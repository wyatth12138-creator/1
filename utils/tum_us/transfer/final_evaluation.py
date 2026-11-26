"""Immutable model-collection locking and resumable, cached final evaluation."""
from __future__ import annotations
import copy,hashlib,json,os
from pathlib import Path
import torch
import portalocker
from .checkpoint import digest,file_hash,load_checkpoint
from . import engine

def _write(path,obj):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_suffix(path.suffix+".tmp")
 with temp.open("w",encoding="utf-8") as f:
  json.dump(obj,f,ensure_ascii=False,indent=2);f.flush();os.fsync(f.fileno())
 os.replace(temp,path)

def _code_hashes():
 from tum_us.paths import source_files
 return {key:file_hash(path) for key,path in source_files().items()}


def _manifest_fingerprint(spec,split):
 path=spec[split+"_manifest"];value=file_hash(path)
 mapping=spec.get("final_frame_map_path",spec.get("frame_map_path")) if split=="final" else spec.get("frame_map_path")
 if mapping:value=hashlib.sha256((value+file_hash(mapping)).encode()).hexdigest()
 return value

def create_lock(checkpoints,path,endpoints=None,final_config=None):
 path=Path(path)
 if path.exists():raise FileExistsError("Model collection lock already exists")
 final_specs=(final_config or {}).get("endpoints",{})
 items={}
 for source in checkpoints:
  checkpoint=Path(source).resolve();payload=torch.load(checkpoint,map_location="cpu",weights_only=False)
  if payload.get("format")!="tum_us.transfer.v1" or digest(payload["config"])!=payload.get("config_hash"):
   raise ValueError("Invalid transfer checkpoint")
  cfg=copy.deepcopy(payload["config"]);registered=payload["spec"]["endpoints"]
  names=list(endpoints or registered)
  if set(names)-registered.keys():raise ValueError("Requested final endpoint was not learned")
  final={}
  for name in names:
   spec=cfg["endpoints"][name]
   for field in ("final_manifest","final_frame_map_path"):
    if field in final_specs.get(name,{}):spec[field]=str(Path(final_specs[name][field]).resolve())
   if not spec.get("final_manifest"):raise ValueError(f"{name}: explicit final_manifest required")
   final[name]={"spec":spec,"fingerprint":_manifest_fingerprint(spec,"final")}
  items[str(checkpoint)]={"checkpoint_sha256":file_hash(checkpoint),"config_hash":payload["config_hash"],
                         "training_manifests":payload["manifests"],"endpoints":final}
 record={"format":"tum_us.final_collection.v1","code_sha256":_code_hashes(),"checkpoints":items}
 record["lock_hash"]=digest(record);_write(path,record);return record

def _groups(path):
 from .data import _read_records
 result=set()
 for r in _read_records(Path(path)):
  group=r.get("group_id") or r.get("entity_uid") or r.get("case_id")
  if group is None:raise ValueError("Missing group in final manifest")
  result.add(str(group))
 return result

def evaluate_locked(lock_path,checkpoint_path,output_dir,endpoints=None,device=None):
 lock_path=Path(lock_path);lock=json.loads(lock_path.read_text(encoding="utf-8"))
 expected=lock.pop("lock_hash");actual=digest(lock);lock["lock_hash"]=expected
 if actual!=expected or lock.get("format")!="tum_us.final_collection.v1":raise ValueError("Collection lock modified")
 if lock["code_sha256"]!=_code_hashes():raise ValueError("Research code changed since model collection lock")
 checkpoint=str(Path(checkpoint_path).resolve())
 if checkpoint not in lock["checkpoints"]:raise ValueError("Checkpoint is not in the locked collection")
 item=lock["checkpoints"][checkpoint]
 if file_hash(checkpoint)!=item["checkpoint_sha256"]:raise ValueError("Locked checkpoint changed")
 names=list(endpoints or item["endpoints"])
 if set(names)-item["endpoints"].keys():raise ValueError("Endpoint outside locked collection")
 for name in names:
  if _manifest_fingerprint(item["endpoints"][name]["spec"],"final")!=item["endpoints"][name]["fingerprint"]:
   raise ValueError("Final manifest or frame map changed")
 journal_path=lock_path.with_suffix(".evaluation.json")
 with portalocker.Lock(str(lock_path)+".guard",timeout=0):
  journal=json.loads(journal_path.read_text(encoding="utf-8")) if journal_path.exists() else {"lock_hash":expected,"models":{}}
  if journal["lock_hash"]!=expected:raise ValueError("Final transaction belongs to another lock")
  run=journal["models"].setdefault(item["checkpoint_sha256"],{"completed":{},"attempts":0})
  if set(names).issubset(run["completed"]):
   return {"checkpoint":checkpoint,"lock_hash":expected,"cached":True,"results":{n:run["completed"][n] for n in names}}
  model,payload=load_checkpoint(checkpoint);cfg=copy.deepcopy(payload["config"])
  if payload["config_hash"]!=item["config_hash"]:raise ValueError("Locked model config differs")
  for key,fp in item["training_manifests"].items():
   ep,split=key.split(":",1)
   if _manifest_fingerprint(cfg["endpoints"][ep],split)!=fp:raise ValueError("Training manifest changed since locking")
  cfg["allow_final_evaluation"]=True;cfg["evaluation_split"]="test"
  if device:cfg["training"]["device"]=device
  run["attempts"]+=1;run["state"]="running";_write(journal_path,journal)
  output=Path(output_dir);output.mkdir(parents=True,exist_ok=True)
  try:
   results={}
   for name in names:
    if name not in run["completed"]:
     spec=item["endpoints"][name]["spec"]
     if _groups(spec["final_manifest"]) & (_groups(spec["train_manifest"]) | _groups(spec["dev_manifest"])):
      raise ValueError("Final groups overlap training/development")
     cfg["endpoints"][name]=spec
     dataset=engine._dataset(cfg,name,"test")
     dev=engine.device_for(cfg);model.to(dev)
     metrics=engine._evaluate(model,dataset,name,dev)
     run["completed"][name]=metrics;_write(journal_path,journal)
    results[name]=run["completed"][name]
   run["state"]="complete";_write(journal_path,journal)
   report={"checkpoint":checkpoint,"lock_hash":expected,"cached":False,"results":results}
   _write(output/"final_evaluation.json",report);return report
  except BaseException as error:
   run["state"]="interrupted";run["error"]=repr(error);_write(journal_path,journal);raise

