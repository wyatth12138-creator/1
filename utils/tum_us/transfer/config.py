"""Portable configuration for the complete transfer-retention study."""
from __future__ import annotations
import copy
import json
from pathlib import Path
from tum_us.paths import ROOT, pretrained_path
from ..data.registry import default_target_id

PROJECT = ROOT
KINDS = {"classification", "segmentation"}

def _endpoint(name, kind, task, anatomy, targets, space=0, budget=None):
    return {"kind": kind, "task": task, "anatomy_id": anatomy,
            "target_ids": targets, "label_space_id": space, "dataset_names": [name],
            "train_manifest": f"manifests/{name}_train.jsonl",
            "dev_manifest": f"manifests/{name}_dev.jsonl", "label_budget": budget}

DEFAULTS = {
 "model": {"backbone":"small", "image_size":256,
           "checkpoint":"pre_checkpoints/sam2.1_hiera_small.pt", "memory_capacity":4,"policy":"fifo"},
 "data_root":"data",
 "training":{"seed":6000,"lr":3e-4,"weight_decay":0.01,"effective_batch":16,
             "A_updates":1200,"head_updates":100,"joint_cycles":600,"probe_cycles":60,
             "warmup_fraction":0.05,"mim_weight":0.1,"device":"auto","amp":True,
             "checkpoint_every":100},
 "discovery_seeds":[6101,6102,6103,6104],
 "confirmation_seeds":[6201,6202,6203,6204,6205],
 "task_order":["A","B","C","D","E","F"],
 "endpoints":{
   "A":_endpoint("BUS-BRA","classification","image_cls",2,[1],budget=148),
   "B":_endpoint("BUSIS","segmentation","image_seg",2,[1]),
   "C":_endpoint("Fetal_HC","segmentation","image_seg",3,[4]),
   "D":_endpoint("Appendix","classification","image_cls",1,[default_target_id("appendix")],space=1),
   "E":_endpoint("CAMUS","segmentation","video_seg",4,[5,6,7]),
   "F":_endpoint("Liver","classification","ceus_cls",6,[default_target_id("liver")])
 },
 "methods":["shared","gain_selected2","drop_a_selected2","random2",
            "same_selected2_frozen_no_copy","private_delta_old_route_frozen"],
 "selector":{"K":2,"tau":0.02,"random_seed":6001},
 "video":{"train_frames":8,"eval_frames":16,"require_native_frame_map":True},
 "output_dir":"runs/transfer",
 "data":{"clip_length":8,"label_subset_seed":6000},
}

for _key, _dataset in {"A":"public/image_cls/BUS-BRA","B":"public/image_seg/BUSIS",
                      "C":"public/image_seg/Fetal_HC","D":"public/image_cls/Appendix",
                      "E":"public/video_seg/CAMUS","F":"private/ceus_cls/Liver"}.items():
    DEFAULTS["endpoints"][_key]["dataset_ids"]=[_dataset]
    DEFAULTS["endpoints"][_key]["train_manifest"]=f"manifests/{_key}_train.jsonl"
    DEFAULTS["endpoints"][_key]["dev_manifest"]=f"manifests/{_key}_dev.jsonl"

def _merge(dst, src):
    for key, value in src.items():
        if isinstance(value, dict) and isinstance(dst.get(key), dict):
            _merge(dst[key], value)
        else:
            dst[key] = copy.deepcopy(value)
    return dst

def load_config(path=None, overrides=None):
    cfg = copy.deepcopy(DEFAULTS)
    if path:
        with Path(path).open(encoding="utf-8-sig") as stream:
            _merge(cfg, json.load(stream))
    if overrides:
        _merge(cfg, overrides)
    cfg["project_root"] = str(PROJECT)
    def resolved(value):
        item=Path(value).expanduser()
        return str((PROJECT/item).resolve() if not item.is_absolute() else item.resolve())
    cfg["model"]["checkpoint"]=str(pretrained_path(cfg["model"]["checkpoint"]))
    cfg["data_root"]=resolved(cfg["data_root"])
    cfg["output_dir"]=resolved(cfg["output_dir"])
    if cfg.get("discovery_archive"):cfg["discovery_archive"]=resolved(cfg["discovery_archive"])
    for key,spec in cfg["endpoints"].items():
        spec["endpoint"]=key
        if spec["kind"] not in KINDS:
            raise ValueError(f"Unsupported endpoint kind: {spec['kind']}")
        for role in ("train_manifest","dev_manifest","final_manifest","frame_map_path","final_frame_map_path"):
            if role in spec:
                spec[role]=resolved(spec[role])
    t=cfg["training"]
    for key in ("effective_batch","A_updates","head_updates","joint_cycles","probe_cycles"):
        if int(t[key])<1:
            raise ValueError(f"{key} must be positive")
    if int(t["probe_cycles"])>int(t["joint_cycles"]):
        raise ValueError("probe_cycles exceeds full schedule")
    if set(cfg["task_order"])-cfg["endpoints"].keys():
        raise ValueError("Task order contains an unregistered endpoint")
    return cfg

