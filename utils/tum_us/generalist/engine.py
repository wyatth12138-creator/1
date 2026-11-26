"""Native task-utility training, collection, comparison and incremental adaptation."""
import copy, hashlib, json, math, os, random, time
from dataclasses import replace
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from . import checkpoint
from tum_us.engine.runtime import autocast_context, make_grad_scaler


def _json(path,obj):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp'); temp.write_text(json.dumps(obj,ensure_ascii=False,indent=2,default=str),encoding='utf-8'); os.replace(temp,path)

def _controller_arch(cfg):
    keys=('descriptor_dim','hidden_dim','num_heads','num_layers','dropout','task_agnostic','semantic_dropout','unseen_semantics_to_unknown')
    return {k:cfg.get('controller',{})[k] for k in keys if k in cfg.get('controller',{})}

def _device(cfg):
    v=cfg.get('training',{}).get('device','auto')
    return torch.device(('cuda' if torch.cuda.is_available() else 'cpu') if v=='auto' else v)

def _seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def _datasets(cfg,split,names):
    from .data import build_endpoint_datasets
    return build_endpoint_datasets(cfg,split,endpoints=list(names))

def _names(cfg,key):
    names=cfg.get('endpoint_sets',{}).get(key,[])
    if not names: raise ValueError('No endpoints configured for '+key)
    return list(names)

def _fingerprints(datasets):
    return {k:v.manifest_fingerprint for k,v in datasets.items()}

def _collection_source_identity():
    from tum_us.paths import source_files
    return {key:hashlib.sha256(path.read_bytes()).hexdigest() for key,path in source_files().items()}


def _identity(cfg):
    c=copy.deepcopy(cfg); c.pop('output_dir',None)
    return hashlib.sha256(json.dumps(c,sort_keys=True,default=str).encode()).hexdigest()

def _policy(model,policy,capacity):
    if hasattr(model,'set_memory_policy'): model.set_memory_policy(policy,capacity)
    else: model.config.policy=policy; model.config.memory_capacity=int(capacity)

def _entity_loss(model,entity,endpoint,device,recompute=False):
    from tum_us.transfer.metrics import supervised_entity_loss
    return supervised_entity_loss(model,entity,endpoint,device,recompute=recompute)


def ceus_segmentation_clips(entity):
    """Group windows by clip and retain the same per-clip loss weighting."""
    windows_by_clip = {}
    labeled_clips = set()
    for batch in entity.batches:
        meta = batch.metadata[0]
        key = (meta['record_id'], meta['sample_index'])
        windows_by_clip.setdefault(key, []).append(batch)
        if bool((batch.mask_valid & batch.frame_valid.unsqueeze(-1)).any()):
            labeled_clips.add(key)
    if not labeled_clips:
        raise ValueError('CEUS segmentation case has no annotated clip')
    return [replace(entity, batches=windows) for key, windows in windows_by_clip.items()
            if key in labeled_clips]


class _TimedModel(torch.nn.Module):
    def __init__(self,model,device): super().__init__(); self.model=model; self.device=device; self.seconds=0.; self.bytes=[]
    def forward(self,batch,endpoint=None):
        if self.device.type=='cuda': torch.cuda.synchronize(self.device)
        start=time.perf_counter(); out=self.model(batch,endpoint=endpoint)
        if self.device.type=='cuda': torch.cuda.synchronize(self.device)
        self.seconds+=time.perf_counter()-start; self.bytes.extend(int(x) for x in out.get('memory_bytes',[])); return out

class _OneEntity:
    def __init__(self,source,index): self.source=source; self.index=index; self.spec=source.spec
    def __len__(self): return 1
    def __getitem__(self,index):
        if index!=0: raise IndexError(index)
        return self.source[self.index]

def _evaluate_source(model,dataset,endpoint,device,bootstrap=0,seed=3407,fit_seed=None,policy_seed=None):
    from tum_us.transfer.metrics import evaluate_entities
    from .analysis import summarize_predictions
    timed=_TimedModel(model,device); records=[]; start=time.perf_counter()
    for i in range(len(dataset)):
        timed.seconds=0.; timed.bytes=[]
        result=evaluate_entities(timed,_OneEntity(dataset,i),endpoint,device); r=result['records'][0]
        r.update({'endpoint':endpoint,'kind':dataset.spec['kind'],'seed':int(seed if fit_seed is None else fit_seed),'policy_seed':None if policy_seed is None else int(policy_seed),'seconds':timed.seconds,'memory_bytes':max(timed.bytes,default=0)})
        records.append(r)
    return {'metrics':summarize_predictions(records,bootstrap=bootstrap,seed=seed),'records':records,'actual_wall_seconds':time.perf_counter()-start,
            'input_size':list(dataset.size),'score_mask_size':list(getattr(dataset,'mask_size',dataset.size)),
            'max_memory_bytes':max((r['memory_bytes'] for r in records),default=0)}

def _macro(model,datasets,device,seed=3407):
    results={k:_evaluate_source(model,ds,k,device,seed=seed,fit_seed=seed) for k,ds in datasets.items()}
    return sum(sum(r['loss'] for r in v['records'])/len(v['records']) for v in results.values())/len(results),results

def _reader_initialization(cfg):
    path=cfg.get('training',{}).get('init_checkpoint')
    if not path:
        return None
    path=Path(path).resolve()
    payload=checkpoint._read(path)
    saved=copy.deepcopy(payload['model_config'])
    saved.setdefault('image_size',512)
    effective={**saved,'image_size':cfg['model']['image_size']}
    return {'checkpoint':str(path),'sha256':checkpoint.file_hash(path),
            'source_stage':payload['stage'],'source_model_config':saved,
            'effective_model_config':effective,'optimizer':'fresh','scheduler':'fresh'}


def _new_reader(cfg,initialization):
    from .model import GeneralistModel
    if initialization is None:
        return GeneralistModel(cfg['model'],_controller_arch(cfg))
    model,_=checkpoint.load(initialization['checkpoint'],image_size=cfg['model']['image_size'])
    actual=model.model_config
    for key,value in cfg['model'].items():
        observed=actual.get(key,False) if key=='private_seg_calibration' else actual.get(key)
        if key!='checkpoint' and observed!=value:
            raise ValueError(f'Initialization model differs at {key}: {observed!r} versus {value!r}')
    return model


def run_readers(cfg,out):
    if cfg.get('training',{}).get('multitask',{}).get('enabled',False):
        from .multitask import run_readers as run_multitask_readers
        return run_multitask_readers(cfg,out)
    from .model import GeneralistModel
    out=Path(out); out.mkdir(parents=True,exist_ok=True); tr=cfg['training']; device=_device(cfg); seed=int(tr.get('seed',3407)); _seed(seed)
    names=_names(cfg,'readers'); train=_datasets(cfg,'train',names); dev=_datasets(cfg,'dev',names); fingerprints={'train':_fingerprints(train),'dev':_fingerprints(dev)}
    progress=out/'readers_progress.pt'; final=out/'readers.pt'; expected={'config':_identity(cfg),'manifests':fingerprints}
    initialization=_reader_initialization(cfg)
    if initialization is not None: expected['initialization']=initialization
    if final.exists():
        _,p=checkpoint.load(final,expected_stage='readers',expected_image_size=cfg['model']['image_size'])
        if p['provenance']!=expected: raise ValueError('Reader checkpoint inputs differ')
        return {'checkpoint':str(final),'reused':True,'selection':p['extra']}
    if progress.exists():
        model,p=checkpoint.load(progress,expected_stage='readers_progress',expected_image_size=cfg['model']['image_size'])
        if p['provenance']!=expected: raise ValueError('Reader resume inputs differ')
    else: model=_new_reader(cfg,initialization); p=None
    model.to(device); model.configure_readers(); params=[p for p in model.parameters() if p.requires_grad]
    if p is None and model.support_rule is not None:
        from .support_profiles import initialize_support_profiles
        selected = initialize_support_profiles(model, train, cfg['support_guidance'], seed, device)
        _json(out / 'support_profiles.json', selected)
    if not params: raise ValueError('No reader parameters trainable')
    opt=torch.optim.AdamW(params,lr=float(tr.get('lr',3e-4)),weight_decay=float(tr.get('weight_decay',.01)))
    steps=int(tr.get('reader_updates',1200)); scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=max(steps,1)); batch=int(tr.get('effective_batch',16)); every=int(tr.get('checkpoint_every',100)); counts={k:0 for k in names}; first=0; best=None; best_loss=float('inf'); best_step=0
    if p:
        opt.load_state_dict(p['optimizer']); scheduler.load_state_dict(p['scheduler']); counts.update(p['cursor']['counts']); first=p['cursor']['step']; best=p['extra']['best']; best_loss=p['extra']['best_loss']; best_step=p['extra']['best_step']; checkpoint.restore_rng(p['rng'])
    scaler=make_grad_scaler(device,tr.get('amp',True))
    if p and p.get('scaler') is not None: scaler.load_state_dict(p['scaler'])
    if p is None and tr.get('post_init_seed') is not None:
        # Optional matched reader RNG after architecture-dependent module init.
        _seed(int(tr['post_init_seed']))
    sampling = tr.get('reader_sampling', 'source_balanced')
    if sampling not in ('source_balanced', 'full_pass'):
        raise ValueError('Unknown reader_sampling: ' + str(sampling))
    entity_pool = [(ep, i) for ep in names for i in range(len(train[ep]))] if sampling == 'full_pass' else []
    first_select_step = (len(entity_pool) + batch - 1) // batch
    if steps < first_select_step:
        raise ValueError('full_pass requires enough updates to cover every training entity')
    shuffled_entities = []
    previous_epoch = -1
    start=time.perf_counter(); step=first
    while step<steps:
        model.train(); opt.zero_grad(set_to_none=True)
        for offset in range(batch):
            position = step * batch + offset
            if sampling == 'full_pass':
                epoch, index = divmod(position, len(entity_pool))
                if epoch != previous_epoch:
                    shuffled_entities = entity_pool.copy()
                    random.Random(seed + epoch * 1009).shuffle(shuffled_entities)
                    previous_epoch = epoch
                ep, entity_index = shuffled_entities[index]
            else:
                ep = names[position % len(names)]
                epoch, index = divmod(counts[ep], len(train[ep]))
                order = list(range(len(train[ep])))
                random.Random(seed + epoch * 1009 + names.index(ep) * 7919).shuffle(order)
                entity_index = order[index]
            counts[ep] += 1
            dataset = train[ep]
            entity = dataset[entity_index]
            parts = [entity]
            if dataset.spec['task'] == 'ceus_seg':
                parts = ceus_segmentation_clips(entity)
            for part in parts:
                with autocast_context(device, tr.get('amp', True)):
                    loss = _entity_loss(model, part, ep, device, recompute=len(part.batches) > 1)
                    loss = loss / (batch * len(parts))
                if not loss.requires_grad:
                    raise ValueError('Reader batch has no supervision')
                scaler.scale(loss).backward()
        scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(params,float(tr.get('clip_grad_norm',1.)),error_if_nonfinite=not scaler.is_enabled()); old_scale=scaler.get_scale(); scaler.step(opt); scaler.update()
        if scaler.get_scale()<old_scale: continue
        scheduler.step()
        log_every = int(tr.get('reader_log_every', 0))
        if log_every and (step == first or (step + 1) % log_every == 0):
            elapsed = time.perf_counter() - start
            print(f'Readers {step + 1}/{steps} updates, {elapsed:.1f} seconds', flush=True)
        if (step+1)%every==0 or step+1==steps:
            if step + 1 >= first_select_step:
                score, metrics = _macro(model, dev, device, seed=seed)
                if score < best_loss:
                    best_loss = score
                    best = copy.deepcopy(model.state_dict())
                    best_step = step + 1
                    _json(out / 'readers_dev_best.json', metrics)
            checkpoint.save(progress,model,cfg,stage='readers_progress',optimizer=opt,scheduler=scheduler,scaler=scaler,cursor={'step':step+1,'counts':counts},provenance=expected,extra={'best':best,'best_loss':best_loss,'best_step':best_step})
        step+=1
    if best is None: raise ValueError('Reader training produced no selectable checkpoint')
    model.load_state_dict(best); checkpoint.save(final,model,cfg,stage='readers',provenance=expected,extra={'best_step':best_step,'macro_loss':best_loss,'training_updates':steps,'reader_sampling':sampling,'source_exposures':counts,'actual_wall_seconds':time.perf_counter()-start})
    return {'checkpoint':str(final),'reused':False,'best_step':best_step,'macro_loss':best_loss}


def _collect(cfg,out,reference,names,*,behavior=None,fixed_state=False):
    from .rollout import collect_examples
    out=Path(out); out.mkdir(parents=True,exist_ok=True); device=_device(cfg); opts=cfg.get('utility',{}); seed=int(cfg['training'].get('seed',3407)); datasets=_datasets(cfg,'train',names)
    model,payload=checkpoint.load(reference,expected_image_size=cfg['model']['image_size']); model.to(device).eval(); model.requires_grad_(False)
    behavior=behavior or opts.get('behavior_policy','fifo')
    coupled=opts.get('tie_continuation_to_behavior',True)
    reuse=opts.get('reuse_prefix',True); objective=opts.get('target_objective','task')
    if type(coupled) is not bool or type(reuse) is not bool: raise ValueError('Utility continuation/prefix switches must be boolean')
    if objective not in ('task','pointer_consistency','output_consistency'): raise ValueError('Invalid utility target objective')
    continuation=opts.get('continuation_policy','fifo') if behavior=='fifo' or not coupled else behavior
    signature={'reference_sha256':checkpoint.file_hash(reference),'manifests':_fingerprints(datasets),'capacities':opts.get('capacities',[2,4,8]),'horizons':opts.get('horizons',[1,2,4]),'max_states':opts.get('max_states',2),'behavior':behavior,'continuation':continuation,'fixed_state':fixed_state,'seed':seed,'endpoints':{name:cfg['endpoints'][name] for name in names},'endpoint_order':list(names),'data':cfg.get('data',{}),'video':cfg.get('video',{}),'data_root':cfg.get('data_root'),'max_entities':opts.get('max_entities')}
    signature.update(image_size=model.config.image_size,reader_protocol=copy.deepcopy(model.model_config),target_objective=objective,proxy_teacher='reader_internal_causal_full_memory' if objective!='task' else None,reuse_prefix=reuse,continuation_mode='legacy_auto' if coupled else 'explicit')
    signature['source_sha256']=_collection_source_identity()
    index=out/'utility_index.json'; ident=hashlib.sha256(json.dumps(signature,sort_keys=True).encode()).hexdigest(); manifest={'format':'tum_us.generalist.utility.v1','fingerprint':ident,'signature':signature,'reference':str(reference),'items':[],'next_entity':0,'next_batch':0,'next':0,'complete':False}
    if index.exists():
        manifest=json.loads(index.read_text(encoding='utf-8'))
        if manifest['fingerprint']!=ident: raise ValueError('Utility cache dependencies changed')
        for item in manifest['items']:
            if checkpoint.file_hash(out/item['file'])!=item['sha256']: raise ValueError('Utility shard integrity failure')
        if manifest['complete']: return {'index':str(index),'examples':sum(i['count'] for i in manifest['items']),'reused':True}
    work=[]; maximum=opts.get('max_entities')
    for ep,ds in datasets.items():
        work.extend((ep,i) for i in range(min(len(ds),int(maximum)) if maximum is not None else len(ds)))
    start=time.perf_counter()
    for entity_pos in range(manifest.get('next_entity',0),len(work)):
        ep,i=work[entity_pos]; entity=datasets[ep][i]
        start_batch=manifest.get('next_batch',0) if entity_pos==manifest.get('next_entity',0) else 0
        for view in range(start_batch,len(entity.batches)):
            pos=manifest['next']; batch=entity.batches[view]; _seed(seed+pos*7919)
            examples=collect_examples(model,batch.to(device),capacities=tuple(signature['capacities']),max_states=int(signature['max_states']),horizons=tuple(signature['horizons']),behavior_policy=behavior,continuation_policy=continuation,fixed_state=fixed_state,reuse_prefix=reuse,target_objective=objective)
            cleaned=[]
            for ex in examples:
                item={k:(v.detach().cpu() if torch.is_tensor(v) else v) for k,v in ex.items()}; item['endpoint']=ep; item['entity_key']=entity.key; item['record_index']=view; cleaned.append(item)
                stats=manifest.setdefault('utility_statistics',{'segmentation':{'observed_states':0,'exact_zero_states':0,'max_abs_delta':0.},'classification':{'observed_states':0,'exact_zero_states':0,'max_abs_delta':0.}})
                for axis,axis_name in enumerate(('segmentation','classification')):
                    eligible=~torch.as_tensor(item['anchor_mask']).bool()
                    if bool(torch.as_tensor(item['valid_tasks'])[axis]) and eligible.any():
                        magnitude=float(torch.as_tensor(item['deltas'])[eligible,axis].abs().max()); row=stats[axis_name]
                        row['observed_states']+=1; row['exact_zero_states']+=int(magnitude==0.); row['max_abs_delta']=max(row['max_abs_delta'],magnitude)
            if cleaned:
                name=f'shard_{pos:07d}.pt'; checkpoint.atomic_save({'examples':cleaned},out/name)
                manifest['items'].append({'file':name,'count':len(cleaned),'sha256':checkpoint.file_hash(out/name)})
            manifest['next']=pos+1; manifest['next_entity']=entity_pos; manifest['next_batch']=view+1; _json(index,manifest)
        manifest['next_entity']=entity_pos+1; manifest['next_batch']=0; _json(index,manifest)
    if not manifest['items']: raise ValueError('No labeled future deletion states; check sequence frames, capacity and annotation coverage')
    manifest['complete']=True; manifest['actual_wall_seconds']=time.perf_counter()-start; _json(index,manifest)
    return {'index':str(index),'examples':sum(i['count'] for i in manifest['items']),'reused':False,'actual_wall_seconds':manifest['actual_wall_seconds']}

def collect(cfg,out):
    out=Path(out); reference=cfg.get('utility',{}).get('reference_checkpoint') or str(out/'readers.pt')
    return _collect(cfg,out/'utility',reference,_names(cfg,'collect'),fixed_state=bool(cfg.get('utility',{}).get('fixed_state',False)))

def _read_cache(path):
    path=Path(path); manifest=json.loads(path.read_text(encoding='utf-8'))
    if manifest.get('format')!='tum_us.generalist.utility.v1' or not manifest.get('complete'): raise ValueError('Incomplete utility cache')
    fingerprint=hashlib.sha256(json.dumps(manifest['signature'],sort_keys=True).encode()).hexdigest()
    if fingerprint!=manifest.get('fingerprint'): raise ValueError('Utility provenance fingerprint mismatch')
    if checkpoint.file_hash(manifest['reference'])!=manifest['signature']['reference_sha256']: raise ValueError('Utility reference checkpoint changed')
    examples=[]
    for shard in manifest['items']:
        p=(path.parent/shard['file']).resolve()
        if p.parent!=path.parent.resolve() or checkpoint.file_hash(p)!=shard['sha256']: raise ValueError('Invalid utility shard')
        examples.extend(torch.load(p,map_location='cpu',weights_only=False)['examples'])
    if not examples: raise ValueError('Empty utility cache')
    return examples,manifest

def _collate(items,device):
    n=max(len(x['descriptors']) for x in items); b=len(items)
    result={'descriptors':torch.zeros(b,n,328),'deltas':torch.zeros(b,n,2),'slot_mask':torch.zeros(b,n,dtype=torch.bool),'anchor_mask':torch.ones(b,n,dtype=torch.bool),'valid_tasks':torch.zeros(b,2,dtype=torch.bool),'active_tasks':torch.zeros(b,2,dtype=torch.bool),'context':torch.zeros(b,5)}
    for i,x in enumerate(items):
        size=len(x['descriptors']); result['descriptors'][i,:size]=x['descriptors']; result['deltas'][i,:size]=x['deltas']; result['slot_mask'][i,:size]=True; result['anchor_mask'][i,:size]=x['anchor_mask']; result['valid_tasks'][i]=x['valid_tasks']; result['active_tasks'][i]=x.get('active_tasks',x['valid_tasks']); result['context'][i]=x['context']
    return {k:v.to(device) for k,v in result.items()}

def controller_objective(pred,batch,scales,temperature=.2,regression_weight=1.,ranking_weight=1.,axis=None):
    if temperature<=0: raise ValueError('Temperature must be positive')
    losses=[[],[]]; ranks=[[],[]]; count=[0,0]
    candidates=batch['slot_mask'] & ~batch['anchor_mask']
    for i in range(len(pred)):
        keep=candidates[i]
        if not keep.any(): continue
        for task in range(2):
            if axis is not None and task!=axis: continue
            if not bool(batch['valid_tasks'][i,task] and batch['active_tasks'][i,task]): continue
            target=batch['deltas'][i,keep,task]/scales[task]; forecast=pred[i,keep,task]; target=target-target.mean(); forecast=forecast-forecast.mean()
            losses[task].append(F.smooth_l1_loss(forecast,target)); teacher=(-target/temperature).softmax(0); ranks[task].append(-(teacher*F.log_softmax(-forecast/temperature,dim=0)).sum()); count[task]+=1
    if not any(losses): raise ValueError('No observed task with evictable candidates')
    reg=torch.stack([torch.stack(x).mean() for x in losses if x]).mean(); rank=torch.stack([torch.stack(x).mean() for x in ranks if x]).mean()
    return {'loss':regression_weight*reg+ranking_weight*rank,'regression':reg,'ranking':rank,'axis_states':count}

def _scales(examples,eps=1e-6):
    groups=_sampling_groups(examples); counts=torch.zeros(2); per_source=[[],[]]
    for entities in groups.values():
        per_entity=[[],[]]
        for items in entities.values():
            values=[[],[]]
            for x in items:
                valid=torch.as_tensor(x['valid_tasks']).bool() & torch.as_tensor(x.get('active_tasks',x['valid_tasks'])).bool(); keep=~torch.as_tensor(x['anchor_mask']).bool(); d=torch.as_tensor(x['deltas']).float()
                for axis in range(2):
                    if valid[axis] and keep.any():
                        v=d[keep,axis]-d[keep,axis].mean(); values[axis].append(v.abs().mean()); counts[axis]+=1
            for axis in range(2):
                if values[axis]: per_entity[axis].append(torch.stack(values[axis]).mean())
        for axis in range(2):
            if per_entity[axis]: per_source[axis].append(torch.stack(per_entity[axis]).mean())
    scales=torch.stack([torch.stack(v).mean() if v else torch.tensor(1.) for v in per_source]).clamp_min(eps)
    return scales,counts
def _reader_hash(model):
    h=hashlib.sha256()
    for name,value in sorted(model.state_dict().items()):
        if name.startswith('controller.'): continue
        tensor=value.detach().cpu().contiguous(); h.update(name.encode()); h.update(str(tensor.dtype).encode()); h.update(str(tuple(tensor.shape)).encode()); h.update(tensor.numpy().tobytes())
    return h.hexdigest()

def _sampling_groups(examples):
    groups={}
    for item in examples:
        valid=torch.as_tensor(item['valid_tasks']).bool() & torch.as_tensor(item.get('active_tasks',item['valid_tasks'])).bool()
        axis=''.join(str(i) for i in range(2) if valid[i]) or 'none'
        source=str(item.get('endpoint','axis:'+axis)); entity=str(item.get('entity_key','unknown'))
        groups.setdefault(source,{}).setdefault(entity,[]).append(item)
    if not groups: raise ValueError('No utility examples available')
    return groups

def _balanced_draw(groups,rng):
    sources=sorted(groups); source=sources[rng.randrange(len(sources))]
    entities=sorted(groups[source]); entity=entities[rng.randrange(len(entities))]
    items=groups[source][entity]
    return items[rng.randrange(len(items))]
def _train_controller(cfg,out,index,reference,*,old_index=None,variant='joint',preserve_scales=False):
    out=Path(out); out.mkdir(parents=True,exist_ok=True); device=_device(cfg); opts=cfg.get('controller',{}); seed=int(cfg['training'].get('seed',3407)); _seed(seed)
    examples,manifest=_read_cache(index); old,old_manifest=_read_cache(old_index) if old_index else ([],None)
    model,parent=checkpoint.load(reference,expected_image_size=cfg['model']['image_size']); model.to(device)
    for cached in (manifest,old_manifest):
        if cached is not None and cached['signature'].get('image_size',512)!=model.config.image_size:
            raise ValueError('Utility labels use a different image_size; recollect them with the current reader')
    if checkpoint.file_hash(reference)!=manifest['signature']['reference_sha256']:
        source,_=checkpoint.load(manifest['reference'],expected_image_size=cfg['model']['image_size'])
        if _reader_hash(source)!=_reader_hash(model): raise ValueError('Utility cache and controller parent use different frozen readers')
    if old_manifest is not None:
        previous,_=checkpoint.load(old_manifest['reference'],expected_image_size=cfg['model']['image_size'])
        if _reader_hash(previous)!=_reader_hash(model): raise ValueError('Old utility replay uses different frozen readers')
    if variant=='task_agnostic': model.controller.task_agnostic=True; model.controller_config['task_agnostic']=True
    axis={'seg_specialist':0,'cls_specialist':1}.get(variant)
    if variant not in ('joint','task_agnostic','seg_specialist','cls_specialist'): raise ValueError('Unknown controller variant')
    if axis is not None:
        examples=[x for x in examples if bool(torch.as_tensor(x['valid_tasks'])[axis]) and bool(torch.as_tensor(x.get('active_tasks',x['valid_tasks']))[axis])]
        old=[x for x in old if bool(torch.as_tensor(x['valid_tasks'])[axis]) and bool(torch.as_tensor(x.get('active_tasks',x['valid_tasks']))[axis])]
        if not examples: raise ValueError('Specialist has no observed examples for its task')
    model.eval(); model.configure_controller(); opts_steps=int(opts.get('updates',1200)); batchsize=int(opts.get('batch_size',8)); replay=float(opts.get('replay_fraction',.5));
    if opts_steps<1 or batchsize<1 or not 0<=replay<=1: raise ValueError('Invalid controller training options')
    if old_index and (not old or not 0<replay<1): raise ValueError('Expansion requires both old and new utility replay in each training batch')
    if old and batchsize<2: raise ValueError('Replay requires controller batch_size >= 2')
    if old_index or preserve_scales: scales=model.controller.scales.detach().clone().cpu(); counts=None
    else: scales,counts=_scales(examples)
    model.controller.set_scales(scales.to(device)); params=list(model.controller.parameters()); opt=torch.optim.AdamW(params,lr=float(opts.get('lr',3e-4)),weight_decay=float(opts.get('weight_decay',.01)))
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=opts_steps); progress=out/'controller_progress.pt'; final=out/'controller.pt'; start_step=0
    provenance={'config':_identity(cfg),'cache_sha256':checkpoint.file_hash(index),'parent_sha256':checkpoint.file_hash(reference),'old_cache_sha256':checkpoint.file_hash(old_index) if old_index else None,'variant':variant,'preserve_scales':preserve_scales}
    if final.exists():
        _,p=checkpoint.load(final,expected_stage='controller',expected_image_size=cfg['model']['image_size']);
        if p['provenance']!=provenance: raise ValueError('Controller completed inputs differ')
        return {'checkpoint':str(final),'reused':True,'complete':True,'updates':p['cursor']['step']}
    if progress.exists():
        model,p=checkpoint.load(progress,expected_stage='controller_progress',expected_image_size=cfg['model']['image_size']);
        if p['provenance']!=provenance: raise ValueError('Controller resume inputs differ')
        model.to(device); model.eval(); model.configure_controller(); params=list(model.controller.parameters()); opt=torch.optim.AdamW(params,lr=float(opts.get('lr',3e-4)),weight_decay=float(opts.get('weight_decay',.01))); scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=opts_steps); opt.load_state_dict(p['optimizer']); scheduler.load_state_dict(p['scheduler']); start_step=int(p['cursor']['step']); checkpoint.restore_rng(p['rng'])
    scaler=make_grad_scaler(device,cfg['training'].get('amp',True))
    if progress.exists() and p.get('scaler') is not None: scaler.load_state_dict(p['scaler'])
    groups=_sampling_groups(examples); old_groups=_sampling_groups(old) if old else None
    started=time.perf_counter(); every=int(opts.get('checkpoint_every',100)); max_seconds=opts.get('max_seconds'); step=start_step; last={}; model.controller.train()
    while step<opts_steps:
        rnd=random.Random(seed+step*1000003); old_n=max(1,min(batchsize-1,round(batchsize*replay))) if old else 0
        selected=[_balanced_draw(groups,rnd) for _ in range(batchsize-old_n)]+[_balanced_draw(old_groups,rnd) for _ in range(old_n)]
        batch=_collate(selected,device); opt.zero_grad(set_to_none=True)
        with autocast_context(device,cfg['training'].get('amp',True)):
            pred=model.controller(batch['descriptors'],batch['slot_mask'],batch['context'])
            loss=controller_objective(pred,batch,model.controller.scales,float(opts.get('temperature',.2)),float(opts.get('regression_weight',1.)),float(opts.get('ranking_weight',1.)),axis)
        scaler.scale(loss['loss']).backward(); scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(params,float(opts.get('clip_grad_norm',1.)),error_if_nonfinite=not scaler.is_enabled()); old_scale=scaler.get_scale(); scaler.step(opt); scaler.update()
        if scaler.get_scale()<old_scale: continue
        scheduler.step(); step+=1
        last={k:(float(v.detach()) if torch.is_tensor(v) else v) for k,v in loss.items()}
        stop=max_seconds is not None and time.perf_counter()-started>=float(max_seconds)
        if step%every==0 or step==opts_steps or stop:
            checkpoint.save(progress,model,cfg,stage='controller_progress',optimizer=opt,scheduler=scheduler,scaler=scaler,cursor={'step':step},parents={'reference':reference},provenance=provenance,extra={'loss':last,'scales':scales.tolist(),'axis_counts':counts.tolist() if counts is not None else None})
        if stop: break
    extra={'loss':last,'scales':scales.tolist(),'axis_counts':counts.tolist() if counts is not None else None,'actual_wall_seconds':time.perf_counter()-started,'variant':variant,'complete':step==opts_steps}
    # A soft time limit still yields a loadable snapshot, while progress retains the optimizer for continuation.
    path=final if step==opts_steps else out/'controller_latest.pt'
    checkpoint.save(path,model,cfg,stage='controller',optimizer=opt,scheduler=scheduler,scaler=scaler,cursor={'step':step},parents={'reference':reference},provenance=provenance,extra=extra)
    return {'checkpoint':str(path),'progress':str(progress),'updates':step,'complete':step==opts_steps,'scales':scales.tolist(),'actual_wall_seconds':extra['actual_wall_seconds']}

def train_controller(cfg,out):
    out=Path(out); opts=cfg.get('controller',{}); index=opts.get('utility_index') or str(out/'utility'/'utility_index.json'); _,manifest=_read_cache(index); reference=opts.get('reference_checkpoint') or manifest['reference']
    return _train_controller(cfg,out,index,reference,variant=opts.get('variant','joint'))

def evaluate(cfg,out):
    out=Path(out); opts=cfg.get('evaluation',{}); device=_device(cfg); path=opts.get('checkpoint') or str(out/'controller.pt'); model,payload=checkpoint.load(path,expected_image_size=cfg['model']['image_size']); model.to(device).eval(); reader_hash=_reader_hash(model)
    reader_protocol=copy.deepcopy(getattr(model,'model_config',cfg['model']))
    refs={'tum':(model,str(path))}; fit_seed=int(payload.get('config',{}).get('training',{}).get('seed',3407))
    for name,other in opts.get('checkpoints',{}).items():
        candidate,_=checkpoint.load(other,expected_image_size=cfg['model']['image_size']); candidate.to(device).eval()
        if _reader_hash(candidate)!=reader_hash: raise ValueError('Policy comparison must use identical frozen readers')
        refs[name]=(candidate,str(other))
    datasets=_datasets(cfg,opts.get('split','dev'),_names(cfg,'evaluate')); policies=opts.get('policies',['none','fifo','random','quality','novelty','tum','full']); capacities=opts.get('capacities',[2,4,8]); reports=[]; started=time.perf_counter(); base_seed=int(opts.get('seed',3407)); boot=int(opts.get('bootstrap',1000))
    variants=list(policies)+[name for name in refs if name!='tum' and name not in policies and name not in ('seg_specialist','cls_specialist')]
    for policy in variants:
        for capacity in capacities:
            per={}; refpath=str(path)
            for ep,ds in datasets.items():
                actual=policy; candidate=model
                if policy=='specialist':
                    selected='seg_specialist' if ds.spec['kind']=='segmentation' else 'cls_specialist'
                    if selected not in refs: raise ValueError('Specialist comparison requires both specialist checkpoints')
                    candidate,refpath=refs[selected]; actual='tum'
                elif policy in refs:
                    candidate,refpath=refs[policy]; actual='tum'
                _seed(base_seed); _policy(candidate,actual,int(capacity)); candidate.eval()
                per[ep]=_evaluate_source(candidate,ds,ep,device,bootstrap=boot,seed=base_seed,fit_seed=fit_seed,policy_seed=base_seed)
            reports.append({'policy':policy,'capacity':int(capacity),'checkpoint':refpath,'endpoints':per})
    from .analysis import paired_comparison
    comparisons={}
    for item in reports:
        baseline=next((x for x in reports if x['policy']=='fifo' and x['capacity']==item['capacity']),None)
        if baseline and item['policy']!='fifo':
            left=[r for ep in item['endpoints'].values() for r in ep['records']]; right=[r for ep in baseline['endpoints'].values() for r in ep['records']]
            comparisons[item['policy']+':'+str(item['capacity'])]=paired_comparison(left,right,bootstrap=boot,seed=base_seed)
    controller_comparisons={}
    for item in reports:
        main=next((x for x in reports if x['policy']=='tum' and x['capacity']==item['capacity']),None)
        if main and item['policy'] not in ('tum','none','fifo','random','quality','novelty','full'):
            left=[r for ep in main['endpoints'].values() for r in ep['records']]; right=[r for ep in item['endpoints'].values() for r in ep['records']]
            controller_comparisons['tum-minus-'+item['policy']+':'+str(item['capacity'])]=paired_comparison(left,right,bootstrap=boot,seed=base_seed)
    result={'evaluation_split':opts.get('split','dev'),'fit_seed':fit_seed,'policy_seed':base_seed,'evaluation_config_sha256':_identity(cfg),'reader_protocol':reader_protocol,'video_protocol':copy.deepcopy(cfg.get('video',{})),'endpoint_protocols':{ep:copy.deepcopy(cfg['endpoints'][ep]) for ep in datasets},'reader_sha256':reader_hash,'manifests':_fingerprints(datasets),'policies':reports,'paired_vs_fifo':comparisons,'paired_controller_comparisons':controller_comparisons,'actual_wall_seconds':time.perf_counter()-started}; _json(out/'evaluation'/'policy_reports.json',result); return result

def expand(cfg,out):
    out=Path(out); opts=cfg.get('controller',{}); reference=opts.get('reference_checkpoint') or str(out/'controller.pt'); old_index=opts.get('old_utility_index') or str(out/'utility'/'utility_index.json')
    cache=_collect(cfg,out/'expansion'/'utility_new',reference,_names(cfg,'expand_new'),behavior='tum')
    trained=_train_controller(cfg,out/'expansion',cache['index'],reference,old_index=old_index,variant=opts.get('variant','joint'))
    if not trained.get('complete',False): return {'collection':cache,'training':trained,'status':'time_limit'}
    fine=_train_controller(cfg,out/'expansion'/'no_replay',cache['index'],reference,preserve_scales=True)
    if not fine.get('complete',False): return {'collection':cache,'training':trained,'no_replay':fine,'status':'time_limit'}
    ev_cfg=copy.deepcopy(cfg); ev_cfg['endpoint_sets']['evaluate']=list(dict.fromkeys(_names(cfg,'expand_old')+_names(cfg,'expand_new'))); ev_cfg.setdefault('evaluation',{})['checkpoint']=trained['checkpoint']; ev_cfg['evaluation']['checkpoints']={'freeze':reference,'no_replay':fine['checkpoint']}
    evaluated=evaluate(ev_cfg,out/'expansion'); return {'collection':cache,'training':trained,'no_replay':fine,'comparison':evaluated}

def run(cfg,out):
    out=Path(out); started=time.perf_counter(); out.mkdir(parents=True,exist_ok=True)
    reader=run_readers(cfg,out); cache=collect(cfg,out); trained=train_controller(cfg,out)
    if not trained.get('complete',True): return {'readers':reader,'collection':cache,'controller':trained,'status':'controller_time_limit','actual_wall_seconds':time.perf_counter()-started}
    checkpoints={}; opts=cfg.get('controller',{}); variants=cfg.get('utility',{}).get('variants',opts.get('variants',['fixed_state','task_agnostic','seg_specialist','cls_specialist']))
    for variant in variants:
        if variant=='fixed_state':
            fixed=_collect(cfg,out/'utility_fixed_state',reader['checkpoint'],_names(cfg,'collect'),fixed_state=True)
            branch=_train_controller(cfg,out/'control_fixed_state',fixed['index'],reader['checkpoint'])
        elif variant in ('task_agnostic','seg_specialist','cls_specialist'):
            branch=_train_controller(cfg,out/('control_'+variant),cache['index'],reader['checkpoint'],variant=variant)
        else: raise ValueError('Unknown requested variant '+str(variant))
        if not branch.get('complete',True): return {'readers':reader,'collection':cache,'controller':trained,'status':'control_time_limit','control':variant,'partial':branch}
        checkpoints[variant]=branch['checkpoint']
    current=trained['checkpoint']; rounds=[]
    for r in range(int(cfg.get('utility',{}).get('recollection_rounds',0))):
        directory=out/f'recollection_{r+1}'; prior=current; new=_collect(cfg,directory/'utility',prior,_names(cfg,'collect'),behavior='tum'); updated=_train_controller(cfg,directory,new['index'],prior,old_index=cache['index']); control=_train_controller(cfg,directory/'no_recollection',cache['index'],prior,preserve_scales=True); rounds.append({'collection':new,'controller':updated,'equal_updates_no_recollection':control})
        if not updated.get('complete',True) or not control.get('complete',True): return {'readers':reader,'collection':cache,'controller':trained,'recollection':rounds,'status':'recollection_time_limit'}
        current=updated['checkpoint']; checkpoints[f'no_recollection_round_{r+1}']=control['checkpoint']
    ev_cfg=copy.deepcopy(cfg); ev_cfg.setdefault('evaluation',{})['checkpoint']=current; ev_cfg['evaluation']['checkpoints']={**ev_cfg['evaluation'].get('checkpoints',{}),**checkpoints}
    if 'seg_specialist' in checkpoints and 'cls_specialist' in checkpoints: ev_cfg['evaluation']['policies']=list(ev_cfg['evaluation'].get('policies',['none','fifo','random','quality','novelty','tum','full']))+['specialist']
    reports=evaluate(ev_cfg,out); result={'readers':reader,'collection':cache,'controller':trained,'controls':checkpoints,'recollection':rounds,'evaluation':reports}
    if cfg.get('utility',{}).get('run_expand',True):
        expand_cfg=copy.deepcopy(cfg); expand_cfg.setdefault('controller',{})['reference_checkpoint']=current; result['expansion']=expand(expand_cfg,out)
        if result['expansion'].get('status'): result['status']='expansion_'+result['expansion']['status']
    result.setdefault('status','complete'); result['actual_wall_seconds']=time.perf_counter()-started; _json(out/'pipeline_result.json',result); return result













