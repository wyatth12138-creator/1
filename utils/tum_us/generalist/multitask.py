"""Opt-in full reader training: unpaired balanced tasks, shared/private updates,
train-only uncertainty/coverage resampling, resumable selection and optimizer.
All later utility/controller/recollection/expansion stages use ordinary engine.
"""
import copy,hashlib,math,random,time
from pathlib import Path
import torch
from . import checkpoint
from .task_optimization import combine_shared,private_correction,higher_order_attention,rank_and_cover
from tum_us.engine.runtime import autocast_context,append_jsonl


def alignment_entity(entity,maximum=4,offset=0):
    """Bound only the extra higher-order probe, not the main supervised sequence.

    Keep original frame ordinals and real supervision; no mask propagation.
    One real view/window per probe, rotating across entity windows.
    """
    from dataclasses import replace
    from types import SimpleNamespace
    if maximum<1: raise ValueError('Positive alignment frame limit required')
    choices=list(entity.batches)
    if entity.label is None: choices=[b for b in choices if (b.mask_valid & b.frame_valid[:,:,None]).any()]
    if not choices: raise ValueError('No supervised alignment window')
    batch=choices[offset%len(choices)]
    if len(batch)!=1: raise ValueError('Entity probe expects one case per batch')
    valid=batch.frame_valid[0].nonzero().flatten().tolist()
    if entity.label is None:
        labelled=(batch.mask_valid[0].any(-1)&batch.frame_valid[0]).nonzero().flatten().tolist()
    else: labelled=[]
    def spread(values,n):
        return [values[i] for i in torch.linspace(0,len(values)-1,min(n,len(values))).round().long().tolist()] if values else []
    keep=spread(labelled,maximum)
    for i in spread(valid,maximum)+valid:
        if len(keep)>=min(maximum,len(valid)): break
        if i not in keep: keep.append(i)
    ids=torch.tensor(sorted(keep),device=batch.media.device,dtype=torch.long)
    changes={k:getattr(batch,k).index_select(1,ids) for k in ('media','frame_valid','masks','mask_valid','frame_indices')}
    return SimpleNamespace(key=entity.key,label=entity.label,batches=[replace(batch,**changes)])


def _add(buffer,grad,weight=1.):
    for i,g in enumerate(grad):
        if g is not None:
            value=g.detach().float()*weight
            if buffer[i] is None: buffer[i]=value
            else: buffer[i].add_(value)


def _candidate_score(model,entity,endpoint,device):
    embeddings=[]; uncertainties=[]
    # Global current-frame tokens exist for both families, including native SAM2.
    def hook(module,args): embeddings.append(args[0].detach().float().reshape(-1,256).mean(0))
    handle=model.global_readout.register_forward_pre_hook(hook)
    try:
        with torch.no_grad():
            for batch in entity.batches:
                b=batch.to(device); output=model(b,endpoint=endpoint)
                if entity.label is not None:
                    p=output['class_logits'].float().softmax(-1).clamp_min(1e-8)
                    uncertainties.append(-(p*p.log()).sum(-1).mean())
                else:
                    # Uses predictions on valid input frames, never annotation
                    # coverage or ground-truth boundaries for acquisition scores.
                    valid=b.frame_valid[:,:,None] & (b.target_ids[:,None,:]>=0)
                    logits=output['mask_logits'][valid].float().reshape(-1)
                    stride=max(1,math.ceil(logits.numel()/4096)); p=logits[::stride].sigmoid().clamp(1e-6,1-1e-6)
                    entropy=-(p*p.log()+(1-p)*(1-p).log())
                    uncertainties.append(torch.quantile(entropy,.9))
    finally: handle.remove()
    if not embeddings or not uncertainties: raise ValueError('Empty candidate prediction')
    return float(torch.stack(uncertainties).mean()),torch.stack(embeddings).mean(0).cpu()


def select_training(model,datasets,settings,device,seed):
    """Resampling already labelled TRAIN examples; not an annotation-saving study."""
    selected={}; report={}; rng=checkpoint.rng_state(); was_training=model.training
    try:
        model.eval()
        for offset,(ep,ds) in enumerate(datasets.items()):
            indices=list(range(len(ds))); random.Random(seed+offset*7919).shuffle(indices)
            indices=indices[:min(len(indices),int(settings.get('candidate_limit',64)))]
            budget=max(1,math.ceil(len(indices)*float(settings.get('fraction',.5))))
            keys=[]; u=[]; f=[]
            for i in indices:
                entity=ds[i]; score,feature=_candidate_score(model,entity,ep,device)
                keys.append(entity.key); u.append(score); f.append(feature)
            choices=rank_and_cover(keys,u,torch.stack(f),budget,int(settings.get('pool_factor',4)))
            selected[ep]=[indices[i] for i in choices]
            report[ep]={'candidate_entities':len(indices),'selected_entities':budget,
                        'selected_keys':[keys[i] for i in choices], 'candidate_keys':keys,
                        'uncertainties':u,'kind':ds.spec['kind']}
    finally:
        checkpoint.restore_rng(rng); model.train(was_training)
    return selected,report


def _validate(options,batch):
    method=options.get('optimizer','consistent')
    if method not in ('mean','pcgrad','consistent'): raise ValueError('Unknown multitask optimizer')
    if batch<2 or batch%2: raise ValueError('Multitask effective_batch must be even and >=2')
    s=options.get('selection',{})
    if s.get('enabled',False):
        if not 0<float(s.get('fraction',.5))<=1: raise ValueError('Selection fraction outside (0,1]')
        if int(s.get('candidate_limit',64))<1 or int(s.get('pool_factor',4))<1: raise ValueError('Invalid selection size')
        if int(s.get('warmup_updates',200))<0 or int(s.get('refresh_every',400))<1: raise ValueError('Invalid selection schedule')
    if int(options.get('alignment_every',1))<1: raise ValueError('Invalid alignment interval')
    if int(options.get('alignment_max_frames',4))<1: raise ValueError('Invalid alignment frame bound')
    if options.get('private_update','alignment') not in ('alignment','supervised_replay'): raise ValueError('Invalid private update control')
    if float(options.get('alignment_weight',1.))<0 or not 0<=float(options.get('max_private_ratio',.1))<=1: raise ValueError('Invalid alignment strength')


def run_readers(cfg,out):
    from . import engine as e
    from .model import GeneralistModel
    out=Path(out); out.mkdir(parents=True,exist_ok=True)
    tr=cfg['training']; options=tr['multitask']; device=e._device(cfg); seed=int(tr.get('seed',3407)); e._seed(seed)
    batch=int(tr.get('effective_batch',16)); _validate(options,batch)
    names=e._names(cfg,'readers'); train=e._datasets(cfg,'train',names); dev=e._datasets(cfg,'dev',names)
    families={k:[ep for ep in names if train[ep].spec['kind']==k] for k in ('classification','segmentation')}
    if any(not v for v in families.values()): raise ValueError('Both independently labelled task families are required')
    expected={'config':e._identity(cfg),'manifests':{'train':e._fingerprints(train),'dev':e._fingerprints(dev)},
              'reader_source':e._collection_source_identity(),
              'trainer_sha256':checkpoint.file_hash(__file__),
              'optimizer_sha256':checkpoint.file_hash(Path(__file__).with_name('task_optimization.py'))}
    initialization=e._reader_initialization(cfg)
    if initialization is not None: expected['initialization']=initialization
    final=out/'readers.pt'; progress=out/'readers_progress.pt'
    if final.exists():
        _,p=checkpoint.load(final,expected_stage='readers',expected_image_size=cfg['model']['image_size'])
        if p['provenance']!=expected: raise ValueError('Multitask reader checkpoint inputs differ')
        return {'checkpoint':str(final),'reused':True,'selection':p['extra']}
    if progress.exists():
        model,p=checkpoint.load(progress,expected_stage='readers_progress',expected_image_size=cfg['model']['image_size'])
        if p['provenance']!=expected: raise ValueError('Multitask reader resume inputs differ')
    else: model=e._new_reader(cfg,initialization); p=None
    model.to(device).configure_readers()
    if p is None and model.support_rule is not None:
        from .support_profiles import initialize_support_profiles
        selected_profiles=initialize_support_profiles(model,train,cfg['support_guidance'],seed,device)
        e._json(out/'support_profiles.json',selected_profiles)
    groups=model.multitask_parameter_groups()
    ordered=[(name,param) for group in groups.values() for name,param in group]; params=[x[1] for x in ordered]
    positions={id(param):i for i,param in enumerate(params)}
    shared=[positions[id(param)] for _,param in groups['shared']]
    private={family:[positions[id(param)] for _,param in groups[family]] for family in families}
    method=options.get('optimizer','consistent')
    if method=='consistent' and any(not private[k] for k in families): raise ValueError('Consistent updates require genuinely private heads for both families')
    probe_names=options.get('probe_parameters')
    if probe_names is None:
        # Top FPN lateral convolution is common to semantic and spatial readers.
        candidates=[name for name,param in groups['shared'] if name.startswith('sam.image_encoder.neck.convs.') and name.endswith('weight')]
        if not candidates: raise ValueError('No common shared-neck probe found')
        # Neck convs are ordered from LOWEST resolution. After SAM2's scalp,
        # the last surviving FPN feature uses conv[scalp], not conv[-1].
        prefix=f'sam.image_encoder.neck.convs.{model.sam.image_encoder.scalp}.'
        probe_names=[name for name in candidates if name.startswith(prefix)]
        if not probe_names: raise ValueError('Cannot identify surviving top FPN probe')
    named=dict(ordered)
    if any(name not in named or positions[id(named[name])] not in shared for name in probe_names): raise ValueError('Probe must be trainable and shared')
    probe=[named[name] for name in probe_names]; probe_indices=[positions[id(v)] for v in probe]
    opt=torch.optim.AdamW(params,lr=float(tr.get('lr',3e-4)),weight_decay=float(tr.get('weight_decay',.01)))
    steps=int(tr.get('reader_updates',1200)); scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=steps)
    every=int(tr.get('checkpoint_every',100)); counts={ep:0 for ep in names}; cursor={family:0 for family in families}
    selected={ep:list(range(len(ds))) for ep,ds in train.items()}; last_selection=-1
    first=0; best=None; best_loss=float('inf'); best_step=0
    if p:
        opt.load_state_dict(p['optimizer']); scheduler.load_state_dict(p['scheduler'])
        c=p['cursor']; first=c['step']; counts=c['counts']; cursor=c['family_cursor']; selected=c['selected']; last_selection=c['last_selection']
        best=p['extra']['best']; best_loss=p['extra']['best_loss']; best_step=p['extra']['best_step']; checkpoint.restore_rng(p['rng'])
    elif tr.get('post_init_seed') is not None: e._seed(int(tr['post_init_seed']))
    # BF16 gradients accumulate in FP32. Devices without BF16 use FP32, avoiding
    # unscaled FP16 gradients in a manually assembled multi-task optimizer.
    amp=bool(tr.get('amp',True) and device.type=='cuda' and torch.cuda.is_bf16_supported())
    e._json(out/'multitask_partition.json',{'parameters':{k:[n for n,_ in v] for k,v in groups.items()},
        'probe_parameters':probe_names,'method':method,'precision':'bf16' if amp else 'float32',
        'family_sources':families,'adaptation':'unpaired parameter-space mixed derivative; not original ConsMTL',
        'selection_scope':'reader train resampling only; later utility stage may use all train labels'})
    start=time.perf_counter()
    for step in range(first,steps):
        step_start=time.perf_counter(); selection_seconds=0.
        s=options.get('selection',{}); warmup=int(s.get('warmup_updates',200)); refresh=int(s.get('refresh_every',400))
        if s.get('enabled',False) and step>=warmup and (last_selection<0 or step-last_selection>=refresh):
            selection_start=time.perf_counter()
            selected,report=select_training(model,train,s,device,seed+step*1009); last_selection=step
            e._json(out/f'selection_{step:06d}.json',{'step':step,'sources':report,'split':'train','conflict_penalty':False,'cross_task_mutual_information':False})
            selection_seconds=time.perf_counter()-selection_start
        model.train(); opt.zero_grad(set_to_none=True); accumulated={}; representatives={}; losses={}
        for family,eps in families.items():
            buffer=[None]*len(params); values=[]
            for offset in range(batch//2):
                ep=eps[cursor[family]%len(eps)]; cursor[family]+=1
                pool=selected[ep]; epoch,index=divmod(counts[ep],len(pool)); order=list(pool)
                random.Random(seed+epoch*1009+names.index(ep)*7919).shuffle(order); counts[ep]+=1
                entity=train[ep][order[index]]
                representatives.setdefault(family,(entity,ep))
                if ep==eps[step%len(eps)]: representatives[family]=(entity,ep)
                with autocast_context(device,amp): loss=e._entity_loss(model,entity,ep,device)
                if not torch.isfinite(loss): raise FloatingPointError('Nonfinite supervised loss')
                grads=torch.autograd.grad(loss,params,allow_unused=True)
                _add(buffer,grads,2./batch); values.append(float(loss.detach()))
                del loss,grads
            accumulated[family]=buffer; losses[family]=sum(values)/len(values)
        a,b=(accumulated[k] for k in families)
        direction,stats=combine_shared([a[i] for i in shared],[b[i] for i in shared],method)
        for i,g in zip(shared,direction): params[i].grad=g
        for family in families:
            for i in private[family]: params[i].grad=None if accumulated[family][i] is None else accumulated[family][i]*.5
        alignment={}
        alignment_start=time.perf_counter()
        if method=='consistent' and step%int(options.get('alignment_every',1))==0:
            target=[params[i].grad for i in probe_indices]; rng=checkpoint.rng_state()
            try:
                for family,(entity,ep) in representatives.items():
                    entity=alignment_entity(entity,int(options.get('alignment_max_frames',4)),step)
                    ids=private[family]
                    # Explicit FP32 math attention for actual mixed derivatives.
                    with higher_order_attention():
                        loss=e._entity_loss(model,entity,ep,device)
                        if options.get('private_update','alignment')=='supervised_replay':
                            from .task_optimization import dot
                            raw=torch.autograd.grad(loss,[params[i] for i in ids],allow_unused=True)
                            extra=[None if g is None else g.detach().float() for g in raw]
                            base=dot([accumulated[family][i] for i in ids],[accumulated[family][i] for i in ids]).sqrt()
                            size=dot(extra,extra).sqrt(); factor=(float(options.get('max_private_ratio',.1))*base/size.clamp_min(1e-20)).clamp(max=1.)
                            extra=[None if g is None else g*factor for g in extra]
                            info={'control':'extra supervised head replay','private_correction_norm':float(size*factor),'private_main_norm':float(base)}
                        else:
                            extra,info=private_correction(loss,probe,[params[i] for i in ids],target,
                                [accumulated[family][i] for i in ids],float(options.get('alignment_weight',1.)),float(options.get('max_private_ratio',.1)))
                    for i,g in zip(ids,extra):
                        if g is not None:
                            if params[i].grad is None: params[i].grad=g*.5
                            else: params[i].grad.add_(g,alpha=.5)
                    alignment[family]=info
                    del loss,extra
            finally: checkpoint.restore_rng(rng)
        norm=torch.nn.utils.clip_grad_norm_(params,float(tr.get('clip_grad_norm',1.)),error_if_nonfinite=True)
        opt.step(); scheduler.step()
        append_jsonl(out/'multitask_updates.jsonl',{'step':step+1,'supervised_loss':losses,'shared':stats,'private':alignment,'gradient_norm':float(norm),'lr':scheduler.get_last_lr()[0],
            'seconds':time.perf_counter()-step_start,'selection_seconds':selection_seconds,
            'alignment_seconds':time.perf_counter()-alignment_start if alignment else 0.,
            'main_entity_exposures':batch,'extra_private_entity_exposures':len(alignment)})
        if (step+1)%every==0 or step+1==steps:
            score,metrics=e._macro(model,dev,device,seed=seed)
            if not math.isfinite(score): raise FloatingPointError('Nonfinite development selection score')
            if score<best_loss: best_loss=score; best=copy.deepcopy({k:v.cpu() for k,v in model.state_dict().items()}); best_step=step+1; e._json(out/'readers_dev_best.json',metrics)
            checkpoint.save(progress,model,cfg,stage='readers_progress',optimizer=opt,scheduler=scheduler,
                cursor={'step':step+1,'counts':counts,'family_cursor':cursor,'selected':selected,'last_selection':last_selection},
                provenance=expected,extra={'best':best,'best_loss':best_loss,'best_step':best_step})
    if best is None: raise ValueError('No selectable reader checkpoint')
    model.load_state_dict(best); checkpoint.save(final,model,cfg,stage='readers',provenance=expected,
        extra={'best_step':best_step,'macro_loss':best_loss,'training_updates':steps,'actual_wall_seconds':time.perf_counter()-start,
               'optimizer':method,'selected_train_indices':selected,'last_selection_update':last_selection})
    return {'checkpoint':str(final),'reused':False,'best_step':best_step,'macro_loss':best_loss,'optimizer':method}
