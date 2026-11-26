import copy, math, random, time
from contextlib import nullcontext
import torch
import numpy as np
from .checkpoint import restore_rng, rng_state
from .selection import score_row, weighted_difference


def device_for(cfg):
    d = cfg['training'].get('device', 'auto')
    return torch.device(('cuda' if torch.cuda.is_available() else 'cpu') if d == 'auto' else d)

def clone_model(model, device):
    out = type(model).from_spec(model.export_spec()); out.load_state_dict(copy.deepcopy(model.state_dict()), strict=True)
    return out.to(device)

_NORMALIZATION = {}
_COSTS = {'normalization_seconds':0.0,'normalization_runs':0,'normalization_cache_hits':0}

def _dataset(cfg, endpoint, split='train', labels=True):
    from .data import EndpointDataset
    ds=EndpointDataset(cfg['endpoints'][endpoint], split, cfg, labels_allowed=labels)
    if split == 'train':
        key=(endpoint,ds.manifest_fingerprint,tuple(k for k,_ in ds.groups),str(cfg.get('model',{}).get('image_size')))
        if key not in _NORMALIZATION:
            source_spec=dict(cfg['endpoints'][endpoint])
            if labels and source_spec.get('label_budget') is not None: source_spec['subset_keys']=[key for key,_ in ds.groups]
            source=EndpointDataset(source_spec, 'train', cfg, labels_allowed=False)
            started=time.perf_counter(); _NORMALIZATION[key]=source.compute_normalization(); _COSTS['normalization_seconds']+=time.perf_counter()-started; _COSTS['normalization_runs']+=1
        else: _COSTS['normalization_cache_hits']+=1
        ds.normalizer,ds.mean_rgb=_NORMALIZATION[key]
    return ds

def _evaluate(model, dataset, endpoint, device):
    from .data import evaluate_entities
    return evaluate_entities(model, dataset, endpoint, device)

def _seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

class Stream:
    def __init__(self, dataset, seed):
        if not len(dataset): raise ValueError('Empty training dataset')
        self.dataset, self.seed, self.cursor = dataset, seed, 0
    def next(self):
        epoch, offset = divmod(self.cursor, len(self.dataset))
        order = list(range(len(self.dataset))); random.Random(self.seed + epoch).shuffle(order)
        self.cursor += 1
        return self.dataset[order[offset]]

def _lr_factor(step, total, warmup):
    warm = max(1, int(total * warmup))
    if step < warm: return (step+1)/warm
    return .5*(1+math.cos(math.pi*(step-warm)/max(1,total-warm)))

def train_joint(model, cfg, endpoint, old_endpoints=(), *, cycles=None, phase='joint', supervised=True, old_replay=True, full_cycles=None, initial_rng=None, callback=None, resume=None):
    from .data import supervised_entity_loss
    tr = cfg['training']; device = device_for(cfg); model.to(device)
    if initial_rng is not None: restore_rng(initial_rng)
    model.train(); model.configure_training(endpoint, phase=phase)
    params = [p for p in model.parameters() if p.requires_grad]
    if not params: raise ValueError('No trainable parameters')
    opt = torch.optim.AdamW(params, lr=tr.get('lr',3e-4), weight_decay=tr.get('weight_decay',.01))
    cycles = int(cycles if cycles is not None else tr.get('joint_cycles',600))
    full_cycles = int(full_cycles if full_cycles is not None else cycles)
    datasets = {endpoint: _dataset(cfg, endpoint, labels=supervised)}
    for old in old_endpoints: datasets[old] = _dataset(cfg, old)
    # Probe each old route once without advancing its sampler or RNG.  A frozen
    # route cannot update any parameter, so it receives no optimizer step.
    if device.type == 'cuda': torch.cuda.synchronize(device)
    preflight_start=time.perf_counter(); prior_rng=rng_state(); eligible={}
    try:
        for old in old_endpoints:
            eligible[old]=bool(supervised_entity_loss(model,datasets[old][0],old,device).requires_grad) if old_replay else False
    finally:
        restore_rng(prior_rng)
    if device.type == 'cuda': torch.cuda.synchronize(device)
    preflight_seconds=time.perf_counter()-preflight_start
    scheduled_old=sum(int(eligible[old_endpoints[i % len(old_endpoints)]]) for i in range(full_cycles)) if old_replay and old_endpoints else 0
    nsteps=full_cycles+scheduled_old
    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lambda step: _lr_factor(step,nsteps,tr.get('warmup_fraction',.05)))
    seed = int(tr.get('seed',6000)); streams = {e: Stream(d, seed + 101*i) for i,(e,d) in enumerate(datasets.items())}
    effective = int(tr.get('effective_batch',16)); weight = float(tr.get('mim_weight',.1))
    counts = {'cycles': 0, 'new_updates': 0, 'old_updates': 0, 'old_skipped': 0, 'optimizer_updates': 0, 'entities': 0, 'mim_forwards': 0}
    if device.type == 'cuda': torch.cuda.synchronize(device); torch.cuda.reset_peak_memory_stats(device)
    start=time.perf_counter()
    first_cycle=0
    if resume is not None:
        model.load_state_dict(resume['state_dict'],strict=True); opt.load_state_dict(resume['optimizer']); scheduler.load_state_dict(resume['scheduler']); restore_rng(resume['rng'])
        counts.update(resume['extra']['counts']); first_cycle=counts['cycles']
        for key,cursor in resume['cursor'].items(): streams[key].cursor=cursor
    def update(ep, use_mim, labels):
        opt.zero_grad(set_to_none=True)
        for _ in range(effective):
            ent = streams[ep].next(); loss = None
            context=torch.autocast(device_type='cuda',dtype=torch.bfloat16) if device.type=='cuda' and tr.get('amp',True) else nullcontext()
            with context:
                if labels: loss = supervised_entity_loss(model, ent, ep, device)
                if use_mim:
                    ds=datasets[ep]; mask_rng=torch.Generator(device='cpu').manual_seed(seed + 100003*streams[ep].cursor); recon = sum(model.reconstruction_loss(batch.to(device), ep, generator=mask_rng, normalizer=ds.normalizer, mean_rgb=ds.mean_rgb) for batch in ent.batches)/len(ent.batches)
                    loss = weight*recon if loss is None else loss+weight*recon
                    counts['mim_forwards'] += len(ent.batches)
            if loss is None: raise ValueError('Empty update')
            if not loss.requires_grad: raise RuntimeError('Route changed gradient eligibility during training')
            (loss/effective).backward(); counts['entities'] += 1
        torch.nn.utils.clip_grad_norm_(params, float(tr.get('clip_grad',1.0)), error_if_nonfinite=True); opt.step(); scheduler.step(); counts['optimizer_updates'] += 1
    for cycle in range(first_cycle,cycles):
        update(endpoint, True, supervised); counts['new_updates'] += 1
        if old_replay and old_endpoints:
            old=old_endpoints[cycle % len(old_endpoints)]
            if eligible[old]:
                update(old, False, True); counts['old_updates'] += 1
            else: counts['old_skipped'] += 1
        counts['cycles'] += 1
        if callback: callback(model, opt, scheduler, counts, {k:s.cursor for k,s in streams.items()})
    if device.type == 'cuda': torch.cuda.synchronize(device)
    counts['seconds'] = time.perf_counter()-start
    counts['eligibility_seconds']=preflight_seconds
    counts['eligible_old_routes']=eligible
    counts['peak_cuda_bytes'] = torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else 0
    return model, counts

def warm_head(model, cfg, endpoint, *, supervised=True, initial_rng=None):
    return train_joint(model,cfg,endpoint,cycles=cfg['training'].get('head_updates',100),phase='head', supervised=supervised,initial_rng=initial_rng)

def train_A(model,cfg,callback=None,resume=None):
    from .data import supervised_entity_loss
    _seed(cfg['training'].get('seed',6000)); device=device_for(cfg); model.to(device); model.configure_training('A',phase='joint')
    tr=cfg['training']; ds=_dataset(cfg,'A'); dev=_dataset(cfg,'A','dev'); stream=Stream(ds,tr.get('seed',6000)); params=[p for p in model.parameters() if p.requires_grad]
    opt=torch.optim.AdamW(params,lr=tr.get('lr',3e-4),weight_decay=tr.get('weight_decay',.01)); updates=int(tr.get('A_updates',1200)); every=int(tr.get('checkpoint_every',100)); batch=int(tr.get('effective_batch',16)); best=None; best_score=float('inf'); start=time.perf_counter()
    scheduler=torch.optim.lr_scheduler.LambdaLR(opt,lambda step:_lr_factor(step,updates,tr.get('warmup_fraction',.05)))
    first_step=0
    if resume is not None:
        model.load_state_dict(resume['state_dict'],strict=True); opt.load_state_dict(resume['optimizer']); scheduler.load_state_dict(resume['scheduler']); restore_rng(resume['rng'])
        stream.cursor=resume['cursor']['A']; first_step=resume['extra']['step']; best=resume['extra']['best']; best_score=resume['extra']['best_score']
    for step in range(first_step,updates):
        model.train(); opt.zero_grad(set_to_none=True)
        for _ in range(batch):
            context=torch.autocast(device_type='cuda',dtype=torch.bfloat16) if device.type=='cuda' and tr.get('amp',True) else nullcontext()
            with context: loss=supervised_entity_loss(model,stream.next(),'A',device)/batch
            loss.backward()
        torch.nn.utils.clip_grad_norm_(params,float(tr.get('clip_grad',1.0)),error_if_nonfinite=True); opt.step(); scheduler.step()
        if (step+1)%every == 0 or step+1 == updates:
            result=_evaluate(model,dev,'A',device); score=result['metrics']['brier']
            if score < best_score: best_score=score; best=(copy.deepcopy(model.state_dict()),rng_state(),step+1)
        if callback and ((step+1)%every==0 or step+1==updates): callback(model,opt,scheduler,{'A':stream.cursor},{'step':step+1,'best':best,'best_score':best_score})
    model.load_state_dict(best[0]); restore_rng(best[1])
    return model, {'updates':updates,'selected_update':best[2],'dev_brier':best_score,'seconds':time.perf_counter()-start}

def train_B_pair(parent,cfg):
    device=device_for(cfg); base_rng=rng_state(); result={}
    for supervised, name in [(False,'minus'),(True,'plus')]:
        m=clone_model(parent,device); restore_rng(base_rng); m.add_endpoint('B',cfg['endpoints']['B'])
        m,w=warm_head(m,cfg,'B',supervised=supervised,initial_rng=base_rng)
        m,c=train_joint(m,cfg,'B',['A'],supervised=supervised)
        result[name]=(m,{'warm':w,'joint':c})
    return result['minus'],result['plus']

def probe(model,cfg,endpoint,old_endpoints,initial_rng=None):
    return train_joint(model,cfg,endpoint,old_endpoints,cycles=cfg['training'].get('probe_cycles',60),full_cycles=cfg['training'].get('joint_cycles',600),initial_rng=initial_rng)

def restore_and_score(model, reference, cfg, old_endpoints, endpoint, gain):
    device=device_for(cfg); datasets={e:_dataset(cfg,e,'dev') for e in list(old_endpoints)+[endpoint]}
    baseline={e:_evaluate(model,ds,e,device) for e,ds in datasets.items()}; rows=[]
    for block in range(3,16):
        with model.restore_modules(reference,[block]):
            recovered={e:_evaluate(model,ds,e,device) for e,ds in datasets.items()}
        delta={}; entity=[]
        for ep in old_endpoints:
            old={r['key']:r for r in baseline[ep]['records']}; new={r['key']:r for r in recovered[ep]['records']}
            if old.keys()!=new.keys(): raise ValueError('Recovery evaluation entity mismatch')
            vals=[old[k]['loss']-new[k]['loss'] for k in old]; delta[ep]=sum(vals)/len(vals)
            if ep=='A': entity=[{'key':k,'recovery':old[k]['loss']-new[k]['loss']} for k in old]
        a,support=weighted_difference(entity,gain); row=score_row(block,delta,a,support)
        pre=baseline[endpoint]['records']; post={r['key']:r for r in recovered[endpoint]['records']}
        row['new_recovery']=sum(r['loss']-post[r['key']]['loss'] for r in pre)/len(pre); rows.append(row)
    return rows










