"""Atomic resumable checkpoints for native TUM generalist training."""
import copy, hashlib, os, random, tempfile
from dataclasses import asdict, is_dataclass
from pathlib import Path
import numpy as np
import torch

FORMAT='tum_us.generalist.v1'
_READER_HASH_CACHE={}

def file_hash(path):
    h=hashlib.sha256()
    with open(path,'rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''): h.update(chunk)
    return h.hexdigest()

def reader_hash(state):
    h=hashlib.sha256()
    for name,value in sorted(state.items()):
        if name.startswith('controller.'): continue
        tensor=value.detach().cpu().contiguous(); h.update(name.encode()); h.update(str(tensor.dtype).encode()); h.update(str(tuple(tensor.shape)).encode()); h.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()

def rng_state():
    return {'python':random.getstate(),'numpy':np.random.get_state(),'torch':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}

def restore_rng(state):
    random.setstate(state['python']); np.random.set_state(state['numpy']); torch.set_rng_state(state['torch'].cpu())
    if torch.cuda.is_available() and state.get('cuda') is not None: torch.cuda.set_rng_state_all([s.cpu() for s in state['cuda']])

def atomic_save(payload,path):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    fd,tmp=tempfile.mkstemp(dir=path.parent,prefix=path.name+'.',suffix='.tmp')
    try:
        with os.fdopen(fd,'wb') as stream:
            torch.save(payload,stream); stream.flush(); os.fsync(stream.fileno())
        os.replace(tmp,path)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)
    return str(path)

def save(path,model,cfg,*,stage,optimizer=None,scheduler=None,scaler=None,cursor=None,parents=None,provenance=None,extra=None):
    mc=getattr(model,'model_config',None)
    if mc is None: mc=getattr(model,'config',cfg['model'])
    if is_dataclass(mc): mc=asdict(mc)
    if mc.get('image_size',512)!=cfg.get('model',{}).get('image_size',mc.get('image_size',512)):
        raise ValueError('Model and saved training configuration use different image_size')
    parent_paths={str(k):str(Path(v).resolve()) for k,v in (parents or {}).items()}; parent_hashes={k:file_hash(v) for k,v in parent_paths.items()}
    state=model.state_dict(); storage={'kind':'full'}
    if stage in ('controller','controller_progress') and 'reference' in parent_paths:
        parent=parent_paths['reference']; sha=parent_hashes['reference']
        if Path(parent)==Path(path).resolve(): raise ValueError('Checkpoint cannot be its own parent')
        if sha not in _READER_HASH_CACHE:
            base=_read(parent); _READER_HASH_CACHE[sha]=reader_hash(base['model'])
        current_hash=reader_hash(state)
        if current_hash!=_READER_HASH_CACHE[sha]: raise ValueError('Frozen reader differs from checkpoint parent')
        storage={'kind':'controller_only','parent_path':parent,'parent_sha256':sha,'reader_sha256':current_hash}
        state={k:v for k,v in state.items() if k.startswith('controller.')}
        if not state: raise ValueError('No controller tensors to save')
    payload={'format':FORMAT,'model_config':copy.deepcopy(mc),'controller_config':copy.deepcopy(getattr(model,'controller_config',cfg.get('controller',{}))),
             'model':state,'storage':storage,'config':copy.deepcopy(cfg),'stage':stage,'cursor':cursor or {},'rng':rng_state(),
             'optimizer':optimizer.state_dict() if optimizer is not None else None,'scheduler':scheduler.state_dict() if scheduler is not None else None,'scaler':scaler.state_dict() if scaler is not None else None,
             'parents':parent_hashes,'provenance':provenance or {},'extra':extra or {}}
    return atomic_save(payload,path)

def _read(path,seen=None):
    path=Path(path).resolve(); seen=set() if seen is None else set(seen)
    if path in seen: raise ValueError('Checkpoint parent cycle')
    seen.add(path)
    payload=torch.load(path,map_location='cpu',weights_only=False)
    if payload.get('format')!=FORMAT: raise ValueError('Invalid generalist checkpoint format')
    storage=payload.get('storage',{'kind':'full'})
    if storage['kind']=='controller_only':
        parent=Path(storage['parent_path']).resolve()
        if file_hash(parent)!=storage['parent_sha256'] or payload['parents'].get('reference')!=storage['parent_sha256']: raise ValueError('Checkpoint parent hash mismatch')
        base=_read(parent,seen)
        if reader_hash(base['model'])!=storage['reader_sha256']: raise ValueError('Frozen reader identity mismatch')
        state=base['model'].copy(); expected={k for k in state if k.startswith('controller.')}
        if set(payload['model'])!=expected: raise ValueError('Controller tensor keys differ from parent')
        state.update(payload['model']); payload['model']=state
    elif storage['kind']!='full': raise ValueError('Unsupported checkpoint storage format')
    return payload

def load(path,model_cls=None,expected_stage=None,expected_parents=None,*,image_size=None,expected_image_size=None):
    """Load tensors at their saved size, or explicitly initialize another size.

    The returned payload retains the original checkpoint metadata. Exact
    continuation uses expected_image_size; image_size only changes construction.
    """
    if model_cls is None:
        from .model import GeneralistModel
        model_cls=GeneralistModel
    payload=_read(path)
    if expected_stage is not None and payload['stage']!=expected_stage: raise ValueError('Wrong checkpoint stage')
    if expected_parents is not None and payload['parents']!={str(k):file_hash(v) for k,v in expected_parents.items()}: raise ValueError('Checkpoint parent mismatch')
    saved_config=copy.deepcopy(payload['model_config'])
    saved_size=saved_config.get('image_size',512)
    if expected_image_size is not None and saved_size!=expected_image_size:
        raise ValueError(f'Checkpoint image_size {saved_size} differs from requested {expected_image_size}; use training.init_checkpoint for scale adaptation')
    if image_size is not None:
        if type(image_size) is not int or image_size<32 or image_size%32:
            raise ValueError('image_size must be a positive multiple of 32')
        saved_config['image_size']=image_size
    else:
        saved_config.setdefault('image_size',saved_size)
    model=model_cls(saved_config,payload['controller_config']); model.load_state_dict(payload['model'],strict=True)
    return model,payload
