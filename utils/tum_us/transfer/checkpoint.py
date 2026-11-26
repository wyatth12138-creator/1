"""Atomic checkpoints; frozen SAM tensors are reconstructed from a hashed base."""
import copy, hashlib, json, os, random, tempfile
from pathlib import Path
import torch
import numpy as np

FORMAT = 'tum_us.transfer.v1'
COMPACT_STORAGE = 'frozen_sam_base.v1'
FULL_STORAGE = 'full.v1'

def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, default=str, separators=(',', ':')).encode()).hexdigest()

def file_hash(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()

def rng_state():
    return {'python': random.getstate(), 'numpy': np.random.get_state(), 'torch': torch.get_rng_state(), 'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}

def restore_rng(state):
    random.setstate(state['python']); np.random.set_state(state['numpy']); torch.set_rng_state(state['torch'].cpu())
    if state.get('cuda') is not None and torch.cuda.is_available(): torch.cuda.set_rng_state_all(state['cuda'])

def _base_checkpoint(spec):
    """Use TUMModel's resolution rule, including relocatable relative paths."""
    config = spec.get('model_config')
    if not isinstance(config, dict):
        raise ValueError('Compact SAM checkpoint requires model_config in topology')
    filename = config.get('checkpoint') or f"weights/sam2.1_hiera_{config.get('backbone', 'small')}.pt"
    from tum_us.paths import pretrained_path
    return pretrained_path(filename)

def _layout(state):
    return {name: {'shape': list(value.shape), 'dtype': str(value.dtype)} for name, value in state.items()}

def _validate_layout(state, expected, context):
    actual = _layout(state)
    missing = sorted(set(expected) - set(actual))
    unexpected = sorted(set(actual) - set(expected))
    mismatched = sorted(k for k in set(actual) & set(expected) if actual[k] != expected[k])
    if missing or unexpected or mismatched:
        raise ValueError(f'{context} layout mismatch: missing={missing}, unexpected={unexpected}, shape/dtype={mismatched}')

def _tensor_bytes(state):
    return sum(t.numel() * t.element_size() for t in state.values())

def _compact_extra(extra, full_state):
    """A's best snapshot also contains SAM; preserve its RNG and selected step."""
    result = dict(extra or {})
    best = result.get('best')
    if best is None: return result, False
    if not isinstance(best, (list, tuple)) or len(best) != 3 or not isinstance(best[0], dict):
        raise ValueError('Malformed A best-model snapshot')
    _validate_layout(best[0], _layout(full_state), 'A best-model state')
    for name, tensor in full_state.items():
        if name.startswith('sam.') and not torch.equal(tensor, best[0][name].to(tensor.device)):
            raise ValueError(f'Frozen SAM differs from A best snapshot: {name}')
    result['best'] = ({k: v for k, v in best[0].items() if not k.startswith('sam.')}, best[1], best[2])
    return result, True

def _storage_payload(model, spec, state, extra):
    sam_state = {k: v for k, v in state.items() if k.startswith('sam.')}
    if not sam_state: return state, {'kind': FULL_STORAGE}, extra or {}
    sam = getattr(model, 'sam', None)
    if sam is None or any(p.requires_grad for p in sam.parameters()):
        raise ValueError('Compact checkpoint requires every original SAM parameter to be frozen')
    base = _base_checkpoint(spec)
    if not base.is_file(): raise FileNotFoundError(f'Official SAM base checkpoint missing: {base}')
    non_sam = {k: v for k, v in state.items() if not k.startswith('sam.')}
    compact_extra, compact_best = _compact_extra(extra, state)
    storage = {'kind': COMPACT_STORAGE,
        'base': {'sha256': file_hash(base), 'size_bytes': base.stat().st_size,
                 'checkpoint': spec['model_config'].get('checkpoint') or
                    f"weights/sam2.1_hiera_{spec['model_config'].get('backbone', 'small')}.pt"},
        'sam_layout': _layout(sam_state), 'non_sam_layout': _layout(non_sam),
        'compact_A_best': compact_best,
        'tensor_bytes': {'saved_current_state': _tensor_bytes(non_sam),
                        'omitted_current_sam': _tensor_bytes(sam_state),
                        'omitted_A_best_sam': _tensor_bytes(sam_state) if compact_best else 0}}
    return non_sam, storage, compact_extra

def save_checkpoint(path, model, config, *, stage, parents=None, manifests=None, optimizer=None, scheduler=None, cursor=None, extra=None):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    parent_hashes = {str(k): file_hash(v) for k, v in (parents or {}).items()}
    spec = model.export_spec()
    state, storage, compact_extra = _storage_payload(model, spec, model.state_dict(), extra)
    payload = {'format': FORMAT, 'spec': spec, 'state_dict': state, 'state_storage': storage, 'config': copy.deepcopy(config), 'config_hash': digest(config), 'manifests': manifests or {}, 'parents': parent_hashes, 'stage': stage, 'rng': rng_state(), 'optimizer': optimizer.state_dict() if optimizer else None, 'scheduler': scheduler.state_dict() if scheduler else None, 'cursor': cursor or {}, 'extra': compact_extra}
    fd, tmp = tempfile.mkstemp(prefix=path.name + '.', suffix='.tmp', dir=path.parent); os.close(fd)
    try:
        with open(tmp, 'wb') as f:
            torch.save(payload, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)
    return str(path)

def load_checkpoint(path, model_cls=None, expected_config=None, expected_parents=None):
    if model_cls is None:
        from .model import TransferModel
        model_cls = TransferModel
    payload = torch.load(path, map_location='cpu', weights_only=False)
    if payload.get('format') != FORMAT: raise ValueError('Unsupported transfer checkpoint')
    if digest(payload['config']) != payload['config_hash']: raise ValueError('Checkpoint config hash mismatch')
    if expected_config is not None and digest(expected_config) != payload['config_hash']: raise ValueError('Configuration differs from checkpoint')
    if expected_parents is not None and {str(k): file_hash(v) for k, v in expected_parents.items()} != payload['parents']: raise ValueError('Parent checkpoint hash mismatch')
    # Missing metadata is the legacy full v1 format; keep strict compatibility.
    storage = payload.get('state_storage', {'kind': FULL_STORAGE})
    kind = storage.get('kind')
    if kind not in (FULL_STORAGE, COMPACT_STORAGE):
        raise ValueError(f'Unsupported checkpoint state storage: {kind}')
    if kind == COMPACT_STORAGE:
        base = _base_checkpoint(payload['spec'])
        if not base.is_file(): raise FileNotFoundError(f'Official SAM base checkpoint missing: {base}')
        if base.stat().st_size != storage['base']['size_bytes'] or file_hash(base) != storage['base']['sha256']:
            raise ValueError('Official SAM base SHA256 mismatch; refusing topology reconstruction')
        _validate_layout(payload['state_dict'], storage['non_sam_layout'], 'Saved non-SAM state')
        if any(k.startswith('sam.') for k in payload['state_dict']):
            raise ValueError('Compact checkpoint unexpectedly contains original SAM tensors')
    model = model_cls.from_spec(payload['spec'])
    if kind == COMPACT_STORAGE:
        initialized = model.state_dict()
        sam_state = {k: v for k, v in initialized.items() if k.startswith('sam.')}
        non_sam = {k: v for k, v in initialized.items() if not k.startswith('sam.')}
        _validate_layout(sam_state, storage['sam_layout'], 'Reconstructed original SAM')
        _validate_layout(non_sam, storage['non_sam_layout'], 'Reconstructed non-SAM topology')
        full_state = {**sam_state, **payload['state_dict']}
        _validate_layout(full_state, _layout(initialized), 'Expanded checkpoint')
        model.load_state_dict(full_state, strict=True)
        # Keep loaded trainable tensors independent of subsequent optimizer writes.
        payload['state_dict'] = full_state
        if storage.get('compact_A_best'):
            best = payload['extra'].get('best')
            if not isinstance(best, (tuple, list)) or len(best) != 3:
                raise ValueError('Compact A best snapshot missing')
            _validate_layout(best[0], storage['non_sam_layout'], 'A best non-SAM state')
            payload['extra']['best'] = ({**sam_state, **best[0]}, best[1], best[2])
    else:
        model.load_state_dict(payload['state_dict'], strict=True)
    return model, payload

