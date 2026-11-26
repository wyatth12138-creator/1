"""Shared source-tree/installed-package paths and legacy pretrained resolution."""
from pathlib import Path
_parent=Path(__file__).resolve().parent.parent
ROOT=_parent.parent if _parent.name=='utils' else _parent
MODEL_ROOT=ROOT/('model' if (ROOT/'model/core').is_dir() else '_tum_us_assets')
VENDOR_ROOT=MODEL_ROOT/'vendor'

def pretrained_path(value):
    path=Path(value).expanduser()
    if not path.is_absolute():
        if path.parts and path.parts[0]=='weights': path=Path('pre_checkpoints',*path.parts[1:])
        path=ROOT/path
    legacy=ROOT/'weights'
    if path.is_relative_to(legacy): path=ROOT/'pre_checkpoints'/path.relative_to(legacy)
    return path.resolve()

def source_files():
    """Hash actual implementations, including model files outside the namespace bridge."""
    paths=list(Path(__file__).resolve().parent.rglob('*.py'))
    for part in ('core','generalist','transfer'): paths+=list((MODEL_ROOT/part).rglob('*.py'))
    paths+=list((ROOT/'utils').glob('*.py'))
    paths += [ROOT/'train.py',ROOT/'test.py']
    return {str(p.relative_to(ROOT)).replace('\\','/'):p for p in sorted(set(paths)) if p.is_file()}
