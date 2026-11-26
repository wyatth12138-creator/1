"""Training utilities used by the project's train.py and test.py scripts."""
from .config import artifact_path, load_config, resolve_config

__all__ = ["load_config", "resolve_config", "artifact_path", "build_utility", "train_controller"]


def __getattr__(name):
    if name in {"build_utility", "train_controller"}:
        from . import utility
        return getattr(utility, name)
    raise AttributeError(name)
