"""Full experiment checkpoints. Loading uses pickle only for our own checkpoint format."""
from __future__ import annotations

import os
import hashlib
import json
import random
from dataclasses import asdict, is_dataclass
from pathlib import Path

import numpy as np
import torch

FORMAT = "tum_us.experiment.v1"


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def capture_rng() -> dict:
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}


def restore_rng(state: dict) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if torch.cuda.is_available() and state.get("cuda") is not None:
        torch.cuda.set_rng_state_all([item.cpu() for item in state["cuda"]])


def save_checkpoint(path, model, *, optimizer=None, scheduler=None, scaler=None,
                    phase: str, config: dict, registry: dict,
                    epoch: int = 0, batch_in_epoch: int = 0, global_step: int = 0,
                    extra: dict | None = None) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    model_config = getattr(model, "config", {})
    if is_dataclass(model_config):
        model_config = asdict(model_config)
    payload = {"format": FORMAT, "model": model.state_dict(), "model_config": model_config,
               "config": config, "registry": registry, "phase": phase,
               "optimizer": optimizer.state_dict() if optimizer is not None else None,
               "scheduler": scheduler.state_dict() if scheduler is not None else None,
               "scaler": scaler.state_dict() if scaler is not None else None,
               "epoch": int(epoch), "batch_in_epoch": int(batch_in_epoch),
               "global_step": int(global_step), "rng": capture_rng(),
               "trainable_parameters": [name for name, p in model.named_parameters() if p.requires_grad],
               "extra": extra or {}}
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)
    # An engine-written sidecar is checked BEFORE invoking the pickle-based loader.
    # This keeps official SAM files on their separate weights_only loader path.
    metadata = path.with_name(path.name + ".metadata.json")
    metadata_tmp = metadata.with_name(metadata.name + ".tmp")
    with metadata_tmp.open("w", encoding="utf-8") as stream:
        json.dump({"format": FORMAT, "sha256": _digest(path)}, stream)
    os.replace(metadata_tmp, metadata)
    return path


def read_checkpoint(path: str | Path) -> dict:
    # This loader is intentionally limited to experiment files made by this engine.
    # Official SAM checkpoints are loaded by the model's own backbone loader.
    path = Path(path)
    metadata_path = path.with_name(path.name + ".metadata.json")
    if not metadata_path.is_file():
        raise ValueError("Missing engine checkpoint metadata sidecar; official SAM weights belong in model.checkpoint.")
    with metadata_path.open(encoding="utf-8") as stream:
        metadata = json.load(stream)
    if metadata.get("format") != FORMAT or metadata.get("sha256") != _digest(path):
        raise ValueError("Checkpoint provenance/integrity verification failed.")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or payload.get("format") != FORMAT:
        raise ValueError("Not a TUM-US experiment checkpoint; use model.checkpoint for official SAM weights.")
    return payload


def load_checkpoint(path, model, *, optimizer=None, scheduler=None, scaler=None,
                    expected_phase: str | None = None, registry: dict | None = None,
                    restore_randomness: bool = True, expected_image_size: int | None = None) -> dict:
    payload = read_checkpoint(path)
    if expected_phase and payload["phase"] != expected_phase:
        raise ValueError(f"Expected {expected_phase} checkpoint, got {payload['phase']}")
    if registry is not None and payload.get("registry") != registry:
        raise ValueError("Checkpoint task/target registry differs from the current registry.")
    saved_size=payload.get('model_config',{}).get('image_size')
    if saved_size is None:
        saved_size=payload.get('config',{}).get('model',{}).get('image_size')
    current_config=getattr(model,'config',{})
    if is_dataclass(current_config): current_config=asdict(current_config)
    model_size=current_config.get('image_size') if isinstance(current_config,dict) else getattr(current_config,'image_size',None)
    current_size=expected_image_size if expected_image_size is not None else model_size
    if saved_size is not None and current_size is not None and saved_size!=current_size:
        raise ValueError(f'Checkpoint image_size {saved_size} differs from current {current_size}; initialize a new training run for scale adaptation')
    model.load_state_dict(payload["model"], strict=True)
    for name, target in (("optimizer", optimizer), ("scheduler", scheduler), ("scaler", scaler)):
        if target is not None and payload.get(name) is not None:
            target.load_state_dict(payload[name])
    if restore_randomness:
        restore_rng(payload["rng"])
    return payload
