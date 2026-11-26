"""Configuration and output paths for reproducible TUM-US experiments."""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path
from tum_us.paths import ROOT, pretrained_path
from typing import Any


DEFAULTS: dict[str, Any] = {
    "project_root": str(ROOT),
    "data_root": "data",
    "manifest": "dataset_task_manifest/dataset_task_manifest.jsonl",
    "output_root": None,
    "model": {"backbone": "small", "image_size": 256, "memory_capacity": 4,
              "policy": "random", "checkpoint": "pre_checkpoints/sam2.1_hiera_small.pt",
              "freeze_trunk": True},
    "data": {"tasks": ["image_seg", "image_cls", "video_seg", "ceus_seg", "ceus_cls"],
             "privacy": "all", "clip_length": 8, "dataset_ids": None,
             "max_records": None, "train_split": "train", "val_split": "val",
             "test_split": "test", "single_mask_policy": "center_reference"},
    "training": {"run_name": "stage1", "epochs": 20, "batch_size": 1,
                 "learning_rate": 0.0001, "fpn_learning_rate": 0.00001, "weight_decay": 0.01,
                 "gradient_accumulation": 4, "clip_grad_norm": 1.0,
                 "amp": True, "seed": 3407, "num_workers": 0,
                 "device": "auto", "steps_per_epoch": None,
                 "max_steps": None, "max_val_batches": None,
                 "validate_every": 1, "checkpoint_every": 1,
                 "visualize_samples": 4, "resume": None},
    "evaluation": {"split": "test", "max_batches": None, "threshold": 0.5,
                   "hd95": False, "classification_unit": "case"},
    "utility": {"name": "utility_train", "split": "train", "max_batches": None,
                "max_states": 4, "reference_checkpoint": None,
                "max_frames_per_group": None},
    "controller": {"run_name": "stage2", "epochs": 20, "batch_size": 16,
                   "learning_rate": 0.0003, "weight_decay": 0.01,
                   "temperature": 0.2, "regression_weight": 1.0,
                   "ranking_weight": 0.25, "scale_epsilon": 0.000001,
                   "utility_path": None, "resume": None,
                   "reference_checkpoint": None},
}


def _merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def resolve_path(value: str | Path, base: str | Path) -> Path:
    path = Path(os.path.expandvars(os.path.expanduser(str(value))))
    return path.resolve() if path.is_absolute() else (Path(base) / path).resolve()


def load_config(path: str | Path | None = None, *, settings: dict | None = None) -> dict[str, Any]:
    raw: dict[str, Any] = {}
    if path is not None:
        config_path = Path(path).resolve()
        with config_path.open("r", encoding="utf-8-sig") as stream:
            if config_path.suffix.lower() in {".yaml", ".yml"}:
                try:
                    import yaml
                except ImportError as exc:
                    raise RuntimeError("YAML configuration requires PyYAML; JSON needs no extra dependency.") from exc
                raw = yaml.safe_load(stream) or {}
            else:
                raw = json.load(stream)
        if not isinstance(raw, dict):
            raise ValueError("Configuration must be an object/mapping.")
    cfg = _merge(_merge(DEFAULTS, settings or {}), raw)
    root = resolve_path(cfg["project_root"], Path.cwd())
    cfg["project_root"] = str(root)
    cfg["data_root"] = str(resolve_path(cfg["data_root"], root))
    cfg["manifest"] = str(resolve_path(cfg["manifest"], cfg["data_root"]))
    output = resolve_path(cfg["output_root"] or ".", root)
    if not output.is_relative_to(root):
        raise ValueError(f"All experiment outputs must stay inside project_root: {root}")
    cfg["output_root"] = str(output)
    if cfg["model"].get("checkpoint"):
        cfg["model"]["checkpoint"] = str(pretrained_path(resolve_path(cfg["model"]["checkpoint"], root)))
    for section, keys in (("training", ("resume",)),
                          ("utility", ("reference_checkpoint",)),
                          ("controller", ("resume", "reference_checkpoint", "utility_path"))):
        for key in keys:
            if cfg[section].get(key):
                cfg[section][key] = str(resolve_path(cfg[section][key], root))
    if int(cfg["training"]["batch_size"]) < 1 or int(cfg["training"]["gradient_accumulation"]) < 1:
        raise ValueError("batch_size and gradient_accumulation must be positive.")
    if int(cfg["model"]["memory_capacity"]) < 2:
        raise ValueError("memory_capacity includes an anchor and must be at least two.")
    return cfg


def resolve_config(settings: dict) -> dict[str, Any]:
    """Resolve a script's parameter dictionary without requiring a config file."""
    return load_config(settings=settings)


def artifact_path(cfg: dict, category: str, *parts: str) -> Path:
    root = Path(cfg["output_root"]).resolve()
    if category == "checkpoints" and parts:
        path = root.joinpath("runs", parts[0], "checkpoints", *parts[1:]).resolve()
    elif category == "reports" and parts:
        path = root.joinpath("runs", parts[0], "evaluation", *parts[1:]).resolve()
    else:
        path = root.joinpath(category, *parts).resolve()
    if not path.is_relative_to(root):
        raise ValueError("Output path escapes the CVPR artifact directory.")
    return path


def write_json(path: str | Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    os.replace(temporary, path)
