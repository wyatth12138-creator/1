"""Randomness, device selection, logging and device-independent batch helpers."""
from __future__ import annotations

import dataclasses
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device(requested: str = "auto") -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return device


def move_batch(batch: Any, device: torch.device) -> Any:
    if hasattr(batch, "to"):
        return batch.to(device)
    if dataclasses.is_dataclass(batch):
        return dataclasses.replace(batch, **{f.name: getattr(batch, f.name).to(device)
            for f in dataclasses.fields(batch) if isinstance(getattr(batch, f.name), torch.Tensor)})
    raise TypeError("A Batch dataclass or an object with .to(device) is required.")


def autocast_context(device: torch.device, enabled: bool):
    enabled = bool(enabled and device.type == "cuda")
    dtype = torch.bfloat16 if enabled and torch.cuda.is_bf16_supported() else torch.float16
    return torch.autocast(device_type=device.type, enabled=enabled, dtype=dtype)


def make_grad_scaler(device: torch.device, enabled: bool):
    return torch.amp.GradScaler("cuda", enabled=bool(enabled and device.type == "cuda"
                                                   and not torch.cuda.is_bf16_supported()))


def finite_losses(losses: dict) -> dict[str, float]:
    result = {}
    for name, value in losses.items():
        if isinstance(value, torch.Tensor) and value.numel() == 1:
            if not torch.isfinite(value).all():
                raise FloatingPointError(f"Non-finite loss: {name}={value.detach().cpu().item()}")
            result[name] = float(value.detach().cpu())
    return result


def gradient_summary(model: torch.nn.Module) -> dict:
    groups: dict[str, dict] = {}
    total_sq = 0.0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        group_name = name.split(".")[0]
        group = groups.setdefault(group_name, {"parameters": 0, "with_grad": 0,
                                                "nonzero_grad": 0, "norm_sq": 0.0})
        group["parameters"] += parameter.numel()
        if parameter.grad is not None:
            gradient = parameter.grad.detach().float()
            if not torch.isfinite(gradient).all():
                raise FloatingPointError(f"Non-finite gradient in {name}")
            group["with_grad"] += parameter.numel()
            if torch.count_nonzero(gradient).item():
                group["nonzero_grad"] += parameter.numel()
            norm_sq = float(torch.sum(gradient * gradient).cpu())
            group["norm_sq"] += norm_sq
            total_sq += norm_sq
    for group in groups.values():
        group["norm"] = group.pop("norm_sq") ** 0.5
    return {"norm": total_sq ** 0.5, "groups": groups}


def append_jsonl(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")


def registry_snapshot() -> dict:
    from tum_us.data import registry
    return {key: list(getattr(registry, key)) for key in
            ("TASKS", "LABEL_SPACES", "ANATOMIES", "TARGETS") if hasattr(registry, key)}


def environment_summary(device: torch.device) -> dict:
    return {"torch": torch.__version__, "cuda_runtime": torch.version.cuda,
            "device": str(device), "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None}
