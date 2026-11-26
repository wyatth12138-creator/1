"""Tensor-only model input contract, independent of the legacy GeneralTMI package."""
from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any

import torch
from torch import Tensor


@dataclass
class Sample:
    media: Tensor                 # [T,V,3,H,W], float32 in [0,1]
    frame_valid: Tensor           # [T], bool
    view_valid: Tensor            # [V], bool
    masks: Tensor                 # [T,Q,Hm,Wm], binary float, Q>=1
    mask_valid: Tensor            # [T,Q], bool
    target_ids: Tensor            # [Q], long; -1 for padding
    class_targets: Tensor         # [], long; -100 when unavailable
    class_valid: Tensor           # [], bool
    label_space_ids: Tensor       # [], long
    task_ids: Tensor              # [], long
    anatomy_ids: Tensor           # [], long
    seg_view_ids: Tensor          # [], long; canonical view index
    frame_indices: Tensor         # [T], long; source indices
    metadata: dict[str, Any]

    def validate(self) -> None:
        if self.media.ndim != 5 or self.media.shape[2] != 3:
            raise ValueError("Sample.media must have shape [T,V,3,H,W]")
        t, v, _, h, w = self.media.shape
        if t < 1 or v < 1 or self.masks.ndim != 4:
            raise ValueError("Samples require at least one frame, view and mask slot")
        q = self.target_ids.numel()
        if q < 1 or self.masks.shape[:2] != (t, q) or min(self.masks.shape[-2:]) < 1 or self.mask_valid.shape != (t, q):
            raise ValueError("Mask and target slot shapes disagree")
        if self.frame_valid.shape != (t,) or self.frame_indices.shape != (t,):
            raise ValueError("Frame validity/index shapes disagree")
        if self.view_valid.shape != (v,):
            raise ValueError("View validity shape disagrees")
        for name in ("class_targets", "class_valid", "label_space_ids", "task_ids", "anatomy_ids", "seg_view_ids"):
            if getattr(self, name).ndim != 0:
                raise ValueError(f"Sample.{name} must be a scalar tensor")


@dataclass
class Batch:
    media: Tensor                 # [B,T,V,3,H,W]
    frame_valid: Tensor           # [B,T]
    view_valid: Tensor            # [B,V]
    masks: Tensor                 # [B,T,Q,Hm,Wm]
    mask_valid: Tensor            # [B,T,Q]
    target_ids: Tensor            # [B,Q]
    class_targets: Tensor         # [B]
    class_valid: Tensor           # [B]
    label_space_ids: Tensor       # [B]
    task_ids: Tensor              # [B]
    anatomy_ids: Tensor           # [B]
    seg_view_ids: Tensor          # [B]
    frame_indices: Tensor         # [B,T]
    metadata: list[dict[str, Any]]

    def to(self, device: str | torch.device, non_blocking: bool = False) -> "Batch":
        return Batch(**{
            item.name: (value.to(device, non_blocking=non_blocking) if isinstance(value, Tensor) else value)
            for item in fields(self) for value in (getattr(self, item.name),)
        })

    def __len__(self) -> int:
        return self.media.shape[0]


def collate_samples(samples: list[Sample]) -> Batch:
    if not samples:
        raise ValueError("Cannot collate an empty list")
    for sample in samples:
        sample.validate()
    b = len(samples)
    t = max(item.media.shape[0] for item in samples)
    v = max(item.media.shape[1] for item in samples)
    q = max(item.target_ids.numel() for item in samples)
    h, w = samples[0].media.shape[-2:]
    if any(item.media.shape[-2:] != (h, w) for item in samples):
        raise ValueError("Samples must share image_size before collation")
    mask_h, mask_w = samples[0].masks.shape[-2:]
    if any(item.masks.shape[-2:] != (mask_h, mask_w) for item in samples):
        raise ValueError("Samples must share mask_size before collation")
    result = {
        "media": torch.zeros((b, t, v, 3, h, w), dtype=torch.float32),
        "frame_valid": torch.zeros((b, t), dtype=torch.bool),
        "view_valid": torch.zeros((b, v), dtype=torch.bool),
        "masks": torch.zeros((b, t, q, mask_h, mask_w), dtype=torch.float32),
        "mask_valid": torch.zeros((b, t, q), dtype=torch.bool),
        "target_ids": torch.full((b, q), -1, dtype=torch.long),
        "frame_indices": torch.full((b, t), -1, dtype=torch.long),
    }
    for index, sample in enumerate(samples):
        st, sv = sample.media.shape[:2]
        sq = sample.target_ids.numel()
        result["media"][index, :st, :sv] = sample.media
        result["frame_valid"][index, :st] = sample.frame_valid
        result["view_valid"][index, :sv] = sample.view_valid
        result["masks"][index, :st, :sq] = sample.masks
        result["mask_valid"][index, :st, :sq] = sample.mask_valid
        result["target_ids"][index, :sq] = sample.target_ids
        result["frame_indices"][index, :st] = sample.frame_indices
    for name in ("class_targets", "class_valid", "label_space_ids", "task_ids", "anatomy_ids", "seg_view_ids"):
        result[name] = torch.stack([getattr(sample, name) for sample in samples])
    return Batch(**result, metadata=[sample.metadata for sample in samples])
