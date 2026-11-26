from __future__ import annotations

from dataclasses import dataclass, replace
import math
import random

import torch
from torch import Tensor
import torch.nn.functional as F


@dataclass
class MemorySlot:
    frame_index: int
    spatial: Tensor
    pos: Tensor
    pointers: Tensor
    global_token: Tensor
    quality: Tensor
    area: Tensor
    anchor: bool = False
    object_tokens: Tensor | None = None

    def detach(self):
        return replace(self, **{name: getattr(self, name).detach() for name in
                               ("spatial", "pos", "pointers", "global_token", "quality", "area")},
                       object_tokens=None if self.object_tokens is None else self.object_tokens.detach())

    @property
    def nbytes(self):
        total=sum(getattr(self, name).numel() * getattr(self, name).element_size()
                  for name in ("spatial", "pos", "pointers", "global_token", "quality", "area"))
        return total+(0 if self.object_tokens is None else self.object_tokens.numel()*self.object_tokens.element_size())


def slot_descriptors(slots: list[MemorySlot], query_index: int | None = None) -> Tensor:
    """Only retained/candidate evidence enters the controller; no labels or future features."""
    if not slots:
        return torch.empty(0, 328)
    now = max(s.frame_index for s in slots) if query_index is None else query_index
    tokens = torch.stack([s.global_token.float() for s in slots])
    mean_token = tokens.mean(dim=0, keepdim=True)
    descriptions = []
    for slot, token in zip(slots, tokens):
        spatial_map = slot.spatial.float()
        spatial = spatial_map.reshape(-1, 64, *spatial_map.shape[-2:]).mean(dim=(0, 2, 3))
        stats = torch.stack([
            slot.area.float().reshape(()), slot.quality.float().reshape(()),
            token.new_tensor((now - slot.frame_index) / max(now, 1)),
            token.new_tensor(float(slot.anchor)),
            F.cosine_similarity(token[None], mean_token, dim=-1).squeeze(0),
            token.new_tensor(math.log1p(max(slot.frame_index, 0)) / 10),
            token.norm() / 16, spatial.norm() / 8,
        ])
        descriptions.append(torch.cat([token, spatial, stats]))
    return torch.stack(descriptions)


class MemoryBank:
    POLICIES = ("fifo", "random", "quality", "novelty", "seg_utility",
                "cls_utility", "weighted_sum", "tum", "none", "full")

    def __init__(self, capacity: int, policy="fifo", controller=None, context=None, rng=None):
        if capacity < 2 and policy not in ("none", "full"):
            raise ValueError("Capacity includes one anchor and must be >=2")
        if policy not in self.POLICIES:
            raise ValueError(f"Unknown memory policy: {policy}")
        self.capacity, self.policy, self.controller = capacity, policy, controller
        self.context, self.rng = context, rng or random
        self._last_frame = None
        self.slots: list[MemorySlot] = []

    def clear(self):
        self.slots.clear()
        self._last_frame = None

    @property
    def nbytes(self):
        return sum(s.nbytes for s in self.slots)

    def update(self, slot: MemorySlot, active_tasks=(True, True)):
        previous = self._last_frame if self._last_frame is not None else (self.slots[-1].frame_index if self.slots else None)
        if previous is not None and slot.frame_index <= previous:
            raise ValueError("Memory writes must be strictly increasing within a case")
        self._last_frame = slot.frame_index
        if self.policy == "none":
            self.slots = []
            return None
        candidates = self.slots + [slot]
        if self.policy == "full" or len(candidates) <= self.capacity:
            self.slots = candidates
            return None
        eligible = [i for i, s in enumerate(candidates) if not s.anchor]
        if not eligible:
            raise ValueError("A full memory bank must contain an evictable slot")
        if self.policy == "random":
            drop = self.rng.choice(eligible)
        elif self.policy == "fifo":
            drop = eligible[0]
        elif self.policy == "quality":
            drop = min(eligible, key=lambda i: float(candidates[i].quality.detach()))
        elif self.policy == "novelty":
            tokens = F.normalize(torch.stack([s.global_token.float() for s in candidates]), dim=-1)
            similarities = tokens @ tokens.T
            similarities.fill_diagonal_(-2)
            redundancy = similarities.max(dim=-1).values
            drop = max(eligible, key=lambda i: float(redundancy[i].detach()))
        else:
            if self.controller is None:
                raise ValueError("Learned memory policies require a UtilityController")
            with torch.no_grad():
                descriptors = slot_descriptors(candidates)
                values = (self.controller(descriptors, context=self.context)
                          if self.context is not None else self.controller(descriptors))
                active = torch.as_tensor(active_tasks, dtype=torch.bool, device=values.device)
                if self.policy == "seg_utility":
                    scores = values[:, 0]
                elif self.policy == "cls_utility":
                    scores = values[:, 1]
                elif not active.any():
                    raise ValueError("Task-utility selection requires at least one requested task")
                elif self.policy == "weighted_sum":
                    scores = values[:, active].mean(dim=-1)
                else:
                    scores = values[:, active].max(dim=-1).values
                drop = min(eligible, key=lambda i: float(scores[i]))
        evicted = candidates[drop].frame_index
        self.slots = [s for i, s in enumerate(candidates) if i != drop]
        return evicted
