from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import Path
import random

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .controller import UtilityController
from .losses import segmentation_loss
from .memory import MemoryBank, MemorySlot, slot_descriptors


@dataclass
class TUMConfig:
    backbone: str = "small"
    image_size: int = 256
    memory_capacity: int = 4
    policy: str = "random"
    checkpoint: str = ""
    freeze_trunk: bool = True
    num_targets: int = 32
    num_anatomies: int = 16
    num_tasks: int = 6
    num_label_spaces: int = 5
    detach_memory: bool = False
    joint_ceus_memory: bool = True


class GlobalReadout(nn.Module):
    def __init__(self, label_spaces=5):
        super().__init__()
        self.view_embeddings = nn.Embedding(2, 256)
        self.view_score = nn.Sequential(nn.Linear(256, 64), nn.GELU(), nn.Linear(64, 1))
        self.history = nn.MultiheadAttention(256, 4, dropout=0.0, batch_first=True)
        self.fuse = nn.Sequential(nn.Linear(512, 256), nn.GELU(), nn.LayerNorm(256))
        self.heads = nn.ModuleList([nn.Linear(256, 2) for _ in range(label_spaces)])

    def fuse_views(self, features: Tensor, valid: Tensor):
        tokens = features.mean(dim=(-2, -1))
        view_ids = torch.arange(len(tokens), device=tokens.device)
        scores = self.view_score(tokens + self.view_embeddings(view_ids)).squeeze(-1)
        scores = scores.masked_fill(~valid.bool(), float("-inf"))
        return (scores.softmax(dim=0)[:, None] * tokens).sum(dim=0)

    def forward(self, current: Tensor, slots: list[MemorySlot], label_space: int,
                query_index: int):
        if slots:
            keys = torch.stack([s.global_token for s in slots])
            ages = current.new_tensor([query_index - s.frame_index for s in slots])
            frequencies = torch.exp(torch.arange(128, device=current.device).float() * (-9.21034 / 127))
            phase = ages[:, None] * frequencies[None]
            time_pos = torch.cat([phase.sin(), phase.cos()], dim=-1).to(keys.dtype)
            context, _ = self.history(current[None, None], (keys + time_pos)[None], keys[None],
                                      need_weights=False)
            context = context[0, 0]
        else:
            context = torch.zeros_like(current)
        return self.heads[label_space](self.fuse(torch.cat([current, context])))[None]


class TUMModel(nn.Module):
    """Native SAM2 components with automatic semantic queries and one bounded history."""

    def __init__(self, config: TUMConfig | dict | None = None):
        super().__init__()
        self.config = TUMConfig(**config) if isinstance(config, dict) else (config or TUMConfig())
        cfg = self.config
        if isinstance(cfg.image_size, bool) or not isinstance(cfg.image_size, int) or cfg.image_size < 32 or cfg.image_size % 32:
            raise ValueError("SAM2 image_size must be a positive integer multiple of 32")
        if cfg.backbone not in ("small", "tiny"):
            raise ValueError("Supported backbones: small, tiny")
        from tum_us.paths import pretrained_path
        from sam2.build_sam import build_sam2
        checkpoint = pretrained_path(cfg.checkpoint or f"pre_checkpoints/sam2.1_hiera_{cfg.backbone}.pt")
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Missing official {cfg.backbone} checkpoint: {checkpoint}")
        letter = {"small": "s", "tiny": "t"}[cfg.backbone]
        self.sam = build_sam2(f"configs/sam2.1/sam2.1_hiera_{letter}.yaml", str(checkpoint),
                              device="cpu", mode="train", apply_postprocessing=False,
                              hydra_overrides_extra=[f"++model.image_size={cfg.image_size}"])
        if cfg.freeze_trunk:
            self.sam.image_encoder.trunk.requires_grad_(False)
        self.target_queries = nn.Embedding(cfg.num_targets, 256)
        self.task_embeddings = nn.Embedding(cfg.num_tasks, 256)
        self.anatomy_embeddings = nn.Embedding(cfg.num_anatomies, 256)
        self.global_readout = GlobalReadout(cfg.num_label_spaces)
        self.controller = UtilityController()
        self.register_buffer("image_mean", torch.tensor([.485, .456, .406]).view(1, 3, 1, 1))
        self.register_buffer("image_std", torch.tensor([.229, .224, .225]).view(1, 3, 1, 1))

    @classmethod
    def from_config(cls, config):
        return cls(config)

    def train(self, mode=True):
        super().train(mode)
        if self.config.freeze_trunk:
            self.sam.image_encoder.trunk.eval()
        return self

    def _encode(self, media: Tensor):
        return self.sam.forward_image((media - self.image_mean) / self.image_std)

    def active_memory_tasks(self, task_id: int):
        if self.config.joint_ceus_memory and task_id in (3, 4):
            return (True, True)
        return (task_id in (0, 2, 3), task_id in (1, 4, 5))

    def _new_bank(self, task, anatomy, label_space, targets, capacity=None, policy=None, rng=None):
        """Extension point for context-conditioned memory without mutable hooks."""
        return MemoryBank(self.config.memory_capacity if capacity is None else capacity,
                          self.config.policy if policy is None else policy, self.controller, rng=rng)

    @staticmethod
    def segmentation_task_id(task: Tensor) -> Tensor:
        # Classification records reuse the supervised segmentation condition
        # for their modality, including when constructing spatial memory.
        return torch.where(task == 1, 0,
                           torch.where(task == 4, 3, torch.where(task == 5, 2, task)))

    def _condition(self, features: dict, view: int, targets: Tensor, slots: list[MemorySlot], frame: int):
        top = features["backbone_fpn"][-1][view:view + 1]
        q, (_, _, h, w) = len(targets), top.shape
        top = top.expand(q, -1, -1, -1)
        if not slots:
            return top + self.sam.no_mem_embed.reshape(1, 256, 1, 1)
        current = top.flatten(2).permute(2, 0, 1)
        current_pos = features["vision_pos_enc"][-1][view:view + 1].expand(q, -1, -1, -1)
        current_pos = current_pos.flatten(2).permute(2, 0, 1)
        memories, positions = [], []
        newest_first = sorted((s for s in slots if not s.anchor), key=lambda s: s.frame_index, reverse=True)
        ranks = {s.frame_index: min(i, self.sam.num_maskmem - 2) for i, s in enumerate(newest_first)}
        for slot in slots:
            spatial = slot.spatial[view] if slot.spatial.ndim == 5 else slot.spatial
            positional = slot.pos[view] if slot.pos.ndim == 5 else slot.pos
            memories.append(spatial.flatten(2).permute(2, 0, 1))
            position = positional.flatten(2).permute(2, 0, 1)
            temporal = self.sam.num_maskmem - 1 if slot.anchor else ranks[slot.frame_index]
            positions.append(position + self.sam.maskmem_tpos_enc[temporal])
        pointers = torch.stack([s.pointers[view] if s.pointers.ndim == 3 else s.pointers for s in slots])
        n = len(slots)
        pointers = pointers.reshape(n, q, 4, 64).permute(0, 2, 1, 3).flatten(0, 1)
        from sam2.modeling.sam2_utils import get_1d_sine_pe
        ages = top.new_tensor([frame - s.frame_index for s in slots]).float()
        pointer_pos = get_1d_sine_pe(ages / max(frame, 1), dim=256)
        pointer_pos = self.sam.obj_ptr_tpos_proj(pointer_pos).to(top.dtype)
        pointer_pos = pointer_pos[:, None].expand(n, q, 64).repeat_interleave(4, dim=0)
        memories.append(pointers)
        positions.append(pointer_pos)
        fused = self.sam.memory_attention(curr=[current], curr_pos=[current_pos],
                    memory=torch.cat(memories), memory_pos=torch.cat(positions), num_obj_ptr_tokens=n * 4)
        return fused.permute(1, 2, 0).reshape(q, 256, h, w)

    def _prepare_object_read(self,query,fused,slots,view,frame):
        return query,fused

    def _read(self, features, view, targets, task, anatomy, label_space, frame, slots, view_valid):
        q = len(targets)
        fused = self._condition(features, view, targets, slots, frame)
        query = (self.target_queries(targets)
                 + self.task_embeddings(self.segmentation_task_id(task))
                 + self.anatomy_embeddings(anatomy))
        query,fused=self._prepare_object_read(query,fused,slots,view,frame)
        dense_prompt = self.sam.sam_prompt_encoder.no_mask_embed.weight.reshape(1, 256, 1, 1)
        dense_prompt = dense_prompt.expand(q, 256, fused.shape[-2], fused.shape[-1])
        high = [x[view:view + 1].expand(q, -1, -1, -1) for x in features["backbone_fpn"][:-1]]
        low, quality, token, _ = self.sam.sam_mask_decoder(
            image_embeddings=fused, image_pe=self.sam.sam_prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=query[:, None], dense_prompt_embeddings=dense_prompt,
            multimask_output=False, repeat_image=False, high_res_features=high)
        masks = F.interpolate(low.float(), (self.config.image_size,) * 2, mode="bilinear", align_corners=False)[:, 0]
        pointers = self.sam.obj_ptr_proj(token[:, 0])
        global_token = self.global_readout.fuse_views(features["backbone_fpn"][-1], view_valid)
        logits = self.global_readout(global_token, slots, label_space, frame)
        return masks, pointers, quality.mean(), global_token, logits

    def _write(self, features, view, masks, pointers, quality, global_token, frame, anchor):
        top = features["backbone_fpn"][-1][view:view + 1].expand(len(masks), -1, -1, -1)
        probabilities = masks.sigmoid()[:, None]
        encoded = self.sam.memory_encoder(top,
            probabilities * self.sam.sigmoid_scale_for_mem_enc + self.sam.sigmoid_bias_for_mem_enc,
            skip_mask_sigmoid=True)
        slot = MemorySlot(frame, encoded["vision_features"], encoded["vision_pos_enc"][-1],
                          pointers, global_token, quality, probabilities.mean(), anchor)
        return slot.detach() if self.config.detach_memory else slot

    def _read_views(self, features, targets, task, anatomy, label_space, frame, slots, view_valid):
        return {view: self._read(features, view, targets, task, anatomy, label_space,
                                frame, slots, view_valid)
                for view in view_valid.nonzero().flatten().tolist()}

    def _write_views(self, features, results, frame, anchor):
        written = [self._write(features, view, *result[:4], frame, anchor)
                   for view, result in results.items()]
        first = written[0]
        if len(written) == 1:
            return first
        return replace(first, spatial=torch.stack([s.spatial for s in written]),
                       pos=torch.stack([s.pos for s in written]),
                       pointers=torch.stack([s.pointers for s in written]),
                       quality=torch.stack([s.quality for s in written]).mean(),
                       area=torch.stack([s.area for s in written]).mean())

    def forward(self, batch):
        b, t, vmax, _, h, w = batch.media.shape
        if (h, w) != (self.config.image_size, self.config.image_size):
            raise ValueError("Batch spatial size differs from model image_size")
        qmax = batch.target_ids.shape[-1]
        batch_masks, batch_view_masks, batch_classes, batch_frame_classes, indices, byte_counts = [], [], [], [], [], []
        for case in range(b):
            valid_targets = batch.target_ids[case] >= 0
            targets = batch.target_ids[case, valid_targets]
            if not len(targets):
                raise ValueError("Each sample needs at least one target query, including classification samples")
            task = batch.task_ids[case]
            anatomy = batch.anatomy_ids[case]
            label_space = int(batch.label_space_ids[case].clamp(min=0))
            bank = self._new_bank(task, anatomy, label_space, targets)
            masks_for_case, view_masks_for_case, logits_for_case = [], [], []
            video = int(task) in (2, 3, 4, 5)
            active_tasks = self.active_memory_tasks(int(task))
            last_logits = None
            for ti in range(t):
                if not bool(batch.frame_valid[case, ti]):
                    masks_for_case.append(batch.media.new_zeros(qmax, h, w))
                    view_masks_for_case.append(batch.media.new_zeros(vmax, qmax, h, w))
                    logits_for_case.append(batch.media.new_zeros(2))
                    continue
                frame = int(batch.frame_indices[case, ti])
                features = self._encode(batch.media[case, ti])
                results = self._read_views(features, targets, task, anatomy, label_space, frame,
                                            bank.slots if video else [], batch.view_valid[case])
                canonical_view = int(batch.seg_view_ids[case])
                if canonical_view not in results:
                    raise ValueError("Canonical segmentation view must be valid")
                masks, _, _, _, logits = results[canonical_view]
                per_view = masks.new_zeros(vmax, qmax, h, w)
                for view, result in results.items():
                    per_view[view] = masks.new_zeros(qmax, h, w).index_copy(
                        0, valid_targets.nonzero().flatten(), result[0])
                padded = per_view[canonical_view]
                masks_for_case.append(padded)
                view_masks_for_case.append(per_view)
                logits_for_case.append(logits[0])
                last_logits = logits[0]
                if video and bank.policy != "none":
                    bank.update(self._write_views(features, results, frame, not bank.slots), active_tasks)
            if last_logits is None:
                raise ValueError("Every case must have a valid frame")
            batch_masks.append(torch.stack(masks_for_case))
            batch_view_masks.append(torch.stack(view_masks_for_case))
            batch_classes.append(last_logits)
            batch_frame_classes.append(torch.stack(logits_for_case))
            indices.append([s.frame_index for s in bank.slots])
            byte_counts.append(bank.nbytes)
        return {"mask_logits": torch.stack(batch_masks), "class_logits": torch.stack(batch_classes),
                "view_mask_logits": torch.stack(batch_view_masks),
                "frame_class_logits": torch.stack(batch_frame_classes), "memory_indices": indices,
                "memory_bytes": byte_counts}

    @torch.no_grad()
    def utility_examples(self, batch, max_states=2):
        """Fixed-reader, fixed-state deletion labels, evaluated on later queries."""
        was_training = self.training
        self.eval()
        examples = []
        try:
            for case in range(len(batch.media)):
                valid_times = batch.frame_valid[case].nonzero().flatten().tolist()
                if len(valid_times) < 3:
                    continue
                valid_targets = batch.target_ids[case] >= 0
                targets = batch.target_ids[case, valid_targets]
                view, task = int(batch.seg_view_ids[case]), batch.task_ids[case]
                anatomy, label_space = batch.anatomy_ids[case], int(batch.label_space_ids[case].clamp(min=0))
                requested_tasks = self.active_memory_tasks(int(task))
                cached = {ti: self._encode(batch.media[case, ti]) for ti in valid_times}
                bank = MemoryBank(self.config.memory_capacity, "random")
                states = 0
                possible = [ti for ti in valid_times[1:-1]
                            if (bool(batch.class_valid[case]) and requested_tasks[1]) or any(
                                requested_tasks[0] and qi > ti and batch.mask_valid[case, qi, valid_targets].any()
                                for qi in valid_times)]
                if not possible:
                    continue
                dual = [ti for ti in possible if any(
                    requested_tasks[0] and qi > ti and batch.mask_valid[case, qi, valid_targets].any() for qi in valid_times)]
                selected_times = []
                if dual:
                    selected_times.append(dual[-1])
                if possible[-1] not in selected_times and len(selected_times) < max_states:
                    selected_times.append(possible[-1])
                rest = [ti for ti in possible if ti not in selected_times]
                selected_times += random.sample(rest, min(len(rest), max(0, max_states - len(selected_times))))
                for ti in valid_times[:-1]:
                    frame = int(batch.frame_indices[case, ti])
                    results = self._read_views(cached[ti], targets, task, anatomy, label_space,
                                               frame, bank.slots, batch.view_valid[case])
                    masks = results[min(results)][0]
                    candidate = self._write_views(cached[ti], results, frame, not bank.slots)
                    candidates = bank.slots + [candidate]
                    future = [qi for qi in valid_times if qi > ti]
                    seg_queries = [qi for qi in future if requested_tasks[0] and batch.mask_valid[case, qi, valid_targets].any()]
                    cls_valid = bool(batch.class_valid[case]) and requested_tasks[1]
                    active = torch.tensor([bool(seg_queries), cls_valid], device=masks.device)
                    if ti in selected_times and len(candidates) >= 2 and active.any() and states < max_states:
                        seg_query = seg_queries[-1] if seg_queries else None
                        cls_query = future[-1] if cls_valid else None

                        def read_cost(memory):
                            costs = masks.new_zeros(2)
                            queries = set(q for q in (seg_query, cls_query) if q is not None)
                            for qi in queries:
                                fm = int(batch.frame_indices[case, qi])
                                pred, _, _, _, cl = self._read(cached[qi], view, targets, task, anatomy,
                                    label_space, fm, memory, batch.view_valid[case])
                                if qi == seg_query:
                                    valid = batch.mask_valid[case, qi, valid_targets]
                                    costs[0] = segmentation_loss(pred[valid], batch.masks[case, qi, valid_targets][valid])
                                if qi == cls_query:
                                    costs[1] = F.cross_entropy(cl.float(), batch.class_targets[case:case + 1])
                            return costs

                        full_cost = read_cost(candidates)
                        deltas = torch.stack([read_cost(candidates[:j] + candidates[j + 1:]) - full_cost
                                              for j in range(len(candidates))])
                        examples.append({"descriptors": slot_descriptors(candidates, frame).float().cpu(),
                            "deltas": deltas.float().cpu(), "valid_tasks": active.cpu(),
                            "active_tasks": torch.tensor(requested_tasks, dtype=torch.bool),
                            "anchor_mask": torch.tensor([s.anchor for s in candidates]),
                            "frame_indices": torch.tensor([s.frame_index for s in candidates])})
                        states += 1
                    bank.update(candidate)
                    if states >= max_states:
                        break
        finally:
            self.train(was_training)
        return examples

    def config_dict(self):
        return asdict(self.config)
