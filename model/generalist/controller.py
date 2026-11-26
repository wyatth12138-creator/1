"""Permutation-equivariant deletion-cost prediction with semantic/budget context."""
from __future__ import annotations

import torch
from torch import nn


class ContextualController(nn.Module):
    """Predict two signed, normalized costs; unobserved axes are trained nowhere.

    Context columns are task, anatomy, label space, target, capacity. A target
    of -1 explicitly denotes multiple/unspecified targets; there is no source ID.
    task_agnostic disables semantic conditioning while retaining the budget.
    Unseen semantic IDs use a trained unknown token rather than random embeddings.
    """

    def __init__(self, descriptor_dim=328, hidden_dim=128, num_heads=4, num_layers=2,
                 dropout=0.0, num_tasks=6, num_anatomies=16, num_label_spaces=5,
                 num_targets=32, task_agnostic=False, semantic_dropout=0.2,
                 unseen_semantics_to_unknown=True):
        super().__init__()
        if hidden_dim % num_heads or num_layers < 1:
            raise ValueError("Controller requires divisible heads and at least one layer")
        self.task_agnostic = bool(task_agnostic)
        self.semantic_dropout = float(semantic_dropout)
        if not 0 <= self.semantic_dropout <= 1:
            raise ValueError("semantic_dropout must lie in [0,1]")
        self.unseen_semantics_to_unknown = bool(unseen_semantics_to_unknown)
        self.semantic_sizes = (num_tasks, num_anatomies, num_label_spaces, num_targets)
        self.seen_buffer_names = ("seen_task_ids", "seen_anatomy_ids", "seen_label_space_ids", "seen_target_ids")
        for name, size in zip(self.seen_buffer_names, self.semantic_sizes):
            self.register_buffer(name, torch.zeros(size, dtype=torch.bool))
        self.semantic_embeddings = nn.ModuleList([nn.Embedding(size + 1, hidden_dim)
                                                   for size in self.semantic_sizes])
        self.budget = nn.Sequential(nn.Linear(1, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, hidden_dim))
        self.input_norm = nn.LayerNorm(descriptor_dim)
        self.project = nn.Linear(descriptor_dim, hidden_dim)
        layer = nn.TransformerEncoderLayer(hidden_dim, num_heads, hidden_dim * 2,
                                          dropout, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers, enable_nested_tensor=False)
        self.output = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, 2))
        self.register_buffer("scales", torch.ones(2))

    @torch.no_grad()
    def set_scales(self, values):
        values = torch.as_tensor(values, dtype=self.scales.dtype, device=self.scales.device)
        if values.shape != (2,) or not torch.isfinite(values).all() or not (values > 0).all():
            raise ValueError("Utility scales must be two positive finite training-source values")
        self.scales.copy_(values)

    def forward(self, descriptors, slot_mask=None, context=None):
        squeeze = descriptors.ndim == 2
        if squeeze:
            descriptors = descriptors[None]
            if slot_mask is not None and slot_mask.ndim == 1:
                slot_mask = slot_mask[None]
        if descriptors.ndim != 3 or descriptors.shape[-1] != self.project.in_features:
            raise ValueError("Expected descriptors [B,N,328]")
        b, n, _ = descriptors.shape
        if n < 1 or not torch.isfinite(descriptors).all():
            raise ValueError("Controller descriptors must be nonempty and finite")
        if slot_mask is None:
            slot_mask = torch.ones(b, n, dtype=torch.bool, device=descriptors.device)
        else:
            slot_mask = slot_mask.to(device=descriptors.device, dtype=torch.bool)
        if slot_mask.shape != (b, n) or not slot_mask.any(dim=1).all():
            raise ValueError("Each memory set requires at least one valid slot")
        if context is None:
            raise ValueError("ContextualController requires explicit semantic/budget context")
        context = torch.as_tensor(context, dtype=torch.float32, device=descriptors.device)
        if context.ndim == 1:
            context = context[None]
        if context.shape != (b, 5) or not torch.isfinite(context).all():
            raise ValueError("Context must have shape [B,5] and finite values")
        if (context[:, 4] < 0).any():
            raise ValueError("Memory capacity cannot be negative")
        semantic = descriptors.new_zeros(b, self.project.out_features)
        for column, (size, embedding, seen_name) in enumerate(zip(
                self.semantic_sizes, self.semantic_embeddings, self.seen_buffer_names)):
            values = context[:, column]
            if not torch.equal(values, values.floor()) or ((values < -1) | (values >= size)).any():
                raise ValueError(f"Semantic context column {column} has invalid IDs")
            ids = values.long()
            known = ids >= 0
            seen = getattr(self, seen_name)
            if self.training:
                # Record supplied training semantics before dropping any ID. Evaluation
                # never expands this vocabulary, including across held-out sources.
                with torch.no_grad():
                    seen[ids[known].unique()] = True
            if self.task_agnostic:
                continue
            if self.training and self.semantic_dropout > 0:
                drop = torch.rand(ids.shape, device=ids.device) < self.semantic_dropout
                ids = ids.masked_fill(drop, -1)
            elif not self.training and self.unseen_semantics_to_unknown:
                observed = seen[ids.clamp(min=0)] & known
                ids = ids.masked_fill(~observed, -1)
            semantic = semantic + embedding(ids + 1)
        budget = self.budget(context[:, 4:5].log1p() / 4)
        x = self.project(self.input_norm(descriptors)) + (semantic + budget)[:, None]
        x = self.encoder(x, src_key_padding_mask=~slot_mask)
        output = self.output(x)
        if self.task_agnostic:
            # One shared score: selecting a task axis at inference cannot restore
            # task dependence after semantic conditioning has been disabled.
            output = output.mean(dim=-1, keepdim=True).expand_as(output)
        output = output.masked_fill(~slot_mask[..., None], 0)
        return output[0] if squeeze else output
