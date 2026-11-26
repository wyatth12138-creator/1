from __future__ import annotations

import torch
from torch import Tensor, nn


class UtilityController(nn.Module):
    """Permutation-equivariant prediction of normalized deletion costs (seg, cls)."""

    def __init__(self, descriptor_dim=328, hidden_dim=128, num_heads=4,
                 num_layers=2, dropout=0.0):
        super().__init__()
        self.input_norm = nn.LayerNorm(descriptor_dim)
        self.project = nn.Linear(descriptor_dim, hidden_dim)
        layer = nn.TransformerEncoderLayer(hidden_dim, num_heads, hidden_dim * 2,
                                          dropout, batch_first=True, norm_first=True)
        self.context = nn.TransformerEncoder(layer, num_layers, enable_nested_tensor=False)
        self.output = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, 2))
        self.register_buffer("scales", torch.ones(2))

    def set_scales(self, values: Tensor) -> None:
        values = torch.as_tensor(values, device=self.scales.device, dtype=self.scales.dtype)
        if values.shape != (2,) or not torch.isfinite(values).all() or not (values > 0).all():
            raise ValueError("Utility scales must be two positive finite numbers")
        self.scales.copy_(values)

    def forward(self, descriptors: Tensor, slot_mask: Tensor | None = None) -> Tensor:
        squeeze = descriptors.ndim == 2
        if squeeze:
            descriptors = descriptors.unsqueeze(0)
            if slot_mask is not None:
                slot_mask = slot_mask.unsqueeze(0)
        if descriptors.ndim != 3 or descriptors.shape[-1] != self.project.in_features:
            raise ValueError("Expected descriptors [B,N,328]")
        if slot_mask is not None and not slot_mask.any(dim=1).all():
            raise ValueError("Each memory set must have at least one valid slot")
        x = self.project(self.input_norm(descriptors))
        x = self.context(x, src_key_padding_mask=None if slot_mask is None else ~slot_mask.bool())
        result = self.output(x)
        return result.squeeze(0) if squeeze else result
