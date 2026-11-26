"""Per-view input-conditioned scaling for the top TUM-US FPN feature."""

import torch
from torch import nn


class InputConditioning(nn.Module):
    """Scale each view using its own spatial channel statistics.

    The static control has exactly the same trainable architecture, but feeds
    zeros to the MLP after standardization. It can therefore learn fixed scales.
    """

    def __init__(self, variant="conditioned", channels=256, hidden=64, eps=1e-6):
        super().__init__()
        if variant not in ("conditioned", "static"):
            raise ValueError(f"Unknown input conditioning variant: {variant}")
        if eps <= 0:
            raise ValueError("eps must be positive")
        self.variant = variant
        self.channels = channels
        self.eps = eps
        self.rule = nn.Sequential(
            nn.Linear(2 * channels, hidden), nn.GELU(), nn.Linear(hidden, channels)
        )
        nn.init.zeros_(self.rule[-1].weight)
        nn.init.zeros_(self.rule[-1].bias)

    def forward(self, top):
        if top.ndim != 4 or top.shape[1] != self.channels:
            raise ValueError(f"Expected [views, {self.channels}, height, width]")
        # Compute moments and normalization in float32 for mixed precision.
        x = top.float()
        mean = x.mean(dim=(-2, -1))
        std = (x.var(dim=(-2, -1), unbiased=False) + self.eps).sqrt()
        stats = torch.cat((mean, std), dim=1)
        center = stats.mean(dim=1, keepdim=True)
        variance = (stats - center).square().mean(dim=1, keepdim=True)
        stats = (stats - center) * torch.rsqrt(variance + self.eps)
        if self.variant == "static":
            stats = torch.zeros_like(stats)
        log_scale = 0.5 * self.rule(stats.to(self.rule[0].weight.dtype)).tanh()
        return top * log_scale.exp().to(top.dtype)[:, :, None, None]

    def transform_features(self, features):
        """Apply once immediately after encoding, before both reading and writing."""
        pyramid = features["backbone_fpn"]
        weighted = self(pyramid[-1])
        return {**features, "backbone_fpn": [*pyramid[:-1], weighted]}

