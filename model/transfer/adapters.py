"""Equal-budget residual adapters and explicit, autograd-safe Hiera routing."""
from __future__ import annotations

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

BLOCK_INDICES = tuple(range(3, 16))
BLOCK_DIMS = {i: 384 if i < 14 else 768 for i in BLOCK_INDICES}
BASE_RANKS = {i: 32 if i < 14 else 16 for i in BLOCK_INDICES}
DELTA_RANKS = (5, 4, 4, 5, 4, 4, 5, 4, 4, 5, 4, 4, 4)


class ResidualAdapter(nn.Module):
    """BHWC residual, affine-free normalization, zero-initialized output."""

    def __init__(self, dim: int, rank: int):
        super().__init__()
        self.dim, self.rank = dim, rank
        self.norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.down = nn.Linear(dim, rank, bias=False)
        self.up = nn.Linear(rank, dim, bias=False)
        self.activation = nn.GELU()
        nn.init.zeros_(self.up.weight)

    def forward(self, value):
        return value + self.up(self.activation(self.down(self.norm(value))))


class RoutedHiera(nn.Module):
    """Wrap an already-loaded Hiera; routing is an argument, never a mutable hook.

    The adapter pool belongs to TransferModel, outside the frozen SAM module.
    Non-reentrant checkpoint closures capture their immutable layer route.
    """

    def __init__(self, trunk: nn.Module, gradient_checkpointing: bool = False):
        super().__init__()
        if len(trunk.blocks) != 16 or list(trunk.stage_ends) != [0, 2, 13, 15]:
            raise ValueError("Transfer adapters require official SAM2.1 Small Hiera [1,2,11,2]")
        for block, dim in BLOCK_DIMS.items():
            if trunk.blocks[block].dim_out != dim:
                raise ValueError(f"Unexpected Hiera width at block {block}")
        self.original = trunk
        self.gradient_checkpointing = bool(gradient_checkpointing)
        self.channel_list = trunk.channel_list
        self.stage_ends = trunk.stage_ends
        self.return_interm_layers = trunk.return_interm_layers
        self.dim_out = trunk.blocks[-1].dim_out

    @property
    def blocks(self):
        return self.original.blocks

    @property
    def patch_embed(self):
        return self.original.patch_embed

    def get_num_layers(self):
        return self.original.get_num_layers()

    def get_layer_id(self, name):
        return self.original.get_layer_id(name)

    def forward(self, value, route_ids=None, adapter_pool=None):
        if route_ids is None or adapter_pool is None:
            raise ValueError("RoutedHiera requires an explicit immutable route and adapter pool")
        route_ids = tuple(tuple(chain) for chain in route_ids)
        if len(route_ids) != len(BLOCK_INDICES):
            raise ValueError("Expected 13 route entries")
        value = self.original.patch_embed(value)
        value = value + self.original._get_pos_embed(value.shape[1:3])
        outputs = []
        for index, block in enumerate(self.original.blocks):
            modules = tuple(adapter_pool[str(index)][v] for v in route_ids[index - 3]) if index >= 3 else ()

            def run(x, frozen_block=block, adapters=modules):
                x = frozen_block(x)
                for adapter in adapters:
                    x = adapter(x)
                return x

            # SAM stays eval, but gradients still flow through all frozen blocks.
            if self.gradient_checkpointing and torch.is_grad_enabled() and index >= 3:
                value = checkpoint(run, value, use_reentrant=False)
            else:
                value = run(value)
            if index == self.stage_ends[-1] or (index in self.stage_ends and self.return_interm_layers):
                outputs.append(value.permute(0, 3, 1, 2))
        return outputs
