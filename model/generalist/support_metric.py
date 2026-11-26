"""Construct a fresh similarity rule from the annotations in each support set."""

import copy

import torch
from torch import nn
from torch.nn import functional as F


VARIANTS = ('fixed', 'independent', 'shared')


def support_key(task, anatomy, label_space, targets):
    return (int(task), int(anatomy), int(label_space), *sorted(targets))


class SupportMetric(nn.Module):
    def __init__(self, channels=256, hidden=64, variant='shared'):
        super().__init__()
        if variant not in VARIANTS:
            raise ValueError(f'Unknown support metric variant: {variant}')
        self.variant = variant
        rule = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.GELU(),
            nn.Linear(hidden, channels),
        )
        nn.init.zeros_(rule[-1].weight)
        nn.init.zeros_(rule[-1].bias)
        self.rule = rule
        if variant == 'independent':
            self.seg_rule = copy.deepcopy(rule)

    def context(self, prototypes, task):
        """Return weighted class prototypes and per-channel feature scales."""
        unit_prototypes = F.normalize(prototypes.float(), dim=-1)
        channel_scales = self.channel_scales(unit_prototypes, task)
        weighted_prototypes = F.normalize(unit_prototypes * channel_scales, dim=-1)
        return weighted_prototypes, channel_scales

    def channel_scales(self, unit_prototypes, task):
        channel_variance = unit_prototypes.var(dim=0, unbiased=False)
        mean_variance = channel_variance.mean().clamp_min(1e-6)
        support_contrast = channel_variance / mean_variance

        if self.variant == 'fixed':
            support_contrast = torch.zeros_like(support_contrast)
        rule = self.rule
        if self.variant == 'independent' and task == 'image_seg':
            rule = self.seg_rule

        log_scales = 0.5 * rule(support_contrast).tanh()
        return log_scales.exp()

    @staticmethod
    def score(features, context):
        weighted_prototypes, channel_scales = context
        weighted_features = F.normalize(features.float() * channel_scales, dim=-1)
        cosine_similarity = weighted_features @ weighted_prototypes.T
        return 16.0 * cosine_similarity


def mask_prototypes(features, masks):
    """Pool each image into background and foreground vectors, in that order."""
    foreground = F.interpolate(masks[:, None].float(), size=features.shape[-2:], mode='area')
    background = 1 - foreground
    region_weights = torch.cat((background, foreground), dim=1).flatten(2)
    pixel_features = features.flatten(2).transpose(1, 2)
    region_area = region_weights.sum(dim=-1)
    region_sums = region_weights @ pixel_features
    region_means = region_sums / region_area.clamp_min(1e-6)[..., None]
    return region_means, region_area > 0
