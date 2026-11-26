from __future__ import annotations

import torch
import torch.nn.functional as F


def segmentation_loss(logits, targets):
    bce = F.binary_cross_entropy_with_logits(logits.float(), targets.float())
    probs = logits.float().sigmoid().flatten(-2)
    targets = targets.float().flatten(-2)
    dice = (2 * (probs * targets).sum(-1) + 1) / (probs.sum(-1) + targets.sum(-1) + 1)
    return bce + 1 - dice.mean()


def supervised_mask_logits(output, batch):
    if "view_mask_logits" not in output:
        return output["mask_logits"]
    views = output["view_mask_logits"]
    indices = torch.arange(len(views), device=views.device)
    return views[indices, :, batch.seg_view_ids]


def compute_losses(output, batch):
    logits, cls = supervised_mask_logits(output, batch), output["class_logits"]
    zero = logits.sum() * 0 + cls.sum() * 0
    mask_valid = batch.mask_valid & batch.frame_valid[:, :, None] & (batch.target_ids[:, None] >= 0)
    seg = segmentation_loss(logits[mask_valid], batch.masks[mask_valid]) if mask_valid.any() else zero
    valid = batch.class_valid.bool() & (batch.class_targets >= 0)
    classification = F.cross_entropy(cls[valid].float(), batch.class_targets[valid]) if valid.any() else zero
    return {"loss": seg + classification, "seg_loss": seg, "cls_loss": classification}
