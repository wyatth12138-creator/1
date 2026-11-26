from collections import defaultdict
from dataclasses import replace
import math
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from tum_us.engine.metrics import _hd95, binary_auc


CLASSIFICATION_THRESHOLD = 0.5
HD95_DEFINITION = (
    "95th percentile of the pooled bidirectional distances between mask boundary "
    "pixels, in scoring-grid pixels. Both-empty masks score 0; exactly-one-empty "
    "pairs score the scoring-grid diagonal between corner pixel centers."
)


def _window_logits(model, batch, endpoint, output_key, recompute):
    if not recompute:
        return model(batch, endpoint=endpoint)[output_key]

    def predict(media):
        window = replace(batch, media=media)
        return model(window, endpoint=endpoint)[output_key]

    return checkpoint(predict, batch.media, use_reentrant=False)


def _classification(model, entity, endpoint, device, recompute=False):
    groups=defaultdict(list)
    for batch in entity.batches:
        logits = _window_logits(model, batch.to(device), endpoint, 'class_logits', recompute)
        meta=batch.metadata[0]
        groups[(meta.get('record_id'),meta.get('sample_index'))].append(logits.reshape(-1,2).mean(0))
    records=defaultdict(list)
    for (record, sample), values in groups.items():
        records[record].append(torch.stack(values).mean(0))
    return torch.stack([torch.stack(values).mean(0) for values in records.values()]).mean(0)

def _segmentation(model,entity,endpoint,device,return_targets=False,recompute=False):
    sequences=defaultdict(lambda: ([], [], []))
    for batch in entity.batches:
        b = batch.to(device)
        out = _window_logits(model, b, endpoint, 'mask_logits', recompute)
        valid=b.mask_valid & b.frame_valid.unsqueeze(-1)
        if valid.any():
            meta=batch.metadata[0]
            group=sequences[(meta.get('record_id'),meta.get('sample_index'))]
            logits, truth = out[valid], b.masks[valid]
            if logits.shape[-2:] != truth.shape[-2:]:
                logits = F.interpolate(logits[:, None].float(), size=truth.shape[-2:],
                                       mode='bilinear', align_corners=False)[:, 0]
            group[0].append(logits); group[1].append(truth)
            if return_targets: group[2].append(b.target_ids[:,None,:].expand_as(valid)[valid])
    if not sequences: raise ValueError('Entity has no valid segmentation masks')
    if return_targets: return [(torch.cat(x),torch.cat(y),torch.cat(ids)) for x,y,ids in sequences.values()]
    return [(torch.cat(x),torch.cat(y)) for x,y,_ in sequences.values()]

def supervised_entity_loss(model,entity,endpoint,device,recompute=False):
    if entity.label is not None:
        logits=_classification(model,entity,endpoint,device,recompute=recompute)
        return F.cross_entropy(logits.unsqueeze(0),torch.tensor([entity.label],device=device))
    losses=[]
    for logits,truth in _segmentation(model,entity,endpoint,device,recompute=recompute):
        p=logits.sigmoid()
        dice=((2*(p*truth).sum(dim=(-2,-1))+1e-6)/(p.sum(dim=(-2,-1))+truth.sum(dim=(-2,-1))+1e-6)).mean()
        losses.append(.5*F.binary_cross_entropy_with_logits(logits,truth)+.5*(1-dice))
    return torch.stack(losses).mean()

def classification_metric_summary(records):
    labels = [record["label"] for record in records]
    probabilities = [record["probability"] for record in records]
    auc = binary_auc(labels, probabilities)
    return {
        "auc": auc,
        "auroc": auc,
        "accuracy": float(np.mean([
            int(probability >= CLASSIFICATION_THRESHOLD) == label
            for label, probability in zip(labels, probabilities)
        ])) if records else None,
        "accuracy_threshold": CLASSIFICATION_THRESHOLD,
        "brier": float(np.mean([
            (probability - label) ** 2 for label, probability in zip(labels, probabilities)
        ])) if records else None,
        "n_positive": labels.count(1),
        "n_negative": labels.count(0),
    }


def _mask_hd95(prediction, target, diagonal):
    pred_present, target_present = prediction.any(), target.any()
    if not pred_present or not target_present:
        return (diagonal, "one_empty_penalized") if pred_present or target_present else (0.0, "both_empty")
    if np.array_equal(prediction, target):
        return 0.0, "defined"

    # Cropping to the union bounding box preserves both boundaries and their distances.
    foreground = prediction | target
    rows, cols = np.flatnonzero(foreground.any(axis=1)), np.flatnonzero(foreground.any(axis=0))
    crop = np.s_[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1]
    return _hd95(prediction[crop], target[crop]), "defined"


def _hd95_count_template():
    return {"scored": 0, "defined": 0, "both_empty": 0, "one_empty_penalized": 0}


def segmentation_entity_scores(sequences):
    """Score (aligned logits, binary masks, target IDs) per sequence, then per entity."""
    sequence_dice, sequence_hd95 = [], []
    target_dice, target_hd95 = defaultdict(list), defaultdict(list)
    counts, target_counts = _hd95_count_template(), defaultdict(_hd95_count_template)
    grid = None
    for logits, truth, target_ids in sequences:
        if logits.ndim != 3 or logits.shape != truth.shape or target_ids.shape != logits.shape[:1]:
            raise ValueError("Expected aligned logits and masks of shape (N, H, W) and N target IDs")
        if not logits.shape[0]:
            raise ValueError("A scored segmentation sequence must have valid masks")
        if grid is None:
            grid = tuple(logits.shape[-2:])
        elif logits.shape[-2:] != grid:
            raise ValueError("Masks within an entity must use one scoring grid")

        prediction = logits.sigmoid() >= 0.5
        binary_truth = truth >= 0.5
        pred_float = prediction.float()
        # Retain the endpoint's existing smoothed Dice and sequence weighting.
        dice = ((2 * (pred_float * truth).sum(dim=(-2, -1)) + 1e-6) /
                (pred_float.sum(dim=(-2, -1)) + truth.sum(dim=(-2, -1)) + 1e-6))
        predictions = prediction.detach().cpu().numpy()
        truths = binary_truth.detach().cpu().numpy()
        ids = target_ids.detach().cpu().numpy()
        diagonal = math.hypot(grid[0] - 1, grid[1] - 1)
        hd95 = np.empty(len(ids), dtype=float)
        for i, (mask, reference, target_id) in enumerate(zip(predictions, truths, ids)):
            hd95[i], status = _mask_hd95(mask, reference, diagonal)
            for group in (counts, target_counts[str(int(target_id))]):
                group["scored"] += 1
                group[status] += 1

        sequence_dice.append(dice.mean())
        sequence_hd95.append(float(hd95.mean()))
        for target_id in np.unique(ids):
            target = str(int(target_id))
            target_dice[target].append(dice[target_ids == int(target_id)].mean())
            target_hd95[target].append(float(hd95[ids == target_id].mean()))
    if not sequence_dice:
        raise ValueError("Entity has no valid segmentation masks")

    return {
        "dice": float(torch.stack(sequence_dice).mean()),
        "dice_by_target": {target: float(torch.stack(values).mean()) for target, values in target_dice.items()},
        "hd95_pixels": float(np.mean(sequence_hd95)),
        "hd95_pixels_by_target": {target: float(np.mean(values)) for target, values in target_hd95.items()},
        "hd95_counts": counts,
        "hd95_counts_by_target": dict(target_counts),
        "score_mask_size": list(grid),
    }


def _average_record_targets(records, field):
    grouped = defaultdict(list)
    for record in records:
        for target, score in record.get(field, {}).items():
            grouped[target].append(score)
    return {target: float(np.mean(grouped[target])) for target in sorted(grouped)}


def segmentation_metric_summary(records):
    summary = {"dice": float(np.mean([r["dice"] for r in records])) if records else None}
    dice_by_target = _average_record_targets(records, "dice_by_target")
    if dice_by_target:
        summary["dice_by_target"] = dice_by_target
    scored = [r for r in records if r.get("hd95_pixels") is not None]
    if not scored:
        return summary
    if len(scored) != len(records):
        raise ValueError("HD95 requires complete entity coverage")
    grids = {tuple(r["score_mask_size"]) for r in scored}
    if len(grids) != 1:
        raise ValueError("HD95 requires one scoring grid per endpoint")

    counts, target_counts = _hd95_count_template(), defaultdict(_hd95_count_template)
    for record in scored:
        for status, count in record["hd95_counts"].items():
            counts[status] += count
        for target, tally in record["hd95_counts_by_target"].items():
            for status, count in tally.items():
                target_counts[target][status] += count
    summary.update({
        "hd95_pixels": float(np.mean([r["hd95_pixels"] for r in scored])),
        "hd95_pixels_by_target": _average_record_targets(scored, "hd95_pixels_by_target"),
        "hd95_counts": counts,
        "hd95_counts_by_target": dict(target_counts),
        "hd95_definition": HD95_DEFINITION,
        "score_mask_size": list(next(iter(grids))),
    })
    return summary


def evaluate_entities(model, dataset, endpoint, device):
    records = []
    is_classification = dataset.spec["kind"] == "classification"
    model.eval()
    with torch.no_grad():
        for i in range(len(dataset)):
            entity = dataset[i]
            if is_classification:
                if entity.label is None:
                    raise ValueError("Classification evaluation requires labels")
                probability = float(_classification(model, entity, endpoint, device).softmax(-1)[1])
                predicted_label = int(probability >= CLASSIFICATION_THRESHOLD)
                loss = (probability - entity.label) ** 2
                records.append({
                    "key": entity.key, "label": entity.label, "probability": probability,
                    "predicted_label": predicted_label, "correct": predicted_label == entity.label,
                    "loss": loss, "difficulty": loss,
                })
            else:
                scores = segmentation_entity_scores(
                    _segmentation(model, entity, endpoint, device, return_targets=True)
                )
                loss = 1 - scores["dice"]
                records.append({
                    "key": entity.key, "label": None, "probability": None, **scores,
                    "loss": loss, "difficulty": loss,
                })
    summary = classification_metric_summary if is_classification else segmentation_metric_summary
    return {"metrics": summary(records), "records": records}

