"""Task/dataset metrics that respect mask_valid and class_valid supervision."""
from __future__ import annotations

from collections import defaultdict
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F


def binary_auc(labels: Iterable[int], probabilities: Iterable[float]) -> float | None:
    labels = np.asarray(list(labels), dtype=np.int64)
    scores = np.asarray(list(probabilities), dtype=np.float64)
    positives, negatives = int((labels == 1).sum()), int((labels == 0).sum())
    if not positives or not negatives:
        return None
    order = np.argsort(scores, kind="stable")
    ranks = np.empty(len(scores), dtype=np.float64)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and scores[order[end]] == scores[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end
    return float((ranks[labels == 1].sum() - positives * (positives + 1) / 2) / (positives * negatives))


def classification_metrics(records: list[dict]) -> dict:
    if not records:
        return {"n": 0, "auc": None, "accuracy": None, "balanced_accuracy": None,
                "f1": None, "brier": None, "nll": None}
    y = np.asarray([r["label"] for r in records], dtype=np.int64)
    p = np.asarray([r["probability"] for r in records], dtype=np.float64)
    predictions = p >= 0.5
    tp, tn = int(((y == 1) & predictions).sum()), int(((y == 0) & ~predictions).sum())
    fp, fn = int(((y == 0) & predictions).sum()), int(((y == 1) & ~predictions).sum())
    recalls = [numerator / denominator for numerator, denominator in
               ((tp, tp + fn), (tn, tn + fp)) if denominator]
    clipped = np.clip(p, 1e-7, 1 - 1e-7)
    return {"n": len(records), "auc": binary_auc(y, p),
            "accuracy": float((predictions == y).mean()),
            "balanced_accuracy": float(np.mean(recalls)) if len(recalls) == 2 else None,
            "f1": float(2 * tp / (2 * tp + fp + fn)) if 2 * tp + fp + fn else 0.0,
            "brier": float(np.square(p - y).mean()),
            "nll": float(-(y * np.log(clipped) + (1 - y) * np.log(1 - clipped)).mean()),
            "confusion_matrix": [[tn, fp], [fn, tp]]}


def _hd95(prediction: np.ndarray, target: np.ndarray) -> float | None:
    if not prediction.any() and not target.any():
        return 0.0
    if not prediction.any() or not target.any():
        return None
    from scipy.ndimage import binary_erosion, distance_transform_edt
    pred_edge = prediction ^ binary_erosion(prediction)
    target_edge = target ^ binary_erosion(target)
    distances = np.concatenate((distance_transform_edt(~target_edge)[pred_edge],
                                distance_transform_edt(~pred_edge)[target_edge]))
    return float(np.percentile(distances, 95))


def segmentation_metrics(records: list[dict]) -> dict:
    if not records:
        return {"n_masks": 0, "dice": None, "iou": None, "hd95_pixels": None}
    hd = [r["hd95"] for r in records if r.get("hd95") is not None]
    return {"n_masks": len(records), "dice": float(np.mean([r["dice"] for r in records])),
            "iou": float(np.mean([r["iou"] for r in records])),
            "hd95_pixels": float(np.mean(hd)) if hd else None,
            "hd95_defined_masks": len(hd)}


def _case_aggregate(records: list[dict]) -> tuple[list[dict], int]:
    groups = defaultdict(list)
    for record in records:
        groups[(record["dataset_id"], record["task"], record["label_space_id"], record["case_id"])].append(record)
    aggregated, conflicts = [], 0
    for items in groups.values():
        if len({r["label"] for r in items}) != 1:
            conflicts += 1
            continue
        item = dict(items[0])
        item["probability"] = float(np.mean([r["probability"] for r in items]))
        item["clips"] = len(items)
        aggregated.append(item)
    return aggregated, conflicts


class MetricAccumulator:
    def __init__(self, threshold: float = 0.5, hd95: bool = False, classification_unit: str = "case"):
        if classification_unit not in {"case", "sample"}:
            raise ValueError("classification_unit must be case or sample")
        self.threshold, self.hd95, self.classification_unit = threshold, hd95, classification_unit
        self.segmentation: list[dict] = []
        self.classification: list[dict] = []
        self.samples = 0
        self.memory_bytes: list[int] = []

    def update(self, output: dict, batch) -> None:
        from tum_us.data.registry import TASKS
        probabilities = output["class_logits"].detach().float().softmax(-1).cpu()
        from tum_us.model.losses import supervised_mask_logits
        logits = supervised_mask_logits(output, batch).detach().float()
        target = batch.masks.detach().float()
        if logits.shape[:3] != target.shape[:3]:
            raise ValueError(f"Prediction/target B,T,Q mismatch: {tuple(logits.shape)} vs {tuple(target.shape)}")
        if logits.shape[-2:] != target.shape[-2:]:
            logits = F.interpolate(logits.flatten(0, 2).unsqueeze(1), size=target.shape[-2:],
                                   mode="bilinear", align_corners=False).squeeze(1).reshape_as(target)
        predictions = (logits.sigmoid() >= self.threshold).cpu().numpy()
        targets = (target >= 0.5).cpu().numpy()
        mask_valid = batch.mask_valid.detach().cpu().bool()
        frame_valid = batch.frame_valid.detach().cpu().bool()
        class_valid = batch.class_valid.detach().cpu().bool()
        task_ids = batch.task_ids.detach().cpu().reshape(-1).tolist()
        frame_indices = batch.frame_indices.detach().cpu()
        target_ids = batch.target_ids.detach().cpu()
        labels = batch.class_targets.detach().cpu().reshape(-1)
        spaces = batch.label_space_ids.detach().cpu().reshape(-1)
        for b, metadata in enumerate(batch.metadata):
            self.samples += 1
            task = TASKS[task_ids[b]]
            base = {"dataset_id": str(metadata.get("dataset_id", "unknown")), "task": task,
                    "privacy": str(metadata.get("privacy", "unknown")),
                    "case_id": str(metadata.get("case_id", metadata.get("entity_uid", metadata.get("record_id", self.samples)))),
                    "record_id": str(metadata.get("record_id", "")),
                    "sample_path": str(metadata.get("sample_path", ""))}
            if class_valid[b]:
                self.classification.append({**base, "label_space_id": int(spaces[b]),
                                            "label": int(labels[b]), "probability": float(probabilities[b, 1])})
            for t, q in torch.nonzero(mask_valid[b], as_tuple=False).tolist():
                if not frame_valid[b, t]:
                    raise ValueError("mask_valid points to a padded/invalid frame.")
                pred, truth = predictions[b, t, q], targets[b, t, q]
                intersection = int(np.logical_and(pred, truth).sum())
                union = int(np.logical_or(pred, truth).sum())
                size = int(pred.sum() + truth.sum())
                self.segmentation.append({**base, "frame_index": int(frame_indices[b, t]),
                    "target_id": int(target_ids[b, q] if target_ids.ndim == 2 else target_ids[q]),
                    "dice": float(2 * intersection / size) if size else 1.0,
                    "iou": float(intersection / union) if union else 1.0,
                    "hd95": _hd95(pred, truth) if self.hd95 else None})
        self.memory_bytes.extend(int(x) for x in output.get("memory_bytes", []))

    def compute(self) -> dict:
        classes, conflicts = _case_aggregate(self.classification) if self.classification_unit == "case" else (self.classification, 0)
        def summarize(seg, cls):
            classification = classification_metrics(cls)
            if len({r["label_space_id"] for r in cls}) > 1:
                classification["auc"] = None
                classification["auc_note"] = "Different clinical label spaces; pooled AUC is not reported."
            return {"segmentation": segmentation_metrics(seg), "classification": classification}
        result = {"overall": summarize(self.segmentation, classes), "counts": {
            "samples": self.samples, "annotated_masks": len(self.segmentation),
            "classification_samples": len(self.classification), "classification_cases": len(classes),
            "conflicting_cases_excluded": conflicts}, "classification_unit": self.classification_unit}
        for key, name in (("task", "by_task"), ("dataset_id", "by_dataset"),
                          ("privacy", "by_privacy"), ("label_space_id", "by_label_space")):
            keys = sorted({str(r[key]) for r in self.segmentation + classes if key in r})
            result[name] = {value: summarize([r for r in self.segmentation if str(r.get(key)) == value],
                                            [r for r in classes if str(r.get(key)) == value]) for value in keys}
        # Classification label spaces encode different clinical questions: the pooled AUC
        # is omitted whenever their binary semantics differ.
        spaces = {r["label_space_id"] for r in classes}
        if len(spaces) > 1:
            result["overall"]["classification"]["auc"] = None
            result["overall"]["classification"]["auc_note"] = "Different label spaces; use by_label_space/by_dataset."
        datasets = list(result["by_dataset"].values())
        for metric, family in (("dice", "segmentation"), ("auc", "classification")):
            values = [item[family][metric] for item in datasets if item[family].get(metric) is not None]
            result.setdefault("macro_dataset", {})[metric] = float(np.mean(values)) if values else None
        task_values = {}
        for task in sorted({r["task"] for r in self.segmentation + classes}):
            members = {r["dataset_id"] for r in self.segmentation + classes if r["task"] == task}
            family, metric = ("segmentation", "dice") if task.endswith("seg") else ("classification", "auc")
            values = [result["by_dataset"][name][family][metric] for name in members
                      if result["by_dataset"][name][family].get(metric) is not None]
            task_values[task] = float(np.mean(values)) if values else None
        defined = [value for value in task_values.values() if value is not None]
        result["macro_task"] = {"by_task": task_values,
                                "score": float(np.mean(defined)) if defined else None,
                                "definition": "Mean of dataset means within each task, then equal mean over defined tasks."}
        result["memory"] = {"peak_retained_bytes": max(self.memory_bytes, default=0),
                            "mean_retained_bytes": float(np.mean(self.memory_bytes)) if self.memory_bytes else 0.0}
        return result
