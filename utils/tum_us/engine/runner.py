"""Shared construction helpers and validation; the training workflow lives in train.py."""
from __future__ import annotations

import copy
import time
from dataclasses import asdict, fields
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .checkpoint import capture_rng, load_checkpoint, read_checkpoint, restore_rng
from .config import write_json
from .metrics import MetricAccumulator
from .runtime import autocast_context, finite_losses, move_batch, registry_snapshot, seed_everything
from .samplers import BalancedGroupBatchSampler, SequentialGroupBatchSampler
from .visualize import save_prediction_panels


def make_dataset(cfg: dict, split: str, *, tasks=None, max_records=None):
    from tum_us.data import build_dataset
    data = cfg["data"]
    return build_dataset(data_root=cfg["data_root"], manifest_path=cfg["manifest"], split=split,
        tasks=data["tasks"] if tasks is None else tasks, privacy=data["privacy"],
        image_size=int(cfg["model"]["image_size"]), clip_length=int(data["clip_length"]),
        dataset_ids=data.get("dataset_ids"),
        max_records=data.get("max_records") if max_records is None else max_records,
        single_mask_policy=data.get("single_mask_policy", "center_reference"))


def make_loader(cfg: dict, dataset, *, train: bool = False):
    from tum_us.data import collate_samples
    options = cfg["training"]
    sampler = BalancedGroupBatchSampler(dataset, options["batch_size"], options.get("steps_per_epoch"), options["seed"]) if train else SequentialGroupBatchSampler(dataset, options["batch_size"])
    generator = torch.Generator().manual_seed(int(options["seed"]) + (11 if train else 29))
    return DataLoader(dataset, batch_sampler=sampler, collate_fn=collate_samples,
        num_workers=int(options["num_workers"]), pin_memory=torch.cuda.is_available(),
        persistent_workers=False, generator=generator)


def build_model(cfg: dict, model_config: dict | None = None):
    from tum_us.model import TUMConfig, TUMModel
    parameters = copy.deepcopy(model_config or cfg["model"])
    supported = {item.name for item in fields(TUMConfig)}
    unknown = set(parameters) - supported
    if unknown:
        raise ValueError(f"Unknown model config fields: {sorted(unknown)}")
    return TUMModel(TUMConfig(**parameters))


def _load_model_weights(cfg: dict, checkpoint: str | None):
    if checkpoint:
        payload = read_checkpoint(checkpoint)
        saved_size = payload["model_config"].get("image_size", 512)
        if saved_size != cfg["model"]["image_size"]:
            raise ValueError("Checkpoint input size differs; use train.py --init-checkpoint for scale adaptation")
        cfg["model"] = copy.deepcopy(payload["model_config"])
        model = build_model(cfg)
        load_checkpoint(checkpoint, model, registry=registry_snapshot(), restore_randomness=False)
        cfg["model"] = asdict(model.config)
        return model
    model = build_model(cfg)
    cfg["model"] = asdict(model.config)
    return model


@torch.no_grad()
def evaluate_loader(model, loader, device: torch.device, cfg: dict, *, output_dir: Path | None = None,
                    max_batches: int | None = None, visualize_samples: int = 0) -> dict:
    from tum_us.model.losses import compute_losses
    previous_training = model.training
    state = capture_rng()
    seed_everything(int(cfg["training"]["seed"]) + 101)
    model.eval()
    options = cfg["evaluation"]
    metrics = MetricAccumulator(options["threshold"], options["hd95"], options["classification_unit"])
    losses_by_group: dict[str, list[float]] = {}
    timings, panels = [], 0
    try:
        for index, cpu_batch in enumerate(loader):
            if max_batches is not None and index >= max_batches:
                break
            batch = move_batch(cpu_batch, device)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            start = time.perf_counter()
            with autocast_context(device, cfg["training"]["amp"]):
                output = model(batch)
                loss_values = finite_losses(compute_losses(output, batch))
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            timings.append(time.perf_counter() - start)
            metrics.update(output, batch)
            group = str(cpu_batch.metadata[0].get("dataset_id", "unknown"))
            losses_by_group.setdefault(group, []).append(loss_values["loss"])
            if output_dir is not None and panels < visualize_samples:
                panels += save_prediction_panels(output, cpu_batch, output_dir / "panels", f"batch_{index:05d}", visualize_samples - panels)
        report = metrics.compute()
        report["loss_by_dataset"] = {key: sum(values) / len(values) for key, values in losses_by_group.items()}
        report["macro_dataset_loss"] = sum(report["loss_by_dataset"].values()) / len(losses_by_group) if losses_by_group else None
        report["batches"] = len(timings)
        report["mean_batch_seconds"] = sum(timings) / len(timings) if timings else None
        report["selection_score"] = report["macro_task"]["score"]
        if output_dir is not None:
            write_json(output_dir / "metrics.json", report)
            for filename, records in (("segmentation_predictions.jsonl", metrics.segmentation),
                                      ("classification_predictions.jsonl", metrics.classification)):
                path = output_dir / filename
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open("w", encoding="utf-8") as stream:
                    import json
                    for record in records:
                        stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
        return report
    finally:
        model.train(previous_training)
        restore_rng(state)
