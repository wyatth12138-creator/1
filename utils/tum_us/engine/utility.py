"""Frozen-reference utility caches and supervised set-conditioned controllers."""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
from collections import OrderedDict
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from .checkpoint import load_checkpoint, read_checkpoint, save_checkpoint
from .config import artifact_path, write_json
from .runner import _load_model_weights, make_dataset, make_loader
from .runtime import (append_jsonl, autocast_context, choose_device, finite_losses,
                      gradient_summary, move_batch, registry_snapshot, seed_everything)

UTILITY_FORMAT = "tum_us.utility.v1"


def file_digest(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_example(example: dict) -> dict:
    keys = ("descriptors", "deltas", "valid_tasks", "anchor_mask", "frame_indices")
    result = {key: torch.as_tensor(example[key]).detach().cpu().contiguous() for key in keys}
    result["active_tasks"] = torch.as_tensor(example.get("active_tasks", result["valid_tasks"])).detach().cpu().bool()
    n = len(result["descriptors"])
    if result["descriptors"].shape != (n, 328) or result["deltas"].shape != (n, 2):
        raise ValueError("Utility examples require descriptors [N,328] and deltas [N,2].")
    if result["valid_tasks"].shape != (2,) or result["active_tasks"].shape != (2,) or result["anchor_mask"].shape != (n,) or result["frame_indices"].shape != (n,):
        raise ValueError("Invalid utility validity/anchor/frame-index shapes.")
    if n < 2 or not (result["valid_tasks"].bool() & result["active_tasks"]).any() or result["anchor_mask"].bool().all():
        raise ValueError("Utility states need an evictable slot and at least one supervised task.")
    if not torch.isfinite(result["descriptors"]).all() or not torch.isfinite(result["deltas"]).all():
        raise FloatingPointError("Utility examples contain non-finite values.")
    result["descriptors"] = result["descriptors"].float()
    result["deltas"] = result["deltas"].float()
    result["valid_tasks"] = result["valid_tasks"].bool()
    result["anchor_mask"] = result["anchor_mask"].bool()
    result["frame_indices"] = result["frame_indices"].long()
    return result


def build_utility(cfg: dict, checkpoint: str | None = None) -> dict:
    cfg = copy.deepcopy(cfg)
    options = cfg["utility"]
    reference = checkpoint or options.get("reference_checkpoint")
    if not reference:
        raise ValueError("build-utility requires a trained stage-one --checkpoint/reference_checkpoint.")
    if options["split"] != cfg["data"]["train_split"]:
        raise ValueError("Controller training utility must be built only on the configured training split.")
    device = choose_device(cfg["training"]["device"])
    seed_everything(int(cfg["training"]["seed"]))
    model = _load_model_weights(cfg, reference).to(device)
    model.requires_grad_(False).eval()
    tasks = [task for task in cfg["data"]["tasks"] if task in {"video_seg", "ceus_seg", "ceus_cls", "video_cls"}]
    if not tasks:
        raise ValueError("No temporal tasks were selected for memory utility generation.")
    dataset = make_dataset(cfg, options["split"], tasks=tasks)
    loader = make_loader(cfg, dataset)
    directory = artifact_path(cfg, "cache", "utility", options["name"])
    index_path = directory / "index.json"
    fingerprint_payload = {"reference_sha256": file_digest(reference),
        "manifest_sha256": file_digest(cfg["manifest"]), "model": cfg["model"],
        "data": cfg["data"], "split": options["split"], "max_states": options["max_states"],
        "max_batches": options.get("max_batches"), "seed": cfg["training"]["seed"],
        "batch_size": cfg["training"]["batch_size"]}
    from tum_us.paths import MODEL_ROOT
    reader_root = MODEL_ROOT / "core"
    fingerprint_payload["reader_source_sha256"] = {
        source.name: file_digest(source) for source in sorted(reader_root.glob("*.py"))
    }
    fingerprint = hashlib.sha256(json.dumps(fingerprint_payload, sort_keys=True).encode()).hexdigest()
    directory.mkdir(parents=True, exist_ok=True)
    manifest = {"format": UTILITY_FORMAT, "fingerprint": fingerprint,
        "provenance": fingerprint_payload, "reference_checkpoint": str(reference),
        "config": cfg, "registry": registry_snapshot(), "split": options["split"],
        "shards": [], "processed_batches": 0, "examples": 0, "complete": False,
        "absolute_delta_sum": [0.0, 0.0], "delta_count": [0, 0]}
    if index_path.exists():
        with index_path.open(encoding="utf-8") as stream:
            previous = json.load(stream)
        if previous.get("fingerprint") != fingerprint:
            raise ValueError("Existing utility cache has different reference/data/settings; choose a new utility name.")
        manifest = previous
        if manifest["complete"]:
            return {"utility_path": str(index_path), "examples": manifest["examples"], "reused": True}
    sums = torch.tensor(manifest["absolute_delta_sum"], dtype=torch.float64)
    counts = torch.tensor(manifest["delta_count"], dtype=torch.long)
    for index, cpu_batch in enumerate(loader):
        if options.get("max_batches") is not None and index >= int(options["max_batches"]):
            break
        if index < manifest["processed_batches"]:
            continue
        # Per-batch seeds make a resumed utility build reproduce the same random subsets.
        seed_everything(int(cfg["training"]["seed"]) + index * 9973)
        batch = move_batch(cpu_batch, device)
        with torch.no_grad(), autocast_context(device, cfg["training"]["amp"]):
            examples = [_validate_example(item) for item in model.utility_examples(batch, max_states=int(options["max_states"]))]
        if examples:
            name = f"shard_{index:07d}.pt"
            payload = {"format": UTILITY_FORMAT, "examples": examples,
                "source_records": [str(item.get("record_id", "")) for item in cpu_batch.metadata]}
            temporary = directory / (name + ".tmp")
            torch.save(payload, temporary)
            os.replace(temporary, directory / name)
            manifest["shards"].append({"path": name, "count": len(examples)})
            for item in examples:
                active = (~item["anchor_mask"])[:, None] & (item["valid_tasks"] & item["active_tasks"])[None]
                sums += (item["deltas"].abs().double() * active).sum(0)
                counts += active.sum(0)
            manifest["examples"] += len(examples)
        manifest["processed_batches"] = index + 1
        manifest["absolute_delta_sum"] = sums.tolist()
        manifest["delta_count"] = counts.tolist()
        write_json(index_path, manifest)
        print(f"utility batch={index + 1} states={len(examples)} total={manifest['examples']}", flush=True)
    if not manifest["examples"]:
        raise RuntimeError("No valid memory deletion states were generated; no controller cache was produced.")
    manifest["complete"] = True
    write_json(index_path, manifest)
    return {"utility_path": str(index_path), "examples": manifest["examples"],
            "delta_count": manifest["delta_count"], "reused": False}


class UtilityDataset(Dataset):
    def __init__(self, index_path: str | Path):
        self.index_path = Path(index_path)
        with self.index_path.open(encoding="utf-8") as stream:
            self.manifest = json.load(stream)
        if self.manifest.get("format") != UTILITY_FORMAT or not self.manifest.get("complete"):
            raise ValueError("Utility index must be a completed TUM-US utility build.")
        self.entries = [(shard["path"], index) for shard in self.manifest["shards"] for index in range(shard["count"])]
        self.cache = OrderedDict()

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, index):
        name, offset = self.entries[index]
        if name not in self.cache:
            path = (self.index_path.parent / name).resolve()
            if not path.is_relative_to(self.index_path.parent.resolve()):
                raise ValueError("Utility shard path escapes its cache directory.")
            payload = torch.load(path, map_location="cpu", weights_only=True)
            if payload.get("format") != UTILITY_FORMAT:
                raise ValueError("Invalid utility shard format.")
            self.cache[name] = payload["examples"]
            if len(self.cache) > 2:
                self.cache.popitem(last=False)
        return self.cache[name][offset]


def collate_utility(examples: list[dict]) -> dict[str, torch.Tensor]:
    maximum = max(len(item["descriptors"]) for item in examples)
    b = len(examples)
    result = {"descriptors": torch.zeros(b, maximum, 328), "deltas": torch.zeros(b, maximum, 2),
              "slot_mask": torch.zeros(b, maximum, dtype=torch.bool),
              "anchor_mask": torch.ones(b, maximum, dtype=torch.bool),
              "valid_tasks": torch.stack([item["valid_tasks"].bool() for item in examples]),
              "active_tasks": torch.stack([item.get("active_tasks", item["valid_tasks"]).bool() for item in examples])}
    for index, item in enumerate(examples):
        n = len(item["descriptors"])
        result["descriptors"][index, :n] = item["descriptors"]
        result["deltas"][index, :n] = item["deltas"]
        result["anchor_mask"][index, :n] = item["anchor_mask"]
        result["slot_mask"][index, :n] = True
    return result


def controller_objective(predictions: torch.Tensor, batch: dict, scales: torch.Tensor,
                         temperature: float = 0.2, regression_weight: float = 1.0,
                         ranking_weight: float = 0.25) -> dict:
    if temperature <= 0:
        raise ValueError("Controller temperature must be positive.")
    candidates = batch["slot_mask"] & ~batch["anchor_mask"]
    requested = batch.get("active_tasks", batch["valid_tasks"])
    known = batch["valid_tasks"] & requested
    active = candidates[:, :, None] & known[:, None, :]
    if not active.any() or not candidates.any(dim=1).all() or not known.any(dim=1).all():
        raise ValueError("Each utility state requires an evictable candidate and a supervised task.")
    target = batch["deltas"] / scales[None, None, :]
    regression = F.smooth_l1_loss(predictions[active], target[active])
    tasks = requested[:, None, :]
    target_cost = target.masked_fill(~tasks, float("-inf")).amax(-1)
    predicted_cost = predictions.masked_fill(~tasks, float("-inf")).amax(-1)
    fully_supervised = ((~requested) | batch["valid_tasks"]).all(-1) & requested.any(-1)
    if fully_supervised.any():
        allowed = candidates[fully_supervised]
        costs = target_cost[fully_supervised]
        teacher_logits = (-costs / temperature).masked_fill(~allowed, float("-inf"))
        logits = (-predicted_cost[fully_supervised] / temperature).masked_fill(~allowed, float("-inf"))
        soft_target = teacher_logits.softmax(-1)
        # Avoid 0 * -inf on anchor and padded entries.
        log_probabilities = logits.log_softmax(-1).masked_fill(~allowed, 0.0)
        ranking = -(soft_target * log_probabilities).sum(-1).mean()
        with torch.no_grad():
            chosen = logits.argmax(-1)
            reference = teacher_logits.argmax(-1)
            agreement = (chosen == reference).float().mean()
            selected_cost = costs.gather(1, chosen[:, None]).squeeze(1)
            best_cost = costs.masked_fill(~allowed, float("inf")).amin(-1)
            excess = (selected_cost - best_cost).mean()
    else:
        ranking = predictions.sum() * 0
        agreement = predictions.new_zeros(())
        excess = predictions.new_zeros(())
    return {"loss": regression_weight * regression + ranking_weight * ranking,
            "regression_loss": regression, "ranking_loss": ranking,
            "action_agreement": agreement, "excess_deletion_cost": excess,
            "ranking_states": fully_supervised.sum().to(predictions.dtype)}


def train_controller(cfg: dict, utility_path: str | None = None, checkpoint: str | None = None) -> dict:
    cfg = copy.deepcopy(cfg)
    options = cfg["controller"]
    index_path = utility_path or options.get("utility_path")
    if not index_path:
        raise ValueError("train-controller requires --utility-path or controller.utility_path.")
    dataset = UtilityDataset(index_path)
    reference = checkpoint or options.get("reference_checkpoint") or dataset.manifest["reference_checkpoint"]
    if file_digest(reference) != dataset.manifest["provenance"]["reference_sha256"]:
        raise ValueError("Controller reference checkpoint differs from utility-label reference.")
    seed_everything(int(cfg["training"]["seed"]))
    device = choose_device(cfg["training"]["device"])
    model = _load_model_weights(cfg, reference).to(device)
    model.requires_grad_(False).eval()
    model.controller.requires_grad_(True)
    model.config.policy = "tum"
    cfg["model"]["policy"] = "tum"
    sums = torch.tensor(dataset.manifest["absolute_delta_sum"], dtype=torch.float32)
    counts = torch.tensor(dataset.manifest["delta_count"], dtype=torch.float32)
    epsilon = float(options["scale_epsilon"])
    scales = torch.where(counts > 0, sums / counts.clamp_min(1), torch.ones_like(sums)).clamp_min(epsilon)
    model.controller.set_scales(scales)
    optimizer = torch.optim.AdamW(model.controller.parameters(), lr=float(options["learning_rate"]),
                                 weight_decay=float(options["weight_decay"]))
    batch_size = int(options["batch_size"])
    if batch_size < 1 or not len(dataset):
        raise ValueError("Controller requires positive batch size and nonempty utility data.")
    batches_per_epoch = math.ceil(len(dataset) / batch_size)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, int(options["epochs"]) * batches_per_epoch))
    registry = registry_snapshot()
    run_dir = artifact_path(cfg, "runs", options["run_name"])
    checkpoint_dir = artifact_path(cfg, "checkpoints", options["run_name"])
    write_json(run_dir / "config.json", cfg)
    write_json(run_dir / "utility_provenance.json", {"index_path": str(index_path),
        "fingerprint": dataset.manifest["fingerprint"], "scales": scales.tolist(), "delta_count": counts.tolist()})
    start_epoch, global_step = 0, 0
    if options.get("resume"):
        payload = load_checkpoint(options["resume"], model, optimizer=optimizer, scheduler=scheduler,
                                  expected_phase="stage2", registry=registry)
        if payload["extra"].get("utility_fingerprint") != dataset.manifest["fingerprint"]:
            raise ValueError("Resuming a controller requires the same utility cache.")
        start_epoch, global_step = payload["epoch"], payload["global_step"]
    for epoch in range(start_epoch, int(options["epochs"])):
        generator = torch.Generator().manual_seed(int(cfg["training"]["seed"]) + epoch * 1000003)
        order = torch.randperm(len(dataset), generator=generator).tolist()
        batches = [order[index:index + batch_size] for index in range(0, len(order), batch_size)]
        loader = DataLoader(dataset, batch_sampler=batches, collate_fn=collate_utility,
                            num_workers=0, generator=generator)
        model.controller.train()
        epoch_values = []
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            predictions = model.controller(batch["descriptors"], batch["slot_mask"])
            losses = controller_objective(predictions, batch, model.controller.scales,
                float(options["temperature"]), float(options["regression_weight"]), float(options["ranking_weight"]))
            values = finite_losses(losses)
            losses["loss"].backward()
            gradients = gradient_summary(model)
            torch.nn.utils.clip_grad_norm_(model.controller.parameters(), float(cfg["training"]["clip_grad_norm"]))
            optimizer.step()
            scheduler.step()
            global_step += 1
            epoch_values.append(values)
            append_jsonl(run_dir / "train.jsonl", {"phase": "stage2", "epoch": epoch,
                "global_step": global_step, "losses": values, "gradients": gradients,
                "learning_rate": optimizer.param_groups[0]["lr"]})
        # Evaluation of the resulting checkpoint uses the learned minimax policy.
        model.config.policy = "tum"
        save_checkpoint(checkpoint_dir / "last.pt", model, optimizer=optimizer, scheduler=scheduler,
            phase="stage2", config=cfg, registry=registry, epoch=epoch + 1,
            global_step=global_step, extra={"utility_fingerprint": dataset.manifest["fingerprint"],
            "scales": model.controller.scales.detach().cpu().tolist(),
            "mean_losses": {key: sum(row[key] for row in epoch_values) / len(epoch_values) for key in epoch_values[0]}})
        print(f"controller epoch={epoch + 1} loss={sum(row['loss'] for row in epoch_values) / len(epoch_values):.5f}", flush=True)
    result = {"phase": "stage2", "global_step": global_step, "checkpoint": str(checkpoint_dir / "last.pt"),
              "run_dir": str(run_dir), "scales": model.controller.scales.detach().cpu().tolist()}
    write_json(run_dir / "result.json", result)
    return result
