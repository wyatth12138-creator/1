"""Entity-level reporting. Dataset endpoints and task metrics remain separate."""
from __future__ import annotations

from collections import defaultdict
import math
import numpy as np

from tum_us.engine.metrics import binary_auc as _auc
from tum_us.transfer.metrics import (
    CLASSIFICATION_THRESHOLD,
    classification_metric_summary,
    segmentation_metric_summary,
)


def _metric(rows):
    if not rows:
        raise ValueError("Cannot report an empty endpoint")
    kind = rows[0]["kind"]
    if any(r["kind"] != kind for r in rows):
        raise ValueError("An endpoint cannot mix classification and segmentation")
    if kind == "classification":
        return _auc([r["label"] for r in rows], [r["probability"] for r in rows])
    if kind != "segmentation":
        raise ValueError(f"Unknown endpoint kind {kind!r}")
    return float(np.mean([r["dice"] for r in rows]))


def _interval(values):
    values = [v for v in values if v is not None and math.isfinite(v)]
    return {"low": float(np.quantile(values, .025)), "high": float(np.quantile(values, .975)),
            "valid_replicates": len(values)} if values else {"low": None, "high": None, "valid_replicates": 0}


def _check_rows(rows):
    keys = [(r["endpoint"], str(r["key"]), int(r.get("seed", 0))) for r in rows]
    if len(set(keys)) != len(keys):
        raise ValueError("Expected one prediction per endpoint/entity/seed; aggregate windows first")
    for r in rows:
        fields = ("probability",) if r["kind"] == "classification" else ("dice",)
        if any(not math.isfinite(float(r[f])) or not 0 <= float(r[f]) <= 1 for f in fields):
            raise ValueError("Predictions must be finite and lie in [0,1]")
        if r["kind"] == "classification" and r["label"] not in (0, 1):
            raise ValueError("Binary endpoint labels must be 0 or 1")
        if r.get("hd95_pixels") is not None and not 0 <= float(r["hd95_pixels"]) < math.inf:
            raise ValueError("HD95 must be finite and nonnegative")


def _seed_groups(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[int(row.get("seed", 0))].append(row)
    return grouped


def _sample_indices(rows, rng):
    # Classification resampling preserves both outcome classes, avoiding
    # undefined AUROC replicates in small development cohorts.
    if rows[0]["kind"] == "classification":
        out = []
        for label in (0, 1):
            indices = [i for i, r in enumerate(rows) if r["label"] == label]
            if indices:
                out.extend(rng.choice(indices, size=len(indices), replace=True).tolist())
        return out
    return rng.integers(0, len(rows), len(rows)).tolist()


def summarize_predictions(records, bootstrap=1000, seed=3407):
    records = list(records)
    _check_rows(records)
    groups = defaultdict(list)
    for row in records:
        groups[row["endpoint"]].append(row)
    rng = np.random.default_rng(seed)
    result = {}
    for endpoint, rows in sorted(groups.items()):
        by_seed = _seed_groups(rows)
        seeds = sorted(by_seed)
        primary_by_seed = {str(s): _metric(by_seed[s]) for s in seeds}
        observed = [value for value in primary_by_seed.values() if value is not None]
        is_classification = rows[0]["kind"] == "classification"
        summary = classification_metric_summary(rows) if is_classification else segmentation_metric_summary(rows)
        if is_classification:
            secondary_metric = "accuracy"
            secondary_by_row = lambda row: float(int(row["probability"] >= CLASSIFICATION_THRESHOLD) == row["label"])
        elif "hd95_pixels" in summary:
            secondary_metric = "hd95_pixels"
            secondary_by_row = lambda row: row["hd95_pixels"]
        else:
            secondary_metric = None
        replicates = []
        secondary_replicates = []
        # Crossed seed/entity bootstrap: the same sampled entity identities are
        # used for every sampled seed; seeds are never treated as new patients.
        keysets = [{str(r["key"]) for r in by_seed[s]} for s in seeds]
        if any(k != keysets[0] for k in keysets[1:]):
            raise ValueError("Seeds must score the same endpoint entities")
        ordered = {s: sorted(by_seed[s], key=lambda r: str(r["key"])) for s in seeds}
        labels = [[r.get("label") for r in ordered[s]] for s in seeds]
        if any(y != labels[0] for y in labels[1:]):
            raise ValueError("Entity labels differ between seeds")
        secondary_values = (
            {s: np.asarray([secondary_by_row(row) for row in ordered[s]]) for s in seeds}
            if secondary_metric else {}
        )
        for _ in range(int(bootstrap)):
            indices = _sample_indices(ordered[seeds[0]], rng)
            sampled_seeds = rng.choice(seeds, size=len(seeds), replace=True)
            values = [_metric([ordered[int(s)][i] for i in indices]) for s in sampled_seeds]
            if all(v is not None for v in values):
                replicates.append(float(np.mean(values)))
            if secondary_metric:
                secondary_replicates.append(float(np.mean([
                    secondary_values[s][indices].mean() for s in sampled_seeds
                ])))
        item = {"kind": rows[0]["kind"], "metric": "auroc" if rows[0]["kind"] == "classification" else "dice",
                "value": float(np.mean(observed)) if observed else None,
                "ci95": _interval(replicates), "entities": len(keysets[0]), "seeds": seeds,
                "per_seed": primary_by_seed}
        item.update(summary)
        item[item["metric"]] = item["value"]
        if is_classification:
            item["auc"] = item["value"]
            item["n_positive"] = labels[0].count(1)
            item["n_negative"] = labels[0].count(0)
        if secondary_metric:
            item[secondary_metric + "_ci95"] = _interval(secondary_replicates)
            item[secondary_metric + "_per_seed"] = {
                str(s): float(secondary_values[s].mean()) for s in seeds
            }
        for field in ("loss", "memory_bytes", "seconds"):
            values = [float(r[field]) for r in rows if r.get(field) is not None]
            if values:
                item["mean_" + field] = float(np.mean(values))
                if field == "memory_bytes":
                    item["max_memory_bytes"] = max(values)
        result[endpoint] = item
    return {"endpoints": result, "bootstrap": int(bootstrap), "bootstrap_seed": seed,
            "unit": "entity; crossed seed/entity bootstrap; class-stratified for AUROC",
            "pooled_task_metric": None}


def paired_comparison(candidate, reference, bootstrap=2000, seed=3407):
    candidate, reference = list(candidate), list(reference)
    _check_rows(candidate); _check_rows(reference)
    def key(r):
        return (r["endpoint"], int(r.get("seed", 0)), str(r["key"]))
    left, right = {key(r): r for r in candidate}, {key(r): r for r in reference}
    if left.keys() != right.keys():
        raise ValueError("Paired comparison requires identical endpoint/entity/seed coverage")
    for k in left:
        if left[k]["kind"] != right[k]["kind"] or left[k].get("label") != right[k].get("label"):
            raise ValueError("Paired outcomes disagree")
    endpoints = sorted({k[0] for k in left})
    rng, result = np.random.default_rng(seed), {}
    for endpoint in endpoints:
        seeds = sorted({k[1] for k in left if k[0] == endpoint})
        keys = {s: sorted([k for k in left if k[:2] == (endpoint, s)]) for s in seeds}
        identities = [[k[2] for k in keys[s]] for s in seeds]
        if any(ids != identities[0] for ids in identities[1:]):
            raise ValueError("All seeds must score the same entities")
        diffs = []
        for s in seeds:
            a, b = _metric([left[k] for k in keys[s]]), _metric([right[k] for k in keys[s]])
            if a is not None and b is not None:
                diffs.append(a-b)
        draws = []
        for _ in range(int(bootstrap)):
            indices = _sample_indices([left[k] for k in keys[seeds[0]]], rng)
            values = []
            for s in rng.choice(seeds, len(seeds), replace=True):
                sampled = [keys[int(s)][i] for i in indices]
                a, b = _metric([left[k] for k in sampled]), _metric([right[k] for k in sampled])
                if a is not None and b is not None:
                    values.append(a-b)
            if len(values) == len(seeds):
                draws.append(float(np.mean(values)))
        result[endpoint] = {"delta": float(np.mean(diffs)) if diffs else None,
                            "ci95": _interval(draws), "seeds": seeds,
                            "entities": len(keys[seeds[0]]), "direction": "higher is better"}
    return {"endpoints": result, "bootstrap": bootstrap, "paired": True}


def utility_diagnostics(predictions, batch):
    """Observed-axis decision regret, never evaluating an unobserved axis."""
    import torch
    result = {}
    eligible = batch["slot_mask"].bool() & ~batch["anchor_mask"].bool()
    for axis, name in enumerate(("segmentation", "classification")):
        known = batch["valid_tasks"][:, axis].bool()
        if "active_tasks" in batch:
            known &= batch["active_tasks"][:, axis].bool()
        if not known.any():
            result[name] = {"states": 0, "agreement": None, "mean_excess_cost": None}
            continue
        pred = predictions[known, :, axis].masked_fill(~eligible[known], float("inf"))
        true = batch["deltas"][known, :, axis].masked_fill(~eligible[known], float("inf"))
        chosen, best = pred.argmin(-1), true.argmin(-1)
        excess = true.gather(1, chosen[:, None]).squeeze(1) - true.min(-1).values
        result[name] = {"states": int(known.sum()), "agreement": float((chosen == best).float().mean()),
                        "mean_excess_cost": float(excess.mean())}
    return result


def reader_policy_interaction(object_tum,object_fifo,native_tum,native_fifo,bootstrap=1000,seed=3407):
    """Paired four-condition interaction, not four independent confidence bands.

    Positive values mean utility selection gains more with the object reader:
    (object/tum - object/fifo) - (native/tum - native/fifo).
    """
    conditions=[list(x) for x in (object_tum,object_fifo,native_tum,native_fifo)]
    for rows in conditions: _check_rows(rows)
    key=lambda r:(r['endpoint'],int(r.get('seed',0)),str(r['key']))
    tables=[{key(r):r for r in rows} for rows in conditions]
    if any(t.keys()!=tables[0].keys() for t in tables[1:]): raise ValueError('Interaction requires identical endpoint/entity/seed coverage')
    for k in tables[0]:
        if any((t[k]['kind'],t[k].get('label'))!=(tables[0][k]['kind'],tables[0][k].get('label')) for t in tables[1:]):
            raise ValueError('Interaction condition outcomes disagree')
    def effect(keys):
        values=[_metric([table[k] for k in keys]) for table in tables]
        return None if any(v is None for v in values) else values[0]-values[1]-values[2]+values[3]
    rng=np.random.default_rng(seed); result={}
    for endpoint in sorted({k[0] for k in tables[0]}):
        seeds=sorted({k[1] for k in tables[0] if k[0]==endpoint})
        keys={s:sorted(k for k in tables[0] if k[:2]==(endpoint,s)) for s in seeds}
        identities=[[k[2] for k in keys[s]] for s in seeds]
        if any(x!=identities[0] for x in identities[1:]): raise ValueError('Interaction seeds must score identical entities')
        observed=[effect(keys[s]) for s in seeds]; valid=[v for v in observed if v is not None]; draws=[]
        for _ in range(int(bootstrap)):
            indices=_sample_indices([tables[0][k] for k in keys[seeds[0]]],rng)
            values=[effect([keys[int(s)][i] for i in indices]) for s in rng.choice(seeds,len(seeds),replace=True)]
            if all(v is not None for v in values): draws.append(float(np.mean(values)))
        result[endpoint]={'interaction':float(np.mean(valid)) if valid else None,'ci95':_interval(draws),'entities':len(keys[seeds[0]]),'seeds':seeds}
    return {'formula':'(object_tum-object_fifo)-(native_tum-native_fifo)','endpoints':result,'paired':True,'bootstrap':bootstrap}
