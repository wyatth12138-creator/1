"""Controlled deletion costs under frozen readers and explicit future policies.

These labels are model-dependent deletion effects, not medical causal effects
or optimal-action ground truth. Future inputs/labels enter only target costs;
controller descriptors are captured at the current overflow state.
"""
from __future__ import annotations

import hashlib
import random
import torch
import torch.nn.functional as F

from ..model.losses import segmentation_loss
from ..model.memory import MemoryBank, slot_descriptors


def _new_bank(model, task, anatomy, label_space, targets, capacity, policy, rng=None):
    if policy not in MemoryBank.POLICIES:
        raise ValueError(f"Unknown continuation/behavior policy: {policy}")
    return model._new_bank(task, anatomy, label_space, targets, capacity=capacity, policy=policy, rng=rng)


def _state_indices(valid_times, capacity, max_states):
    # The first overflow occurs after capacity already-retained frames. Selection
    # uses positions only, never annotation presence or future image content.
    positions = valid_times[capacity:-1]
    if len(positions) <= max_states:
        return set(positions)
    if max_states == 1:
        return {positions[len(positions) // 2]}
    return {positions[round(i * (len(positions) - 1) / (max_states - 1))] for i in range(max_states)}


def _future_windows(future, horizons, include_terminal):
    requests = list(horizons)
    if include_terminal and "terminal" not in requests:
        requests.append("terminal")
    seen = set()
    for horizon in requests:
        times = tuple(future if horizon == "terminal" else future[:horizon])
        if times and times not in seen:
            seen.add(times)
            yield horizon, times


def _stream_seed(root_seed, *identity):
    value = repr((root_seed, *identity)).encode("utf-8")
    return int.from_bytes(hashlib.sha256(value).digest()[:8], "little")


@torch.no_grad()
def collect_examples(model, batch, capacities=(2, 4, 8), max_states=2, horizons=(1, 2, 4),
                     behavior_policy="fifo", continuation_policy="fifo", fixed_state=False, reuse_prefix=False,
                     target_objective="task"):
    """Collect centered eligible-action costs at actual capacity overflows.

    ``max_states`` limits decision states per case/capacity, not horizon labels.
    A branch removes one non-anchor candidate, then re-reads and re-writes every
    future valid frame under the same continuation policy. Cached image encoding
    is shared because it is independent of memory; all memory writes are fresh.

    ``target_objective`` defaults to native task losses without schema changes.
    The two consistency controls use a frozen same-reader causal full-memory
    teacher. Their values do not use GT values, but observation timestamps and
    axes are matched to task mode. Pointer cost averages cosine distance across
    valid views/targets; output cost is canonical-view soft BCE or terminal KL.
    This is a reader-internal teacher control, not a SPACE/JEPA reproduction.

    ``reuse_prefix`` shares deterministic continuation prefixes when enabled;
    random continuation retains separate per-window streams. This assumes pure,
    deterministic eval readers/writers (as in the current native implementation).

    Native class labels describe an examination, so classification cost is read
    only at its terminal valid frame. A terminal horizon is added automatically
    for classification-capable labeled cases; intermediate frames still undergo
    read/write continuation, without invented per-frame classification truth.
    """
    # These internal full-memory teacher controls are not SPACE/JEPA reproductions.
    if target_objective not in ("task", "pointer_consistency", "output_consistency"):
        raise ValueError("Unknown target_objective")
    # Opt-in: only deterministic continuation streams can share prefixes.
    if type(reuse_prefix) is not bool:
        raise ValueError("reuse_prefix must be boolean")
    capacities = tuple(int(c) for c in capacities)
    if not capacities or any(c < 2 for c in capacities) or len(set(capacities)) != len(capacities):
        raise ValueError("Collection capacities must be distinct integers >=2")
    if max_states < 1:
        raise ValueError("max_states must be positive")
    horizons = tuple(horizons)
    if not horizons or any(h != "terminal" and (not isinstance(h, int) or isinstance(h, bool) or h < 1)
                           for h in horizons):
        raise ValueError("Horizons must be positive step counts or 'terminal'")
    if behavior_policy in ("none", "full") or behavior_policy not in MemoryBank.POLICIES:
        raise ValueError("Utility collection needs a bounded behavior policy with overflow decisions")
    if continuation_policy not in MemoryBank.POLICIES:
        raise ValueError("Unknown continuation policy")
    was_training = model.training
    python_rng = random.getstate()
    root_seed = random.getrandbits(63)
    model.eval()
    examples = []
    try:
        for case in range(len(batch.media)):
            task = batch.task_ids[case]
            if int(task) not in (2, 3, 4, 5):
                continue  # T=1 image tasks have no temporal deletion decision.
            times = batch.frame_valid[case].nonzero().flatten().tolist()
            if len(times) < 4:
                continue
            frames = [int(batch.frame_indices[case, ti]) for ti in times]
            if any(b <= a for a, b in zip(frames, frames[1:])):
                raise ValueError("Rollout requires strictly increasing native frame indices")
            valid_targets = batch.target_ids[case] >= 0
            targets = batch.target_ids[case, valid_targets]
            if not len(targets):
                raise ValueError("Rollout requires at least one semantic target")
            anatomy = batch.anatomy_ids[case]
            label_space = int(batch.label_space_ids[case].clamp(min=0))
            view_valid = batch.view_valid[case]
            canonical_view = int(batch.seg_view_ids[case])
            if not 0 <= canonical_view < len(view_valid) or not bool(view_valid[canonical_view]):
                raise ValueError("Canonical segmentation view must be valid")
            requested = tuple(bool(v) for v in model.active_memory_tasks(int(task)))
            cls_available = requested[1] and bool(batch.class_valid[case]) and int(batch.class_targets[case]) >= 0
            final_time = times[-1]
            cached = {}

            def features(ti):
                if ti not in cached:
                    cached[ti] = model._encode(batch.media[case, ti])
                return cached[ti]

            def read(ti, memory):
                return model._read_views(features(ti), targets, task, anatomy, label_space,
                                         int(batch.frame_indices[case, ti]), memory, view_valid)

            teacher = {}

            def ensure_teacher():
                # Called only for proxy objectives with an observed deletion window.
                # The full stream is read causally before writing each current frame.
                # No annotation value or deletion-branch result enters this teacher.
                if teacher:
                    return
                full = _new_bank(model, task, anatomy, label_space, targets, 2, "full")
                for qi in times:
                    result = read(qi, full.slots)
                    teacher[qi] = {view: {
                        "pointers": r[1].detach().float().clone(),
                        "mask_probability": r[0].detach().float().sigmoid(),
                        "class_probability": r[4].detach().float().softmax(dim=-1),
                    } for view, r in result.items()}
                    slot = model._write_views(features(qi), result,
                                              int(batch.frame_indices[case, qi]), not full.slots)
                    full.update(slot, requested)

            def pointer_cost(result, qi, target_mask=None):
                values = []
                for view, r in result.items():
                    actual = r[1].float()
                    reference = teacher[qi][view]["pointers"]
                    if target_mask is not None:
                        actual, reference = actual[target_mask], reference[target_mask]
                    actual = F.normalize(actual, dim=-1)
                    reference = F.normalize(reference, dim=-1)
                    values.append((1 - (actual * reference).sum(dim=-1)).mean())
                return torch.stack(values).mean()

            def branch_cost(candidates, remove, future, capacity, seed, prefix_lengths=()):
                branch = _new_bank(model, task, anatomy, label_space, targets, capacity,
                                   continuation_policy, random.Random(seed))
                branch.slots = [slot for j, slot in enumerate(candidates) if j != remove]
                if continuation_policy == "none":
                    branch.slots = []
                branch._last_frame = max(slot.frame_index for slot in candidates)
                losses = [[], []]
                snapshots = {}
                def aggregate():
                    counts = torch.tensor([len(values) for values in losses], dtype=torch.long)
                    costs = batch.media.new_zeros(2, dtype=torch.float32)
                    for axis, values in enumerate(losses):
                        if values:
                            costs[axis] = torch.stack(values).mean()
                    return costs, counts

                for step, qi in enumerate(future, 1):
                    result = read(qi, branch.slots)
                    pred, _, _, _, logits = result[canonical_view]
                    mask_valid = batch.mask_valid[case, qi, valid_targets].bool()
                    if requested[0] and mask_valid.any():
                        # One actual annotated query can contain several targets.
                        if target_objective == "task":
                            value = segmentation_loss(pred[mask_valid],
                                batch.masks[case, qi, valid_targets][mask_valid]).float()
                        elif target_objective == "pointer_consistency":
                            value = pointer_cost(result, qi, mask_valid)
                        else:
                            value = F.binary_cross_entropy_with_logits(pred[mask_valid].float(),
                                teacher[qi][canonical_view]["mask_probability"][mask_valid])
                        losses[0].append(value)
                    if cls_available and qi == final_time:
                        if target_objective == "task":
                            value = F.cross_entropy(logits.float(), batch.class_targets[case:case + 1])
                        elif target_objective == "pointer_consistency":
                            value = pointer_cost(result, qi)
                        else:
                            value = F.kl_div(logits.float().log_softmax(dim=-1),
                                teacher[qi][canonical_view]["class_probability"], reduction="batchmean")
                        losses[1].append(value)
                    if not fixed_state and continuation_policy != "none":
                        slot = model._write_views(features(qi), result, int(batch.frame_indices[case, qi]),
                                                  not branch.slots)
                        branch.update(slot, requested)
                    if step in prefix_lengths:
                        snapshots[step] = aggregate()
                return snapshots if prefix_lengths else aggregate()

            for capacity in capacities:
                selected = _state_indices(times, capacity, max_states)
                if not selected:
                    continue
                behavior = _new_bank(model, task, anatomy, label_space, targets, capacity, behavior_policy,
                                     random.Random(_stream_seed(root_seed, "behavior", case, capacity)))
                for index, ti in enumerate(times[:-1]):
                    current_frame = int(batch.frame_indices[case, ti])
                    result = read(ti, behavior.slots)
                    candidate = model._write_views(features(ti), result, current_frame, not behavior.slots)
                    candidates = behavior.slots + [candidate]
                    if ti in selected and len(candidates) > capacity:
                        eligible = [j for j, slot in enumerate(candidates) if not slot.anchor]
                        # Snapshot before looking at any future feature/annotation.
                        descriptors = slot_descriptors(candidates, current_frame).detach().float().cpu().clone()
                        context = (behavior.context.detach().float().cpu().clone() if behavior.context is not None
                                   else torch.tensor([int(task), int(anatomy), label_space,
                                        int(targets[0]) if len(targets) == 1 else -1, capacity], dtype=torch.float32))
                        future = times[index + 1:]
                        windows = [(h, window) for h, window in _future_windows(future, horizons, cls_available)
                                   if ((requested[0] and any(batch.mask_valid[case, qi, valid_targets].any()
                                                           for qi in window))
                                       or (cls_available and final_time in window))]
                        if windows and target_objective != "task":
                            ensure_teacher()
                        prefix_costs = None
                        if reuse_prefix and continuation_policy != "random" and windows:
                            longest = max((w for _, w in windows), key=len)
                            lengths = {len(w) for _, w in windows}
                            seed = _stream_seed(root_seed, "continuation", case, capacity,
                                                current_frame, tuple(longest))
                            prefix_costs = {action: branch_cost(candidates, action, longest, capacity, seed, lengths)
                                            for action in eligible}
                        for requested_horizon, future_window in windows:
                            # Branch generation cannot consume the behavior stream;
                            # all actions within this window share continuation draws.
                            seed = _stream_seed(root_seed, "continuation", case, capacity,
                                                current_frame, tuple(future_window))
                            costs = batch.media.new_full((len(candidates), 2), float("nan"), dtype=torch.float32)
                            query_counts = None
                            for action in eligible:
                                action_cost, count = (prefix_costs[action][len(future_window)] if prefix_costs is not None
                                                      else branch_cost(candidates, action, future_window, capacity, seed))
                                costs[action] = action_cost
                                if query_counts is not None and not torch.equal(query_counts, count):
                                    raise RuntimeError("Deletion branches evaluated different supervision sets")
                                query_counts = count
                            valid_axes = query_counts > 0
                            deltas = torch.zeros_like(costs)
                            centered = costs[eligible] - costs[eligible].mean(dim=0, keepdim=True)
                            deltas[eligible] = centered
                            deltas[:, ~valid_axes.to(deltas.device)] = 0
                            examples.append({"descriptors": descriptors.clone(), "deltas": deltas.cpu(),
                                "valid_tasks": valid_axes, "active_tasks": torch.tensor(requested),
                                "anchor_mask": torch.tensor([s.anchor for s in candidates]),
                                "frame_indices": torch.tensor([s.frame_index for s in candidates]), "context": context,
                                "raw_action_costs": costs.cpu(), "supervised_query_counts": query_counts,
                                "capacity": capacity, "case": case, "state_frame": current_frame,
                                "horizon": len(future_window), "horizon_requested": requested_horizon,
                                "future_frame_indices": [int(batch.frame_indices[case, q]) for q in future_window],
                                "classification_query_frame": int(batch.frame_indices[case, final_time]) if valid_axes[1] else None,
                                "classification_interval": int(batch.frame_indices[case, final_time]) - current_frame if valid_axes[1] else None,
                                "behavior_policy": behavior_policy, "continuation_policy": continuation_policy,
                                "fixed_state": bool(fixed_state), "candidate_bytes": sum(s.nbytes for s in candidates),
                                **({"target_objective": target_objective,
                                    "proxy_teacher": "reader_internal_causal_full_memory"}
                                   if target_objective != "task" else {})})
                    behavior.update(candidate, requested)
                    if ti >= max(selected):
                        break
    finally:
        random.setstate(python_rng)
        model.train(was_training)
    return examples
