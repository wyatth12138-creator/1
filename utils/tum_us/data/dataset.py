"""Manifest-driven adapters for the five ultrasound tasks.

No imports from GeneralTMI are required. A row is a case/task group; the
selected source item and exact sampled frame indices are retained in metadata.
"""
from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict, deque
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset

from .registry import (
    ACTIVE_TASKS, ANATOMY_TO_ID, DEFAULT_SINGLE_MASK_POLICY, TASK_TO_ID,
    classification_label, default_target_id, target_id,
)
from .schema import Sample


def _as_set(value):
    if value is None or value == "all":
        return set()
    return {value} if isinstance(value, str) else set(value)


def _read_records(path: Path) -> list[dict]:
    if path.suffix.casefold() == ".csv":
        with path.open(encoding="utf-8-sig", newline="") as stream:
            result = list(csv.DictReader(stream))
        nested = {"sample_paths", "annotation_paths", "temporal_lengths", "segmentation_targets",
                  "segmentation_raw_values", "classification_labels", "linked_record_ids",
                  "entity_available_tasks", "modality_components"}
        for record in result:
            for key in nested.intersection(record):
                record[key] = json.loads(record[key]) if record[key] else ([] if key != "segmentation_raw_values" else {})
        return result
    with path.open(encoding="utf-8-sig") as stream:
        if path.suffix.casefold() == ".json":
            contents = json.load(stream)
            return contents if isinstance(contents, list) else contents["records"]
        return [json.loads(line) for line in stream if line.strip()]


def _balanced_prefix(records: list[dict], maximum: int | None) -> list[dict]:
    """For smoke limits, round-robin tasks and then datasets within each task."""
    if maximum is None or maximum >= len(records):
        return records
    if maximum < 1:
        raise ValueError("max_records must be positive")
    grouped: dict[str, dict[str, deque]] = defaultdict(lambda: defaultdict(deque))
    for record in records:
        grouped[record["task"]][record["dataset_id"]].append(record)
    tasks = deque(task for task in ACTIVE_TASKS if task in grouped)
    datasets = {task: deque(sorted(grouped[task])) for task in tasks}
    chosen = []
    while tasks and len(chosen) < maximum:
        task = tasks.popleft()
        names = datasets[task]
        name = names.popleft()
        chosen.append(grouped[task][name].popleft())
        if grouped[task][name]:
            names.append(name)
        if names:
            tasks.append(task)
    return chosen


def select_frames(total: int, length: int | None, required: Iterable[int] = ()) -> list[int]:
    if total < 1:
        raise ValueError("A source video has no frames")
    if not length or length >= total:
        return list(range(total))
    if length < 1:
        raise ValueError("clip_length must be positive or None")
    mandatory = sorted({int(index) for index in required if 0 <= int(index) < total})
    if len(mandatory) > length:
        positions = np.linspace(0, len(mandatory) - 1, length).round().astype(int)
        return [mandatory[index] for index in positions]
    chosen = set(mandatory)
    for index in np.linspace(0, total - 1, length).round().astype(int):
        if len(chosen) >= length:
            break
        chosen.add(int(index))
    for index in range(total):
        if len(chosen) >= length:
            break
        chosen.add(index)
    return sorted(chosen)


def _media_tensor(value: np.ndarray, size: tuple[int, int]) -> torch.Tensor:
    """Input is T,V,C,H,W. Preserve RGB, including CEUS palettes."""
    array = np.array(value, dtype=np.float32, copy=True)
    if array.shape[2] == 1:
        array = np.repeat(array, 3, axis=2)
    if array.shape[2] == 4:
        array = array[:, :, :3]
    if array.shape[2] != 3:
        raise ValueError("Media requires one, three or four channels")
    tensor = torch.from_numpy(array)
    if not torch.isfinite(tensor).all():
        raise ValueError("Media contains non-finite values")
    if tensor.min() < 0:
        raise ValueError("Media must contain nonnegative raw intensities")
    if tensor.max() > 1:
        tensor = tensor / 255.0
    tensor = tensor.clamp(0, 1)
    t, v, c, h, w = tensor.shape
    tensor = F.interpolate(tensor.reshape(t * v, c, h, w), size=size, mode="bilinear", align_corners=False)
    return tensor.reshape(t, v, c, *size)


def _video_array(path: Path, ceus: bool) -> np.ndarray:
    if path.suffix.casefold() != ".npy":
        raise ValueError(f"Video adapter expects a NumPy .npy source, received {path.suffix}")
    value = np.load(path, allow_pickle=False, mmap_mode="r")
    if ceus:
        if value.ndim == 3:
            return value[..., None]
        if value.ndim == 4 and value.shape[-1] in (1, 3, 4):
            return value  # T,H,W,C
        raise ValueError(f"CEUS video must be T,H,W,C; shape={value.shape}")
    if value.ndim == 3:
        return value[:, None]  # T,1,H,W
    if value.ndim == 4 and value.shape[0] in (1, 3, 4):
        return value.transpose(1, 0, 2, 3)  # C,T,H,W -> T,C,H,W
    if value.ndim == 4 and value.shape[1] in (1, 3, 4):
        return value
    if value.ndim == 4 and value.shape[-1] in (1, 3, 4):
        return value.transpose(0, 3, 1, 2)
    raise ValueError(f"Unsupported B-mode video shape: {value.shape}")


def _bmode_source_view(video: np.ndarray, record: dict) -> tuple[int, str]:
    """Determine input semantics using input content/config only, never masks."""
    for key in ("source_bmode_view_index", "bmode_view_index", "bmode_view"):
        value = record.get(key)
        if value is not None and str(value) in {"0", "1"}:
            return int(value), f"manifest:{key}"
    if video.shape[-1] >= 3:
        # Streaming order must be determined before future frames arrive.
        # Explicit source metadata is preferred; the fallback sees frame 0 only.
        frames = np.asarray(video[:1], dtype=np.float32)
        height, width = frames.shape[1:3]
        half = width // 2
        scores = []
        for side in (0, 1):
            x0 = side * half + max(1, round(half * 0.04))
            x1 = (side + 1) * half - max(1, round(half * 0.02))
            panel = frames[:, round(height * 0.1):max(round(height * 0.1) + 1, round(height * 0.96)), x0:x1, :3]
            scores.append(float((np.abs(panel[..., 0] - panel[..., 1]) + np.abs(panel[..., 1] - panel[..., 2]) + np.abs(panel[..., 0] - panel[..., 2])).mean()))
        if abs(scores[0] - scores[1]) > 1e-5:
            return int(scores[0] > scores[1]), "first_stored_frame_rgb_chromaticity"
    # The source convention is a last-resort input-only rule for achromatic panels.
    return (1 if record.get("anatomy") == "prostate" else 0), "source_anatomy_convention"


def _ceus_media(video: np.ndarray, selected: list[int], bmode_side: int, size) -> torch.Tensor:
    frames = np.asarray(video[selected])
    width = frames.shape[2]
    half = width // 2
    left = frames[:, :, :half]
    right = frames[:, :, width - half:]
    panels = (left, right)
    views = np.stack((panels[bmode_side], panels[1 - bmode_side]), axis=1)
    return _media_tensor(views.transpose(0, 1, 4, 2, 3), size)


def _frame_array(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value)
    if value.ndim == 3 and value.shape[-1] in (3, 4):
        value = value[..., :3].max(axis=-1)
    if value.ndim != 2:
        raise ValueError(f"An individual mask must be H,W, received {value.shape}")
    return value


def _mask_sequence(value: np.ndarray) -> list[np.ndarray]:
    value = np.asarray(value)
    if value.ndim == 2:
        return [value]
    if value.ndim == 4 and value.shape[1] == 1:
        value = value[:, 0]
    if value.ndim == 3:
        return [np.asarray(item) for item in value]
    raise ValueError(f"Unsupported mask sequence shape: {value.shape}")


def _explicit_indices(archive: dict, record: dict, count: int, sample_index: int) -> list[int] | None:
    for source in (archive, record):
        for key in ("frame_indices", "annotated_frame_indices", "mask_frame_indices", "frame_index", "annotated_frame_index", "mask_frame_index", "fnum", "fnums"):
            if key not in source or source[key] is None:
                continue
            value = source[key]
            if source is record and isinstance(value, dict):
                value = value.get(str(sample_index), value.get(sample_index))
                if value is None:
                    continue
            try:
                numbers = np.asarray(value).reshape(-1)
                if len(numbers) != count:
                    continue
                values = [int(index) for index in numbers]
            except (TypeError, ValueError):
                continue
            base = int(source.get("frame_index_base", 0))
            return [index - base for index in values]
    return None


def _load_masks(path: Path, total: int, record: dict, sample_index: int, policy: str) -> tuple[dict[int, np.ndarray], dict]:
    suffix = path.suffix.casefold()
    if suffix not in {".npz", ".npy"}:
        with Image.open(path) as image:
            raw = np.asarray(image).copy()
        return {0: _frame_array(raw)}, {"mask_frame_mapping": "static_image", "annotation_source_frames": [0]}
    if suffix == ".npy":
        archive = {"mask": np.load(path, allow_pickle=False)}
    else:
        # Official local cardiac annotations are object dictionaries. Only paths
        # resolved within the caller's explicitly selected data root reach here.
        with np.load(path, allow_pickle=True) as source:
            archive = {key: source[key] for key in source.files if key in {
                "fnum_mask", "mask", "masks", "label", "labels", "arr_0",
                "frame_indices", "annotated_frame_indices", "mask_frame_indices", "frame_index",
                "annotated_frame_index", "mask_frame_index", "fnum", "fnums", "frame_index_base",
            }}
    if "fnum_mask" in archive:
        raw = archive["fnum_mask"]
        value = raw.item() if raw.shape == () else raw.tolist()
        if not isinstance(value, dict):
            raise TypeError("fnum_mask must be an indexed dictionary")
        masks = {int(index): _frame_array(mask) for index, mask in value.items()}
        if any(index < 0 or index >= total for index in masks):
            raise ValueError("A fnum_mask index is outside the source video")
        return masks, {"mask_frame_mapping": "explicit_fnum_mask", "annotation_source_frames": sorted(masks)}
    key = next((name for name in ("mask", "masks", "label", "labels", "arr_0") if name in archive), None)
    if key is None:
        raise KeyError("Annotation contains no recognized mask array")
    sequence = _mask_sequence(archive[key])
    count = len(sequence)
    explicit = _explicit_indices(archive, record, count, sample_index)
    original_count = count
    duplicate_reference = False
    if explicit is None and count != total and count > 1 and all(
        np.array_equal(sequence[0], item) for item in sequence[1:]
    ):
        # Some source containers repeat a single reference mask. They are one
        # reference, rather than independent annotations for every video frame.
        sequence, count, duplicate_reference = sequence[:1], 1, True
        explicit = _explicit_indices(archive, record, count, sample_index)
    convention = False
    if explicit is not None:
        indices, mapping = explicit, "explicit_frame_indices"
    elif count == total:
        indices, mapping = list(range(total)), "dense_same_length"
    elif count == 1 and policy == "center_reference":
        indices, mapping, convention = [total // 2], "single_reference_center_convention", True
    elif count == 1 and policy == "first_reference":
        indices, mapping, convention = [0], "single_reference_first_convention", True
    elif policy == "unlabeled":
        return {}, {"mask_frame_mapping": "unindexed_annotation", "annotation_source_frames": [], "unindexed_mask_count": count}
    else:
        raise ValueError("Mask/video lengths differ without explicit frame indices; configure a single-mask policy or provide indices")
    if len(set(indices)) != len(indices) or any(index < 0 or index >= total for index in indices):
        raise ValueError("Annotation indices are duplicated or outside the source video")
    return dict(zip(indices, map(_frame_array, sequence))), {
        "mask_frame_mapping": mapping, "annotation_source_frames": indices,
        "mask_frame_index_is_convention": convention, "single_mask_policy": policy,
        "source_mask_count": original_count,
        "duplicate_reference_collapsed": duplicate_reference,
    }


def _target_definitions(record: dict) -> tuple[list[str], list[int], list[int], dict]:
    names = list(record.get("segmentation_targets") or ("foreground",))
    anatomy = str(record.get("anatomy", "unknown"))
    local_ids = []
    for index, name in enumerate(names, 1):
        prefix = str(name).split(":", 1)[0]
        local_ids.append(int(prefix) if prefix.isdigit() else index)
    targets = [target_id(name, anatomy) for name in names]
    return names, local_ids, targets, record.get("segmentation_raw_values") or {}


def _mask_channels(mask: np.ndarray, local_ids: list[int], raw_values: dict, size) -> torch.Tensor:
    unique = set(np.unique(mask).tolist())
    channels = []
    for local in local_ids:
        raw = raw_values.get(str(local))
        if raw is not None and int(raw) in unique:
            channel = mask == int(raw)
        elif local in unique:
            channel = mask == local
        elif len(local_ids) == 1:
            channel = mask > 0
        else:
            channel = np.zeros_like(mask, dtype=bool)
        channels.append(torch.from_numpy(np.array(channel, dtype=np.float32, copy=True)))
    tensor = torch.stack(channels)[:, None]
    return F.interpolate(tensor, size=size, mode="nearest")[:, 0]


class UltrasoundDataset(Dataset):
    def __init__(self, data_root, manifest_path, split="train", tasks=None, privacy="all", image_size=256,
                 clip_length=8, dataset_ids=None, max_records=None, single_mask_policy=DEFAULT_SINGLE_MASK_POLICY):
        self.data_root = Path(data_root).expanduser().resolve()
        self.manifest_path = Path(manifest_path).expanduser().resolve()
        self.image_size = (image_size, image_size) if isinstance(image_size, int) else tuple(image_size)
        self.clip_length = clip_length
        self.single_mask_policy = single_mask_policy
        if single_mask_policy not in {"center_reference", "first_reference", "unlabeled", "error"}:
            raise ValueError("Unsupported single_mask_policy")
        if len(self.image_size) != 2 or min(self.image_size) < 1:
            raise ValueError("image_size must be positive")
        if clip_length is not None and clip_length < 1:
            raise ValueError("clip_length must be positive or None")
        if not self.data_root.is_dir():
            raise NotADirectoryError(self.data_root)
        task_filter = _as_set(tasks)
        if task_filter.difference(ACTIVE_TASKS):
            raise ValueError("Only the five active tasks are supported; video_cls is reserved")
        split_filter, dataset_filter = _as_set(split), _as_set(dataset_ids)
        privacy_filter = _as_set(privacy)
        if privacy_filter.difference({"public", "private"}):
            raise ValueError("privacy must be all, public or private")
        source_records = _read_records(self.manifest_path)
        # Link labels at the case level and within the requested split. Never
        # infer temporal alignment from a case-level link.
        label_sets: dict[str, set] = defaultdict(set)
        for record in source_records:
            if split_filter and record["split"] not in split_filter:
                continue
            if privacy_filter and record["privacy"] not in privacy_filter:
                continue
            value, valid, space = classification_label(record)
            if valid:
                label_sets[record["entity_uid"]].add((value, space))
        self._linked_labels = {key: next(iter(values)) for key, values in label_sets.items() if len(values) == 1}
        chosen = []
        seen = set()
        for record in source_records:
            if record.get("task") not in ACTIVE_TASKS:
                continue
            if split_filter and record["split"] not in split_filter:
                continue
            if privacy_filter and record["privacy"] not in privacy_filter:
                continue
            if task_filter and record["task"] not in task_filter:
                continue
            if dataset_filter and record["dataset_id"] not in dataset_filter:
                continue
            if record["record_id"] in seen:
                raise ValueError("Manifest contains duplicate selected record_id")
            seen.add(record["record_id"])
            if not record.get("sample_paths"):
                raise ValueError("A selected manifest record has no sample_paths")
            chosen.append(record)
        self.records = _balanced_prefix(chosen, max_records)
        if not self.records:
            raise ValueError("Manifest filters selected no records")
        groups = defaultdict(list)
        for index, record in enumerate(self.records):
            groups[(record["task"], record["dataset_id"])].append(index)
        self.groups = dict(groups)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self):
        return len(self.records)

    def _resolve(self, relative: str) -> Path:
        path = (self.data_root / str(relative).replace("\\", "/")).resolve()
        if path != self.data_root and self.data_root not in path.parents:
            raise ValueError("Manifest path escapes the configured data_root")
        return path

    def _sample_index(self, record: dict) -> int:
        count = len(record["sample_paths"])
        if count == 1 or record["split"] != "train":
            return 0
        token = f"2026:{self.epoch}:{record['record_id']}".encode("utf-8")
        return int.from_bytes(hashlib.blake2b(token, digest_size=8).digest(), "little") % count

    def __getitem__(self, index: int) -> Sample:
        record = self.records[index]
        task = record["task"]
        anatomy = str(record.get("anatomy", "unknown"))
        source_index = self._sample_index(record)
        source_relative = record["sample_paths"][source_index]
        path = self._resolve(source_relative)
        segmentation = task.endswith("seg")
        ceus = task.startswith("ceus")
        masks_by_frame: dict[int, np.ndarray] = {}
        mask_metadata = {}
        seg_view = 0
        if task.startswith("image"):
            with Image.open(path) as image:
                raw = np.asarray(image.convert("RGB")).copy()
            total, selected = 1, [0]
            media = _media_tensor(raw.transpose(2, 0, 1)[None, None], self.image_size)
        else:
            video = _video_array(path, ceus)
            total = len(video)
            if ceus:
                bmode_side, view_rule = _bmode_source_view(video, record)
        if segmentation:
            annotation_paths = record.get("annotation_paths") or ()
            if source_index >= len(annotation_paths):
                raise ValueError("Selected segmentation sample lacks its paired annotation path")
            annotation_path = self._resolve(annotation_paths[source_index])
            masks_by_frame, mask_metadata = _load_masks(annotation_path, total, record, source_index, self.single_mask_policy)
        if not task.startswith("image"):
            selected = select_frames(total, self.clip_length, masks_by_frame)
            if ceus:
                media = _ceus_media(video, selected, bmode_side, self.image_size)
            else:
                media = _media_tensor(np.asarray(video[selected])[:, None], self.image_size)
        if segmentation:
            names, local_ids, targets, raw_values = _target_definitions(record)
        else:
            names, local_ids, targets, raw_values = [], [1], [default_target_id(anatomy)], {}
        q = len(targets)
        mask_tensor = torch.zeros((len(selected), q, *self.image_size), dtype=torch.float32)
        valid = torch.zeros((len(selected), q), dtype=torch.bool)
        if ceus and masks_by_frame:
            first_mask = next(iter(masks_by_frame.values()))
            source_width = video.shape[2]
            half = source_width // 2
            if first_mask.shape[1] == source_width:
                left_energy = sum(np.count_nonzero(value[:, :half]) for value in masks_by_frame.values())
                right_energy = sum(np.count_nonzero(value[:, source_width - half:]) for value in masks_by_frame.values())
                mask_side = int(right_energy > left_energy) if left_energy or right_energy else bmode_side
                # Annotation selects the supervised output view only. It never
                # changes the input's B-mode/CEUS order.
                seg_view = 0 if mask_side == bmode_side else 1
                masks_by_frame = {frame: value[:, :half] if mask_side == 0 else value[:, source_width - half:]
                                  for frame, value in masks_by_frame.items()}
                mask_metadata["annotation_source_view_index"] = mask_side
            else:
                seg_view = int(record.get("seg_view_id", record.get("seg_view_index", 0)))
        for out_index, frame in enumerate(selected):
            if frame in masks_by_frame:
                mask_tensor[out_index] = _mask_channels(masks_by_frame[frame], local_ids, raw_values, self.image_size)
                valid[out_index] = True
        class_value, class_valid, space = classification_label(record)
        class_source = "record"
        if segmentation and not class_valid and record["entity_uid"] in self._linked_labels:
            class_value, space = self._linked_labels[record["entity_uid"]]
            class_valid, class_source = True, "linked_case_label"
        metadata = {
            "entity_uid": record["entity_uid"], "record_id": record["record_id"],
            "dataset_id": record["dataset_id"], "dataset_name": record.get("dataset_name"),
            "task": task, "privacy": record["privacy"], "split": record["split"],
            "case_id": record.get("case_id"), "anatomy": anatomy,
            "sample_path": source_relative, "sample_index": source_index,
            "source_frame_indices": selected, "source_frame_count": total,
            "classification_label_source": class_source if class_valid else "unavailable",
            "classification_labels": list(record.get("classification_labels") or ()),
            "target_names": names, "annotation_granularity": record.get("annotation_granularity"),
            "seg_view_id": seg_view, **mask_metadata,
        }
        if ceus:
            metadata.update({"view_order": ["bmode", "ceus"], "source_bmode_view_index": bmode_side,
                             "view_order_rule": view_rule, "color_policy": "preserve_rgb"})
        sample = Sample(
            media=media, frame_valid=torch.ones(len(selected), dtype=torch.bool),
            view_valid=torch.ones(media.shape[1], dtype=torch.bool), masks=mask_tensor, mask_valid=valid,
            target_ids=torch.tensor(targets, dtype=torch.long), class_targets=torch.tensor(class_value, dtype=torch.long),
            class_valid=torch.tensor(class_valid, dtype=torch.bool), label_space_ids=torch.tensor(space, dtype=torch.long),
            task_ids=torch.tensor(TASK_TO_ID[task], dtype=torch.long),
            anatomy_ids=torch.tensor(ANATOMY_TO_ID.get(anatomy, 0), dtype=torch.long),
            seg_view_ids=torch.tensor(seg_view, dtype=torch.long),
            frame_indices=torch.tensor(selected, dtype=torch.long), metadata=metadata,
        )
        sample.validate()
        return sample


def build_dataset(data_root, manifest_path, split="train", tasks=None, privacy="all", image_size=256,
                  clip_length=8, dataset_ids=None, max_records=None,
                  single_mask_policy=DEFAULT_SINGLE_MASK_POLICY) -> UltrasoundDataset:
    return UltrasoundDataset(data_root, manifest_path, split, tasks, privacy, image_size,
                             clip_length, dataset_ids, max_records, single_mask_policy)
