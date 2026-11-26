"""Stable task, label-space and segmentation-target IDs for TUM-US.

IDs are serialized in checkpoints. Append new entries; never reorder them.
Dataset names select annotation semantics in the data layer, not model inputs.
"""
from __future__ import annotations

TASKS = ("image_seg", "image_cls", "video_seg", "ceus_seg", "ceus_cls", "video_cls")
TASK_TO_ID = {name: index for index, name in enumerate(TASKS)}
NUM_TASKS = len(TASKS)
ACTIVE_TASKS = TASKS[:5]

ANATOMIES = (
    "unknown", "appendix", "breast", "fetal_head", "heart", "kidney",
    "liver", "ovary", "prostate", "thyroid",
)
ANATOMY_TO_ID = {name: index for index, name in enumerate(ANATOMIES)}
NUM_ANATOMIES = len(ANATOMIES)

LABEL_SPACES = ("malignancy", "appendicitis", "fatty_liver", "cirrhosis", "stage")
LABEL_SPACE_TO_ID = {name: index for index, name in enumerate(LABEL_SPACES)}
LABEL_CLASSES = (
    ("benign", "malignant"),
    ("no_appendicitis", "appendicitis"),
    ("non_fatty_liver", "fatty_liver"),
    ("no_cirrhosis", "cirrhosis"),
    ("zao", "zhong"),
)
NUM_LABEL_SPACES = len(LABEL_SPACES)
NUM_CLASSES = 2

TARGET_NAMES = (
    "foreground", "breast_nodule", "thyroid_nodule", "thyroid_gland",
    "fetal_head", "lv_endo", "lv_epi", "left_atrium", "cardiac_structure",
    "kidney", "kidney_capsule", "prostate", "prostate_capsule", "ovarian_nodule",
    "liver_foreground", "breast_foreground", "thyroid_foreground",
    "prostate_foreground", "heart_foreground", "appendix", "liver", "ovary",
    "kidney_foreground", "fetal_head_foreground",
    "lv_myocardium", "reserved_25", "reserved_26", "reserved_27",
    "reserved_28", "reserved_29", "reserved_30", "reserved_31",
)
TARGETS = TARGET_NAMES
TARGET_TO_ID = {name: index for index, name in enumerate(TARGET_NAMES)}
NUM_TARGETS = len(TARGET_NAMES)
DEFAULT_SINGLE_MASK_POLICY = "center_reference"

_DEFAULT_TARGETS = {
    "breast": "breast_nodule", "thyroid": "thyroid_nodule",
    "fetal_head": "fetal_head", "heart": "cardiac_structure", "kidney": "kidney",
    "prostate": "prostate", "liver": "liver_foreground", "ovary": "ovarian_nodule",
    "appendix": "appendix", "unknown": "foreground",
}


def normalize_label(value: str) -> str:
    value = str(value).split(":", 1)[-1].strip().casefold()
    return "_".join(value.replace("-", " ").replace("_", " ").split())


def default_target_id(anatomy: str) -> int:
    return TARGET_TO_ID[_DEFAULT_TARGETS.get(anatomy, "foreground")]


def target_id(raw_name: str, anatomy: str) -> int:
    name = normalize_label(raw_name)
    if name in TARGET_TO_ID:
        return TARGET_TO_ID[name]
    if name in {"foreground", "unresolved_foreground", "unknown", ""}:
        candidate = f"{anatomy}_foreground"
        return TARGET_TO_ID.get(candidate, TARGET_TO_ID["foreground"])
    if name in {"nodule", "nodules", "lesion", "tumor", "tumour"}:
        return default_target_id(anatomy)
    aliases = {
        "head": "fetal_head", "thyroid": "thyroid_gland", "gland": "thyroid_gland",
        "lv_endo": "lv_endo", "left_ventricle_endocardium": "lv_endo",
        "lv_epi": "lv_epi", "left_ventricle_epicardium": "lv_epi",
        "la": "left_atrium", "left_atrium": "left_atrium", "cardiac": "cardiac_structure",
        "myocardium": "lv_myocardium",
    }
    if name == "capsule":
        return TARGET_TO_ID["kidney_capsule" if anatomy == "kidney" else "prostate_capsule"]
    return TARGET_TO_ID.get(aliases.get(name, ""), default_target_id(anatomy))


def label_space_id(record: dict) -> int:
    """Resolve a record's local classification space without conflating 0/1 labels."""
    labels = record.get("classification_labels") or ()
    names = {normalize_label(value) for value in labels}
    dataset_name = str(record.get("dataset_name", ""))
    if names & {"no_appendicitis", "appendicitis"} or dataset_name == "Appendix":
        return LABEL_SPACE_TO_ID["appendicitis"]
    if names & {"non_fatty_liver", "fatty_liver"} or dataset_name == "Fatty-Liver":
        return LABEL_SPACE_TO_ID["fatty_liver"]
    if names & {"no_cirrhosis", "cirrhosis"} or dataset_name == "LiverCirrhosisYang":
        return LABEL_SPACE_TO_ID["cirrhosis"]
    if names & {"zao", "zhong"} or dataset_name == "JH_HC":
        return LABEL_SPACE_TO_ID["stage"]
    return LABEL_SPACE_TO_ID["malignancy"]


def classification_label(record: dict) -> tuple[int, bool, int]:
    space = label_space_id(record)
    labels = record.get("classification_labels") or ()
    if record.get("classification_label_status", "complete") == "conflict" or len(labels) != 1:
        return -100, False, space
    normalized = normalize_label(labels[0])
    if normalized in LABEL_CLASSES[space]:
        return LABEL_CLASSES[space].index(normalized), True, space
    # A numeric-only label remains local to its registered two-class space.
    prefix = str(labels[0]).split(":", 1)[0]
    if normalized in {"0", "1"} and prefix in {"0", "1"}:
        return int(prefix), True, space
    return -100, False, space
