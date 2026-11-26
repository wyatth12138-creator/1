# Standalone five-task data interface

`build_dataset` reads the canonical JSONL manifest (CSV and JSON record lists
are also supported). Paths resolve relative to `data_root`; the legacy
GeneralTMI package is never imported. All selected records are kept in
`dataset.records`, grouped indices in `dataset.groups[(task, dataset_id)]`.

- Image input becomes `T=1,V=1`; B-mode video becomes `T>=1,V=1`.
- CEUS keeps RGB and becomes `V=2` in the fixed order B-mode, CEUS. Source
  view identity comes from explicit manifest configuration or input RGB
  chromaticity, with a source/anatomy convention fallback. Ground-truth
  masks do not determine the model's input ordering.
- Mask pixels map through the manifest's `segmentation_raw_values` and
  `segmentation_targets` into stable registry target IDs. CAMUS retains three
  targets. Unnamed source annotations retain a foreground target.
- `fnum_mask` and explicit `frame_indices`/`frame_index` identify supervised
  source frames. Equal mask/video sequence lengths use direct frame indices.
- `single_mask_policy='center_reference'` is the prototype's explicit input
  convention for an unindexed single mask: supervise only source frame
  `T//2`. It is **not** a verified source frame annotation. The metadata
  marks `mask_frame_index_is_convention=True`. `first_reference`, `unlabeled`
  and `error` are alternative policies. Explicit indices always take priority.
- If unequal-length source arrays repeat exactly the same mask, those copies
  collapse to one reference before applying the single-mask policy. The
  source count and collapse operation are recorded in metadata.
- Segmentation samples can inherit a consistent same-entity classification
  label from the requested split. This is case-level label sharing, not
  temporal registration or alignment of two source clips.
- Classification samples carry one default anatomical query target and no
  valid segmentation labels. `mask_valid` and `class_valid` determine loss
  participation. Batch padding uses target/frame ID `-1` and invalid masks.

Registry IDs are checkpoint-stable. Five active tasks occupy IDs 0–4;
`video_cls=5` is reserved. There are five independent binary label spaces
and 32 segmentation target/query IDs.

Configure data paths and endpoint filters in the JSON configuration. See the project README for the dataset preparation and evaluation commands.
