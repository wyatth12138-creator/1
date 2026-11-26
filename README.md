# TUM-US

TUM-US is a joint ultrasound model for still-image classification and segmentation, B-mode video segmentation, and dual-view contrast-enhanced ultrasound (CEUS) classification and segmentation.

The V2 reader uses a shared SAM 2.1 Hiera Small encoder at 256 × 256 pixels per view. Training-set support prototypes reweight the top-level features. The SAM mask decoder produces segmentation masks and foreground/background evidence for classification. Video windows use four memory slots, with the first valid frame kept as an anchor and the other slots updated by FIFO. B-mode and CEUS views share the same retained frame indices.

## Installation

Use Python 3.10+ and install a PyTorch build appropriate for the machine. Run the following from the repository root.

```sh
python -m pip install -e ".[preprocess,evaluation]"
python train.py --help
python test.py --help
```

Download `sam2.1_hiera_small.pt` from the [official SAM 2 checkpoint list](https://github.com/facebookresearch/sam2#download-checkpoints) and save it at `pre_checkpoints/sam2.1_hiera_small.pt`. Alternatively, set `model.checkpoint` in the configuration to a local file. Training checkpoints, data, and experiment outputs are kept locally.

## V2 reader

The [V2 template](config/v2_readers.json) describes 37 endpoint definitions and the support-prototype settings. Supply an inventory for the data you are authorized to use. Paths in the inventory are relative to `--data-root`. Video segmentation requires a verified frame map linking stored frames to annotated masks. See the [data interface](utils/tum_us/data/README.md) and [CAMUS and EchoNet instructions](config/docs/CAMUS_EchoNet处理说明.md).

```sh
python utils/prepare.py --config config/v2_readers.json --data-root data --inventory data/inventory.jsonl --frame-map data/frame_map.json --output run/v2_inputs
python train.py --config run/v2_inputs/prepared_config.json --stage readers --output run/v2_readers
python test.py --config run/v2_inputs/prepared_config.json --checkpoint run/v2_readers/readers.pt --output run/v2_evaluation
```

Check `readiness_report.json` before training. To use a different set of endpoints, update `endpoints`, `endpoint_sets`, and the keys in `model.support_profiles` together. The latter describe combinations of task, anatomy, label space, and target IDs. Every retained classification class or segmentation profile needs enough training cases to initialize its prototypes.

The commands above evaluate the development split. For a held-out test, prepare a fixed inventory with `--include-test`, generate the final manifests, and set `allow_final_evaluation` to `true` and `evaluation.split` to `"test"` in the scoring configuration. The public template describes the model and inputs; exact internal results require the original locked case splits and research checkpoints.

## Other study entry points

- `config/study.json` configures the task-utility controller study. Prepare inputs with the study configuration and use the stages listed by `train.py --help`.
- `config/object_state.json` configures the object-state comparison. See the [object-state instructions](config/docs/对象状态适配说明.md).
- `config/consistent_mtl.json` configures optional multitask optimization. See the [multitask instructions](config/docs/通用训练优化.md).
- `config/support_prompt.json` configures the optional episodic prompt study. Set `reference_config` to your prepared configuration (relative to the prompt config file or absolute) and update `heldout` and `evaluate` for its endpoints.

Third-party provenance and notices are listed in [SOURCES.md](SOURCES.md) and `model/vendor/`.
