# Runtime sources

- Optional object-state bridge: query propagation inspired by [VidEoMT](https://github.com/tue-mps/videomt/tree/f351ccbfb2096c71bf3bce6241c8eb257c5cffeb), and foreground/background object abstractions inspired by [Cutie](https://github.com/hkchengrex/Cutie/tree/ec5cdd4cf16f75c73ad785a2f96fb97dbad4125a). The implementation is a small SAM2 decoder-side adaptation, not a reproduction of either complete architecture. Upstream MIT notices are retained in model/vendor/VidEoMT_LICENSE and model/vendor/Cutie_LICENSE. See config/docs/对象状态适配说明.md.

- Cardiac data protocols are adapted from [MemSAM](https://github.com/dengxl0520/MemSAM/tree/36f1b988f14b518a6346d2edb1267340387e646c), pinned commit `36f1b988f14b518a6346d2edb1267340387e646c`: utils/preprocess_camus.py, utils/preprocess_echonet.py and utils/tum_us/data/cardiac_preprocess.py. Copyright (c) 2024 DengXL; MIT license retained at model/vendor/MemSAM_LICENSE. TUM-US adds explicit safe numeric annotation indices, geometry/provenance records, unique-frame sampling and separate dense/sparse supervision. See config/docs/CAMUS_EchoNet处理说明.md for exact differences and raw-data commands.

- SAM2.1 Small: download the official checkpoint separately. The bundled SAM2 code has its license under model/vendor/SAM2_LICENSE. Compact controller checkpoints validate their parent checkpoint SHA256 and the complete frozen-reader tensor identity.
- generalist modules implement contextual task utility, unpaired observed-axis supervision, future read/write rollout and controller adaptation. transfer modules retain previous data/metric helpers and the earlier optional gain-guided transfer implementation; they do not define the current main research pipeline.

# v0.5 training references (independently implemented adaptation)

- ConsMTL / MTLlib: https://github.com/jianke0604/MTLlib , inspected commit `e64779272aade3b89e8ad5437ec1e33d1b3bdf5e`, `methods/weight_methods.py`. Our parameter-coordinate cosine probe with PCGrad is not its paired-representation least-squares algorithm.
- ODS: https://github.com/Juho-Jung/multitask_active_learning_official , inspected commit `86667800a0fff48b68fc673c0c023fb3c194b447`. Selection and optimization separation only; no unpaired CTMI or annotation-budget claim.
- RADIO: https://github.com/NVlabs/RADIO , inspected commit `4ffaf15d0ff48197c194e306772bdb5fb92db230`, `examples/common/phi_s.py`. Work2 design only; no RADIO code or weights included.

