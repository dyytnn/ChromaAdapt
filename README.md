# ChromaBridge

Code and model release for **Synthetic Pretraining and Adversarial Domain
Adaptation for Overlapping Chromosome Pair Segmentation in Metaphase Images**.

ChromaBridge uses a dual-head MASS-Net to predict two chromosome instances and
their overlap region. The repository includes the training and evaluation source
code together with four CPU-ready ONNX models corresponding to the paper's SuSy,
SuRe, SuSy-to-SuRe, and SuSy-to-SeDANN regimes.

## Public release contents

This repository contains:

- model, training, domain-adaptation, evaluation, and data-loader source code;
- standalone karyotype preprocessing and synthetic-pair generation code;
- architecture and pipeline configuration files;
- four exported ONNX segmentation models;
- a standalone ONNX Runtime inference script.

It intentionally does **not** contain patient images, masks, generated datasets,
original PyTorch checkpoints, training logs, or internal experiment artifacts.
The `data/` directory contains Python dataset-loader code only.

## Quick ONNX inference

No PyTorch installation or dataset is needed for the smoke test:

```bash
cd onnx_models
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt

python infer_onnx.py \
  --model massnet_susy_to_sure.onnx \
  --smoke-test \
  --output-dir outputs \
  --benchmark-runs 20
```

To process a local chromosome-pair crop or a directory of crops, replace
`--smoke-test` with `--input /path/to/image_or_directory`.

See [onnx_models/README.md](onnx_models/README.md) for model I/O,
preprocessing, postprocessing, validation results, and checksums.

## Included ONNX models

| Paper regime | File |
|---|---|
| SuSy | `onnx_models/massnet_susy.onnx` |
| SuRe | `onnx_models/massnet_sure.onnx` |
| SuSy to SuRe | `onnx_models/massnet_susy_to_sure.onnx` |
| SuSy to SeDANN | `onnx_models/massnet_susy_to_sedann.onnx` |

The SeDANN export contains only the segmentation network required at inference;
the training-only domain classifier and gradient-reversal path are excluded.

## Source layout

```text
.
├── abs_interface/      # Pipeline and model interfaces
├── configs/            # Architecture, training, evaluation, and inference YAML
├── data/               # Dataset-loader source code (no dataset files)
├── generation/         # Extraction, binarization, gap/hole repair, and synthesis
├── manager/            # Pipeline orchestration
├── onnx_models/        # Public ONNX weights and standalone CPU inference
├── registry/           # Builders, training, evaluation, DANN, and KoLeo code
├── source/             # MASS-Net modules and losses
├── utils/              # Export, distributed-training, and visualization tools
├── main.py             # PyTorch training entry point
└── evaluation.py       # PyTorch evaluation/inference entry point
```

## Training and evaluation data

Datasets are not redistributed because the real metaphase images are subject to
their original access and privacy conditions. Users must supply their own data
paths in the YAML files under `configs/`.

See [DATA_AVAILABILITY.md](DATA_AVAILABILITY.md) for the public-release data
statement.

Expected labeled real-data layout:

```text
dataset_root/
├── images/
│   └── example.png
└── masks/
    └── example.png
```

Expected synthetic-pair layout:

```text
dataset_root/
├── images/
│   ├── OVERLAP/
│   └── TOUCHING/
└── masks/
    ├── OVERLAP/
    └── TOUCHING/
```

No real or synthetic image from the study is committed to this repository. The
source needed to recreate a compatible synthetic dataset from an authorized
karyotype collection is documented in
[generation/README.md](generation/README.md).

## PyTorch entry points

After installing the training dependencies and replacing the generic path
placeholders in a selected YAML file:

```bash
python main.py \
  --pipeline_config configs/train_pipeline_config/segment_synthetic_data.yaml

python evaluation.py \
  --pipeline_config configs/eval_pipeline_config/evaluation/instance_segmentation_multihead_real.yaml
```

The supplied supervised-training configurations write generated artifacts under
`outputs/`, which is ignored by Git.

## Citation

The final journal citation and DOI will be added after publication.
