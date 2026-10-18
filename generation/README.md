# Data-generation pipeline

This directory contains the standalone preprocessing and synthetic-pair code
used by the paper. It contains source code only: no karyotype image, chromosome
crop, mask, or generated dataset is included.

## 1. Extract single chromosomes

Install the lightweight generation dependencies from the repository root:

```bash
python -m pip install -r generation/requirements.txt
```

Extract a single karyotype image or all supported images below a directory:

```bash
python -m generation.preprocessing \
  --input /path/to/karyotype_images \
  --output outputs/chromosome_pool
```

The defaults reproduce the public paper pipeline: global dark-pixel threshold
225, eight-pixel bounding-box padding, four-row spatial ordering, weak-bridge
splitting, sudden component-area filtering, and gap/hole repair. Every input is
written to its own subdirectory:

```text
outputs/chromosome_pool/
├── images/<karyotype_id>/chromosome_category_<n>.png
└── masks/<karyotype_id>/chromosome_category_<n>.png
```

Masks are binary PNGs whose pixel values are 0 or 1.

## 2. Generate synthetic pairs

Generate the 10,000 pairs used by the paper protocol:

```bash
python -m generation.synthetic_pairs \
  --input outputs/chromosome_pool \
  --output outputs/synthetic_pairs \
  --count 10000 \
  --sampling same-karyotype \
  --mode both \
  --touching-probability 0.8 \
  --minimum-overlap 0.2 \
  --maximum-overlap 0.5 \
  --feather-size 5 \
  --seed 42
```

`same-karyotype` sampling keeps acquisition and staining characteristics
consistent within a pair. The default 0.8 touching probability produces an
exact 4:1 touching-to-overlap quota (subject only to integer rounding) used by
the production configuration. Use `random` or `same-class` only for a different
experimental protocol.

The generator independently rotates the two instances, rejects insufficiently
distinct orientations, translates them into either exact contact or a controlled
overlap, and uses distance-transform blending with Gaussian feathering. It
writes the layout consumed by `Instance_segmentation_touch_overlap`:

```text
outputs/synthetic_pairs/
├── images/
│   ├── OVERLAP/*.png
│   └── TOUCHING/*.png
├── masks/
│   ├── OVERLAP/*.png
│   └── TOUCHING/*.png
└── manifest.jsonl
```

Mask channel 0 is unused, channel 1 is chromosome 1, and channel 2 is
chromosome 2. The overlap target is their logical intersection. The manifest
records relative source identifiers, classes, rotations, and measured overlap;
it never stores an absolute local path.

Generated outputs belong under `outputs/`, which is ignored by Git. Do not add
locally generated crops or datasets to the public repository.

## Smoke tests

The tests construct arrays in memory and do not require or create study data:

```bash
python -m unittest tests.test_generation
```
