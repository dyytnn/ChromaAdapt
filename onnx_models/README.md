# MASS-Net ONNX reviewer package

This directory contains the four trained models evaluated in the paper. The
models run on CPU with ONNX Runtime; the reviewer does not need this repository's
PyTorch training environment.

## Quick start

Python 3.9 or newer is recommended.

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

The smoke test generates a neutral image in memory and creates the following
files under `outputs/smoke_test/`:

- `chromosome_labels.png`: raw class IDs 0, 1, and 2;
- `overlap_mask.png`: binary overlap mask encoded as 0 or 255;
- `overlay.png`: visual overlay (instance 1 blue, instance 2 green, overlap red).

No image or dataset is included in this package. To run a local chromosome-pair
crop, replace `--smoke-test` with `--input /path/to/image`. `--input` can also
point to a directory; all supported images immediately in that directory are
processed.

## Included models

| Paper regime | File | Training data |
|---|---|---|
| SuSy | `massnet_susy.onnx` | Synthetic pairs only |
| SuRe | `massnet_sure.onnx` | Labeled real pairs only |
| SuSy to SuRe | `massnet_susy_to_sure.onnx` | Synthetic pretraining, then supervised real-data fine-tuning |
| SuSy to SeDANN | `massnet_susy_to_sedann.onnx` | Synthetic supervision with unlabeled-real adversarial adaptation |

The SeDANN file contains the segmentation network used at inference. Its
training-only gradient-reversal layer and domain classifier are intentionally
excluded.

## Model interface and preprocessing

All four files use ONNX opset 16 and the same interface:

| Name | Type and shape | Meaning |
|---|---|---|
| `input` | `float32 [N, 3, 224, 224]` | Normalized image batch; `N` is dynamic |
| `chromosome_logits` | `float32 [N, 3, 224, 224]` | Background plus two chromosome-instance logits |
| `overlap_logits` | `float32 [N, 1, 224, 224]` | Binary overlap logits |

The reference script reproduces the reported evaluation preprocessing:

1. Read a chromosome-pair crop with OpenCV in **BGR** order.
2. Resize it to 224 x 224 using bilinear interpolation.
3. Convert values to `[0, 1]` and normalize the three channels with mean
   `(0.485, 0.456, 0.406)` and standard deviation `(0.229, 0.224, 0.225)`.
4. Transpose from HWC to NCHW.
5. Apply `argmax` to `chromosome_logits`; apply sigmoid to `overlap_logits` and
   threshold at `0.85`.

The two foreground chromosome IDs identify the two instances in a pair; their
numeric order is not biologically meaningful. The model expects cropped
touching/overlapping chromosome pairs, not an uncropped full metaphase image.

## Python adapters

`adapters.py` exposes one adapter class per released regime. Each class loads
its matching ONNX file by default, and all return the same `MassNetPrediction`
with raw logits plus decoded chromosome and overlap masks:

```python
from onnx_models.adapters import SuSyToSuReOnnxAdapter
import cv2

adapter = SuSyToSuReOnnxAdapter()  # CPUExecutionProvider; uses the bundled ONNX
image = cv2.imread("pair_crop.png")  # OpenCV BGR, same preprocessing as CLI
prediction = adapter.predict_image(image)
print(prediction.chromosome_labels.shape)  # [1, H, W], values 0/1/2
print(prediction.overlap_masks.shape)      # [1, H, W], values 0/1
```

The other classes are `SuSyOnnxAdapter`, `SuReOnnxAdapter`, and
`SuSyToSeDANNOnnxAdapter`. Alternatively, choose a bundled model dynamically:
`OnnxMassNetAdapter.from_model("susy_to_sure")`. To use an ONNX file at a
custom location, pass `model_path=...` to any adapter. `infer_tensor()` accepts
already-normalized float32 NCHW batches; `predict_path()` loads one image from
disk and returns `(original_bgr_image, prediction)`.

Use `--color-order rgb` only when deliberately testing RGB input. The default
`bgr` setting matches the real-data evaluation loader used for the paper.

## Integrity and export validation

Each checkpoint was loaded with strict parameter-key matching. Every exported
graph passed `onnx.checker`, CPU ONNX Runtime inference, a dynamic-batch smoke
test, and a numerical comparison against its PyTorch source:

| Model | Max difference, chromosome | Max difference, overlap |
|---|---:|---:|
| SuSy | 8.941e-06 | 2.766e-05 |
| SuRe | 5.484e-06 | 1.907e-06 |
| SuSy to SuRe | 1.335e-05 | 1.669e-05 |
| SuSy to SeDANN | 9.060e-06 | 5.341e-05 |

SHA-256 checksums:

```text
8ec608f8ccd8c3249ea1ce1acef6315710b45d899496d50d93e5c001bb256a78  massnet_susy.onnx
5505706b3b3ad43dda7609a2d839bcdc51aa433a47ada9edc32581ca317ff698  massnet_sure.onnx
6eda7765c117806601aac7255c119f150204a33d64222f1f2022a2e34e66b195  massnet_susy_to_sure.onnx
4ca3832f91f40015aa9cda3de7d377eff5208148d0f09129524f156c00ae96b8  massnet_susy_to_sedann.onnx
```

They can be checked directly with `sha256sum -c checksums.sha256`.

## Re-exporting from PyTorch

Authors with the full repository and its PyTorch environment can regenerate all
four files from original checkpoints stored outside the public repository:

```bash
python utils/export_paper_onnx.py \
  --susy-checkpoint /path/to/susy.pth \
  --sure-checkpoint /path/to/sure.pth \
  --susy-sure-checkpoint /path/to/susy_to_sure.pth \
  --susy-sedann-checkpoint /path/to/susy_to_sedann.pth
```

To export one checkpoint, use `utils/convert_to_onnx.py`. The exporter rejects
missing or unexpected parameter keys instead of silently exporting a partially
loaded model.
