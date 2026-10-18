# Data availability

This public repository does not distribute any image, annotation, mask, patient
record, or generated training dataset used in the study.

The real metaphase images remain subject to their original access, privacy, and
ethical-use conditions. The synthetic training dataset is also not packaged in
this source-code release.

The `data/` directory contains Python dataset loaders, and the `generation/`
directory contains the standalone karyotype preprocessing and synthetic-pair
generator. These directories contain source code only and do not contain study
data. Outputs produced by the generator are ignored by Git.

The released ONNX models can be checked without data by using the in-memory
smoke test documented in `onnx_models/README.md`. Users who are authorized to
use compatible chromosome-pair images can provide their own local paths at run
time; such paths and files are ignored by Git.
