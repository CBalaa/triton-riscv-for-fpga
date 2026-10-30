# Capturing and assembling the complete model

The model sources were ported from branch `vit-base-patch16-224`, commit
`478766cfddb11e61560c44d68320ee21279b90ae` of `CBalaa/buddy-mlir-for-fpga`.
The current 43 entries, including quantized attention primitives and
independent integer/layout primitives, live in their individual case directories. Shared
configuration lives in each `kernel.py`; `../cases.py` enumerates it without
importing Triton. Compiler stages and dependencies live in `../common.mk`.
The original monolithic `triton/kernels.py` is not used.

The current cleanup checks inventory, dependencies, references, Python syntax
and diffs to prepare later standalone operator verification. It establishes
no new numerical or FPGA result. The production commands below remain
available for subsequent work.

Run from the repository root with a Python environment that contains PyTorch,
Transformers, SafeTensors and the matching built Buddy bindings:

```bash
VIT=examples/BOSCAMEFPGA/vit-base-patch16-224
export PYTHONPATH="$PWD/buddy-mlir/build/python_packages${PYTHONPATH:+:$PYTHONPATH}"
python "$VIT/tools/fetch_assets.py" --output "$VIT/assets"
python "$VIT/tools/import_model.py" --assets "$VIT/assets" \
  --checkpoint "$VIT/assets/model.safetensors" \
  --output "$VIT/build/import" --save-mlir
make -C "$VIT" boscame PYTHON=python
python "$VIT/tools/simcheck.py" --all
python "$VIT/tools/replace_triton.py" --import-dir "$VIT/build/import" \
  --triton-build "$VIT/build/kernels" --output "$VIT/build/replacement-w8a8" \
  --require-complete
python "$VIT/tools/lower_model.py" --replacement "$VIT/build/replacement-w8a8" \
  --output "$VIT/build/lowered"
python "$VIT/tools/model_simcheck.py" --mode w8a8 \
  --replacement "$VIT/build/replacement-w8a8"
```

Patch projection and all 73 dense linears use
quantize -> signed i8 AME `matmul_i8_*` -> dequantize. QK and PV are model
functions expanded into four actual primitive calls per head. Runtime K/V
use compiled quantize-transpose kernels; physical i8 RHS is `[N,K]`.
QK adds compiled unpack and PV adds compiled pack to preserve dense public
197x197 head layouts with 64-byte aligned primitive slices. There are no
standalone attention executable cases or configuration dispatch containers.
The generated adapters only validate descriptors and dispatch one primitive.
Model-owned scratch enters the graph as mutable memrefs. Standard
tensor/buffer bridges preserve the activation/output boundaries; production
workspace checks verify that each primitive reads its producer's storage
without workspace copies. This is model adaptation and requires no compiler
extension.

Integer names encode destination semantics: `matmul_i8_*` overwrites C and
ignores dirty C; `gemm_i8_*` accumulates and remains independent. All production
matrix routes require overwrite. QK consumes 0.125 once in its dequantizer;
PV applies no additional factor. Floating-point bias, residual, normalization
and activation operations retain graph order. FPGA FP32-reference modes are
disabled; the independent host baseline below remains available. See
[MODEL.md](../MODEL.md) for primitive names, head offsets and workspace lifetime.

The pinned checkpoint is about 346 MB and is not checked in. `fetch_assets.py` verifies its SHA256
and reuses an already verified copy. Model import requires the real checkpoint;
random initialized weights are not a substitute.

Each numerical kernel follows `TTIR -> standard bufferized contract ->
BOSCAME plan -> plan verify -> emit`. The graph lowerer handles the remaining
standard graph plumbing around those external calls; it neither selects
kernel routes nor compiles numerical operations through a host fallback.

`model_simcheck.py` is currently a prerequisite audit. It checks the 32 required
case results, emitted-module hashes and the current explicit primitive dependency closure.
It returns `SKIP` (exit status 2) for any
blocked/missing kernel, incomplete launch coverage or stale evidence. Even if
all kernels pass, it cannot claim full-model simulation until a simulator
executes the complete graph with those emitted kernels and the pinned model
assets. Host graph check scripts remain optional host checks. Standalone board runs
are separate evidence and do not execute the full model.


## Reproducible accuracy comparison

Provide a JSON sample manifest with a `samples` list. Each item has a unique
`id`, an image `path` relative to the manifest, and an optional zero-based
ImageNet `label` in `[0,999]`. Missing labels are allowed; top-1/top-5 accuracy
is then reported only for the labeled subset. Both execution paths consume
the exact prepared NPZ bytes and recorded sample order.

```bash
python "$VIT/tools/compare_accuracy.py" prepare --assets "$VIT/assets" \
  --samples samples.json --output "$VIT/build/accuracy-inputs.npz"
python "$VIT/tools/compare_accuracy.py" reference --assets "$VIT/assets" \
  --inputs "$VIT/build/accuracy-inputs.npz" \
  --output "$VIT/build/fp32-logits.npz"
# Run the complete emitted model on accuracy-inputs.npz, producing w8a8-logits.npz.
python "$VIT/tools/compare_accuracy.py" compare \
  --inputs "$VIT/build/accuracy-inputs.npz" \
  --reference "$VIT/build/fp32-logits.npz" \
  --candidate "$VIT/build/w8a8-logits.npz" \
  --report "$VIT/build/accuracy-comparison.json"
```

The reference command verifies checkpoint/config hashes and runs pure FP32
patch/dense operations. The prepare command verifies the pinned processor,
records original image hashes and saves the processor's deterministic output.
Candidate NPZ must contain finite `[samples,1000]` `logits`. Its `.json` sidecar
must copy `model_id`, `revision`, `checkpoint_sha256`, `inputs_sha256`,
`preprocessor_sha256` and `sample_ids` from the prepared-input sidecar, provide
`logits_sha256`, `module_sha256` and `execution_report_sha256`, and explicitly
record `execution="boscame-emitted-simulator"` or `"boscame-hardware"`,
`full_model_executed=true`, and `all_numerical_kernels_emitted=true` from the
actual runner. Host graph checks cannot satisfy this protocol. The complete
emitted-model runner remains unavailable; passing standalone prerequisites does not execute the graph.

The comparison reports max/mean/RMS logit error, prediction agreement and
labeled top-1/top-5 changes in percentage points. `MEASURED` records metrics;
it does not imply acceptable accuracy. No dataset or checkpoint is included,
and no full-model accuracy result has been established by these scripts.
