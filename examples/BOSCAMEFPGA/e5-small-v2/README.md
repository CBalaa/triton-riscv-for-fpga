# E5-small-v2 on BOSCAME FPGA

This directory adapts the pinned `intfloat/e5-small-v2` BERT encoder using the
same independent operator layout, `int launch(void)` entry, typed MemRef ABI,
Triton AOT exporter, ordered-reduction policy and NH/E6 link flow as
`../vit-base-patch16-224`. Compiler and host checks run without FPGA access.
[VALIDATION.md](VALIDATION.md) records the offline checks;
[BOARD_RESULTS.md](BOARD_RESULTS.md) records subsequent fpga112 operator runs
with a 60-second execution deadline. The newer complete-output download and
local-oracle results are in [RETURNED_OUTPUT_RESULTS.md](RETURNED_OUTPUT_RESULTS.md).
The complete 12-layer model also passed fpga112 execution and full-output local
validation; see [MODEL_RESULTS.md](MODEL_RESULTS.md). All three complete output
BINs, including 13 hidden-state checkpoints, match the emitted implementation
reference byte for byte.

The source layout follows the ViT example:

| Path | Contents |
| --- | --- |
| `<operator>/` | Independent Triton kernel, C launcher and makefile |
| `cases.py`, `common.mk`, `board.mk`, `board*.ld` | Inventory, compiler stages and NH/E6 linking |
| `model/`, `model_runtime/` | Small pinned upstream metadata and the complete C model graph |
| `tools/` | Export/build tools, host checks and optional board/local validation |
| `VALIDATION.md`, `*_RESULTS.md`, `MODEL_RESULTS.json` | Validation scope and portable results |

Weights, downloaded tokenizer files, compiled images and raw execution evidence
remain in ignored `assets/` and `build/`. They are preserved locally and are not
included in the source checkout. The committed result reports retain task IDs,
output hashes and numerical criteria; their raw evidence paths refer to that
local archive.

The fixed specialization is batch 1, sequence length 128, 12 encoder layers,
hidden width 384, FFN width 1536, 12 attention heads of width 32, FP32 LayerNorm
with epsilon `1e-12`, and exact-definition GELU. Input IDs, masks and type IDs
use contiguous INT32 arrays; the host tokenizer performs the INT64 conversion
where Hugging Face requires it. The actual checkpoint accepts up to 512 tokens;
this specialization pads or truncates to 128. Dropout is disabled for inference.

Each of the 30 operator directories contains `kernel.py`, `launch.c` and
`makefile`. Literal `KERNEL_CONFIG` dictionaries describe shapes, arguments,
constexprs, complete launch grids and output arguments. `cases.py` reads them
without importing Triton or PyTorch. Every C launcher initializes distinct
64-byte-aligned operands, calls its generated `_mlir_ciface_kernel_*` adapter
and compares all outputs against an independent scalar reference.

The scheduler board build preserves this original launcher for host validation.
Its generated board launcher keeps initialization and dispatch, captures every
output in immutable DDR snapshots, and returns before the scalar oracle. The
scheduler downloads these snapshots as BIN files; the local verifier compiles
the unchanged `launch.c` and `support.c` with `HOST_TEST`, fills the typed output
descriptors from those actual files, and runs the complete original oracle
twice. Quantization exports both INT8 values and FP32 scales. Metadata, shape,
byte length and SHA256 must match the scheduler artifact manifest; missing,
short or corrupted files fail validation.

The production arithmetic contract follows ViT's W8A8 path. Dense layers and
per-head QK/PV products use per-row symmetric quantization with ties away from
zero, signed INT8 overwrite matmul (`C = A @ B.T` with physical `B[N,K]`), then
FP32 dequantization. QK dequantization multiplies by `1/sqrt(32)` once. PV
quantizes a transposed runtime V operand; it applies no attention scale.
Every matmul tile stays within its logical M/N/K dimensions, checked by
`cases.py`. QK's K=32 is below AME's K=64 minimum, so that specialization uses
an exact scalar INT32 accumulation kernel with BM=BN=1/BK=32; other matrix
specializations retain AME tiles (PV uses BN=32).
Residuals, GELU, LayerNorm, Softmax and embeddings remain FP32. This is an
explicit quantization adaptation of an FP32 checkpoint; task accuracy has not
been established.

The encoder includes word lookup, absolute position/type embeddings,
post-residual LayerNorm, noncausal key padding masks and head split/merge.
The 384-dimensional sentence output uses the attention-mask-weighted mean of
all valid final tokens, including CLS/SEP, then L2 normalization. Padding is
excluded from the mean and key attention; padded query states still exist as
in BERT. The adapter validates ID ranges and binary masks before dispatch.
The complete model rejects an all-zero mask. The standalone pooling primitive
returns zero for that defensive edge case.

E5 requires a text prefix before tokenization: use `query: ` for queries and
symmetric similarity/features, and `passage: ` for retrieved passages.
`tools/encode.py --kind` adds exactly one prefix and uses the pinned local
tokenizer. BERT's pooler is not used for E5 sentence vectors.

```bash
# Run from the repository root with the existing boscame environment installed.
conda activate boscame
export BUDDY_DIR="$PWD/buddy-mlir"
export BUDDY_ROOT="$PWD/buddy-mlir"
source scripts/triton-riscv-env.sh
E5=examples/BOSCAMEFPGA/e5-small-v2

# Inventory and host-only checks (no board access).
make -C "$E5" help
python "$E5/cases.py"
make -C "$E5" hostcheck PYTHON=python
make -C "$E5" modelcheck PYTHON=python
make -C "$E5" model-runtime-check PYTHON=python

# Compiler stages and optional strict emitted-code simulation.
make -C "$E5" asm PYTHON=python
make -C "$E5" simcheck CASES='embedding_lookup_128x384 matmul_i8_128x128x32 mean_pool_128x384' PYTHON=python

# Build ELF/bin files and run NR ELF audits without executing them.
make -C "$E5" bin PYTHON=python
make -C "$E5/mean_pool_128x384" bin PYTHON=python

# Optional pinned checkpoint/tokenizer download and real-weight host checks.
make -C "$E5" assets PYTHON=python
python "$E5/tools/model_reference.py" --assets "$E5/assets" --output "$E5/build/modelcheck-checkpoint"
python "$E5/tools/encode.py" --text 'how much protein should a female eat' --kind query --mode w8a8

# Offline complete-model build, using the pinned assets and all 30 real kernels.
make -C "$E5" model-bin PYTHON=python
make -C "$E5" model-host PYTHON=python
```

The host tools require NumPy, PyTorch, Transformers and safetensors; kernel
checks also require the in-tree Triton package/plugin. Compiler stages use
the repository's Buddy compiler, restricted-assembly tools and NH/E6 runtime.
`BUDDY_ROOT` selects that checkout and `LLVM_BIN` can override its LLVM tool
directory for the complete-model builders. The shared
`../tools/contract_frontend.py`, `scheduler_outputs.py` and
`scheduler_output_runtime.h`/`.inc` provide the same export and returned-output
infrastructure used by the other BOSCAME examples.

After explicitly selecting fpga112 and establishing its scheduler connection,
the optional runner below submits the compiled operator images, downloads every
declared output, and invokes the local oracle. It requires an online READY
device and keeps the foreground scheduler lease alive until each task ends.

```bash
# One operator, or omit --case to run all 30 operators.
python examples/BOSCAMEFPGA/tools/run_returned_outputs.py \
  --model e5-small-v2 --case matmul_i8_128x384x384

# Repeat only local validation of an already downloaded result.
E5_RUN="$E5/build/board-runs/fpga112-returned-outputs-20261103/matmul_i8_128x384x384"
python examples/BOSCAMEFPGA/tools/verify_returned_outputs.py \
  --model "$E5" --case matmul_i8_128x384x384 \
  --artifacts "$E5_RUN/artifacts" --config "$E5_RUN/board_config.json" \
  --output "$E5_RUN/local-recheck"
```

`LOCAL_ORACLE_PASS` means every downloaded output passed the original C oracle
and both complete local replays. The runner also verifies the real scheduler
job, UART output declarations, exit status and artifact metadata before
reporting board operator `PASS`. Independent host fixtures are labeled
`SIMULATED_SELFTEST_PASS`; they do not establish board numerical correctness.
These per-operator checks do not establish target execution of the complete
E5 model; its separate successful execution is recorded in
[MODEL_RESULTS.md](MODEL_RESULTS.md). The original 60-second board evidence remains in its separate run
directory and [BOARD_RESULTS.md](BOARD_RESULTS.md).

`hostcheck` executes real Triton bodies in the CPU interpreter and separately
checks every C launcher's descriptor/grid ABI with mock raw entries. The ABI
check makes no numerical claim. `simcheck` executes emitted BOSCAME modules in
the strict DMA/SPM/REG interpreter with poisoned SPM and logical-range guards.
Compiler success, interpreter checks and FPGA execution are distinct evidence.

`model_runtime/model.c` provides the complete 12-layer C entry `e5_infer`.
It binds the same typed adapters, reuses a caller-owned workspace, dynamically
quantizes weights and activations, and optionally copies final hidden states.
`model_runtime/model.h` documents the weight pointer structs and caller buffer
contracts. [model_runtime/README.md](model_runtime/README.md) maps those fields
to checkpoint tensors.

The complete host forward in `tools/model_reference.py` binds all 30 operator
contracts to checkpoint tensors and writes an explicit 1,649-call W8A8 trace.
Its FP32 decomposition is compared to an independent Hugging Face `BertModel`.
Its W8A8 mode evaluates the quantized contracts with PyTorch on the host; it
does not dispatch the emitted object files.

`tools/build_model_board.py` packs the pinned real checkpoint and tokenizer
input, links the complete C model with the 30 original emitted kernel objects
and typed adapters, and audits the final `build/model-board/e5_model.elf` before
producing its BIN. Every dispatch uses the same NH/E6 cache/fence brackets as
ViT. The image exports the 384-value sentence vector, all 128 final hidden
states, and 13 complete hidden-state checkpoints (embedding LayerNorm and each
of 12 encoder layers). Host reference files are emitted alongside the image;
they are not used to compute target outputs. `board_model.ld` reserves a
256 MiB region, keeps both stacks and control BSS outside the switchable
operand window, and checks all boundaries at link time. The audited complete
image is 137,044,992 bytes, including 132,848,640 bytes of FP32 parameters.

After fpga112 has explicitly been selected, the separate optional
`tools/run_model_board.py` submits that image, retains its foreground lease,
downloads the three full output BINs and compares every element locally with
the frozen W8A8 and independent Hugging Face FP32 references. It also checks
that the final layer snapshot equals the exported hidden state exactly and
that masked-mean/L2 recomputation from downloaded hidden states reproduces the
sentence vector. `--validate-only --output PATH` repeats just that local
validation without scheduler access. The full-model execution evidence is
separate from the standalone operator results. `tools/build_model_host.py`
executes the actual C graph and emitted scalar LLVM locally, using exact C
INT8/INT32 dot products for the matrix kernels. Supply its output directory
with `--emitted-reference PATH` during local validation to require byte equality
for all three returned tensors; both the original W8A8 and HF FP32 comparisons
are retained.

For this complete-model runner, `--client`, `--endpoint` and `--token-file`
override the scheduler executable, selected fpga112 URL and local credential
file. Their environment equivalents are `FPGA_SCHEDULER_CLIENT`,
`FPGA_SCHEDULER_URL` and `FPGA_SCHEDULER_TOKEN_FILE`. Defaults preserve the sibling
`fpga-scheduler/.venv/bin/fpga-scheduler` installation (or `fpga-scheduler` on
PATH), `http://127.0.0.1:18762` and the repository-local
`.agent/secrets/fpga112.token`. Credentials stay outside this example. New runs
default to a fresh UTC-dated evidence directory; an explicit `--output` can select
another location. `--validate-only` requires the archived directory through
`--output`. The dated path in the result report contains the original evidence.
`--help` and `--validate-only` make no scheduler requests.

`board.mk` links the same NH/E6 boot, cache, UART and DMA runtime as ViT, with
an explicit single-E6 private-DDR DMA guard. `board.ld` preserves NH/E6 entry
addresses and expands the standalone operand arena to 64 MiB inside a 96 MiB
link region, because the FP32 vocabulary table alone uses about 44.7 MiB.
The standalone operator board results are recorded separately; this map does
not establish fit for a complete model and its weights. Assets and build
products stay under ignored `assets/` and `build/` directories.
