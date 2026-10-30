# ViT graph contract

The target is the pinned `google/vit-base-patch16-224` checkpoint: batch 1,
input `[1,3,224,224]`, 196 patches plus CLS, 12 encoder blocks, hidden width
768, 12 heads of width 64, MLP width 3072 and 1000 logits. `model.json`
records the revision and checkpoint digest. Preprocessing is outside the graph.

`tools/import_model.py` preserves the 200 checkpoint parameters, explicit
QK -> scale 0.125 -> Softmax -> PV ordering, separate dense biases, residual
connections, exact GELU and LayerNorm epsilon `1e-12`.

## Production matrix routes

All 98 model matrix products use quantization, signed i8 AME matrix
multiplication and dequantization: patch projection=1, dense=73, QK=12,
PV=12. LayerNorm, GELU, Softmax, bias, residual/add and dequantization remain
floating point. The graph contains 1543 actual external calls after expansion:
219 dense primitive calls, 1176 attention primitive/layout calls and 148 other
numerical calls. There is no executable attention case or attention dispatch
container; shape matching is independent of retired case names.

`match_linear` recognizes `[M,N,K]` shapes `[197,768,768]`,
`[197,3072,768]`, `[197,768,3072]` and `[1,1000,768]`. It consumes the physical
`[N,K]` checkpoint weight, bypasses its graph transpose and reuses existing
`quantize_*`, `matmul_i8_*`, `dequantize_*` specializations. Weight i8/scales
are packed parameters; activations are quantized at runtime. Bias/residual
users retain their original graph positions after dequantization.

`matmul_i8_*` computes `C = A @ B^T` with `family="matmul_i8"` and
`accumulation="overwrite"`, ignores old C and owns its zero initializer.
`gemm_i8_*` computes `C += A @ B^T` and remains an independent accumulating
example outside production routes. Both use physical RHS `[N,K]`.

The five standalone FP32 Linear cases have been retired. Patch projection
already composes im2col, quantization, i8 matmul, dequantization, bias and
NCHW output conversion. FPGA FP32 reference modes are disabled. The independent
host FP32 baseline remains `tools/compare_accuracy.py reference`.

## Attention model expansion

The matcher retains these f32 public layouts, including leading unit batches:

| Model function | Public input/output | Four actual primitive calls per head |
| --- | --- | --- |
| QK | `[12,197,64]`, `[12,64,197]` -> `[12,197,197]` | `quantize_197x64`, `quantize_transpose_64x197`, `matmul_i8_197x197x64`, `dequantize_197x197` |
| PV | `[12,197,197]`, `[12,197,64]` -> `[12,197,64]` | `quantize_197x197`, `quantize_transpose_197x64`, `matmul_i8_197x64x197`, `dequantize_197x64` |

Q and probabilities use row scales. Runtime K and V use column scales over
the complete reduction axis; the compiled transpose quantizers produce
contiguous i8 `[N,K]`. Quantization is symmetric signed i8 `[-127,127]`,
nearest with ties away from zero and zero-vector scale 1. Integer matmul is
overwrite. QK dequantization computes
`((float(acc) * row_scale) * column_scale) * 0.125`; PV has factor 1. The QK
matcher consumes the original scale operation together with its product,
so 0.125 occurs once. Masks cover all 197 tails.

Each model attention site becomes 49 actual `CallExternalOp` graph nodes:
12 heads times four primitive calls plus one compiled layout call. A dense
197*197 f32 head occupies 155236 bytes, which is not a multiple of 64.
QK dequantizes into `[12,38816]` staging, then the compiled
`unpack_heads_12x197x197` restores dense public layout. PV first runs
`pack_heads_12x197x197`, then quantizes aligned staged heads. All other
head strides are 197*64 elements and already 64-byte aligned. The generated
C adapters only validate/build bounded descriptors and call one primitive;
they perform no candidate arithmetic, transpose, copies or head loops.

Each site owns caller-supplied 64-byte aligned quantized-left,
quantized-right, row-scale, column-scale, i32 accumulator, padded staging and
output buffers for the entire model invocation. Scratch is reused across its
heads in graph-body order. There is one graph partition, whose operation list
preserves body order; Buddy imports that list sequentially. External calls
have side effects and are not marked pure. Model-owned workspace arguments
use mutable memrefs before one-shot bufferization. Read-only tensor inputs
receive `to_buffer read_only` bridges; an output receives one
`to_tensor restrict writable` bridge after its final writer. This avoids
copying a tensor destination while later primitive calls read the original
workspace. The adapter layer performs this conversion with standard MLIR
operations; the compiler is unchanged.

Production workspace handling checks primitive storage roots after
bufferization and rejects disconnected readers/writers or workspace copies.
Descriptor adapters enforce offsets, strides and alignment. Scratch remains
live across all four calls and all 12 heads.

`attention-primitive-calls.json` records the actual generated graph calls,
primitive symbols, root buffers, dtypes, shapes, element offsets, stages,
heads and workspace ownership. Its hash binds the replacement report and its
adapter hash binds generated C. This manifest describes model expansion;
it is not an executable attention configuration.

## Production gates and evidence limits

The inventory contains 43 standalone entries, including independent integer
examples; the model dependency closure contains 32. The kernel index requires
current source/adapter/module hashes and completed lowering. The replacement
audits 73/12/12/1 matrix routes, all 1176 ordered attention calls, exact head
views, scale roots and layout staging, and rejects residual matrix operations.
The lowerer validates the manifest and generated adapters before lowering,
then rejects matrix operations before and after TOSA conversion so they cannot
silently become scalar f32 loops. Partial or stale reports fail closed.

The current cleanup retains model shape matching, primitive expansion,
workspace materialization and production lowering. Dedicated local/full-model
capture and model-call-chain validation tools have been removed. This stage
checks inventory, dependencies, references, Python syntax and diffs only;
standalone numerical verification remains separate work.

`tools/model_simcheck.py` audits dependency hashes and complete per-round
launch coverage and records `SKIP`: complete emitted-model execution remains
unimplemented. Standalone/chain simulation, graph replacement and native
adapter compilation do not establish a complete-model or FPGA execution PASS.
Pinned-checkpoint accuracy additionally requires prepared samples and outputs
from actual complete-model execution. Historical fused/composition and FPGA
records remain archived and do not describe the current model expansion.

Historical local graph measurements and their original scope are recorded in
[ATTENTION_PRIMITIVE_MODEL.md](ATTENTION_PRIMITIVE_MODEL.md). Earlier reports,
diagnoses, submitted images and hashes retain their original contents.
