# Quantized ViT production update — 2026-10-29

The production path now requires quantization → signed i8 AME overwrite GEMM
→ dequantization for every matrix product. Following the final requested
scope, **all 45 operators pass compilation, ABI smoke, ELF audit and numerical
validation**. Numerical checks execute current emitted IR; patch embedding
also executes its generated C adapter natively. **FPGA coverage is 1/45 PASS;
44 operators remain blocked by the platform. Complete-model execution and
model accuracy are not established.** No further full-graph compilation was
performed after the scope was narrowed to individual operators.

Evidence root: [build/quantized-attention-20261029](build/quantized-attention-20261029/).
Historical `BOARD_RESULTS.md`, old build artifacts and diagnostics retain
their original meaning and were not replaced by these results.

## Inventory and model wiring

The following standalone FP32 executables were deleted:

- `linear_196x768x768`
- `linear_197x768x768`
- `linear_197x3072x768`
- `linear_197x768x3072`
- `linear_1x1000x768`

The 196-row case is absent from the current model's dense matcher; patch
projection already composes quantization and integer GEMM. The four other
shapes remain recognized by `match_linear` independently of retired case
names, because the W8A8 rewriter needs that recognition. Production FP32
reference modes were disabled. The independent host FP32 baseline remains.

Existing dense primitives are reused without duplicate specializations:

| Logical product [M,N,K] | Runtime activation quantizer | Overwrite GEMM | Dequantizer |
| --- | --- | --- | --- |
| [197,768,768] | `quantize_197x768` | `matmul_i8_overwrite_197x768x768` | `dequantize_197x768` |
| [197,3072,768] | `quantize_197x768` | `matmul_i8_overwrite_197x3072x768` | `dequantize_197x3072` |
| [197,768,3072] | `quantize_197x3072` | `matmul_i8_overwrite_197x768x3072` | `dequantize_197x768` |
| [1,1000,768] | `quantize_1x768` | `matmul_i8_overwrite_1x1000x768` | `dequantize_1x1000` |

Dense weights/scales use the existing model packing path. Patch projection
retains `quantize_196x768`, `quantize_768x768`,
`matmul_i8_overwrite_196x768x768` and `dequantize_196x768`.
Bias and residual consumers retain their graph positions. LayerNorm, GELU,
Softmax, add, bias_add and the unrelated integer GEMM examples are retained.

The two attention names and public f32 layouts remain unchanged:

| Composition | A / B / C | New compiled numeric chain per head |
| --- | --- | --- |
| `attention_qk_12x197x197x64` | [12,197,64] / [12,64,197] / [12,197,197] | `quantize_197x64` → `quantize_transpose_64x197` → `matmul_i8_overwrite_197x197x64` → `dequantize_197x197` |
| `attention_pv_12x197x64x197` | [12,197,197] / [12,197,64] / [12,197,64] | `quantize_197x197` → `quantize_transpose_197x64` → `matmul_i8_overwrite_197x64x197` → `dequantize_197x64` |

These eight primitives plus `pack_heads_12x197x197` and
`unpack_heads_12x197x197` are the ten additions. Head layout staging is
compiled Triton: 197×197 f32 elements have an unaligned head stride, so the
private layout uses 38,816 elements per head while preserving the public
compact layout of 38,809 elements. PV packs before quantization; QK unpacks
after dequantization. The seven padding elements remain untouched.

Transpose quantization computes runtime column scales and writes physical
`B[N,K]`; GEMM computes `A @ B.T`. It is not a C or NumPy production transpose.
Quantization uses symmetric [-127,127], ties away from zero, and zero-vector
scale 1. The 197-column dequantizer processes one masked row per program.
QK applies 0.125 once during dequantization; PV applies no extra factor.
The C composition contains only descriptor setup, workspace and dispatch.

The current inventory has 45 entries; production dependency closure has 34.
The expected full graph has 73 dense routes, 12 QK, 12 PV and one patch route.
Dense rewrite fixtures verify 73 expansions/219 calls. Real Buddy capture of
the attention expression verifies both model call substitutions, including
rank-4 descriptors and consuming the QK scale once. Rejection tests cover
remaining graph products, malformed replacement reports, stale dependency
hashes and matrix operations introduced during lowering. These checks are
graph/ABI evidence, not complete-model numerical execution.

A further capture uses the exact `model/config.json` and `TraceableViT`
source with 200 deterministic random-weight parameters (seed 20261029),
without downloading or substituting a claimed checkpoint. This actual full
source graph replaces **73 dense + 12 QK + 12 PV + one patch** products,
produces 391 external calls and has zero remaining matrix products. It
preserves 73 bias, 25 add, 25 LayerNorm, 12 Softmax and 12 GELU calls.
[untrained-route-check/report.json](build/quantized-attention-20261029/untrained-route-check/report.json)
and the generated replacement MLIR record this graph routing check. It does
not execute the emitted full model or validate checkpoint accuracy.
The actual `lower_model.py` then lowers this replaced graph through the local
Buddy tools to LLVM successfully, retaining 391 external calls and passing
the matrix-fallback rejection gates before scalar loop lowering. The
[lowering report](build/quantized-attention-20261029/untrained-route-check/lowered/lower.json)
also explicitly reports no numerical or complete-model execution validation.

## Compilation, AME and ABI evidence

[kernels/build-report.json](build/quantized-attention-20261029/kernels/build-report.json)
records **45/45 PASS** through the build pipeline: 43 numerical entries use
the real frontend, contract verification, plan verification, emit, LLVM,
restricted assembly and object generation. The two attention container IRs
are explicitly dispatch-only; numerical evidence comes from their separately
compiled dependencies, not the empty containers.

The new GEMM emitted IR contains `bosc_ame.tile.mma` with i8 inputs and i32
accumulators. The emitted overwrite route explicitly fills its private DDR
accumulator with i32 zero and loads it before MMA. It does not require a
literal `tile.zero_acc` instruction or a clean external C. Strict dynamic
access checks additionally confirm zero reads of external C.

Both attention ELF audits pass, each with **136 AME instructions**. The QK
GEMM standalone ELF also passes with 136 AME instructions and four AME store
instructions. The complete image audits, assembly, adapters, ELF/map files
and hashes are preserved. The dependency objects used by the two composition
builds match those of the standalone builds.

[launchcheck/report.json](build/quantized-attention-20261029/launchcheck/report.json)
records **45/45 ABI_SMOKE_PASS**. The attention checks run the actual generated
C adapter with nonnumerical child mocks: all 49 child calls, 12 head slices,
64-byte alignment, physical right operand layout, scale axes, disjoint/reused
workspaces and pack/unpack order are checked. This is explicitly not a
numerical test. Each real board composition has one outer timer covering all
heads and layout staging; child wrappers are untimed.

## Strict emitted-IR simulation

[all-operators-validation.md](build/quantized-attention-20261029/all-operators-validation.md)
lists every current operator and its separate compilation, ABI, ELF/image,
numerical and FPGA status. The corresponding
[JSON](build/quantized-attention-20261029/all-operators-validation.json)
binds those results to current source/module hashes and retained evidence.
All 45 numerical checks pass, totaling **148,951 executed programs and
25,947,767 output comparisons**. These counts include repeated stress rounds,
layout padding checks and final composition outputs; they are not counts of
unique model outputs. Intermediate attention checks are reported separately.

The expanded operator checks include the existing kernels, rather than only
the ten additions and two attention compositions:

| Additional scope | Executed programs | Output comparisons | Evidence |
| --- | ---: | ---: | --- |
| Existing 20 add/bias/LayerNorm/quantize/dequantize operators | 30,739 | 10,741,805 | [pointwise](build/quantized-attention-20261029/all-pointwise-simcheck.json) |
| Ten existing integer GEMMs | 3,624 | 6,355,824 | [integer GEMMs, including the two new ones](build/quantized-attention-20261029/all-integer-simcheck.json) |
| GELU | 14,184 | 1,815,552 | [nonlinear](build/quantized-attention-20261029/all-nonlinear-simcheck.json) |
| Softmax | 7,092 | 1,397,124 | [nonlinear](build/quantized-attention-20261029/all-nonlinear-simcheck.json) |
| Complete patch projection | 7,260 | 451,584 final | [patch chain](build/quantized-attention-20261029/patch-chain-check.json) |

GELU and Softmax execute three complete rounds with poisoned SPM retained
per shard. Their maximum absolute errors are respectively 4.43347e-7 and
1.60728e-7 under the unchanged `atol=rtol=2e-5`. Existing pointwise kernels
use their retained complete-grid fixtures; quantizers, dequantizers and
integer GEMMs run three rounds.

Patch projection is a mixed native-C/strict-IR check: the actual generated
C adapter performs dispatch, bias and NCHW layout conversion, while im2col
and all four numerical child kernels execute emitted IR. No NumPy result is
used as candidate kernel output. Im2col follows the adapter's forward order;
the children vary forward/tail-first/reverse order over three rounds, with
guards, unchanged inputs and dirty/reused workspace checked. This is not
target assembly execution.

[primitives-simcheck.json](build/quantized-attention-20261029/primitives-simcheck.json)
and [attention-chain-check.json](build/quantized-attention-20261029/attention-chain-check.json)
record all ten primitives and both full compositions passing three complete
rounds. Candidate values come from executing emitted dependency IR in the
strict DDR/SPM/REG simulator; NumPy is used for fixtures and independent
references only. Calling the empty composition container as a numeric kernel
is rejected.

| Scope | Executed programs, three rounds | Output comparisons |
| --- | ---: | ---: |
| Ten standalone primitives | 25,140 | 3,334,866 |
| QK, all 12 heads per round | 33,228 | 1,397,124 final; 5,113,332 intermediate |
| PV, all 12 heads per round | 27,684 | 453,888 final; 4,165,560 intermediate |

Tests cover all-zero vectors, signed values, exact half ties and adjacent
f32 values, finite extremes, 197 tails, scale axes, transposed byte packing,
integer dot, ordered dequantization, guards, unchanged inputs, dirty C and
workspace reuse. Each dependency retains its poisoned simulator Machine
across heads and rounds. Forward, tail-first and reverse execution pass; the
two identical stress-input rounds have equal output hashes. Quantized values,
scales, intermediate integer products and final restored values match the
independent algorithm oracle exactly.

The existing classifier chain was also executed against the freshly compiled
reused primitives: quantize → i8 overwrite GEMM → dequantize → bias → residual,
with zero old-C reads. Its separate
[reused-classifier-chain.json](build/quantized-attention-20261029/reused-classifier-chain.json)
records PASS without claiming complete-model execution.

The collector checks current module/dependency hashes and complete
head/stage/grid/round coverage, and rejects deliberately damaged evidence.
[attention-validation-selfcheck.json](build/quantized-attention-20261029/attention-validation-selfcheck.json)
records those checks.

## Quantization error is a separate result

No original FP32 tolerance was widened. The unchanged comparison budget is
`abs(error) <= 0.005 + 0.005*abs(FP32 reference)`. Exact agreement with the
quantized oracle establishes implementation correctness, not FP32 equivalence
or complete-model accuracy.

| Strict simulator normal fixture | Max absolute | MAE | RMSE | Relative L2 | Outside original tolerance |
| --- | ---: | ---: | ---: | ---: | ---: |
| QK, random normal inputs | 0.0403970 | 0.00595041 | 0.00754263 | 0.00821655 | 126,272 / 465,708 |
| PV, probability-shaped A | 0.00351345 | 0.000546069 | 0.000687115 | 0.00559205 | 0 / 151,296 |

Signed extreme stress fixtures have larger absolute FP32 errors; their full
metrics are in [attention-chain-summary.json](build/quantized-attention-20261029/attention-chain-summary.json).
They are not marked accuracy PASS. The original small deterministic launcher
fixtures have separate host C baseline measurements: QK max absolute error
0.0000517550, PV 0.000275328, each with zero old-tolerance violations. Those
smaller fixture errors do not supersede the random/stress observations.

## FPGA execution and remaining work

The `fpga-board-run` workflow used the existing scheduler and `run_bin.py`
with its **120-second execution deadline**; queue waiting is separate.
[board-coverage.json](build/quantized-attention-20261029/board-coverage.json)
retains the initial 12-case attention subset. The final 45-operator table
records **1/45 executed PASS, 44/45 platform-blocked**: one task was cancelled
before start, and 43 images were not submitted.

`quantize_197x64` passed two actual invocations, cycles **436,338 / 435,731**,
25,610 checked values, zero errors. Task:
`5d98171cb5884dc092909daca2fa8510`. Its image SHA256, UART, task details,
ELF audit, validation count and each invocation's cycle are preserved under
[board-runs/quantize_197x64](build/quantized-attention-20261029/board-runs/quantize_197x64/).
Cycles measure the complete typed adapter/grid/DMA/fence and exclude fixture
initialization, reference calculation, comparison and UART.

After that run, the scheduler reported FPGA4 **QUARANTINED**:
`FPGA4 (B1.F0) missing from UVHS inventory; reported FPGA units: 0,1,2,3`.
The next task `1a2b5522a17e4f4db20426f244624429` stayed queued, never started,
and was cancelled after stopping only this batch's parent. Unrelated tasks
were not cancelled. No timeout or hang is inferred for this or any unsubmitted
image. [board-platform-final.json](build/quantized-attention-20261029/board-platform-final.json)
preserves the state. Remote service, runner, bitstream and hardware selection
were not changed.

All 45 ready board images, audits and SHA256 values are frozen in
[prepared-board-images](build/quantized-attention-20261029/prepared-board-images/).
The original twelve-image manifest is retained as `attention-manifest.json`;
`manifest.json` covers all 45. The remaining 44 operators still require real
board execution after the platform recovers. Their compilation and simulation
results do not substitute for board results. A final read-only scheduler
probe at 02:57 UTC also failed with connection resets while the existing
SSH forward was listening; see
[board-platform-recheck.json](build/quantized-attention-20261029/board-platform-recheck.json).
The saved quarantine observation remains the last confirmed device state.

Full emitted-model simulation/FPGA inference, checkpoint-backed output
comparison and top-1 accuracy remain unvalidated. The model prerequisite gate
records SKIP because a complete emitted-model runner is unavailable and the
pinned checkpoint is missing; existing operator and graph tests cannot turn
that result into PASS.

## Preservation and reproduction

The old diagnostic batch was stopped/cancelled. `preexisting.diff`,
`source-before.tar.gz` and `retired-source/` preserve pre-task source changes
and removed sources. Old board/hang evidence and the default `build/kernels`
were not overwritten. New artifacts use the isolated evidence root above.
`source-snapshot/` records the board build inputs. A later metadata/docstring
cleanup changed no emitted numerical IR or object bytes, verified in
`metadata-cleanup-verification.json`.

Use the `boscame` conda environment and this repository's Buddy/LLVM build.
The relevant entries are `make asm`, `tools/check_launch.py`,
`tools/simcheck.py`, `tools/check_quantized_attention.py`,
`tools/check_patch_chain.py`, `tools/check_all_operators.py`,
`tools/check_w8a8_rewrite.py`, `tools/check_model_routes.py`,
`tools/model_simcheck.py` and `tools/run_board_suite.py`. Pass the isolated
kernel/report paths to avoid mixing historical and current evidence.
`make operatorcheck KERNELS=/absolute/evidence/kernels
REPORT_DIR=/absolute/evidence` only collects saved per-operator evidence;
`make patch-chaincheck` runs the complete patch adapter check.
