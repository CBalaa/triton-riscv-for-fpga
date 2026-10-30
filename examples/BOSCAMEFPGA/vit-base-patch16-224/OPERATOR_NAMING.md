# Integer operator naming migration — 2026-10-29

The project uses `gemm_i8_<shape>` for `C = old C + A @ B.T` and
`matmul_i8_<shape>` for `C = A @ B.T`. Both retain physical `B[N,K]`
storage and signed i8 inputs with i32 results. This migration changes names;
the numerical bodies, tiling, masks, quantization and bias/residual wiring
remain unchanged.

## Old-to-new map

These are historical source names, not compatibility aliases. In particular,
an old accumulating `matmul_i8_<shape>` and the current pure-product
`matmul_i8_<shape>` have different contracts. Use the map and the evidence
timestamp when reading an old report.

| Previous name | Current name | Preserved operation |
| --- | --- | --- |
| `matmul_i8_196x768x768` | `gemm_i8_196x768x768` | `C += A @ B.T` |
| `matmul_i8_197x3072x768` | `gemm_i8_197x3072x768` | `C += A @ B.T` |
| `matmul_i8_197x768x3072` | `gemm_i8_197x768x3072` | `C += A @ B.T` |
| `matmul_i8_197x768x768` | `gemm_i8_197x768x768` | `C += A @ B.T` |
| `matmul_i8_1x1000x768` | `gemm_i8_1x1000x768` | `C += A @ B.T` |
| `matmul_i8_overwrite_196x768x768` | `matmul_i8_196x768x768` | `C = A @ B.T` |
| `matmul_i8_overwrite_197x197x64` | `matmul_i8_197x197x64` | `C = A @ B.T` |
| `matmul_i8_overwrite_197x3072x768` | `matmul_i8_197x3072x768` | `C = A @ B.T` |
| `matmul_i8_overwrite_197x64x197` | `matmul_i8_197x64x197` | `C = A @ B.T` |
| `matmul_i8_overwrite_197x768x3072` | `matmul_i8_197x768x3072` | `C = A @ B.T` |
| `matmul_i8_overwrite_197x768x768` | `matmul_i8_197x768x768` | `C = A @ B.T` |
| `matmul_i8_overwrite_1x1000x768` | `matmul_i8_1x1000x768` | `C = A @ B.T` |

[OPERATOR_NAMING_MAP.json](OPERATOR_NAMING_MAP.json) contains the explicit case
and family mapping. The Triton function names change from `matmul_i8` to
`gemm_i8`, and from `matmul_i8_overwrite` to `matmul_i8`, respectively. Typed
and raw exported symbols use the corresponding new case names. No symlink
aliases are created. All source directories were first moved to unique
temporary names before any destination was populated.

The inventory remains **45 operators**: 5 accumulating `gemm_i8` examples,
7 overwrite `matmul_i8` kernels and 33 other entries. The production dependency
closure remains **34**, with no accumulating GEMM in production routes.
Production selection requires both `family="matmul_i8"` and
`accumulation="overwrite"`; small rejection tests cover a falsely named
accumulating kernel. No complete-model compilation is required for this check.

`attention_qk_12x197x197x64` and `attention_pv_12x197x64x197` keep their names
and public layouts. They are batch matmul compositions of two-dimensional
quantization, transpose, integer matmul and dequantization primitives; **12
means attention heads**. Their integer dependencies are now
`matmul_i8_197x197x64` and `matmul_i8_197x64x197`. QK still applies 0.125 once;
PV applies no extra factor. [MODEL.md](MODEL.md) documents the complete ABI.

## Source, compilation and regression evidence

All new evidence is isolated under
[build/operator-naming-20261029](build/operator-naming-20261029/).
The initial process check found no remaining local validation or board client.
The previous batch's pending task had already been confirmed cancelled before
this migration. No submitted or running image was overwritten.

[operator-naming-audit.md](build/operator-naming-20261029/operator-naming-audit.md)
and its JSON companion establish:

- 45/45 complete kernel ASTs/configurations, launchers, makefiles and inventory
  descriptors are identical after the explicit name mapping.
- 495/495 generated adapter, IR and assembly comparisons are equivalent.
  Normalization only maps names, temporary TTIR source locations and padding
  before assembly comments. Instructions, constants, types, masks, strides
  and memory operations are compared unchanged.
- Fresh compilation passes for all 45 entries through contract verification,
  plan verification, emit, LLVM, restricted assembly and object generation.
  The two attention containers remain dispatch-only; their numerical work
  resides in compiled dependencies.
- 45 new binaries link and pass ELF audit. Their hashes and image copies are
  frozen under `prepared-board-images/`. Image preparation is not execution.

[all-operators-validation.md](build/operator-naming-20261029/all-operators-validation.md)
records **45/45 PASS** for compilation, ABI smoke, ELF/image audit and fresh
numerical regression, and **0/45 FPGA execution**. Numerical checks execute
**148,951 programs and 25,947,767 output comparisons**, including retained
stress rounds, layout padding and final composition outputs. The
[45-operator equivalence summary](build/operator-naming-20261029/naming-regression-summary.json)
confirms every round's output hash equals its pre-rename actual result.
Each numerical check executes fresh
emitted IR, with the existing mixed native-C/strict-IR method for the complete
patch adapter. ABI smoke uses mocks and is reported separately.

The independent fresh regressions retain their original fixtures, tolerances,
guards and scheduling checks. Integer tests include dirty old C and retained
SPM; `gemm_i8` still reads the logical old C, while `matmul_i8` reads none.
Evidence for numerical equivalence is split by scope:

| Scope | Evidence |
| --- | --- |
| All 12 integer cases, 36 rounds; independent reconstruction of 24 input fixtures | [equivalence.json](build/operator-naming-20261029/equivalence.json) |
| 28 pointwise/quantize/dequantize/layout cases | [pointwise regression](build/operator-naming-20261029/pointwise-regression/report.json) |
| GELU and Softmax, including actual archived output bytes | [nonlinear equivalence](build/operator-naming-20261029/nonlinear-semantic-equivalence.json) |
| QK/PV and complete patch projection | [composition equivalence](build/operator-naming-20261029/composition-semantic-equivalence.json) |
| 73 dense rewrites, 219 calls; attention routing and four integration regressions | [model naming review](build/operator-naming-20261029/model-naming-review.json) |

[elf-symbol-naming.json](build/operator-naming-20261029/elf-symbol-naming.json)
additionally audits the 12 renamed integer ELFs and the three composition
ELFs: all expected typed/raw symbols and child symbols are defined, with no
unresolved symbols or obsolete overwrite names. The attention empty raw
containers are correctly absent from linked numerical execution.

The regression compares against the independent numerical oracle and the
pre-rename actual output hashes. Equality to pre-rename quantized results does
not establish equivalence to FP32. The quantization errors documented in the
unchanged historical attention report retain their meaning.

## Historical evidence and FPGA boundary

The pre-rename source archive and original working-tree diff/status are saved
in the new evidence root. The audit verifies 212 archived source files,
495 historical compilation artifacts, 249 frozen board files and 231 saved
report references without changing their hashes. `BOARD_RESULTS.md`,
`VALIDATION.md`, `QUANTIZED_ATTENTION_VALIDATION.md`, diagnostic reports and
all prior build/simulator/UART/task records keep their original names and
content. Their old paths are not rewritten to point at a new implementation.
The independent example outside this directory, `../matmul_i8_128x1024x128`,
also keeps its name.

**No newly named image has completed FPGA execution.** The read-only
scheduler probe failed; the last confirmed FPGA4 state was QUARANTINED
because it was absent from the UVHS inventory. Restoring the missing SSH
forward was attempted, but the remote host closed the connection. The new
[platform record](build/operator-naming-20261029/board-platform-final.json)
retains those observations. No FPGA task was submitted, and no remote
service, runner or bitstream was changed. Future execution retains the
120-second deadline, with queue waiting separate.

The previous successful `quantize_197x64` task remains historical and is not
counted as a new naming-regression board PASS. Simulator results and ELF
audits do not establish board correctness. No full graph was compiled or
executed for this migration, and no model accuracy or latency is claimed.

## Reproduction

Use the existing `boscame` environment and in-tree Buddy tools. Set
`BUDDY_ROOT` and `BUDDY_OPT`, and use an explicit fresh `KERNELS` directory.
`make asm`, `tools/check_launch.py`, `tools/simcheck.py`,
`tools/check_quantized_attention.py` and `tools/check_patch_chain.py` produce
the corresponding per-operator checks. `tools/run_naming_pointwise_regression.py`
runs the 28 independent pointwise/layout cases. After generating evidence,
`tools/check_operator_naming.py` audits the old/new source and build identity;
`make operatorcheck KERNELS=/absolute/new/kernels REPORT_DIR=/absolute/new`
collects current evidence without running a model or submitting board work.
