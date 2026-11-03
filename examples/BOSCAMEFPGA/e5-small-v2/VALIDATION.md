# Validation performed on 2026-11-03

This file records the offline validation stage. Subsequent per-operator
execution on fpga112 with a 60-second application deadline is recorded in
[BOARD_RESULTS.md](BOARD_RESULTS.md). The offline checks below used
`/home/chh/miniconda3/envs/boscame/bin/python`, the in-tree triton-riscv plugin
and this repository's `buddy-mlir` compiler/runtime.

The later scheduler returned-output stage passed all 30 operators, downloading
38 complete BIN snapshots and checking all 3,313,184 output elements in each
of two unchanged native C oracle replays. Actual job IDs, FPGA ports, kernel
cycles and artifact evidence are recorded in
[RETURNED_OUTPUT_RESULTS.md](RETURNED_OUTPUT_RESULTS.md).
The later complete-model stage also passed fpga112 execution: all 1,649 calls
and 13 checkpoints completed, and all 688,512 returned FP32 elements agree
byte for byte with the local emitted implementation reference. The original
W8A8 acceptance thresholds also pass. See [MODEL_RESULTS.md](MODEL_RESULTS.md)
and [MODEL_RESULTS.json](MODEL_RESULTS.json) for the separate complete-model
evidence and FP32 quantization comparison.

| Check | Result and scope |
| --- | --- |
| Literal inventory | 30 independent operators; all matmul BM/BN/BK are bounded by logical M/N/K |
| Target compilation | All 30 pass through Contract IR, BOSCAME, LLVM, restricted NR assembly and RISC-V object generation |
| Offline images | All 30 independent operator ELF/bin images link; all 30 NR ELF audits pass |
| Scheduler output build | All 30 images rebuild with 38 complete output snapshots; every output config and ELF SHA audit passes |
| Native returned-output replay infrastructure | All 30 unchanged C launch oracles accept independently computed host fixtures in two complete replays; representative missing/short/corrupted-file checks fail as expected. This is explicitly a simulated selftest, not board-return evidence |
| Native launcher ABI | All 30 independent C launchers pass descriptor, stride, nonalias and complete-grid mock checks |
| Triton CPU numerical interpretation | All 30 actual bodies pass independent NumPy/PyTorch references; the final QK/PV tile implementations were checked again separately with exact integer results |
| Actual C model routing | `e5_infer` passes all 1,649 typed calls, exact call order, descriptors, alignment, buffer bounds and producer checks; seven invalid-request checks dispatch zero kernels |
| C model workspace | 7,479,296 bytes, caller-owned and reusable |
| Strict emitted simulation | Nine operators pass complete grids in reverse order: embedding lookup/add, QK/PV matmuls, key masking, split/merge heads, masked mean and L2 normalization |
| Host model decomposition | Complete 12-layer FP32 decomposition agrees with Hugging Face `BertModel`; all 30 operators are covered by the host W8A8 schedule |
| Checkpoint assets | Pinned `ffb93f3bd4047442299a41ebb6fa998a38507c52`; safetensors SHA256 `45bfa60070649aae2244fbc9d508537779b93b6f353c17b0f95ceccb1c5116c1` |

QK uses exact scalar INT32 accumulation with BM=BN=1/BK=32 because the AME
backend requires K to be a multiple of 64. Its final emitted strict simulation
checks every one of 16,384 output elements across 16,384 programs, 1,064,960
logical DDR accesses and zero DMA transfers, with zero numerical error. PV
uses BM=32/BN=32/BK=128; its four programs check all 4,096 outputs exactly.
Both simulations preserve input bytes and verify output guards. This final
fallback replaces an earlier oversize attention tile; the ViT sources were
not changed.

For the seeded random full-size checkpoint, FP32 hidden-state maximum absolute
error was `1.9073486328125e-6` and sentence-vector error was
`2.2351741790771484e-8`. For the pinned real checkpoint on one deterministic
synthetic token/mask/type input, both FP32 errors were zero; host W8A8 sentence
cosine similarity to FP32 was `0.9985458850860596`, with hidden-state maximum
absolute error `2.397855281829834`. These are arithmetic checks on one input,
not a retrieval benchmark or task-accuracy claim. A real English query encoded
with the prefix/tokenizer tool produced a finite unit sentence vector.

The strict emitted simulation uses poisoned SPM, full logical buffer bounds,
read-only input checks and output guards. It covers the nine listed cases.
The C model test uses typed mocks and establishes routing/ABI only. The Python
W8A8 forward evaluates the operator arithmetic on the host. These original
checks concern offline linking and host evaluation; the later standalone and
complete-model board evidence are recorded in their separate reports.

Reproducible commands are in [README.md](README.md). Detailed local evidence
lives in ignored `build/`: `build-report.json`, per-case `board/elf-audit.json`,
`hostcheck/report.json`, `hostcheck-final-matmul/report.json`,
`model-runtime-check/report.json`, `modelcheck-checkpoint/report.json`,
`modelcheck-checkpoint/operator_trace.json`, `simcheck-subset-report.json`,
and `simcheck-final-matmul-report.json`. Build artifacts and downloaded weights
are intentionally excluded from source control.

The scheduler-output rebuild is recorded in
`.cache/e5-returned-output-build-audit-20261103.json` and
`.cache/e5-returned-output-build-20261103.log`. Native replay infrastructure
selftests live in `.cache/returned-output-replay-selftests/e5-all/report.json`;
the representative E5 matmul/quantization and ELECTRA two-invocation
quantization negative regressions are under that cache's model directories.
These host-generated fixtures are labeled `SIMULATED_SELFTEST_PASS`.
Actual downloaded outputs require matching scheduler job/artifact metadata
and a separate `LOCAL_ORACLE_PASS` before they count as board numerical
evidence.
