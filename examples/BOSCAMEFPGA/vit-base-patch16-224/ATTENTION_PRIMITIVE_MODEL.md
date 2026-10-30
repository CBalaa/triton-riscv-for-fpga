# ViT explicit attention primitive validation

Collection: **OFFLINE_MODEL_CHAIN_FRAGMENTS_PASS_FPGA_NOT_RUN**. Full model execution and checkpoint accuracy: **not demonstrated**.

Inventory/dependencies: PASS (43 cases). Current ASM artifacts: 43/43. Primitive strict checks: 10/10. Actual generated model adapter chains: PASS_MODEL_CHAIN_FRAGMENTS_ONLY (scope=captured_attention_fragment).

The model graph expands each QK/PV site into 49 existing primitive calls. Generated C adapters only construct descriptors. Numerical execution uses current strict emitted primitive IR. No independent executable attention case or host candidate arithmetic implements the model route.

The five retired FP32 Linear executables and both executable attention directories are absent. Dense Linear reuses the existing quantize → overwrite matmul_i8 → dequantize route. QK reuses quantize_197x64, quantize_transpose_64x197, matmul_i8_197x197x64 and dequantize_197x197; PV reuses quantize_197x197, quantize_transpose_197x64, matmul_i8_197x64x197 and dequantize_197x64. pack/unpack_heads_12x197x197 provide compiled layout staging for aligned 197-column head slices. The QK dequantizer applies 0.125 once; PV applies no QK factor.

Full source graph: PASS_SOURCE_GRAPH_ONLY; routes={'dense_w8a8': 73, 'qk_w8a8': 12, 'pv_w8a8': 12, 'patch_w8a8': 1}; attention sites=24; primitive calls=1176. Host C adapter object compile: PASS. The complete source graph uses deterministic random untrained weights; checkpoint verification and model accuracy are pending. This establishes replacement and call contracts only.

Local bufferized graph: 279 workspace operands and 146 destinations resolve to their caller-owned roots; 0 workspace copies and 3 verified readonly layout copies. Typed memref boundaries prevent tensor bufferization from disconnecting primitive writers and readers.

Native ABI mock: 10/10; local ELF/audit: 10/10. matmul_i8_197x197x64: 8 emitted signed-i8/i32 AME MMA operations, initializer=private DDR zero stores then tile.load_acc; matmul_i8_197x64x197: 8 emitted signed-i8/i32 AME MMA operations, initializer=private DDR zero stores then tile.load_acc. matmul_i8_197x197x64: 136 AME instructions in audited ELF; matmul_i8_197x64x197: 136 AME instructions in audited ELF.

FPGA: **NOT_RUN_USER_FPGA_BROKEN**. No task, UART, cycles or hardware check count was generated. Historical images and board evidence remain separate. The execution limit remains 120 seconds.

| Captured actual model chain | Heads × rounds | Programs | Final outputs checked | Quantized oracle |
|---|---:|---:|---:|---|
| attention_qk | 12 × 3 | 33228 | 1397124 | PASS |
| attention_pv | 12 × 3 | 27684 | 453888 | PASS |

Numerical fragments checked 9278892 intermediate output elements; 338688 native guard bytes and 970752 simulator guard bytes. They checked 11035152 native input bytes and 32470308 simulator readonly bytes for changes. Guard/input/padded-tail errors and reads of old external C were zero. Per-head scratch and simulator SPM remained dirty across heads and all three rounds; repeated stress results were identical.

Quantized implementation correctness and FP32 error are separate. The original absolute/relative 0.005 tolerances are retained; no tolerance violation is converted to FP32 PASS. The signed/ties/extremes rounds deliberately use values with magnitude up to 130048.0078125, including non-probability PV inputs, to stress arithmetic and tails; their absolute errors are not model accuracy measurements. Relative L2 is ||quantized−FP32||₂/||FP32||₂. The outside-tolerance count uses |quantized−FP32| > 0.005 + 0.005×|FP32| element by element.

| Chain / round | Max abs | Mean abs | Relative L2 | Outside original tolerance |
|---|---:|---:|---:|---:|
| attention_qk / normal | 0.0403970387 | 0.00595040913 | 0.0082165486 | 126272/465708 |
| attention_qk / signed-ties-extremes-tail-first | 190218240 | 105153.989 | 0.00270080854 | 98150/465708 |
| attention_qk / same-input-dirty-workspace-reverse | 190218240 | 105153.989 | 0.00270080854 | 98150/465708 |
| attention_pv / normal | 0.00351344794 | 0.000546068845 | 0.00559204614 | 0/151296 |
| attention_pv / signed-ties-extremes-tail-first | 4.56759706e+09 | 2004699.15 | 0.00277218025 | 33366/151296 |
| attention_pv / same-input-dirty-workspace-reverse | 4.56759706e+09 | 2004699.15 | 0.00277218025 | 33366/151296 |

Current source, generated adapters, graph call manifest, simulator and oracle hashes are bound in the saved chain report. The read-only collector rejects partial, stale or altered saved evidence. Local attention fragment execution does not establish execution of the complete ViT model, bias/residual graph, checkpoint accuracy or FPGA behavior.

Saved-evidence self-check: PASS; 35 incomplete/stale/altered evidence mutations rejected. This self-check executes the read-only collector and independent expected-value oracles only.

Evidence: [strict adapter/IR fragments](build/attention-primitive-model-20261029/strict-model-chains/report.json), [primitive strict checks](build/attention-primitive-model-20261029/primitive-strict-report.json), [evidence self-check](build/attention-primitive-model-20261029/attention-evidence-selfcheck.json), [complete source graph](build/attention-primitive-model-20261029/model-checks/full-source-graph/report.json), [local ELF/image hashes](build/attention-primitive-model-20261029/local-images/manifest.json), [historical evidence preservation](build/attention-primitive-model-20261029/historical-report-preservation.json), [compiler rollback](build/attention-primitive-model-20261029/compiler-rollback/summary.json).
