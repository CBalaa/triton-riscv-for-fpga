# Direct quantized attention validation

Current evidence status: **INCOMPLETE_LOCAL_EVIDENCE**. FPGA execution: **NOT_RUN_USER_FPGA_BROKEN**. Full model execution and checkpoint accuracy remain unverified.

QK and PV retain their dense f32 public descriptors. Their real `@triton.jit` bodies quantize runtime operands, execute signed-i8 AME matrix multiplication with an INT32 zero initializer, and dequantize to f32. There is no child-dispatch composition container. Quantization uses [-127,127], nearest rounding with ties away from zero and scale=1 for a zero vector. QK applies 0.125 once; PV applies no additional factor.

| Case | Shape (heads,M,N,K) | Grid | Build | AME IR | Strict simulator | ABI | Local image |
| --- | --- | --- | --- | --- | --- | --- | --- |
| attention_pv_12x197x64x197 | [12, 197, 64, 197] | [13, 1, 12] | PASS | PRESENT | INCOMPLETE | ABI_SMOKE_PASS | LOCAL_IMAGE_PREPARED_NOT_EXECUTED |
| attention_qk_12x197x197x64 | [12, 197, 197, 64] | [13, 4, 12] | PASS | PRESENT | INCOMPLETE | ABI_SMOKE_PASS | LOCAL_IMAGE_PREPARED_NOT_EXECUTED |

Inventory: PASS; 45 cases; 0 unresolved dependencies. The five retired FP32 Linear directories remain absent. Dense and patch projection continue to use the existing quantize / pure i8 matmul / dequantize chains.

Strict correctness is bit-exact against an independent quantized algorithm oracle. Saved full-grid evidence must include all 12 heads, normal and signed/tie/extreme fixtures, 197 tails, guards, input preservation, zero old-C reads and repeated execution on dirty state with reversed program order. Collection reassembles saved shards and rejects stale source/module/output hashes. ABI smoke uses a mock and establishes no numerical result.

- attention_pv_12x197x64x197: simulator INCOMPLETE; programs=unavailable, outputs=unavailable; old-composition output comparison=INCOMPLETE.
- attention_qk_12x197x197x64: simulator INCOMPLETE; programs=unavailable, outputs=unavailable; old-composition output comparison=INCOMPLETE.

AME evidence is taken from generated IR and the linked executable:

- attention_pv_12x197x64x197: 2 signed-i8 AME matmul routes; 8 emitted i8/i32 `tile.mma` operations; 8 `mqma.b.mm` assembly instructions; linked ELF audit=PASS, AME instructions=120. [IR](build/direct-attention-20261029/kernels/attention_pv_12x197x64x197/kernel.boscame.mlir), [ELF audit](build/direct-attention-20261029/kernels/attention_pv_12x197x64x197/board/elf-audit.json).
- attention_qk_12x197x197x64: 4 signed-i8 AME matmul routes; 4 emitted i8/i32 `tile.mma` operations; 4 `mqma.b.mm` assembly instructions; linked ELF audit=PASS, AME instructions=84. [IR](build/direct-attention-20261029/kernels/attention_qk_12x197x197x64/kernel.boscame.mlir), [ELF audit](build/direct-attention-20261029/kernels/attention_qk_12x197x197x64/board/elf-audit.json).

FP32 approximation error is a separate measurement. The original absolute and relative tolerances remain 0.005 each; they are not relaxed to obtain a PASS. Extreme stress inputs test quantization semantics and can have large absolute errors. Neither an exact quantized implementation nor the values below establish model accuracy.

| Case / round | Max absolute error | Mean absolute error | Relative L2 | Outside original tolerance |
| --- | ---: | ---: | ---: | ---: |
| Current direct-kernel measurements pending | unavailable | unavailable | unavailable | unavailable |

Local ELF/bin preparation uses each case’s existing Makefile and `GOLDEN=1`. The NR ELF audit binds its result to the actual linked ELF hash; saved image manifests also bind sources, kernel object, generated descriptors, runtime objects, golden values and disassembly. A prepared image and host golden generation establish no FPGA result. No task, UART, cycle or hardware comparison count is invented: task=null, UART=null, cycles=[], hardware_checked=0. The configured scheduler execution deadline remains 120 seconds.

Model checks: PASS_GRAPH_CHECKS_ONLY; 24 production dependency cases. These cover route/fallback gates and generated descriptors. The required matrix routes are patch=1, dense=73, QK=12 and PV=12. They do not demonstrate execution of the complete model or pinned-checkpoint accuracy.

Failed prototypes and build logs, the original source archive, old composition records and historical FPGA images are retained. Historical composition results appear only as hash comparisons and are never relabeled as direct-kernel or new hardware execution.

Evidence: [JSON record](build/direct-attention-20261029/collected-evidence.json), [direct build directory](build/direct-attention-20261029), [historical quantized validation](QUANTIZED_ATTENTION_VALIDATION.md).

Refresh saved evidence without compiling or running a candidate:

```bash
BUDDY_ROOT=/home/chh/gitprojects/triton-riscv/buddy-mlir \
/home/chh/miniconda3/envs/boscame/bin/python tools/collect_direct_attention.py \
  --document DIRECT_ATTENTION.md
```
