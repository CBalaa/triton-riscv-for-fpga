# 验证范围与历史记录：2026-11-03

最新结果：32/32 独立算子通过回传 BIN 后的本地原始 oracle；整模型正常结束，
emitted 实现一致性 PASS、独立 Torch W8A8 数值对照 FAIL。
整模型数值验证尚未通过，详见 [MODEL_BOARD_RESULTS.md](MODEL_BOARD_RESULTS.md)。

下表记录初始主机与离线验证阶段，使用 `boscame` 环境、本仓库 Triton
plugin 以及本仓库 `buddy-mlir` 的 compiler/runtime。后续全部32个独立算子
已在 fpga112 执行，结果为26 PASS、6项60秒超时，详见
[BOARD_RESULTS.md](BOARD_RESULTS.md)。板测还修正了4个launcher的旧元素计数，
完成host fixture数值回归、ABI、重新链接/审计及上板复测，首次失败证据单独保留。

| 检查 | 结果与范围 |
| --- | --- |
| 字面量配置/源码检查 | 32 个独立算子，所有 matmul 的 BM/BN/BK≤M/N/K；Python AST 与 C syntax 检查通过 |
| 真实 Triton CPU interpreter | 32/32 通过，对照独立 NumPy 数值参考，INT8/INT32 精确相等 |
| 独立 C launcher ABI | 32/32 通过，检查 descriptor、尺寸、stride、64 字节对齐、非别名与完整 grid；raw entry 为 mock |
| 真实目标编译 | 32/32 通过 Triton AOT、Contract verification、BOSCAME plan/emit、LLVM、NR 指令过滤与 RISC-V object |
| 独立离线镜像 | 32/32 ELF/bin 链接及 NR ELF audit 通过，NH/E6 entry 地址保留；所有 source/artifact hashes 已复核 |
| Strict emitted BOSCAME 模拟 | 8 个算子的完整逆序 grid 通过，包括 embedding、masked softmax、head pack、N=1 bias、QK、N=1 dot、QK dequantize、PV transpose quantize |
| 完整 C 模型 ABI | 实际 `model.c` 执行 12 层和 discriminator，1,645 次 typed 调用与主机顺序完全一致，shape/stride/alignment/nonalias 检查通过；非法 ID 和不足工作区被拒绝 |
| C 目标对象 | 完整 C 调度交叉编译为 RV64 freestanding object |
| 完整主机 FP32 对照 | 随机权重与真实 checkpoint 两种模式；各检查含 padding/type 与全有效 token 两组输入，对照 HF `ElectraForPreTraining` |
| 权重准备 | 实际 checkpoint 的 201 个 FP32 tensor 已打包，共 437,932,036 bytes；逐 tensor 64 字节对齐、名称/shape/offset/SHA256 完整，生成的 C binder syntax 检查通过 |
| Tokenizer | pinned 本地 tokenizer 将真实英文文本转换为固定 `[128]` INT32 输入文件 |

真实 checkpoint 固定为
`google/electra-base-discriminator@1ae76a97c7e84a4e640876a07453fccd636f0667`，
`pytorch_model.bin` SHA256 为
`ff381b988006468d184eb8c1c090be15553155daa960587e8d86465969f1a0c1`。
主机 FP32 logits 与 HF 的最大绝对差分别为 `3.3855438232421875e-5`
和 `4.5299530029296875e-5`。HF 的融合 bias/addmm 与独立 matmul/bias
阶段存在 FP32 舍入顺序差，检查 tolerance 为 atol=rtol=1e-4。
随机完整模型的两组输入最大差均为 0。

真实权重的主机 W8A8 契约与 FP32 最大 logits 差分别为
`1.5490951538085938` 和 `1.3111376762390137`。这些是两组人工构造有效
token ID 输入的数值检查，不是任务精度评估；当前量化没有校准数据集。

所有 matmul 的 logical tile 均符合用户要求。QK K=64 使用 BK=64。
最后 N=1 的 dot 用 scalar INT32 嵌套循环，BM=BN=1、BK=128，每次只读取
一个有效 i8 元素并精确累加，不使用超大 tile。其 ELF 含 0 个 AME、
0 个 RVV 指令，完整 strict simulation 的 128 个程序输出精确相等。
其他矩阵使用 AME，所有矩阵语义均为 overwrite。

Strict 模拟验证完整 grid、persistent poisoned SPM、逻辑 DDR 地址范围、
guard、输入未改动及输出数值。C 模型 ABI 使用 typed mock，主机整模
W8A8 使用 PyTorch 数值契约；两者均不等于完整 emitted 模型执行。
上述为早期离线阶段范围。2026-11-03 已在 fpga112 完成真实整模型：
12 层及 discriminator 全部 1,571 次调用正常返回，模型 3,168,993,218 cycles。
14 个输出 BIN（embedding、12 层 hidden、logits）回传本地，与实际 emitted LLVM
和精确 INT8/I32 dot oracle 共 1,278,080 个 FP32 元素逐字节相同。
原独立 Torch W8A8 参考阈值失败仍保留；未通过放宽容差获得实现一致性结果。
与 HF FP32 的量化差异和当前单句对覆盖范围见
[MODEL_BOARD_RESULTS.md](MODEL_BOARD_RESULTS.md)，尚未建立数据集任务精度。

复现步骤见 [README.md](README.md)。本地详细证据位于被忽略的 `build/`：
`build-report.json`、`host-kernels.json`、`launch-check/report.json`、
`model-host.json`、`model-abi/report.json`、`simcheck-primitives.json`、
`simcheck-matmul.json`、`weights/manifest.json`，以及每个算子的
`board/elf-audit.json`。权重和编译产物不纳入源码。
