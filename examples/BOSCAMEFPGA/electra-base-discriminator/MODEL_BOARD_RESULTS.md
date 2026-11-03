# ELECTRA 完整模型回传验证

2026-11-03，fpga112 / FPGA7，真实 pinned checkpoint、真实 tokenizer 句对、
batch=1、sequence=128（27 有效 token）、eval。全部 12 层 encoder 和
discriminator dense/GELU/prediction 在板上运行，1,571 次真实 typed adapter
调用全部返回。板端只推理和快照；14 份 BIN 全由 scheduler 下载后在本地比较。

**实现一致性 PASS：14/14 张量、1,278,080 个 FP32 元素逐字节一致，最大误差 0。**
参考在本地运行实际 `model_runtime/model.c`、实际 adapter 和实际 emitted
scalar LLVM；仅 AME 整数乘法由独立精确 signed-i8 / INT32 dot oracle 实现。
K≤3072，不发生 INT32 溢出。独立算子的真实硬件数值证据仍见
[RETURNED_OUTPUT_RESULTS.md](RETURNED_OUTPUT_RESULTS.md)。

| 项目 | 结果 |
|---|---|
| Scheduler task | `2f68c53895904e529c0a74d1370cd3d2` |
| 退出状态 | completed / exited / app_exit_code=0 / exit_code=0 |
| BIN 大小 | 190,079,812 bytes |
| BIN SHA256 | `5c7dd741b8c11349f0dce9eb1651d3828c2f001eb2d64581604709cd7d25f037` |
| 模型 cycles | **3,168,993,218** |
| Typed adapter cycles 总和 | 3,167,297,213 |
| 快照 cycles | 412,884,919 |
| 推理加快照 cycles | 3,581,878,137 |
| 回传 | embedding + 12 层 hidden + logits，14 BIN / 5,112,320 bytes |

模型 cycles 覆盖真实 C 调度和全部 typed kernel/grid/DMA/fence，扣除层快照复制；
不包含权重装载、scheduler 捕获/下载或本地 oracle。约 5.5296 MHz 下，
模型计算约 573 秒。60 秒不能作为该模型 hang 的依据。

固定 dense 权重使用与运行时相同的逐行对称 INT8 量化，74 个矩阵已逐张量与
独立 Torch 量化对照，整数结果全部相同。权重 scale、bias、embedding、
LayerNorm 和激活保持 FP32；激活和 QK/PV 右操作数仍在板端动态量化。
预量化减少 74 次固定权重量化调用，原动态权重版本的 1,645 次调度 ABI 仍通过。
`board_model.ld` 把回传快照放在 `0x80100000`，权重和工作区放入 PMP operand
窗口，NH/E6 stack 和控制状态保持稳定缓存策略。

## 独立数学参考与 FP32 精度

原先冻结的 PyTorch W8A8 参考容差比较 **FAIL，未放宽容差，也未覆盖旧记录**。
实际结果与该参考的最终 logits 最大差为 0.867895、RMSE 为 0.308100；
同一 emitted LLVM 本地运行产生完全相同的差异，而板端与 emitted 参考全部
逐字节相同。这说明硬件实现复现了当前编译后的数值语义；不能将其表述为与
PyTorch W8A8 或 HF FP32 逐值相同。

实际 logits 对 HF FP32 最大差 0.958637、RMSE 0.410748；27 个有效 token
最大差 0.415571。本例按 logit≥0 的二值判断在 128 个位置均一致。
这些是一个句对的回归结果，未建立数据集上的 discriminator 精度结论。

## 复现与证据

使用 boscame Python 环境，从仓库根目录执行；先构建独立算子对象并准备 assets/weights：

```bash
E=examples/BOSCAMEFPGA/electra-base-discriminator
make -C "$E" model-bin PYTHON=python
make -C "$E" model-host PYTHON=python
python "$E/tools/run_model_board.py" --output "$E/build/model-board-runs/new-run" --timeout 1800
```

新的 runner 保留独立 PyTorch 对照状态，再运行严格 emitted 参考比较。
历史首任务的 `result.json` 保留 `LOCAL_VALIDATION_FAILED`，最终两个判据分别在
`verification-summary.json` 记录，不将独立参考失败抹成 PASS。

证据根目录：
[fpga112-checkpoint-pair-20261103](build/model-board-runs/fpga112-checkpoint-pair-20261103/)。
其中 `task.json`/`uart.log` 是原始任务，`artifacts/manifest.json` 保存全部
shape/dtype/length/SHA，`local-validation.json` 保存原独立参考失败，
`local-emitted-validation.json` 保存全部逐字节一致结果，
`verification-summary.json` 汇总两种参考与 cycles。真实输入与 checkpoint
绑定在 `params-manifest.json`；源码、镜像、ELF、审计和命令均保留。
