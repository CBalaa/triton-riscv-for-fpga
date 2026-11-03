# MobileNetV3 示例合入与 Review

本次仅导入上游 `integrate-mobilenetv3` 的
`examples/BOSCAMEFPGA/mobilenetv3_small_100.lamb_in1k/`，来源提交
[`350219b`](https://github.com/CBalaa/triton-riscv-for-fpga/commit/350219b824303172c7e91fa88b90cf57110b99ea)。
未导入其 `scripts/run-fpga-scheduler.py` 和对应测试。NH 启动、UART 驱动及
START/RESULT 协议属于镜像运行时，保留在示例内；不包含调度客户端或服务器。

## 与已经跑通的 ViT 对比

| 项目 | ViT-B/16 | 本示例 MobileNetV3-Small |
|---|---|---|
| 整图 | 1,543 次调用、28 个实际使用的特化 | 121 次调用、88 个特化 |
| 数值路径 | INT8 GEMM + FP32 图运算 | 全部 FP32 |
| 默认执行核 | E6 运算、NH 输出 | NH 单核标量 |
| 加速器与内存 | AME/DMA、private DDR、核间结果发布 | 不使用 AME/DMA/private DDR，不进行核间结果发布 |
| 调用诊断 | 逐次 BEGIN/END、当前 kernel、trap 上下文 | 已补齐序号/总数、symbol/case/节点名、BEGIN/END、当前调用与 trap 上下文 |
| 单算子诊断 | adapter 周期、returned/validation/checked/errors | 已补齐 adapter 周期、returned/validation/checked/errors |
| 数值参考 | 独立算子 oracle 与整图参考 logits | 独立 C 算子 oracle 与原始 timm logits |
| 本次本地验收 | 保留原有实现与既有板测证据 | 88/88 host、88/88 NH 镜像审计、完整模型 host 通过 |

ViT 的 cache policy、private DDR release 和 NH/E6 独占 cache line 修复不能
机械搬到默认 NH 单核模型。该模型没有相应的跨核路径；保留的旧 NH/E6 单算子
profile 则将 invocation 与 validation 分为独立的 64B volatile 对象，且用
write-through 策略发布状态。此 profile 不等于启用 AME 加速。

## 已修复的问题

| 问题 | 影响 | 修复与验证 |
|---|---|---|
| NH/E6 profile 共用最终 ELF/BIN，切换时缺少配置依赖 | 可能拿到另一 profile 的旧镜像 | profile identity 强制重链；NH→E6→NH 回归；链接前移除旧 BIN |
| 旧 E6 构建忽略自定义输出目录及部分 flags/header 变化 | 路径错误或复用旧对象 | 尊重 KERNELS/工具参数；配置身份与自动头文件依赖 |
| 模型自定义 OUT 作为命令行参数泄露到子 Make | 各算子可能写入同一模型目录，造成覆盖或构建失败 | 隔离模型/算子输出目录；增加显式外部 OUT 构建回归 |
| 整模型直接链接已有 kernel.o | 算子源码/编译参数变化可能不触发重建 | 模型 host/bin 对实际引用的 case 重新进入 Make 检查依赖 |
| 新增弱 trace hooks 生成 .got，但 NH/E6 linker 未显式收集它 | 单算子 BIN 被填充到约 17 MiB，并消耗过多上传与服务器配额 | 把 .got 收进 .data；保留原始板测镜像，另行构建紧凑镜像并验证相同 kernel.o/adapter.c |
| ld.lld 固定在 LLVM_BIN 下 | 本地工具链没有此文件时无法链接 | LINKER 覆盖与 PATH fallback，与 ViT 一致 |
| 构建报告未验证 adapter/contract 变动 | 被修改的工件仍可能显示构建 PASS | 校验源码、生成器、adapter 和 contract manifest 哈希 |
| launcher 只打印数字编号，缺少返回/完整校验计数 | 无法直接对应到具体调用或识别提前结束 | 完整 BEGIN/END 与汇总、首批错误 logits 位型、trap 当前调用 |
| 单算子输出预填零 | 预期为零的位置漏写也可能通过 | 88 个 launcher 全部使用 NaN sentinel，加入漏写负例 |
| host/QEMU 仅凭 PASS 子串判断 | 可能接受错 case、部分比较或异常数值 | 严格匹配 case、checked、errors、validation_executed、有限最大误差 |
| 整模型 host 对外部绝对 OUT 使用 relative_to(ROOT) | 文档支持的外部导出目录直接异常 | 兼容绝对路径；编译执行外部目录回归 |
| 参考文件、内嵌 blob 和 manifest 缺少交叉检查 | 错误组合的参考可能被使用 | 校验 blob SHA/长度、内嵌 golden 字节、参考 logits 与 top5 |
| host 缺少超时、失败报告与执行期间输入稳定性检查 | 检查挂住或证据不完整 | 加入 timeout、失败 JSON 与前后 SHA 一致性 |
| Graph.emit 的 zip 会静默截断 ABI 参数 | 参数数目不一致时校验不完整 | 显式检查参数数量和重复输出名，加入回归 |

算子源码按 10 类逐类 review；未发现已被实际验证证实的 kernel 计算错误。
43 个卷积/linear 按真实维度标量循环，没有 BM/BN/BK 矩阵分块。另按用户要求，
把两项 7×7 mean 的 BLOCK 从 128 调为 32，使分块不超过实际 SPATIAL=49；
仅改变这两项未上板算子的配置，保留原算法和 ViT 配置。BN 融合、SE、残差、池化、分类头均由真实
Triton adapter 串联执行，host 验证没有用 C oracle 替代图中的算子。

## 本地验证结果

- `python -m unittest discover -s tools -p 'test_*.py' -v`：20 项通过。
- 88/88 算子完成 TTIR→BOSCAME→LLVM→RV64 汇编/对象构建。
- 88/88 NH BIN 构建及 ELF 审计通过，审计 SHA 与当前 ELF 一致。
- 88/88 实际 Triton LLVM IR + 原 adapter + 独立 C oracle 的 host 检查通过。
- 真实预训练模型 host：121 对 BEGIN/END 完整，`checked=1000 errors=0`；
  logits `mismatch_count=0`，最大绝对误差 `7.510185241699219e-06`；
  top5 为 `[904,905,753,858,807]`，与 timm 一致。
- 模型 NH BIN 已构建并通过两项 ELF 审计：无 AME、无 RVV，entry 为 `0x80000000`。
- 导出测试使用 timm 1.0.22、torch 2.10.0+cpu、torchvision 0.25.0+cpu；
  输入为默认确定性 tensor，BN 融合误差 `7.62939453125e-06`。

完整模型 activations 为 2,007,040 B；常量 blob 为 10,753,952 B；NH BIN 为
14,948,256 B。权重与临时 Python 依赖均保存在忽略的 build 目录，没有改动
现有 Python 环境或将权重纳入源码。

证据均在本例的忽略目录中：

- `build/review-tests-final.log`
- `build/full-asm.log`、`build/full-bin-final.log`、`build/bin-audit-summary.json`
- `build/host-check-review/summary.json`
- `build/model/host-model-report.json`、`build/model/host/run.log`
- `build/model/elf-audit.json`、`build/model/nh-audit.json`

本地没有可用的 QEMU/RISC-V Linux GCC，未执行 QEMU 检查。Host 数值校验与
ELF 审计不能代替实板运行，也不能用 host ticks 作为 FPGA cycle。

## 计时与后续板测

整模型每条 END 的 cycles 包含 descriptor 构造、adapter 检查和完整 grid，
不含 BEGIN/END UART 打印。整图 cycles 包含逐调用打印，不含最终 logits 校验。
单算子 cycles 测量完整 typed adapter，含最终同步，不含初始化和独立 C oracle。

本轮板测按用户要求使用 fpga112 已有 scheduler，逐个执行 88 个单算子 BIN，
CPU 释放复位后 60 秒仍未退出即记录超时/hang；上传与 DDR 装载不计入这 60 秒。
若 END 已打印而随后超时，保留周期并单列为后置 oracle/报告阶段超时，不能
把这种情况当作完整数值验收通过。板测结果另存于
`build/fpga112-operators-review/` 和 `build/fpga112-operators-compact/`，
合并核验及周期表见 `build/fpga112-operators-complete/REPORT.md`。

## 首两批实板结果（配额释放前）

两批均通过 fpga112 的 scheduler 执行，实际分配 local FPGA4。
前 12 个原镜像和后 11 个紧凑镜像共 **23/88 完整通过，0 超时/hang，0 数值失败**。
余下 65 项未执行，不能报告为通过或 hang。两批分别在下一个任务的 admission
阶段因 `resident_storage_quota_exceeded` 停止；两项被取消的任务均
`hardware_started=false`，CPU 没有释放复位。

紧凑版只改变 linker 的 GOT 放置；当时构建的 88 个 kernel.o 和 adapter.c 与第一批构建的
SHA 完全一致，BIN 从 17,825,840 B 降为 5,680–7,344 B。原镜像、UART 和任务证据
均保留，未用新镜像覆盖旧结果。服务器每任务还有额外的保留数据占用，
紧凑镜像仍不足以让余下任务进入固定存储预留。没有修改服务器配置或清理
他人的任务。随后管理员释放并扩大配额，余下 65 项的补测结果见下文。

逐项 cycles、完整退出状态、镜像 SHA 与任务 ID，见
`build/fpga112-operators-complete/REPORT.md` 和 `cycles.csv`；
这 23 项均完整数值验证、UART 协议和调度器终态通过。

## 分块参数收敛

两项 7×7 mean 的 `BLOCK=32` 配置完成独立 host oracle 2/2 验证，真实模型
重新运行 121 次调用：1000 logits 无超差，最大绝对误差
`6.67572021484375e-06`，top5 仍为 `[904,905,753,858,807]`。
增加回归保证所有 mean 的 BLOCK 为正数、2 的幂且不超过实际 SPATIAL。

新配置的构建与验证保存在 `build/kernels-reduction-fit/`、
`build/host-check-reduction-fit/` 和 `build/model-reduction-fit/`。
原模型验证和两批实板镜像均保持不变。两项 BLOCK32 mean 已在配额恢复后的
补测中通过；首两批 23 个已通过项不包含 mean，其 FPGA cycles 不受此调整影响。

最终自定义 OUT 构建已通过：`build/model-reduction-fit/mobilenet.bin` 完成
两项 ELF 审计（36,157 条指令，0 AME、0 RVV），20 项离线回归全部通过。
完整模型已完成 host 与 FPGA 验证；全部 88 个独立算子也已通过 FPGA 验证。

## 配额恢复后的补测

2026-11-03（UTC）通过 fpga112 scheduler 补测剩余 65 项，没有重复前23项。
新批次 65/65 完整通过，三批合计 **88/88 PASS、0 hang、0 数值失败**。
所有实际执行项均为 `completed / exited / app_exit_code=0 / exit_code=0`；
每项完整 BEGIN/END、精确 checked、errors=0，并校验 DDR readback 和 reset。
执行上限为 CPU 释放复位后60秒，排除 DDR装载；每个算子只测一次。

本次 scheduler 实际分配 local FPGA4 和 FPGA6，两项7×7池化采用 BLOCK32新版：

| 算子 | cycles | task ID |
|---|---:|---|
| mean_hw_c288_h7_w7 | 118600 | 6656513601c6404487f43295ffd14ff2 |
| mean_hw_c576_h7_w7 | 235307 | d5e06b9fb32f4a22be578f0b794d3e50 |

新原始证据在 `build/fpga112-operators-resumed-20261103/`；
三批合并及独立核验见 `build/fpga112-operators-complete/REPORT.md`、
`cycles.csv` 和 `verified-summary.json`。前两次 admission 阶段的配额取消
仍保留为历史 attempt，均未启动CPU，不计入 hang或算子失败。

本客户端的65项任务均已完成。最终服务快照中有其他任务使用 FPGA6，
没有干预该任务；全服务活动任务数与本客户端未结束任务数分别报告。

## 整模型 FPGA 验证

最终 BLOCK32 镜像在 fpga112 / local FPGA4 首跑通过，任务
`e741fcd891be408981dc5d644326cdd4`，BIN SHA-256 为
`e5f7be89a9cb34149f4ea62f6a0433cf9f4b5e70dbf57997ae79cc25b2b05a95`。
调度器严格终态 `completed / exited / app_exit_code=0 / exit_code=0`，
DDR readback 与 reset 均确认。默认确定性输入，batch=1，float32 NH 标量路径。

- 121/121 次调用均有顺序完整的 BEGIN/END。
- `kernel_returned=1 validation_executed=1 checked=1000 errors=0`。
- top1=904，top5=`[904,905,753,858,807]`，与 timm 参考一致。
- 最大绝对误差 `6.67572021484375e-06`，板上比较容差 atol=0.002、rtol=0.001。
- logits FNV-1a-64 为 `0x162ef27ad01e3fa8`，与真实 Triton host 输出相同。
- 整图 **330,772,222 cycles**，包含逐调用打印，不包含最终 logits 校验。
- 算子累计 **314,577,201 cycles**，为整图内121次实际调用周期之和，
  排除各次BEGIN/END打印，包含descriptor构造和typed adapter/grid。

原始 task/result/UART、独立验证及逐调用周期在
`build/fpga112-model-20261103/`。本次预算600秒；所有调用正常完成，未发生hang。
本任务已结束，自有18764转发已关闭；其他客户端的活动任务未干预。
