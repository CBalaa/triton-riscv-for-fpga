# Quantize 上板超时调查 — 2026-10-27

已确认 1×768 的原始 kernel 在首次调用的 Scale 输出路径停滞：指令探针把停点定位到从 SPM 取 Scale 的 scalar `lw` 区间，量化计算主体已经走完。绕过该 SPM 中转并显式设置 DDR 写策略的独立诊断镜像可以返回，下载的 Q/Scale 由 host 全量校验。六个尺寸都存在相同的静态访问路径，但没有对六项分别完成指令探针。

用户补充的 [patch7 发版说明](../../../.tmp/patch7-1db13c0.md) 第 2.5 节明确允许 E6/CPU 对 SPM 做子行访问。因此，不能把此停点解释为“SPM 硬件不支持 scalar load”，也不能仅凭 LLVM 将单元素 vector 拷贝标量化就判定编译器错误。当前确认的是发生停滞的软件访问位置；具体是缓存属性、子行/整行访问交互，还是当前硬件的事务响应问题，尚无证据唯一区分。

已按用户指示停止：本轮剩余三个排队任务均已取消且 `hardware_started=false`，执行脚本均已退出。原始算子、launcher、编译器未作修复；下列诊断均为隔离的临时镜像。等待用户下一步指示，不继续提交或修改。

原始镜像和诊断镜像分别记录。调度器状态 `timeout` 且已有应用 UART，在采集工具中记为 `TIMEOUT`；本文的“卡住”描述指令级诊断结论。

## 原始镜像与诊断镜像

| 算子 | 原始镜像本轮状态 | 同 SHA256 的前次原始镜像 | 诊断镜像 E6 cycles（两轮） | Host 校验 | 原始 scalar SPM load PC（静态） |
|---|---|---|---:|---|---|
| quantize_1x768 | TIMEOUT | TIMEOUT | 35,260, 35,060 | PASS / 原始 C PASS | `0x80011396` |
| quantize_196x768 | TIMEOUT | TIMEOUT | 6,776,954, 6,776,673 | PASS / 原始 C PASS | `0x8001146c` |
| quantize_197x768 | TIMEOUT | TIMEOUT | 已取消，未执行 | 未校验 | `0x80011472` |
| quantize_197x3072 | TIMEOUT | TIMEOUT | 未提交，已停止 | 未校验 | `0x8001145e` |
| quantize_768x768 | 已取消，未执行 | TIMEOUT | 未提交，已停止 | 未校验 | `0x80011460` |
| quantize_1000x768 | 未提交，已停止 | TIMEOUT | 未提交，已停止 | 未校验 | `0x80011472` |

前次证据位于 `build/board-runs/20261027-suite/`；已逐项重算当前 `build/kernels/<case>/board/<case>.bin` 的 SHA256，与前次记录一致。前次 TIMEOUT 不计作本轮重新执行。

## 精确停点与证据

1. `probe-stage` 在向量量化循环结束后超时。此时 DDR→SPM 的输入 DMA、abs/max、除法与舍入循环都已经走完；尚未进入原始 launcher 的比较循环。
2. `probe-fine/retry-console.log` 出现 `0xd074`，证明单个 scale 的 `vse32.v` 以及后续 fence 已返回。不能把停滞归因于 `mf2` 或 VL=1 store。
3. `probe-memory/uart.log` 最后停在 `0xd081`（scalar SPM load 前），没有执行后标记 `0xd082`。其原始指令为 `lw a4, 0(a4)`；1×768 对应原始 ELF `0x80011396`，首行读取物理地址 `0x20401000`。没有收到 trap 报告。这里以插桩前后标记和反汇编定位区间，没有硬件 PC 采样；插桩镜像的指令地址也与原始 ELF 不同。

```asm
lui a3, 132097          # 0x20401000, SPM1 + 4096
add a4, a0, a3
lw  a4, 0(a4)          # 停点
add a5, a2, a0
sw  a4, 0(a5)          # DDR Scale[row]
```

这是上游 `vector<1xf32>` 的 Scale 拷贝在 LLVM/指令选择后被标量化的路径：`kernel.boscame.mlir` 中的单元素 `vector.load/store`，最终成为 SPM scalar `lw` + DDR scalar `sw`。六个 specialization 都含有同一条路径；3072 列的 Scale 中转地址为 SPM1 + 16384。精确 ELF/源码快照和各自 PC 见 `instruction-map.json`、`source-snapshots/`。

## 与 SPM 发版约定的对照

- `0x20401000`（以及 3072 列时的 `0x20404000`）在 SPM1 的合法区间内，4-byte load 自然对齐。发版要求 64-byte 对齐及长度为 64 的倍数的是 DMA，不能把这一要求直接套到 CPU `lw`。
- 发版明确 E6、RVV、AME 具有独立访问路径，并明确 SPM 无硬件 cache coherence。本例先由 RVV 以 VL=1 写入 4-byte Scale，再由 E6 scalar `lw` 读回；`fence` 不能等同于跨 master 的 cache clean/invalidate。
- 原始 wrapper 在 kernel 入口设置 CSR `0xbc0=0x0f0f0f0f0f0f0f0f`；在这条 `lw` 前没有另一次策略切换或 cache maintenance。这里是静态记录，不能仅据 CSR 常量推断当前硬件内部实际发出的 AXI 事务。
- 发版建议先用 DMA 初始化完整 64-byte line，再进行子行访问；Scale 的这条路径直接 RVV 写 4 byte，没有看到针对该 Scale line 的 DMA 整行初始化。这是与建议做法的差异，尚未证明它造成此次停滞。

因此，本次足以排除“120 秒主要消耗在量化循环或 host/板端校验”，并定位到 Scale 的 RVV→SPM→E6 标量读回交接；尚不足以判定是哪段 RTL 或哪项缓存/初始化约定导致事务不完成。发版文件描述 patch7，任务 API 没有给出当前 bitstream 版本，也不能把两者直接视为已核实相同。

## 诊断对照和 host 校验

`direct-scale-export` 保留量化计算主体和 Q 的 DMA 回写；Scale 已在浮点寄存器中，改为切换 CSR `0xbc0` 至平台的写策略 `0x0f0f0f0f0f0f0f0f` 后直接写 DDR，跳过 Scale 的 SPM 中转。这里只构建独立诊断 ELF/bin，没有替换生产 kernel 或修改编译器。

每项仍使用原始 launch.c 的两轮输入生成代码。为了保持两轮输出并适配服务默认的前 4 MiB 导出窗口，诊断 launcher 把 Q0/S0/Q1/S1 放在 arena 前部；输入初始化放在其后。E6 完成 DMA/fence 后由 NH 发出 `@@FPGA_OUTPUT`。通过 SDK 下载真实的 `Q0.bin`、`S0.bin`、`Q1.bin`、`S1.bin`，检查下载长度与 SHA256。

Host 首先按 FP32 逐步运算计算参考值，然后把下载字节交给未修改的原始 `launch.c` + `support.c` 再执行全部 oracle/比较。INT8 要求逐位相同，Scale 保持原始 `atol=rtol=1e-6`；每项总比较数为 `2*(M*K+M)`。1×768 另做单字节损坏检查，原始 C 校验正确拒绝。板端临时成功文本只表示执行/导出成功，数值 PASS 以这两项 host 检查为准。

停点以 `probe-memory` 的前后 mailbox 标记和最终反汇编为准。早期追加探针使用 `vmv.x.s`，未通过本地目标白名单约束且数值结果失败，已排除；保存在 `unsupported-vector-load-export` 目录。发版说明称支持全部 RVV 1.0，不能由本地白名单推断该指令在硬件上不受支持，也不能把该探针的错误值算作原始算子的第二个问题。改用 `vfmv.f.s` / `fmv.x.w` 的后续单 load 对照镜像已通过静态审计，但按用户指示在排队阶段取消，不能声称它已板测通过。

成功的 direct-scale 诊断同时绕过 SPM 中转并显式设置写策略，证明存在可正常执行并得到正确结果的量化计算路径；它不是只改变一条 `lw` 的严格单变量对照，不能据此宣布已找出唯一底层成因。

Cycles 来自 E6 `rdcycle` 包围完整 typed adapter，包括网格派发、DMA、最终 fence，排除输入初始化、host 校验和 UART。原始调用未返回，因此没有完整 cycles。UART 的 5.5296 MHz 为标注频率，平台未提供实测时钟或 bitstream 版本，不将墙钟或硬编码标签当作纯 kernel 测量。

## 平台与复现

全部任务经用户的 `../run_bin.py` / 调度队列，`timeout=120`；任务实际使用 node `local` / FPGA4。每次 task.json 保留 session、镜像 SHA256、DDR readback、reset 和 application_started 的实际证据，硬件版本未提供。

中途出现 `Input transfer or verification failed; hardware was not started`，只读远端 service.log 确认是 Resident storage quota exceeded。它发生在 prepare 阶段，不属于 kernel hang。已把已结束 quantize 任务冗余的 padded image/readback 两份文件移至远端 `~/fpga-quantize-evidence-20261027/`，逐文件核对 SHA256；日志、数据库、原上传镜像和输出 artifact 保留。可根据归档 manifest 原路移回；未修改服务配置、runner、Tcl、bitstream，也未绕过队列。

完整证据目录：`build/board-runs/quantize-diagnosis-20261027/`。原始镜像在 `original/` 和 `original-retry/`，诊断镜像在对应 `direct-scale-export-<case>/`。每项保存 task ID、bin/ELF、hash、完整 console/UART、task.json、输出 bytes 和 host 检查记录。

```bash
conda activate boscame
VIT=examples/BOSCAMEFPGA/vit-base-patch16-224
D=$VIT/build/board-runs/quantize-diagnosis-20261027
python "$D/build_export.py" quantize_1x768 direct-scale-export
PYTHONPATH=/home/chh/gitprojects/fpga-scheduler/src python "$D/run_export.py" "$D/direct-scale-export-quantize_1x768"
python "$D/check_original_c.py" "$D/direct-scale-export-quantize_1x768"
```

`PYTHONPATH` 仅选择已有的本地 SDK 版本以支持 DDR artifact 下载；没有改动用户维护的 run_bin.py。

## 任务关联

| 算子 | 原始任务 | 诊断任务 |
|---|---|---|
| quantize_1x768 | `335372cc705d44af9945650c8a7501be` | `59b5712605b94c2d89005f40a596ece4` |
| quantize_196x768 | `4901842e61614d23960294da719e3cab` | `15bd6972b07f476586becb23a0ca1410` |
| quantize_197x768 | `8563cdd248294b2dbf5c8ddf546dd297` | `93d65b0b54114c86bc7e7f1c50c7b593`（已取消） |
| quantize_197x3072 | `e45984024c0d46d1817034b8c0efbc61` | 未提交，已停止 |
| quantize_768x768 | `2421b4f0a2cd49c29d205eec2e2ffe9f`（已取消） | 未提交，已停止 |
| quantize_1000x768 | 未提交，已停止 | 未提交，已停止 |

阶段/指令探针的 task ID 保存在各探针目录的 task.json。平台拒绝和取消的任务也保留原记录，不计作实际 kernel 执行。

## 2026-10-28：存储规划 v2 镜像对照

本轮新构建的 `quantize_1x768` 镜像（SHA256 `526507f43a145c3af961657060004843cae81813339fdc257e749ec0a493e032`）在首次 kernel 调用期间停在 UART `E6 progress=0x1001`。任务 `5f09ab06d12c40f29daa52e41bae6c5b` 运行约 404.7 秒后被人工取消；没有 kernel cycle、调用返回或数值校验，因此只记为未定位的 kernel-call stall，不记作 scheduler timeout 或正式 HANG。

这份 v2 编译产物**没有复现旧版的 SPM scalar load 路径**：`kernel.planned.mlir` 报告两组 SPM high-water 均为 0；逻辑上的标量归约结果由 `kernel.boscame.mlir` 的 SSA 值传递，后续没有从 scratch memref 执行 `memref.load`；`kernel.nr.s` 中唯一的 `lw` 是从栈读取 ABI 参数，Scale 使用 `fsw` 写到 DDR descriptor，向量读写也是 DDR。对应计划 IR、发射 IR、NR 汇编、提交镜像及任务日志已归档在 `plan/validation-evidence/board/vit/quantize_1x768/`。

所以旧镜像在 `0x20401000` 的 SPM scalar load 停点不能解释本次 v2 任务。为了节省板卡时间，没有重新提交此例；当前 v2 stall 的具体原因仍未确定。


## 2026-10-28：本轮 SPM 复查与镜像布局对照

当前 SHA256 526507f43a145c3af961657060004843cae81813339fdc257e749ec0a493e032 镜像经用户 run_bin.py / 调度队列两次实际执行，均在首次 kernel 调用期间 120 秒 TIMEOUT。当前 plan 的两组 SPM high-water 均为 0；最终 fsw 写 DDR 0x80100c00，没有 SPM 标量 load/store。此前 2026-10-27 旧镜像的 0x20401000 标量 lw 结论不能解释本轮镜像。

隔离 .p2align 1→6 变体两次 PASS，原始 kernel 指令逐字节未变；仅在 kernel 前放 34 B padding 的变体也 PASS。更严格的对照保留所有函数 PC 和全部原始对象，仅把链接脚本副本的 .rodata 起点从 0x80011210 改为 0x80011238（.data 随后移动），同样 PASS。因此已经确认**镜像布局敏感**，不能据此认定函数对齐要求或具体取指/RTL 根因。

所有 PASS 均完成两次真实调用与原始 1538 项校验，零错误；原始镜像没有完整 cycle。仅移动 .rodata 的变体为 22,905 / 22,841 E6 cycles。生产源码、编译器及原始 bin 未修改。

完整任务 ID、镜像/hash、console/UART、各级 IR、ELF、反汇编、布局比较及复现脚本见 [本轮报告](build/board-runs/quantize-spm-investigation-20261028/REPORT.md)；已通过的最小链接布局变体为 build/board-runs/quantize-spm-investigation-20261028/controls/data-pad40/image.bin。全部任务使用调度器分配的 local/FPGA4，已核对镜像 SHA、DDR readback、reset 和应用 START 证据；硬件版本和实测频率未提供。
