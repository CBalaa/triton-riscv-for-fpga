# Dequantize 上板数值问题调查 — 2026-10-27

四个原始镜像均已上板复现数值失败。直接原因定位到 **Row scale 的 E6 scalar store → RVV load 交接没有得到预期值**：单元素 `vector<1xf32>` 拷贝在 LLVM 后端被标量化为 SPM `sw`，随后 RVV 读取并广播错误的缩放因子。将这个写入位置改成显式 VL=1 的 RVV store 后，四个尺寸的真实 DDR 输出共 **908,008 个 FP32 元素全部逐位一致**，没有放宽原始容限。

这是已定位的软件访问路径和已验证的规避方案。不能由此断言“硬件不支持 scalar SPM”：本地 patch7 发版说明明确允许 CPU 对 SPM 进行子行访问，并说明不同 master 之间没有硬件 cache coherence。目前还未唯一区分实际缓存策略、子行事务/初始化交互或硬件响应问题，生产编译器未修改。

## 原始板测与向量写入对照

| 算子 | 原始板端 oracle：错误数 / 检查数 | 原始 E6 cycles | vector-store：错误数 / 检查数 | vector-store E6 cycles | 逐位差异 / 最大绝对误差 |
|---|---:|---:|---:|---:|---:|
| dequantize_1x1000 | 999 / 1,000 | 40,500 | 0 / 1,000 | 40,469 | 0 / 0 |
| dequantize_196x768 | 150,381 / 150,528 | 5,794,922 | 0 / 150,528 | 5,805,675 | 0 / 0 |
| dequantize_197x768 | 151,148 / 151,296 | 5,867,661 | 0 / 151,296 | 5,873,259 | 0 / 0 |
| dequantize_197x3072 | 604,592 / 605,184 | 23,303,610 | 0 / 605,184 | 23,338,208 | 0 / 0 |

原始四项均 `execution_status=exited, app_exit_code=1`，真实 kernel 已返回、板端 oracle 遍历完全部输出，没有 timeout/trap。错误数恰好等于参考值非零元素数，但仅凭计数不能反推 baseline 输出均为零；baseline 没有导出 C。

vector-store 四项均 `exited, app_exit_code=0`，实际 node `local` / FPGA4。Host 核验输出长度、服务端 artifact SHA256、镜像 SHA256、完整 invocation/cycle 记录以及前后各 64-byte guards，全部通过。再将下载的 C.bin 交给**未修改的原始 launch.c + support.c** 执行全部比较，四项也全部通过；1×1000 人为损坏一个元素时，原始 C oracle 正确报出 1 项错误。

每个尺寸本次为 1 轮既有输入：`X[k]=((k%1023)-511)*7919`，`Row[i]=(i%17+1)/256`，`Column[j]=(j%31+1)/512`。FP32 参考顺序为 `(float(X)*Row)*Column`，`atol=1e-5, rtol=1e-6`。结论覆盖这四组完整输出，不扩展为任意输入或完整 ViT 模型已通过。

## 错误 scale 精确解释输出

相同 DDR 导出诊断 harness 下，1×1000 使用**未经改动的原始 kernel.nr.s**：

- Task `131b8954c20240f499dd0883b88b4dd9`，local / FPGA4，40,380 cycles。
- `SPM1 + 512 = 0x20400200` 应写入 row scale `1/256`，FP32 bits `0x3b800000`。
- kernel 返回后以 RVV 读取该槽位，实际为 `1/64`，bits `0x3c800000`。
- C[0] 实际 `-123.49270629882812`，期望 `-30.87317657470703`，正好为 4 倍。
- 用 snapshot 的 `1/64` 计算全部 1,000 项，**逐位差异为 0**；生成字节的 SHA256 与下载 C.bin 相同。
- 原始参考下错误 999 项，最大绝对误差 `2702.6416015625`，guards 正常。

vector-store 对照的该槽位恢复为 `0x3b800000`，全部 1,000 项正确。196 行和 197 行最终 scale snapshot 分别为 `0x3d100000`（9/256）、`0x3d200000`（10/256），均符合最后一行。未正确更新的 SPM 状态可解释不同执行读到零或其他旧值；未采集的历史运行不能当作已证实的旧值来源。

## 从源码到最终指令

Triton 数学表达式正确：`tl.store(Out + index, (value * row) * column, valid)`。以 1×1000 为例：

1. `build/kernels/dequantize_1x1000/kernel.boscame.mlir:47–51`：DDR `vector<1xf32>` load → SPM1 vector store → SPM1 vector load/extract。
2. `kernel.ll:91–99` 仍为 `load/store <1 x float>`，目标指针为 `inttoptr (i64 541065728 to ptr)`，即 `0x20400200`。
3. `kernel.nr.s:81–96` 拷贝被后端变为 scalar `lw` + `sw`，消费者仍为 RVV：

```asm
lw          a4, 0(a4)          # DDR Row[row] 的 FP32 bits
sw          a4, 512(a5)        # a5=0x20400000，E6 scalar 写 SPM1
...
vsetivli    zero, 1, e32, mf2, ta, ma
fence       rw, rw
vle32.v     v8, (a2)           # a2=0x20400200，RVV 读同一 scale
fence       rw, rw
vfmv.f.s    fa5, v8
```

只将这个 `sw` 替换为以下片段，保留 DDR scalar load、全部计算、列 scale、网格与 DMA：

```asm
addi        a5, a5, 512
vsetivli    zero, 1, e32, m1, ta, ma
vmv.v.x     v16, a4
fence       rw, rw
vse32.v     v16, (a5)
fence       rw, rw
addi        a5, a5, -512
```

`v16` 在原 kernel 中未使用，下一段消费者自行重设 vtype。1×1000 在 full/tail 两个静态分支各替换一次，其他尺寸各一处；每项 `kernel.diff` 保存精确差异。所有诊断 ELF 均通过本地 `check_nr_elf.py` 审计；数值结论以板测与 host 比较为准。

## 补充对照与边界

为分开顺序屏障和写策略，对 1×1000 另做两项独立诊断。调度器选择到 local / FPGA5，因此它们不是 FPGA4 原始/向量对照的同板重复。

| 变体 | Task ID | 实际结果 |
|---|---|---|
| scalar-fence：原 `sw` 前后加 `fence iorw,iorw` | `14fde5e98d244dd68fb0c7111e3d4848` | 正常返回，40,201 cycles；host 999/1,000 项错误，scale snapshot 为 0，guards 正常 |
| scalar-write-policy：保存 CSR 0xbc0，切写策略后 `sw`，fence 后恢复原策略 | `3750dd586bc040e8a96b44f375cb5281` | 120 秒期限到达，已有应用 UART，记为 TIMEOUT；kernel 未返回，没有完整 cycles 或 C.bin |

第一项说明这次 FPGA5 运行仅加 fence 未消除错误。第二项没有指令级进度探针，不能宣称具体某条 `sw`/CSR/fence 卡住，也不能把 timeout 当数值 FAIL/PASS，或宣布“切写策略即可修复”。未继续扩大板测或修改远端设施。

`bosc_ame_dma_wait` 在 STATUS 读取前切到读策略；最终指令在它返回后没有为 singleton scalar store 显式恢复写策略，这是需要检查的上下文。它是候选解释，尚无总线事务/缓存状态证据把它定为唯一原因。现有 snapshot + 成对替换已把数值故障收敛到 scalar→RVV 的 scale 交接。

## 原有检查的缺口

- `BOSCAMEPlanEmit.cpp:239–257` 按精确 lane 数生成 vector load/store，此例 lane 数为 1。
- `SpmDmaCheck.cpp:838–845` 在 **MLIR 软件契约** 中拒绝 scalar memref 访问 SPM，但接受 typed vector。
- `LegalizeForLLVMExport.cpp:670–674` 在 SPM 标注存在时校验；随后普通 LLVM fixed vector 可在指令选择时标量化。最终 `sw` 不再携带原 SPM 契约信息。这是软件访问约束没有保持到最终指令的缺口，不是硬件禁止 scalar 的证明。
- strict simulator 执行 `kernel.boscame.mlir`，没有执行 LLVM 标量化后的指令或模拟真实 E6/RVV 缓存交接。
- `check_nr_elf.py` 审计 NR AME/RVV 编码，不能证明标量 SPM 地址及跨 master 可见性正确。

本地 `.tmp/patch7-1db13c0.md` 第 2.5 节允许 E6/CPU 访问 SPM 子行，建议先 DMA 初始化完整 64-byte line，明确 SPM 无硬件 cache coherence。不能把 DMA 的 64-byte 对齐规则套给自然对齐的 CPU 4-byte store。task API 未提供当前 bitstream/RuntimeDB 版本或实测时钟；patch7 文档与当前硬件版本未核实相同。

## 修复方向

近期可复用已板测通过的向量写入方案，确保 singleton SPM 写入在**最终机器指令**中保持 RVV，或在合法化阶段直接让已加载的 scale 留在寄存器中供 broadcast 使用。正式实现需遵守访存/调度契约，不能简单扩大访问宽度而越过真实边界。

长期应补 final-assembly/目标 intrinsic 回归，覆盖长度 1 的 DDR→SPM 与 SPM→DDR 拷贝，并加入 scalar/RVV/DMA 交接可见性测试。若保留 scalar SPM 路径，需要按实际硬件契约明确 cache policy、必要维护与 line 初始化，分别做读/写对照。当前 vector-store 是隔离诊断镜像；**原始 kernel、launcher、Makefile、编译器及远端 runner/bitstream 未由本调查修复或替换**。

## 证据与复核

证据根目录：`build/board-runs/dequantize-investigation-20261027/`。

- `summary.json`：baseline、DDR 数值结果、补充 TIMEOUT、task ID、hash、cycles。
- `baseline/<case>/`：原始提交 bin、完整 console/UART、task.json、结果。
- `ddr/<variant>/<case>/`：独立 ELF/bin、kernel.s、build.json、artifact 元数据、C.bin、Before/After/D.bin、两套 host 检查。
- `scale-reconstruction.json`：错误 scale 精确解释全部输出的记录。
- `source-snapshots/` 与 `source-manifest.json`：Triton、MLIR、LLVM、最终汇编、runtime/checker 源码快照与 SHA256。

| 算子 | 原始 Task ID | vector-store Task ID |
|---|---|---|
| dequantize_1x1000 | `d8fb15f208944bffa2389878ed59da41` | `5811a29b3d1e4fa786dc7a00736c4be9` |
| dequantize_196x768 | `64652919ac354ee1876ba116613ee62c` | `f2dc2b55a47141fda4f6853e4fffceeb` |
| dequantize_197x768 | `400e236a0be144b3acfe20d8bb7c3422` | `1086d65af8c942ce9f6252de8c4a326f` |
| dequantize_197x3072 | `308c2dc6983b4d94852721657e0be957` | `12c10af363c04bec9e7767c08edbbac8` |

全部执行经既有 `127.0.0.1:18765` 隧道和调度队列；DDR 输出按 scheduler 的 `examples/ddr_output/README.md` 流程导出、下载、host 校验。诊断 UART 中 `validation_executed=0`，其 `PASS/app_exit_code=0` **仅表示执行/导出成功**；本报告数值 PASS 来自下载字节的完整比较。

复核已有输出，不启动板测：

```bash
conda activate boscame
VIT=examples/BOSCAMEFPGA/vit-base-patch16-224
D=$VIT/build/board-runs/dequantize-investigation-20261027
python $VIT/tools/dequantize_board_diagnose.py \
  --case dequantize_1x1000 --case dequantize_196x768 \
  --case dequantize_197x768 --case dequantize_197x3072 \
  --variant vector-store --scheduler-root /home/chh/gitprojects/fpga-scheduler \
  --report-dir $D/ddr --collect
python $D/check_original_c.py
```

诊断工具非 collect 模式会构建并提交；已有 task.json 时拒绝覆盖，有意重测需新 report-dir。选择 scheduler 源码 SDK 是为了 artifact 支持，没有修改用户维护的 run_bin.py。

最初两个 DDR prepare 任务 `b7a714e1c9694ca08aee004e79ca1afc` / `f8fd9e8e3ae7443b85c69f588b20b0ad` 因 resident 配额失败，硬件未启动，不计作算子错误。仅将本调查四个已结束 baseline 任务目录移至远端 `~/fpga-resident-deploy/backups/dequantize-investigation-20261027/`；没有删除证据、操作其他用户任务或更改服务配置，失败 JSON 保留。

Cycles 为 E6 `rdcycle` 包围完整 typed adapter，含 grid dispatch、DMA 与最终 fence，排除初始化、oracle、snapshot、host 校验与 UART。不同 harness/状态下的 cycle 差异不作为性能优化结论；没有从硬编码 UART 频率换算秒数。
