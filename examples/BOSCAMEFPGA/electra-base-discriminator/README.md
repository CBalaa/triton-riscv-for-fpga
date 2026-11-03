# ELECTRA-base-discriminator / BOSCAME

参考 `../vit-base-patch16-224` 适配
[`google/electra-base-discriminator`](https://hf-mirror.com/google/electra-base-discriminator)。
每个独立算子保留真实 `@triton.jit`、字面量 `KERNEL_CONFIG`、
`int launch(void)` 和 `makefile`；沿用 typed MemRef → unranked adapter、
完整三维 grid dispatch、64 字节对齐及非别名检查、分阶段 AOT 编译、
NH/E6 boot/cache/UART/DMA runtime 和链接脚本。

当前验证状态：**32/32 独立算子回传验证通过；整模型执行正常，emitted 实现
一致性 PASS，独立 PyTorch W8A8 数值对照 FAIL。整模型数值验证尚未通过。**
独立算子最新结果见 [RETURNED_OUTPUT_RESULTS.md](RETURNED_OUTPUT_RESULTS.md)，
旧 60 秒期限测试另存于 [BOARD_RESULTS.md](BOARD_RESULTS.md)。

目录结构：

```text
<operator>/         kernel.py + launch.c + makefile，每个算子独立构建
model/             固定上游配置、revision 和资产清单
model_runtime/     完整 C 调度、typed ABI 和 NH/E6 模型入口
tools/             导出、构建、权重/输入准备、回传与本地验证工具
common.mk          共用 AOT 编译规则
board.mk/board.ld  独立算子 NH/E6 连接规则与布局
board_model.ld     完整模型连接布局
assets/            本地下载的 checkpoint/tokenizer（忽略）
build/             编译产物、权重、回传 BIN 和原始证据（忽略）
```

[工具索引](tools/README.md) 列出保留脚本的职责。Git 仅保存源码、配置和结果摘要；
报告中指向 `build/` 的路径为本地证据位置，不随源码上传，整理时也未删除这些证据。

固定 batch=1、sequence=128、eval。原配置为 12 层、hidden/embedding=768、
12 heads × 64、FFN=3072、vocabulary=30522、max positions=512、
LayerNorm epsilon=1e-12、GELU。输出是 `[128]` 逐 token 替换判别原始 logits，
包括 padding 位置；它不是句子分类或生成模型。Dropout 在推理时关闭。

32 个算子目录包含 embedding、head pack/unpack、masked softmax、post-LN、
残差、FFN、discriminator dense/GELU/prediction，以及 W8A8 数值原语。
矩阵使用物理 `B[N,K]`、overwrite `C=A@B.T`，权重无需从 HF 的 `[out,in]`
布局转置。逐行量化使用对称 signed INT8、ties away from zero、零行 scale=1。
QK 在 `dequantize_128x128` 应用一次 `0.125`，PV 使用运行时 V 转置量化，
不另加缩放。Embedding、LayerNorm、bias、GELU、softmax、残差保持 FP32。

`cases.py` 强制 `BM≤M、BN≤N、BK≤K`。QK 的 BK=64；输出宽度为 1 的
`matmul_i8_128x1x768` 用 BM=BN=1、BK=128 的精确 scalar INT32 累加，
避开 AME 的最小 N 限制。其他矩阵保留 AME `tl.dot`。独立的
`quantize_transpose_128x128` 是额外布局原语，不参与默认模型。
这里的 W8A8 是明确的量化适配，任务精度尚未评估。

完整模型已在 fpga112 执行：12 层及 discriminator 共 1,571 次真实调用，
回传 embedding、12 层 hidden 和 logits，全部 14 个 BIN 与本地 emitted LLVM
加独立精确整数 dot 参考逐字节一致；模型耗时 3,168,993,218 cycles。
独立 PyTorch W8A8 对照超过预设容差，最终 logits 最大差 0.867895，
因此整模型数值验证仍为 FAIL；这不是只拿 INT8 输出与 FP32 比较而产生的结论。
结果、实际输入、任务和复现命令见 [MODEL_BOARD_RESULTS.md](MODEL_BOARD_RESULTS.md)。

`model_runtime/model.c` 的 `electra_infer()` 串联完整 12 层及 discriminator，
共 1,645 次 typed 算子调用。调用者提供权重指针、INT32 input/type/mask、
16 MiB 可复用工作区和输出。输入 mask 必须为二值且非全零，token ID/type
在进入 kernel 前验证。每个张量连续、64 字节对齐、遵守非别名约定。
完整 checkpoint 的权重及模型工作区单独管理，不占独立 launcher 的 arena。

从仓库根目录执行：

```bash
conda activate boscame
export BUDDY_DIR="$PWD/buddy-mlir"
export BUDDY_ROOT="$PWD/buddy-mlir"
source scripts/triton-riscv-env.sh
ELECTRA=examples/BOSCAMEFPGA/electra-base-discriminator

make -C "$ELECTRA" hostcheck       # 真实 Triton CPU interpreter 数值
make -C "$ELECTRA" launchcheck     # 独立 C launch/descriptor/grid ABI
make -C "$ELECTRA" modelcheck      # 完整随机权重主机 HF 对照
make -C "$ELECTRA" model-abicheck  # 实际 C 调度 vs 完整主机调用序列
make -C "$ELECTRA" asm             # Contract→BOSCAME→LLVM→受限 RISC-V 对象
make -C "$ELECTRA" simcheck CASES='matmul_i8_128x1x768 masked_softmax_12x128x128'
make -C "$ELECTRA" bin             # 离线 ELF/bin 链接和 NR ELF 审计

make -C "$ELECTRA" assets
python "$ELECTRA/tools/model_reference.py" --assets "$ELECTRA/assets" --w8a8
make -C "$ELECTRA" weights         # 生成64字节对齐FP32权重blob及C binder
python "$ELECTRA/tools/prepare_inputs.py" --text 'The quick brown fox jumps.'

# 完整模型：先完成 assets、weights 和 asm，再离线连接并建立本地实现参考。
make -C "$ELECTRA" model-bin
make -C "$ELECTRA" model-host
```

可选编译阶段为 `ttir/boscame/llvm/asm`，`CASES='...'` 选择子集，
`KERNELS=/absolute/path` 重定向产物。每个独立算子默认构建 `bin`。
`launchcheck` 和 `model-abicheck` 使用 mock，只建立 ABI/连接证据；
`hostcheck` 执行真实 Triton 源码；`simcheck` 执行 emitted BOSCAME，
检查完整 grid、poisoned SPM、DDR 越界、guard 和输入未改动。
`model_reference.py` 用 PyTorch 计算完整 FP32/W8A8 契约，不运行 emitted 整模。

板测验证采用 scheduler 回传 BIN 后在本地运行原始 C oracle，板端只执行
fixture初始化、真实算子及各轮独立输出快照。数值通过需同时满足真实任务成功、
下载元数据/长度/SHA核对以及本地oracle通过。构建后运行：

```bash
python examples/BOSCAMEFPGA/tools/run_returned_outputs.py \
  --model electra-base-discriminator --timeout 180
```

可重复传入`--case`选择算子；独立构建目录可用`--kernels`指定。
回传结果与判据见 [RETURNED_OUTPUT_RESULTS.md](RETURNED_OUTPUT_RESULTS.md)，
旧 60 秒记录及发生超时的阶段见 [BOARD_RESULTS.md](BOARD_RESULTS.md)。
独立算子与完整12层模型的证据分别记录，整模型结果见上文链接。

完整模型使用 `tools/run_model_board.py`，同样先回传全部 BIN，再在本地分别运行
独立 Torch 和 emitted 实现参考；任一检查失败，runner 返回非零。
用户选定 fpga112 并建立 HTTP 转发后，可从仓库根目录运行：

```bash
export FPGA_SCHEDULER_CLIENT="$PWD/../fpga-scheduler/.venv/bin/fpga-scheduler"
export FPGA_SCHEDULER_URL=http://127.0.0.1:18762
export FPGA_SCHEDULER_TOKEN_FILE="$PWD/.agent/secrets/fpga112.token"
python "$ELECTRA/tools/run_model_board.py" \
  --output "$ELECTRA/build/model-board-runs/new-run" --timeout 1800
```

三个连接设置也可用 `--client/--endpoint/--token-file` 显式指定。
此 runner 的服务器标签和默认连接配置为 fpga112；它不会建立 SSH 隧道。
原始失败结果仍保留在历史证据中；本次整理未重跑模型或修改容差。

资产固定到 revision `1ae76a97c7e84a4e640876a07453fccd636f0667`，
下载后记录 SHA256。`tools/prepare_weights.py` 写出 `weights.bin`、
`weights.h` 和带原始 checkpoint 名称/shape/offset/hash 的 manifest，
通过生成的 `electra_bind_weights()` 绑定 `ElectraWeights`。
Tokenizer 在主机运行，`tools/prepare_inputs.py` 输出三个 little-endian INT32 文件。

`board.ld` 保留 NH `0x80000000`、E6 `0x80010000` 入口及 NOLOAD/DMA/PMP
符号约定。完整 `[30522,768]` FP32 词表约 89.4 MiB，独立算子 arena
因此扩大到 112 MiB，离线链接区为 128 MiB，E6 stack 起于 `0x87800000`。
该地址布局已离线链接；独立算子的实际执行结果及超时阶段见板测报告。
完整模型使用专用 `board_model.ld`：固定矩阵权重按相同契约预量化，
FP32 embedding 保留完整词表，16 MiB scratch 与输出快照单独分配。
默认动态权重 C API 和其 1,645 次 ABI 路由仍保留；整模型板端预量化版本
省去 74 次固定权重量化。验证范围见 [VALIDATION.md](VALIDATION.md)。
