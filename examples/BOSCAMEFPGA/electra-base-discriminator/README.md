# ELECTRA-base-discriminator / BOSCAME

参考 `../vit-base-patch16-224` 适配
[`google/electra-base-discriminator`](https://hf-mirror.com/google/electra-base-discriminator)。
每个独立算子保留 `@triton.jit`、字面量 `KERNEL_CONFIG`、`int launch(void)`
和 `makefile`，沿用 typed MemRef → unranked adapter、完整 grid dispatch、
64 字节对齐及非别名检查、NH/E6 runtime 和链接脚本。

当前 FPGA 路径默认使用 C++ lowering 的 FP32 max 标量化。
完整模型导出 embedding、12 层 hidden 和 logits，共 14 个张量，
通过 scheduler 下载后在本地与 emitted 执行结果比较。

验收使用实际 C 模型调度和编译器产出的 scalar LLVM 在本地生成结果，再与
scheduler 下载的实际 BIN 比较。AME INT8/INT32 dot 在本地主机执行时使用
精确 C 整数累加；板端继续执行正常编译的 AME 算子。
每个输出的 FP32 uint32 位模式和全部字节必须相同，任何一个 bit 不同均失败。
这项检查验证该编译实现的 FPGA 执行一致性。

```text
<operator>/         kernel.py + launch.c + makefile；32 个独立算子
model/             固定上游配置、revision 和资产清单
model_runtime/     完整 C 调度、typed ABI 和 NH/E6 模型入口
tools/             导出、构建、参数准备、回传与本地验证工具
common.mk          共用 AOT 编译规则
board.mk/board.ld  独立算子 NH/E6 链接规则与布局
board_model.ld     完整模型链接布局
assets/            checkpoint/tokenizer（Git 忽略）
build/             编译产物、权重与回传 BIN（Git 忽略）
```

固定 batch=1、sequence=128、eval；12 层、hidden/embedding=768、12 heads × 64、
FFN=3072、vocabulary=30522、max positions=512、LayerNorm epsilon=1e-12。
输出是 `[128]` 逐 token 替换判别原始 logits，包含 padding 位置。
Dropout 在推理时关闭。

矩阵采用物理 `B[N,K]`、overwrite `C=A@B.T`。W8A8 逐行量化使用对称
signed INT8、ties away from zero、零行 scale=1。QK 在 dequantize 时应用一次
`0.125`，PV 使用运行时 V 转置量化。Embedding、LayerNorm、bias、GELU、
softmax 和残差保持 FP32。`cases.py` 检查 `BM≤M、BN≤N、BK≤K`，QK 的 BK=64。
最终 N=1 dot 使用精确 scalar INT32 累加；其他矩阵保留 AME `tl.dot`。
独立 `quantize_transpose_128x128` 是额外布局原语，不参与默认模型。

完整预量化板端模型执行 1,571 次算子调用，回传 embedding、12 层 hidden 和
logits。动态权重 C API `electra_infer()` 仍保留，执行 1,645 次 typed 调用；
预量化固定权重省去其中 74 次权重量化。调用者提供 64 字节对齐、非别名张量、
合法 INT32 token/type/mask 和 16 MiB 工作区。

从仓库根目录构建，使用独立输出目录：

```bash
conda activate boscame
export BUDDY_DIR="$PWD/buddy-mlir"
export BUDDY_ROOT="$PWD/buddy-mlir"
source scripts/triton-riscv-env.sh
ELECTRA="$PWD/examples/BOSCAMEFPGA/electra-base-discriminator"
OUT="$ELECTRA/build/example"

make -C "$ELECTRA" launchcheck KERNELS="$OUT/kernels"
make -C "$ELECTRA" model-abicheck
make -C "$ELECTRA" asm bin KERNELS="$OUT/kernels"
make -C "$ELECTRA" assets
python "$ELECTRA/tools/prepare_weights.py" --output "$OUT/weights"
python "$ELECTRA/tools/prepare_model_board.py" \
  --weights "$OUT/weights" --output "$OUT/model-board"
python "$ELECTRA/tools/check_model_weights.py" \
  --weights "$OUT/weights" --build "$OUT/model-board"
python "$ELECTRA/tools/build_model_bin.py" \
  --kernels "$OUT/kernels" --output "$OUT/model-board"
python "$ELECTRA/tools/build_model_host.py" \
  --kernels "$OUT/kernels" --build "$OUT/model-board" --output "$OUT/model-host"
```

`ttir/boscame/llvm/asm` 为可单独选择的阶段，`CASES='...'` 选择算子子集。
每个算子目录默认构建 ELF/bin。`launchcheck` 和 `model-abicheck` 检查 descriptor、
grid 和实际 C 调度 ABI；最终数值验收由 scheduler 回传和本地 emitted 比较完成。
固定 checkpoint revision 为 `1ae76a97c7e84a4e640876a07453fccd636f0667`。
参数和 tokenizer 输入的 dtype、shape、offset、原始名称及 SHA 均写入 manifest。

共享 FPGA 策略默认 `ALLOW_VECTOR_MAX_REDUCTIONS=0`，向 `lower-bosc-ame`
传入 `scalar-f32-max-reductions=true`。C++ pass 使用有界 scratch、volatile
向量快照、逐 lane 标量加载和 SSA 累加器，保留其他向量运算。切换策略会使
LLVM 及后续产物重新编译；汇编过滤和最终 ELF 机器码审计拒绝残留
`vfredmax.vs`。该策略独立于 ordered add 绕过；非 FPGA 编译器默认行为不变。
更多说明见 [编译器策略文档](../../../buddy-mlir/examples/FPGA-BOSCAME/tools/README.scalar-max-reductions.md)。

按仓库 scheduler 技能建立 fpga112 的转发与 token 配置后，前台执行：

```bash
export FPGA_SCHEDULER_CLIENT="$PWD/../fpga-scheduler/.venv/bin/fpga-scheduler"
export FPGA_SCHEDULER_URL=http://127.0.0.1:18762
export FPGA_SCHEDULER_TOKEN_FILE="$PWD/.agent/secrets/fpga112.token"

# 独立算子；可重复 --case 选择子集。
python examples/BOSCAMEFPGA/tools/run_returned_outputs.py \
  --model electra-base-discriminator --kernels "$OUT/kernels" --timeout 1800

# 完整模型；执行、下载所有 BIN、核对元数据/SHA 后，在本地严格验收。
python "$ELECTRA/tools/run_model_board.py" \
  --build "$OUT/model-board" --reference "$OUT/model-host" \
  --output "$OUT/model-run" --timeout 7200
```

runner 的 `--client/--endpoint/--token-file` 可覆盖连接配置，它不建立 SSH 隧道。
输出目录必须是新目录。板端只计算并快照数据；runner 下载实际 BIN，
核对成功退出、镜像与输出元数据，再进行本地数值比较。

已有下载可以完全离线复验，验证结果写入新目录：

```bash
OLD="$OUT/model-run"
python "$ELECTRA/tools/run_model_board.py" \
  --validate-only "$OLD" --reference "$OUT/model-host" \
  --output "$ELECTRA/build/offline-review-new"

# 也可只产生一份新的严格逐位报告。
python "$ELECTRA/tools/verify_model_emitted.py" \
  --run "$OLD" --reference "$OUT/model-host" \
  --output "$ELECTRA/build/offline-review-single.json"
```

`verify_model_emitted.py` 拒绝覆盖现有报告，并检查任务、镜像、参数、参考源码与
对象之间的哈希关联。通过时 runner 返回 0；数值或溯源检查失败返回非零。
正常构建及验收入口见[工具索引](tools/README.md)。

`board.ld` 保留 NH `0x80000000`、E6 `0x80010000` 入口及 NOLOAD/DMA/PMP
约定。完整 FP32 词表约 89.4 MiB，独立算子 arena 为 112 MiB，离线链接区
128 MiB，E6 stack 起于 `0x87800000`。整模型使用专用 `board_model.ld`，
参数、16 MiB scratch 与输出快照独立分配。Git 仅保存源码、配置和教程；
`assets/`、`build/`、Python cache 保持忽略。
