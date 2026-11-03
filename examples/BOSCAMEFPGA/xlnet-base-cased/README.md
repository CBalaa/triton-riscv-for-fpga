# XLNet-Base-Cased on BOSCAME

本例按 `../vit-base-patch16-224` 的 launcher、Triton 算子、typed adapter、
NH/E6 runtime 和链接脚本适配
[`xlnet/xlnet-base-cased`](https://hf-mirror.com/xlnet/xlnet-base-cased)。
每个算子目录包含 `kernel.py`、字面量 `KERNEL_CONFIG`、`int launch(void)` 和
`makefile`；`common.mk`、`board.mk` 负责共享编译与裸机链接。

固定配置为 batch=1、sequence=128、12 层、hidden=768、12 heads、head dim=64、
intermediate=3072、vocabulary=32000，eval 双向 content stream，`use_mems=False`。
支持 attention mask 与 token type ids 0/1/2；不支持训练 dropout、query stream、
循环记忆、动态形状或单向配置。

完整 word embedding 使用 FP32，线性层采用逐行对称 W8A8、signed i8/i32 matmul
和 FP32 dequantization。量化范围 [-127,127]，舍入为 nearest ties away from zero，
全零行 scale=1。`model_runtime/model.c` 串联完整 encoder 与可选 LM head；LM head
分 32 个 1024 词 tile，最后保留 256 词，输出 `logits[128,32000]`。

Q/K/V/R 的 `[input,head,dim]` 权重转为物理 `[output,input]`；O 已为 `[N,K]`。
QK、位置分数、segment 分数和 PV 逐 head 执行，缩放 0.125 在 AC+BD+EF 合并后
应用一次。Segment N=2 使用 BM=BN=1、BK=64 的标量 int32 row-dot；其余矩阵
使用 AME。`cases.py` 检查所有 BM/BN/BK 均不超过实际 M/N/K。

## 准备环境与输入

从仓库根目录执行，需要已构建的 LLVM、Buddy、Triton CPU backend，以及模型
输入准备所需的 Python 环境。模型配置与固定 revision/hash 见
[model/config.json](model/config.json) 和 [model/manifest.json](model/manifest.json)。
`assets/` 保存下载的 checkpoint/tokenizer，`build/` 保存生成文件，均被 Git 忽略。

```bash
conda activate boscame
source scripts/triton-riscv-env.sh
export BUDDY_ROOT="$PWD/buddy-mlir"
X=examples/BOSCAMEFPGA/xlnet-base-cased
K="$PWD/$X/build/kernels"

python "$X/tools/fetch_assets.py" --weights
python "$X/tools/prepare_weights.py" --lm-head --output "$X/build/weights-lm"
python "$X/tools/prepare_model_input.py" --lm-head --output "$X/build/model-input-lm"
```

`prepare_weights.py` 生成物理布局明确的权重及 scale，保留 FP32 embedding。
`prepare_model_input.py` 生成左 padding 到 128 的真实句对输入。
Relative positional encoding 在主机离线生成位置序列 128 到 -127。
content stream 的 attention mask 取消对角线屏蔽，包括 padding query 行；
`model.c` 将 mask[128] 转为 `non_target[128,128]`。

## 编译算子与完整模型

```bash
make -C "$X" launchcheck
make -C "$X" model-runtime-object
make -C "$X" -j4 asm KERNELS="$K"
make -C "$X" -j4 bin KERNELS="$K"
make -C "$X" bincheck KERNELS="$K"

python "$X/tools/build_model_bin.py" --lm-head --kernels "$K" \
  --weights "$X/build/weights-lm" --model-input "$X/build/model-input-lm" \
  --output "$X/build/model-bin-lm"
python "$X/tools/build_model_host.py" --build "$X/build/model-bin-lm" \
  --kernels "$K" --output "$X/build/model-host-lm"
```

`launchcheck` 检查 grid 和 descriptor ABI。`bincheck` 核对源码、编译产物与 ELF
审计哈希，并从最终 ELF 重新提取 BIN 逐字节比较。完整模型 builder 链接真实
`model.c`、AOT 对象、typed adapter、NH/E6 runtime 与 freestanding math。
`build_model_host.py` 在主机执行相同 C graph 和 emitted LLVM，仅用精确
signed-i8/i32 dot 执行 AME 整数矩阵乘法；输出清单直接来自模型的 `output-layout.json`。

需要 encoder 的每层 hidden 快照时，权重与模型构建均去掉 `--lm-head`，输入准备
也可去掉该选项，并使用对应的新输出目录。encoder 导出 embedding 和 12 层 hidden，
LM 镜像导出完整词表 logits。`--kernels` 与 `--output` 可分离不同构建。

FPGA 构建默认启用 C++ lowering 的 `scalar-f32-max-reductions=true` 与既有的
ordered-add 标量化。FP32 max 使用有界 scratch、volatile 标量加载和标量 accumulator，
其余向量计算保留；最终 ELF 审计拒绝残留 `vfredmax.vs`。两项策略独立，切换策略会
触发 LLVM 阶段起重新编译。这是软件绕过，硬件根因尚未确定。

所有 entry 要求连续、非别名、64 字节对齐并匹配固定形状；token id 范围为
[0,32000)。完整模型使用 `board_model.ld`，NH/E6 entry 为
`0x80000000` / `0x80010000`，参数从 `0x81000000` 开始。链接器 ASSERT 检查代码、
快照、参数、scratch 和栈边界。LM 的 16,384,000 字节输出需要至少 16 MiB
scheduler 捕获窗口。

## 执行与回传验证

验收标准为：**本地 emitted 执行输出与 scheduler 实际回传 BIN 逐位一致**。
板端只计算和捕获，下载后在本地按 FP32 位模式与整数原始字节比较。

运行前选择 FPGA 目标，确认 SSH/API 转发、token 和设备 READY。下面 URL 对应
fpga112 的本地转发；其他部署应使用其明确配置。

```bash
export FPGA_SCHEDULER_URL=http://127.0.0.1:18762
export FPGA_SCHEDULER_TOKEN_FILE="$PWD/.agent/secrets/fpga112.token"

python "$X/tools/run_model_board.py" --build "$X/build/model-bin-lm" \
  --reference-emitted "$X/build/model-host-lm" \
  --output "$X/build/model-runs/run-001" --timeout 7200
```

客户端默认使用相邻 `fpga-scheduler` checkout，缺失时从 PATH 查找
`fpga-scheduler`。可以传入 `--client`、`--endpoint`、`--token-file`，或设置
`FPGA_SCHEDULER_CLIENT`、`FPGA_SCHEDULER_URL`、`FPGA_SCHEDULER_TOKEN_FILE`。
前台客户端持续保活到任务结束；运行、恢复原任务和下载使用相同配置。
相同 `--output` 会查询既有任务并核对冻结文件，不重复提交。

只验证已经下载的文件时，将新报告写入独立位置，不访问 scheduler：

```bash
python "$X/tools/run_model_board.py" --validate-only "$X/build/model-runs/run-001" \
  --validation-output "$X/build/model-reviews/run-001.json"
```

正常单算子回传也使用共享客户端，可按需选择算子：

```bash
python examples/BOSCAMEFPGA/tools/run_returned_outputs.py \
  --model xlnet-base-cased --kernels "$K" --case quantize_128x768 \
  --output "$X/build/operator-runs"
python "$X/tools/verify_quantize_bits.py" \
  --runs "$X/build/operator-runs" --case quantize_128x768 \
  --output "$X/build/operator-reviews/quantize_128x768.json"
```

量化的 Q/Scale 快照按完整缓存行隔离，后续调用不会覆盖先前输出。下载器核对
name、dtype、shape、nbytes 与 SHA；缺失、截短或损坏文件会使本地验证失败。

## 已知差异

严格 encoder 验证中，第三层末 `layernorm_128x768[34,334]` 存在 1 ulp 差异：
expected 为 `-1.411880612373352` / `0xbfb4b881`，实际为
`-1.4118804931640625` / `0xbfb4b880`。硬件根因尚未确定，严格比较保留该差异，
不放宽容差。
