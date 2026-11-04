# EfficientNet-B3 FPGA 示例

本目录提供 `timm/efficientnet_b3.ra2_in1k` 的 99 个独立算子，以及完整模型的
编译、上板和数值验证工具。模型输入为 `1×3×288×288`，完整图包含 308 个逻辑
节点和 503 次物理调用。

37 个 INT8 点卷积保留原有 packed GEMM。4 个普通卷积、14 个 depthwise 和
分类器保留 FP32 公共 ABI，内部使用量化后的 **i8×i8→i32 MMA**，卷积采用
im2col，再反量化并加 bias。GAP 使用标量 Torch 求和顺序，SiLU 使用 RVV
直接写最终输出。默认构建绕开有数值问题的向量 max/add reduction；不要关闭
归约检查。

每个算子目录的 `kernel.py`、`launch.c`、`makefile` 分别定义固定配置、裸机
入口和构建方式；共享编译链在 `common.mk`、`board.mk`、`tools/` 中，整模型
工具在 `diagnostics/model_integration/`。生成的 IR、对象、镜像和链路元数据
统一放在被 Git 忽略的 `build/` 中。

## 环境

下面的命令均从仓库根目录执行。使用已安装本仓库 Triton backend 的
`boscame` 环境，以及本地 Buddy/LLVM 工具链：

```bash
conda activate boscame
export BUDDY_DIR="$PWD/buddy-mlir"
source scripts/triton-riscv-env.sh
export BUDDY_ROOT="$BUDDY_DIR"

EFFNET=examples/BOSCAMEFPGA/efficientnet_b3.ra2_in1k
MODEL="$EFFNET/build/model-integration"
TOOLS="$EFFNET/diagnostics/model_integration"
```

整模型 CPU 参考需要 NumPy、Torch 2.10、torchvision、safetensors 和
huggingface_hub；GAP 顺序实现对应 Torch commit
`449b1768410104d3ed79d3bcfe4ba1d65c7f22c0`。timm 使用下文单独安装的
`1.0.22` 版本。工具位置可通过 `BUDDY_OPT`、`LLVM_BIN`、`LLVM_OPT`、
`LLC`、`CLANG`、`LINKER` 等 Make 变量覆盖。

## 编译与本地检查

```bash
# 全部 99 个算子，编译、链接并审计最终 ELF：
make -C "$EFFNET" -j4 bin

# 单个算子：
make -C "$EFFNET/conv2d_288x288_3x40_k3_s2" bin

# 指定子集：
make -C "$EFFNET" -j4 bin CASES='add_1x232x9x9 silu_1x6x1x1'

# 描述符、grid 和非法 ABI 检查：
make -C "$EFFNET" launchcheck

# 生成的 BOSCAME IR 与独立参考比较；大形状全网格仿真较慢：
make -C "$EFFNET" simcheck CASES='silu_1x6x1x1 gap_9x9x816 quantize_1x1536'
```

`asm` 只编译到 `kernel.o`；`ttir`、`boscame`、`llvm` 可停在中间阶段。
每个算子的 BIN 位于 `build/kernels/<case>/<case>.bin`，IR 和汇编位于同目录。
18 个卷积默认插入逐 patch DDR 读写阶段，并执行关闭 loop/SLP 自动向量化的
LLVM O2。GAP、SiLU 的默认 object 来自 case 显式绑定的 C 实现。

`launchcheck` 使用 mock raw entry 检查 ABI；`simcheck` 检查 IR 数学。
二者不能代替板端数值验证。部分裸机 launcher 只是 smoke test，退出码 0
仅表示执行返回；完整模型按后文回读全部 logits 后独立验收。

## fpga112 连接与单算子执行

使用 `fpga-scheduler` 客户端，并确认 host 隧道指向 fpga112。
已有正确 HTTP 转发时沿用；没有时建立自己的转发：

```bash
ssh -o BatchMode=yes -o ConnectTimeout=8 -o ConnectionAttempts=1 fpga112 true
ssh -fNT -o BatchMode=yes -o ConnectTimeout=8 \
  -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 -o ServerAliveCountMax=3 \
  -L 127.0.0.1:18762:127.0.0.1:18765 fpga112

export FPGA_SCHEDULER_URL=http://127.0.0.1:18762
export FPGA_SCHEDULER_TOKEN_FILE="$PWD/.agent/secrets/fpga112.token"
FPGA_CLIENT=/home/chh/gitprojects/fpga-scheduler/.venv/bin/fpga-scheduler

"$FPGA_CLIENT" --endpoint "$FPGA_SCHEDULER_URL" \
  --token-file "$FPGA_SCHEDULER_TOKEN_FILE" health
"$FPGA_CLIENT" --endpoint "$FPGA_SCHEDULER_URL" \
  --token-file "$FPGA_SCHEDULER_TOKEN_FILE" devices
```

确认有 READY 设备后执行：

```bash
"$FPGA_CLIENT" --endpoint "$FPGA_SCHEDULER_URL" \
  --token-file "$FPGA_SCHEDULER_TOKEN_FILE" \
  --lease-seconds 300 --heartbeat-seconds 2 --request-timeout 180 \
  run "$EFFNET/build/kernels/conv2d_288x288_3x40_k3_s2/conv2d_288x288_3x40_k3_s2.bin" \
  --timeout 600 --queue-timeout 3600
```

客户端必须前台存活到任务结束。执行期限、客户端租约和 HTTP deadline 是
不同设置。单算子超过 60 秒按约定标为 HANG；放宽执行期限用于确认最终能否
返回。`launch_cycles` 包含 launcher 工作，不等同于纯 kernel cycles。
发生断线时先恢复隧道，再查询原任务 ID，避免直接重复提交。

## 准备完整模型

先完成上面的全部算子编译。安装固定 timm，并下载脚本要求的官方权重与配置：

```bash
python3 -m pip install --no-deps --target "$MODEL/python-deps" timm==1.0.22
python3 - <<'ASSETS'
from huggingface_hub import hf_hub_download
for filename in ("model.safetensors", "config.json"):
    hf_hub_download(
        repo_id="timm/efficientnet_b3.ra2_in1k",
        revision="0366a75518620e0f2077789202073759f2d61393",
        filename=filename,
        local_dir="examples/BOSCAMEFPGA/efficientnet_b3.ra2_in1k/build/model-integration/assets",
    )
ASSETS

python3 "$TOOLS/model_graph.py"
python3 "$TOOLS/reference.py"
python3 "$TOOLS/check_connectors.py" --buddy-opt "$BUDDY_DIR/build/bin/buddy-opt"
python3 "$TOOLS/physical_plan.py"
python3 "$TOOLS/audit_model_plan.py"
python3 "$TOOLS/build_connectors.py" --jobs 2 --output "$MODEL/connectors-ddr-policy"
python3 "$TOOLS/generate_runner.py" --plan-dir "$MODEL/physical-plan" --output "$MODEL/runner"
```

`reference.py` 会校验固定权重/config 的 SHA，生成原 FP32 与 W8A8 CPU 参考。
默认输入是 `((i*17)%257-128)/64` 的确定性张量；它不是图片，不代表 ImageNet
准确率。可用 `reference.py --input-f32 PATH` 指定已预处理的 little-endian
FP32 `[1,3,288,288]` 输入，然后重新生成 physical plan、runner 和镜像。

构建两个额外形状需要的 SiLU，以及 RVV 反量化连接算子：

```bash
python3 "$TOOLS/prepare_prefix.py"
python3 "$TOOLS/check_silu_direct_output.py" \
  --buddy-opt "$BUDDY_DIR/build/bin/buddy-opt" --stem "$MODEL/prefix8/stem.f32"
python3 "$TOOLS/build_silu_direct_output.py" \
  --output "$MODEL/silu-rvv-direct-output-overlay"
python3 "$TOOLS/check_dequant_candidate.py"
python3 "$TOOLS/build_dequant_overlay.py" \
  --plan "$MODEL/physical-plan/physical-plan.json" \
  --source-connectors "$MODEL/connectors-ddr-policy" --output "$MODEL/dequant-rvv-overlay"
```

GAP 使用 `build/kernels/` 中由标准 Make 生成的默认 Torch-order 实现。
上述检查会重新生成所需的 host 验证和对象 manifest，无需历史上板记录。

## 构建与运行完整模型

```bash
IMAGE="$MODEL/model-image"
python3 "$TOOLS/build_model_image.py" --jobs 4 \
  --host-numerical-validation --strict-probe \
  --plan-dir "$MODEL/physical-plan" --runner "$MODEL/runner" \
  --kernels "$EFFNET/build/kernels" --conv-overlay "$EFFNET/build/kernels" \
  --silu-overlay "$MODEL/silu-rvv-direct-output-overlay" \
  --silu-source "$TOOLS/silu_rvv_direct_output_candidate.c" \
  --dequant-overlay "$MODEL/dequant-rvv-overlay" \
  --connectors "$MODEL/connectors-ddr-policy" --output "$IMAGE"
python3 "$TOOLS/audit_image.py" --image-dir "$IMAGE" --plan-dir "$MODEL/physical-plan"
```

完成单算子检查后，通过 scheduler 提交完整模型：

```bash
python3 "$TOOLS/run_collect.py" --phase-gates-complete \
  --image-dir "$IMAGE" --reference-dir "$MODEL/physical-plan" \
  --output "$MODEL/board" --endpoint "$FPGA_SCHEDULER_URL" \
  --token-file "$FPGA_SCHEDULER_TOKEN_FILE" \
  --timeout 7200 --queue-timeout 3600 \
  --lease-seconds 300 --heartbeat-seconds 2 --request-timeout 180
```

该命令持续前台保活，完成后收集 UART、全部 1,000 个 `logits_f32` 和
503 个 `call_cycles_u64`。将下面的 `TASK_ID` 替换为本次输出的任务 ID，
独立检查完整输出、镜像/参考 hash 及最终 UART 与 DDR cycles 是否一致：

```bash
TASK_ID=替换为本次任务ID
RUN="$MODEL/board/$TASK_ID"
python3 "$TOOLS/audit_board_run.py" --run-dir "$RUN" \
  --reference-dir "$MODEL/physical-plan" --primary-reference-dir "$MODEL/physical-plan"
python3 "$TOOLS/summarize_model_runtime.py" --run-dir "$RUN" --plan-dir "$MODEL/physical-plan"
```

`--strict-probe` 使用 `atol=rtol=2e-6`；`--host-numerical-validation` 允许完整
执行后导出结果，由主机数值审计决定是否通过。板端 app0、编译/ISA 检查通过
都不等同于数值通过。重新构建后需验证新镜像，不继承旧对象的运行结论。
