# ELECTRA 工具索引

脚本从自身位置查找样例目录。所有默认输出均写入被忽略的 `../build/`，
checkpoint/tokenizer 写入 `../assets/`；不需要将权重或 BIN 纳入 Git。

| 工作 | 入口 |
| --- | --- |
| 读取字面量算子配置与检查 tile | `../cases.py` |
| Triton AOT、contract 与 typed adapter 导出 | `export.py`、`prepare_adapters.py` |
| staged build 状态与哈希 | `report.py` |
| 独立算子 board 配置与回传输出声明 | `board_config.py`、`output_spec.py` |
| NH/E6 启动、SPM/DMA 与 launcher 支持 | `board_support.c/.h`、`launch_checks.h`、`attention_reference.h` |
| 独立源码数值、launcher ABI/fixture、emitted 模拟 | `check_kernels.py`、`check_launch.py`、`check_launch_numerics.py`、`simcheck.py` |
| pinned HF 资产与普通 FP32 权重 blob | `fetch_assets.py`、`prepare_weights.py` |
| tokenizer 输入与整模型预量化参数 | `prepare_inputs.py`、`prepare_model_board.py` |
| 独立 PyTorch FP32/W8A8 模型参考 | `model_reference.py` |
| 实际 C 模型的 typed ABI mock 检查 | `check_model_abi.py` |
| 逐张量打包参数核对 | `check_model_weights.py` |
| 完整 NH/E6 ELF/BIN 构建与 NR 审计 | `build_model_bin.py` |
| 实际 C 调度、emitted scalar LLVM 与独立精确整数 dot 本地执行 | `build_model_host.py` |
| 实际 FPGA 运行、全量 BIN 下载和本地双参考验证 | `run_model_board.py` |
| 下载张量对照独立 PyTorch W8A8 | `verify_model_board.py` |
| 下载张量对照本地 emitted 实现 | `verify_model_emitted.py` |

构建、依赖和 scheduler 连接配置见 [README](../README.md)。共享 AOT 前端与
scheduler 输出实现位于 `../../tools/`，模型算子自身的 kernel/launcher/连接规则
保留在本目录。完整模型 runner 仅在两个本地参考都通过时返回 0；
ELECTRA 历史结果仍是 emitted 实现一致性 PASS、独立 Torch W8A8 FAIL。
