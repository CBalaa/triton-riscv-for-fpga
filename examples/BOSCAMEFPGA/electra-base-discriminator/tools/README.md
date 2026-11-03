# ELECTRA 工具索引

脚本从自身位置查找样例目录。默认产物写入被忽略的 `../build/`，
checkpoint/tokenizer 写入 `../assets/`。
不将权重、编译产物或回传 BIN 纳入 Git。

| 工作 | 入口 |
| --- | --- |
| 字面量配置与 tile 检查 | `../cases.py` |
| Triton AOT、contract 与 typed adapter 导出 | `export.py`、`prepare_adapters.py` |
| staged build 状态、命令与哈希 | `report.py` |
| 独立算子 board 配置和输出声明 | `board_config.py`、`output_spec.py` |
| NH/E6、SPM/DMA 与 launcher 支持 | `board_support.c/.h`、`launch_checks.h`、`attention_reference.h` |
| launcher 与 typed ABI 检查 | `check_launch.py`、`check_model_abi.py` |
| pinned HF 资产与权重 blob | `fetch_assets.py`、`prepare_weights.py` |
| tokenizer 输入与整模型预量化参数 | `prepare_inputs.py`、`prepare_model_board.py` |
| 逐张量参数核对 | `check_model_weights.py` |
| 完整 NH/E6 ELF/BIN 构建与机器码审计 | `build_model_bin.py` |
| 本地执行实际 C 调度、emitted scalar LLVM 和精确整数 dot | `build_model_host.py` |
| FPGA 执行、全量 BIN 下载、本地逐位验收；已有下载离线复验 | `run_model_board.py` |
| 下载张量与本地 emitted 输出的 uint32/字节严格比较 | `verify_model_emitted.py` |

完整模型默认验收仅调用 emitted 严格比较。`run_model_board.py --validate-only`
必须指定已有参考与新输出目录，不访问 scheduler，也不改写原运行记录。
`verify_model_emitted.py --output` 指定新报告路径，已存在的报告不会覆盖。

构建、连接配置和验收命令见 [README](../README.md)。共享 AOT 前端与
scheduler 输出协议位于 `../../tools/`；算子 kernel、launcher 和链接规则
保留在本目录。
