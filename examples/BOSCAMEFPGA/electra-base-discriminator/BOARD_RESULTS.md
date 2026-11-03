# ELECTRA 算子板测（fpga112，2026-11-03）

本文保留早期 60 秒期限测试，不代表最新验证状态。后续全部 32 个独立算子
均完成回传 BIN 和本地验证，见 [RETURNED_OUTPUT_RESULTS.md](RETURNED_OUTPUT_RESULTS.md)。

按本仓库本地 `fpga-scheduler-client` 技能执行全部32个独立算子，scheduler自动分配端口4/6/7。每项执行期限60秒，不含排队与DDR装载。最终 **26 PASS、6 HANG_60S、0 TRAP、0 EXITED_FAIL**。

6项超时中：embedding仅完成START/start，未见进入typed adapter前的0x1001，属于输入初始化超时；3个大matmul已返回完整内核cycle，后续标量oracle未在60秒内完成；2个大量化仅完成第1次内核，后续初始化/校验阶段超时，第2次尚未进入。没有以这些记录声称kernel本体卡死或数值校验通过。

Cycle覆盖完整typed adapter（grid dispatch、DMA和最终fence），不含输入初始化、标量校验及UART。量化正常项的两个cycle分别对应两次调用。

| 算子 | 结果 | cycle(s)，十进制 | 端口 | Job ID | state / execution_status / app_exit / exit_code |
| --- | --- | ---: | ---: | --- | --- |
| [add_128x768](build/board-runs/fpga112-60s-20261103/add_128x768/result.json) | PASS | 1106274 | 6 | 8e712c491e094e7da100636a6946da74 | completed / exited / 0 / 0 |
| [bias_add_128x1](build/board-runs/fpga112-60s-20261103/bias_add_128x1/result.json) | PASS | 11232 | 7 | 751215d6fdf0495a93d9d9667b4b985c | completed / exited / 0 / 0 |
| [bias_add_128x3072](build/board-runs/fpga112-60s-20261103/bias_add_128x3072/result.json) | PASS | 3726317 | 7 | 92650dfc3ec04ef0bd1be3b29a8cb2c3 | completed / exited / 0 / 0 |
| [bias_add_128x768](build/board-runs/fpga112-60s-20261103/bias_add_128x768/result.json) | PASS | 937524 | 6 | 8432f54c9a394da09f88a0b4defb0d3a | completed / exited / 0 / 0 |
| [dequantize_128x1](build/board-runs/fpga112-60s-20261103/dequantize_128x1/result.json) | PASS | 68369 | 4 | 78da64605472406da73ae5aa54add9fb | completed / exited / 0 / 0 |
| [dequantize_128x128](build/board-runs/fpga112-60s-20261103/dequantize_128x128/result.json) | PASS | 272341 | 4 | f8b2ffd08b7f4807899d6b5937760392 | completed / exited / 0 / 0 |
| [dequantize_128x3072](build/board-runs/fpga112-60s-20261103/dequantize_128x3072/result.json) | PASS | 4452244 | 7 | f5a71149e5ca450a9cb070e505a9bfa1 | completed / exited / 0 / 0 |
| [dequantize_128x64](build/board-runs/fpga112-60s-20261103/dequantize_128x64/result.json) | PASS | 164277 | 7 | 948aff4687c343a6b4a1d78a889c41e9 | completed / exited / 0 / 0 |
| [dequantize_128x768](build/board-runs/fpga112-60s-20261103/dequantize_128x768/result.json) | PASS | 1168008 | 7 | 2375de2206de4b18912f1c248eb43fef | completed / exited / 0 / 0 |
| [embedding_128x768](build/board-runs/fpga112-60s-20261103/embedding_128x768/result.json) | HANG：初始化 | — | 4 | 4aebe64b06b143b0857418d10ed3a6a2 | failed / timeout / None / 124 |
| [gelu_128x3072](build/board-runs/fpga112-60s-20261103/gelu_128x3072/result.json) | PASS | 67289545 | 6 | 793d98aaea8d4e00aa5f20ceb24dea6b | completed / exited / 0 / 0 |
| [gelu_128x768](build/board-runs/fpga112-60s-20261103/gelu_128x768/result.json) | PASS | 16948763 | 6 | 9fdb903cb40b4f4496d514e528b6b4e7 | completed / exited / 0 / 0 |
| [layernorm_128x768](build/board-runs/fpga112-60s-20261103/layernorm_128x768/result.json) | PASS | 6993899 | 4 | 040a827009f8465abec903441535d2d1 | completed / exited / 0 / 0 |
| [masked_softmax_12x128x128](build/board-runs/fpga112-60s-20261103/masked_softmax_12x128x128/result.json) | PASS | 66354634 | 6 | 35e37005f3e5403fa8400796882fbf0b | completed / exited / 0 / 0 |
| [matmul_i8_128x128x64](build/board-runs/fpga112-60s-20261103/matmul_i8_128x128x64/result.json) | PASS | 52583 | 4 | 62074c5135bd4cf1b61db5445bb592c7 | completed / exited / 0 / 0 |
| [matmul_i8_128x1x768](build/board-runs/fpga112-60s-20261103/matmul_i8_128x1x768/result.json) | PASS | 1348527 | 4 | 97651ea2fde14cd09d7216d838a51a85 | completed / exited / 0 / 0 |
| [matmul_i8_128x3072x768](build/board-runs/fpga112-60s-20261103/matmul_i8_128x3072x768/result.json) | HANG：内核已返回，校验未完成 | 9313474 | 4 | 22fb9daf2d9f454da25e73f1871ffb7d | failed / timeout / None / 124 |
| [matmul_i8_128x64x128](build/board-runs/fpga112-60s-20261103/matmul_i8_128x64x128/result.json) | PASS | 41848 | 4 | 22e8ab9bff844beda7001b4b3827d4e8 | completed / exited / 0 / 0 |
| [matmul_i8_128x768x3072](build/board-runs/fpga112-60s-20261103/matmul_i8_128x768x3072/result.json) | HANG：内核已返回，校验未完成 | 9237616 | 4 | 62bbfdce4de943e6aa86b6d848957067 | failed / timeout / None / 124 |
| [matmul_i8_128x768x768](build/board-runs/fpga112-60s-20261103/matmul_i8_128x768x768/result.json) | HANG：内核已返回，校验未完成 | 2328574 | 4 | db299b8d2a744daeb6a83b5c32190a34 | failed / timeout / None / 124 |
| [pack_heads_12x128x64](build/board-runs/fpga112-60s-20261103/pack_heads_12x128x64/result.json) | PASS | 905568 | 6 | 0630b3dbb73c48608dc57149c9d63130 | completed / exited / 0 / 0 |
| [quantize_128x128](build/board-runs/fpga112-60s-20261103/quantize_128x128/result.json) | PASS | 517373 / 517236 | 6 | a3286f86cf19457683770538aec81751 | completed / exited / 0 / 0 |
| [quantize_128x3072](build/board-runs/fpga112-60s-20261103/quantize_128x3072/result.json) | PASS | 11289980 / 11288932 | 7 | f2aa62ec41b84d4fac8c0916848259b6 | completed / exited / 0 / 0 |
| [quantize_128x64](build/board-runs/fpga112-60s-20261103/quantize_128x64/result.json) | PASS | 283670 / 283331 | 7 | aeafcc1a6c764c89969802d33e04ec29 | completed / exited / 0 / 0 |
| [quantize_128x768](build/board-runs/fpga112-60s-20261103/quantize_128x768/result.json) | PASS | 2870920 / 2870502 | 6 | acb67acb1bdf40fca170c98d3b9af016 | completed / exited / 0 / 0 |
| [quantize_1x768](build/board-runs/fpga112-60s-20261103/quantize_1x768/result.json) | PASS | 22907 / 22857 | 6 | 979508ba759949808dcd36ea70892434 | completed / exited / 0 / 0 |
| [quantize_3072x768](build/board-runs/fpga112-60s-20261103/quantize_3072x768/result.json) | HANG：1/2次内核已返回 | 68890676 | 4 | 8df76978dfa64bd4911447158e6f1fda | failed / timeout / None / 124 |
| [quantize_768x3072](build/board-runs/fpga112-60s-20261103/quantize_768x3072/result.json) | HANG：1/2次内核已返回 | 67737294 | 6 | ed2cc724733b40b4ad4f44163c0d8242 | failed / timeout / None / 124 |
| [quantize_768x768](build/board-runs/fpga112-60s-20261103/quantize_768x768/result.json) | PASS | 17221513 / 17220505 | 4 | 7f50be24341340dbb0083eee7043e835 | completed / exited / 0 / 0 |
| [quantize_transpose_128x128](build/board-runs/fpga112-60s-20261103/quantize_transpose_128x128/result.json) | PASS | 2085170 | 4 | 48fc938854da4ea49dd61501c7782e8c | completed / exited / 0 / 0 |
| [quantize_transpose_128x64](build/board-runs/fpga112-60s-20261103/quantize_transpose_128x64/result.json) | PASS | 1042493 | 4 | d71727e714324bf485ec23bdec0ab123 | completed / exited / 0 / 0 |
| [unpack_heads_12x128x64](build/board-runs/fpga112-60s-20261103/unpack_heads_12x128x64/result.json) | PASS | 905481 | 6 | 1347a272aa6548268adde3f05388db59 | completed / exited / 0 / 0 |

## Launcher 计数修正与复测

首次板测的add_128x768、bias_add_128x768、gelu_128x3072、gelu_128x768返回但校验失败，原因是C launcher残留ViT的151296/605184元素计数，在当前输出shape之外继续校验。已分别改为128×768或128×3072，保持Triton kernel与matmul tile不变。

修复后4项通过实际logical-shape host stub fixture数值回归、launcher ABI检查、重新链接与NR ELF审计，并在fpga112重新执行，均PASS。上表为最终复测结果；最初失败的BIN、UART、旧launch.c及job结果保存在`launcher-count-before-fix-attempts/`，未删除或将首次失败改写为成功。

| 算子 | 首次校验错误数 | 首次Job ID |
| --- | ---: | --- |
| add_128x768 | 52089 | 85406749e1d94a5bb8691f7d45a8816d |
| bias_add_128x768 | 51486 | b30e8d3a4c7344709b69d0589285e3cd |
| gelu_128x3072 | 211814 | 16f8d41919744dd98cfe72b5bc85c5f6 |
| gelu_128x768 | 495087 | 2c535299bfa542319d0c133d9cd856e0 |

## 连接中断恢复

旧HTTP转发曾退出，gelu_128x3072的原任务`1e1688a2d1b54c0c979f5172cb42d7da`因客户端失联被取消。SSH主链可达，重建本地转发后先查询该原ID确认cancelled，再用新ID重提。中断原尝试保存在`connection-interrupted-attempts/`，不计HANG。

全部原始证据位于`build/board-runs/fpga112-60s-20261103/`：每项包含冻结BIN、镜像SHA、ELF审计、board_config、CLI stdout/stderr、完整UART、task.json和result.json。`verification.json`确认全部32项BIN与服务器image_sha256、当前kernel源哈希一致；job JSON保留完整details。本次不验证完整模型执行或任务精度。
