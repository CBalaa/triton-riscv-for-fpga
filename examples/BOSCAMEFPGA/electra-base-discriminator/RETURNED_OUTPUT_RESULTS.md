# ELECTRA 输出回传与本地验证结果

32个独立算子重新构建并在fpga112使用180秒执行期限运行；每项都下载真实输出BIN，再执行原始launch.c的本地C oracle。旧60秒证据保持不变。

| 算子 | 结果 | cycles（十进制） | 输出BIN | 端口 | Task ID |
| --- | --- | ---: | ---: | ---: | --- |
| [add_128x768](build/board-runs/fpga112-returned-outputs-20261103/add_128x768/result.json) | PASS | 1106261 | 1 | 6 | a1ae8439c6614583875960ac384b1265 |
| [bias_add_128x1](build/board-runs/fpga112-returned-outputs-20261103/bias_add_128x1/result.json) | PASS | 11092 | 1 | 6 | bd4c254cad1849f6a8a375605f68c026 |
| [bias_add_128x3072](build/board-runs/fpga112-returned-outputs-20261103/bias_add_128x3072/result.json) | PASS | 3712317 | 1 | 7 | 107dd1f4f1bd41d49b0171c4a0c0d1b3 |
| [bias_add_128x768](build/board-runs/fpga112-returned-outputs-20261103/bias_add_128x768/result.json) | PASS | 937443 | 1 | 6 | 4736cf4c12224c36942b61f12e9c0bb1 |
| [dequantize_128x1](build/board-runs/fpga112-returned-outputs-20261103/dequantize_128x1/result.json) | PASS | 68261 | 1 | 6 | a9aefb528b2e4ef58691713037d06371 |
| [dequantize_128x128](build/board-runs/fpga112-returned-outputs-20261103/dequantize_128x128/result.json) | PASS | 271579 | 1 | 6 | 103083c33c7941eeb8ffdc3fb9e4f750 |
| [dequantize_128x3072](build/board-runs/fpga112-returned-outputs-20261103/dequantize_128x3072/result.json) | PASS | 4447353 | 1 | 7 | fe4b0f46c94440b9968fcb002cccbcfe |
| [dequantize_128x64](build/board-runs/fpga112-returned-outputs-20261103/dequantize_128x64/result.json) | PASS | 167463 | 1 | 6 | e26916a035cb46cea54aaa30889c178f |
| [dequantize_128x768](build/board-runs/fpga112-returned-outputs-20261103/dequantize_128x768/result.json) | PASS | 1163262 | 1 | 6 | 60693e1060a74945a236923e1e18c62b |
| [embedding_128x768](build/board-runs/fpga112-returned-outputs-20261103/embedding_128x768/result.json) | PASS | 1207542 | 1 | 6 | cdfaee32d8364201a420185aaf48237c |
| [gelu_128x3072](build/board-runs/fpga112-returned-outputs-20261103/gelu_128x3072/result.json) | PASS | 67288090 | 1 | 6 | 84012e7e3450453cb75bd06a7cada9e6 |
| [gelu_128x768](build/board-runs/fpga112-returned-outputs-20261103/gelu_128x768/result.json) | PASS | 16821665 | 1 | 6 | 45c0b97d744d407fa5db825db1779471 |
| [layernorm_128x768](build/board-runs/fpga112-returned-outputs-20261103/layernorm_128x768/result.json) | PASS | 6997435 | 1 | 7 | e422055d71004dc8be8fcdd6f1a88ac6 |
| [masked_softmax_12x128x128](build/board-runs/fpga112-returned-outputs-20261103/masked_softmax_12x128x128/result.json) | PASS | 66544010 | 1 | 7 | ef972c7b5cfd40d1acfc8092e69c051a |
| [matmul_i8_128x128x64](build/board-runs/fpga112-returned-outputs-20261103/matmul_i8_128x128x64/result.json) | PASS | 52411 | 1 | 6 | efeddecf88914c98a10db6afb4eab57f |
| [matmul_i8_128x1x768](build/board-runs/fpga112-returned-outputs-20261103/matmul_i8_128x1x768/result.json) | PASS | 1349471 | 1 | 7 | c577f03f93ee4ede84e3cde027f66f67 |
| [matmul_i8_128x3072x768](build/board-runs/fpga112-returned-outputs-20261103/matmul_i8_128x3072x768/result.json) | PASS | 9617042 | 1 | 7 | aee0801122174722af79c624d7eb5dba |
| [matmul_i8_128x64x128](build/board-runs/fpga112-returned-outputs-20261103/matmul_i8_128x64x128/result.json) | PASS | 41972 | 1 | 7 | dd24442064a04d53b5a190fa4316cbc2 |
| [matmul_i8_128x768x3072](build/board-runs/fpga112-returned-outputs-20261103/matmul_i8_128x768x3072/result.json) | PASS | 8928537 | 1 | 7 | 41cd53c023654494a0f87addded3a053 |
| [matmul_i8_128x768x768](build/board-runs/fpga112-returned-outputs-20261103/matmul_i8_128x768x768/result.json) | PASS | 2407074 | 1 | 4 | 7102667f248e44ecbd42e42d75a2b496 |
| [pack_heads_12x128x64](build/board-runs/fpga112-returned-outputs-20261103/pack_heads_12x128x64/result.json) | PASS | 970211 | 1 | 7 | ab43a27a1082447aa5e61805d84c1869 |
| [quantize_128x128](build/board-runs/fpga112-returned-outputs-20261103/quantize_128x128/result.json) | PASS | 517368 / 517190 | 4 | 4 | 8f49b4b9c9fd42fc805ac82b4dbf0f55 |
| [quantize_128x3072](build/board-runs/fpga112-returned-outputs-20261103/quantize_128x3072/result.json) | PASS | 11290144 / 11288894 | 4 | 4 | fd4e95badad446ecaa634137075f7887 |
| [quantize_128x64](build/board-runs/fpga112-returned-outputs-20261103/quantize_128x64/result.json) | PASS | 276729 / 276599 | 4 | 4 | 3f33e6116d6048adb4f73bacb06bf7f4 |
| [quantize_128x768](build/board-runs/fpga112-returned-outputs-20261103/quantize_128x768/result.json) | PASS | 2871154 / 2870689 | 4 | 7 | 399e57f624714e4e9ec2e4dfa187b9e9 |
| [quantize_1x768](build/board-runs/fpga112-returned-outputs-20261103/quantize_1x768/result.json) | PASS | 22875 / 22773 | 4 | 7 | 52e97ba1f80f4ccbafa47eac40d5b9ae |
| [quantize_3072x768](build/board-runs/fpga112-returned-full-16m-20261103/quantize_3072x768/result.json) | PASS | 68894512 / 68892438 | 4 | 6 | c776bdac60e74aa69912de32c00a9a29 |
| [quantize_768x3072](build/board-runs/fpga112-returned-full-16m-20261103/quantize_768x3072/result.json) | PASS | 67732757 / 67731868 | 4 | 6 | b84238b086ce4b4aa0e56751914d0ce4 |
| [quantize_768x768](build/board-runs/fpga112-returned-outputs-20261103/quantize_768x768/result.json) | PASS | 17223335 / 17222517 | 4 | 6 | da5b6d83b79f4852a0027e892876cefc |
| [quantize_transpose_128x128](build/board-runs/fpga112-returned-outputs-20261103/quantize_transpose_128x128/result.json) | PASS | 2084957 | 2 | 4 | 014ae962c8df4bf0a5a72011c61b310a |
| [quantize_transpose_128x64](build/board-runs/fpga112-returned-outputs-20261103/quantize_transpose_128x64/result.json) | PASS | 1042907 | 2 | 4 | 323b942254b047ca84aab315a1be3f50 |
| [unpack_heads_12x128x64](build/board-runs/fpga112-returned-outputs-20261103/unpack_heads_12x128x64/result.json) | PASS | 911046 | 1 | 4 | 88bd46944f9248acb6d47fb344fa3223 |

PASS要求scheduler应用成功、输出捕获成功、下载清单/长度/SHA匹配，以及本地原oracle完整两次重放通过。各轮量化Q/Scale分别持久化；负回归验证缺失、短文件和错误输出会失败。cycles不含输入初始化、快照复制、下载及本地oracle。

以上是独立算子验证；旧 60 秒期限记录见 [BOARD_RESULTS.md](BOARD_RESULTS.md)。
完整模型已正常执行，但独立 Torch W8A8 对照失败，详见
[MODEL_BOARD_RESULTS.md](MODEL_BOARD_RESULTS.md)；数据集任务精度尚未评估。
表内 `build/` 链接指向本地原始证据，BIN 和编译产物未纳入 Git。
