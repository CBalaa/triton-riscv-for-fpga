# E5-small-v2 board results — fpga112, 2026-11-03

Target: `fpga112`; scheduler endpoint: `http://127.0.0.1:18762`.
Each task uses a 60-second application execution deadline, excluding queue wait and DDR loading; queue timeout is 3,600 seconds.

Coverage: **HANG_60S: 3**, **PASS: 27**.

Completed kernel cycle records: **30/30**. Timeout after adapter return: **3**; timeout without observed adapter return: **0**; TRAP: **0**; normal exit with validation failure: **0**.

An initial API connection failure was recovered by querying the original task ID and confirming `Task not found` before a new submission. That unexecuted attempt is archived under `connection-interrupted-attempts/` and excluded from the execution counts.

| Operator | Status | Timeout phase | Kernel cycles | Job ID | FPGA port |
| --- | --- | --- | ---: | --- | --- |
| `add_128x384` | PASS | — | 553405 | `91a047ffbe9d4bd78a0aa12f82505c23` | 7 |
| `attention_mask_12x128x128` | PASS | — | 2782602 | `cb8c75369b7c4127b364339fb42b70d7` | 7 |
| `bias_add_128x1536` | PASS | — | 1841779 | `dba2feb042ca417a9928e74c9419b43f` | 7 |
| `bias_add_128x384` | PASS | — | 488871 | `21cf5d5b61434394aef436e624a4c594` | 4 |
| `dequantize_128x128` | PASS | — | 272360 | `3216026020764db09644886b9b60f354` | 4 |
| `dequantize_128x1536` | PASS | — | 2262843 | `aacded79af3e4cf3a4df657114f179b4` | 4 |
| `dequantize_128x32` | PASS | — | 118597 | `200f2e782f0a416d9418a5e7da8c8bd9` | 7 |
| `dequantize_128x384` | PASS | — | 621819 | `6b6344507ce04339a8438a36423f41e0` | 7 |
| `embedding_add_128x384` | PASS | — | 658393 | `c8105a5b1e994ed89cb667486c2c477c` | 7 |
| `embedding_lookup_128x384` | PASS | — | 337521 | `8f5396459675490199ce6b291d232a35` | 6 |
| `gelu_128x1536` | PASS | — | 33703410 | `37043ac8132d4b9e9f58ffc40c4ea8bc` | 7 |
| `l2_normalize_1x384` | PASS | — | 15427 | `36cab029b2eb4db9b7e18fee0b9a3047` | 6 |
| `layernorm_128x384` | PASS | — | 3239782 | `bafbc7d21d7846ec91a3a7144d903770` | 7 |
| `matmul_i8_128x128x32` | PASS | — | 7602596 | `b44ae0d9770d4cd5b7aa9f613831afd3` | 7 |
| `matmul_i8_128x1536x384` | HANG_60S | kernel_returned_before_timeout | 2531967 | `03a627acc05745d9a463ce8ddc16ef71` | 6 |
| `matmul_i8_128x32x128` | PASS | — | 34701 | `d464401287d14a30837f30e597d3d10d` | 4 |
| `matmul_i8_128x384x1536` | HANG_60S | kernel_returned_before_timeout | 2341507 | `413197f4a7734553b3ff84a27a7aaa6e` | 6 |
| `matmul_i8_128x384x384` | HANG_60S | kernel_returned_before_timeout | 634107 | `b8bc888700a241eeb584c80436300d90` | 4 |
| `mean_pool_128x384` | PASS | — | 1993646 | `85c620ee24cd449abc60c3a76183601b` | 4 |
| `merge_heads_12x128x32` | PASS | — | 619418 | `14900565db8d4536b4d06604115af40e` | 4 |
| `quantize_128x128` | PASS | — | 517295 | `9f64f5143f764776b3b979913281f5a4` | 4 |
| `quantize_128x1536` | PASS | — | 5679249 | `2677efb66aae442ba6befc3154600cc3` | 6 |
| `quantize_128x32` | PASS | — | 166660 | `7dd8cffaeeb344bc900fda92093565f1` | 4 |
| `quantize_128x384` | PASS | — | 1466583 | `229445971bba49da97d17646042810ac` | 4 |
| `quantize_1536x384` | PASS | — | 17604908 | `2454d42342ef4bd3ac5a2b08dc698106` | 6 |
| `quantize_384x1536` | PASS | — | 17036719 | `62cde96e493a4ab3b6d34a98102099d0` | 4 |
| `quantize_384x384` | PASS | — | 4401069 | `8d9227fdcb844476ac3342e20e936b94` | 4 |
| `quantize_transpose_128x32` | PASS | — | 345402 | `3787048ef7334e65bb91ca00379cb69f` | 4 |
| `softmax_12x128x128` | PASS | — | 63181551 | `2b266b1f656a4c4cb7ce7a1265136ee9` | 6 |
| `split_heads_128x384` | PASS | — | 610934 | `91d5c2f17d7d458fa6ee0ead1fb5b7ff` | 6 |

`PASS` requires scheduler `exited`, application exit code zero, complete adapter returns and cycles, validation execution, the expected checked count and zero errors. `HANG_60S` means the application execution deadline expired; `TRAP` is reported separately. `EXITED_FAIL` means normal application termination failed the validation contract.

For a timeout, `kernel_returned_before_timeout` means the typed adapter completed and a cycle count was captured; whole-launch completion and numerical validation remain unconfirmed. `no_completed_kernel_cycle` means no adapter completion was observed. Neither connection failures nor queue/loading time are counted as application hangs.

Kernel cycles measure the complete typed adapter, including program-grid dispatch, DMA and final fence. They exclude input initialization, scalar validation and UART. A task with no completed adapter has no measured kernel cycle count.

Per-case evidence is under `build/board-runs/fpga112-60s-20261103/<operator>/`: immutable image, SHA256 in `result.json`, ELF audit, board configuration, scheduler `task.json`, client stdout/stderr and `uart.log`. Missing final task/UART files indicate an unconfirmed or unexecuted submission.
