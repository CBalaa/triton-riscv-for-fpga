# E5 scheduler returned-output validation — fpga112, 2026-11-03

**30/30 standalone operators PASS.** The scheduler captured and downloaded all 38 output BIN snapshots (3,313,184 elements per complete local replay). Every downloaded file matched its declared dtype, shape, byte length and SHA256 and passed the unchanged C launcher oracle in two complete native replays.

All jobs finished with `state=completed`, `execution_status=exited`, `app_exit_code=0` and `exit_code=0`. Each image dispatched its complete kernel grid once, captured the full output, and returned before the scalar numerical oracle; validation ran locally over the actual downloaded bytes. Quantizers exported both `Q` and `Scale`. Every case also rejected a missing BIN, a one-byte-short BIN and a corrupted numeric value.

The FPGA column is the port actually assigned by fpga112. Kernel cycles cover the typed adapter, complete grid, DMA and final fence; they exclude fixture initialization, output capture, scheduler loading, download and local validation. The application deadline was 180 seconds. This validates the individual operators, not target execution of the complete E5 model.

| Operator | Output tensors | Elements per replay | Kernel cycles | FPGA | Scheduler job | Result |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| `add_128x384` | 1 | 49,152 | 551825 | 4 | `dec81ca1dc464d07ae13dd6cedb1a605` | PASS |
| `attention_mask_12x128x128` | 1 | 196,608 | 2782521 | 4 | `022c251a40cd40ac903fe20d52900035` | PASS |
| `bias_add_128x1536` | 1 | 196,608 | 1845815 | 4 | `bde9501ed4114009b91220ad501d1810` | PASS |
| `bias_add_128x384` | 1 | 49,152 | 489153 | 6 | `6ff770be93ff4f7bace157f2964aa1b3` | PASS |
| `dequantize_128x128` | 1 | 16,384 | 272448 | 6 | `d192596f92ec4af69aaaa1ddd23c8dd3` | PASS |
| `dequantize_128x1536` | 1 | 196,608 | 2262915 | 4 | `95f57ee2a414427d96c5961f8df0a2a6` | PASS |
| `dequantize_128x32` | 1 | 4,096 | 118712 | 7 | `3b3c632e134248069b2ee67b9b1a8b8a` | PASS |
| `dequantize_128x384` | 1 | 49,152 | 619549 | 6 | `46039fbde57c4b5baffd297cff6ac3c8` | PASS |
| `embedding_add_128x384` | 1 | 49,152 | 658826 | 6 | `3c8c4d6e87654981b256ced6117dd7aa` | PASS |
| `embedding_lookup_128x384` | 1 | 49,152 | 334702 | 4 | `872e82937f8b4ccb94518bab59ee0383` | PASS |
| `gelu_128x1536` | 1 | 196,608 | 36817128 | 4 | `40702a6612714718871c21e07ef36e37` | PASS |
| `l2_normalize_1x384` | 1 | 384 | 15454 | 4 | `4b6656843f344555a64c138a73cb9964` | PASS |
| `layernorm_128x384` | 1 | 49,152 | 3545779 | 6 | `336ab672b6d1482ab42bd694bc32c87a` | PASS |
| `matmul_i8_128x128x32` | 1 | 16,384 | 7618903 | 7 | `04e11d1c455648a8a8503cf8ef885d5f` | PASS |
| `matmul_i8_128x1536x384` | 1 | 196,608 | 2531386 | 7 | `213b8f69fc8a4d2a939537ee0196908e` | PASS |
| `matmul_i8_128x32x128` | 1 | 4,096 | 34684 | 7 | `bc2594fd706743df901eec5b29e9dba8` | PASS |
| `matmul_i8_128x384x1536` | 1 | 49,152 | 2341292 | 4 | `55043dd1f7954cf29778c670af752339` | PASS |
| `matmul_i8_128x384x384` | 1 | 49,152 | 634158 | 4 | `574f9f7df8b44082b389f5dc150d9a13` | PASS |
| `mean_pool_128x384` | 1 | 384 | 1993628 | 6 | `ca47d65c85e24d1bb76033122f9ee16b` | PASS |
| `merge_heads_12x128x32` | 1 | 49,152 | 609678 | 7 | `92ab59bf737d4027842741f1d393fbfa` | PASS |
| `quantize_128x128` | 2 | 16,512 | 509887 | 7 | `de1ba69055b84d8bb2048f5251dc9018` | PASS |
| `quantize_128x1536` | 2 | 196,736 | 5672808 | 7 | `afaac2be30ef4b8ba2ee4124fb52e6c4` | PASS |
| `quantize_128x32` | 2 | 4,224 | 160444 | 7 | `366592c254284e6f8c967162e6ac638c` | PASS |
| `quantize_128x384` | 2 | 49,280 | 1460877 | 4 | `e57e09bcc5014b1883cf594b5d7728d5` | PASS |
| `quantize_1536x384` | 2 | 591,360 | 17521514 | 4 | `dc2a0ecbcbea448bb56115ebee149ec3` | PASS |
| `quantize_384x1536` | 2 | 590,208 | 17017474 | 7 | `4fa947655a504c2f86a9399f73980f22` | PASS |
| `quantize_384x384` | 2 | 147,840 | 4381593 | 7 | `68c1a4b54f4e4df8a3c0da549fcf9382` | PASS |
| `quantize_transpose_128x32` | 2 | 4,128 | 345422 | 4 | `868dce05264a474d9cf26bb696928d78` | PASS |
| `softmax_12x128x128` | 1 | 196,608 | 63181085 | 4 | `b4d6f8fc37644a4f85afd9d5fd78cce9` | PASS |
| `split_heads_128x384` | 1 | 49,152 | 609698 | 4 | `9753f152304447f5ad3a22bb8cac7f28` | PASS |

The three dense matmuls that exhausted the original 60-second board-oracle deadline now have complete downloaded-output numerical evidence. This does not rewrite their original timeout records or claim the original board oracle finished in 60 seconds. The unchanged smallest original image separately completed under a 300-second deadline in approximately 145 seconds. Its original timeout occurred after the adapter returned, during the remaining launcher validation work; the original statuses and measured kernel cycles remain in [BOARD_RESULTS.md](BOARD_RESULTS.md).

Evidence is under [the isolated returned-output run directory](build/board-runs/fpga112-returned-outputs-20261103/). Each case preserves `image.bin`, `image.elf`, `elf-audit.json`, `board_config.json`, `frontend.json`, frozen source files, submission/client logs, `task.json`, `uart.log`, `result.json`, downloaded `artifacts/*.bin`, scheduler `artifacts/manifest.json`, and `local-validation/report.json` with the two full replay records and three negative regressions. The local verifier records the original launcher/support source hashes, downloaded BIN hashes, scheduler job ID and verified manifest hash.

The original 60-second images, UART and results remain separate in [BOARD_RESULTS.md](BOARD_RESULTS.md). No original kernel or original numerical launcher was replaced to obtain this result. Reproduction commands are in [README.md](README.md).
