# ViT operator FPGA results — 2026-10-27

All 40 inventory operators were submitted sequentially through the user-provided `../run_bin.py`: **FAIL=4, PASS=22, TIMEOUT=14**.
Test interval (UTC): `2026-10-27T07:03:18.401373+00:00` through `2026-10-27T07:41:26.417959+00:00`.

## Execution and interpretation

- Environment: `boscame` conda; Buddy/LLVM/runtime from this repository’s `buddy-mlir` checkout.
- Scheduler endpoint: local SSH forward `127.0.0.1:18765` to `fpga`; actual task node `local`, FPGA4. No tunnel restart was required.
- UART reports 5.5296 MHz. The scheduler evidence does not expose a bitstream path or revision; no bitstream revision is inferred.
- Execution deadline: 120 s as configured in `run_bin.py`; image load/reset count toward this budget, queue wait does not. UART output does not extend it.
- PASS requires real adapter return, full-output validation completion, zero errors and all expected cycle records. FAIL means the numerical oracle ran and found incorrect results.
- TIMEOUT means the 120 s deadline expired after application UART output. HANG means the deadline expired with no UART output. A kernel that has not returned has no completed cycle count.
- Cycles are E6 `rdcycle` around the entire typed adapter/grid dispatch, DMA and final fence. Input initialization, scalar reference, result comparison and UART are excluded. They are one measured invocation, not a repeated-run average.
- Each case uses its original deterministic `launch.c` fixture. This is operator validation, not complete ViT inference or model accuracy.

## Operator results

| Operator | Status | E6 cycles | Checked | Errors | Max absolute error | Oracle |
|---|---|---:|---:|---:|---:|---|
| add_197x768 | PASS | 2,887,892 | 151296 | 0 | 0.0 | on-board C reference |
| add_1x1000 | PASS | 21,441 | 1000 | 0 | 0.0 | on-board C reference |
| add_1x768 | PASS | 15,231 | 768 | 0 | 0.0 | on-board C reference |
| attention_pv_12x197x64x197 | TIMEOUT | unavailable | — | — | — | embedded C reference |
| attention_qk_12x197x197x64 | TIMEOUT | unavailable | — | — | — | embedded C reference |
| bias_add_196x768 | PASS | 3,332,983 | 150528 | 0 | 0.0 | on-board C reference |
| bias_add_197x3072 | PASS | 7,844,898 | 605184 | 0 | 0.0 | on-board C reference |
| bias_add_197x768 | PASS | 3,350,023 | 151296 | 0 | 0.0 | on-board C reference |
| bias_add_1x1000 | PASS | 25,253 | 1000 | 0 | 0.0 | on-board C reference |
| bias_add_1x768 | PASS | 17,594 | 768 | 0 | 0.0 | on-board C reference |
| dequantize_196x768 | FAIL | 5,795,579 | 150528 | 150381 | 1063410.375 | on-board C reference |
| dequantize_197x3072 | FAIL | 23,303,636 | 605184 | 604592 | 1065324.5 | on-board C reference |
| dequantize_197x768 | FAIL | 5,867,669 | 151296 | 151148 | 1063410.375 | on-board C reference |
| dequantize_1x1000 | FAIL | 40,513 | 1000 | 999 | 1002781.0 | on-board C reference |
| gelu_197x3072 | PASS | 242,708,461 | 605184 | 0 | 5.960464477539063e-08 | on-board C reference |
| layernorm_197x768 | PASS | 12,882,541 | 151296 | 0 | 2.384185791015625e-07 | on-board C reference |
| layernorm_1x768 | PASS | 66,368 | 768 | 0 | 2.384185791015625e-07 | on-board C reference |
| linear_196x768x768 | TIMEOUT | unavailable | — | — | — | embedded C reference |
| linear_197x3072x768 | TIMEOUT | unavailable | — | — | — | embedded C reference |
| linear_197x768x3072 | TIMEOUT | unavailable | — | — | — | embedded C reference |
| linear_197x768x768 | TIMEOUT | unavailable | — | — | — | embedded C reference |
| linear_1x1000x768 | TIMEOUT | unavailable | — | — | — | on-board C reference |
| matmul_i8_196x768x768 | PASS | 8,923,258 | 150528 | 0 | 0.0 | embedded C reference |
| matmul_i8_197x3072x768 | PASS | 35,769,861 | 605184 | 0 | 0.0 | embedded C reference |
| matmul_i8_197x768x3072 | PASS | 25,724,284 | 151296 | 0 | 0.0 | embedded C reference |
| matmul_i8_197x768x768 | PASS | 8,891,752 | 151296 | 0 | 0.0 | embedded C reference |
| matmul_i8_1x1000x768 | PASS | 1,786,783 | 1000 | 0 | 0.0 | on-board C reference |
| matmul_i8_overwrite_196x768x768 | PASS | 6,364,359 | 150528 | 0 | 0.0 | embedded C reference |
| matmul_i8_overwrite_197x3072x768 | PASS | 25,760,042 | 605184 | 0 | 0.0 | embedded C reference |
| matmul_i8_overwrite_197x768x3072 | PASS | 22,636,326 | 151296 | 0 | 0.0 | embedded C reference |
| matmul_i8_overwrite_197x768x768 | PASS | 6,396,499 | 151296 | 0 | 0.0 | embedded C reference |
| matmul_i8_overwrite_1x1000x768 | PASS | 1,394,120 | 1000 | 0 | 0.0 | on-board C reference |
| patch_embed_196x768 | TIMEOUT | unavailable | — | — | — | embedded C reference |
| quantize_1000x768 | TIMEOUT | unavailable | — | — | — | on-board C reference |
| quantize_196x768 | TIMEOUT | unavailable | — | — | — | on-board C reference |
| quantize_197x3072 | TIMEOUT | unavailable | — | — | — | on-board C reference |
| quantize_197x768 | TIMEOUT | unavailable | — | — | — | on-board C reference |
| quantize_1x768 | TIMEOUT | unavailable | — | — | — | on-board C reference |
| quantize_768x768 | TIMEOUT | unavailable | — | — | — | on-board C reference |
| softmax_12x197x197 | PASS | 242,654,538 | 465708 | 0 | 1.3969838619232178e-09 | on-board C reference |

## Validation contracts

The original launcher’s numerical tolerances are unchanged. Float comparisons use `abs(actual-expected) <= atol + rtol*abs(expected)`.

| Family / cases | atol | rtol |
|---|---:|---:|
| i8 matmul and overwrite matmul | exact i32 | exact i32 |
| add_1x768, add_197x768, bias_add_*x768 | 1e-6 | 1e-6 |
| add_1x1000, bias_add_1x1000, bias_add_197x3072 | 1e-5 | 1e-5 |
| dequantize | 1e-5 | 1e-6 |
| LayerNorm and Softmax | 4e-4 | 4e-4 |
| GELU | 3e-4 | 3e-4 |
| dense float Linear and composed patch embedding | 3e-3 | 3e-3 |
| QK / PV attention | 5e-3 | 5e-3 |
| quantize scale; quantize i8 output | 1e-6; exact i8 | 1e-6; exact i8 |

Expensive cases use `GOLDEN=1`: the host evaluates only the original scalar expected formula, with a deliberately no-op kernel adapter. The FPGA runs the real adapter on the original inputs and traverses every output against the embedded result. The submitted-image reference bytes and source/hash provenance were checked. The two 1x1000 integer matmuls and `linear_1x1000x768` used their original on-board oracle images, even though their current build directories were later rebuilt in GOLDEN mode. Archived submitted images are authoritative.

The separate 18-case host fixture/checker validation accepts independent expected bytes and rejects one intentionally corrupted output for each case. Those checks establish initialization/checker behavior, not FPGA success.

## Failures and unfinished kernels

The four dequantize adapters returned and completed their full checks. Their error counts equal the number of nonzero expected elements; their maximum errors are around one million, well beyond the original tolerance. Root cause is not established by these runs.

Timeouts retain their last UART marker in each `result.json` and `uart.log`. No completed cycle is synthesized from wall time or the timeout budget. No diagnostic probe is substituted for a production result.

## Reproduction and evidence

Build/run examples are in [README.md](README.md#build-and-run-on-fpga). Machine-readable results and the exact submitted images are under [build/board-runs/20261027-suite/](build/board-runs/20261027-suite/). The generated [REPORT.md](build/board-runs/20261027-suite/REPORT.md) and [report.json](build/board-runs/20261027-suite/report.json) include the complete result set. Each case preserves console output, relative-time events, UART, scheduler task details, ELF audit, image hash and validation mode.

| Operator | Scheduler task ID | Submitted image SHA256 |
|---|---|---|
| add_197x768 | `472488bbd7404a39a1288efae5966cb6` | `c9fb34ba576faf6afe4d69e505d63ea18e926bc7d96c3968f1790d148776e3b7` |
| add_1x1000 | `805167b5d0f44f5b9cbc25b61ee507d7` | `b2bddde12085e5ff724a24dc0e26a66311a701f780e11bcdef9c4e2c1c596694` |
| add_1x768 | `dd1bdeaac52e46aa8dbe86b05889b8b6` | `0c7a6a0f8ba5964468e0a6f8b873d108fa52dcb70dd1b321f54aaa2b785258c0` |
| attention_pv_12x197x64x197 | `587c31833f514c0c87a3ed587c5355f7` | `8fd8f18c0d4597a28291a4f68711ff8c5c0711c39d4d13aa513d16c249949df8` |
| attention_qk_12x197x197x64 | `397628745b684e4c8a0ef2e0421a3e21` | `ed7128414658d89930ae7157020ac5b95dff0add135d40935187070c699be305` |
| bias_add_196x768 | `9f237b5472e445ce9a78f67f350b18c4` | `4526c0325ba616f954e6444db3ce85eb81b838bba70de27c8e7ffe1a4c5f7aff` |
| bias_add_197x3072 | `3cc8f9d16af04aadb5eddba8cfa5d12d` | `595e25f7ff4b739c61276e14ae895d358df61a02ae43af9b6a3a0a414b9dd3da` |
| bias_add_197x768 | `b15acbba139e4926a7eca00a9df01bc4` | `551b54e2471300aaae4bed6ad9b052513f14b22b8afcf8e3f50bebc75dac77de` |
| bias_add_1x1000 | `e366635560dd4f709c1ddafe0283b1c9` | `77d0711587c03d6db4c9f9ef66b3f60c59b920fe14d20f646700ddf2f16df2c3` |
| bias_add_1x768 | `db7074a46b7147bd9581b112a6c99ae3` | `a027e21efaa31b5dd6144b727b8b1277b338af8e02cbbb82a9c75b3614c1a990` |
| dequantize_196x768 | `9eb38037fe234603a4ab14d786eab26e` | `e9545d516d9f294d172acb7f1a298799a6e1eceb5e435885025098346fdd8b2d` |
| dequantize_197x3072 | `84c473c1f42d4c5698dc551007cf1389` | `d38624acc3007c62d36c575b3d5fa45cb287f87292ff6e3b9cd02fb301ac0d3e` |
| dequantize_197x768 | `009bf9ef943f4e92973bfd405bffbb06` | `4cf7893a388ccba1484da2f91edda676066d44e0ad002efaa708bc559c2c1c78` |
| dequantize_1x1000 | `8a01c377007f44c292bd871c0c21cc9b` | `c787fa555fefc18fa60473a2ff339c99ffce51808df3ac03346f4387171524c9` |
| gelu_197x3072 | `ad8d108d03424ff998df7b336f3476a5` | `ffb8495a49bbbb216b3fc150f33a940c3e915d4878c883be75866eb0216e3772` |
| layernorm_197x768 | `5f76e1c18119419bbf503aba8cc757ae` | `657a1c16785dc38c4b004d061c22aec7670579953fa0956f13b352e057c17daf` |
| layernorm_1x768 | `e62b3f7d7af44a9f912f9f93b6fdf4e1` | `776a683810fcead9a03cabfc8d7d2fa563cae915b734a8409c0f83fb1749af52` |
| linear_196x768x768 | `3ae16045421b4f3bb6659aed37b227c4` | `9109f1c6d2dde6682f84faf39b8df998de74ae4302ed84f46f973c67017a2911` |
| linear_197x3072x768 | `7d90bfffcf7a4572834432234ef05928` | `26b01a8aaccd06032188a073cd0a05f12cc11bfac78d08cf4076279e114912f8` |
| linear_197x768x3072 | `6d116a15133046c89266f1b184450549` | `31da044d7e8f20442480cd4f01e5206a57a50a4ca59d971a3c3b5d88dcb6c2a2` |
| linear_197x768x768 | `28e4a53e3c424dacb16797aa588e233a` | `c87b1da1d74ebc1c981a80aca718aae4f7a2dc5b6ae72f8745eca9a1bb800748` |
| linear_1x1000x768 | `ec3c3e1f5f474e80b978f5c7bed22860` | `f63ab8361a8e45b1b5eafdaae889d5cea4975407e9743f12effede40ee7cc5f4` |
| matmul_i8_196x768x768 | `45fb199781a14ef9854f5bd3356d8ece` | `b54d5c2eaf144d67fdf67745d59b3a680a53ed7839ff4676eeb6f1d72af778be` |
| matmul_i8_197x3072x768 | `c82a3002a51d41bebea84afb27e57a10` | `12e11d9f73a270ee7b55d93f4d1b213abab629722092d435dcac5d955fb67145` |
| matmul_i8_197x768x3072 | `17228991f0694b36b6a61d9cf270c553` | `9d76624cf251047e005d060835dcc6ec199732a33903d94f24e1ced6d5f5336b` |
| matmul_i8_197x768x768 | `5d3f47aab87f45eca03992e192a76466` | `5bf02825e013a3c6bbef8bba4ba1fc57d14d903a263b14f02701fa4ee96eb210` |
| matmul_i8_1x1000x768 | `703496086bfb4c91a1d569a6de4b65da` | `118c03c4e5f97c7b749b1f1e8452fa97dd38fbbdf15858f68d84c91c09fee5e6` |
| matmul_i8_overwrite_196x768x768 | `d7df78c987444588bd614222ec85a78f` | `57781d7b32992261a9d43c0871fd5eb02b3c420f492dc5e4523fd5b900ffeb12` |
| matmul_i8_overwrite_197x3072x768 | `ddae45fa7408482b84b9cca2bdae28da` | `7ad99870a81c990c83b09364d3109b784cc778a18beb4292881095143085e785` |
| matmul_i8_overwrite_197x768x3072 | `7e1b837551e84d54ba2fb92235e63806` | `9700b92cf43acc003582fcddf02491ec0ed84339267ed4677d248aef715513f1` |
| matmul_i8_overwrite_197x768x768 | `9faf2fb8feaf4c4ca4619cb723b9f188` | `27f3df05163d885cbec4d8c651780f4397a5d6e4853ba81f4741ebbe9f3ac8f3` |
| matmul_i8_overwrite_1x1000x768 | `98444d98d5da40878a9021f17cd53661` | `1d686e0f0e11ca85db5ae6937d4fe83d5b7f6da31a461e27d787d251a8118f8b` |
| patch_embed_196x768 | `8a8608818bf44d289e576f06a127fe67` | `659e37a979e14d843f116c681f5e0c29e2c940eaeffe2c0215b0d5bf95a690dc` |
| quantize_1000x768 | `c460179bf38a4e60a196d83dd18e61a9` | `bbab728c004dc2f9f8fd6d98aa0077dce320a00b9dcf3fb14c51a8cd223cd267` |
| quantize_196x768 | `7e5898e5c5ff4f1ca1bfc85230e65feb` | `307d8b78251e51fe5ec0872554c1a40b30e46217c91e30d72a76bbefd8ce896d` |
| quantize_197x3072 | `9cb5390ce3c24f4ebc4d9cc782b9744e` | `37f695245baf94a962a3b13ffe7c4d56fe09ec0a846c7444600e8d6d2e151bf5` |
| quantize_197x768 | `85884913e30b4046bf496e891dbf8b76` | `27451ee6d684fadb7fb82b70d2f8f4e64d3e7ec96dd1d38ac6ff6a138b635f79` |
| quantize_1x768 | `17be9872a7104d0e94a9d5708c63220d` | `10a7ba64b08e3d9014069cb96ff8318a2b5c1f53a9303667e18525b398540038` |
| quantize_768x768 | `a41bcd2d899c44208e077f0899b6bd0b` | `0bdc5ad13bd0158dbfe7b78bbddb150839c7164db913720893dbc6bae49643ed` |
| softmax_12x197x197 | `4fee9aa2fdcf49f5abfe81329f23d64f` | `8c999813622dae1e161e70eb801fe03dbc77f815ae302c7a0d7cffabb5f73889` |
