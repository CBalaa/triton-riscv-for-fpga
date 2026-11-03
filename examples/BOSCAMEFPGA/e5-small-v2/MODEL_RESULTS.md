# Complete E5 model validation on fpga112

The complete 12-layer E5 model passed actual FPGA execution and full downloaded
output validation on 2026-11-03. All three returned BINs match the local emitted
implementation reference **byte for byte**, including every hidden-state
checkpoint. The original frozen PyTorch W8A8 acceptance checks also pass.

| Execution evidence | Value |
| --- | --- |
| Selected server / actual FPGA | fpga112 / 6 |
| Scheduler task ID | `e39c7be6daea49c5b72c71bd5cfa602e` |
| State / execution status | `completed` / `exited` |
| Application / runtime exit codes | `0` / `0` |
| Image bytes | 137,044,992 |
| Image SHA256 | `ca76e585921d67c1d2088c7379550273122e7017d4cc0024b77ef8a2c9da0d67` |
| DDR image readback | SHA matches uploaded image |
| Real typed kernel calls / complete checkpoints | 1,649 / 13 |
| Model cycles / summed kernel cycles | 3,500,492,248 / 3,407,672,428 |
| Scheduler execution time | 636.369 seconds, including 2.450 seconds of image loading |
| UTC creation / completion | 07:15:49 / 07:26:27 on 2026-11-03 |

The foreground scheduler client stayed alive through terminal completion.
UART contains valid `@@FPGA_START`, all 13 layer progress markers, three
`@@FPGA_OUTPUT` declarations, and `@@FPGA_RESULT` with application exit code 0.
There was one model submission; no model trap or hardware retry was needed.

The model-cycle interval brackets `e5_model_launch()` on E6. It includes input
and weight validation, workspace/descriptor binding, all adapter dispatches,
13 complete checkpoint copies and their progress markers, and the final
196,608-byte hidden-state copy. It excludes platform startup, image loading,
NH UART/output declarations and scheduler download/local validation. Snapshot
cycles were not measured separately, so this value is not a pure-inference
measurement and cannot be compared directly with another model's counter that
subtracts snapshots. The summed kernel counter covers the real adapter/kernel
calls and their closing fences; it excludes checkpoints and most C graph
management.

The image uses pinned `intfloat/e5-small-v2` revision
`ffb93f3bd4047442299a41ebb6fa998a38507c52`, with safetensors SHA256
`45bfa60070649aae2244fbc9d508537779b93b6f353c17b0f95ceccb1c5116c1`.
The actual tokenizer input was
`query: how much protein should a female eat`, with 11 valid tokens padded to
128. The target links all 30 original emitted kernel objects and their typed
adapters to the actual C model graph. No host reference supplies target outputs.

| Downloaded tensor | Shape | Elements | Bytes | Emitted-reference max error |
| --- | --- | ---: | ---: | ---: |
| `sentence_embedding` | `[384]` | 384 | 1,536 | 0 |
| `last_hidden` | `[128,384]` | 49,152 | 196,608 | 0 |
| `layer_hidden` | `[13,128,384]` | 638,976 | 2,555,904 | 0 in all 13 layers |
| Total | | 688,512 | 2,754,048 | 0 |

Each downloaded file has its exact declared shape, dtype and byte length; its
SHA256 equals the scheduler artifact manifest. The local implementation
reference executes the same `model_runtime/model.c`, generated checkpoint/input
bindings, original typed adapters and actual emitted scalar `kernel.ll` files
on the host. Matrix products use independently written exact INT8/INT32 C dot
products; their maximum accumulation fits INT32. Host `rsqrtf` uses
`1.0f/sqrtf(x)`. This reference is explicitly host execution, with its source,
input, checkpoint and output hashes preserved. Its exact agreement checks the
whole target computation; independent Hugging Face FP32 and PyTorch W8A8
comparisons separately assess the arithmetic adaptation.

| Complete-output comparison | Max absolute error | RMSE | Cosine |
| --- | ---: | ---: | ---: |
| Sentence vs PyTorch W8A8 | 0.0068060830 | 0.0022457808 | 0.9990316418 |
| Final hidden vs PyTorch W8A8 | 0.1003159285 | 0.0191333546 | 0.9985823456 |
| All layer checkpoints vs PyTorch W8A8 | 0.1416830719 | 0.0138853315 | 0.9996165765 |
| Sentence vs independent HF FP32 | 0.0057844333 | 0.0021503892 | 0.9991121585 |
| Final hidden vs independent HF FP32 | 3.7583963275 | 0.1644089266 | 0.9192406017 |

PyTorch W8A8 and emitted scalar arithmetic have different FP32 reduction/math
evaluation orders. Their small early differences can cross discrete INT8
rounding thresholds in later layers. The per-layer W8A8 maximum differences
start at `2.38e-6`, grow to `0.141683`, and end at `0.100316`; the corresponding
emitted-reference differences are zero in every layer. FP32 differences include
W8A8 quantization error. The full final-hidden comparison includes padded query
states; over the 11 valid tokens its FP32 maximum error is `0.1043904424` and
RMSE `0.0227458101`. All tokens are retained in the complete-output checks.

The original acceptance thresholds remain sentence maximum error `0.01` and
cosine at least `0.999`, final-hidden maximum error `0.2` and RMSE `0.03`,
and sentence norm error at most `1e-4`. The actual sentence norm is
`0.9999999067`. The last checkpoint equals `last_hidden` exactly. Independently
recomputing masked mean and L2 normalization in FP64 from the downloaded hidden
states reproduces the sentence vector within `2.77e-8`. With the additional
emitted reference, all three complete files must also agree byte for byte.

Output SHA256 values:

- Sentence: `873bd9ab82cead4c2b22811c2c1f239a2273b2e19f53e8596d143033dfc138da`
- Final hidden: `9b78a4390bbbf84a3c3c23db969f1bfe77e42fe0024acd04894dc9ca0ac52b59`
- Layer checkpoints: `ce83c58b776299390be397a595f0849769e0d7c4c99cc8d0c83c51b8ac0b8a85`

Detailed evidence is under the ignored directory
`build/model-board-runs/fpga112-checkpoint-query-20261103/`: `task.json`,
`result.json`, `uart.log`, the audited ELF/BIN, original input/reference files,
`artifacts/manifest.json`, all downloaded BINs, `emitted_manifest.json`, frozen
emitted-reference sources and `local-validation.json`.
[MODEL_RESULTS.json](MODEL_RESULTS.json) preserves the portable machine-readable
summary. The earlier standalone operator evidence remains in
[RETURNED_OUTPUT_RESULTS.md](RETURNED_OUTPUT_RESULTS.md).

Reproduce the offline build and local validation without another FPGA run:

```bash
E5=examples/BOSCAMEFPGA/e5-small-v2
PYTHON=python

# Activate the Python/compiler environment from README.md first; pinned assets are required.
make -C "$E5" model-bin PYTHON="$PYTHON"
"$PYTHON" -B "$E5/tools/build_model_host.py"
"$PYTHON" -B "$E5/tools/run_model_board.py" --validate-only \
  --emitted-reference "$E5/build/model-host-emitted" \
  --output "$E5/build/model-board-runs/fpga112-checkpoint-query-20261103"
```

This establishes complete-model implementation correctness for this real input
and fixed batch-1/sequence-128 specialization. It does not establish retrieval
benchmark accuracy, general dynamic shapes, or performance optimization.
