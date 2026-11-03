`e5_infer` calls the 30 generated typed adapter families for the complete
12-layer encoder and masked-mean/L2 output. It makes 1,649 adapter calls. Matrix
products always use the overwrite INT8 family, including per-head attention.
All tensor math is dispatched to those adapters; the C glue only validates
input IDs, assigns descriptors and optionally copies the final hidden state.

The caller supplies a contiguous 64-byte-aligned arena of at least
`e5_workspace_bytes()` bytes. Weight arrays, IDs, output arrays and arena must
be disjoint and remain valid throughout the call. Linear weights keep their
Hugging Face `[output,input]` layout. Buffers for query/key/value, attention
probabilities, residual states and quantization temporaries are reused after
their consumers finish. The entry returns `-1` before dispatch for invalid
IDs, type/position IDs, masks, pointer alignment or insufficient workspace.
At least one token must be valid.

Populate the `E5Weights` fields from these checkpoint names:

| C field | Hugging Face state dictionary key |
| --- | --- |
| `word_embeddings` | `embeddings.word_embeddings.weight` |
| `position_embeddings` | `embeddings.position_embeddings.weight` |
| `type_embeddings` | `embeddings.token_type_embeddings.weight` |
| `embedding_norm` | `embeddings.LayerNorm.{weight,bias}` |
| `layers[i].query/key/value` | `encoder.layer.i.attention.self.{query,key,value}.{weight,bias}` |
| `layers[i].attention_output` | `encoder.layer.i.attention.output.dense.{weight,bias}` |
| `layers[i].attention_norm` | `encoder.layer.i.attention.output.LayerNorm.{weight,bias}` |
| `layers[i].intermediate` | `encoder.layer.i.intermediate.dense.{weight,bias}` |
| `layers[i].output` | `encoder.layer.i.output.dense.{weight,bias}` |
| `layers[i].output_norm` | `encoder.layer.i.output.LayerNorm.{weight,bias}` |

Position IDs normally contain `0..127`; token type IDs normally contain zero.
Apply `query: ` or `passage: ` before tokenization, include CLS/SEP and pad to
128. Token IDs/masks/types/positions are INT32 for this C ABI. Sentence output
contains 384 FP32 values. BERT's pooler weights are deliberately unused.

`../tools/build_model_board.py` provides the complete target integration. It
binds all real typed adapters and their original emitted kernel objects,
packs the pinned checkpoint, and links the NH/E6 image with
`../board_model.ld`. Cache/fence wrappers bracket every dispatch. An optional
`E5_TARGET_CHECKPOINTS` hook copies all hidden states after the embedding
LayerNorm and each encoder layer; it adds no replacement arithmetic.

The host mock checker establishes all 1,649 calls and their ABI, alignment,
buffer bounds, provenance and call order. It does not establish numerical
target execution. The separate model runner downloads actual sentence,
hidden-state and 13-layer checkpoint BINs for full local comparison, and
`--validate-only` repeats that check without invoking the scheduler.
It requires a previously executed emitted reference and writes results into a
new review directory; archived execution evidence is never overwritten.
Complete-output byte equality against locally executed emitted scalar LLVM
and exact INT8/INT32 dot operations is the numerical acceptance criterion.
The normal FPGA compiler uses the C++ scalar FP32 max workaround. Build, run,
download and local verification commands are in the [main README](../README.md).
