The pinned source is `intfloat/e5-small-v2` at
`ffb93f3bd4047442299a41ebb6fa998a38507c52`, retrieved through
`https://hf-mirror.com`. `config.json` is the upstream model configuration;
`manifest.json` records the immutable revision and E5 output contract.

`python tools/fetch_assets.py` fetches the immutable revision's safetensors
checkpoint and BERT tokenizer into `assets/`, verifies the configuration and
records file SHA256 hashes. `--metadata-only` skips checkpoint download.
No weights are checked into this example.

Inputs are English text with `query: ` or `passage: ` prefixes, WordPiece
tokenization with special tokens, and padding/truncation to 128 for this
specialization. Upstream's maximum is 512. Output is the masked mean of final
encoder tokens followed by L2 normalization, never BERT's pooler output.
