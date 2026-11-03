import triton
import triton.language as tl


@triton.jit
def l2_normalize(X, Out, WIDTH: tl.constexpr, BLOCK: tl.constexpr, EPS: tl.constexpr):
    col = tl.arange(0, BLOCK)
    valid = col < WIDTH
    x = tl.load(X + col, valid, other=0.0)
    norm = tl.sqrt(tl.sum(x * x, 0))
    denominator = tl.maximum(norm, EPS)
    tl.store(Out + col, x / denominator, valid)


KERNEL_CONFIG = {
    "family": "l2_normalize",
    "kernel": "l2_normalize",
    "arguments": [
        {"name": "X", "rank": 1, "dtype": "f32"},
        {"name": "Out", "rank": 1, "dtype": "f32"},
    ],
    "shape": [384],
    "argument_shapes": [[384], [384]],
    "constexprs": {"WIDTH": 384, "BLOCK": 512, "EPS": 1e-12},
    "grid": [1, 1, 1],
    "roles": ["optional_unit_sentence_embedding"],
    "outputs": [1],
    "checked_elements": 384,
    "metadata": {
        "model": "thenlper/gte-small",
        "batch": 1,
        "sequence_length": 128,
        "target": "nr-fpga",
    },
}
