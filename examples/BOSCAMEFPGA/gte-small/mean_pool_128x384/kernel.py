import triton
import triton.language as tl


@triton.jit
def mean_pool(
    X, Mask, Out, SEQ: tl.constexpr, WIDTH: tl.constexpr, BLOCK: tl.constexpr
):
    channel = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = channel < WIDTH
    total = tl.full((BLOCK,), 0.0, tl.float32)
    count = tl.full((), 0.0, tl.float32)
    # Traverse rows with contiguous hidden tiles. Scalar mask values broadcast
    # directly; no strided gather or rank-two broadcast is required.
    for row in range(SEQ):
        keep = tl.load(Mask + row).to(tl.float32)
        x = tl.load(X + row * WIDTH + channel, valid, other=0.0)
        total = total + x * keep
        count = count + keep
    denominator = tl.maximum(count, 1e-9)
    tl.store(Out + channel, total / denominator, valid)


KERNEL_CONFIG = {
    "family": "mean_pool",
    "kernel": "mean_pool",
    "arguments": [
        {"name": "X", "rank": 2, "dtype": "f32"},
        {"name": "Mask", "rank": 1, "dtype": "i32"},
        {"name": "Out", "rank": 1, "dtype": "f32"},
    ],
    "shape": [128, 384],
    "argument_shapes": [[128, 384], [128], [384]],
    "constexprs": {"SEQ": 128, "WIDTH": 384, "BLOCK": 128},
    "grid": [3, 1, 1],
    "roles": ["attention_mask_weighted_mean_pooling"],
    "outputs": [2],
    "checked_elements": 384,
    "metadata": {
        "model": "thenlper/gte-small",
        "batch": 1,
        "sequence_length": 128,
        "target": "nr-fpga",
    },
}
