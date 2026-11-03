import triton
import triton.language as tl


@triton.jit
def merge_heads(
    X,
    Out,
    LENGTH: tl.constexpr,
    WIDTH: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK: tl.constexpr,
):
    h = tl.program_id(0)
    row = tl.program_id(1)
    c = tl.arange(0, BLOCK)
    value = tl.load(X + (h * LENGTH + row) * HEAD_DIM + c)
    tl.store(Out + row * WIDTH + h * HEAD_DIM + c, value)


KERNEL_CONFIG = {
    "family": "merge_heads",
    "kernel": "merge_heads",
    "arguments": [
        {"name": "X", "rank": 3, "dtype": "f32"},
        {"name": "Out", "rank": 2, "dtype": "f32"},
    ],
    "argument_shapes": [[12, 128, 64], [128, 768]],
    "shape": [128, 768],
    "constexprs": {"LENGTH": 128, "WIDTH": 768, "HEAD_DIM": 64, "BLOCK": 64},
    "grid": [12, 128, 1],
    "roles": ["xlnet_model"],
    "metadata": {"model": "xlnet/xlnet-base-cased"},
}
