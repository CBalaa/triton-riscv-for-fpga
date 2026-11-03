import triton
import triton.language as tl


@triton.jit
def head_bias_add(
    X, Bias, Out, LENGTH: tl.constexpr, HEAD_DIM: tl.constexpr, BLOCK: tl.constexpr
):
    h = tl.program_id(0)
    row = tl.program_id(1)
    c = tl.arange(0, BLOCK)
    value = tl.load(X + (h * LENGTH + row) * HEAD_DIM + c)
    bias = tl.load(Bias + h * HEAD_DIM + c)
    tl.store(Out + (h * LENGTH + row) * HEAD_DIM + c, value + bias)


KERNEL_CONFIG = {
    "family": "head_bias_add",
    "kernel": "head_bias_add",
    "arguments": [
        {"name": "X", "rank": 3, "dtype": "f32"},
        {"name": "Bias", "rank": 2, "dtype": "f32"},
        {"name": "Out", "rank": 3, "dtype": "f32"},
    ],
    "argument_shapes": [[12, 128, 64], [12, 64], [12, 128, 64]],
    "shape": [12, 128, 64],
    "constexprs": {"LENGTH": 128, "HEAD_DIM": 64, "BLOCK": 64},
    "grid": [12, 128, 1],
    "roles": ["xlnet_model"],
    "metadata": {"model": "xlnet/xlnet-base-cased"},
}
