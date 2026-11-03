import triton
import triton.language as tl


@triton.jit
def embedding(
    Ids,
    Table,
    Out,
    SEQ: tl.constexpr,
    WIDTH: tl.constexpr,
    VOCAB: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    c = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    token = tl.load(Ids + row)
    # Host/model contract requires 0 <= token < VOCAB before entry.
    value = tl.load(Table + token * WIDTH + c)
    tl.store(Out + row * WIDTH + c, value)


KERNEL_CONFIG = {
    "family": "embedding",
    "kernel": "embedding",
    "arguments": [
        {"name": "Ids", "rank": 1, "dtype": "i32"},
        {"name": "Table", "rank": 2, "dtype": "f32"},
        {"name": "Out", "rank": 2, "dtype": "f32"},
    ],
    "argument_shapes": [[128], [32000, 768], [128, 768]],
    "shape": [128, 768],
    "constexprs": {"SEQ": 128, "WIDTH": 768, "VOCAB": 32000, "BLOCK": 128},
    "grid": [128, 6, 1],
    "roles": ["xlnet_model"],
    "metadata": {"model": "xlnet/xlnet-base-cased"},
}
