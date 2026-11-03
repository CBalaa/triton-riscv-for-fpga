import triton
import triton.language as tl


@triton.jit
def embedding_add(
    Word,
    PositionIds,
    TypeIds,
    Position,
    Type,
    Out,
    ROWS: tl.constexpr,
    WIDTH: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    valid = col < WIDTH
    position = tl.load(PositionIds + row)
    segment = tl.load(TypeIds + row)
    w = tl.load(Word + row * WIDTH + col, valid, other=0.0)
    p = tl.load(Position + position * WIDTH + col, valid, other=0.0)
    s = tl.load(Type + segment * WIDTH + col, valid, other=0.0)
    tl.store(Out + row * WIDTH + col, (w + p) + s, valid)


KERNEL_CONFIG = {
    "family": "embedding_add",
    "kernel": "embedding_add",
    "arguments": [
        {"name": "Word", "rank": 2, "dtype": "f32"},
        {"name": "PositionIds", "rank": 1, "dtype": "i32"},
        {"name": "TypeIds", "rank": 1, "dtype": "i32"},
        {"name": "Position", "rank": 2, "dtype": "f32"},
        {"name": "Type", "rank": 2, "dtype": "f32"},
        {"name": "Out", "rank": 2, "dtype": "f32"},
    ],
    "shape": [128, 384],
    "argument_shapes": [[128, 384], [128], [128], [512, 384], [2, 384], [128, 384]],
    "constexprs": {"ROWS": 128, "WIDTH": 384, "BLOCK": 512},
    "grid": [128, 1, 1],
    "roles": ["absolute_position_embeddings", "token_type_embeddings"],
    "outputs": [5],
    "checked_elements": 49152,
    "metadata": {
        "model": "thenlper/gte-small",
        "batch": 1,
        "sequence_length": 128,
        "target": "nr-fpga",
    },
}
