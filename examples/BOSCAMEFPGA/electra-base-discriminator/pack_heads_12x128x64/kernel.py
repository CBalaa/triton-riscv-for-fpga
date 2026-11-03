import triton
import triton.language as tl

@triton.jit
def pack_heads(X, Out, HEADS:tl.constexpr, SEQ:tl.constexpr, DIM:tl.constexpr, BLOCK:tl.constexpr):
    row = tl.program_id(0)
    head = tl.program_id(1)
    col = tl.arange(0, BLOCK)
    x = tl.load(X + row * HEADS * DIM + head * DIM + col, col < DIM, other=0.0)
    tl.store(Out + head * SEQ * DIM + row * DIM + col, x, col < DIM)

KERNEL_CONFIG = {'family': 'pack_heads',
 'kernel': 'pack_heads',
 'arguments': [{'name': 'X', 'rank': 2, 'dtype': 'f32'},
               {'name': 'Out', 'rank': 3, 'dtype': 'f32'}],
 'argument_shapes': [[128, 768], [12, 128, 64]],
 'shape': [12, 128, 64],
 'constexprs': {'HEADS': 12, 'SEQ': 128, 'DIM': 64, 'BLOCK': 64},
 'grid': [128, 12, 1],
 'roles': ['model'],
 'metadata': {'model': 'google/electra-base-discriminator'}}
