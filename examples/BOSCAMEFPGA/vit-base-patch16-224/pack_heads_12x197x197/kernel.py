import triton
import triton.language as tl
@triton.jit
def head_layout(X, Out, PACK:tl.constexpr, BLOCK:tl.constexpr):
    head=tl.program_id(0)
    i=tl.program_id(1)*BLOCK+tl.arange(0,BLOCK)
    valid=i<38809
    src=head*(38809 if PACK else 38816)+i
    dst=head*(38816 if PACK else 38809)+i
    tl.store(Out+dst,tl.load(X+src,valid,other=0.0),valid)


KERNEL_CONFIG = {'family': 'head_layout',
 'kernel': 'head_layout',
 'arguments': [{'name': 'X', 'rank': 2, 'dtype': 'f32'},
               {'name': 'Out', 'rank': 2, 'dtype': 'f32'}],
 'shape': [12, 38809],
 'argument_shapes': [[12, 38809], [12, 38816]],
 'constexprs': {'PACK': True, 'BLOCK': 128},
 'grid': [12, 304, 1],
 'roles': ['attention_head_alignment']}
