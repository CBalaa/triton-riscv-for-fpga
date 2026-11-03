# ELECTRA typed operator scheduling

`electra_infer()` in `model.c` calls the real `_mlir_ciface_kernel_*` entries
through embedding, 12 encoder layers and the discriminator head. No scalar
model fallback is substituted. `kernels.h` lists the fixed typed ABI.

All weight tensors are FP32. `tools/prepare_weights.py` maps each checkpoint
name to `ElectraWeights`, preserves `[output,input]` matrix storage, and writes
an aligned `weights.bin` plus `weights.h`. Load the blob at a 64-byte-aligned
address and call `electra_bind_weights(&weights, blob, blob_bytes)`.

Inputs are distinct aligned INT32 arrays of length 128. Masks are binary and
nonempty; IDs are in `[0,30522)` and token types in `[0,2)`. Output contains
128 raw logits. Supply a separate 16 MiB aligned scratch buffer. The buffer
must not overlap inputs, weights or output; it may be reused after return.
All dense operations dynamically quantize both operands. QK scale 0.125 is
applied once by dequantization. PV performs explicit right-transpose
quantization. The last N=1 dot uses exact scalar INT32 accumulation.

`tools/check_model_abi.py` executes the actual C scheduling against typed
mocks. It checks every argument's shape, stride, alignment and nonaliasing,
and compares all 1,645 calls with the independent Python schedule. It also
rejects insufficient scratch and invalid token IDs. This establishes ABI and
call order; mocked kernels do not establish complete-model numerical results.

`make model-runtime-object` cross-compiles this scheduling to an RV64 object.
`tools/build_model_bin.py` now links a complete NH/E6 image with actual audited
objects/adapters. Defining `ELECTRA_PREQUANTIZED_WEIGHTS` adds cached signed-i8
matrix/row-scale bindings, prepared by `prepare_model_board.py`, and skips the
74 constant-weight quantizations. The ordinary FP32-weight ABI stays available.
Defining `ELECTRA_MODEL_BOARD` enables immutable layer snapshots for scheduler
download. There is no numerical oracle on the FPGA.

The default FPGA build enables C++ scalar FP32 max reduction lowering and
audits the final ELF for residual `vfredmax.vs`. The default runner downloads
all 14 tensors and accepts only byte/FP32-bit equality against local execution
of the actual graph/emitted scalar LLVM plus exact C integer dots, after
checking scheduler and build provenance. See the [main README](../README.md)
for build, execution, download and offline verification commands.
