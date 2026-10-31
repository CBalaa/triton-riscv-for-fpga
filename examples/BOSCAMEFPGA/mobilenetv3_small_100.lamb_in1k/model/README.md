# Complete MobileNetV3-Small inference

This runner executes the pretrained `timm` model
`mobilenetv3_small_100.lamb_in1k`, with batch size 1 and a 224×224 RGB input.
The graph has 121 calls to all 88 existing Triton specializations. Inference
batch normalization is folded into convolution weights and biases; squeeze and
excitation, residual connections, global pooling and the classifier remain in
the graph. Computation and model parameters are float32.

`tools/export_model.py` generates the graph, fused weights, input, original timm
golden logits and C runner. It first checks the fused graph against the original
model. Every activation has a 64-byte-aligned allocation, and the liveness
planner prohibits overlapping input/output storage, as required by the existing
adapters. The current graph needs 2,007,040 bytes of activation storage and a
10,753,952-byte constant blob, including its input and golden logits.

Install `torch`, `torchvision`, `timm`, `numpy`, `Pillow` and the timm download
dependencies in the Python environment. Export the pretrained model with:

```sh
python tools/export_model.py --output build/model
```

The default input is a deterministic tensor of exact binary fractions, described
in the generated manifest. Supply `--image /path/to/image.jpg` to use the model's
official timm validation preprocessing (bicubic resize, center crop, ImageNet
normalization). `--checkpoint /path/to/state_dict.pth` loads an already downloaded
pretrained state dictionary without network access. The manifest records the
state dictionary hash, input identity, preprocessing, package versions, complete
graph and memory allocation.

After building the operator LLVM IR, validate the **actual compiled Triton code**
and original ABI adapters on the host:

```sh
python tools/check_model_host.py --clang /path/to/matching/llvm/bin/clang
```

The checker recompiles each `build/kernels/*/kernel.ll` for the host and links it
with `adapter.c` and the generated model runner. It executes the complete model,
checks all 1,000 logits with `atol=0.002, rtol=0.001`, requires the expected top
five classes, and writes `build/model/host-model-report.json`. This is a numerical
and ABI check, not FPGA execution evidence. It does not substitute C reference
operators for Triton kernels. For a different export directory, pass
`--model build/model-image`.

The board runner checks all logits and top one against the same original timm
golden, prints top five class indices, and records each operator's elapsed cycle
count. Indices use the checkpoint's 1,000-class ImageNet order. A model pass
establishes correctness for the supplied input; it does not measure ImageNet
dataset accuracy. Host timing is not an estimate of FPGA inference latency.

Model source is kept separate from generated weight blobs and binaries under
`build/`. Weights are downloaded from the upstream timm pretrained distribution,
whose model configuration identifies the checkpoint license as Apache-2.0.

Build an NH image after the operator `asm` goal has completed:

```sh
make -f model/model.mk bin PYTHON=python3 \
  BUDDY_ROOT=/path/to/buddy-mlir LLVM_BIN=/path/to/matching/llvm/bin
```

This produces `build/model/mobilenet.bin` with the shared `runtime/nh` scheduler
UART protocol. The entire model runs on NH using scalar float32 code; its ELF
audits reject vector/AME instructions and verify the entry point and protocol.
The binary contains pretrained weights, so it is approximately 14.3 MiB. Its
layout reserves 2 MiB for activations and 1 MiB for the NH stack. The output
includes the model verdict and `@@FPGA_RESULT` with the application exit code.

Use `OUT=/absolute/path/to/build/model-image` for another exported input. The
makefile preserves that export's existing image and checkpoint by default.
Changing `IMAGE=/path/to/new.jpg` or `CHECKPOINT=/path/to/new.pth` invalidates the
export; explicit `IMAGE=` switches back to the deterministic tensor. Compiler
flags and export choices are recorded to prevent stale artifact reuse.

`tools/check_riscv.py --model build/model --model build/model-image` also runs
the actual RISC-V `kernel.o` objects under QEMU, using a compatible RISC-V Linux
compiler and emulator selected by `--cc` and `--qemu`. Its model reports include
all-logit differences, top-five agreement and executable/input provenance.
QEMU is a separate check from uploading the NH binary to physical hardware.
