# MobileNetV3-Small on FPGA

This suite contains 88 Triton operator specializations and a complete pretrained
`mobilenetv3_small_100.lamb_in1k` runner. The full model calls these operators 121
times. See [model/README.md](model/README.md) for weights, input preprocessing,
memory layout and model validation.

Source the repository's build environment first. `BUDDY_ROOT` (or `BUDDY_DIR`)
must point to the pinned Buddy source checkout; `LLVM_BIN` (or
`LLVM_BINARY_DIR`) and `BUDDY_MLIR_BINARY_DIR` select its built tools. Run these
commands from this directory:

```sh
make -j16 asm PYTHON="$(command -v python)"
make -j16 bin PYTHON="$(command -v python)"
```

The default images run on NH alone and emit the scheduler UART v1 start/result
records. They do not release E6 or use DMA/AME. The float32 kernels are lowered
without RVV, including their generated ABI wrappers. A change to AME execution
requires a scheduler/hardware profile that supports it; the existing scheduler
only supports NH-only jobs. The old two-core build remains available through
`BOARD_PROFILE=nh-e6` for separately managed hardware.

The output for each case is `build/kernels/<case>/<case>.bin`, alongside the
ELF, assembly, frontend contracts, audit and build logs. Each image executes
its independent C numerical oracle and returns failure if any comparison fails.

Before board execution, the generated code can also be validated locally:

```sh
python tools/check_host.py --clang "$LLVM_BIN/clang"
python tools/check_riscv.py \
  --cc /path/to/riscv64-unknown-linux-gnu-gcc \
  --qemu /path/to/qemu-riscv64
```

The host check compiles emitted LLVM IR for the host. The QEMU check links the
actual RISC-V `kernel.o` without recompiling it. Both use the original adapters
and independent operator oracles. Their reports explicitly identify the
execution environment. Neither checks physical FPGA runtime behavior or timing.

Build and execute the full model using the commands in
[model/README.md](model/README.md). Model images contain pretrained weights,
one input and all 1,000 reference logits, so board results can be checked without
DDR downloads. The runner prints per-operator cycles and top-five indices.

Keep authentication tokens outside tracked files. Scheduler submissions must
keep the client alive for its lease heartbeat, and must verify terminal state,
execution status and both exit codes. A printed `PASS` alone does not establish
a completed scheduler job. The execution timeout includes image loading and
DDR readback as well as inference.

From this directory, submit the operator images followed by the model image:

```sh
python ../../../scripts/run-fpga-scheduler.py \
  --sdk-root /path/to/fpga-scheduler --url http://board-host:18765 \
  --token-file /path/to/.token --no-proxy --timeout 600 \
  build/kernels build/model/mobilenet.bin
```

Use `--dry-run` to record image hashes without connecting, or `--health` with
no image arguments to check the service. The script retains the SDK heartbeat,
runs jobs sequentially and saves each complete scheduler result. It independently
checks the UART protocol against the service's exit status and stops on failure
by default. Pass an explicit model image file; `model-data.bin` is a constant
blob for linking, not a bootable program.
