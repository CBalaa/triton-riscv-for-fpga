# add_1x768 board check

`make bin` links the existing Triton kernel and typed adapter with the
repository-local Buddy NH/E6 boot, PMP/cache setup, UART and DMA runtime.
`board.c` and `link.ld` follow the standalone
`../../matmul_i8_128x1024x128` example. The existing `launch.c` initializes
both inputs and compares all 768 output values with scalar addition.

Run from the repository root with the `boscame` conda environment activated:

```bash
source scripts/triton-riscv-env.sh
export BUDDY_ROOT="$PWD/buddy-mlir"
VIT=examples/BOSCAMEFPGA/vit-base-patch16-224
make -C "$VIT/add_1x768" bin
python examples/BOSCAMEFPGA/run_bin.py \
  "$VIT/build/kernels/add_1x768/board/add_1x768.bin"
```

The scheduler endpoint requires the user-managed SSH tunnel on port 18765.
The image emits UART v1 START/RESULT records so the scheduler can distinguish
normal application exit from a timeout.

## Board result: 2026-10-27

- FPGA: 4; task: `6df45aa8c69d4d81927a9f99139ab2cb`.
- Application and scheduler: PASS, `exited 0 True`.
- Checked: 768 outputs; errors: 0; maximum absolute error: 0.
- Kernel: **15,233 cycles** (`0x3b81`), one complete adapter call covering
  all six programs, including its DMA transfers and final completion fence.
  Input initialization, scalar validation and UART output are outside timing.
- Image SHA256: `0c7a6a0f8ba5964468e0a6f8b873d108fa52dcb70dd1b321f54aaa2b785258c0`.
- Full ELF instruction audit: PASS (1,325 instructions; 5 RVV; 0 AME).

Local evidence is under `../build/kernels/add_1x768/board/`:
`uart.log`, `task.json`, `board-result.json`, `image.sha256` and
`elf-audit.json`. Cycle count is a single board measurement.
