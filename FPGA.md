# Buddy NR FPGA toolchain

This fork preserves the triton-riscv frontend and carries the four-patch Triton
integration series used by `buddy-mlir-for-fpga`:

1. CPU-only build without NVIDIA/AMD backends.
2. The triton-shared-opt frontend entry point.
3. Triton frontend API compatibility.
4. MLIR symbol traits and the LLVM InlineAsmOp convergent argument.

The upstream Triton base is recorded in `triton-hash.txt`. Patched Triton sources
are reconstructed from that base and these patches; a separate patched Triton
fork is not required. The fourth patch targets the parent project's LLVM
`6ceae32b77a2c0cc3c929ae8924254525155a26c`, rather than the release toolchain
downloaded by the upstream `scripts/build.sh`.

From a checkout of `buddy-mlir-for-fpga`, build its LLVM/Buddy first, then run:

```bash
git submodule update --init thirdparty/triton-riscv
conda activate boscame
examples/FPGA-BOSCAME/common/triton/setup-triton.sh --jobs=24
examples/FPGA-BOSCAME/common/triton/setup-triton.sh --check
```

The parent `common/triton/toolchain-lock.json` records exact revisions and patch
hashes. Its patch verifier replays the overlapping series against the pristine
base, verifies full patched file contents, and preserves unrelated local edits.
Use that setup script for this target; the upstream build script selects a
different LLVM/Buddy release.
