# INT8 accumulator validation

`accumulator_validation.py` drives the existing Makefiles and strict simulator.
Default selection is the six plan06 failures plus four overwrite regressions.
`--all` selects the complete 40-case inventory. Build destinations must be new;
an old object or a historical PASS cannot satisfy a new build.

```bash
VIT=examples/BOSCAMEFPGA/vit-base-patch16-224
python3 "$VIT/tools/accumulator_validation.py" build \
  --freeze-tools --output "$PWD/.tmp/accumulator-build"
python3 "$VIT/tools/accumulator_validation.py" simulate \
  --kernels "$PWD/.tmp/accumulator-build/kernels" \
  --output "$PWD/.tmp/accumulator-build/simulation"
python3 "$VIT/tools/accumulator_validation.py" images \
  --kernels "$PWD/.tmp/accumulator-build/kernels" \
  --output "$PWD/.tmp/accumulator-production"
python3 "$VIT/tools/accumulator_validation.py" images \
  --kernels "$PWD/.tmp/accumulator-build/kernels" --accumulator-board \
  --output "$PWD/.tmp/accumulator-diagnostic"
```

`--freeze-tools` copies the compiler executables and Clang resource headers,
records SHA256, and passes those exact executables to Make. Source identities
include compiler headers/sources, runtime and linker inputs, board support,
assembly helpers, the frontend identity and tracked changes in all three Git
repositories. `source-snapshot/sources.tar.gz` preserves the selected source
bytes, including untracked implementation files; HEADs and binary patches
preserve the tracked worktree state. Every recorded source/patch/frontend hash
must remain unchanged through a build or image-flavor run. Frozen Clang
resource headers are rechecked as well as compiler executables.
`images` reuses checked kernel artifacts and rejects any changed kernel object;
production and diagnostic images therefore contain the same generated kernel.
Production images use the original launch scalar reference through `GOLDEN=1`.
An interrupted or source-changing build remains incomplete in its JSON report.

The diagnostic is opt-in (`ACCUMULATOR_VALIDATION=1`). Its separate linker
keeps two independent reference grids, a guarded operand arena and a 256 KiB
managed private frame below the stable E6 stack. It performs three full typed
adapter calls: normal forward, signed extremes with i32 wrap and tail-first,
then the same stress inputs in reverse. Overwrite's third call changes old C.
Each call compares all output bits, all A/B bytes, and 64-byte guards before
and after every argument. The independent host oracle uses signed i64 products
and a final modulo-i32 cast; it never executes a candidate kernel.

The full-grid strict simulator also retains one Machine across the three
matmul rounds. Its report includes external argument read/write bytes, private
frame allocation/release and CPU/AME byte counts. Private frames are separately
audited; they cannot weaken logical argument bounds or input immutability.

Use `check_accumulator_harness.py --output <report.json>` to test the harness
on the host. This checks all ten real grids and independent scalar output
corners, plus deliberate output, guard and input corruption. Its entry is an
explicit stub, so its PASS proves harness behavior only.

Board execution continues through the existing `run_bin.py` queue:

```bash
python3 "$VIT/tools/run_board_suite.py" \
  --kernels "$PWD/.tmp/accumulator-diagnostic/kernels" \
  --case gemm_i8_197x768x768 --report-dir "$PWD/.tmp/accumulator-board-A1"
```

Each run preserves the submitted image/hash, task details, UART, cycles,
embedded oracle manifest, generated IR/assembly, object, ELF and link map.
Diagnostic PASS requires all three complete round records and matching output
hashes as well as zero output/guard/input errors. A timeout with partial cycles
does not establish correctness. `--collect` summarizes existing results without
submitting another task. Run each comparison A/B/A/B with the same input, grid,
measurement window, cache policy and hardware; queue wait is outside cycles.

If three full diagnostic rounds exceed the board's 120-second task budget,
build three separate images with `images --accumulator-board
--accumulator-round 0`, then rounds `1` and `2`, each into a fresh output root.
Every image still runs the full program grid and all output/input/guard checks.
Round IDs remain 0/1/2 in UART and manifests. Each split image must retain the
same kernel object; partial results from a timed-out image stay excluded.
Aggregate existing results without submitting jobs:

```bash
python3 "$VIT/tools/collect_accumulator_evidence.py" diagnostics \
  --result board-round0/operator/result.json \
  --result board-round1/operator/result.json \
  --result board-round2/operator/result.json --output diagnostic-report.json
```

The collector requires distinct complete round IDs, unchanged kernel object
and source identities, matching stress-round output hashes, embedded oracle
bytes and successful scheduler image/exit/UART evidence. A missing round leaves
the aggregate INCOMPLETE. Original TIMEOUT records can be supplied and remain
listed as excluded failures.

The performance collector consumes a JSON manifest of completed runs:

```json
{
  "runs": [
    {
      "result": "board-A1/gemm_i8_197x768x768/result.json",
      "route": "ddr",
      "comparison": "ddr-spm",
      "cache_policy": "compiled wrapper: writes before adapter, uncached reads after",
      "route_evidence": "ddr/kernel.planned.mlir",
      "route_evidence_sha256": "<actual SHA256>",
      "required_route_fragments": ["<actual private accumulator placement marker>"],
      "forbidden_route_fragments": ["<marker for an unintended accumulator path>"]
    }
  ]
}
```

Paths are relative to the manifest. Include at least four entries for each
`ddr-spm` or `ddr-acc` comparison in actual A/B/A/B order. Use precise generated
accumulator markers reviewed against the fixture; an unrelated SPM A/B staging
buffer does not prove P resides in SPM. The collector checks image/task hashes,
route evidence, distinct task IDs, fixed image per route, independent input and
reference identities, timestamps, hardware, cycle scope and alternating order.

```bash
python3 "$VIT/tools/accumulator_validation.py" performance \
  --manifest performance-runs.json --output performance-report.json
```

The report includes every sample, median, spread and cycle delta. A median
improvement within observed spread is explicitly marked insufficient evidence
of speedup. Route-specific compiler proofs, simulator execution and actual
private DDR/SPM/ACC board runs remain required independently.

`collect_accumulator_evidence.py calibration --manifest <build/manifest.json>
--task <collected/task.json> --output <report.json>` handles the within-image
calibration UART schema. It recognizes both the historical eight-call V2-DDR
versus V1-SPM experiment and the sixteen-call V2 DDR/SPM/ACC experiment. Normal
and signed-extreme inputs each use DDR/SPM/DDR/SPM, then DDR/ACC/DDR/ACC. The
collector verifies generated route hashes, independent output hashes, input
and guard checks, frame acquire/release counts, actual task/image identity and
complete invocation ordering before computing medians and spread.

Calibration manifests must explicitly pair `mode: "overwrite"` with
`post_add: false`, or `mode: "normal-post-add"` with `post_add: true`. The
three-route input/reference manifest must declare the same pair. The collector
rejects missing, malformed or inconsistent tags and preserves them in both
PASS and numerical FAIL reports, so performance tables cannot silently combine
the two modes. Historical records without these tags are not inferred to be
overwrite evidence.

For a long K24 calibration, the builder's `--stress 0` or `--stress 1` creates
an eight-call image containing one complete DDR/SPM/DDR/SPM and
DDR/ACC/DDR/ACC schedule. The default image retains all sixteen calls. New
manifests and input/reference manifests declare identical `calls`, `stress`
(`null`, `0` or `1`) and exact `stress_schedule` fields. Split images require
the full `[8, 4096]` little-endian i32 DDR ledger: all 32,768 C/snapshot words
are checked against the independent normal/stress goldens. Collection reports
the selected stress explicitly and computes only its two route comparisons.
An incomplete sixteen-call image cannot be collected as a split image; old
full manifests without stress-selection fields keep the complete sixteen-call
contract. Both split stresses remain necessary for a two-stress calibration.

The report preserves plan versions and physical SPM constraints. A calibration
using explicit fixed SPM is not proof of automatic profitability selection.
The three-route fixture has a common two-output observable contract; its ACC
adapter performs the second output copy inside the measured window. Compiler
fusion of a two-output source graph is not claimed.

Readmem calibration images declare an immutable packed output ledger inside
the scheduler's existing DDR window. Download outputs from the completed task,
then validate all actual C and snapshot values before collecting performance:

```bash
python3 "$VIT/tools/collect_accumulator_evidence.py" artifacts \
  --task board-run/task.json --output board-run/ddr
python3 "$VIT/tools/collect_accumulator_evidence.py" calibration \
  --manifest calibration-build/manifest.json --task board-run/task.json \
  --artifacts board-run/ddr/artifacts.json --output board-run/calibration.json
```

The download action uses only the original task ID and saved HTTP artifacts;
it submits no job and reads no live hardware. Its download report establishes
transport length/SHA256 only. Calibration checks task metadata, UART output
declarations, dtype, shape, byte count and every little-endian i32 value against
the independently generated normal/stress grids. A wrong output suppresses
performance comparisons. Missing readmem evidence leaves the new calibration
unverified. Historical images without output declarations retain their original
UART validation scope. Existing production and three-round diagnostics keep
their full board-side comparisons and do not require readmem outputs.

Actual scheduler profile and available clock/runtime details are preserved
verbatim. Cycles are not converted to seconds using a historical frequency.

## 2026-10-28 acceptance evidence

The current repair uses a fresh default-V2 40-object build and ten affected
matmul cases for full-grid numerical and FPGA acceptance. The other thirty
cases have fresh build coverage; they are not implicitly reported as numerical
or board passes. Full-grid simulation covers forward, tail-first and reverse
orders: 3624 programs, 6355824 exact i32 outputs, zero difference. Production,
three-round diagnostic and large-case single-round images retain the same
frozen kernel object. Final FPGA acceptance passed all 28 unique tasks:
10 original production and 18 diagnostic tasks, 40 complete kernel calls,
and all ten cases with unique complete diagnostic rounds 0/1/2. Production
compared 2118608 i32 values and diagnostics compared 6355824, for 8474432
firmware comparisons with zero output/input/guard errors. No final acceptance
task failed, retried, timed out or trapped. The integration evidence is
`plan/validation-evidence/vit-accumulator-repair-final-20261028.json`; these
firmware comparisons are separate from host DDR readmem checks below.

The automatic hot-P microbenchmarks are complete: six unique tasks, 64 calls,
262144 downloaded i32 values, zero mismatches. The repository evidence is
`plan/validation-evidence/vit-hot-p-automatic-20261028.json`. Physical SPM/ACC
use the final compiler, while physical DDR controls retain the earlier frozen
compiler; a separate final-compiler offline 28-artifact identity bridge is
recorded explicitly. The measured profile is narrowly matched, and ACC
priority is not a claim that it always has the lowest cycle count. Diagnostic
round-zero versus stress rounds may change inputs/order, so their overall
cycle range is not same-input run-to-run variation. The alternating
microbenchmark pairs provide the separately reported timing spreads.
