#!/usr/bin/env python3
"""Execute the complete model using native code from actual Triton LLVM IR.

This validates the compiler output, ABI adapters, memory reuse and pretrained
model logits. It is host numerical evidence, not an FPGA execution claim.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import struct
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=ROOT / "build/model")
    parser.add_argument("--kernels", type=Path, default=ROOT / "build/kernels")
    parser.add_argument("--clang", default="clang")
    parser.add_argument("--jobs", type=int, default=8)
    args = parser.parse_args()
    model = args.model.resolve()
    manifest = json.loads((model / "model-manifest.json").read_text())
    host = model / "host"
    host.mkdir(exist_ok=True)
    report_file = model / "host-model-report.json"
    report_file.unlink(missing_ok=True)
    common = [args.clang, "-O2", "-ffp-contract=off", "-DHOST_TEST",
              "-I" + str(ROOT), "-I" + str(ROOT / "model")]
    commands, objects, source_hashes = [], [], {}
    for case in manifest["used_cases"]:
        for suffix, source in [("kernel", args.kernels / case / "kernel.ll"),
                               ("adapter", args.kernels / case / "adapter.c")]:
            target = host / f"{case}.{suffix}.o"
            commands.append(common + ["-c", str(source), "-o", str(target)])
            objects.append(target)
            source_hashes[f"{case}/{source.name}"] = digest(source)
    for source in [ROOT / "support.c", ROOT / "model/model_support.c",
                   model / "runner.c", model / "model_blob.S"]:
        target = host / (source.stem + ".o")
        commands.append(common + ["-c", str(source), "-o", str(target)])
        objects.append(target)
        source_hashes[str(source.relative_to(ROOT))] = digest(source)

    def compile(command):
        result = subprocess.run(command, text=True, capture_output=True)
        if result.returncode:
            raise RuntimeError(" ".join(command) + "\n" + result.stdout + result.stderr)

    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        list(pool.map(compile, commands))
    executable = host / "mobilenet-host"
    compile(common + list(map(str, objects)) + ["-lm", "-o", str(executable)])
    actual_path = host / "actual-logits.f32"
    actual_path.unlink(missing_ok=True)
    environment = {**os.environ, "MODEL_LOGITS_FILE": str(actual_path)}
    start = time.monotonic()
    result = subprocess.run([str(executable)], env=environment, text=True, capture_output=True)
    seconds = time.monotonic() - start
    (host / "run.log").write_text(result.stdout + result.stderr)
    report = {"model": manifest["model"], "scope": "host native execution of compiled Triton LLVM IR and original adapters",
              "board_execution": False, "returncode": result.returncode, "wall_seconds": seconds,
              "model_manifest_sha256": digest(model / "model-manifest.json"),
              "model_blob_sha256": digest(model / "model-data.bin"),
              "golden_logits_sha256": digest(model / "golden-logits.f32"),
              "compiler": subprocess.check_output([args.clang, "--version"], text=True).splitlines()[0],
              "executable_sha256": digest(executable), "sources_sha256": source_hashes}
    if result.returncode == 0 and actual_path.exists():
        actual = struct.unpack("<1000f", actual_path.read_bytes())
        golden = struct.unpack("<1000f", (model / "golden-logits.f32").read_bytes())
        errors = [i for i, (a, g) in enumerate(zip(actual, golden))
                  if not abs(a-g) <= manifest["logits_atol"] + manifest["logits_rtol"] * abs(g)]
        top5 = sorted(range(1000), key=lambda i: actual[i], reverse=True)[:5]
        report.update(max_abs_error=max(abs(a-g) for a, g in zip(actual, golden)),
                      mismatch_count=len(errors), top5=top5, expected_top5=manifest["expected_top5"],
                      passed=not errors and top5 == manifest["expected_top5"])
    else:
        report.update(passed=False, output=result.stdout + result.stderr)
    report_file.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "sources_sha256"}, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
