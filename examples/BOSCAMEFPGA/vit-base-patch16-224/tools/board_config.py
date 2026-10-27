#!/usr/bin/env python3
"""Generate the board ABI wrappers from the literal per-case configuration."""

import argparse
import json
import math
import re
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from cases import describe

PATCH_DEPENDENCIES = (
    "quantize_196x768", "quantize_768x768",
    "matmul_i8_overwrite_196x768x768", "dequantize_196x768",
)


def dependencies(case):
    return PATCH_DEPENDENCIES if case["family"] == "patch_embed" else ()


def configuration(case):
    family, shape = case["family"], case["shape"]
    trials = 2 if family == "per_token_quantization" else 1
    if family in ("linear", "matmul_i8", "matmul_i8_overwrite"):
        checked = shape[0] * shape[1]
    elif family.startswith("attention_"):
        checked = math.prod(shape[:3])
    elif family == "patch_embed":
        checked = 196 * 768
    elif family == "per_token_quantization":
        checked = trials * (math.prod(shape) + shape[0])
    else:
        checked = math.prod(shape)
    return {
        "case": case["name"], "expected_invocations": trials,
        "expected_checked": checked, "dependencies": list(dependencies(case)),
        "cycle_scope": "complete typed adapter including grid dispatch and DMA; final fence; excludes initialization, validation, UART",
    }


def wrapper(case, timed):
    name = case["name"]
    params = ", ".join(f"MemRef{a['rank']} *a{i}" for i, a in enumerate(case["arguments"]))
    args = ", ".join(f"a{i}" for i in range(len(case["arguments"])))
    lines = [f"extern void vit_board_impl_{name}({params});",
             f"void _mlir_ciface_kernel_{name}({params}) {{"]
    if timed:
        lines += ["  unsigned slot = vit_board_kernel_begin();"]
    lines += ["  nh_e6_cache_writes();"]
    if timed:
        lines += ["  uint64_t begin = nr_cycles();"]
    lines += [f"  vit_board_impl_{name}({args});",
              '  __asm__ volatile("fence iorw, iorw" ::: "memory");']
    if timed:
        lines += ["  uint64_t end = nr_cycles();",
                  "  vit_board_kernel_end(slot, end - begin);"]
    lines += ["  nh_e6_cache_uncached_loads();", "}"]
    return "\n".join(lines)


def write_if_changed(path, text):
    if not path.exists() or path.read_text() != text:
        path.write_text(text)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--dependencies", action="store_true")
    parser.add_argument("--adapter-source", type=Path)
    args = parser.parse_args()
    case = describe(args.case)
    if args.dependencies:
        print(" ".join(dependencies(case)))
        return
    if args.output is None:
        parser.error("--output is required")
    args.output.mkdir(parents=True, exist_ok=True)
    if args.adapter_source is not None:
        source = args.adapter_source.read_text()
        if case["family"] == "patch_embed":
            original = (
                "  for (int p = 0; p < 196; ++p)\n"
                "    for (int o = 0; o < 768; ++o)\n"
                "      ((float *)p3)[o * 196 + p] = patch_gemm[p * 768 + o] + ((const float *)p2)[o];"
            )
            replacement = (
                "  /* Stage each row on the stable stack before switching to DDR writes. */\n"
                "  float row[768];\n"
                "  for (int p = 0; p < 196; ++p) {\n"
                "    nh_e6_cache_uncached_loads();\n"
                "    for (int o = 0; o < 768; ++o)\n"
                "      row[o] = patch_gemm[p * 768 + o] + ((const float *)p2)[o];\n"
                "    nh_e6_cache_writes();\n"
                "    for (int o = 0; o < 768; ++o)\n"
                "      ((float *)p3)[o * 196 + p] = row[o];\n"
                "  }"
            )
            if source.count(original) != 1:
                raise ValueError("patch adapter projection changed; cache staging requires review")
            source = source.replace(original, replacement)
        write_if_changed(args.output / "adapter.c", source)
        return
    config = configuration(case)
    header = (
        "/* Generated from kernel.py by tools/board_config.py. */\n"
        f'#define VIT_BOARD_CASE "{case["name"]}"\n'
        f'#define VIT_BOARD_EXPECTED_INVOCATIONS {config["expected_invocations"]}u\n'
        f'#define VIT_BOARD_EXPECTED_CHECKED {config["expected_checked"]}u\n'
    )
    source = ["/* Wrappers preserve the typed adapter's numerical body. */",
              '#include "board_support.h"', wrapper(case, True)]
    source.extend(wrapper(describe(dep), False) for dep in dependencies(case))
    write_if_changed(args.output / "board_case.h", header)
    write_if_changed(args.output / "board_wrapper.c", "\n\n".join(source) + "\n")
    write_if_changed(args.output / "board_config.json", json.dumps(config, indent=2) + "\n")
    launch = Path(case["launch_source"]).read_text()
    if case["family"] == "per_token_quantization":
        launch, matches = re.subn(
            r'(for \(int trial = 0; trial < 2; trial\+\+\) \{)',
            r'\1\n    nh_e6_cache_writes();', launch, count=1,
        )
        if matches != 1:
            raise ValueError("quantization trial loop changed; cache staging requires review")
    write_if_changed(args.output / "board_launch.c", launch)


if __name__ == "__main__":
    main()
