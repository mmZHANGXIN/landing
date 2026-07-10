#!/usr/bin/env python3
"""Accept Orin CUDA depth projection parity log evidence."""

from __future__ import annotations

import argparse
from pathlib import Path


REQUIRED_SNIPPETS = (
    "OK center/body-axis/z-buffer/invalid",
    "OK translated yawed pose",
    "OK single-point frame",
    "OK HALSS-aligned body ROI",
    "CUDA depth projection parity PASSED",
)

FORBIDDEN_SNIPPETS = (
    "CUDA is not available",
    "Traceback",
    "[FAIL]",
    "AssertionError",
    "RuntimeError",
)


def validate(text: str):
    failures = []
    for snippet in REQUIRED_SNIPPETS:
        if snippet not in text:
            failures.append(f"missing required CUDA parity output: {snippet}")
    for snippet in FORBIDDEN_SNIPPETS:
        if snippet in text:
            failures.append(f"forbidden failure output present: {snippet}")
    return failures


def main():
    parser = argparse.ArgumentParser(description="Accept test_depth_projection_cuda.py tee log")
    parser.add_argument("log", help="experiments/logs/depth_projection_cuda.log")
    args = parser.parse_args()

    path = Path(args.log)
    if not path.is_file():
        print(f"[FAIL] missing CUDA projection log: {path}")
        return 1
    text = path.read_text(encoding="utf-8", errors="replace")
    failures = validate(text)
    print("CUDA depth projection acceptance summary")
    print(f"  log: {path}")
    if failures:
        for failure in failures:
            print(f"[FAIL] {failure}")
        return 1
    print("[OK] CUDA depth projection parity evidence accepted")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
