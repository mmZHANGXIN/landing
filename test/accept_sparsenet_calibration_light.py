#!/usr/bin/env python3
"""Accept a JSON report written by calibrate_sparsenet_scale.py."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def _load(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _finite_positive(value):
    return isinstance(value, (int, float)) and math.isfinite(float(value)) and float(value) > 0.0


def _error(report, section, key):
    value = (report.get(section) or {}).get(key)
    return float(value) if isinstance(value, (int, float)) and math.isfinite(float(value)) else math.inf


def accept_report(report, args):
    failures = []
    scale = report.get("suggested_output_scale")
    if not _finite_positive(scale):
        failures.append(f"suggested_output_scale must be positive finite, got {scale!r}")
    elif float(scale) > args.max_scale:
        failures.append(f"suggested_output_scale {float(scale):.6g} > {args.max_scale:.6g}")

    pixels = report.get("pixels_used")
    if not isinstance(pixels, int) or pixels < args.min_pixels:
        failures.append(f"pixels_used={pixels!r}; need at least {args.min_pixels}")

    valid_ratio = report.get("valid_ratio")
    if not isinstance(valid_ratio, (int, float)) or float(valid_ratio) < args.min_valid_ratio:
        failures.append(f"valid_ratio={valid_ratio!r}; need at least {args.min_valid_ratio:.4f}")

    if report.get("encoding") != args.encoding:
        failures.append(f"encoding={report.get('encoding')!r} expected={args.encoding!r}")
    dmax = report.get("dmax")
    if not _finite_positive(dmax):
        failures.append(f"dmax must be positive finite, got {dmax!r}")

    for key in ("ratio_p10", "ratio_p50", "ratio_p90"):
        value = report.get(key)
        if not _finite_positive(value):
            failures.append(f"{key} must be positive finite, got {value!r}")

    raw_median = _error(report, "raw_error", "median_abs_m")
    scaled_median = _error(report, "scaled_error", "median_abs_m")
    scaled_p90 = _error(report, "scaled_error", "p90_abs_m")
    if scaled_median > args.max_scaled_median_abs_m:
        failures.append(
            f"scaled median error {scaled_median:.3f}m > {args.max_scaled_median_abs_m:.3f}m"
        )
    if scaled_p90 > args.max_scaled_p90_abs_m:
        failures.append(
            f"scaled p90 error {scaled_p90:.3f}m > {args.max_scaled_p90_abs_m:.3f}m"
        )
    if args.require_improvement and scaled_median > raw_median:
        failures.append(
            f"scaled median error {scaled_median:.3f}m is worse than raw {raw_median:.3f}m"
        )

    return failures


def main():
    parser = argparse.ArgumentParser(description="Accept SparseNet calibration JSON evidence")
    parser.add_argument("report", help="JSON written by calibrate_sparsenet_scale.py --out-json")
    parser.add_argument("--encoding", default="inverse_unit")
    parser.add_argument("--min-pixels", type=int, default=100)
    parser.add_argument("--min-valid-ratio", type=float, default=0.005)
    parser.add_argument("--max-scale", type=float, default=100.0)
    parser.add_argument("--max-scaled-median-abs-m", type=float, default=0.5)
    parser.add_argument("--max-scaled-p90-abs-m", type=float, default=1.5)
    parser.add_argument("--require-improvement", action="store_true")
    args = parser.parse_args()

    report = _load(Path(args.report))
    print("SparseNet calibration acceptance summary")
    print(f"  source: {report.get('source')}")
    print(f"  encoding/dmax: {report.get('encoding')}/{report.get('dmax')}")
    print(f"  pixels_used: {report.get('pixels_used')} valid_ratio={report.get('valid_ratio')}")
    print(f"  suggested_output_scale: {report.get('suggested_output_scale')}")
    print(f"  raw_median_abs_m: {_error(report, 'raw_error', 'median_abs_m'):.3f}")
    print(f"  scaled_median_abs_m: {_error(report, 'scaled_error', 'median_abs_m'):.3f}")

    failures = accept_report(report, args)
    if failures:
        for failure in failures:
            print(f"[FAIL] {failure}")
        return 1
    print("[OK] SparseNet calibration evidence accepted")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
