#!/usr/bin/env python3
"""Lightweight tests for SparseNet calibration JSON acceptance."""

import json
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PYTHON = sys.executable


def _report(**overrides):
    report = {
        "source": "target_depth_m=3.000",
        "encoding": "inverse_unit",
        "dmax": 30.0,
        "pixels_used": 240,
        "valid_ratio": 0.025,
        "suggested_output_scale": 12.0,
        "ratio_p10": 10.0,
        "ratio_p50": 12.0,
        "ratio_p90": 14.0,
        "raw_error": {"median_abs_m": 2.0, "mean_abs_m": 2.2, "p90_abs_m": 3.0},
        "scaled_error": {"median_abs_m": 0.25, "mean_abs_m": 0.3, "p90_abs_m": 0.8},
    }
    for key, value in overrides.items():
        if "." in key:
            section, subkey = key.split(".", 1)
            report.setdefault(section, {})[subkey] = value
        else:
            report[key] = value
    return report


def _run(report, *args):
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "scale.json"
        path.write_text(json.dumps(report), encoding="utf-8")
        return subprocess.run(
            [PYTHON, str(ROOT / "accept_sparsenet_calibration_light.py"), str(path), *args],
            cwd=str(ROOT),
            text=True,
            capture_output=True,
        )


def test_good_report_passes():
    result = _run(_report(), "--require-improvement")
    text = result.stdout + result.stderr
    assert result.returncode == 0, text
    assert "[OK] SparseNet calibration evidence accepted" in text


def test_large_scale_fails():
    result = _run(_report(suggested_output_scale=150.0))
    text = result.stdout + result.stderr
    assert result.returncode != 0, text
    assert "suggested_output_scale" in text


def test_too_few_pixels_fails():
    result = _run(_report(pixels_used=20))
    text = result.stdout + result.stderr
    assert result.returncode != 0, text
    assert "pixels_used=20" in text


def test_scaled_error_fails():
    result = _run(_report(**{"scaled_error.median_abs_m": 0.9}))
    text = result.stdout + result.stderr
    assert result.returncode != 0, text
    assert "scaled median error" in text


def main():
    test_good_report_passes()
    test_large_scale_fails()
    test_too_few_pixels_fails()
    test_scaled_error_fails()
    print("=== Lightweight SparseNet calibration JSON acceptance ===")
    print("  OK good report passes")
    print("  OK too-large scale fails")
    print("  OK too-few pixels fail")
    print("  OK excessive scaled error fails")
    print("=== ALL PASSED ===")


if __name__ == "__main__":
    main()
