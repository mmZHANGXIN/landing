#!/usr/bin/env python3
"""Lightweight tests for CUDA depth projection log acceptance."""

import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PYTHON = sys.executable


GOOD_LOG = """\
  OK center/body-axis/z-buffer/invalid: valid=3 max_diff=0.00e+00
  OK translated yawed pose: valid=3 max_diff=0.00e+00
  OK single-point frame: valid=1 max_diff=0.00e+00
=== CUDA depth projection parity PASSED ===
"""


def _run(path):
    return subprocess.run(
        [PYTHON, str(ROOT / "accept_depth_projection_cuda_light.py"), str(path)],
        cwd=str(ROOT),
        text=True,
        capture_output=True,
    )


def test_good_log_passes():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "depth_projection_cuda.log"
        path.write_text(GOOD_LOG, encoding="utf-8")
        result = _run(path)
        text = result.stdout + result.stderr
        assert result.returncode == 0, text
        assert "[OK] CUDA depth projection parity evidence accepted" in text


def test_missing_case_fails():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "depth_projection_cuda.log"
        path.write_text(
            GOOD_LOG.replace("  OK single-point frame: valid=1 max_diff=0.00e+00\n", ""),
            encoding="utf-8",
        )
        result = _run(path)
        text = result.stdout + result.stderr
        assert result.returncode != 0, text
        assert "missing required CUDA parity output: OK single-point frame" in text


def test_cuda_unavailable_fails():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "depth_projection_cuda.log"
        path.write_text("CUDA is not available; torch_cuda depth projection cannot be accepted.\n", encoding="utf-8")
        result = _run(path)
        text = result.stdout + result.stderr
        assert result.returncode != 0, text
        assert "forbidden failure output present: CUDA is not available" in text


def main():
    test_good_log_passes()
    test_missing_case_fails()
    test_cuda_unavailable_fails()
    print("=== Lightweight CUDA depth projection acceptance ===")
    print("  OK good CUDA parity log passes")
    print("  OK missing parity case fails")
    print("  OK CUDA-unavailable log fails")
    print("=== ALL PASSED ===")


if __name__ == "__main__":
    main()
