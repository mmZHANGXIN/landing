#!/usr/bin/env python3
"""Lightweight tests for timing log acceptance."""

import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PYTHON = sys.executable


def _line(seq, total):
    return (
        f"2026-06-07 13:00:00,000 [INFO] [{seq:04d}] act=3(W) "
        f"H=20ms D=3ms C=18ms RL=8ms total={total}ms | yaw=10deg"
    )


def _run(log_text, *args):
    with tempfile.TemporaryDirectory() as tmp:
        log = Path(tmp) / "timing.log"
        log.write_text(log_text, encoding="utf-8")
        return subprocess.run(
            [PYTHON, str(ROOT / "analyze_timing_log.py"), str(log), *args],
            cwd=str(ROOT),
            text=True,
            capture_output=True,
        )


def test_good_timing_log_passes():
    lines = [_line(i + 1, 70 + (i % 5)) for i in range(60)]
    result = _run("\n".join(lines), "--budget-ms", "100", "--max-p95-ms", "90", "--min-samples", "30")
    text = result.stdout + result.stderr
    assert result.returncode == 0, text
    assert "[OK] timing gates passed" in text


def test_short_slow_timing_log_fails():
    lines = [_line(i + 1, 140) for i in range(5)]
    result = _run("\n".join(lines), "--budget-ms", "100", "--max-p95-ms", "100", "--min-samples", "30")
    text = result.stdout + result.stderr
    assert result.returncode != 0, text
    assert "only 5 samples parsed" in text
    assert "total P95" in text
    assert "over-budget ratio" in text


def main():
    test_good_timing_log_passes()
    test_short_slow_timing_log_fails()
    print("=== Lightweight timing log acceptance ===")
    print("  OK good timing log passes")
    print("  OK short/slow timing log fails")
    print("=== ALL PASSED ===")


if __name__ == "__main__":
    main()
