#!/usr/bin/env python3
"""Lightweight tests for Orin environment report acceptance."""

import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PYTHON = sys.executable


def _valid_report():
    return "\n".join([
        "# Orin Environment Check",
        "",
        "- required_failures: `0`",
        "",
        "| Key | Required | Status | Detail | Fix |",
        "| --- | --- | --- | --- | --- |",
        "| jetpack | True | PASS | R35 |  |",
        "| nvcc | True | PASS | CUDA 11.4 |  |",
        "| ros2_cli | True | PASS | ros2 0.9 |  |",
        "| ros_setup | True | PASS | /opt/ros/galactic/setup.bash |  |",
        "| torch_cuda | True | PASS | cuda=True |  |",
        "| opencv | True | PASS | cv2=4.5 DISPLAY set |  |",
        "| numpy | True | PASS | 1.22.4 |  |",
        "| yaml | True | PASS | 6.0 |  |",
        "| stable_baselines3 | True | PASS | 1.7 |  |",
        "| mavsdk | True | PASS | import ok |  |",
        "| rclpy | True | PASS | import ok |  |",
    ])


def _run(text):
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "orin_env.md"
        path.write_text(text, encoding="utf-8")
        return subprocess.run(
            [PYTHON, str(ROOT / "accept_orin_env_light.py"), str(path)],
            cwd=str(ROOT),
            text=True,
            capture_output=True,
        )


def test_valid_report_passes():
    result = _run(_valid_report())
    text = result.stdout + result.stderr
    assert result.returncode == 0, text
    assert "[OK] strict Orin environment evidence accepted" in text


def test_non_strict_report_fails():
    result = _run(_valid_report().replace("| jetpack | True | PASS |", "| jetpack | False | WARN |"))
    text = result.stdout + result.stderr
    assert result.returncode != 0, text
    assert "jetpack was not required" in text
    assert "jetpack status=WARN expected=PASS" in text


def test_cuda_failure_fails():
    result = _run(_valid_report().replace("| torch_cuda | True | PASS |", "| torch_cuda | True | FAIL |"))
    text = result.stdout + result.stderr
    assert result.returncode != 0, text
    assert "torch_cuda status=FAIL expected=PASS" in text


def test_missing_mavsdk_fails():
    result = _run(_valid_report().replace("| mavsdk | True | PASS | import ok |  |\n", ""))
    text = result.stdout + result.stderr
    assert result.returncode != 0, text
    assert "missing row: mavsdk" in text


def main():
    test_valid_report_passes()
    test_non_strict_report_fails()
    test_cuda_failure_fails()
    test_missing_mavsdk_fails()
    print("=== Lightweight Orin environment acceptance ===")
    print("  OK strict report passes")
    print("  OK non-strict report fails")
    print("  OK CUDA failure report fails")
    print("  OK missing MAVSDK report row fails")
    print("=== ALL PASSED ===")


if __name__ == "__main__":
    main()
