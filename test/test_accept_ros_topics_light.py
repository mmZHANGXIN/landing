#!/usr/bin/env python3
"""Lightweight tests for ROS topic evidence acceptance."""

import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PYTHON = sys.executable


def _hz_log(rate, window, min_s=0.09, max_s=0.11, std=0.005):
    return (
        f"average rate: {rate:.3f}\n"
        f"\tmin: {min_s:.3f}s max: {max_s:.3f}s std dev: {std:.5f}s window: {window}\n"
    )


def _run(*args):
    return subprocess.run(
        [PYTHON, str(ROOT / "accept_ros_topics_light.py"), *args],
        cwd=str(ROOT),
        text=True,
        capture_output=True,
    )


def test_good_topic_logs_pass():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        files = {
            "hz_livox_lidar.log": _hz_log(10.0, 15),
            "hz_livox_imu.log": _hz_log(200.0, 40, min_s=0.004, max_s=0.006, std=0.0004),
            "hz_cloud_registered.log": _hz_log(10.0, 15),
            "hz_odometry.log": _hz_log(20.0, 20, min_s=0.04, max_s=0.06),
        }
        for name, text in files.items():
            (root / name).write_text(text, encoding="utf-8")
        result = _run(
            str(root / "hz_livox_lidar.log"),
            str(root / "hz_livox_imu.log"),
            str(root / "hz_cloud_registered.log"),
            str(root / "hz_odometry.log"),
            "--require-all",
        )
        text = result.stdout + result.stderr
        assert result.returncode == 0, text
        assert "[OK] ROS topic evidence accepted" in text


def test_low_rate_and_missing_topic_fail():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "hz_cloud_registered.log").write_text(_hz_log(1.5, 3), encoding="utf-8")
        result = _run(str(root / "hz_cloud_registered.log"), "--require-all")
        text = result.stdout + result.stderr
        assert result.returncode != 0, text
        assert "average rate 1.50Hz" in text
        assert "window 3" in text
        assert "missing required topic logs" in text


def main():
    test_good_topic_logs_pass()
    test_low_rate_and_missing_topic_fail()
    print("=== Lightweight ROS topic acceptance ===")
    print("  OK good topic logs pass")
    print("  OK low-rate/missing-topic logs fail")
    print("=== ALL PASSED ===")


if __name__ == "__main__":
    main()
