#!/usr/bin/env python3
"""Lightweight tests for the full field acceptance wrapper."""

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PYTHON = sys.executable


def test_dry_run_prints_full_field_acceptance_commands():
    result = subprocess.run(
        [
            PYTHON,
            str(ROOT / "run_field_acceptance.py"),
            "--expected-yaw-rate", "0.35",
            "--gis-bounds", "120.0,30.0,121.0,31.0",
            "--strict-flight-ready",
            "--require-landing",
            "--dry-run",
        ],
        cwd=str(ROOT),
        text=True,
        capture_output=True,
    )
    text = result.stdout + result.stderr
    assert result.returncode == 0, text
    for needle in (
        "check_orin_env.py --strict --require-jetson",
        "--out-md experiments/logs/orin_env.md",
        "field_evidence_status.py --strict",
        "--validate-artifacts",
        "--expected-yaw-rate 0.35",
        "--max-flight-action-run 60",
        "--max-halss-p95-ms 70",
        "--max-depth-p95-ms 15",
        "--max-completion-p95-ms 45",
        "--max-rl-p95-ms 30",
        "--max-sync-p95-ms 100",
        "--max-yaw-transform-error 0.15",
        "--safe-point-tolerance-m 2.0",
        "--out-md experiments/logs/field_evidence_status.md",
        "--gis-bounds 120.0,30.0,121.0,31.0",
        "run_acceptance_light.py",
        "--gis-prior experiments/logs",
        "--gis-bounds 120.0,30.0,121.0,31.0",
        "--nocontrol-log experiments/logs/nocontrol.log",
        "--frame-dir experiments/frames",
        "--drl-diagnosis-json experiments/logs/drl_live_frame.json",
        "--sparsenet-calibration-json experiments/logs/sparsenet_scale.json",
        "--depth-projection-cuda-log experiments/logs/depth_projection_cuda.log",
        "--orin-env-md experiments/logs/orin_env.md",
        "--flight-log experiments/logs/pipeline.log",
        "--max-flight-action-run 60",
        "--max-halss-p95-ms 70",
        "--max-depth-p95-ms 15",
        "--max-completion-p95-ms 45",
        "--max-rl-p95-ms 30",
        "--require-global-guidance",
        "--expected-yaw-rate 0.35",
        "--require-action-probs",
        "--safe-point-tolerance-m 2.0",
        "--require-landing",
        "--strict-flight-ready",
        "--ros-topic-log /livox/lidar=experiments/logs/hz_livox_lidar.log",
        "--ros-topic-log /livox/imu=experiments/logs/hz_livox_imu.log",
        "--ros-topic-log /cloud_registered=experiments/logs/hz_cloud_registered.log",
        "--ros-topic-log /Odometry=experiments/logs/hz_odometry.log",
        "--require-all-ros-topics",
    ):
        assert needle in text, needle + "\n" + text


def test_dry_run_can_skip_env_check_for_bench_debug():
    result = subprocess.run(
        [
            PYTHON,
            str(ROOT / "run_field_acceptance.py"),
            "--expected-yaw-rate", "0.35",
            "--skip-env-check",
            "--dry-run",
        ],
        cwd=str(ROOT),
        text=True,
        capture_output=True,
    )
    text = result.stdout + result.stderr
    assert result.returncode == 0, text
    assert "check_orin_env.py --strict --require-jetson" not in text
    assert "field_evidence_status.py --strict" in text
    assert "--validate-artifacts" in text
    assert "--skip-orin-env" in text
    assert "--orin-env-md" not in text


def test_dry_run_passes_explicit_safe_point():
    result = subprocess.run(
        [
            PYTHON,
            str(ROOT / "run_field_acceptance.py"),
            "--expected-yaw-rate", "0.35",
            "--expected-safe-point", "31.0,121.0",
            "--safe-point-tolerance-m", "1.0",
            "--dry-run",
        ],
        cwd=str(ROOT),
        text=True,
        capture_output=True,
    )
    text = result.stdout + result.stderr
    assert result.returncode == 0, text
    assert "--expected-safe-point 31.0,121.0" in text
    assert "--safe-point-tolerance-m 1.0" in text


def main():
    test_dry_run_prints_full_field_acceptance_commands()
    test_dry_run_can_skip_env_check_for_bench_debug()
    test_dry_run_passes_explicit_safe_point()
    print("=== Lightweight field acceptance wrapper ===")
    print("  OK dry-run prints full evidence status and unified acceptance commands")
    print("  OK dry-run can skip Orin environment check for bench debugging")
    print("  OK dry-run passes explicit expected safe point")
    print("=== ALL PASSED ===")


if __name__ == "__main__":
    main()
