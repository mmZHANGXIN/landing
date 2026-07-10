#!/usr/bin/env python3
"""Lightweight tests for field evidence status summary."""

import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PYTHON = sys.executable


def _touch(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("ok\n", encoding="utf-8")


def _populate(logs, frames):
    for name in (
        "global_prior_20260607.json",
        "hz_livox_lidar.log",
        "hz_livox_imu.log",
        "hz_cloud_registered.log",
        "hz_odometry.log",
        "depth_projection_cuda.log",
        "nocontrol.log",
        "drl_live_frame.json",
        "sparsenet_scale.json",
        "pipeline.log",
        "acceptance_light_20260607_120000.json",
    ):
        _touch(logs / name)
    (logs / "orin_env.md").write_text(
        "\n".join([
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
        ]),
        encoding="utf-8",
    )
    (logs / "depth_projection_cuda.log").write_text(
        "\n".join([
            "  OK center/body-axis/z-buffer/invalid: valid=3 max_diff=0.00e+00",
            "  OK translated yawed pose: valid=3 max_diff=0.00e+00",
            "  OK single-point frame: valid=1 max_diff=0.00e+00",
            "=== CUDA depth projection parity PASSED ===",
        ]),
        encoding="utf-8",
    )
    for name in (
        "000000_calib_frame.npz",
        "000000_binary_semantic.png",
        "000000_depth.png",
    ):
        _touch(frames / name)


def _run(logs, frames, *args):
    return subprocess.run(
        [
            PYTHON,
            str(ROOT / "field_evidence_status.py"),
            "--log-dir",
            str(logs),
            "--frame-dir",
            str(frames),
            "--gis-prior",
            str(logs),
            *args,
        ],
        cwd=str(ROOT),
        text=True,
        capture_output=True,
    )


def test_complete_evidence_passes_strict():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        logs = tmp / "logs"
        frames = tmp / "frames"
        _populate(logs, frames)
        result = _run(logs, frames, "--strict", "--no-hints")
        text = result.stdout + result.stderr
        assert result.returncode == 0, text
        assert "required_missing: 0" in text
        assert "[PASS] nocontrol_log" in text


def test_missing_evidence_fails_strict():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        logs = tmp / "logs"
        frames = tmp / "frames"
        _touch(logs / "hz_livox_lidar.log")
        result = _run(logs, frames, "--strict")
        text = result.stdout + result.stderr
        assert result.returncode != 0, text
        assert "required_missing:" in text
        assert "missing_keys:" in text
        assert "produce:" in text


def test_markdown_report_is_written():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        logs = tmp / "logs"
        frames = tmp / "frames"
        _populate(logs, frames)
        report = tmp / "field_status.md"
        result = _run(logs, frames, "--strict", "--out-md", str(report))
        text = result.stdout + result.stderr
        assert result.returncode == 0, text
        assert "markdown_report:" in text
        md = report.read_text(encoding="utf-8")
        assert "# Field Evidence Status" in md
        assert "required_missing: `0`" in md
        assert "| gis_prior | PASS |" in md
        assert "python accept_gis_prior_light.py" in md
        assert "requirement_audit" in md
        assert "audit_experiment_requirements_light.py --strict-local" in md
        assert "--max-halss-p95-ms <70>" in md
        assert "--max-depth-p95-ms <15>" in md
        assert "--max-sync-p95-ms <100>" in md
        assert "--max-yaw-transform-error <0.15>" in md
        assert "diagnose_nocontrol_action_log.py" in md
        assert "--fail-on-issues" in md
        assert "--frame-glob 'experiments/frames/*_calib_frame.npz'" in md
        assert "--require-live-frame --require-probs --min-items 8" in md
        assert "depth_projection_cuda" in md
        assert "accept_depth_projection_cuda_light.py" in md


def test_non_strict_orin_env_report_fails_strict_status():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        logs = tmp / "logs"
        frames = tmp / "frames"
        _populate(logs, frames)
        (logs / "orin_env.md").write_text(
            "\n".join([
                "# Orin Environment Check",
                "",
                "- required_failures: `0`",
                "",
                "| Key | Required | Status | Detail | Fix |",
                "| --- | --- | --- | --- | --- |",
                "| jetpack | False | WARN | /etc/nv_tegra_release missing |  |",
                "| torch_cuda | False | WARN | import failed |  |",
                "| rclpy | False | WARN | import failed |  |",
            ]),
            encoding="utf-8",
        )
        result = _run(logs, frames, "--strict", "--no-hints")
        text = result.stdout + result.stderr
        assert result.returncode != 0, text
        assert "orin_env" in text
        assert "jetpack was not required" in text


def test_skip_orin_env_allows_bench_status():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        logs = tmp / "logs"
        frames = tmp / "frames"
        _populate(logs, frames)
        (logs / "orin_env.md").unlink()
        result = _run(logs, frames, "--strict", "--skip-orin-env")
        text = result.stdout + result.stderr
        assert result.returncode == 0, text
        assert "required_missing: 0" in text
        assert "orin_env" not in text


def test_validate_artifacts_rejects_placeholder_files():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        logs = tmp / "logs"
        frames = tmp / "frames"
        _populate(logs, frames)
        result = _run(
            logs,
            frames,
            "--strict",
            "--validate-artifacts",
            "--expected-yaw-rate",
            "0.35",
            "--gis-bounds",
            "120.0,30.0,121.0,31.0",
            "--no-hints",
        )
        text = result.stdout + result.stderr
        assert result.returncode != 0, text
        assert "required_missing:" in text
        assert "gis_prior" in text
        assert "nocontrol_log" in text
        assert "pipeline_log" in text
        assert "invalid:" in text


def test_nocontrol_validator_includes_action_diagnosis_not_safe_point():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        logs = tmp / "logs"
        frames = tmp / "frames"
        _populate(logs, frames)
        result = _run(
            logs,
            frames,
            "--validate-artifacts",
            "--expected-yaw-rate",
            "0.35",
            "--expected-safe-point",
            "31.0,121.0",
            "--no-hints",
        )
        text = result.stdout + result.stderr
        assert result.returncode == 0, text
        assert "diagnose_nocontrol_action_log.py" in text
        assert "--fail-on-issues" in text
        assert "accept_nocontrol_log.py" in text
        assert "--expected-safe-point" not in text.split("nocontrol_log", 1)[1].split("pipeline_log", 1)[0]


def main():
    test_complete_evidence_passes_strict()
    test_missing_evidence_fails_strict()
    test_markdown_report_is_written()
    test_non_strict_orin_env_report_fails_strict_status()
    test_skip_orin_env_allows_bench_status()
    test_validate_artifacts_rejects_placeholder_files()
    test_nocontrol_validator_includes_action_diagnosis_not_safe_point()
    print("=== Lightweight field evidence status acceptance ===")
    print("  OK complete evidence passes strict status")
    print("  OK missing evidence fails strict status with hints")
    print("  OK markdown evidence report is written")
    print("  OK non-strict Orin environment report fails strict status")
    print("  OK skip-orin-env allows bench evidence status")
    print("  OK validate-artifacts rejects placeholder evidence files")
    print("  OK no-control validator includes action diagnosis without safe-point arguments")
    print("=== ALL PASSED ===")


if __name__ == "__main__":
    main()
