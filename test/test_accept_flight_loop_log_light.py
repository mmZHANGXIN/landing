#!/usr/bin/env python3
"""Lightweight tests for closed-loop flight log acceptance."""

import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PYTHON = sys.executable


def _frame(seq, action, name, total, yaw_sp, yr=0.35, vx=-0.2, vy=1.0,
           h=20, d=3, c=18, rl=8, sync=25):
    return (
        f"2026-06-07 13:00:00,000 [INFO] [{seq:04d}] act={action}({name}) "
        f"H={h}ms D={d}ms C={c}ms RL={rl}ms total={total}ms "
        f"| yaw=10deg yaw_sp={yaw_sp:.0f}deg yr={yr:.2f} sync={sync}ms "
        f"v_body=[0.0,1.0,0.0] v_ned=[{vx:.1f},{vy:.1f},0.0] "
        "| depth=3.0/12.0/30.0m obsD=0.01/0.05/0.12 obsS=0.12/0.50/0.98 "
        "valid=0.04 sem_safe=0.55 sem_danger=0.45 conf=0.72 p=3:W:0.72,7:E:0.20"
    )


def _run(log_text, *args):
    with tempfile.TemporaryDirectory() as tmp:
        log = Path(tmp) / "pipeline.log"
        log.write_text(log_text, encoding="utf-8")
        return subprocess.run(
            [PYTHON, str(ROOT / "accept_flight_loop_log.py"), str(log), *args],
            cwd=str(ROOT),
            text=True,
            capture_output=True,
        )


def _mapping(sign=-1, act3="W"):
    return f"2026-06-07 13:00:00,050 [INFO] [Init] Action mapping frame=body lateral_sign={sign} (act3={act3})"


def test_good_global_guidance_log_passes():
    lines = [
        "2026-06-07 13:00:00,000 [INFO] [GlobalPrior] Safe point from GIS: lat=31.0000000 lon=121.0000000",
        _mapping(),
        "2026-06-07 13:00:00,100 [INFO] [FlightReady] Strict experiment gates passed.",
        "2026-06-07 13:00:01,000 [INFO] [Pipeline] FAST-LIO ready.",
        "2026-06-07 13:00:02,000 [INFO] [Pipeline] Global guidance target: 31.0000000, 121.0000000",
        "2026-06-07 13:00:02,100 [INFO] [GOTO_SAFE] Target NED: 5.0, 2.0 tolerance=2.0m",
        "2026-06-07 13:00:03,000 [INFO] [GOTO_SAFE] Distance 4.00m yaw_sp=1.0deg",
        "2026-06-07 13:00:04,000 [INFO] [GOTO_SAFE] Arrived. XY error=1.20m",
        "2026-06-07 13:00:04,100 [INFO] [Pipeline] Starting DRL descent control loop...",
    ]
    names = {1: "N", 3: "W", 7: "E"}
    for i in range(30):
        action = (1, 3, 7)[i % 3]
        lines.append(_frame(i + 1, action, names[action], 70 + (i % 3), yaw_sp=5 + i))
    lines.append("2026-06-07 13:00:20,000 [INFO] [Pipeline] Target altitude reached: 0.45m")
    result = _run(
        "\n".join(lines),
        "--require-global-guidance",
        "--require-landing",
        "--require-action-probs",
        "--expected-yaw-rate",
        "0.35",
        "--expected-safe-point",
        "31.0,121.0",
        "--min-drl-frames",
        "20",
    )
    text = result.stdout + result.stderr
    assert result.returncode == 0, text
    assert "[OK] flight-loop log acceptance passed" in text


def test_expected_safe_point_mismatch_fails():
    lines = [
        "2026-06-07 13:00:00,000 [INFO] [GlobalPrior] Safe point from GIS: lat=31.0000000 lon=121.0000000",
        _mapping(),
        "2026-06-07 13:00:00,100 [INFO] [FlightReady] Strict experiment gates passed.",
        "2026-06-07 13:00:01,000 [INFO] [Pipeline] FAST-LIO ready.",
        "2026-06-07 13:00:02,000 [INFO] [Pipeline] Global guidance target: 31.0000000, 121.0000000",
        "2026-06-07 13:00:02,100 [INFO] [GOTO_SAFE] Target NED: 5.0, 2.0 tolerance=2.0m",
        "2026-06-07 13:00:04,000 [INFO] [GOTO_SAFE] Arrived. XY error=1.20m",
        "2026-06-07 13:00:04,100 [INFO] [Pipeline] Starting DRL descent control loop...",
    ]
    names = {1: "N", 3: "W", 7: "E"}
    for i in range(30):
        action = (1, 3, 7)[i % 3]
        lines.append(_frame(i + 1, action, names[action], 70 + (i % 3), yaw_sp=5 + i))
    result = _run(
        "\n".join(lines),
        "--require-global-guidance",
        "--require-action-probs",
        "--expected-yaw-rate",
        "0.35",
        "--expected-safe-point",
        "31.01,121.01",
        "--safe-point-tolerance-m",
        "2.0",
        "--min-drl-frames",
        "20",
    )
    text = result.stdout + result.stderr
    assert result.returncode != 0, text
    assert "safe-point GPS mismatch" in text
    assert "global guidance target GPS mismatch" in text


def test_goto_arrival_error_above_tolerance_fails():
    lines = [
        "2026-06-07 13:00:00,000 [INFO] [GlobalPrior] Safe point from GIS: lat=31.0000000 lon=121.0000000",
        _mapping(),
        "2026-06-07 13:00:00,100 [INFO] [FlightReady] Strict experiment gates passed.",
        "2026-06-07 13:00:01,000 [INFO] [Pipeline] FAST-LIO ready.",
        "2026-06-07 13:00:02,000 [INFO] [Pipeline] Global guidance target: 31.0000000, 121.0000000",
        "2026-06-07 13:00:02,100 [INFO] [GOTO_SAFE] Target NED: 5.0, 2.0 tolerance=1.0m",
        "2026-06-07 13:00:04,000 [INFO] [GOTO_SAFE] Arrived. XY error=1.20m",
        "2026-06-07 13:00:04,100 [INFO] [Pipeline] Starting DRL descent control loop...",
    ]
    names = {1: "N", 3: "W", 7: "E"}
    for i in range(30):
        action = (1, 3, 7)[i % 3]
        lines.append(_frame(i + 1, action, names[action], 70 + (i % 3), yaw_sp=5 + i))
    result = _run(
        "\n".join(lines),
        "--require-global-guidance",
        "--require-action-probs",
        "--expected-yaw-rate",
        "0.35",
        "--min-drl-frames",
        "20",
    )
    text = result.stdout + result.stderr
    assert result.returncode != 0, text
    assert "GOTO_SAFE arrived XY error 1.20m > tolerance 1.00m" in text


def test_goto_target_near_zero_fails():
    lines = [
        "2026-06-07 13:00:00,000 [INFO] [GlobalPrior] Safe point from GIS: lat=31.0000000 lon=121.0000000",
        _mapping(),
        "2026-06-07 13:00:00,100 [INFO] [FlightReady] Strict experiment gates passed.",
        "2026-06-07 13:00:01,000 [INFO] [Pipeline] FAST-LIO ready.",
        "2026-06-07 13:00:02,000 [INFO] [Pipeline] Global guidance target: 31.0000000, 121.0000000",
        "2026-06-07 13:00:02,100 [INFO] [GOTO_SAFE] Target NED: 0.0, 0.0 tolerance=2.0m",
        "2026-06-07 13:00:04,000 [INFO] [GOTO_SAFE] Arrived. XY error=0.20m",
        "2026-06-07 13:00:04,100 [INFO] [Pipeline] Starting DRL descent control loop...",
    ]
    names = {1: "N", 3: "W", 7: "E"}
    for i in range(30):
        action = (1, 3, 7)[i % 3]
        lines.append(_frame(i + 1, action, names[action], 70 + (i % 3), yaw_sp=5 + i))
    result = _run(
        "\n".join(lines),
        "--require-global-guidance",
        "--require-action-probs",
        "--expected-yaw-rate",
        "0.35",
        "--min-drl-frames",
        "20",
    )
    text = result.stdout + result.stderr
    assert result.returncode != 0, text
    assert "GOTO_SAFE target NED is near zero" in text


def test_bypass_zero_yaw_log_fails():
    lines = [
        "2026-06-07 13:00:00,100 [WARNING] [FlightReady] Strict gates bypassed by --allow-incomplete-experiment",
        _mapping(),
        "2026-06-07 13:00:01,000 [INFO] [Pipeline] FAST-LIO ready.",
        "2026-06-07 13:00:01,100 [INFO] [Pipeline] Starting DRL descent control loop...",
    ]
    for i in range(5):
        lines.append(_frame(i + 1, 3, "W", 70, yaw_sp=0, yr=0.0, vx=0.0, vy=0.0))
    result = _run("\n".join(lines), "--require-global-guidance", "--min-drl-frames", "10")
    text = result.stdout + result.stderr
    assert result.returncode != 0, text
    assert "strict FlightReady pass log is missing" in text
    assert "strict FlightReady gates were bypassed" in text
    assert "required global-guidance event missing" in text
    assert "yaw_rate is zero" in text
    assert "no nonzero NED velocity commands logged" in text


def test_mirrored_action_label_fails_by_default():
    lines = [
        "2026-06-07 13:00:00,000 [INFO] [GlobalPrior] Safe point from GIS: lat=31.0000000 lon=121.0000000",
        _mapping(sign=1, act3="E"),
        "2026-06-07 13:00:00,100 [INFO] [FlightReady] Strict experiment gates passed.",
        "2026-06-07 13:00:01,000 [INFO] [Pipeline] FAST-LIO ready.",
        "2026-06-07 13:00:02,000 [INFO] [Pipeline] Global guidance target: 31.0000000, 121.0000000",
        "2026-06-07 13:00:02,100 [INFO] [GOTO_SAFE] Target NED: 5.0, 2.0 tolerance=2.0m",
        "2026-06-07 13:00:04,000 [INFO] [GOTO_SAFE] Arrived. XY error=1.20m",
        "2026-06-07 13:00:04,100 [INFO] [Pipeline] Starting DRL descent control loop...",
    ]
    names = {1: "N", 3: "E", 7: "W"}
    for i in range(30):
        action = (1, 3, 7)[i % 3]
        lines.append(_frame(i + 1, action, names[action], 70 + (i % 3), yaw_sp=5 + i))
    result = _run(
        "\n".join(lines),
        "--require-global-guidance",
        "--require-action-probs",
        "--expected-yaw-rate",
        "0.35",
        "--min-drl-frames",
        "20",
    )
    text = result.stdout + result.stderr
    assert result.returncode != 0, text
    assert "action id/name mismatch" in text
    assert "action mapping evidence invalid" in text
    assert "act=3 logged=E expected=W" in text


def test_missing_action_mapping_fails():
    lines = [
        "2026-06-07 13:00:00,000 [INFO] [GlobalPrior] Safe point from GIS: lat=31.0000000 lon=121.0000000",
        "2026-06-07 13:00:00,100 [INFO] [FlightReady] Strict experiment gates passed.",
        "2026-06-07 13:00:01,000 [INFO] [Pipeline] FAST-LIO ready.",
        "2026-06-07 13:00:02,000 [INFO] [Pipeline] Global guidance target: 31.0000000, 121.0000000",
        "2026-06-07 13:00:02,100 [INFO] [GOTO_SAFE] Target NED: 5.0, 2.0 tolerance=2.0m",
        "2026-06-07 13:00:04,000 [INFO] [GOTO_SAFE] Arrived. XY error=1.20m",
        "2026-06-07 13:00:04,100 [INFO] [Pipeline] Starting DRL descent control loop...",
    ]
    names = {1: "N", 3: "W", 7: "E"}
    for i in range(30):
        action = (1, 3, 7)[i % 3]
        lines.append(_frame(i + 1, action, names[action], 70 + (i % 3), yaw_sp=5 + i))
    result = _run(
        "\n".join(lines),
        "--require-global-guidance",
        "--expected-yaw-rate",
        "0.35",
        "--min-drl-frames",
        "20",
    )
    text = result.stdout + result.stderr
    assert result.returncode != 0, text
    assert "runtime action mapping log is missing" in text


def test_single_action_nonzero_yaw_fails():
    lines = [
        "2026-06-07 13:00:00,000 [INFO] [GlobalPrior] Safe point from GIS: lat=31.0000000 lon=121.0000000",
        _mapping(),
        "2026-06-07 13:00:00,100 [INFO] [FlightReady] Strict experiment gates passed.",
        "2026-06-07 13:00:01,000 [INFO] [Pipeline] FAST-LIO ready.",
        "2026-06-07 13:00:02,000 [INFO] [Pipeline] Global guidance target: 31.0000000, 121.0000000",
        "2026-06-07 13:00:02,100 [INFO] [GOTO_SAFE] Target NED: 5.0, 2.0 tolerance=2.0m",
        "2026-06-07 13:00:04,000 [INFO] [GOTO_SAFE] Arrived. XY error=1.20m",
        "2026-06-07 13:00:04,100 [INFO] [Pipeline] Starting DRL descent control loop...",
    ]
    for i in range(30):
        lines.append(_frame(i + 1, 3, "W", 70, yaw_sp=5 + i, yr=0.35, vx=0.2, vy=0.9))
    result = _run(
        "\n".join(lines),
        "--require-global-guidance",
        "--require-action-probs",
        "--expected-yaw-rate",
        "0.35",
        "--min-drl-frames",
        "20",
        "--max-action-run",
        "20",
    )
    text = result.stdout + result.stderr
    assert result.returncode != 0, text
    assert "all parsed DRL frames selected one action" in text
    assert "longest repeated action run" in text


def test_module_p95_budget_fails_even_when_total_passes():
    lines = [
        "2026-06-07 13:00:00,000 [INFO] [GlobalPrior] Safe point from GIS: lat=31.0000000 lon=121.0000000",
        _mapping(),
        "2026-06-07 13:00:00,100 [INFO] [FlightReady] Strict experiment gates passed.",
        "2026-06-07 13:00:01,000 [INFO] [Pipeline] FAST-LIO ready.",
        "2026-06-07 13:00:02,000 [INFO] [Pipeline] Global guidance target: 31.0000000, 121.0000000",
        "2026-06-07 13:00:02,100 [INFO] [GOTO_SAFE] Target NED: 5.0, 2.0 tolerance=2.0m",
        "2026-06-07 13:00:04,000 [INFO] [GOTO_SAFE] Arrived. XY error=1.20m",
        "2026-06-07 13:00:04,100 [INFO] [Pipeline] Starting DRL descent control loop...",
    ]
    names = {1: "N", 3: "W", 7: "E"}
    for i in range(30):
        action = (1, 3, 7)[i % 3]
        lines.append(_frame(i + 1, action, names[action], 90, yaw_sp=5 + i, h=20, d=20, c=10, rl=5))
    result = _run(
        "\n".join(lines),
        "--require-global-guidance",
        "--require-action-probs",
        "--expected-yaw-rate",
        "0.35",
        "--min-drl-frames",
        "20",
        "--max-total-p95-ms",
        "100",
        "--max-depth-p95-ms",
        "15",
    )
    text = result.stdout + result.stderr
    assert result.returncode != 0, text
    assert "depth projection P95 20.0ms > 15.0ms" in text


def test_cloud_odom_sync_budget_fails():
    lines = [
        "2026-06-07 13:00:00,000 [INFO] [GlobalPrior] Safe point from GIS: lat=31.0000000 lon=121.0000000",
        _mapping(),
        "2026-06-07 13:00:00,100 [INFO] [FlightReady] Strict experiment gates passed.",
        "2026-06-07 13:00:01,000 [INFO] [Pipeline] FAST-LIO ready.",
        "2026-06-07 13:00:02,000 [INFO] [Pipeline] Global guidance target: 31.0000000, 121.0000000",
        "2026-06-07 13:00:02,100 [INFO] [GOTO_SAFE] Target NED: 5.0, 2.0 tolerance=2.0m",
        "2026-06-07 13:00:04,000 [INFO] [GOTO_SAFE] Arrived. XY error=1.20m",
        "2026-06-07 13:00:04,100 [INFO] [Pipeline] Starting DRL descent control loop...",
    ]
    names = {1: "N", 3: "W", 7: "E"}
    for i in range(30):
        action = (1, 3, 7)[i % 3]
        lines.append(_frame(i + 1, action, names[action], 70, yaw_sp=5 + i, sync=180))
    result = _run(
        "\n".join(lines),
        "--require-global-guidance",
        "--require-action-probs",
        "--expected-yaw-rate",
        "0.35",
        "--min-drl-frames",
        "20",
        "--max-sync-p95-ms",
        "100",
    )
    text = result.stdout + result.stderr
    assert result.returncode != 0, text
    assert "cloud/odom sync P95 180.0ms > 100.0ms" in text


def test_yaw_transform_error_fails():
    lines = [
        "2026-06-07 13:00:00,000 [INFO] [GlobalPrior] Safe point from GIS: lat=31.0000000 lon=121.0000000",
        _mapping(),
        "2026-06-07 13:00:00,100 [INFO] [FlightReady] Strict experiment gates passed.",
        "2026-06-07 13:00:01,000 [INFO] [Pipeline] FAST-LIO ready.",
        "2026-06-07 13:00:02,000 [INFO] [Pipeline] Global guidance target: 31.0000000, 121.0000000",
        "2026-06-07 13:00:02,100 [INFO] [GOTO_SAFE] Target NED: 5.0, 2.0 tolerance=2.0m",
        "2026-06-07 13:00:04,000 [INFO] [GOTO_SAFE] Arrived. XY error=1.20m",
        "2026-06-07 13:00:04,100 [INFO] [Pipeline] Starting DRL descent control loop...",
    ]
    names = {1: "N", 3: "W", 7: "E"}
    for i in range(30):
        action = (1, 3, 7)[i % 3]
        lines.append(_frame(i + 1, action, names[action], 70, yaw_sp=5 + i, vx=0.0, vy=1.0))
    result = _run(
        "\n".join(lines),
        "--require-global-guidance",
        "--require-action-probs",
        "--expected-yaw-rate",
        "0.35",
        "--min-drl-frames",
        "20",
        "--max-yaw-transform-error",
        "0.15",
    )
    text = result.stdout + result.stderr
    assert result.returncode != 0, text
    assert "yaw transform error P95" in text


def main():
    test_good_global_guidance_log_passes()
    test_expected_safe_point_mismatch_fails()
    test_goto_arrival_error_above_tolerance_fails()
    test_goto_target_near_zero_fails()
    test_bypass_zero_yaw_log_fails()
    test_mirrored_action_label_fails_by_default()
    test_missing_action_mapping_fails()
    test_single_action_nonzero_yaw_fails()
    test_module_p95_budget_fails_even_when_total_passes()
    test_cloud_odom_sync_budget_fails()
    test_yaw_transform_error_fails()
    print("=== Lightweight flight-loop log acceptance ===")
    print("  OK good GIS-to-DRL log passes")
    print("  OK GIS prior safe-point mismatch fails")
    print("  OK GOTO_SAFE arrival error above tolerance fails")
    print("  OK near-zero GOTO_SAFE target fails")
    print("  OK bypass/zero-yaw log fails")
    print("  OK mirrored action labels fail by default")
    print("  OK missing startup action mapping log fails")
    print("  OK nonzero-yaw single-action log fails")
    print("  OK module-level timing budget fails independently of total P95")
    print("  OK cloud/odom sync budget fails")
    print("  OK yaw transform error fails")
    print("=== ALL PASSED ===")


if __name__ == "__main__":
    main()
