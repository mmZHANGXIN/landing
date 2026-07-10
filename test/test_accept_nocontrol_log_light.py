#!/usr/bin/env python3
"""Lightweight tests for no-control log acceptance."""

import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PYTHON = sys.executable


def _line(seq, action, name, yr, total, valid=0.05, safe=0.55, danger=0.45,
          h=20, d=3, c=25, rl=8):
    return (
        f"2026-06-05 19:00:00,000 [INFO] [{seq:04d}] act={action}({name}) "
        f"H={h}ms D={d}ms C={c}ms RL={rl}ms total={total}ms | yaw=10° yr={yr:.2f} sync=25ms "
        "v_body=[0.0,1.0,0.0] v_ned=[-0.2,1.0,0.0] "
        "| depth=3.0/12.0/30.0m obsD=0.01/0.05/0.12 obsS=0.12/0.50/0.98 "
        f"valid={valid:.2f} sem_safe={safe:.2f} sem_danger={danger:.2f} "
        "conf=0.73 p=3:W:0.73,7:E:0.20,1:N:0.07"
    )


def _run(log_text, *args):
    with tempfile.TemporaryDirectory() as tmpdir:
        log = Path(tmpdir) / "nocontrol.log"
        log.write_text(log_text, encoding="utf-8")
        return subprocess.run(
            [PYTHON, str(ROOT / "accept_nocontrol_log.py"), str(log), *args],
            cwd=str(ROOT),
            text=True,
            capture_output=True,
        )


def _mapping(sign=-1, act3="W"):
    return f"2026-06-05 19:00:00,000 [INFO] Action mapping: frame=body lateral_sign={sign} act3={act3}"


def test_good_log_passes():
    lines = [_mapping()]
    names = {1: "N", 3: "W", 7: "E"}
    for i in range(90):
        action = (1, 3, 7)[i % 3]
        lines.append(_line(i + 1, action, names[action], 0.35, 70 + (i % 5)))
    result = _run(
        "\n".join(lines),
        "--min-samples", "60",
        "--max-total-p95-ms", "90",
        "--expected-yaw-rate", "0.35",
        "--require-action-probs",
    )
    text = result.stdout + result.stderr
    assert result.returncode == 0, text
    assert "[OK] no-control log acceptance passed" in text


def test_single_action_zero_yaw_fails():
    lines = [_mapping()]
    lines.extend(_line(i + 1, 3, "W", 0.0, 70) for i in range(80))
    result = _run("\n".join(lines), "--min-samples", "60", "--max-action-run", "20")
    text = result.stdout + result.stderr
    assert result.returncode != 0, text
    assert "yaw_rate is zero" in text
    assert "all parsed frames selected one action" in text
    assert "longest repeated action run" in text


def test_mirrored_action_label_fails_by_default():
    lines = [_mapping(sign=1, act3="E")]
    names = {1: "N", 3: "E", 7: "W"}
    for i in range(90):
        action = (1, 3, 7)[i % 3]
        lines.append(_line(i + 1, action, names[action], 0.35, 70 + (i % 5)))
    result = _run(
        "\n".join(lines),
        "--min-samples", "60",
        "--max-total-p95-ms", "90",
        "--expected-yaw-rate", "0.35",
        "--require-action-probs",
    )
    text = result.stdout + result.stderr
    assert result.returncode != 0, text
    assert "action id/name mismatch" in text
    assert "action mapping evidence invalid" in text
    assert "act=3 logged=E expected=W" in text


def test_mirrored_action_label_can_be_explicitly_accepted():
    lines = [_mapping(sign=1, act3="E")]
    names = {1: "N", 3: "E", 7: "W"}
    for i in range(90):
        action = (1, 3, 7)[i % 3]
        lines.append(_line(i + 1, action, names[action], 0.35, 70 + (i % 5)))
    result = _run(
        "\n".join(lines),
        "--min-samples", "60",
        "--max-total-p95-ms", "90",
        "--expected-yaw-rate", "0.35",
        "--require-action-probs",
        "--action-lateral-sign", "1",
    )
    text = result.stdout + result.stderr
    assert result.returncode == 0, text
    assert "[OK] no-control log acceptance passed" in text


def test_missing_action_mapping_fails():
    lines = []
    names = {1: "N", 3: "W", 7: "E"}
    for i in range(90):
        action = (1, 3, 7)[i % 3]
        lines.append(_line(i + 1, action, names[action], 0.35, 70 + (i % 5)))
    result = _run(
        "\n".join(lines),
        "--min-samples", "60",
        "--max-total-p95-ms", "90",
        "--expected-yaw-rate", "0.35",
    )
    text = result.stdout + result.stderr
    assert result.returncode != 0, text
    assert "runtime action mapping log is missing" in text


def test_module_p95_budget_fails_even_when_total_passes():
    lines = [_mapping()]
    names = {1: "N", 3: "W", 7: "E"}
    for i in range(90):
        action = (1, 3, 7)[i % 3]
        lines.append(_line(i + 1, action, names[action], 0.35, 90, h=80, d=3, c=5, rl=2))
    result = _run(
        "\n".join(lines),
        "--min-samples", "60",
        "--max-total-p95-ms", "100",
        "--max-halss-p95-ms", "70",
        "--expected-yaw-rate", "0.35",
        "--require-action-probs",
    )
    text = result.stdout + result.stderr
    assert result.returncode != 0, text
    assert "HALSS P95 80.0ms > 70.0ms" in text


def test_cloud_odom_sync_budget_fails():
    lines = [_mapping()]
    names = {1: "N", 3: "W", 7: "E"}
    for i in range(90):
        action = (1, 3, 7)[i % 3]
        line = _line(i + 1, action, names[action], 0.35, 70 + (i % 5))
        lines.append(line.replace("sync=25ms", "sync=180ms"))
    result = _run(
        "\n".join(lines),
        "--min-samples", "60",
        "--expected-yaw-rate", "0.35",
        "--max-sync-p95-ms", "100",
        "--require-action-probs",
    )
    text = result.stdout + result.stderr
    assert result.returncode != 0, text
    assert "cloud/odom sync P95 180.0ms > 100.0ms" in text


def test_yaw_transform_error_fails():
    lines = [_mapping()]
    names = {1: "N", 3: "W", 7: "E"}
    for i in range(90):
        action = (1, 3, 7)[i % 3]
        line = _line(i + 1, action, names[action], 0.35, 70 + (i % 5))
        lines.append(line.replace("v_ned=[-0.2,1.0,0.0]", "v_ned=[0.0,1.0,0.0]"))
    result = _run(
        "\n".join(lines),
        "--min-samples", "60",
        "--expected-yaw-rate", "0.35",
        "--max-yaw-transform-error", "0.15",
        "--require-action-probs",
    )
    text = result.stdout + result.stderr
    assert result.returncode != 0, text
    assert "yaw transform error P95" in text


def test_legacy_act3_e_log_gets_specific_diagnosis():
    lines = [_mapping(sign=1, act3="E")]
    for i in range(12):
        lines.append(
            f"2026-06-04 18:15:{20 + i:02d},000 [INFO] [{146 + i:04d}] "
            "act=3(E) H=67ms D=6ms C=32ms RL=24ms total=129ms "
            "| yaw=10° yr=0.00 v_body=[-0.0,1.0,0.0] v_ned=[-0.2,1.0,0.0]"
        )
    result = _run("\n".join(lines), "--min-samples", "1")
    text = result.stdout + result.stderr
    assert result.returncode != 0, text
    assert "legacy no-control log format detected" in text
    assert "act=3 logged=E expected=W" in text
    assert "action mapping evidence invalid" in text


def main():
    test_good_log_passes()
    test_single_action_zero_yaw_fails()
    test_mirrored_action_label_fails_by_default()
    test_mirrored_action_label_can_be_explicitly_accepted()
    test_missing_action_mapping_fails()
    test_module_p95_budget_fails_even_when_total_passes()
    test_cloud_odom_sync_budget_fails()
    test_yaw_transform_error_fails()
    test_legacy_act3_e_log_gets_specific_diagnosis()
    print("=== Lightweight no-control log acceptance ===")
    print("  OK good log passes")
    print("  OK zero-yaw single-action log fails")
    print("  OK mirrored action labels fail unless explicitly requested")
    print("  OK missing startup action mapping log fails")
    print("  OK module-level timing budget fails independently of total P95")
    print("  OK cloud/odom sync budget fails")
    print("  OK yaw transform error fails")
    print("  OK legacy act=3(E) logs get specific diagnosis")
    print("=== ALL PASSED ===")


if __name__ == "__main__":
    main()
