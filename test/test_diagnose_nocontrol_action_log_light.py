#!/usr/bin/env python3
"""Lightweight tests for no-control action-collapse diagnosis."""

import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PYTHON = sys.executable


def _mapping(sign=-1, act3="W"):
    return f"2026-06-05 19:00:00,000 [INFO] Action mapping: frame=body lateral_sign={sign} act3={act3}"


def _modern_line(seq, action=3, name="W", yr=0.35, valid=0.06, safe=0.55, danger=0.45,
                 probs=True):
    suffix = "conf=0.74 p=3:W:0.740,7:E:0.130,1:N:0.080" if probs else "conf=n/a"
    return (
        f"2026-06-05 19:00:00,000 [INFO] [{seq:04d}] act={action}({name}) "
        "H=30ms D=4ms C=20ms RL=10ms total=76ms | yaw=10deg "
        f"yr={yr:.2f} sync=22ms v_body=[0.0,-1.0,0.0] v_ned=[0.2,-1.0,0.0] "
        "| depth=3.0/12.0/30.0m obsD=0.01/0.05/0.12 obsS=0.12/0.50/0.98 "
        f"valid={valid:.2f} sem_safe={safe:.2f} sem_danger={danger:.2f} {suffix}"
    )


def _run(log_text, *args):
    with tempfile.TemporaryDirectory() as tmpdir:
        log = Path(tmpdir) / "nocontrol.log"
        log.write_text(log_text, encoding="utf-8")
        return subprocess.run(
            [PYTHON, str(ROOT / "diagnose_nocontrol_action_log.py"), str(log), *args],
            cwd=str(ROOT),
            text=True,
            capture_output=True,
        )


def test_legacy_act3_e_zero_yaw_is_explained():
    lines = [_mapping(sign=1, act3="E")]
    for i in range(12):
        lines.append(
            f"2026-06-04 18:15:{20 + i:02d},000 [INFO] [{146 + i:04d}] "
            "act=3(E) H=67ms D=6ms C=32ms RL=24ms total=129ms "
            "| yaw=10deg yr=0.00 v_body=[-0.0,1.0,0.0] v_ned=[-0.2,1.0,0.0]"
        )
    result = _run("\n".join(lines), "--fail-on-issues")
    text = result.stdout + result.stderr
    assert result.returncode == 1, text
    assert "ACTION_MAPPING_MISMATCH" in text
    assert "ACTION_NAME_MISMATCH" in text
    assert "LEGACY_LOG_FORMAT" in text
    assert "ZERO_YAW_RATE" in text
    assert "SINGLE_ACTION_COLLAPSE" in text
    assert "act=3 logged=E expected=W" in text


def test_modern_single_action_reports_live_frame_diagnosis_next_step():
    lines = [_mapping()]
    lines.extend(_modern_line(i + 1) for i in range(20))
    result = _run("\n".join(lines), "--expected-yaw-rate", "0.35", "--fail-on-issues")
    text = result.stdout + result.stderr
    assert result.returncode == 1, text
    assert "SINGLE_ACTION_COLLAPSE" in text
    assert "diagnose_drl_policy.py --scan-modes" in text
    assert "ZERO_YAW_RATE" not in text
    assert "LEGACY_LOG_FORMAT" not in text


def test_modern_multiaction_log_passes_diagnosis():
    lines = [_mapping()]
    names = {1: "N", 3: "W", 7: "E"}
    for i in range(18):
        action = (1, 3, 7)[i % 3]
        lines.append(_modern_line(i + 1, action=action, name=names[action]))
    result = _run("\n".join(lines), "--expected-yaw-rate", "0.35", "--fail-on-issues")
    text = result.stdout + result.stderr
    assert result.returncode == 0, text
    assert "[OK] No action-collapse diagnosis issues detected" in text


def test_one_class_semantic_is_reported():
    lines = [_mapping()]
    for i in range(18):
        action = (1, 3, 7)[i % 3]
        name = {1: "N", 3: "W", 7: "E"}[action]
        lines.append(_modern_line(i + 1, action=action, name=name, safe=0.99, danger=0.01))
    result = _run("\n".join(lines), "--expected-yaw-rate", "0.35", "--fail-on-issues")
    text = result.stdout + result.stderr
    assert result.returncode == 1, text
    assert "SEMANTIC_ONE_CLASS" in text
    assert "HALSS semantic map is nearly one class" in text


def main():
    test_legacy_act3_e_zero_yaw_is_explained()
    test_modern_single_action_reports_live_frame_diagnosis_next_step()
    test_modern_multiaction_log_passes_diagnosis()
    test_one_class_semantic_is_reported()
    print("=== Lightweight no-control action diagnosis ===")
    print("  OK legacy act=3(E), zero-yaw logs get actionable findings")
    print("  OK modern single-action logs point to live-frame DRL diagnosis")
    print("  OK multi-action yaw-fault logs pass")
    print("  OK one-class semantic observations are reported")
    print("=== ALL PASSED ===")


if __name__ == "__main__":
    main()
