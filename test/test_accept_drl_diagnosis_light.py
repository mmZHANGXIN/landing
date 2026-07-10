#!/usr/bin/env python3
"""Lightweight tests for DRL diagnosis JSON acceptance."""

import json
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PYTHON = sys.executable


def _row(case, action, name):
    return {
        "case": case,
        "action": action,
        "action_name": name,
        "confidence": 0.7,
        "top3": f"p={action}:{name}:0.700",
        "depth_norm_min": 0.01,
        "depth_norm_mean": 0.05,
        "depth_norm_max": 0.12,
        "sem_norm_min": 0.12,
        "sem_norm_mean": 0.5,
        "sem_norm_max": 0.98,
    }


def _report(collapsed=False, sign=-1, wrong_name=False):
    names = ["HOVER", "N", "NW", "W", "SW", "S", "SE", "E", "NE", "DESCEND"]
    rows = [
        _row("near_safe", 1, "N"),
        _row("mid_safe", 3, "E" if wrong_name else "W"),
    ]
    unique = [3] if collapsed else [1, 3]
    return {
        "policy": "weights/last_step_model_sb3.zip",
        "input_set": "live frames",
        "items": ["000000_calib_frame.npz", "000001_calib_frame.npz"],
        "action_frame": "body",
        "action_lateral_sign": sign,
        "action_names": names,
        "encodings": [
            {
                "depth_mode": "meters_div255",
                "semantic_mode": "gray_unit",
                "unique_actions": unique,
                "unique_action_names": [names[action] for action in unique],
                "collapsed": collapsed,
                "rows": rows if not collapsed else [_row("near_safe", 3, "W"), _row("mid_safe", 3, "W")],
            }
        ],
        "collapsed_encodings": (
            [{"depth_mode": "meters_div255", "semantic_mode": "gray_unit", "action": 3, "action_name": "W"}]
            if collapsed else []
        ),
    }


def _run(report, *args):
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "drl.json"
        path.write_text(json.dumps(report), encoding="utf-8")
        return subprocess.run(
            [PYTHON, str(ROOT / "accept_drl_diagnosis_light.py"), str(path), *args],
            cwd=str(ROOT),
            text=True,
            capture_output=True,
        )


def test_good_report_passes():
    result = _run(_report(), "--require-live-frame", "--require-probs")
    text = result.stdout + result.stderr
    assert result.returncode == 0, text
    assert "[OK] DRL diagnosis evidence accepted" in text


def test_collapsed_report_fails():
    result = _run(_report(collapsed=True), "--require-live-frame")
    text = result.stdout + result.stderr
    assert result.returncode != 0, text
    assert "collapsed encodings present" in text
    assert "produced 1 unique actions" in text


def test_wrong_action_name_fails():
    result = _run(_report(wrong_name=True))
    text = result.stdout + result.stderr
    assert result.returncode != 0, text
    assert "action 3 name='E' expected='W'" in text


def main():
    test_good_report_passes()
    test_collapsed_report_fails()
    test_wrong_action_name_fails()
    print("=== Lightweight DRL diagnosis JSON acceptance ===")
    print("  OK good report passes")
    print("  OK collapsed report fails")
    print("  OK action-name mismatch fails")
    print("=== ALL PASSED ===")


if __name__ == "__main__":
    main()
