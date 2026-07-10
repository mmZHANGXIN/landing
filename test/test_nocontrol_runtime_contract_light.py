#!/usr/bin/env python3
"""Lightweight source checks for the no-control live runtime contract."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _source():
    return (ROOT / "test_live_nocontrol.py").read_text(encoding="utf-8")


def test_yaw_rate_gate_exists_before_ros_init():
    text = _source()
    assert 'parser.add_argument("--require-yaw-rate"' in text
    assert "yaw_rate_rad_s is zero and --require-yaw-rate is set." in text
    assert text.index("args.require_yaw_rate") < text.index("bridge = FastLIOBridge()")


def test_cloud_odom_sync_uses_runtime_budget():
    text = _source()
    assert 'max_cloud_odom_sync_ms", 100.0' in text
    assert "Drop stale cloud/odom pair" in text
    assert "if sync_ms > max_sync_ms" in text
    assert "if sync_ms > 500.0" not in text


def test_startup_log_exposes_no_control_gates():
    text = _source()
    assert "max_cloud_odom_sync_ms=%.0f" in text
    assert "No-control config: yaw_rate_rad_s" in text
    assert "Action mapping: frame=%s lateral_sign=%d act3=%s" in text


def main():
    test_yaw_rate_gate_exists_before_ros_init()
    test_cloud_odom_sync_uses_runtime_budget()
    test_startup_log_exposes_no_control_gates()
    print("=== Lightweight no-control runtime contract acceptance ===")
    print("  OK --require-yaw-rate rejects zero yaw-fault runs before ROS init")
    print("  OK cloud/odom sync uses runtime.max_cloud_odom_sync_ms")
    print("  OK startup logs expose yaw rate, sync budget, and action mapping")
    print("=== ALL PASSED ===")


if __name__ == "__main__":
    main()
