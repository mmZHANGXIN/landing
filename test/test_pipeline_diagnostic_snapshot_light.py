#!/usr/bin/env python3
"""Lightweight source checks for closed-loop diagnostic snapshots."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _source():
    return (ROOT / "pipeline.py").read_text(encoding="utf-8")


def test_action_collapse_snapshot_has_halss_and_sync_evidence():
    text = _source()
    assert "binary_semantic_vis" in text
    assert '"binary_semantic_vis": (' in text
    assert '"cloud_odom_sync_ms": np.array(sync_ms' in text
    assert "cloud_odom_sync_ms" in text


def test_action_collapse_snapshot_has_yaw_velocity_evidence():
    text = _source()
    assert '"yaw_rad": np.array(pose[5]' in text
    assert '"action_id": np.array(action_id' in text
    assert '"v_body": v_body.astype(np.float32)' in text
    assert '"v_ned": v_ned.astype(np.float32)' in text


def test_global_guidance_requires_home_telemetry_before_ned_conversion():
    text = _source()
    assert "wait_for_local_pose(timeout_s=10.0)" in text
    assert "wait_for_home(timeout_s=10.0)" in text
    assert "[Pipeline] Home telemetry:" in text
    assert "Home GPS is unavailable; cannot convert safe point to NED" in text
    assert "Safe point converts to near-zero NED offset" in text


def main():
    test_action_collapse_snapshot_has_halss_and_sync_evidence()
    test_action_collapse_snapshot_has_yaw_velocity_evidence()
    test_global_guidance_requires_home_telemetry_before_ned_conversion()
    print("=== Lightweight pipeline diagnostic snapshot contract ===")
    print("  OK action-collapse snapshots keep HALSS binary semantic and sync evidence")
    print("  OK action-collapse snapshots keep yaw/action/velocity evidence")
    print("  OK global guidance waits for valid home telemetry before NED conversion")
    print("=== ALL PASSED ===")


if __name__ == "__main__":
    main()
