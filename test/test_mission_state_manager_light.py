#!/usr/bin/env python3
"""Lightweight tests for mission-state transitions."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from control.mission_state_manager import MissionState, MissionStateManager, StateInputs


def _inputs(z=-1.5, ground=1.5, step=0):
    return StateInputs(
        now=1.0,
        pose_xyz=(0.0, 0.0, z),
        yaw_rad=0.25,
        cloud_odom_sync_ms=10.0,
        perception_ok=True,
        offboard_active=True,
        armed=True,
        step_count=step,
        max_steps=100,
        ground_clearance_p05_m=ground,
        ground_clearance_min_m=ground,
    )


def test_fastlio_z_triggers_direct_land():
    mgr = MissionStateManager({
        "height_axis": "neg_z",
        "ground_z_ref_m": 0.0,
        "direct_land_trigger_height_m": 0.8,
        "ground_crosscheck_action": "warn",
    })
    mgr.start_after_takeoff(False)
    decision = mgr.update(_inputs(z=-0.7, ground=0.72))
    assert decision.state == MissionState.DIRECT_LAND
    assert decision.direct_land
    assert decision.reason == "height_below_direct_land_trigger"
    assert decision.land_reference_xy_yaw == (0.0, 0.0, 0.25)


def test_ground_crosscheck_warn_does_not_block():
    mgr = MissionStateManager({
        "height_axis": "neg_z",
        "direct_land_trigger_height_m": 0.8,
        "ground_crosscheck_action": "warn",
        "ground_crosscheck_max_error_m": 0.2,
    })
    mgr.start_after_takeoff(False)
    decision = mgr.update(_inputs(z=-0.7, ground=2.0))
    assert decision.state == MissionState.DIRECT_LAND
    assert decision.reason.endswith("crosscheck_warn")


def test_ground_crosscheck_block_can_block():
    mgr = MissionStateManager({
        "height_axis": "neg_z",
        "direct_land_trigger_height_m": 0.8,
        "ground_crosscheck_action": "block",
        "ground_crosscheck_max_error_m": 0.2,
    })
    mgr.start_after_takeoff(False)
    decision = mgr.update(_inputs(z=-0.7, ground=2.0))
    assert decision.state == MissionState.DRL_DESCENT
    assert not decision.direct_land


def test_pointcloud_low_without_low_pose_does_not_trigger():
    mgr = MissionStateManager({
        "height_axis": "neg_z",
        "direct_land_trigger_height_m": 0.8,
    })
    mgr.start_after_takeoff(False)
    decision = mgr.update(_inputs(z=-2.0, ground=0.2))
    assert decision.state == MissionState.DRL_DESCENT
    assert not decision.direct_land


def test_pose_timeout_aborts_in_descent():
    mgr = MissionStateManager({"pose_timeout_s": 0.2})
    mgr.start_after_takeoff(False)
    decision = mgr.update(StateInputs(
        now=1.0,
        pose_xyz=(0.0, 0.0, -1.0),
        pose_age_ms=250.0,
        offboard_active=True,
        armed=True,
    ))
    assert decision.state == MissionState.ABORT
    assert decision.abort


def main():
    test_fastlio_z_triggers_direct_land()
    test_ground_crosscheck_warn_does_not_block()
    test_ground_crosscheck_block_can_block()
    test_pointcloud_low_without_low_pose_does_not_trigger()
    test_pose_timeout_aborts_in_descent()
    print("=== Mission state manager light tests ===")
    print("  OK fastlio z triggers direct land")
    print("  OK ground crosscheck warn/block behavior")
    print("  OK pointcloud-only low outlier does not trigger")
    print("  OK pose timeout aborts in descent")
    print("=== ALL PASSED ===")


if __name__ == "__main__":
    main()
