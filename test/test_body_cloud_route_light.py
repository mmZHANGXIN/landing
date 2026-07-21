#!/usr/bin/env python3
"""Dependency-light regression tests for the outdoor body-cloud route."""

import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from perception.halss_preprocess import body_cloud_to_level_body_roi
from pipeline import _limited_axis_velocity
from control.mission_state_manager import MissionState, MissionStateManager, StateInputs


def _cfg():
    return {
        "body_R_from_lidar_imu": [1, 0, 0, 0, 1, 0, 0, 0, 1],
        "body_T_from_lidar_imu": [0, 0, 0],
        "halss_min_down_m": 0.05,
        "halss_max_down_m": 30.0,
    }


def test_body_cloud_does_not_require_world_pose():
    points = np.array([[1.0, 0.0, -10.0], [50.0, 0.0, -10.0]], dtype=np.float32)
    roi, stats = body_cloud_to_level_body_roi(
        points, 0.0, 0.0, _cfg(), half_x=5.0, half_y=5.0,
    )
    assert roi.shape == (1, 3)
    assert np.allclose(roi[0], [1.0, 0.0, 10.0])
    assert stats["roi_type"] == "deskewed_body_level"


def test_nonfinite_body_points_are_rejected_and_reported():
    points = np.array([[0.0, 0.0, -5.0], [np.nan, 0.0, -5.0]], dtype=np.float32)
    roi, stats = body_cloud_to_level_body_roi(points, 0.0, 0.0, _cfg(), 5.0, 5.0)
    assert len(roi) == 1
    assert math.isclose(stats["finite_ratio"], 0.5)


def test_vertical_velocity_is_symmetric_and_capped():
    assert _limited_axis_velocity(10.0, 1.0) == 1.0
    assert _limited_axis_velocity(-10.0, 1.0) == -1.0
    assert math.isclose(_limited_axis_velocity(0.2, 1.0), 0.2)


def test_manual_hold_yields_to_offboard_exit_without_abort():
    mgr = MissionStateManager({"height_axis": "pos_z", "ground_z_ref_m": 0.0})
    mgr.reset(MissionState.HOLD_FOR_MANUAL, "goto_timeout_wait_manual")
    decision = mgr.update(StateInputs(
        now=1.0, pose_xyz=(0.0, 0.0, 30.0),
        offboard_active=False, armed=True,
    ))
    assert decision.state == MissionState.IDLE
    assert decision.reason == "manual_takeover"
    assert not decision.abort

