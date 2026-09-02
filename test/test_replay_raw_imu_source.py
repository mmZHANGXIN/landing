#!/usr/bin/env python3
"""Dependency-light tests for the raw Livox + IMU replay source."""

from types import SimpleNamespace
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from replay_compare_common import (  # noqa: E402
    BagFrameSource,
    custom_msg_to_raw_arrays,
)


def _point(x, y, z, offset_ns=0, tag=0, reflectivity=1):
    return SimpleNamespace(x=x, y=y, z=z, offset_time=offset_ns,
                           tag=tag, reflectivity=reflectivity)


def _msg(points, stamp=0.0):
    secs = int(stamp)
    nsecs = int(round((stamp - secs) * 1e9))
    return SimpleNamespace(
        point_num=len(points), points=points,
        header=SimpleNamespace(
            seq=7, stamp=SimpleNamespace(secs=secs, nsecs=nsecs)))


def test_raw_arrays_filter_zero_fill_and_keep_offsets():
    msg = _msg([
        _point(1.0, 2.0, 3.0, offset_ns=100, reflectivity=9),
        _point(0.0, 0.0, 0.0, offset_ns=200),
        _point(2.0, 2.0, 2.0, offset_ns=300, tag=0x20),
        _point(float("nan"), 1.0, 1.0, offset_ns=400),
    ])
    xyz, intensity, offsets = custom_msg_to_raw_arrays(msg)
    assert xyz.shape == (1, 3)
    assert np.allclose(xyz[0], [1.0, 2.0, 3.0])
    assert intensity.tolist() == [9.0]
    assert np.allclose(offsets, [1e-7])


def test_exp_so3_rotates_z_axis():
    rot = BagFrameSource._exp_so3(np.array([0.0, 0.0, np.pi / 2]))
    assert np.allclose(rot @ np.array([1.0, 0.0, 0.0]), [0.0, 1.0, 0.0], atol=1e-7)


def test_deskew_uses_scan_end_imu_rotation_and_keeps_imu_frame():
    source = object.__new__(BagFrameSource)
    source._max_sync_ms = 0.2 * 1000.0  # seconds are converted below by method
    source._imu_t = np.array([0.0, 1.0], dtype=np.float64)
    source._imu_rot = np.stack([
        np.eye(3),
        BagFrameSource._exp_so3(np.array([0.0, 0.0, np.pi / 2])),
    ])
    source._pose_t = np.array([0.0, 1.0], dtype=np.float64)
    source._pose_xyz = np.zeros((2, 3), dtype=np.float64)
    source._pose6 = np.zeros((2, 6), dtype=np.float32)
    source._pose_quat = np.array([[0.0, 0.0, 0.0, 1.0]] * 2)
    source._cfg_body_rotation = np.eye(3)
    source._log = SimpleNamespace(warning=lambda *args, **kwargs: None)

    result = source._deskew_raw(_msg([
        _point(1.0, 0.0, 0.0, offset_ns=0),
        _point(1.0, 0.0, 0.0, offset_ns=1_000_000_000),
    ]))
    assert result is not None
    deskewed = result[0]
    # R(ref)^T R(point) maps the point from t=0 into the scan-end frame.
    assert np.allclose(deskewed[0], [0.0, -1.0, 0.0], atol=1e-6)
    assert result[3] == 1.0  # output timestamp is scan end, not scan start
