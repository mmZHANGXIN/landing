#!/usr/bin/env python3
"""DepthProjector geometry tests for the down-looking projection path."""

import numpy as np

from perception.depth_projection import DepthProjector


def _projector():
    return DepthProjector(
        img_width=128,
        img_height=128,
        max_range=30.0,
        mode="perspective",
        fx=64.0,
        fy=64.0,
        cx=63.5,
        cy=63.5,
    )


def test_center_depth():
    projector = _projector()
    points = np.array([
        [0.0, 0.0, 5.0],
        [1.0, 0.0, 5.0],
        [0.0, 1.0, 5.0],
    ], dtype=np.float32)
    pose = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)

    depth = projector.project(points, pose)

    assert depth.shape == (128, 128)
    assert np.isclose(depth[63, 63], 5.0), depth[63, 63]
    assert int(np.sum(depth < 30.0)) == 3


def test_single_point_is_not_dropped():
    projector = _projector()
    points = np.array([[0.0, 0.0, 5.0]], dtype=np.float32)
    pose = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)

    depth = projector.project(points, pose)

    assert np.isclose(depth[63, 63], 5.0), depth[63, 63]
    assert int(np.sum(depth < 30.0)) == 1


def test_body_to_camera_axis_mapping():
    projector = _projector()
    points = np.array([
        [0.0, 0.0, 5.0],  # body down -> optical axis
        [1.0, 0.0, 5.0],  # body forward -> camera +Y -> image row down
        [0.0, 1.0, 5.0],  # body right -> camera +X -> image col right
    ], dtype=np.float32)
    pose = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)

    depth = projector.project(points, pose)

    assert np.isclose(depth[63, 63], 5.0), depth[63, 63]
    assert np.isclose(depth[76, 63], 5.0), depth[76, 63]
    assert np.isclose(depth[63, 76], 5.0), depth[63, 76]


def test_pose_translation_uses_inverse_transform():
    projector = _projector()
    points = np.array([
        [10.0, 20.0, 8.0],
        [10.0, 21.0, 8.0],
    ], dtype=np.float32)
    pose = np.array([10.0, 20.0, 3.0, 0.0, 0.0, 0.0], dtype=np.float32)

    depth = projector.project(points, pose)

    assert np.isclose(depth[63, 63], 5.0), depth[63, 63]
    assert np.isclose(depth[63, 76], 5.0), depth[63, 76]


def test_z_buffer_keeps_nearest():
    projector = _projector()
    points = np.array([
        [0.0, 0.0, 8.0],
        [0.0, 0.0, 5.0],
        [1.0, 0.0, 5.0],
    ], dtype=np.float32)
    pose = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)

    depth = projector.project(points, pose)

    assert np.isclose(depth[63, 63], 5.0), depth[63, 63]


def test_invalid_points_remain_dmax():
    projector = _projector()
    points = np.array([
        [0.0, 0.0, -1.0],
        [100.0, 0.0, 5.0],
        [0.0, 0.0, 30.0],
    ], dtype=np.float32)
    pose = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)

    depth = projector.project(points, pose)

    assert int(np.sum(depth < 30.0)) == 0


def test_yaw_transform_keeps_body_forward_semantics():
    projector = _projector()
    yaw = np.pi / 2.0
    pose = np.array([0.0, 0.0, 0.0, 0.0, 0.0, yaw], dtype=np.float32)
    # World east is body forward when yaw=90deg, so it projects on camera +Y.
    points = np.array([
        [0.0, 0.0, 5.0],
        [0.0, 1.0, 5.0],
        [1.0, 0.0, 5.0],
    ], dtype=np.float32)

    depth = projector.project(points, pose)

    assert np.isclose(depth[63, 63], 5.0), depth[63, 63]
    assert depth[76, 63] < 30.0


if __name__ == "__main__":
    test_center_depth()
    test_single_point_is_not_dropped()
    test_body_to_camera_axis_mapping()
    test_pose_translation_uses_inverse_transform()
    test_z_buffer_keeps_nearest()
    test_invalid_points_remain_dmax()
    test_yaw_transform_keeps_body_forward_semantics()
    print("DepthProjector tests passed")
