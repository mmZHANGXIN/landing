#!/usr/bin/env python3
"""Lightweight down-looking depth projection acceptance without numpy/cv2."""

import math

W = 128
H = 128
DMAX = 30.0
FX = 64.0
FY = 64.0
CX = 63.5
CY = 63.5

# Body I: x=forward, y=right, z=down. Camera C: X=right, Y=forward, Z=down.
R_I_TO_C = (
    (0.0, 1.0, 0.0),
    (1.0, 0.0, 0.0),
    (0.0, 0.0, 1.0),
)


def _mat_vec_mul(mat, vec):
    return tuple(sum(mat[r][c] * vec[c] for c in range(3)) for r in range(3))


def _rot_zyx(roll, pitch, yaw):
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return (
        (cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr),
        (sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr),
        (-sp, cp * sr, cp * cr),
    )


def _world_to_body(point_g, pose):
    px, py, pz, roll, pitch, yaw = pose
    dx = point_g[0] - px
    dy = point_g[1] - py
    dz = point_g[2] - pz
    r_gi = _rot_zyx(roll, pitch, yaw)
    # DepthProjector uses row-vector multiplication by R_gi, equivalent to R_gi^T * delta.
    return (
        dx * r_gi[0][0] + dy * r_gi[1][0] + dz * r_gi[2][0],
        dx * r_gi[0][1] + dy * r_gi[1][1] + dz * r_gi[2][1],
        dx * r_gi[0][2] + dy * r_gi[1][2] + dz * r_gi[2][2],
    )


def project(points_world, pose):
    depth = [[DMAX for _ in range(W)] for _ in range(H)]
    for point in points_world:
        point_i = _world_to_body(point, pose)
        xc, yc, zc = _mat_vec_mul(R_I_TO_C, point_i)
        if not (0.01 < zc < DMAX):
            continue
        u = int(math.floor(FX * xc / zc + CX))
        v = int(math.floor(FY * yc / zc + CY))
        if 0 <= u < W and 0 <= v < H and zc < depth[v][u]:
            depth[v][u] = zc
    return depth


def _valid_pixels(depth):
    return sum(1 for row in depth for value in row if value < DMAX)


def test_center_depth():
    points = [
        (0.0, 0.0, 5.0),
        (1.0, 0.0, 5.0),
        (0.0, 1.0, 5.0),
    ]
    depth = project(points, (0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
    assert depth[63][63] == 5.0, depth[63][63]
    assert _valid_pixels(depth) == 3, _valid_pixels(depth)


def test_single_point_is_not_dropped():
    depth = project([(0.0, 0.0, 5.0)], (0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
    assert depth[63][63] == 5.0, depth[63][63]
    assert _valid_pixels(depth) == 1, _valid_pixels(depth)


def test_body_to_camera_axis_mapping():
    points = [
        (0.0, 0.0, 5.0),  # body down -> optical axis
        (1.0, 0.0, 5.0),  # body forward -> camera +Y -> image row down
        (0.0, 1.0, 5.0),  # body right -> camera +X -> image col right
    ]
    depth = project(points, (0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
    assert depth[63][63] == 5.0, depth[63][63]
    assert depth[76][63] == 5.0, depth[76][63]
    assert depth[63][76] == 5.0, depth[63][76]


def test_pose_translation_uses_inverse_transform():
    pose = (10.0, 20.0, 3.0, 0.0, 0.0, 0.0)
    points = [
        (10.0, 20.0, 8.0),
        (10.0, 21.0, 8.0),
    ]
    depth = project(points, pose)
    assert depth[63][63] == 5.0, depth[63][63]
    assert depth[63][76] == 5.0, depth[63][76]


def test_z_buffer_keeps_nearest():
    points = [
        (0.0, 0.0, 8.0),
        (0.0, 0.0, 5.0),
        (1.0, 0.0, 5.0),
    ]
    depth = project(points, (0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
    assert depth[63][63] == 5.0, depth[63][63]


def test_invalid_points_remain_dmax():
    points = [
        (0.0, 0.0, -1.0),   # behind the down-looking camera
        (100.0, 0.0, 5.0),  # outside image bounds
        (0.0, 0.0, DMAX),   # clipped at max range
    ]
    depth = project(points, (0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
    assert _valid_pixels(depth) == 0, _valid_pixels(depth)


def test_yaw_transform_keeps_body_forward_semantics():
    yaw = math.pi / 2.0
    points = [
        (0.0, 0.0, 5.0),
        (0.0, 1.0, 5.0),
        (1.0, 0.0, 5.0),
    ]
    depth = project(points, (0.0, 0.0, 0.0, 0.0, 0.0, yaw))
    assert depth[63][63] == 5.0, depth[63][63]
    assert depth[76][63] < DMAX, depth[76][63]


def main():
    test_center_depth()
    test_single_point_is_not_dropped()
    test_body_to_camera_axis_mapping()
    test_pose_translation_uses_inverse_transform()
    test_z_buffer_keeps_nearest()
    test_invalid_points_remain_dmax()
    test_yaw_transform_keeps_body_forward_semantics()
    print("=== Lightweight depth projection acceptance ===")
    print("  OK center pixel depth=5m")
    print("  OK single valid point is projected")
    print("  OK body axes map through R_I_to_C")
    print("  OK pose translation uses inverse transform")
    print("  OK z-buffer keeps nearest point")
    print("  OK invalid/behind/out-of-frame points stay at dmax")
    print("  OK yaw=90deg maps world east to body forward camera row")
    print("=== ALL PASSED ===")


if __name__ == "__main__":
    main()
