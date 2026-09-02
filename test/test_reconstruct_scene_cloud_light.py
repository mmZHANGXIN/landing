#!/usr/bin/env python3
"""Dependency-light regression tests for reconstruct_scene_cloud.py.

纯 NumPy 测试 (不依赖 rosbag): 四元数 SLERP、刚体变换、体素索引/累积、
空输入、PointCloud2 字段解析 (合成消息)、位姿时间插值.
"""

import math
import sys
import types
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from reconstruct_scene_cloud import (  # noqa: E402
    VoxelMap,
    interpolate_pose,
    parse_pc2,
    quat_slerp,
    quat_to_euler,
    to_world,
)

_IDENTITY = np.eye(3, dtype=np.float32)
_ZERO = np.zeros(3, dtype=np.float32)


def _q(x, y, z, w):
    q = np.array([x, y, z, w], dtype=np.float64)
    return q / np.linalg.norm(q)


# ── 四元数 SLERP ──
def test_slerp_endpoints_and_midpoint():
    q0 = _q(0, 0, math.sin(math.radians(45)), math.cos(math.radians(45)))
    q1 = _q(0, 0, math.sin(math.radians(-45)), math.cos(math.radians(-45)))
    assert np.allclose(quat_slerp(q0, q1, 0.0), q0)
    assert np.allclose(quat_slerp(q0, q1, 1.0), q1)
    mid = quat_slerp(q0, q1, 0.5)
    # 90° 总转角的中点应为 0 转 (绕 z 的 +45° 与 -45° 中点)
    assert abs(mid[2]) < 1e-6 and mid[3] > 0.99
    assert math.isclose(np.linalg.norm(mid), 1.0, rel_tol=1e-9)


def test_slerp_shortest_path_and_small_angle():
    # 点积为负: 取反走最短弧, 中点取正 w 侧 (不绕远)
    q0 = _q(0, 0, 0, 1)
    q1 = np.array([0.0, 0.0, -math.sqrt(0.5), -math.sqrt(0.5)])  # -90° 绕 z (负 w)
    mid = quat_slerp(q0, q1, 0.5)
    assert mid[2] > 0.3 and mid[3] > 0.9  # 短弧中点 ≈ 45° 绕 z
    assert math.isclose(np.linalg.norm(mid), 1.0, rel_tol=1e-9)
    # 小夹角: 线性插值归一化, 无 NaN
    a = _q(0, 0, 0, 1)
    b = _q(0, 0, 1e-3, math.sqrt(1 - 1e-6))
    out = quat_slerp(a, b, 0.5)
    assert np.isfinite(out).all()
    assert math.isclose(np.linalg.norm(out), 1.0, rel_tol=1e-9)


def test_quat_to_euler_roundtrip():
    r, p, y = quat_to_euler(0, 0, 0, 1)
    assert math.isclose(r, 0.0, abs_tol=1e-9)
    assert math.isclose(p, 0.0, abs_tol=1e-9)
    assert math.isclose(y, 0.0, abs_tol=1e-9)
    # 绕 z 90° → yaw = π/2
    q = _q(0, 0, math.sin(math.pi / 4), math.cos(math.pi / 4))
    r, p, y = quat_to_euler(q[0], q[1], q[2], q[3])
    assert math.isclose(y, math.pi / 2, abs_tol=1e-6)
    assert abs(r) < 1e-9 and abs(p) < 1e-9


# ── 刚体变换 (ENU 世界, 完整 roll/pitch/yaw) ──
def test_world_transform_identity():
    pts = np.array([[1.0, 2.0, -5.0], [0.5, 0.0, -10.0]], dtype=np.float32)
    out = to_world(pts, np.zeros(6, dtype=np.float32), _IDENTITY, _ZERO)
    assert np.allclose(out, pts)


def test_world_transform_extrinsic_translation():
    pts = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
    t_bl = np.array([0.13, 0.0, 0.08], dtype=np.float32)
    # 零位姿: 仅外参平移生效
    out = to_world(pts, np.zeros(6, dtype=np.float32), _IDENTITY, t_bl)
    assert np.allclose(out[0], [1.13, 0.0, 0.08])
    # 位姿平移: 世界坐标再加 [px, py, pz]
    pose = np.array([10.0, -5.0, 3.0, 0.0, 0.0, 0.0], dtype=np.float32)
    out = to_world(pts, pose, _IDENTITY, t_bl)
    assert np.allclose(out[0], [11.13, -5.0, 3.08])


def test_world_transform_full_rpy_no_flattening():
    # roll=90°: 机体系 -z (朝下) 点应转到世界 +y — 完整 roll 生效, 不水平化
    pts = np.array([[0.0, 0.0, -1.0]], dtype=np.float32)
    pose = np.array([0, 0, 0, math.pi / 2, 0, 0], dtype=np.float32)
    out = to_world(pts, pose, _IDENTITY, _ZERO)
    assert np.allclose(out[0], [0.0, 1.0, 0.0], atol=1e-5)
    # yaw=90°: +x → +y (ENU z-up, z 不变)
    pts = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
    pose = np.array([0, 0, 0, 0, 0, math.pi / 2], dtype=np.float32)
    out = to_world(pts, pose, _IDENTITY, _ZERO)
    assert np.allclose(out[0], [0.0, 1.0, 0.0], atol=1e-5)
    # 位姿高度 z>0 直接上移: ENU z 向上 (不翻转)
    pts = np.array([[0.0, 0.0, -5.0]], dtype=np.float32)
    pose = np.array([0, 0, 20.0, 0, 0, 0], dtype=np.float32)
    out = to_world(pts, pose, _IDENTITY, _ZERO)
    assert math.isclose(out[0, 2], 15.0, abs_tol=1e-5)


def test_world_transform_matches_direct_matrix_formula():
    """p_world = R_world_body @ (R_body_lidar @ p + T_body_lidar) + t"""
    pts = np.array([[0.1, -0.2, -3.0], [1.5, 2.0, -1.0]], dtype=np.float32)
    r_bl = np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
                    dtype=np.float32)
    t_bl = np.array([0.13, 0.0, 0.08], dtype=np.float32)
    pose = np.array([12.0, -3.0, 2.0, 0.2, -0.1, 0.5], dtype=np.float32)
    out = to_world(pts, pose, r_bl, t_bl)
    # 直接构造列向量公式
    cr, sr = math.cos(0.2), math.sin(0.2)
    cp, sp = math.cos(-0.1), math.sin(-0.1)
    cy, sy = math.cos(0.5), math.sin(0.5)
    r_wb = np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ])
    t_wb = pose[:3]
    expect = np.array([r_wb @ (r_bl @ p + t_bl) + t_wb for p in pts])
    assert np.allclose(out, expect, atol=1e-4)


def test_world_transform_empty_input():
    out = to_world(np.empty((0, 3), dtype=np.float32),
                   np.zeros(6, dtype=np.float32), _IDENTITY, _ZERO)
    assert out.shape == (0, 3)


# ── 体素索引 / 累积 ──
def test_voxel_map_centroid_and_intensity_mean():
    vm = VoxelMap(0.5)
    # 同一体素 3 点 (全为正象限, 不跨边界) → 质心; 强度均值
    pts = np.array([[0.0, 0.0, 0.0], [0.3, 0.1, 0.2], [0.2, 0.3, 0.1]],
                   dtype=np.float32)
    ints = np.array([10.0, 20.0, 30.0], dtype=np.float32)
    vm.add(pts, ints)
    # 另一个体素 (x 偏移 2.0)
    pts2 = np.array([[2.1, 0.0, 0.0], [2.3, 0.2, 0.1]], dtype=np.float32)
    vm.add(pts2, np.array([5.0, 7.0], dtype=np.float32))
    out, intensity, counts = vm.finalize()
    assert len(out) == 2
    assert np.allclose(counts, [3, 2])
    assert np.allclose(intensity, [20.0, 6.0], atol=1e-4)
    c0 = out[np.argmin(out[:, 0])]
    assert np.allclose(c0, [0.5 / 3.0, 0.4 / 3.0, 0.3 / 3.0], atol=1e-5)


def test_voxel_keys_boundary_and_packing():
    keys = VoxelMap.voxel_keys(np.array([[0.49, -0.01, 5.0]], dtype=np.float32), 0.5)
    # floor(0.49/0.5)=0, floor(-0.01/0.5)=-1, floor(5.0/0.5)=10
    k = int(keys[0])
    assert (k & VoxelMap._MASK) == 0
    assert ((k >> VoxelMap._SHIFT) & VoxelMap._MASK) == VoxelMap._MASK  # -1 的 21 位补码
    assert ((k >> (2 * VoxelMap._SHIFT)) & VoxelMap._MASK) == 10
    with np.testing.assert_raises(ValueError):
        VoxelMap.voxel_keys(np.array([[1e12, 0.0, 0.0]], dtype=np.float32), 0.5)


def test_voxel_map_empty_and_no_intensity():
    vm = VoxelMap(0.1)
    assert vm.add(np.empty((0, 3), dtype=np.float32)) == 0
    pts, intensity, counts = vm.finalize()
    assert len(pts) == 0 and len(counts) == 0
    assert intensity is None
    # with_intensity=False: 不输出 intensity
    vm2 = VoxelMap(0.1, with_intensity=False)
    vm2.add(np.array([[1.0, 1.0, 1.0]], dtype=np.float32),
            np.array([7.0], dtype=np.float32))
    _, ints2, _ = vm2.finalize()
    assert ints2 is None


# ── PointCloud2 字段解析 (合成消息) ──
def _synthetic_pc2():
    """3 点, point_step=20 (x@0 y@4 z@8 intensity@12, 尾部 4 字节填充)."""
    n = 3
    data = bytearray(n * 20)
    pts = np.array([[1.0, 2.0, 3.0], [np.nan, 0.0, 0.0], [4.0, 5.0, 6.0]],
                   dtype=np.float32)
    ints = np.array([10.0, 20.0, 30.0], dtype=np.float32)
    for i in range(n):
        off = i * 20
        data[off:off + 12] = pts[i].tobytes()
        data[off + 12:off + 16] = ints[i].tobytes()
    return types.SimpleNamespace(
        fields=[types.SimpleNamespace(name=k, offset=o, datatype=7)
                for k, o in (("x", 0), ("y", 4), ("z", 8), ("intensity", 12))],
        width=n, height=1, point_step=20, row_step=20, is_bigendian=False,
        data=bytes(data))


def test_parse_pc2_offsets_and_nan_filter():
    msg = _synthetic_pc2()
    pts, ints = parse_pc2(msg)
    assert pts.shape == (2, 3)
    assert np.allclose(pts[0], [1.0, 2.0, 3.0])
    assert np.allclose(pts[1], [4.0, 5.0, 6.0])
    # NaN 行滤除后 intensity 与 xyz 保持对齐
    assert np.allclose(ints, [10.0, 30.0])


def test_parse_pc2_missing_intensity_and_bad_fields():
    msg = _synthetic_pc2()
    msg.fields = msg.fields[:3]  # 去掉 intensity
    pts, ints = parse_pc2(msg)
    assert ints is None
    assert pts.shape[0] == 2
    msg2 = types.SimpleNamespace(fields=[], width=1, height=1, point_step=12,
                                 row_step=12, is_bigendian=False,
                                 data=b"\x00" * 12)
    pts2, ints2 = parse_pc2(msg2)
    assert len(pts2) == 0 and ints2 is None


# ── 位姿时间插值 ──
def _odom_fixture():
    stamps = np.array([10.0, 10.05, 10.10, 10.15], dtype=np.float64)
    poses = np.array([[0, 0, 1, 0, 0, 0],
                      [1, 0, 1, 0, 0, 0],
                      [2, 0, 1, 0, 0, 0],
                      [3, 0, 1, 0, 0, 0]], dtype=np.float32)
    quats = np.array([[0, 0, 0, 1]] * 4, dtype=np.float64)
    return stamps, poses, quats


def test_interpolate_pose_bracketing():
    stamps, poses, quats = _odom_fixture()
    pose6, sync_s, status = interpolate_pose(stamps, poses, quats, 10.075, 0.1)
    assert status == "interp"
    assert np.allclose(pose6[:2], [1.5, 0.0], atol=1e-5)
    assert math.isclose(sync_s, 0.025)


def test_interpolate_pose_nearest_and_skip():
    stamps, poses, quats = _odom_fixture()
    # 末端无夹住样本: 取最近位姿 (t=10.25, 最近 10.15, 差 0.10 ≤ 0.1)
    pose6, sync_s, status = interpolate_pose(stamps, poses, quats, 10.25, 0.1)
    assert status == "nearest"
    assert pose6[0] == 3.0 and math.isclose(sync_s, 0.10)
    # 最近样本超限 → 跳过
    pose6, _, status = interpolate_pose(stamps, poses, quats, 10.30, 0.1)
    assert pose6 is None and status == "skip_nearest_pose_exceed_sync_ms"
    # 夹住但跨度超限 (odom 丢包) → 跳过
    pose6, _, status = interpolate_pose(stamps, poses, quats, 10.075, 0.02)
    assert pose6 is None and status == "skip_bracketing_span_exceed_sync_ms"
    # 无比位姿 → 跳过
    pose6, _, status = interpolate_pose(np.empty(0), poses, quats, 10.0, 0.1)
    assert pose6 is None and status == "skip_no_pose_samples"


def _run_all():
    tests = [(k, v) for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS {name}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"FAIL {name}: {exc}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(_run_all())
