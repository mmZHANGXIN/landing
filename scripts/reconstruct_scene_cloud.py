#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Orin Landing — 实验场景重建: 去畸变点云 → 统一世界坐标场景地图
================================================================
从实验目录的 input.bag 读取:
  /cloud_registered_body          FAST-LIO 去畸变点云 (sensor_msgs/PointCloud2)
  /mavros/local_position/odom     PX4 ENU 位姿 (nav_msgs/Odometry)
结合实验配置中的 LiDAR-IMU 外参 (body_R_from_lidar_imu / body_T_from_lidar_imu),
将每帧点云变换到统一世界坐标系并体素降采样, 输出完整飞行轨迹覆盖的
场景点云, 同时给出 PLY 和 PCD 两种格式 (CloudCompare / Open3D / PCL / ROS
均可读取).

坐标约定: 输出保持 PX4 的 ENU 世界坐标 (z 向上), 使用完整 roll/pitch/yaw:

  p_world = R_world_body * (R_body_lidar * p_lidar + T_body_lidar) + t_world_body

不做额外水平化, 不翻转 z 轴 (与 replay_* 脚本的 z-down 世界不同).

位姿同步: 先单遍读入全部 odom, 对每帧点云在自身 header 时间戳处做
位置线性插值 + 四元数 SLERP; 无夹住样本时回退时间最近位姿. 插值跨度
或最近样本时间差超过 --max-sync-ms (默认 100 ms) 时跳过该帧并记录原因.

体素累积: 哈希表按体素格累积 (整数索引 → 坐标和/强度/点数), 每个体素
保留质心和平均 intensity, 不在内存保存全部原始点.

默认参数:
  --bag      experiments/20260807_162946_orin_landing/input.bag
  --config   对应目录下的 experiment_config_snapshot.yaml
  --cloud-topic /cloud_registered_body
  --pose-topic  /mavros/local_position/odom
  --voxel-size  0.05 m
  --output-dir  实验目录下的 scene_map/

用法:
  python reconstruct_scene_cloud.py \
    --bag /home/ifsc_orin/evelyn/landing/experiments/runs/20260807_162946_orin_landing/input.bag \
    --config /home/ifsc_orin/evelyn/landing/experiments/runs/20260807_162946_orin_landing/experiment_config_snapshot.yaml \
    [--start-time T0] [--end-time T1] [--voxel-size 0.05]
    [--output-dir /home/ifsc_orin/evelyn/landing/experiments/runs/20260807_162946_orin_landing/scene_map] [--max-sync-ms 100] [--no-intensity]
    [--max-frames N]

仅依赖 rosbag / NumPy / PyYAML, 不要求启动 ROS 节点, 不依赖 Open3D.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from pathlib import Path

import numpy as np
import yaml

logger = logging.getLogger("ReconstructSceneCloud")


# ──────────────────────────────────────────────
# 四元数 / 欧拉角 (纯 NumPy, 与 replay_compare_common.py 同式复制,
# 改动需保持同步; 本脚本自包含, 不跨模块取用)
# ──────────────────────────────────────────────
def quat_to_euler(x, y, z, w):
    """四元数 → 欧拉角 (roll, pitch, yaw) rad."""
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)
    sinp = 2.0 * (w * y - z * x)
    if abs(sinp) < 1:
        pitch = math.asin(max(-1.0, min(1.0, sinp)))
    else:
        pitch = math.copysign(math.pi / 2, sinp)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return roll, pitch, yaw


def quat_slerp(q0, q1, t: float) -> np.ndarray:
    """四元数球面线性插值 (q 为 [x,y,z,w], t ∈ [0,1]), 返回单位四元数.

    点积为负时对 q1 取反走最短弧路径; 夹角过小 (cos > 0.9995) 时线性
    插值后归一化, 避免 SLERP 除零/数值抖动.
    """
    q0 = np.asarray(q0, dtype=np.float64)
    q1 = np.asarray(q1, dtype=np.float64)
    dot = float(np.clip(np.dot(q0, q1), -1.0, 1.0))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    if dot > 0.9995:
        q = q0 + t * (q1 - q0)
    else:
        theta = math.acos(dot)
        sin_theta = math.sin(theta)
        q = (math.sin((1.0 - t) * theta) * q0 + math.sin(t * theta) * q1) / sin_theta
    return q / np.linalg.norm(q)


def _rot_zyx(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Body-to-world rotation Rz(yaw)@Ry(pitch)@Rx(roll), 行向量 p @ R.T 约定.

    与 replay_compare_common.py / perception/halss_preprocess.py 的
    _rot_zyx 同式 (改动需保持同步). 对 PX4 ENU 位姿, 即 R_world_body.
    """
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ], dtype=np.float32)


def _cfg_vec3(cfg: dict, key: str, default) -> np.ndarray:
    arr = np.asarray(cfg.get(key, default), dtype=np.float32)
    if arr.shape != (3,):
        raise ValueError(f"{key} must be a 3-element vector")
    return arr


def _cfg_mat3(cfg: dict, key: str, default) -> np.ndarray:
    arr = np.asarray(cfg.get(key, default), dtype=np.float32)
    if arr.size != 9:
        raise ValueError(f"{key} must contain 9 values")
    return arr.reshape(3, 3)


# ──────────────────────────────────────────────
# 消息解析 (不依赖 rospy / 消息包)
# ──────────────────────────────────────────────
def stamp_to_sec(stamp) -> float:
    return float(stamp.secs) + float(stamp.nsecs) * 1e-9


def parse_pc2(msg):
    """sensor_msgs/PointCloud2 → (pts (N,3) float32, intensity (N,) float32 | None).

    按字段 offset + point_step 解析; NaN/Inf 行滤除, intensity 与 xyz
    同一掩码过滤保持对齐. intensity 字段缺失或为非浮点类型时返回 None
    (该话题输出不含 intensity).
    """
    fields = list(getattr(msg, "fields", []))
    offsets = {f.name: f.offset for f in fields}
    if not all(k in offsets for k in ("x", "y", "z")):
        return np.empty((0, 3), dtype=np.float32), None
    n = int(msg.width) * int(msg.height)
    if n == 0 or msg.point_step < 12 or msg.row_step < msg.point_step:
        return np.empty((0, 3), dtype=np.float32), None
    endian = ">" if msg.is_bigendian else "<"
    names = ["x", "y", "z"]
    formats = [endian + "f4"] * 3
    offs = [offsets[k] for k in names]
    if "intensity" in offsets:
        datatype = next((f.datatype for f in fields if f.name == "intensity"), 7)
        if datatype in (6, 7):  # FLOAT64 / FLOAT32
            names.append("intensity")
            formats.append(endian + ("f8" if datatype == 6 else "f4"))
            offs.append(offsets["intensity"])
    arr = np.frombuffer(msg.data, dtype=np.dtype({
        "names": names, "formats": formats, "offsets": offs,
        "itemsize": msg.point_step,
    }), count=n)
    pts = np.column_stack([arr[k] for k in ("x", "y", "z")]).astype(np.float32, copy=False)
    mask = np.isfinite(pts).all(axis=1)
    pts = pts[mask]
    ints = None
    if "intensity" in names:
        ints = np.asarray(arr["intensity"], dtype=np.float32)[mask]
    return pts, ints


def odom_to_pose6(msg):
    """nav_msgs/Odometry → ([x,y,z,roll,pitch,yaw] f32, 四元数 [x,y,z,w] f64).

    四元数保留原始值供点云时间戳处 SLERP 插值; 非 Odometry 返回 (None, None).
    """
    try:
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
    except AttributeError:
        return None, None
    roll, pitch, yaw = quat_to_euler(q.x, q.y, q.z, q.w)
    pose6 = np.array([p.x, p.y, p.z, roll, pitch, yaw], dtype=np.float32)
    quat = np.array([q.x, q.y, q.z, q.w], dtype=np.float64)
    return pose6, quat


# ──────────────────────────────────────────────
# 位姿插值 (点云时间戳处: 位置线性 + 姿态 SLERP)
# ──────────────────────────────────────────────
def interpolate_pose(stamps, poses, quats, t: float, max_sync_s: float):
    """点云时间戳处位姿.

    返回 (pose6, sync_s, status):
      - status="interp": 夹住样本线性位置 + 四元数 SLERP, sync_s 为到
        两侧样本的最大时间距离;
      - status="nearest": 无夹住样本 (首末段), 取时间最近位姿, sync_s
        为时间差;
      - status="skip_*": 时间差超过 max_sync_s (odom 丢包跨度太大或
        最近样本太远) 或无比位姿, pose6 为 None.
    """
    if len(stamps) == 0:
        return None, 0.0, "skip_no_pose_samples"
    stamps = np.asarray(stamps, dtype=np.float64)
    if t < stamps[0] or t > stamps[-1]:
        nearest = int(np.argmin(np.abs(stamps - t)))
        dt = abs(float(stamps[nearest] - t))
        if dt > float(max_sync_s):
            return None, dt, "skip_nearest_pose_exceed_sync_ms"
        return poses[nearest], dt, "nearest"
    i = int(np.searchsorted(stamps, t, side="right")) - 1
    t0, t1 = float(stamps[i]), float(stamps[i + 1])
    if t1 <= t0:  # 重复时间戳
        return poses[i], 0.0, "nearest"
    span = t1 - t0
    if span > float(max_sync_s):
        return None, span, "skip_bracketing_span_exceed_sync_ms"
    frac = (t - t0) / (t1 - t0)
    p0, p1 = poses[i], poses[i + 1]
    xyz = p0[:3] + frac * (p1[:3] - p0[:3])
    q = quat_slerp(quats[i], quats[i + 1], frac)
    roll, pitch, yaw = quat_to_euler(q[0], q[1], q[2], q[3])
    pose6 = np.array([xyz[0], xyz[1], xyz[2], roll, pitch, yaw], dtype=np.float32)
    return pose6, max(t - t0, t1 - t), "interp"


# ──────────────────────────────────────────────
# 坐标变换
# ──────────────────────────────────────────────
def to_world(pts: np.ndarray, pose6: np.ndarray, r_body_lidar: np.ndarray,
             t_body_lidar: np.ndarray) -> np.ndarray:
    """body 点云 → ENU 世界坐标 (行向量约定 p @ R.T).

    p_world = R_world_body * (R_body_lidar * p_lidar + T_body_lidar) + t_world_body
    完整 roll/pitch/yaw, 不水平化, 不翻转 z 轴 (PX4 ENU, z 向上).
    """
    pts = np.asarray(pts, dtype=np.float32)
    if len(pts) == 0:
        return pts
    r_world_body = _rot_zyx(float(pose6[3]), float(pose6[4]), float(pose6[5]))
    pts_base = pts @ r_body_lidar.T + t_body_lidar
    pts_w = pts_base @ r_world_body.T
    pts_w[:, 0] += float(pose6[0])
    pts_w[:, 1] += float(pose6[1])
    pts_w[:, 2] += float(pose6[2])
    return pts_w.astype(np.float32, copy=False)


# ──────────────────────────────────────────────
# 哈希体素累积 (整数键 → 坐标和/强度/点数)
# ──────────────────────────────────────────────
class VoxelMap:
    """体素格哈希累积: 每帧先 numpy 归并到帧内唯一体素, 再并入全局哈希.

    add() 后内存只随体素数增长, 不保存全部原始点; finalize() 输出每个
    体素的质心与平均 intensity (按体素键排序, 输出确定性).
    """

    _SHIFT = 21          # 每轴位宽
    _MASK = (1 << _SHIFT) - 1
    _LIMIT = 1 << (_SHIFT - 1)  # 坐标范围 ±2^20 个体素 (0.05 m 时约 ±52 km)

    def __init__(self, voxel_size: float, with_intensity: bool = True):
        self.voxel_size = float(voxel_size)
        self.with_intensity = bool(with_intensity)
        self._acc: dict = {}   # int64 键 → [sx, sy, sz, si, count] float64
        self._intensity_seen = False

    @property
    def occupied_voxels(self) -> int:
        return len(self._acc)

    @classmethod
    def voxel_keys(cls, pts: np.ndarray, voxel_size: float) -> np.ndarray:
        """点 → 打包的 int64 体素键 (x | y<<21 | z<<42, 21 位每轴)."""
        v = np.floor(np.asarray(pts, dtype=np.float64) / float(voxel_size)).astype(np.int64)
        if len(v) and (np.abs(v) >= cls._LIMIT).any():
            raise ValueError(
                f"voxel index out of range (±{cls._LIMIT} cells, voxel "
                f"{voxel_size} m): max |v|={np.abs(v).max()}")
        return ((v[:, 0] & cls._MASK)
                | ((v[:, 1] & cls._MASK) << cls._SHIFT)
                | ((v[:, 2] & cls._MASK) << (2 * cls._SHIFT)))

    def add(self, pts: np.ndarray, ints: np.ndarray | None = None) -> int:
        """并入一帧变换后点云; 返回该帧保留的有效点数."""
        pts = np.asarray(pts, dtype=np.float32)
        if len(pts) == 0:
            return 0
        finite = np.isfinite(pts).all(axis=1)
        if not finite.all():
            pts = pts[finite]
            if ints is not None:
                ints = np.asarray(ints, dtype=np.float32)[finite]
        keys = self.voxel_keys(pts, self.voxel_size)
        order = np.argsort(keys, kind="stable")
        ks = keys[order]
        first = np.flatnonzero(np.concatenate(([True], ks[1:] != ks[:-1])))
        cnt = np.diff(np.concatenate((first, [len(ks)]))).astype(np.float64)
        sxyz = np.add.reduceat(pts[order].astype(np.float64), first, axis=0)
        if self.with_intensity and ints is not None:
            self._intensity_seen = True
            si = np.add.reduceat(np.asarray(ints, dtype=np.float64)[order], first)
        else:
            si = np.zeros(len(first), dtype=np.float64)
        acc = self._acc
        new = np.column_stack([sxyz, si, cnt])
        for j, k in enumerate(ks[first]):
            key = int(k)
            cur = acc.get(key)
            if cur is None:
                acc[key] = new[j]
            else:
                cur += new[j]
        return int(len(keys))

    def finalize(self):
        """→ (points (M,3) f32, intensity (M,) f32 | None, counts (M,) int64)."""
        if not self._acc:
            empty = np.empty((0, 3), dtype=np.float32)
            return empty, (np.empty(0, dtype=np.float32)
                           if self.with_intensity and self._intensity_seen
                           else None), np.empty(0, dtype=np.int64)
        items = sorted(self._acc.items())
        arrs = np.stack([a for _, a in items])          # (M, 5) [sx,sy,sz,si,cnt]
        counts = arrs[:, 4].astype(np.int64)
        pts = (arrs[:, :3] / arrs[:, 4:5]).astype(np.float32)
        ints = None
        if self.with_intensity and self._intensity_seen:
            ints = (arrs[:, 3] / arrs[:, 4]).astype(np.float32)
        return pts, ints, counts


# ──────────────────────────────────────────────
# 输出 (ASCII PLY / PCD, CloudCompare 与 PCL 均可读)
# ──────────────────────────────────────────────
def write_ply(path: Path, pts: np.ndarray, ints: np.ndarray | None):
    props = ["x", "y", "z"] + (["intensity"] if ints is not None else [])
    with open(path, "w") as f:
        f.write("ply\nformat ascii 1.0\n"
                "comment reconstructed from input.bag (FAST-LIO + PX4 ENU)\n"
                f"element vertex {len(pts)}\n")
        for p in props:
            f.write(f"property float {p}\n")
        f.write("end_header\n")
        if ints is not None:
            np.savetxt(f, np.column_stack([pts, ints]), fmt="%.6f %.6f %.6f %.4f")
        else:
            np.savetxt(f, pts, fmt="%.6f %.6f %.6f")


def write_pcd(path: Path, pts: np.ndarray, ints: np.ndarray | None):
    fields = ["x", "y", "z"] + (["intensity"] if ints is not None else [])
    n = len(fields)
    with open(path, "w") as f:
        f.write("# .PCD v0.7 - Point Cloud Data file format\n"
                "VERSION 0.7\n"
                f"FIELDS {' '.join(fields)}\n"
                f"SIZE {' '.join(['4'] * n)}\n"
                f"TYPE {' '.join(['F'] * n)}\n"
                f"COUNT {' '.join(['1'] * n)}\n"
                f"WIDTH {len(pts)}\n"
                "HEIGHT 1\n"
                "VIEWPOINT 0 0 0 1 0 0 0\n"
                f"POINTS {len(pts)}\n"
                "DATA ascii\n")
        if ints is not None:
            np.savetxt(f, np.column_stack([pts, ints]), fmt="%.6f %.6f %.6f %.4f")
        else:
            np.savetxt(f, pts, fmt="%.6f %.6f %.6f")


# ──────────────────────────────────────────────
# bag 读取
# ──────────────────────────────────────────────
def read_odom(bag_path: str, pose_topic: str):
    """单遍读入全部 odom → (stamps f64 (M,), poses f32 (M,6), quats f64 (M,4), bad)."""
    import rosbag
    stamps, poses, quats = [], [], []
    bad = 0
    with rosbag.Bag(str(bag_path), "r") as bag:
        for _, msg, _ in bag.read_messages(topics=[pose_topic]):
            pose6, quat = odom_to_pose6(msg)
            if pose6 is None:
                bad += 1
                continue
            stamps.append(stamp_to_sec(msg.header.stamp))
            poses.append(pose6)
            quats.append(quat)
    return (np.asarray(stamps, dtype=np.float64),
            np.asarray(poses, dtype=np.float32),
            np.asarray(quats, dtype=np.float64), bad)


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def setup_logging(level: int = logging.INFO):
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s")


def build_parser() -> argparse.ArgumentParser:
    default_bag = "experiments/20260807_162946_orin_landing/input.bag"
    parser = argparse.ArgumentParser(
        description="离线场景重建: 去畸变点云 + PX4 ENU 位姿 → 统一世界坐标 "
                    "体素场景地图 (PLY + PCD + JSON 统计)")
    parser.add_argument("--bag", type=str, default=default_bag,
                        help=f"输入 rosbag (默认 {default_bag})")
    parser.add_argument("--config", type=str, default=None,
                        help="实验配置快照 yaml (默认 bag 同目录 "
                             "experiment_config_snapshot.yaml)")
    parser.add_argument("--cloud-topic", type=str, default="/cloud_registered_body",
                        help="去畸变点云话题 (默认 /cloud_registered_body)")
    parser.add_argument("--cloud-frame", choices=["body", "world"], default="body",
                        help="点云坐标: body=结合 --pose-topic 和机体外参转到世界系 "
                             "(默认); world=输入已经是 FAST-LIO 世界系，直接体素累积")
    parser.add_argument("--pose-topic", type=str, default="/mavros/local_position/odom",
                        help="位姿话题 (默认 /mavros/local_position/odom)")
    parser.add_argument("--voxel-size", type=float, default=0.05,
                        help="体素尺寸米 (默认 0.05)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="输出目录 (默认 bag 所在目录下的 scene_map/)")
    parser.add_argument("--max-sync-ms", type=float, default=100.0,
                        help="位姿同步时间差上限毫秒 (默认 100): 插值跨度或"
                             "最近样本时间差超过则跳过该帧")
    parser.add_argument("--start-time", type=float, default=None,
                        help="起始点云时间 (秒, header 时间戳裁剪, 默认不限)")
    parser.add_argument("--end-time", type=float, default=None,
                        help="结束点云时间 (秒, header 时间戳裁剪, 默认不限)")
    parser.add_argument("--no-intensity", action="store_true",
                        help="不解析/不输出 intensity 字段")
    parser.add_argument("--max-frames", type=int, default=0,
                        help="最多处理帧数 (冒烟测试用, 默认不限)")
    return parser


def main():
    args = build_parser().parse_args()
    setup_logging()

    bag_path = Path(args.bag)
    if not bag_path.is_file():
        sys.exit(f"[Reconstruct] bag not found: {bag_path}")
    config_path = Path(args.config) if args.config else bag_path.parent / "experiment_config_snapshot.yaml"
    if args.cloud_frame == "body" and not config_path.is_file():
        sys.exit(f"[Reconstruct] config not found: {config_path}")
    out_dir = Path(args.output_dir) if args.output_dir else bag_path.parent / "scene_map"
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.start_time is not None and args.end_time is not None \
            and args.start_time > args.end_time:
        sys.exit("[Reconstruct] --start-time must be <= --end-time")
    if args.voxel_size <= 0:
        sys.exit("[Reconstruct] --voxel-size must be > 0")
    max_sync_s = args.max_sync_ms / 1000.0

    if args.cloud_frame == "body":
        cfg = load_config(str(config_path))
        perc = cfg.get("perception", {})
        r_bl = _cfg_mat3(perc, "body_R_from_lidar_imu",
                         [1, 0, 0, 0, 1, 0, 0, 0, 1])
        t_bl = _cfg_vec3(perc, "body_T_from_lidar_imu", [0, 0, 0])
    else:
        r_bl = np.eye(3, dtype=np.float32)
        t_bl = np.zeros(3, dtype=np.float32)

    import rosbag
    info = rosbag.Bag(str(bag_path)).get_type_and_topic_info()
    available = set(info.topics.keys())
    required_topics = [(args.cloud_topic, "cloud")]
    if args.cloud_frame == "body":
        required_topics.append((args.pose_topic, "pose"))
    for topic, need in required_topics:
        if topic not in available:
            sys.exit(f"[Reconstruct] {need} topic {topic} not in bag. "
                     f"Available: {', '.join(sorted(available))}")
    if args.cloud_frame == "body":
        logger.info("[Bag] cloud=%s (%d msgs) pose=%s (%d msgs)",
                    args.cloud_topic, info.topics[args.cloud_topic].message_count,
                    args.pose_topic, info.topics[args.pose_topic].message_count)
    else:
        logger.info("[Bag] world cloud=%s (%d msgs), no external pose applied",
                    args.cloud_topic, info.topics[args.cloud_topic].message_count)

    # ── pass 1: 全部 odom ──
    if args.cloud_frame == "body":
        odom_stamps, odom_poses, odom_quats, odom_bad = read_odom(
            str(bag_path), args.pose_topic)
        if len(odom_stamps) == 0:
            sys.exit("[Reconstruct] no valid odometry samples in bag")
        logger.info("[Odom] %d samples, %d unparseable, t=[%.3f, %.3f]",
                    len(odom_stamps), odom_bad, odom_stamps[0], odom_stamps[-1])
    else:
        odom_stamps = odom_poses = odom_quats = np.empty(0)

    # ── pass 2: 逐帧点云 → 世界坐标 → 体素累积 ──
    voxel = VoxelMap(args.voxel_size, with_intensity=not args.no_intensity)
    skip_counts = {}
    sync_ms_list = []
    time_start = time_end = None
    n_input = 0
    n_frames = 0
    n_processed = 0
    input_frame_id = None
    t0 = time.perf_counter()

    def skip(reason: str, count: int = 1):
        skip_counts[reason] = skip_counts.get(reason, 0) + count

    with rosbag.Bag(str(bag_path), "r") as bag:
        for _, msg, _ in bag.read_messages(topics=[args.cloud_topic]):
            stamp = stamp_to_sec(msg.header.stamp)
            if args.start_time is not None and stamp < args.start_time:
                skip("outside_time_window")
                continue
            if args.end_time is not None and stamp > args.end_time:
                skip("outside_time_window")
                continue
            n_frames += 1
            if time_start is None:
                time_start = stamp
            time_end = stamp
            pts, ints = parse_pc2(msg)
            if len(pts) == 0:
                skip("empty_cloud")
                continue
            if input_frame_id is None:
                input_frame_id = str(getattr(msg.header, "frame_id", ""))
            if args.cloud_frame == "body":
                pose6, sync_s, status = interpolate_pose(
                    odom_stamps, odom_poses, odom_quats, stamp, max_sync_s)
                if status.startswith("skip"):
                    skip(status)
                    continue
                sync_ms_list.append(sync_s * 1000.0)
                pts_w = to_world(pts, pose6, r_bl, t_bl)
            else:
                sync_s = 0.0
                pts_w = pts
            n_input += voxel.add(pts_w, ints)
            n_processed += 1
            if args.max_frames > 0 and n_processed >= args.max_frames:
                logger.info("[Replay] Max frames reached: %d", n_processed)
                break
            if n_processed % 50 == 0:
                logger.info("[%05d] pts=%d voxels=%d sync=%.1f ms",
                            n_processed, n_input, voxel.occupied_voxels,
                            sync_s * 1000.0)
    elapsed = time.perf_counter() - t0

    # ── finalize + 输出 ──
    points, intensity, counts = voxel.finalize()
    logger.info("[Done] frames=%d processed=%d skipped=%d pts_in=%d "
                "voxels=%d pts_out=%d (%.1f s)",
                n_frames, n_processed, sum(skip_counts.values()), n_input,
                voxel.occupied_voxels, len(points), elapsed)

    out_ply = out_dir / "scene_map.ply"
    out_pcd = out_dir / "scene_map.pcd"
    out_json = out_dir / "scene_map_stats.json"
    if len(points):
        write_ply(out_ply, points, intensity)
        write_pcd(out_pcd, points, intensity)
    else:
        logger.warning("[Output] 无有效点, 不写 PLY/PCD")
        out_ply.write_text("ply\nformat ascii 1.0\nelement vertex 0\n"
                           "property float x\nproperty float y\n"
                           "property float z\nend_header\n")
        out_pcd.write_text("# .PCD v0.7\nVERSION 0.7\nFIELDS x y z\n"
                           "SIZE 4 4 4\nTYPE F F F\nCOUNT 1 1 1\n"
                           "WIDTH 0\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\n"
                           "POINTS 0\nDATA ascii\n")

    sync_ms = np.asarray(sync_ms_list, dtype=np.float64)
    bounds = None
    bounds_robust = None
    if len(points):
        p_min = points.min(axis=0)
        p_max = points.max(axis=0)
        bounds = {"min_m": [float(v) for v in p_min],
                  "max_m": [float(v) for v in p_max],
                  "size_m": [float(v) for v in (p_max - p_min)]}
        # 稳健边界: 源数据偶发坏点会拉满 raw bounds, 分位数边界供验收参考
        lo = np.percentile(points, 0.01, axis=0)
        hi = np.percentile(points, 99.99, axis=0)
        bounds_robust = {"min_m": [float(v) for v in lo],
                         "max_m": [float(v) for v in hi],
                         "size_m": [float(v) for v in (hi - lo)]}
    stats = {
        "bag": str(bag_path),
        "config": str(config_path),
        "cloud_topic": args.cloud_topic,
        "cloud_frame": args.cloud_frame,
        "input_frame_id": input_frame_id,
        "pose_topic": (args.pose_topic if args.cloud_frame == "body" else None),
        "voxel_size_m": args.voxel_size,
        "extrinsics": {
            "body_R_from_lidar_imu": r_bl.reshape(-1).tolist(),
            "body_T_from_lidar_imu": t_bl.tolist(),
        },
        "coordinate_frame": ("PX4 ENU (z-up, full roll/pitch/yaw, no flattening)"
                             if args.cloud_frame == "body"
                             else "input FAST-LIO world frame (no external transform)"),
        "frames": {
            "total": n_frames,
            "processed": n_processed,
            "skipped": sum(skip_counts.values()),
            "skip_reasons": {k: v for k, v in sorted(skip_counts.items())},
        },
        "time_range_s": ({"start": float(time_start), "end": float(time_end)}
                         if time_start is not None else None),
        "pose_sync_error_ms": {
            "mean": float(sync_ms.mean()) if len(sync_ms) else None,
            "median": float(np.median(sync_ms)) if len(sync_ms) else None,
            "p95": float(np.percentile(sync_ms, 95)) if len(sync_ms) else None,
            "max": float(sync_ms.max()) if len(sync_ms) else None,
            "max_sync_ms": args.max_sync_ms,
        },
        "point_cloud": {
            "input_points": int(n_input),
            "output_points": int(len(points)),
            "occupied_voxels": voxel.occupied_voxels,
            "bounds_m": bounds,
            "bounds_m_robust_p001_p9999": bounds_robust,
        },
        "intensity": ("mean_per_voxel" if intensity is not None
                      else "disabled_or_missing"),
        "elapsed_s": elapsed,
    }
    with open(out_json, "w") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    logger.info("[Output] ply=%s pcd=%s stats=%s (%d points)",
                out_ply, out_pcd, out_json, len(points))


if __name__ == "__main__":
    main()
