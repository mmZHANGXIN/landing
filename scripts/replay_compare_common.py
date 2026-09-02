#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Orin Landing — 三种离线对比脚本公共模块
=========================================
供 replay_bev_bayesian.py / replay_bev_geometry.py / replay_window10.py 共用:

  - 流式读取 rosbag: 默认输入 /cloud_registered_body (PointCloud2),
    或选择 /livox/lidar + /livox/imu 做原始扫描逐点去畸变; 位姿话题默认按
    /mavros/local_position/odom → /ali_odom → /Odometry 自动选择.
  - cloud/pose 时间同步: 滑窗最近时间 + max_cloud_odom_sync_ms 上限.
  - BEV 粗糙度保持降采样 (128 网格, 每单元保留 z-down 最大点/最小点/高度差/点数).
  - 几何语义分支 (坡度 + 粗糙度) 与 Bayesian HALSS 语义分支 (惰性导入, 无 torch 环境降级).
  - training-camera 深度投影 + NN-fill, ONNX DRL 推理, 三窗口可视化, 每帧保存.

坐标约定 (与 perception/halss_preprocess.py 完全一致):
  level-body: x=forward, y=lateral, z=down (z-down 正值向下).

cv2 / torch / onnxruntime / perception 模块全部惰性导入:
  仅 numpy 依赖的公共解析与降采样逻辑可在无 cv2/torch 的环境独立测试.
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import sys
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# 项目包根目录 (orinlanding/) 加入 sys.path, 与 replay_bag_offline.py 一致
_PACKAGE_ROOT = Path(__file__).resolve().parent.parent
if str(_PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(_PACKAGE_ROOT))

# 纯 numpy 最近有效单元填充 (utils 无 cv2/torch 依赖, 保持本模块
# 可在无 cv2/torch 环境独立测试的性质)
from utils.valid_nearest import fill_valid_nearest  # noqa: E402

logger = logging.getLogger("Compare")

# BEV 降采样网格分辨率 (观测尺寸, 与 DRL 输入/深度投影对齐)
BEV_GRID_RES_DEFAULT = 128


def setup_logging(level=logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


# ──────────────────────────────────────────────
# 配置加载 (与 pipeline.py / replay_bag_offline.py 完全一致)
# ──────────────────────────────────────────────
def _merge_config_overrides(cfg: dict, overrides: dict) -> dict:
    for section, values in overrides.items():
        if isinstance(values, dict) and isinstance(cfg.get(section), dict):
            cfg.setdefault(section, {})
            _merge_config_overrides(cfg[section], values)
        else:
            cfg[section] = values
    return cfg


def load_config(path: str) -> dict:
    config_path = Path(path).resolve()
    if not config_path.is_file() and config_path.parent.name != "runs":
        # Orin 上实验数据在 experiments/runs/<name>/, 开发机在 experiments/<name>/
        alt = (config_path.parents[1] / "runs" / config_path.parent.name / config_path.name)
        if alt.is_file():
            logger.warning("[Config] %s 不存在, 改用 %s", config_path, alt)
            config_path = alt
    import yaml
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    parent = cfg.pop("extends", None)
    if parent:
        parent_path = Path(parent)
        if not parent_path.is_absolute():
            parent_path = config_path.parent / parent_path
        base = load_config(str(parent_path))
        return _merge_config_overrides(base, cfg)
    return cfg


def _resolve_path_candidates(path: str) -> Path | None:
    """相对路径依次尝试: 原样 → 包根目录 → 仓库根目录."""
    candidates = [Path(path)]
    if not Path(path).is_absolute():
        candidates.append(_PACKAGE_ROOT / path)
        candidates.append(_PACKAGE_ROOT.parent / path)
    for cand in candidates:
        if cand.is_file():
            return cand
    return None


# ──────────────────────────────────────────────
# 消息解析 (不依赖 rospy / 消息包)
# ──────────────────────────────────────────────
def pc2_to_numpy(msg) -> np.ndarray:
    """sensor_msgs/PointCloud2 → (N,3) float32, 滤除 NaN/Inf.

    按字段 offset + point_step 解析 (与 pipeline.py 一致), 不依赖
    ros_numpy, 可独立测试.
    """
    field_offsets = {f.name: f.offset for f in msg.fields}
    if not all(k in field_offsets for k in ("x", "y", "z")):
        return np.empty((0, 3), dtype=np.float32)
    n = int(msg.width) * int(msg.height)
    if n == 0 or msg.point_step < 12 or msg.row_step < msg.point_step:
        return np.empty((0, 3), dtype=np.float32)
    endian = ">f4" if msg.is_bigendian else "<f4"
    dtype = np.dtype({
        "names": ["x", "y", "z"],
        "formats": [endian, endian, endian],
        "offsets": [field_offsets["x"], field_offsets["y"], field_offsets["z"]],
        "itemsize": msg.point_step,
    })
    arr = np.frombuffer(msg.data, dtype=dtype, count=n)
    pts = np.column_stack((arr["x"], arr["y"], arr["z"])).astype(np.float32, copy=False)
    return pts[np.isfinite(pts).all(axis=1)]


def custom_msg_to_numpy(msg) -> np.ndarray:
    """livox_ros_driver2/CustomMsg → (N,3) float32, 滤除 NaN.

    duck-typed 读取 points[i].x/y/z: 不依赖 livox 消息包 (rosbag 可在
    无消息包时动态反序列化). 用于原始点云参考统计和可视化.
    """
    points = getattr(msg, "points", None)
    if points is None or len(points) == 0:
        return np.empty((0, 3), dtype=np.float32)
    pts = np.array([[p.x, p.y, p.z] for p in points], dtype=np.float32)
    if len(pts) == 0:
        return pts
    return pts[np.isfinite(pts).all(axis=1)]


def custom_msg_to_raw_arrays(msg):
    """Livox CustomMsg -> valid xyz, intensity and per-point offset seconds.

    Livox messages use a fixed-size point array; invalid samples are commonly
    zero-filled.  The tag/range filtering matches FAST-LIO's Avia preprocessor
    while retaining the point timestamp needed for scan deskewing.
    """
    points = getattr(msg, "points", None)
    if points is None or len(points) == 0:
        return (np.empty((0, 3), dtype=np.float32),
                np.empty(0, dtype=np.float32),
                np.empty(0, dtype=np.float64))
    n = min(int(getattr(msg, "point_num", len(points))), len(points))
    points = points[:n]
    xyz = np.asarray([(p.x, p.y, p.z) for p in points], dtype=np.float32)
    intensity = np.asarray([p.reflectivity for p in points], dtype=np.float32)
    offsets = np.asarray([p.offset_time for p in points], dtype=np.float64) * 1e-9
    tag_class = np.asarray([p.tag for p in points], dtype=np.uint8) & 0x30
    valid = ((tag_class == 0x00) | (tag_class == 0x10))
    valid &= np.isfinite(xyz).all(axis=1) & np.isfinite(offsets)
    valid &= np.einsum("ij,ij->i", xyz, xyz) > 0.1 * 0.1
    return xyz[valid], intensity[valid], offsets[valid]


def stamp_to_sec(stamp) -> float:
    return float(stamp.secs) + float(stamp.nsecs) * 1e-9


def quat_to_euler(x, y, z, w):
    """四元数 → 欧拉角 (roll, pitch, yaw) rad (与 pipeline.py 一致)."""
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)
    sinp = 2.0 * (w * y - z * x)
    pitch = math.asin(max(-1.0, min(1.0, sinp))) if abs(sinp) < 1 else math.copysign(math.pi / 2, sinp)
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


# ──────────────────────────────────────────────
# 旋转矩阵 (公式与 perception/halss_preprocess.py 的 _rot_zyx/_rot_z 完全一致,
# 为独立测试不跨模块取私有函数, 改动需保持同步)
# ──────────────────────────────────────────────
def _rot_zyx(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Body-to-world rotation for roll/pitch/yaw in NED convention (row-vector p @ R.T)."""
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ], dtype=np.float32)


def _rot_z(yaw: float) -> np.ndarray:
    """Yaw-only rotation (2D z rotation embedded in 3D), row-vector convention."""
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([
        [c, -s, 0.0],
        [s,  c, 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float32)


def world_to_level_body(world_points: np.ndarray, pose: np.ndarray) -> np.ndarray:
    """统一世界坐标 W' → 当前帧水平机体坐标 (x 前 / y 侧 / z 下).

    仅 yaw 旋转 + 平移 (roll/pitch 已在世界帧中消去), z 轴只平移.
    与 replay_window10 的私有实现同源 (本函数迁自该脚本, 改动需保持同步).
    """
    pts = np.asarray(world_points, dtype=np.float32)
    if len(pts) == 0:
        return pts
    out = pts.copy()
    out[:, 0] -= float(pose[0])
    out[:, 1] -= float(pose[1])
    out[:, 2] += float(pose[2])  # z-down 世界 → 机下深度
    yaw = float(pose[5])
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        out = out @ _rot_z(yaw)
    return out.astype(np.float32, copy=False)


# ──────────────────────────────────────────────
# rosbag 流式读取 + cloud/pose 时间同步
# ──────────────────────────────────────────────
@dataclass
class CloudFrame:
    """一帧去畸变点云 + 时间最近位姿 + 可选最近原始帧统计."""
    cloud_pts: np.ndarray            # (N,3) float32, /cloud_registered_body 原值
    cloud_stamp: float
    cloud_seq: int
    pose: np.ndarray                 # [x,y,z,roll,pitch,yaw] (ENU pos, PX4 欧拉)
    pose_stamp: float
    sync_ms: float
    raw_pts: np.ndarray | None       # /livox/lidar 最近帧 (仅参考)
    raw_stamp: float | None
    raw_cloud_delta_ms: float | None
    cloud_source: str = "fastlio"


class BagFrameSource:
    """单遍流式读取: cloud + pose 主遍历, raw 参考话题同遍匹配.

    raw 话题缺失/反序列化失败时跳过并告警一次, 不影响主输入.
    """

    POSE_CANDIDATES = ("/mavros/local_position/odom", "/ali_odom", "/Odometry")

    def __init__(self, bag_path: str, cfg: dict, cloud_topic: str = None,
                 pose_topic: str = None, raw_topic: str = None,
                 imu_topic: str = "/livox/imu", cloud_source: str = "fastlio",
                 max_sync_ms: float = 100.0, pose_window: int = 100):
        import rosbag
        self._log = logger
        self._max_sync_ms = float(max_sync_ms)
        self._bag_path = str(bag_path)
        self._cloud_source = str(cloud_source).lower()
        if self._cloud_source not in {"fastlio", "raw_imu"}:
            raise ValueError(f"unsupported cloud_source: {cloud_source}")
        self._bag = rosbag.Bag(self._bag_path, "r")
        perc_cfg = cfg.get("perception", {}) if isinstance(cfg, dict) else {}
        self._cfg_body_rotation = np.asarray(
            perc_cfg.get("body_R_from_lidar_imu", [1, 0, 0, 0, 1, 0, 0, 0, 1]),
            dtype=np.float64).reshape(3, 3)
        info = self._bag.get_type_and_topic_info()
        self._available = set(info.topics.keys())
        self._types = {t: ti.msg_type for t, ti in info.topics.items()}

        self._imu_topic = str(imu_topic or "/livox/imu")
        self._cloud_topic = (
            str(raw_topic or "/livox/lidar")
            if self._cloud_source == "raw_imu"
            else str(cloud_topic or "/cloud_registered_body")
        )
        if self._cloud_topic not in self._available:
            raise RuntimeError(
                f"[Bag] cloud topic {self._cloud_topic} not in bag. "
                f"Available: {', '.join(sorted(self._available))}")
        self._pose_topic = self._resolve_pose_topic(pose_topic)
        self._raw_topic = raw_topic if raw_topic and raw_topic in self._available else None
        if raw_topic and raw_topic not in self._available:
            self._log.warning("[Bag] raw topic %s not in bag — 原始参考禁用", raw_topic)

        self._pose_window = deque(maxlen=int(pose_window))
        self._raw_window = deque(maxlen=50)
        self._raw_skipped = 0
        self._raw_warned = False
        self._pose_bad = 0

        self._imu_t = np.empty(0, dtype=np.float64)
        self._imu_rot = np.empty((0, 3, 3), dtype=np.float64)
        self._pose_t = np.empty(0, dtype=np.float64)
        self._pose_xyz = np.empty((0, 3), dtype=np.float64)
        self._pose6 = np.empty((0, 6), dtype=np.float32)
        self._pose_quat = np.empty((0, 4), dtype=np.float64)
        if self._cloud_source == "raw_imu":
            if self._imu_topic not in self._available:
                raise RuntimeError(
                    f"[Bag] IMU topic {self._imu_topic} not in bag. "
                    f"Available: {', '.join(sorted(self._available))}")
            self._load_raw_imu_state()

        read_topics = [self._cloud_topic, self._pose_topic]
        if self._raw_topic and self._cloud_source == "fastlio":
            read_topics.append(self._raw_topic)
        self._read_topics = read_topics
        self._log.info("[Bag] source=%s cloud=%s pose=%s raw=%s imu=%s",
                       self._cloud_source, self._cloud_topic, self._pose_topic,
                       self._raw_topic or "n/a", self._imu_topic if self._cloud_source == "raw_imu" else "n/a")

    # ── 访问器 ──
    @property
    def cloud_topic(self) -> str:
        return self._cloud_topic

    @property
    def pose_topic(self) -> str:
        return self._pose_topic

    @property
    def raw_topic(self) -> str | None:
        return self._raw_topic

    @property
    def raw_type(self) -> str | None:
        return self._types.get(self._raw_topic) if self._raw_topic else None

    @property
    def cloud_source(self) -> str:
        return self._cloud_source

    def close(self):
        try:
            self._bag.close()
        except Exception:
            pass

    def _resolve_pose_topic(self, explicit: str | None) -> str:
        if explicit:
            if explicit not in self._available:
                raise RuntimeError(
                    f"[Bag] pose topic {explicit} not in bag. "
                    f"Available: {', '.join(sorted(self._available))}")
            return explicit
        for cand in self.POSE_CANDIDATES:
            if cand in self._available:
                self._log.info("[Bag] 自动选择 pose topic: %s (%s)", cand, self._types[cand])
                return cand
        raise RuntimeError(
            f"[Bag] 未找到位姿话题 (查找 {', '.join(self.POSE_CANDIDATES)}), "
            f"请用 --pose-topic 显式指定")

    @staticmethod
    def _exp_so3(rotvec: np.ndarray) -> np.ndarray:
        """Rodrigues exponential for one IMU angular increment."""
        theta = float(np.linalg.norm(rotvec))
        if theta < 1e-12:
            x, y, z = map(float, rotvec)
            return np.array([[1.0, -z, y], [z, 1.0, -x],
                             [-y, x, 1.0]], dtype=np.float64)
        axis = np.asarray(rotvec, dtype=np.float64) / theta
        x, y, z = axis
        K = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
        return np.eye(3) + math.sin(theta) * K + (1.0 - math.cos(theta)) * (K @ K)

    def _load_raw_imu_state(self):
        """Preload pose/IMU samples so each raw scan can use scan-end data."""
        imu_t, gyro = [], []
        pose_t, pose6, pose_q = [], [], []
        for topic, msg, _ in self._bag.read_messages(
                topics=[self._imu_topic, self._pose_topic]):
            stamp = stamp_to_sec(msg.header.stamp)
            if topic == self._imu_topic:
                imu_t.append(stamp)
                gyro.append((msg.angular_velocity.x, msg.angular_velocity.y,
                             msg.angular_velocity.z))
            else:
                p, q = self._odom_to_pose6(msg)
                if p is not None:
                    pose_t.append(stamp); pose6.append(p); pose_q.append(q)
        if len(imu_t) < 2:
            raise RuntimeError(f"[Bag] IMU topic {self._imu_topic} has fewer than 2 samples")
        if not pose_t:
            raise RuntimeError(f"[Bag] no valid odometry samples on {self._pose_topic}")
        imu_t = np.asarray(imu_t, dtype=np.float64)
        gyro = np.asarray(gyro, dtype=np.float64)
        order = np.argsort(imu_t, kind="stable")
        imu_t, gyro = imu_t[order], gyro[order]
        keep = np.r_[True, np.diff(imu_t) > 1e-9]
        self._imu_t = imu_t[keep]
        gyro = gyro[keep]
        self._imu_rot = np.empty((len(self._imu_t), 3, 3), dtype=np.float64)
        self._imu_rot[0] = np.eye(3)
        for i in range(1, len(self._imu_t)):
            dt = float(np.clip(self._imu_t[i] - self._imu_t[i - 1], 0.0, 0.05))
            self._imu_rot[i] = self._imu_rot[i - 1] @ self._exp_so3(gyro[i - 1] * dt)

        pose_t = np.asarray(pose_t, dtype=np.float64)
        pose6 = np.asarray(pose6, dtype=np.float32)
        pose_q = np.asarray(pose_q, dtype=np.float64)
        order = np.argsort(pose_t, kind="stable")
        pose_t, pose6, pose_q = pose_t[order], pose6[order], pose_q[order]
        keep = np.r_[True, np.diff(pose_t) > 1e-9]
        self._pose_t, self._pose6, self._pose_quat = pose_t[keep], pose6[keep], pose_q[keep]
        self._pose_xyz = self._pose6[:, :3].astype(np.float64)
        self._log.info("[RawIMU] loaded imu=%d pose=%d", len(self._imu_t), len(self._pose_t))

    def _pose_at_full(self, t: float):
        """Return interpolated pose, nearest pose stamp, and sync error."""
        if not len(self._pose_t):
            return None, None, float("inf")
        if t < self._pose_t[0] or t > self._pose_t[-1]:
            i = int(np.argmin(np.abs(self._pose_t - t)))
            err = abs(float(self._pose_t[i] - t))
            if err > self._max_sync_ms / 1000.0:
                return None, None, err
            return self._pose6[i].copy(), float(self._pose_t[i]), err
        i = int(np.searchsorted(self._pose_t, t, side="right")) - 1
        if i >= len(self._pose_t) - 1:
            i = len(self._pose_t) - 1
            return self._pose6[i].copy(), float(self._pose_t[i]), abs(float(self._pose_t[i] - t))
        t0, t1 = self._pose_t[i], self._pose_t[i + 1]
        if t1 <= t0:
            return self._pose6[i].copy(), float(t0), 0.0
        frac = (t - t0) / (t1 - t0)
        xyz = self._pose_xyz[i] + frac * (self._pose_xyz[i + 1] - self._pose_xyz[i])
        q = quat_slerp(self._pose_quat[i], self._pose_quat[i + 1], float(frac))
        r, p, y = quat_to_euler(*q)
        return np.array([xyz[0], xyz[1], xyz[2], r, p, y], dtype=np.float32), float(t), max(t - t0, t1 - t)

    def _deskew_raw(self, msg):
        xyz, intensity, offsets = custom_msg_to_raw_arrays(msg)
        if len(xyz) == 0:
            return None
        stamp = stamp_to_sec(msg.header.stamp)
        point_t = stamp + offsets
        ref_t = float(point_t.max())
        ref_pose, pose_stamp, sync_s = self._pose_at_full(ref_t)
        if ref_pose is None:
            return None
        # Nearest integrated IMU rotations are sufficient at the recorded
        # ~200 Hz rate and avoid introducing a scipy runtime dependency.
        idx = np.searchsorted(self._imu_t, point_t, side="left").clip(0, len(self._imu_t) - 1)
        left = np.maximum(idx - 1, 0)
        idx[np.abs(self._imu_t[left] - point_t) < np.abs(self._imu_t[idx] - point_t)] = left[
            np.abs(self._imu_t[left] - point_t) < np.abs(self._imu_t[idx] - point_t)]
        ref_idx = int(np.argmin(np.abs(self._imu_t - ref_t)))
        r_rel = np.einsum("ij,njk->nik", self._imu_rot[ref_idx].T, self._imu_rot[idx])
        deskewed = np.einsum("nij,nj->ni", r_rel, xyz.astype(np.float64))

        # Compensate translation into the scan-end IMU frame using odometry.
        point_xyz = np.column_stack([
            np.interp(point_t, self._pose_t, self._pose_xyz[:, k]) for k in range(3)
        ])
        r_world_body = _rot_zyx(float(ref_pose[3]), float(ref_pose[4]), float(ref_pose[5])).astype(np.float64)
        r_body_from_imu = np.asarray(self._cfg_body_rotation, dtype=np.float64)
        r_world_imu = r_world_body @ r_body_from_imu
        delta_world = point_xyz - ref_pose[:3].astype(np.float64)
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            delta_imu = delta_world @ r_world_imu
        deskewed = (deskewed + delta_imu).astype(np.float32)
        if not np.isfinite(deskewed).all():
            self._log.warning("[RawIMU] non-finite deskewed scan at t=%.6f; skip", stamp)
            return None
        return deskewed, intensity, stamp, ref_t, ref_pose, pose_stamp, sync_s

    def pose_at_interp(self, cloud_stamp: float) -> np.ndarray | None:
        """点云时间戳处的位姿: 位置线性插值 + 姿态四元数 SLERP.

        取 _pose_window 中夹住 cloud_stamp 的两个 odom 样本插值; 窗口内
        无夹住样本 (首帧 / odom 与 cloud 同帧交错) 或两侧同一样本时返回
        None, 调用方回退到时间最近位姿 (不外推). 返回 [x,y,z,roll,pitch,yaw]
        float32.
        """
        if not self._pose_window:
            return None
        times = np.array([t for t, *_ in self._pose_window])
        if cloud_stamp <= times[0] or cloud_stamp >= times[-1]:
            return None  # 无夹住样本: 不外推
        i = int(np.searchsorted(times, cloud_stamp, side="right")) - 1
        t0, p0, q0 = self._pose_window[i]
        t1, p1, q1 = self._pose_window[i + 1]
        if t1 <= t0:
            return p0.copy()
        frac = (cloud_stamp - t0) / (t1 - t0)
        xyz = p0[:3] + frac * (p1[:3] - p0[:3])
        q = quat_slerp(q0, q1, frac)
        roll, pitch, yaw = quat_to_euler(q[0], q[1], q[2], q[3])
        return np.array([xyz[0], xyz[1], xyz[2], roll, pitch, yaw],
                        dtype=np.float32)

    @staticmethod
    def _odom_to_pose6(msg):
        """nav_msgs/Odometry → ([x,y,z,roll,pitch,yaw], 四元数 [x,y,z,w]).

        四元数保留原始值供点云时间戳处 SLERP 插值 (pose_at_interp);
        非 Odometry 返回 (None, None).
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

    def __iter__(self):
        if self._cloud_source == "raw_imu":
            for _, msg, _ in self._bag.read_messages(topics=[self._cloud_topic]):
                result = self._deskew_raw(msg)
                if result is None:
                    continue
                deskewed, _, raw_stamp, cloud_stamp, pose, pose_stamp, sync_s = result
                raw_xyz, _, _ = custom_msg_to_raw_arrays(msg)
                yield CloudFrame(
                    cloud_pts=deskewed,
                    cloud_stamp=float(cloud_stamp),
                    cloud_seq=int(getattr(msg.header, "seq", 0)),
                    pose=pose,
                    pose_stamp=float(pose_stamp),
                    sync_ms=float(sync_s * 1000.0),
                    raw_pts=raw_xyz,
                    raw_stamp=float(raw_stamp),
                    raw_cloud_delta_ms=abs(float(cloud_stamp - raw_stamp)) * 1000.0,
                    cloud_source="raw_imu",
                )
            return

        it = self._bag.read_messages(topics=self._read_topics)
        while True:
            try:
                topic_name, msg, ros_stamp = next(it)
            except StopIteration:
                break
            except Exception as exc:
                # 未知消息类型等反序列化失败: 仅可能来自 raw 参考话题, 跳过
                self._raw_skipped += 1
                if not self._raw_warned:
                    self._raw_warned = True
                    self._log.warning(
                        "[Bag] 消息反序列化失败 (%s) — 已跳过 %d 条; "
                        "原始参考话题可能不可用", exc, self._raw_skipped)
                continue

            stamp = (stamp_to_sec(msg.header.stamp)
                     if hasattr(msg, "header") and msg.header is not None
                     else stamp_to_sec(ros_stamp))

            if topic_name == self._pose_topic:
                pose6, quat = self._odom_to_pose6(msg)
                if pose6 is None:
                    self._pose_bad += 1
                    if self._pose_bad == 1:
                        self._log.warning(
                            "[Bag] pose topic %s 不是 Odometry 格式 — 该话题被忽略",
                            self._pose_topic)
                    continue
                self._pose_window.append((stamp, pose6, quat))

            elif self._raw_topic and topic_name == self._raw_topic:
                try:
                    raw_pts = custom_msg_to_numpy(msg)
                except Exception as exc:
                    self._raw_skipped += 1
                    if not self._raw_warned:
                        self._raw_warned = True
                        self._log.warning("[Bag] raw 消息解析失败 (%s)", exc)
                    continue
                self._raw_window.append((stamp, raw_pts))

            elif topic_name == self._cloud_topic:
                if not self._pose_window:
                    continue  # 尚无位姿, 跳过
                cloud_pts = pc2_to_numpy(msg)
                if len(cloud_pts) == 0:
                    continue
                # 找时间最近位姿
                pose_times = np.array([t for t, *_ in self._pose_window])
                nearest = int(np.argmin(np.abs(pose_times - stamp)))
                pose_stamp = float(pose_times[nearest])
                sync_ms = abs(pose_stamp - stamp) * 1000.0
                if sync_ms > self._max_sync_ms:
                    self._log.warning(
                        "[Sync] %.0fms > %.0fms at t=%.3f, skip",
                        sync_ms, self._max_sync_ms, stamp)
                    continue
                _, pose6, _ = self._pose_window[nearest]
                # 最近原始帧 (仅统计)
                raw_pts = raw_stamp = raw_delta = None
                if self._raw_window:
                    raw_times = np.array([t for t, _ in self._raw_window])
                    r_nearest = int(np.argmin(np.abs(raw_times - stamp)))
                    raw_stamp = float(raw_times[r_nearest])
                    raw_delta = abs(raw_stamp - stamp) * 1000.0
                    raw_pts = self._raw_window[r_nearest][1]
                yield CloudFrame(
                    cloud_pts=cloud_pts,
                    cloud_stamp=float(stamp),
                    cloud_seq=int(getattr(msg.header, "seq", 0)),
                    pose=pose6,
                    pose_stamp=pose_stamp,
                    sync_ms=sync_ms,
                    raw_pts=raw_pts,
                    raw_stamp=raw_stamp,
                    raw_cloud_delta_ms=raw_delta,
                    cloud_source="fastlio",
                )


# ──────────────────────────────────────────────
# 感知参数提取 / 动态 ROI (与 replay_bag_offline.py 主循环一致)
# ──────────────────────────────────────────────
def perception_params(cfg: dict) -> dict:
    perc = cfg.get("perception", {})
    obs = cfg.get("observation", {})
    runtime = cfg.get("runtime", {})
    depth = cfg.get("depth_projection", {})
    return {
        "obs_h": int(obs.get("img_height", 128)),
        "obs_w": int(obs.get("img_width", 128)),
        "dmax": float(depth.get("max_range", 30.0)),
        "safe_id": int(perc.get("safe_class_id", 1)),
        "danger_id": int(perc.get("danger_class_id", 9)),
        "max_sync_ms": float(runtime.get("max_cloud_odom_sync_ms", 100.0)),
        "projection_mode": str(depth.get("mode", "training_camera")).lower(),
        "roi_half_x": float(perc.get("halss_roi_half_x_m", 5.0)),
        "roi_half_y": float(perc.get("halss_roi_half_y_m", 5.0)),
        "roi_dynamic": bool(perc.get("halss_roi_dynamic_enabled", True)),
        "roi_fov_half_rad": math.radians(float(perc.get("halss_roi_fov_half_deg", 45.0))),
        "roi_min_half": float(perc.get("halss_roi_min_half_m", 0.05)),
        "roi_max_half": float(perc.get("halss_roi_max_half_m", 30.0)),
        "bev_grid_res": int(perc.get("compare_bev_grid_res", BEV_GRID_RES_DEFAULT)),
        "perc_cfg": perc,
    }


def roi_bounds(half_x: float, half_y: float) -> dict:
    return {"x_min": -half_x, "x_max": half_x, "y_min": -half_y, "y_max": half_y}


def dynamic_roi_half_extents(params: dict, pose_z: float, ground_z: float,
                             camera, projection_mode: str = None):
    """动态 ROI 半宽: training_camera 用训练相机 FOV 对地投影, 其余用 FOV 锥.

    与 replay_bag_offline.py 主循环完全一致的钳制逻辑.
    """
    if not params["roi_dynamic"]:
        return params["roi_half_x"], params["roi_half_y"]
    H = abs(float(pose_z) - float(ground_z))
    H = max(H, 0.1)
    mode = projection_mode or params["projection_mode"]
    if mode == "training_camera":
        cur_half_x, cur_half_y = camera.ground_half_extents(H)
        cur_half_x = max(params["roi_min_half"], min(params["roi_max_half"], cur_half_x))
        cur_half_y = max(params["roi_min_half"], min(params["roi_max_half"], cur_half_y))
        return cur_half_x, cur_half_y
    half = H * math.tan(params["roi_fov_half_rad"])
    half = max(params["roi_min_half"], min(params["roi_max_half"], half))
    return half, half


# ──────────────────────────────────────────────
# BEV 粗糙度保持降采样 (三脚本共用, 网格行 0 = +y, 与 HALSS 语义方向一致)
# ──────────────────────────────────────────────
@dataclass
class BevGrid:
    """BEV 网格: 每单元保留 z-down 最大点/最小点/高度差/单元点数.

    points 为各占用单元 zmax 点 + zmin 点拼接 (M,3), HALSS 与深度分支
    共用同一批点, 保证对比只反映采样方式变化.

    cell_x_m/cell_y_m: 物理栅格尺寸 (米), 按 (G-1) 等分 (与降采样网格化
    口径一致); 缺省 None 时由 bounds/grid_res 惰性计算.
    """
    points: np.ndarray          # (M,3) float32
    z_max: np.ndarray           # (G,G) float32, 空单元 NaN
    z_min: np.ndarray           # (G,G) float32, 空单元 NaN
    z_diff: np.ndarray          # (G,G) float32, z_max - z_min, 空单元 NaN
    count: np.ndarray           # (G,G) int32
    grid_res: int
    bounds: dict
    stats: dict
    cell_x_m: float = None      # 物理栅格尺寸 x (米)
    cell_y_m: float = None      # 物理栅格尺寸 y (米)

    @property
    def occupied(self) -> np.ndarray:
        return self.count > 0

    @property
    def cell_size_m(self) -> tuple[float, float]:
        """物理栅格尺寸 (cell_x, cell_y) 米; 与降采样网格化 (G-1) 等分口径一致."""
        if self.cell_x_m is None or self.cell_y_m is None:
            g = max(int(self.grid_res) - 1, 1)
            x_span = max(float(self.bounds["x_max"]) - float(self.bounds["x_min"]), 1e-6)
            y_span = max(float(self.bounds["y_max"]) - float(self.bounds["y_min"]), 1e-6)
            return x_span / g, y_span / g
        return float(self.cell_x_m), float(self.cell_y_m)


def bev_roughness_downsample(points: np.ndarray, bounds: dict,
                             grid_res: int = BEV_GRID_RES_DEFAULT) -> BevGrid:
    """向量化 BEV 粗糙度保持降采样.

    每单元保留:
      - z-down 最大点 (单元内 z 最大的原始点);
      - z-down 最小点 (单元内 z 最小的原始点);
      - 高度差 z_max - z_min;
      - 单元点数.
    单点单元只保留一个点 (zmax == zmin), 避免向 HALSS/深度分支注入重复点.
    """
    G = int(grid_res)
    if G <= 0:
        raise ValueError("grid_res must be positive")
    x_min, x_max = float(bounds["x_min"]), float(bounds["x_max"])
    y_min, y_max = float(bounds["y_min"]), float(bounds["y_max"])
    x_span = max(x_max - x_min, 1e-6)
    y_span = max(y_max - y_min, 1e-6)
    cell_x_m = x_span / max(G - 1, 1)
    cell_y_m = y_span / max(G - 1, 1)

    stats = {
        "input_points": 0, "output_points": 0, "occupied_cells": 0,
        "grid_res": G, "height_diff_mean_m": float("nan"),
        "height_diff_max_m": float("nan"),
        "cell_x_m": cell_x_m, "cell_y_m": cell_y_m,
    }
    empty_grid = np.full((G, G), np.nan, dtype=np.float32)
    if points is None or len(points) == 0:
        return BevGrid(np.empty((0, 3), dtype=np.float32), empty_grid.copy(),
                       empty_grid.copy(), empty_grid.copy(),
                       np.zeros((G, G), dtype=np.int32), G, bounds, stats,
                       cell_x_m=cell_x_m, cell_y_m=cell_y_m)

    pts = np.asarray(points, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[1] < 3:
        raise ValueError("points must have shape (N,3) or (N,>=3)")
    pts = pts[:, :3]
    pts = pts[np.isfinite(pts).all(axis=1)]
    stats["input_points"] = int(len(pts))
    if len(pts) == 0:
        return BevGrid(np.empty((0, 3), dtype=np.float32), empty_grid.copy(),
                       empty_grid.copy(), empty_grid.copy(),
                       np.zeros((G, G), dtype=np.int32), G, bounds, stats,
                       cell_x_m=cell_x_m, cell_y_m=cell_y_m)

    # 行 0 = +y (与 HALSS flipud 语义方向一致, 供标签采样对齐)
    col = np.rint((pts[:, 0] - x_min) / x_span * (G - 1)).astype(np.int32)
    row_unflipped = np.rint((pts[:, 1] - y_min) / y_span * (G - 1)).astype(np.int32)
    row = (G - 1) - row_unflipped
    inside = (row >= 0) & (row < G) & (col >= 0) & (col < G)
    pts = pts[inside]
    row, col = row[inside], col[inside]
    if len(pts) == 0:
        # The dynamic ROI can collapse to its minimum size on the first
        # ground frame.  All finite input points may then lie outside the
        # current BEV bounds; return a valid empty grid before reductions.
        return BevGrid(np.empty((0, 3), dtype=np.float32), empty_grid.copy(),
                       empty_grid.copy(), empty_grid.copy(),
                       np.zeros((G, G), dtype=np.int32), G, bounds, stats,
                       cell_x_m=cell_x_m, cell_y_m=cell_y_m)
    z = pts[:, 2]
    flat = row * G + col

    count = np.bincount(flat, minlength=G * G).reshape(G, G).astype(np.int32)
    z_max = np.full(G * G, -np.inf, dtype=np.float32)
    z_min = np.full(G * G, np.inf, dtype=np.float32)
    np.maximum.at(z_max, flat, z)
    np.minimum.at(z_min, flat, z)
    z_max = z_max.reshape(G, G)
    z_min = z_min.reshape(G, G)
    z_diff = z_max - z_min
    occ = count > 0
    z_max[~occ] = np.nan
    z_min[~occ] = np.nan
    z_diff[~occ] = np.nan

    # 每占用单元: 首个按 z 降序的点 = zmax 点, 首个按 z 升序的点 = zmin 点
    order_desc = np.argsort(-z, kind="stable")
    _, first_max = np.unique(flat[order_desc], return_index=True)
    max_idx = order_desc[first_max]
    order_asc = np.argsort(z, kind="stable")
    _, first_min = np.unique(flat[order_asc], return_index=True)
    min_idx = order_asc[first_min]
    both = np.concatenate([max_idx, min_idx])
    # 单点单元 zmax==zmin → 去重
    out = np.unique(pts[both], axis=0).astype(np.float32, copy=False)

    if occ.any():
        # A non-empty occupancy mask is not sufficient to guarantee finite
        # height statistics: malformed/degenerate input can leave all values
        # in z_diff[occ] non-finite.  Never reduce an empty finite subset.
        occupied_diff = z_diff[occ]
        finite_diff = occupied_diff[np.isfinite(occupied_diff)]
        stats.update({
            "output_points": int(len(out)),
            "occupied_cells": int(occ.sum()),
            "height_diff_mean_m": (
                float(np.mean(finite_diff)) if finite_diff.size else float("nan")
            ),
            "height_diff_max_m": (
                float(np.max(finite_diff)) if finite_diff.size else float("nan")
            ),
        })
    # 空占用 (如起飞前动态 ROI 过小) → 保持 stats 初始 0/NaN, 不崩溃
    return BevGrid(out, z_max, z_min, z_diff, count, G, bounds, stats,
                   cell_x_m=cell_x_m, cell_y_m=cell_y_m)


# ──────────────────────────────────────────────
# BEV 分辨率参数化 + 物理栅格 → 模型网格上采样
# ──────────────────────────────────────────────
def bev_grid_res_from_cell(bounds: dict, cell_m: float, min_res: int = 8) -> int:
    """物理栅格尺寸 (米) → 方形 BEV 分辨率 (按 ROI 较大边计算, 向上取整).

    --bev-cell-size-m > 0 时用该函数自动决定融合网格; 与降采样网格化
    (G-1) 等分口径一致, cell_x/cell_y 实际值随 ROI 纵横比而不同.
    """
    cell = float(cell_m)
    if cell <= 0.0:
        raise ValueError("cell_m must be positive")
    x_span = max(float(bounds["x_max"]) - float(bounds["x_min"]), 1e-6)
    y_span = max(float(bounds["y_max"]) - float(bounds["y_min"]), 1e-6)
    g = int(math.ceil(max(x_span, y_span) / cell)) + 1
    return max(int(min_res), g)


def upsample_nearest_map(grid_res: int, model_res: int) -> np.ndarray:
    """最近邻上采样源索引映射 (方形网格, 行/列共用).

    dst[i] = 源网格中与 (i+0.5)*G/M 最近的下标 (中心对齐), 边界钳制.
    上采样得到的数值/掩码只会复制源单元, 不会插值产生新值.
    """
    G, M = int(grid_res), int(model_res)
    if M <= 0 or G <= 0:
        raise ValueError("grid_res/model_res must be positive")
    src = np.clip(np.rint((np.arange(M, dtype=np.float64) + 0.5) * G / M - 0.5),
                  0, G - 1).astype(np.int32)
    return src


def upsample_grid_nearest(values: np.ndarray, grid_res: int,
                          model_res: int) -> np.ndarray:
    """把 (G,G) 网格按最近邻上采样到 (M,M); 掩码/数值通用, 不生成新有效单元."""
    src = upsample_nearest_map(int(grid_res), int(model_res))
    return np.asarray(values)[src][:, src]


def bev_upsample_to_model(bev: BevGrid, model_res: int) -> BevGrid:
    """物理融合网格 BEV → 模型网格 (默认 128×128) BEV, 最近邻上采样.

    要求:
      - z_min/z_max/z_diff/count 全部最近邻复制 (源单元的值原样搬移);
      - occupied (count>0) 只来自原始有效单元的复制, 未知单元绝不上采样
        成已观测单元 (不做双线性插值);
      - points 保持物理代表点不变 (深度投影与语义按物理点 + 模型网格工作).
    """
    G = int(bev.grid_res)
    M = int(model_res)
    if M == G:
        return bev
    if M <= 0 or G <= 0:
        raise ValueError("grid_res/model_res must be positive")
    src = upsample_nearest_map(G, M)
    stats = dict(bev.stats)
    stats["grid_res"] = M
    stats["upsampled_from_grid"] = G
    x_span = max(float(bev.bounds["x_max"]) - float(bev.bounds["x_min"]), 1e-6)
    y_span = max(float(bev.bounds["y_max"]) - float(bev.bounds["y_min"]), 1e-6)
    return BevGrid(
        # NumPy 1.x (used by the Jetson environment) does not accept the
        # NumPy-2-only ``copy`` keyword on np.asarray().  The input is already
        # a NumPy array in normal operation, so omitting it preserves the
        # desired no-copy behavior while remaining version compatible.
        np.asarray(bev.points, dtype=np.float32),
        np.asarray(bev.z_max)[src][:, src],
        np.asarray(bev.z_min)[src][:, src],
        np.asarray(bev.z_diff)[src][:, src],
        np.asarray(bev.count)[src][:, src],
        M, dict(bev.bounds), stats,
        cell_x_m=x_span / max(M - 1, 1), cell_y_m=y_span / max(M - 1, 1))


# ──────────────────────────────────────────────
# 几何语义分支 (脚本二/三, 不调用 HALSS Bayesian)
# ──────────────────────────────────────────────
def geometric_semantic_map(bev: BevGrid, slope_threshold_deg: float,
                           roughness_threshold_m: float, safe_id: int = 1,
                           danger_id: int = 9, smooth_ksize: int = 5):
    """BEV 网格 → 几何安全/危险语义图 (128×128).

    管线: 地面高度栅格 (z_min) → 局部填洞 (仅观测足迹内) → 高斯局部平滑 →
    Sobel 梯度 (换算 m/m) → 法向量 → 坡度 deg → 粗糙度 (局部残差 + 单元高度差)
    → safe = slope < 阈值 and roughness < 阈值.

    空洞区域不做全局均值补全: 足迹外统一标记为 danger_id 且
    semantic_valid_mask=False, 避免产生虚假的安全地面.

    Returns: (sem_map uint8, semantic_valid_mask bool, slope_deg, roughness)
    """
    import cv2

    G = bev.grid_res
    valid = bev.occupied
    height = bev.z_min

    # ── 观测足迹 (凸包) ──
    footprint = np.zeros((G, G), dtype=np.uint8)
    coords = np.column_stack(np.where(valid))[:, ::-1].astype(np.int32)
    if len(coords) >= 3:
        cv2.fillConvexPoly(footprint, cv2.convexHull(coords), 1)
    else:
        footprint[valid] = 1
    footprint = footprint.astype(bool)

    # ── 局部填洞 (仅数值连续性): 全部 NaN 单元用最近有效单元高度填充 ──
    # 距离变换输入必须为 (~valid): 零像素 = 真实有效单元 (被打 label), 非零像素
    # (空洞) 携带最近有效单元的 label. 传 valid.astype(uint8) 会把 label 打到
    # 空洞上, fill_valid_nearest 取到的是空洞组件 id, 填充结果错误 (cv2 5.0.0
    # 实测 label 语义见 utils/valid_nearest.py 的 fill_valid_nearest 注释).
    #
    # 足迹外的 NaN 单元也一并做数值填充: 高斯平滑/Sobel 卷积核会跨越 NaN
    # 区域, 把 NaN 扩散进足迹边缘 (cv2 平滑对 NaN 无保护). 填洞只服务于
    # 局部几何计算的数值连续性, 不改变真实观测掩码: safe 仍要求 valid,
    # semantic_valid_mask 仍是观测足迹, 未观测单元不会被升级为安全.
    filled = height.copy()
    if (~valid).any():
        # DIST_LABEL_PIXEL 的 label 语义因 cv2 版本而异:
        #   cv2 < 5.0  — 每个 zero(有效)像素独立 label (文档语义)
        #   cv2 >= 5.0 — PIXEL ≡ CCOMP, 按 8-连通分量打 label (5.0.0 实测)
        # fill_valid_nearest 对两种语义都取「label 组内 L2 最近有效单元」:
        # 4.x 下与紧凑索引法结果一致, 5.x 下修正分量语义偏差.
        _, labels = cv2.distanceTransformWithLabels(
            (~valid).astype(np.uint8), cv2.DIST_L2, 5, cv2.DIST_LABEL_PIXEL)
        filled = fill_valid_nearest(height, valid, labels)

    # ── 局部平滑 ──
    ksize = int(smooth_ksize)
    if ksize % 2 == 0:
        ksize += 1
    ksize = max(3, ksize)
    smoothed = cv2.GaussianBlur(filled.astype(np.float32), (ksize, ksize), 0)

    # ── Sobel 梯度 (换算为 m/m: 每像素高度 / 每像素实际米) ──
    # ksize=3 核 = [1,2,1]ᵀ ⊗ [-1,0,1] 的未归一化卷积: 垂直平滑和 4 ×
    # 中心差分因子 2 = 8× 放大, 除 8 还原每像素导数
    # (与参考 halss_original.py 的 np.gradient 真梯度口径一致, 保证
    # slope_threshold_deg 的单位含义相同)
    x_min, x_max = bev.bounds["x_min"], bev.bounds["x_max"]
    y_min, y_max = bev.bounds["y_min"], bev.bounds["y_max"]
    cell_x = max((x_max - x_min) / max(G - 1, 1), 1e-6)
    cell_y = max((y_max - y_min) / max(G - 1, 1), 1e-6)
    dx = cv2.Sobel(smoothed, cv2.CV_32F, 1, 0, ksize=3) / (8.0 * cell_x)
    dy = cv2.Sobel(smoothed, cv2.CV_32F, 0, 1, ksize=3) / (8.0 * cell_y)
    grad_mag = np.hypot(dx, dy)
    slope_deg = np.degrees(np.arctan(grad_mag))

    # 法向量 (用于诊断)
    nx, ny = -dx, -dy
    nz = np.ones_like(nx)
    norm = np.sqrt(nx ** 2 + ny ** 2 + nz ** 2) + 1e-8

    # ── 粗糙度: 局部残差均值 + 单元高度差 (高度差 NaN 视为 0) ──
    resid = np.abs(height - smoothed)
    resid = np.nan_to_num(resid, nan=0.0, posinf=0.0, neginf=0.0)
    rough_local = cv2.blur(resid.astype(np.float32), (ksize, ksize))
    zdiff_local = np.nan_to_num(bev.z_diff, nan=0.0, posinf=0.0, neginf=0.0)
    roughness = rough_local + zdiff_local

    # ── 判定 ──
    safe = (slope_deg < float(slope_threshold_deg)) & (roughness < float(roughness_threshold_m))
    safe &= valid
    sem_map = np.full((G, G), danger_id, dtype=np.uint8)
    sem_map[safe] = safe_id
    sem_map[~footprint] = danger_id
    semantic_valid_mask = footprint

    slope_out = np.full((G, G), np.nan, dtype=np.float32)
    rough_out = np.full((G, G), np.nan, dtype=np.float32)
    slope_out[footprint] = slope_deg[footprint]
    rough_out[footprint] = roughness[footprint]
    return sem_map, semantic_valid_mask, slope_out, rough_out


# ──────────────────────────────────────────────
# 语义分支封装 (脚本一/三 Bayesian, 脚本二/三几何)
# ──────────────────────────────────────────────
class BayesianSemanticBranch:
    """HALSS Bayesian UNet + MC Dropout 语义分支 (惰性导入 torch)."""

    def __init__(self, cfg: dict, obs_w: int = 128, obs_h: int = 128,
                 danger_id: int = 9):
        from perception.halss_bayesian import HALSSBayesianEvaluator
        from perception.semantic_generator import SemanticGenerator
        perc_cfg = dict(cfg.get("perception", {}))
        # 权重路径: 相对路径依次尝试原样/包根/仓库根
        wpath = perc_cfg.get("halss_weight_path")
        if wpath:
            resolved = _resolve_path_candidates(wpath)
            if resolved is None:
                logger.warning("[HALSS] 权重 %s 未找到 (尝试相对路径), 将按原样传入", wpath)
            else:
                perc_cfg["halss_weight_path"] = str(resolved)
        self.halss = HALSSBayesianEvaluator(perc_cfg)
        self.sem_gen = SemanticGenerator(
            {**perc_cfg, "img_width": obs_w, "img_height": obs_h})
        self.danger_id = int(danger_id)

    def __call__(self, bev: BevGrid, bounds: dict):
        t0 = time.perf_counter()
        result = self.halss.evaluate(bev.points, fixed_bounds=bounds)
        if result is None:
            sem_map = np.full((self.sem_gen.img_h, self.sem_gen.img_w),
                              self.danger_id, dtype=np.uint8)
        else:
            sem_map = self.sem_gen.generate(result["bev_data"])
        return sem_map, {
            "branch": "bayesian",
            "latency_ms": (time.perf_counter() - t0) * 1000.0,
        }


class GeometrySemanticBranch:
    """几何语义分支 (坡度 + 粗糙度), 纯 numpy/cv2."""

    def __init__(self, slope_threshold_deg: float, roughness_threshold_m: float,
                 safe_id: int = 1, danger_id: int = 9, smooth_ksize: int = 5):
        self.slope_th = float(slope_threshold_deg)
        self.rough_th = float(roughness_threshold_m)
        self.safe_id = int(safe_id)
        self.danger_id = int(danger_id)
        self.ksize = int(smooth_ksize)

    def __call__(self, bev: BevGrid, bounds: dict):
        t0 = time.perf_counter()
        sem_map, semantic_valid, slope_deg, roughness = geometric_semantic_map(
            bev, self.slope_th, self.rough_th,
            safe_id=self.safe_id, danger_id=self.danger_id,
            smooth_ksize=self.ksize)
        return sem_map, {
            "branch": "geometry",
            "latency_ms": (time.perf_counter() - t0) * 1000.0,
            "slope_deg_mean": float(np.nanmean(slope_deg)),
            "slope_deg_max": float(np.nanmax(slope_deg)),
            "roughness_mean_m": float(np.nanmean(roughness)),
            "sem_valid_ratio": float(np.mean(semantic_valid)),
        }


# ──────────────────────────────────────────────
# 十帧融合几何语义分支 (脚本三, 尺度明确的局部平面拟合)
# ──────────────────────────────────────────────
@dataclass
class FusedWindowResult:
    """十帧窗口融合结果 (当前帧水平机体坐标, 当前动态 ROI 裁剪).

    points:            小尺度体素去重后的融合点 (每体素最近表面点 + 垂直
                       跨度远点), 已按 cell_index 升序排列;
    projection_points: 全部窗口点原样拼接, 供深度 z-buffer 与 NN-fill
                       (不平均历史点云, 避免遮挡关系被破坏);
    valid:             (G,G) bool 真实观测掩码 (十帧观测并集);
    count:             (G,G) int32 每格点数;
    frame_count_map:   (G,G) int32 每格来源帧数;
    z_min/z_max:       (G,G) float32 每格最近表面 / 最远表面 (垂直高度跨度);
    cell_index:        (N,) int64 fused.points 每点所属格 flat 索引
                       (row*G+col), 供语义分支按格切片取点;
    frames_used:       实际入窗帧数;
    frame_span_s:      窗口时间跨度 (秒).
    """

    points: np.ndarray
    projection_points: np.ndarray
    valid: np.ndarray
    count: np.ndarray
    frame_count_map: np.ndarray
    z_min: np.ndarray
    z_max: np.ndarray
    cell_index: np.ndarray
    grid_res: int
    bounds: dict
    frames_used: int
    frame_span_s: float
    stats: dict


def _batch_robust_plane_fit(hw, valid_w, ox, oy, min_pts, iters=4):
    """批量 IRLS (Tukey bisquare) 平面拟合 z = a·ox + b·oy + c.

    hw/valid_w: (N, K) 每行一个单元的窗口观测高度与观测掩码;
    ox/oy:      (K,) 窗口单元相对偏移 (米, 各格共用同一偏移量).
    Returns (params (N,3), resid_scale (N,), support (N,) bool):
      - params: [a, b, c], 坡度 = atan(hypot(a, b));
      - resid_scale: 点到拟合平面的鲁棒残差尺度 (1.4826·MAD, 米), 即粗糙度;
      - support: 窗口内观测单元数 >= min_pts (支持充分才参与判定).
    无效单元不进入 A/b; 支持不足的行不拟合 (params=0, scale=NaN).
    """
    N, K = hw.shape
    X = np.column_stack([ox, oy, np.ones(K)]).astype(np.float32)
    h = np.where(valid_w, hw, 0.0).astype(np.float32)
    support = valid_w.sum(axis=1) >= min_pts
    params = np.zeros((N, 3), dtype=np.float32)
    scale = np.full(N, np.nan, dtype=np.float32)
    if not support.any():
        return params, scale, support
    sel = np.flatnonzero(support)
    h_s, v_s = h[sel], valid_w[sel].astype(np.float32)
    # errstate: 本机 float32 matmul SIMD 内核内部换算产生外观性除零/溢出告警
    # (与 replay_window10.body_to_world 同因, 结果正确, 仅静默).
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        p, med = _irls_plane_fit(h_s, v_s, X, K, iters)
    params[sel] = p
    scale[sel] = 1.4826 * med
    return params, scale, support


def _irls_plane_fit(h_s, v_s, X, K, iters):
    """批量 IRLS 主体: 返回 (params (N,3), 每行残差中位数 med (N,)).

    X 为共享设计矩阵 (K,3) = [ox, oy, 1]; h_s/v_s 已按支持行切片.
    性能: 本机 numpy 2.0.2 的 einsum 不走 BLAS (批量 A 矩阵 5.7×、b 3×
    慢于 matmul, 实测), 批量最小二乘全部改用显式维度的 matmul.
    本机 numpy 2.0.2 对批量 solve(A (N,3,3), b (N,3)) 还会误把 b 首维当
    核心维 (实测 (8,3,3)@(8,3) 报 size 8 != 3); 显式补尾维 (N,3,1) 后正常.
    """
    N = len(h_s)
    Xv = X[None, :, :] * v_s[:, :, None]
    A = np.matmul(Xv.transpose(0, 2, 1), Xv) + np.eye(3)[None] * 1e-6
    b = np.matmul(Xv.transpose(0, 2, 1), h_s[..., None])[..., 0]
    p = np.linalg.solve(A, b[..., None])[..., 0]
    for _ in range(iters):
        r = h_s - np.matmul(p, X.T)
        rv = np.where(v_s > 0, np.abs(r), np.inf)
        srt = np.sort(rv, axis=1)
        nv = v_s.sum(axis=1).astype(np.int32)
        med = srt[np.arange(N), np.clip(nv // 2, 0, K - 1)]
        s = np.clip(1.4826 * med, 1e-4, None)
        u = np.abs(r) / s[:, None]
        wgt = np.where(u < 4.685, (1.0 - (u / 4.685) ** 2) ** 2, 0.0) * v_s
        WX = wgt[:, :, None] * X[None, :, :]
        A = np.matmul(WX.transpose(0, 2, 1), X[None, :, :]) \
            + np.eye(3)[None] * 1e-6
        b = np.matmul(WX.transpose(0, 2, 1), h_s[..., None])[..., 0]
        p_new = np.linalg.solve(A, b[..., None])[..., 0]
        if np.allclose(p_new, p, atol=1e-5, rtol=1e-4):
            p = p_new
            break
        p = p_new
    # 最终粗糙度: 拟合后残差的鲁棒尺度 (无效/离群点不进入中位数)
    r = h_s - np.matmul(p, X.T)
    rv = np.where(v_s > 0, np.abs(r), np.inf)
    srt = np.sort(rv, axis=1)
    med = srt[np.arange(N), np.clip(nv // 2, 0, K - 1)]
    return p, med


def fused_geometric_semantic_map(fused: FusedWindowResult,
                                 slope_threshold_deg: float,
                                 roughness_threshold_m: float,
                                 prominence_threshold_m: float = 0.15,
                                 safe_id: int = 1, danger_id: int = 9,
                                 fine_radius_cells: int = 3,
                                 coarse_radius_cells: int = 15,
                                 coarse_stride: int = 2,
                                 min_support_pts: int = 8):
    """十帧融合结果 → 尺度明确的几何安全判定.

    每个候选格 (真实观测) 用周围真实观测单元 (每格最近表面 z_min) 鲁棒拟合
    z = a·x + b·y + c (Tukey bisquare IRLS, 批量):
      - 坡度   = atan(sqrt(a² + b²));
      - 粗糙度 = 点到拟合平面的鲁棒残差尺度 (1.4826·MAD, 米);
      - 突出高度 = 平面高度 - 单元最近表面 (z 向下为正, 正值表示突出).
    双尺度:
      - 细窗口 (2·fine_radius+1)² 判坡度/粗糙度/细突出 (台阶、窄柱体);
      - 粗窗口 (2·coarse_radius+1)² 隔 coarse_stride 取列, 判相对局部地面
        高度 (宽柱体/平台边缘), 同时供显示着色.
    危险突出需要邻域空间支持: 单元内 ≥2 个点高出阈值, 或相邻单元同样突出;
    单个孤立异常点不直接生成障碍 (区分草叶噪声与柱形障碍物).
    safe = 观测 & 支持充分 & 坡度<th & 粗糙度<th & 无细/粗空间连续突出.
    十帧均未观测的单元保持 unknown (semantic_valid_mask=False), 凸包最近邻
    填充不得将其升级为安全区.

    Returns (sem_map, semantic_valid_mask, maps):
      maps = {slope_deg, roughness, prominence_fine, prominence_coarse,
              rel_height, obs_count, obs_frames, confidence, valid}
    各图 (G,G), 未观测单元为 NaN (valid 掩码除外).
    """
    import cv2

    G = int(fused.grid_res)
    valid = fused.valid
    z_min = fused.z_min
    x_min, x_max = fused.bounds["x_min"], fused.bounds["x_max"]
    y_min, y_max = fused.bounds["y_min"], fused.bounds["y_max"]
    x_span = max(x_max - x_min, 1e-6)
    y_span = max(y_max - y_min, 1e-6)
    cell_x = x_span / max(G - 1, 1)
    cell_y = y_span / max(G - 1, 1)

    def _empty_map():
        return np.full((G, G), np.nan, dtype=np.float32)

    slope_out, rough_out = _empty_map(), _empty_map()
    prom_fine_out, prom_coarse_out = _empty_map(), _empty_map()
    rel_out = _empty_map()

    def _window_fit(radius, stride):
        """对全部观测单元批量拟合窗口平面, 返回 (a, b, c, resid_scale, support)."""
        R = int(radius)
        K = ((2 * R) // stride + 1) ** 2
        padded_h = np.pad(z_min, R, mode="constant", constant_values=np.nan)
        padded_v = np.pad(valid, R, mode="constant", constant_values=False)
        view_h = np.lib.stride_tricks.sliding_window_view(padded_h, (2 * R + 1, 2 * R + 1))
        view_v = np.lib.stride_tricks.sliding_window_view(
            padded_v.astype(np.uint8), (2 * R + 1, 2 * R + 1)).astype(bool)
        rows, cols = np.where(valid)
        hw = np.ascontiguousarray(view_h[rows, cols][:, ::stride, ::stride]).reshape(
            len(rows), K)
        vw = np.ascontiguousarray(view_v[rows, cols][:, ::stride, ::stride]).reshape(
            len(rows), K)
        rng = np.arange(2 * R + 1) - R
        ox = (rng[::stride] * cell_x).astype(np.float32)
        oy = (rng[::stride] * cell_y).astype(np.float32)
        # 窗口展平的 axis0=行(y), axis1=列(x); meshgrid(oy, ox, "ij") 使
        # oxy[0] 沿行变化 (y 偏移), oxy[1] 沿列变化 (x 偏移). 返回顺序必须为
        # (x 偏移, y 偏移), 与 _batch_robust_plane_fit 的 X=[ox, oy, 1] 对齐,
        # 否则 5° 斜坡会因 a/b 错位产生 ~0.15 m 的虚假突出 (实测定位).
        oxy = np.array(np.meshgrid(oy, ox, indexing="ij")).reshape(2, -1)
        return rows, cols, hw, vw, oxy[1], oxy[0]

    # ── 细尺度: 坡度 / 粗糙度 / 细突出 ──
    rows, cols, hw, vw, ox, oy = _window_fit(fine_radius_cells, 1)
    params_fine, rough_fine, support_fine = _batch_robust_plane_fit(
        hw, vw, ox, oy, min_support_pts)
    if len(rows):
        slope_out[rows, cols] = np.degrees(
            np.arctan(np.hypot(params_fine[:, 0], params_fine[:, 1])))
        rough_out[rows, cols] = rough_fine

    # ── 粗尺度: 相对局部地面高度 (粗突出, 供判定与显示) ──
    rows_c, cols_c, hw_c, vw_c, ox_c, oy_c = _window_fit(coarse_radius_cells,
                                                         coarse_stride)
    params_coarse, _, support_coarse = _batch_robust_plane_fit(
        hw_c, vw_c, ox_c, oy_c, min_support_pts)

    if len(rows):
        # 窗口以本格为原点 (局部偏移 ox/oy), 平面在本格处的高度 = 截距 c.
        # 不能把 a·xc + b·yc + c 在绝对坐标 (xc,yc ∈ ±2 m) 上求值: 噪声使
        # 拟合坡度带误差, 绝对坐标会把误差放大成虚假突出 (网格边缘实测
        # ~0.15 m 假突出, L 形截断窗口放大到 ~0.65 m). 截距 c 是本格高度
        # 的鲁棒局部估计, 与绝对坐标无关.
        z_plane_fine = params_fine[:, 2]
        z_plane_coarse = params_coarse[:, 2]
        zcell = z_min[rows, cols]
        rel_fine = z_plane_fine - zcell
        rel_coarse = z_plane_coarse - zcell
        prom_fine_out[rows, cols] = np.maximum(rel_fine, 0.0)
        prom_coarse_out[rows, cols] = np.maximum(rel_coarse, 0.0)
        rel_out[rows, cols] = rel_coarse

        # 每格自身点集中高出 (平面 - prom_th) 的点数: 突出必须有点支撑
        n_high_f = np.zeros(G * G, dtype=np.int32)
        n_high_c = np.zeros(G * G, dtype=np.int32)
        if len(fused.points):
            pf = np.asarray(fused.points, dtype=np.float32)
            cell_of = np.asarray(fused.cell_index, dtype=np.int64)
            point_valid = (
                (cell_of >= 0)
                & (cell_of < G * G)
                & np.isfinite(pf[:, :3]).all(axis=1)
            )
            pf = pf[point_valid]
            cell_of = cell_of[point_valid]
            z_th_fine = np.full(G * G, np.inf, dtype=np.float32)
            z_th_coarse = np.full(G * G, np.inf, dtype=np.float32)
            flat_idx = rows * G + cols
            z_th_fine[flat_idx] = z_plane_fine - float(prominence_threshold_m)
            z_th_coarse[flat_idx] = z_plane_coarse - float(prominence_threshold_m)
            if len(pf):
                above_f = pf[:, 2] < z_th_fine[cell_of]
                above_c = pf[:, 2] < z_th_coarse[cell_of]
                n_high_f = np.bincount(cell_of[above_f], minlength=G * G)
                n_high_c = np.bincount(cell_of[above_c], minlength=G * G)
        n_high_f = n_high_f.reshape(G, G)
        n_high_c = n_high_c.reshape(G, G)
        prom_th = float(prominence_threshold_m)
        # 突出单元: 单元高度高出局部平面超过阈值, 且单元内存在高出点
        prom_above_f = np.zeros((G, G), dtype=bool)
        prom_above_c = np.zeros((G, G), dtype=bool)
        sel_f = (rel_fine > prom_th) & (n_high_f[rows, cols] >= 1)
        sel_c = (rel_coarse > prom_th) & (n_high_c[rows, cols] >= 1)
        prom_above_f[rows[sel_f], cols[sel_f]] = True
        prom_above_c[rows[sel_c], cols[sel_c]] = True
        # 空间连续支持: 3×3 邻域内突出格计数 (含自身) >= 2
        neigh_f = cv2.filter2D(prom_above_f.astype(np.float32), -1,
                               np.ones((3, 3), np.float32))
        neigh_c = cv2.filter2D(prom_above_c.astype(np.float32), -1,
                               np.ones((3, 3), np.float32))
        # 危险突出 = 突出格本身, 且 (本格 >= 2 个突出点) 或 (邻域格同样突出).
        # 单个孤立异常点 (1 格 1 点) 不直接生成障碍; 周围草坪格不进入危险集.
        prom_danger_f = prom_above_f & ((n_high_f >= 2) | (neigh_f >= 2))
        prom_danger_c = prom_above_c & ((n_high_c >= 2) | (neigh_c >= 2))

        slope_ok = slope_out[rows, cols] < float(slope_threshold_deg)
        rough_ok = rough_out[rows, cols] < float(roughness_threshold_m)
        safe_mask = (slope_ok & rough_ok & support_fine & support_coarse
                     & ~prom_danger_f[rows, cols] & ~prom_danger_c[rows, cols])
        safe = np.zeros((G, G), dtype=bool)
        safe[rows[safe_mask], cols[safe_mask]] = True
    else:
        safe = np.zeros((G, G), dtype=bool)

    sem_map = np.full((G, G), int(danger_id), dtype=np.uint8)
    sem_map[safe] = int(safe_id)
    # 有真实点但局部平面支持不足时，几何结论尚不成立，应保持 unknown，
    # 不能因为默认 danger_id 而显示为黑色危险散点。
    classification_valid = np.zeros((G, G), dtype=bool)
    if len(rows):
        supported = support_fine & support_coarse
        classification_valid[rows[supported], cols[supported]] = True
    semantic_valid_mask = valid & classification_valid

    frames = fused.frame_count_map.astype(np.float32)
    confidence = np.full((G, G), np.nan, dtype=np.float32)
    confidence[valid] = np.minimum(frames[valid] / 3.0, 1.0)
    maps = {
        "slope_deg": slope_out,
        "roughness": rough_out,
        "prominence_fine": prom_fine_out,
        "prominence_coarse": prom_coarse_out,
        "rel_height": rel_out,
        "obs_count": fused.count.astype(np.float32),
        "obs_frames": frames,
        "confidence": confidence,
        "valid": valid,
        "semantic_valid": semantic_valid_mask,
    }
    return sem_map, semantic_valid_mask, maps


class FusedGeometrySemanticBranch:
    """十帧融合几何语义分支: 鲁棒局部平面拟合 (坡度/粗糙度/突出高度).

    不调用 HALSS Bayesian; 输入为 fuse_window 的结构化融合结果.
    """

    def __init__(self, slope_threshold_deg: float, roughness_threshold_m: float,
                 prominence_threshold_m: float = 0.15,
                 safe_id: int = 1, danger_id: int = 9,
                 fine_radius_cells: int = 3, coarse_radius_cells: int = 15,
                 min_support_pts: int = 8):
        self.slope_th = float(slope_threshold_deg)
        self.rough_th = float(roughness_threshold_m)
        self.prom_th = float(prominence_threshold_m)
        self.safe_id = int(safe_id)
        self.danger_id = int(danger_id)
        self.fine_radius = int(fine_radius_cells)
        self.coarse_radius = int(coarse_radius_cells)
        self.min_support = int(min_support_pts)

    def __call__(self, fused: FusedWindowResult):
        t0 = time.perf_counter()
        sem_map, semantic_valid, maps = fused_geometric_semantic_map(
            fused, self.slope_th, self.rough_th,
            prominence_threshold_m=self.prom_th,
            safe_id=self.safe_id, danger_id=self.danger_id,
            fine_radius_cells=self.fine_radius,
            coarse_radius_cells=self.coarse_radius,
            min_support_pts=self.min_support)

        def _finite_stat(values, reducer):
            arr = np.asarray(values, dtype=np.float32)
            finite = arr[np.isfinite(arr)]
            return float(reducer(finite)) if finite.size else float("nan")

        return sem_map, {
            "branch": "fused_geometry",
            "latency_ms": (time.perf_counter() - t0) * 1000.0,
            # 稀疏启动帧可能没有足够邻域支持，诊断图会全 NaN。显式检查
            # 有限值，避免 np.nanmean/nanmax 在 Orin 上刷 RuntimeWarning。
            "slope_deg_mean": _finite_stat(maps["slope_deg"], np.mean),
            "slope_deg_max": _finite_stat(maps["slope_deg"], np.max),
            "roughness_mean_m": _finite_stat(maps["roughness"], np.mean),
            "sem_valid_ratio": float(np.mean(semantic_valid)),
            "safe_ratio": float(np.mean(sem_map == self.safe_id)),
            "maps": maps,
        }


# ──────────────────────────────────────────────
# 深度分支 (与 pipeline.py / replay_bag_offline.py 完全一致)
# ──────────────────────────────────────────────
def render_sparse_depth(sparse_depth, valid_mask, dmax, min_valid=5, median_ksize=5):
    """NN-fill 深度补全: 距离变换最近有效像素填充 + 中值平滑.

    填充只取真实观测像素的值 (valid_mask), 纯补全深度不参与任何安全语义
    (安全语义只由几何/Bayesian 分支的真实点云判定, 本函数仅服务 DRL 输入).
    """
    import cv2
    if valid_mask.sum() < min_valid:
        return np.where(valid_mask, sparse_depth, dmax).astype(np.float32)
    _, labels = cv2.distanceTransformWithLabels(
        (~valid_mask).astype(np.uint8), distanceType=cv2.DIST_L2, maskSize=5,
        labelType=cv2.DIST_LABEL_PIXEL)
    # 零像素 = 真实有效像素 (被打 label), 无效像素携带最近有效像素的 label.
    # 直接用 valid_coords[label-1] 紧凑索引在 cv2>=5.0 (label=8-连通分量) 下
    # 会错选分量内栅格序第一个像素; fill_valid_nearest 对两种 label 语义都取
    # 「label 组内 L2 最近」, 4.x 下输出与原紧凑索引法逐位一致.
    filled = fill_valid_nearest(sparse_depth, valid_mask, labels)
    if median_ksize >= 3:
        smoothed = cv2.medianBlur(filled.astype(np.float32), int(median_ksize))
    else:
        smoothed = filled.astype(np.float32)
    rendered = np.where(valid_mask, sparse_depth, smoothed)
    return np.clip(rendered, 0.0, dmax).astype(np.float32)


def render_depth_fixed_gray(depth_m, vmax_m=30.0):
    """固定量程灰度显示: 0 m→黑, vmax m→白 (纯显示, 不改输入)."""
    import cv2
    depth = np.nan_to_num(
        np.asarray(depth_m, dtype=np.float32),
        nan=vmax_m, posinf=vmax_m, neginf=0.0,
    )
    norm = np.clip(depth / max(float(vmax_m), 1e-6), 0.0, 1.0)
    return cv2.cvtColor(np.round(norm * 255.0).astype(np.uint8), cv2.COLOR_GRAY2BGR)


def render_bev_bgr(bev: BevGrid, vmax_m: float = None) -> np.ndarray:
    """BEV 高度场 (z_min) inferno 着色, 空单元黑色. 用于左侧处理点云 BEV 窗口."""
    import cv2
    height = bev.z_min
    valid = bev.occupied
    if vmax_m is None:
        vmax_m = float(np.nanmax(height)) if valid.any() else 30.0
    vmax_m = max(float(vmax_m), 1e-3)
    img = np.zeros((bev.grid_res, bev.grid_res), dtype=np.float32)
    img[valid] = np.clip(height[valid] / vmax_m, 0.0, 1.0)
    return cv2.applyColorMap(np.round(img * 255.0).astype(np.uint8), cv2.COLORMAP_INFERNO)


def render_rel_height_bgr(rel_height: np.ndarray, valid: np.ndarray,
                          half_range_m: float = 0.3) -> np.ndarray:
    """相对局部地面高度固定量程着色 (发散色带, 与逐帧归一化的绝对高度区分).

    蓝 = 低于局部地面, 近黑 = 与局部地面齐平, 红 = 突出 (柱体/台阶).
    量程固定 (±half_range_m), 不会随帧内最大值逐帧变化, 避免把相近颜色
    误解为已经证明平坦; 无效/NaN 单元黑色.
    """
    import cv2
    half = max(float(half_range_m), 1e-3)
    img = np.zeros(rel_height.shape[:2], dtype=np.uint8)
    m = np.asarray(valid, dtype=bool) & np.isfinite(rel_height)
    if m.any():
        v = np.clip(rel_height[m] / half, -1.0, 1.0)  # -1..1
        img[m] = ((v * 127.5) + 127.5).astype(np.uint8)  # 0..255
    # OpenCV 自定义色表布局必须是 (256, 1, 3)，最后一维才是 BGR。
    # 旧实现用 lut[i, channel] 索引，把 channel 错放在 size=1 的轴上，
    # 因此在 channel=1 时触发 IndexError。
    lut = np.zeros((256, 1, 3), dtype=np.uint8)
    for i in range(256):
        t = i / 255.0
        below = np.clip((0.5 - t) / 0.5, 0.0, 1.0) ** 0.8
        above = np.clip((t - 0.5) / 0.5, 0.0, 1.0) ** 1.1
        base = 18.0
        lut[i, 0, 0] = int(base + below * 215.0 + above * 20.0)       # B
        lut[i, 0, 1] = int(base + above * 60.0 + below * 45.0)        # G
        lut[i, 0, 2] = int(base + above * 190.0 + below * 18.0)       # R
    out = cv2.applyColorMap(img, lut)
    out[~m] = 0
    return out


def render_diag_map(values: np.ndarray, valid: np.ndarray,
                    vmin: float, vmax: float) -> np.ndarray:
    """诊断图 (坡度/粗糙度/突出高度/观测数): inferno 着色, 无效/NaN 黑色."""
    import cv2
    m = np.asarray(valid, dtype=bool) & np.isfinite(values)
    img = np.zeros(values.shape[:2], dtype=np.float32)
    if m.any():
        img[m] = np.clip((values[m] - float(vmin)) / max(float(vmax) - float(vmin), 1e-6),
                         0.0, 1.0)
    return cv2.applyColorMap(np.round(img * 255.0).astype(np.uint8),
                             cv2.COLORMAP_INFERNO)


def project_depth(points, sem_map, bounds, camera, danger_id: int = 9,
                  fill_unobserved: bool = True,
                  semantic_bev_valid=None,
                  semantic_fill_radius_px: float = 0.0):
    """training-camera 深度投影 + HALSS 标签采样 (与 replay_bag_offline 同一函数).

    fill_unobserved=False 时启用保守模式: 未被射线命中的投影像素保持
    danger_id 且 semantic_valid_mask=False (禁止凸包最近邻填充把未观测区域
    升级为安全区). 十帧回放默认启用该保守模式; 单帧脚本保持默认 True.
    Returns: sparse_depth, valid_mask, sem_map, semantic_valid_mask
    """
    from perception.training_camera_projection import project_training_camera
    return project_training_camera(points, sem_map, bounds, camera,
                                   danger_id=danger_id,
                                   fill_unobserved=fill_unobserved,
                                   semantic_bev_valid=semantic_bev_valid,
                                   semantic_fill_radius_px=semantic_fill_radius_px)


# ──────────────────────────────────────────────
# ONNX DRL (与 pipeline.py / replay_bag_offline.py 完全一致)
# ──────────────────────────────────────────────
CLASS_TO_GRAY = {
    -1: 0, 0: 10, 1: 30, 2: 60, 3: 70, 4: 20,
     5: 40, 6: 80, 7: 90, 8: 50, 9: 250,
}


def make_binary_semantic_vis(sem_map, safe_id=1, danger_id=9):
    sem_vis = np.full(sem_map.shape, 128, dtype=np.uint8)
    sem_vis[sem_map == safe_id] = 255
    sem_vis[sem_map == danger_id] = 0
    return sem_vis


def top3_str(probs, action_names=None, k=3):
    if probs is None:
        return "p=n/a"
    pairs = sorted(enumerate(probs), key=lambda item: item[1], reverse=True)[:k]
    if action_names is None:
        return "p=" + ",".join(f"{idx}:{prob:.2f}" for idx, prob in pairs)
    return "p=" + ",".join(f"{action_names[idx]}:{prob:.2f}" for idx, prob in pairs)


class ONNXDRL:
    """ONNX PPO 推理: 输入 128×128×2 原值 (图内 /255), 输出 10 类 logits."""

    def __init__(self, onnx_path: str, obs_h=128, obs_w=128, dmax=30.0,
                 depth_norm_mode="raw_meters_graph_scaled",
                 semantic_norm_mode="raw_gray_graph_scaled"):
        import onnxruntime as ort
        resolved = _resolve_path_candidates(onnx_path)
        if resolved is None:
            raise FileNotFoundError(f"ONNX model not found: {onnx_path}")
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        opts.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(resolved), opts)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        in_shape = self.session.get_inputs()[0].shape
        self.layout = "chw" if len(in_shape) == 4 and in_shape[1] in (2, 3) else "hwc"
        self.obs_h, self.obs_w = obs_h, obs_w
        self.dmax = dmax
        if depth_norm_mode != "raw_meters_graph_scaled":
            raise ValueError("ONNX graph already contains input/truediv")
        if semantic_norm_mode != "raw_gray_graph_scaled":
            raise ValueError("ONNX graph already contains input/truediv")
        dummy = np.zeros((1, obs_h, obs_w, 2), dtype=np.float32)
        self._forward(dummy)
        logger.info("[ONNX] model=%s layout=%s warmup=OK", resolved, self.layout)

    def _forward(self, obs_raw):
        if self.layout == "chw":
            inp = np.transpose(obs_raw, (0, 3, 1, 2)).astype(np.float32)
        else:
            inp = obs_raw.astype(np.float32)
        return self.session.run([self.output_name], {self.input_name: inp})[0]

    def predict(self, depth_map, sem_map):
        depth_clipped = np.clip(
            np.nan_to_num(depth_map, nan=self.dmax, posinf=self.dmax, neginf=0.0),
            0.0, self.dmax,
        )
        depth_ch = depth_clipped.astype(np.float32)
        sem_int = np.clip(sem_map, -1, 9).astype(np.int16)
        sem_ch = np.zeros_like(sem_int, dtype=np.float32)
        for class_id, gray_val in CLASS_TO_GRAY.items():
            sem_ch[sem_int == class_id] = float(gray_val)
        obs = np.expand_dims(np.stack([depth_ch, sem_ch], axis=-1), axis=0)
        logits = self._forward(obs)[0]
        logits = np.asarray(logits, dtype=np.float32)
        exps = np.exp(logits - float(np.max(logits)))
        probs = exps / max(float(np.sum(exps)), 1e-12)
        action = int(np.argmax(logits))
        return action, {
            "softmax_probs": probs.astype(float).tolist(),
            "action_probs": probs.astype(float).tolist(),
            "confidence": float(np.max(probs)),
            "logits": logits.astype(float).tolist(),
        }


# ──────────────────────────────────────────────
# 三窗口可视化 (左 BEV / 中语义 / 右深度) + 可选原始点云 3D 对比
# ──────────────────────────────────────────────
def sem_crop_bounds(binary_vis):
    """语义图非灰色 (≠128) 包围盒裁剪范围, 三窗口共用同一空间对齐.

    返回 (r_min, r_max, c_min, c_max) 或 None (全灰色, 不裁剪).
    """
    valid = np.asarray(binary_vis) != 128
    if not valid.any():
        return None
    rows = np.any(valid, axis=1)
    cols = np.any(valid, axis=0)
    r_min, r_max = np.where(rows)[0][[0, -1]]
    c_min, c_max = np.where(cols)[0][[0, -1]]
    h, w = valid.shape[:2]
    pad = 2
    r_min, r_max = max(0, r_min - pad), min(h - 1, r_max + pad)
    c_min, c_max = max(0, c_min - pad), min(w - 1, c_max + pad)
    return int(r_min), int(r_max), int(c_min), int(c_max)


def build_three_window_displays(bev_bgr, sem_vis, depth_bgr,
                                rotate_bev: bool = True):
    """三窗口共享显示链: BEV 逆时针 90° → 最近邻缩放至语义尺寸 → 共享裁剪.

    坐标约定 (机体 +x 向前 / +y 向左 / z 向下):
      - BEV 网格行 0 = +y (顶), 列随 +x 增大 (右) — 显示前逆时针旋转 90°,
        使 +x (机头) 朝上、+y (机体左) 朝左, 与相机投影方向一致;
      - 语义/深度图保持 training-camera 投影方向不变;
      - 旋转后的 BEV 先最近邻缩放到语义/深度尺寸 (修复旧版用 128×128 语义
        裁剪坐标直接切 64×64 BEV 的错位), 再执行三窗口共享裁剪;
      - 纯显示层: 不改 BEV 数据与模型输入.

    sem_vis: 语义灰度图 (128=unknown, 裁剪锚点); bev_bgr/depth_bgr 为 BGR.
    Returns (bev_disp, sem_disp, depth_disp): 三者同尺寸, 像素位置一一对应.
    """
    import cv2
    sem_vis = np.asarray(sem_vis)
    sem_bgr = cv2.cvtColor(sem_vis.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    bounds = sem_crop_bounds(sem_vis)
    imgs = [bev_bgr, sem_bgr, depth_bgr]
    if rotate_bev and imgs[0] is not None:
        imgs[0] = cv2.rotate(imgs[0], cv2.ROTATE_90_COUNTERCLOCKWISE)
    h, w = sem_bgr.shape[:2]
    if imgs[0] is not None and imgs[0].shape[:2] != (h, w):
        imgs[0] = cv2.resize(imgs[0], (w, h),
                             interpolation=cv2.INTER_NEAREST)
    if bounds is not None:
        r_min, r_max, c_min, c_max = bounds
        imgs = [img[r_min:r_max + 1, c_min:c_max + 1] for img in imgs]
    return tuple(imgs)


def draw_frame_markers(img, text="", top_label="+x", left_label="+y",
                       color=(255, 255, 255)):
    """在窗口图上叠加方向标记: 顶部 top_label (机头向前 +x), 左侧 left_label
    (机体左 +y). 纯显示函数 (不改数据/模型输入), 供三窗口共用与合成测试
    逐像素断言; text 为可选主文本 (黑描边白字).
    """
    import cv2
    out = np.asarray(img).copy()
    h, w = out.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    if text:
        cv2.putText(out, text, (5, 16), font, 0.45, (0, 0, 0), 3)
        cv2.putText(out, text, (5, 16), font, 0.45, color, 1)
    if top_label:
        x0 = max(2, w // 2 - 10)
        cv2.putText(out, top_label, (x0, 16), font, 0.45, (0, 0, 0), 3)
        cv2.putText(out, top_label, (x0, 16), font, 0.45, color, 1)
    if left_label:
        y0 = max(16, h // 2 + 4)
        cv2.putText(out, left_label, (4, y0), font, 0.45, (0, 0, 0), 3)
        cv2.putText(out, left_label, (4, y0), font, 0.45, color, 1)
    return out


def draw_status_footer(img, text="", color=(235, 235, 235)):
    """Append a status strip below an image, keeping pixels in the image clear.

    The three replay windows show dense BEV/semantic/depth content; drawing the
    frame/action string over the top-left corner hides exactly the details that
    are useful for judging a landing scene.  Keep the image untouched and add
    one or two wrapped status lines in a dark footer instead.
    """
    import cv2
    image = np.asarray(img).copy()
    if not text:
        return image
    h, w = image.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.38
    thickness = 1
    max_width = max(20, w - 10)
    words = str(text).split()
    lines, line = [], ""
    for word in words:
        candidate = word if not line else f"{line} {word}"
        width = cv2.getTextSize(candidate, font, scale, thickness)[0][0]
        if line and width > max_width:
            lines.append(line)
            line = word
        else:
            line = candidate
    if line:
        lines.append(line)
    # Bound the strip for unusual diagnostic strings while retaining the tail.
    lines = lines[:2]
    footer_h = 7 + 17 * len(lines)
    footer = np.zeros((footer_h, w, image.shape[2]), dtype=image.dtype)
    out = np.vstack((image, footer))
    for i, line in enumerate(lines):
        baseline = h + 14 + i * 17
        cv2.putText(out, line, (5, baseline), font, scale, (0, 0, 0), 3,
                    cv2.LINE_AA)
        cv2.putText(out, line, (5, baseline), font, scale, color, thickness,
                    cv2.LINE_AA)
    return out


def render_depth_local_gray(dense_depth, valid_mask, dmax=30.0,
                            p_low=2.0, p_high=98.0, min_span_m=0.5,
                            min_valid=50):
    """近地深度局部自适应灰度显示 (右窗口默认).

    - 从真实投影射线 valid_mask 统计深度, 2%~98% 分位数作为显示范围;
    - 最小显示跨度 min_span_m (默认 0.5 m), 使柱体与地面的小深度差明显呈现;
    - 深度补全只在真实深度射线的凸包内显示, 凸包外保持黑色 (避免 NN-fill
      把整幅画面填成几乎相同灰度);
    - 有效点不足 min_valid 时回退固定 0~dmax.
    纯显示: 不改动 dense_depth 输入 (ONNX 仍接收原始米制数组).
    Returns (bgr, (near_m, far_m)) 显示范围.
    """
    import cv2
    depth = np.asarray(dense_depth, dtype=np.float32)
    valid = np.asarray(valid_mask, dtype=bool)
    dmax = float(dmax)
    near_m, far_m = 0.0, dmax
    if valid.sum() >= max(1, int(min_valid)):
        vals = depth[valid]
        lo, hi = np.percentile(vals, [float(p_low), float(p_high)])
        span = max(float(hi) - float(lo), float(min_span_m))
        center = (float(hi) + float(lo)) * 0.5
        near_m = max(0.0, center - span * 0.5)
        far_m = center + span * 0.5

    hull = np.zeros(depth.shape[:2], dtype=np.uint8)
    coords = np.column_stack(np.where(valid))[:, ::-1].astype(np.int32)
    if len(coords) >= 3:
        cv2.fillConvexPoly(hull, cv2.convexHull(coords), 1)
    else:
        hull[valid] = 1
    hull = hull.astype(bool)

    span = max(far_m - near_m, 1e-6)
    norm = np.zeros_like(depth)
    norm[hull] = np.clip((depth[hull] - near_m) / span, 0.0, 1.0)
    gray = np.round(norm * 255.0).astype(np.uint8)
    bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    return bgr, (float(near_m), float(far_m))


class MplScatterWindow:
    """Matplotlib 3D 散点窗口 (惰性导入, 与 replay_raw_livox_visualization.py 生命周期一致)."""

    def __init__(self, window_title: str, axes_labels: tuple, show: bool = True):
        self._mpl = None
        self._fig = None
        self._ax = None
        self._scatter = None
        self._window_title = window_title
        self._axes_labels = axes_labels
        if not show:
            return
        try:
            import matplotlib
            matplotlib.use("TkAgg")
            import matplotlib.pyplot as plt
            import mpl_toolkits.mplot3d  # noqa: F401
            self._mpl = plt
        except Exception as exc:
            logger.warning("[Vis] Matplotlib unavailable (%s); 3D window %s "
                           "disabled.", exc, window_title)
            self._mpl = None

    def _init_window(self):
        self._mpl.ion()
        fig = self._mpl.figure(num=self._window_title, figsize=(6, 6))
        ax = fig.add_subplot(111, projection="3d")
        ax.set_xlabel(self._axes_labels[0])
        ax.set_ylabel(self._axes_labels[1])
        ax.set_zlabel(self._axes_labels[2])
        ax.set_title("0 pts")
        fig.canvas.draw_idle()
        fig.canvas.flush_events()
        self._fig, self._ax = fig, ax

    def update(self, pts, stamp=None):
        if self._mpl is None:
            return
        if self._fig is None:
            try:
                self._init_window()
            except Exception as exc:
                logger.warning("[Vis] 3D window init failed (%s); disabled.", exc)
                self._mpl = None
                return
        if self._ax is None:
            return
        if self._scatter is not None:
            self._scatter.remove()
            self._scatter = None
        n = 0
        if pts is not None:
            arr = np.asarray(pts, dtype=np.float32)
            if arr.ndim == 2 and arr.shape[1] >= 3 and len(arr) > 0:
                arr = arr[:, :3]
                n = len(arr)
                self._scatter = self._ax.scatter(
                    arr[:, 0], arr[:, 1], arr[:, 2],
                    c=arr[:, 2], cmap="inferno", s=1.0, depthshade=False)
        title = f"{n} pts"
        if stamp is not None:
            title += f"  ·  t={float(stamp):.3f}s"
        self._ax.set_title(title)
        try:
            self._fig.canvas.draw_idle()
            self._fig.canvas.flush_events()
        except Exception as exc:
            logger.warning("[Vis] Window closed or unavailable (%s); disabled.", exc)
            self._mpl = None

    def close(self):
        if self._fig is not None:
            try:
                self._mpl.close(self._fig)
            except Exception:
                pass
            self._fig = None
            self._ax = None
            self._scatter = None


class CompareVisualizer:
    """三窗口: 1.Processed BEV (左) / 2.Semantic Map (中) / 3.Depth (NN-fill) (右).

    坐标统一 (机体 +x 向前 / +y 向左 / z 向下): 三个窗口统一显示为
    "机头向上、机体左侧在画面左边" — BEV 显示前逆时针旋转 90° (只旋转显示
    图, 不改 BEV 数据/模型输入), 语义/深度保持 training-camera 投影方向;
    三窗口共用同一裁剪框 (以语义图非灰色包围盒为准), 标题标明 FWD ↑ / LEFT ←,
    每帧叠加 +x (顶部) / +y (左侧) 方向标记.

    可选 Matplotlib 3D 窗口: Raw Livox 原始点云 + Deskewed Body 去畸变点云
    (--show-raw-compare 开启; --no-pointcloud 关闭 3D).
    """

    WINDOW_BEV = "1.Processed BEV [FWD ↑ / LEFT ←]"
    WINDOW_SEM = "2.Semantic Map [FWD ↑ / LEFT ←]"
    WINDOW_DEPTH = "3.Depth (NN-fill) [FWD ↑ / LEFT ←]"
    STATUS_FOOTER_H = 41

    def __init__(self, dmax=30.0, display_width=300, display_height=300,
                 show_pointcloud=True, show_raw_compare=False):
        self.dmax = float(dmax)
        self.display_width = int(display_width)
        self.display_height = int(display_height)
        self.show_pointcloud = bool(show_pointcloud)
        self.show_raw_compare = bool(show_raw_compare) and self.show_pointcloud
        self._windows_ready = False
        self._mpl_windows = []
        if self.show_raw_compare:
            self._mpl_windows = [
                MplScatterWindow(
                    "4.Raw Livox",
                    ("x (m) — raw Livox frame", "y (m) — raw Livox frame",
                     "z (m) — raw Livox frame")),
                MplScatterWindow(
                    "5.Deskewed Body",
                    ("x forward (m)", "y lateral (m)", "z down (m)")),
            ]

    def _init_windows(self):
        import cv2
        for name in (self.WINDOW_BEV, self.WINDOW_SEM, self.WINDOW_DEPTH):
            cv2.namedWindow(name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(name, self.display_width,
                             self.display_height + self.STATUS_FOOTER_H)
        cv2.moveWindow(self.WINDOW_BEV, 20, 50)
        cv2.moveWindow(self.WINDOW_SEM, 20 + self.display_width + 10, 50)
        cv2.moveWindow(self.WINDOW_DEPTH, 20 + 2 * (self.display_width + 10), 50)
        self._windows_ready = True

    @staticmethod
    def _crop_bounds(binary_vis):
        """语义图非灰色 (≠128) 包围盒裁剪, 三窗口共用同一空间对齐."""
        return sem_crop_bounds(binary_vis)

    def update(self, bev_bgr, sem_map, semantic_valid_mask, depth_map, text="",
               depth_bgr=None, depth_text="", rotate_bev=True):
        import cv2
        if not self._windows_ready:
            self._init_windows()
        dw, dh = self.display_width, self.display_height

        sem_vis = make_binary_semantic_vis(sem_map)
        if semantic_valid_mask is not None:
            sem_vis = sem_vis.copy()
            sem_vis[~np.asarray(semantic_valid_mask, dtype=bool)] = 128
        # depth_bgr: 调用方预渲染的右窗口图 (如局部自适应灰度 + near/far 标注),
        # 缺省回退固定量程灰度
        if depth_bgr is None:
            depth_bgr = render_depth_fixed_gray(depth_map, vmax_m=self.dmax)
        bev_disp, sem_disp, depth_disp = build_three_window_displays(
            bev_bgr, sem_vis, depth_bgr, rotate_bev=rotate_bev)
        disp = []
        labels = [(bev_disp, text), (sem_disp, text),
                  (depth_disp, depth_text or text)]
        for img, label in labels:
            img = cv2.resize(img, (dw, dh), interpolation=cv2.INTER_NEAREST)
            # Direction markers remain on the image; status text is appended
            # below it so it cannot obscure terrain/semantic/depth pixels.
            img = draw_frame_markers(img, text="")
            img = draw_status_footer(img, text=label)
            disp.append(img)
        cv2.imshow(self.WINDOW_BEV, disp[0])
        cv2.imshow(self.WINDOW_SEM, disp[1])
        cv2.imshow(self.WINDOW_DEPTH, disp[2])
        cv2.waitKey(1)

    def update_3d(self, raw_pts=None, raw_stamp=None, deskewed_pts=None,
                  deskewed_stamp=None):
        if not self._mpl_windows:
            return
        self._mpl_windows[0].update(raw_pts, raw_stamp)
        self._mpl_windows[1].update(deskewed_pts, deskewed_stamp)

    def update_diag(self, maps: dict, text="", rotate: bool = True):
        """可选诊断窗口 (坡度/粗糙度/突出/观测数等), 惰性创建.

        maps: {窗口标题: BGR 图}. 仅在调用时显示, 不影响主三窗口.
        rotate: BEV 约定网格图 (行 0 = +y) 显示前逆时针旋转 90°, 与主
        三窗口的 FWD ↑ / LEFT ← 方向一致 (融合掩码/坡度/粗糙度诊断图同规则).
        """
        import cv2
        if not getattr(self, "_diag_windows", None):
            self._diag_windows = {}
            x0 = 20 + 3 * (self.display_width + 10)
            for k, name in enumerate(maps):
                cv2.namedWindow(name, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(name, self.display_width,
                                 self.display_height + self.STATUS_FOOTER_H)
                cv2.moveWindow(
                    name, x0,
                    50 + k * (self.display_height + self.STATUS_FOOTER_H + 10))
                self._diag_windows[name] = True
        for name, img in maps.items():
            if img is not None:
                img = img.copy()
                if rotate:
                    img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
                img = draw_status_footer(img, text=text)
                cv2.imshow(name, img)
        cv2.waitKey(1)

    def close(self):
        import cv2
        for w in self._mpl_windows:
            w.close()
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass


# ──────────────────────────────────────────────
# 每帧结果保存
# ──────────────────────────────────────────────
class FrameSaver:
    """<save-dir>/<strategy>/frame_XXXX.npz + summary.csv.

    --save-raw-arrays 额外保存 raw_points / deskewed_points / processed_points /
    fused_points (大数组), 其余小数组恒保存.
    """

    CSV_HEADER = ("frame_idx,cloud_stamp,pose_stamp,raw_stamp,sync_ms,cloud_seq,"
                  "action_id,n_raw,n_deskewed,n_processed,window_frames,"
                  "depth_valid_ratio,sem_safe_ratio,onnx_ms,total_ms")

    def __init__(self, save_dir: str, strategy: str):
        self.dir = Path(save_dir) / strategy
        self.dir.mkdir(parents=True, exist_ok=True)
        self.strategy = strategy
        self._csv_path = self.dir / "summary.csv"
        with open(self._csv_path, "w", encoding="utf-8") as f:
            f.write(self.CSV_HEADER + "\n")

    def save(self, frame_idx: int, data: dict, save_raw_arrays: bool = False,
             summary: dict = None) -> None:
        arrays = {}
        # 十帧融合分支的观测/几何诊断图: 坡度图、粗糙度图、突出高度图、
        # 相对高度图、观测掩码、来源帧数、每格点数、置信度、融合点数
        for key in ("sparse_depth", "dense_depth", "sem_map", "binary_semantic",
                    "valid_mask", "semantic_valid_mask", "pose", "action_probs",
                    "obs_valid_mask", "obs_point_count", "obs_frame_count",
                    "slope_map", "roughness_map", "prominence_map",
                    "rel_height_map", "confidence_map",
                    "observed_mask", "inferred_mask", "unknown_mask",
                    "current_mask", "history_fill_mask",
                    "anchor_mask", "conflict_mask", "surface_map",
                    "anchor_surface"):
            if key in data and data[key] is not None:
                arrays[key] = np.asarray(data[key])
        for key in ("cloud_stamp", "pose_stamp", "raw_stamp", "sync_ms",
                    "cloud_seq", "action_id", "fused_count",
                    "reg_accepted", "reg_resid_before", "reg_resid_after",
                    "added_cells", "dup_skipped_cells", "fused_coverage"):
            if key in data and data[key] is not None:
                arrays[key] = np.asarray(data[key], dtype=np.float32)
        if save_raw_arrays:
            for key in ("raw_points", "deskewed_points", "processed_points",
                        "fused_points"):
                if key in data and data[key] is not None:
                    arrays[key] = np.asarray(data[key], dtype=np.float32)
        np.savez(self.dir / f"frame_{frame_idx:04d}.npz", **arrays)

        s = summary or {}
        row = (f"{frame_idx},{s.get('cloud_stamp', '')},{s.get('pose_stamp', '')},"
               f"{s.get('raw_stamp', '')},{s.get('sync_ms', '')},"
               f"{s.get('cloud_seq', '')},{s.get('action_id', '')},"
               f"{s.get('n_raw', '')},{s.get('n_deskewed', '')},"
               f"{s.get('n_processed', '')},{s.get('window_frames', '')},"
               f"{s.get('depth_valid_ratio', '')},{s.get('sem_safe_ratio', '')},"
               f"{s.get('onnx_ms', '')},{s.get('total_ms', '')}")
        with open(self._csv_path, "a", encoding="utf-8") as f:
            f.write(row + "\n")


# ──────────────────────────────────────────────
# 终端输出 (按规格样例格式)
# ──────────────────────────────────────────────
def print_frame_block(strategy: str, frame_idx: int, *, n_raw=None, n_deskewed,
                      n_processed, window_frames=None, sync_ms, depth_valid_ratio,
                      sem_safe_ratio, action_id, action_name, probs, action_names,
                      onnx_ms, total_ms, cloud_source="fastlio") -> None:
    """打印一帧结果块 (与规格示例一致)."""
    print(f"[strategy={strategy} source={cloud_source} frame={frame_idx:04d}]")
    if n_raw is None:
        print("raw=n/a")
    else:
        print(f"raw=/livox/lidar: {n_raw}")
    print(f"deskewed_source={cloud_source}: {n_deskewed}")
    print(f"processed: {n_processed}")
    if window_frames is not None:
        print(f"window={window_frames} sync={sync_ms:.0f}ms")
    else:
        print(f"sync={sync_ms:.0f}ms")
    print(f"depth_valid={depth_valid_ratio:.2f} sem_safe={sem_safe_ratio:.2f}")
    print(f"ACTION={action_id}({action_name})")
    if probs is not None:
        pairs = sorted(enumerate(probs), key=lambda item: item[1], reverse=True)[:3]
        top3 = " ".join(f"{action_names[i]}:{p:.2f}" for i, p in pairs)
    else:
        top3 = "n/a"
    print(f"top3={top3}")
    print(f"onnx={onnx_ms:.0f}ms total={total_ms:.0f}ms")


def log_summary(strategy: str, frames_processed: int, elapsed_s: float,
                action_names, action_counts) -> None:
    logger.info("=" * 60)
    logger.info("Replay summary [%s]:", strategy)
    logger.info("  Total frames: %d", frames_processed)
    logger.info("  Elapsed: %.1fs", elapsed_s)
    if frames_processed > 0:
        logger.info("  Avg fps: %.1f", frames_processed / max(elapsed_s, 0.01))
    total_acts = sum(action_counts)
    if total_acts > 0:
        logger.info("  Action distribution:")
        for i, cnt in enumerate(action_counts):
            if cnt > 0:
                logger.info("    %s: %d (%.1f%%)", action_names[i], cnt,
                            100.0 * cnt / total_acts)
    logger.info("=" * 60)


# ──────────────────────────────────────────────
# 公共 CLI 参数
# ──────────────────────────────────────────────
def make_common_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--bag", type=str, required=True,
                        help="输入的 rosbag 路径 (experiments/*/input.bag)")
    parser.add_argument("--config", type=str, required=True,
                        help="实验配置路径 (experiment_config_snapshot.yaml)")
    parser.add_argument("--cloud-topic", type=str, default="/cloud_registered_body",
                        help="主输入去畸变点云话题 (默认 /cloud_registered_body)")
    parser.add_argument("--cloud-source", type=str, default="fastlio",
                        choices=["fastlio", "raw_imu"],
                        help="点云源: fastlio=/cloud_registered_body (默认), "
                             "raw_imu=/livox/lidar + /livox/imu 去畸变")
    parser.add_argument("--pose-topic", type=str, default=None,
                        help="位姿话题; 默认自动选择 "
                             "(/mavros/local_position/odom → /ali_odom → /Odometry)")
    parser.add_argument("--raw-topic", type=str, default="/livox/lidar",
                        help="原始 Livox 参考话题 (默认 /livox/lidar, 仅统计/可视化)")
    parser.add_argument("--imu-topic", type=str, default="/livox/imu",
                        help="raw_imu 模式的 IMU 话题 (默认 /livox/imu)")
    parser.add_argument("--onnx-model", type=str, default="weights/ppo2_policy.onnx",
                        help="ONNX DRL 策略模型路径")
    parser.add_argument("--dmax", type=float, default=None,
                        help="深度最大距离 (米), 默认取配置 max_range")
    parser.add_argument("--ground-z", type=float, default=None,
                        help="手动指定地面 Z (米); 缺省由起飞前稳定 odom 自动"
                             "估计 (本 bag 预期约 3.47 m), 未稳定时回退首帧 "
                             "pose.z; 不再信任快照 mission_state.ground_z_ref_m "
                             "(陈旧值 0.0)")
    parser.add_argument("--rate", type=float, default=0.0,
                        help="回放速率倍率 (0=尽可能快, 1=实时)")
    parser.add_argument("--max-frames", type=int, default=0,
                        help="最多处理 N 帧 (0=处理全部)")
    parser.add_argument("--skip-frames", type=int, default=0,
                        help="跳过前 N 帧 (如起飞阶段 ROI 为空), 默认 0")
    parser.add_argument("--no-display", action="store_true",
                        help="关闭全部可视化窗口 (且不导入 Matplotlib/打开 OpenCV 窗口)")
    parser.add_argument("--no-pointcloud", action="store_true",
                        help="仅关闭 Matplotlib 3D 原始/去畸变点云对比窗口")
    parser.add_argument("--show-raw-compare", action="store_true",
                        help="可选: 增加 Raw Livox 原始点云与 Deskewed Body "
                             "去畸变点云 3D 对比窗口")
    parser.add_argument("--save-dir", type=str, default=None,
                        help="保存目录; 每帧 npz 保存到 <save-dir>/<strategy>/")
    parser.add_argument("--save-raw-arrays", action="store_true",
                        help="额外保存 raw_points/deskewed_points/"
                             "processed_points/fused_points 大数组")
    return parser


def cfg_value(args_value, cfg_value, default):
    """CLI 显式值 > 配置值 > 默认值."""
    if args_value is not None:
        return args_value
    return cfg_value if cfg_value is not None else default


# ──────────────────────────────────────────────
# 脚本一/二共用主循环
# ──────────────────────────────────────────────
def run_standard_replay(args, strategy: str, branch_factory):
    """标准链: 读帧 → pose 匹配 → body→level ROI → 动态 ROI → BEV 降采样 →
    语义分支 → training-camera 深度投影 → NN-fill → ONNX DRL → 打印/显示/保存.

    branch_factory(cfg) -> 可调用 (bev, bounds) -> (sem_map, sem_info)
    """
    cfg = load_config(args.config)
    params = perception_params(cfg)
    if args.dmax is not None:
        params["dmax"] = float(args.dmax)
    dmax = params["dmax"]

    from perception.training_camera_projection import TrainingCameraModel
    camera = TrainingCameraModel.from_config(
        cfg.get("depth_projection", {}).get("training_camera", {}),
        output_width=params["obs_w"], output_height=params["obs_h"], far_m=dmax)

    source = BagFrameSource(
        args.bag, cfg, cloud_topic=args.cloud_topic, pose_topic=args.pose_topic,
        raw_topic=args.raw_topic, imu_topic=args.imu_topic,
        cloud_source=args.cloud_source, max_sync_ms=params["max_sync_ms"])
    # finally 引用的状态在 try 前初始化, 避免中途初始化异常时 NameError
    vis = None
    action_names = []
    action_counts = []
    total_processed = 0
    start_wall = time.perf_counter()
    try:
        drl = ONNXDRL(
            args.onnx_model, obs_h=params["obs_h"], obs_w=params["obs_w"],
            dmax=dmax,
            depth_norm_mode=str(cfg.get("observation", {}).get(
                "depth_norm_mode", "raw_meters_graph_scaled")),
            semantic_norm_mode=str(cfg.get("observation", {}).get(
                "semantic_norm_mode", "raw_gray_graph_scaled")))

        from control.action_decomposer import ActionDecomposer
        decomposer = ActionDecomposer(cfg.get("uav", {}))
        action_names = decomposer.action_names

        branch = branch_factory(cfg)

        saver = FrameSaver(args.save_dir, strategy) if args.save_dir else None
        vis = None if args.no_display else CompareVisualizer(
            dmax=dmax, show_pointcloud=not args.no_pointcloud,
            show_raw_compare=args.show_raw_compare)

        ground_z = args.ground_z
        frame_count = 0
        action_counts = [0] * len(action_names)
        grid_res = params["bev_grid_res"]

        skipped = 0
        for frame in source:
            if args.skip_frames > 0 and skipped < args.skip_frames:
                skipped += 1
                continue
            if ground_z is None:
                ground_z = float(frame.pose[2])
                logger.info("[Ground] First frame pose_z=%.2f set as ground_z",
                            ground_z)

            half_x, half_y = dynamic_roi_half_extents(
                params, float(frame.pose[2]), float(ground_z), camera)
            bounds = roi_bounds(half_x, half_y)

            # ── body 点云 → level-body ROI (外参 + roll/pitch 水平化 + z-down) ──
            from perception.halss_preprocess import body_cloud_to_level_body_roi
            level_pts, _ = body_cloud_to_level_body_roi(
                frame.cloud_pts, float(frame.pose[3]), float(frame.pose[4]),
                params["perc_cfg"], half_x=half_x, half_y=half_y)
            if len(level_pts) < 10:
                logger.debug("[Frame] Sparse ROI: %d points, skip", len(level_pts))
                continue

            # ── BEV 粗糙度保持降采样 (HALSS 与深度共用同一批点) ──
            bev = bev_roughness_downsample(level_pts, bounds, grid_res=grid_res)
            if len(bev.points) == 0:
                logger.debug("[Frame] Empty BEV, skip")
                continue

            # ── 语义分支 ──
            t0 = time.perf_counter()
            sem_map, sem_info = branch(bev, bounds)

            # ── training-camera 深度投影 (z-buffer + 语义标签) ──
            sparse_depth, valid_mask, sem_map, semantic_valid_mask = project_depth(
                bev.points, sem_map, bounds, camera, danger_id=params["danger_id"])
            dense_depth = render_sparse_depth(sparse_depth, valid_mask, dmax)

            # ── ONNX DRL ──
            t_onnx = time.perf_counter()
            action_id, rl_info = drl.predict(dense_depth, sem_map)
            onnx_ms = (time.perf_counter() - t_onnx) * 1000.0
            action_name = decomposer.action_id_to_name(action_id)
            action_counts[action_id] += 1
            total_ms = (time.perf_counter() - t0) * 1000.0
            frame_count += 1
            total_processed += 1

            sem_safe_ratio = float(np.mean(sem_map == params["safe_id"]))
            depth_valid_ratio = float(np.mean(valid_mask))

            print_frame_block(
                strategy, frame_count,
                n_raw=len(frame.raw_pts) if frame.raw_pts is not None else None,
                n_deskewed=int(len(frame.cloud_pts)),
                n_processed=int(len(bev.points)),
                sync_ms=frame.sync_ms,
                depth_valid_ratio=depth_valid_ratio,
                sem_safe_ratio=sem_safe_ratio,
                action_id=action_id, action_name=action_name,
                probs=rl_info["action_probs"], action_names=action_names,
                onnx_ms=onnx_ms, total_ms=total_ms,
                cloud_source=frame.cloud_source)

            if frame_count % 5 == 0:
                logger.info(
                    "[%04d] %s source=%s act=%d(%s) pts=%d sem_safe=%.2f depth_valid=%.2f "
                    "conf=%.2f %s sem=%s %.0fms total=%.0fms",
                    frame_count, strategy, frame.cloud_source, action_id, action_name,
                    int(len(bev.points)), sem_safe_ratio, depth_valid_ratio,
                    rl_info.get("confidence", 0.0),
                    top3_str(rl_info.get("action_probs"), action_names),
                    sem_info, total_ms)

            # ── 可视化 ──
            if vis is not None:
                text = (f"[{strategy}/{frame.cloud_source} #{frame_count}] sync={frame.sync_ms:.0f}ms "
                        f"act={action_id}({action_name})")
                vis.update(render_bev_bgr(bev), sem_map,
                           semantic_valid_mask, dense_depth, text=text)
                if vis.show_raw_compare:
                    vis.update_3d(raw_pts=frame.raw_pts, raw_stamp=frame.raw_stamp,
                                  deskewed_pts=frame.cloud_pts,
                                  deskewed_stamp=frame.cloud_stamp)

            # ── 保存 ──
            if saver is not None:
                binary_semantic = make_binary_semantic_vis(
                    sem_map, safe_id=params["safe_id"], danger_id=params["danger_id"])
                binary_semantic[~semantic_valid_mask] = 128
                saver.save(frame_count, {
                    "sparse_depth": sparse_depth,
                    "dense_depth": dense_depth,
                    "sem_map": sem_map,
                    "binary_semantic": binary_semantic,
                    "valid_mask": valid_mask,
                    "semantic_valid_mask": semantic_valid_mask,
                    "pose": frame.pose,
                    "cloud_stamp": frame.cloud_stamp,
                    "pose_stamp": frame.pose_stamp,
                    "raw_stamp": frame.raw_stamp,
                    "sync_ms": frame.sync_ms,
                    "cloud_seq": frame.cloud_seq,
                    "action_id": action_id,
                    "action_probs": np.asarray(rl_info["action_probs"], dtype=np.float32),
                    "raw_points": frame.raw_pts,
                    "deskewed_points": frame.cloud_pts,
                    "processed_points": bev.points,
                }, save_raw_arrays=args.save_raw_arrays, summary={
                    "cloud_stamp": frame.cloud_stamp, "pose_stamp": frame.pose_stamp,
                    "raw_stamp": frame.raw_stamp or "", "sync_ms": frame.sync_ms,
                    "cloud_seq": frame.cloud_seq, "action_id": action_id,
                    "n_raw": len(frame.raw_pts) if frame.raw_pts is not None else "",
                    "n_deskewed": len(frame.cloud_pts),
                    "n_processed": len(bev.points),
                    "depth_valid_ratio": f"{depth_valid_ratio:.3f}",
                    "sem_safe_ratio": f"{sem_safe_ratio:.3f}",
                    "onnx_ms": f"{onnx_ms:.1f}", "total_ms": f"{total_ms:.1f}"})

            # ── 速率控制 / 帧数上限 ──
            if args.rate > 0:
                elapsed = time.perf_counter() - t0
                sleep = max(0.0, (1.0 / args.rate) - elapsed)
                if sleep > 0:
                    time.sleep(sleep)
            if args.max_frames > 0 and frame_count >= args.max_frames:
                logger.info("[Replay] Max frames reached: %d", args.max_frames)
                break
    finally:
        source.close()
        elapsed = time.perf_counter() - start_wall
        log_summary(strategy, total_processed, elapsed, action_names, action_counts)
        if vis is not None:
            vis.close()
    return {"frames_processed": total_processed, "elapsed_s": elapsed,
            "action_counts": action_counts}
