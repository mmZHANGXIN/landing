#!/usr/bin/env python3
"""
矫正安装参数后的地面点云提取 & 可视化脚本
============================================
安装参数 (用户确认):
  - 位移:  LiDAR 相对机体 IMU [前=0.13, 右=0, 下=0.08] m
  - 俯仰:  pitch=116° (圆顶相对竖直方向前倾 116°, 即 26° 低于水平面, 指向前下方)

处理管线:
  1. 读取 rosbag → 提取 /livox/lidar (CustomMsg) + /livox/imu (四元数)
  2. LiDAR 系 (x=前,y=左,z=上) → 机体系 (x=前,y=右,z=下)
     旋转: R_body_from_lidar = R_axis @ Ry(+116°)
     平移: pts_body = pts_lidar @ R_bl.T + [0.13, 0, 0.08]
  3. IMU 偏航补偿 (yaw-only leveling)
  4. FOV 过滤: 保留与圆顶探测主轴夹角 < fov_half 的点
  5. ROI 过滤: 半径 + 高度范围
  6. 可视化输出到 bag 目录

用法:
  conda activate fylanding
  python3 visualize_ground_points_raw.py
  python3 visualize_ground_points_raw.py --bag-dir bags/rosbag2_2026_06_12-16_05_19
"""

from __future__ import annotations

import argparse
import os
import sys
import logging
from pathlib import Path

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("GroundVis")

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# 安装参数 (用户确认)
# ============================================================
LIDAR_POS_BODY = np.array([0.13, 0.0, 0.08], dtype=np.float32)  # [前, 右, 下] m
LIDAR_PITCH_DEG = 116.0  # 圆顶相对竖直前倾角 (0°=正上, 90°=水平, 116°=前下方26°低于水平)
AZIMUTH_HALF_DEG = 45.0   # LiDAR 系 +x 方向左右方位角半角
ROI_RADIUS_M = 25.0       # 感知半径
MIN_DOWN_M = 0.05         # 最小机体下距离
MAX_DOWN_M = 30.0         # 最大机体下距离


# ============================================================
# 旋转矩阵构建
# ============================================================

def _rot_zyx(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Body-to-world rotation for roll/pitch/yaw (NED convention)."""
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ], dtype=np.float32)


def build_R_body_from_lidar(pitch_deg: float) -> np.ndarray:
    """构建 LiDAR 系 → 机体系 旋转矩阵.

    LiDAR 系:  x=前, y=左, z=上 (圆顶方向)
    机体系:    x=前, y=右, z=下

    步骤:
      1. Ry(pitch): 将 LiDAR z 轴 (圆顶) 绕 y 轴前倾 pitch_deg
      2. R_axis:    坐标系翻转 (y 取反, z 取反)

    R_body_from_lidar = R_axis @ Ry(pitch)

    参数:
      pitch_deg: 圆顶相对竖直方向的前倾角
                0° = 正上, 90° = 水平前, 116° = 前下方 (26° 低于水平)

    返回:
      (3,3) float32 旋转矩阵
    """
    pitch = np.deg2rad(pitch_deg)
    cp, sp = np.cos(pitch), np.sin(pitch)

    # R_axis: LiDAR (x=fwd,y=left,z=up) → 机体 (x=fwd,y=right,z=down)
    R_axis = np.array([
        [1,  0,  0],
        [0, -1,  0],
        [0,  0, -1],
    ], dtype=np.float32)

    # Ry(pitch): 绕 LiDAR y 轴旋转 pitch
    Ry = np.array([
        [cp, 0, sp],
        [0,  1, 0],
        [-sp, 0, cp],
    ], dtype=np.float32)

    R_bl = R_axis @ Ry

    # 验证: LiDAR 各轴在机体下的方向
    lidar_z_body = R_bl[:, 2]  # 圆顶
    lidar_x_body = R_bl[:, 0]  # LiDAR +x
    dome_angle_below_horizon = np.degrees(np.arctan2(lidar_z_body[2], lidar_z_body[0]))
    lidar_x_angle = np.degrees(np.arctan2(lidar_x_body[2], lidar_x_body[0]))
    logger.info(
        "  R_body_from_lidar (pitch=%.0f°):", pitch_deg,
    )
    logger.info(
        "    dome (+z) → body [%+.4f, %+.4f, %+.4f]  (%.1f° below horizon, %.1f° from body-down)",
        lidar_z_body[0], lidar_z_body[1], lidar_z_body[2],
        dome_angle_below_horizon,
        90.0 - dome_angle_below_horizon,
    )
    logger.info(
        "    LiDAR +x → body [%+.4f, %+.4f, %+.4f]  (%.1f° from horizontal)",
        lidar_x_body[0], lidar_x_body[1], lidar_x_body[2],
        lidar_x_angle,
    )
    return R_bl


# ============================================================
# 坐标变换
# ============================================================

def transform_lidar_to_body(
    points_lidar: np.ndarray,
    imu_rpy: np.ndarray | None,
    R_bl: np.ndarray,
    t_bl: np.ndarray,
    filter_axis: str = "dome",
    azimuth_half_deg: float = 45.0,
    roi_radius: float = ROI_RADIUS_M,
    min_down: float = MIN_DOWN_M,
    max_down: float = MAX_DOWN_M,
    yaw_only: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """完整坐标变换: LiDAR 系 → 机体系下视 ROI。

    过滤策略:
      filter_axis="dome":  以 LiDAR +z (圆顶) 为参考轴, ±azimuth_half_deg 锥形过滤
                           (圆顶指向机体前方偏下, 适合提取前方地面)
      filter_axis="lidar_x": 以 LiDAR +x 为参考轴, ±azimuth_half_deg 方位角过滤
                           (pitch>90°时 LiDAR +x 指向后方偏下, 适合提取近处地面)

    返回:
      pts_body:   (M,3) 机体系 ROI 内点云
      pts_fov:    (K,3) 过滤后点云 (ROI 前)
      pts_all:    (N,3) 旋转+平移后所有点 (过滤前)
      stats:      统计字典
    """
    stats = {
        "input_points": 0,
        "fov_passed": 0,
        "output_points": 0,
    }

    if points_lidar is None:
        empty = np.empty((0, 3), dtype=np.float32)
        return empty, empty, empty, stats

    pts = np.asarray(points_lidar, dtype=np.float32)
    if pts.size == 0:
        empty = np.empty((0, 3), dtype=np.float32)
        return empty, empty, empty, stats
    if pts.ndim == 1:
        pts = pts[:3].reshape(1, 3)
    elif pts.ndim == 2 and pts.shape[1] >= 3:
        pts = pts[:, :3]

    stats["input_points"] = int(len(pts))
    stats["filter_axis"] = filter_axis

    # ---- Step 1: LiDAR 系 → 机体系 (旋转 + 平移) ----
    # pts_body = pts_lidar @ R_bl.T + t_bl
    pts_body_all = pts @ R_bl.T + t_bl  # (N,3)

    # ---- Step 2: IMU 偏航补偿 (yaw-only leveling) ----
    if imu_rpy is not None:
        roll, pitch, yaw = float(imu_rpy[0]), float(imu_rpy[1]), float(imu_rpy[2])
        if yaw_only:
            R_level = _rot_zyx(0.0, 0.0, -yaw)
        else:
            R_level = _rot_zyx(-roll, -pitch, -yaw)
        pts_body_all = pts_body_all @ R_level.T
        stats["imu_yaw_deg"] = np.degrees(yaw)
    else:
        R_level = np.eye(3, dtype=np.float32)

    # ---- Step 3: 方向过滤 ----
    R_eff = R_level @ R_bl
    pts_lidar_aligned = (pts_body_all - t_bl) @ R_eff  # (N,3) 逆变换回 LiDAR 系
    x_l, y_l, z_l = pts_lidar_aligned[:, 0], pts_lidar_aligned[:, 1], pts_lidar_aligned[:, 2]

    azimuth_half_rad = np.deg2rad(azimuth_half_deg)

    if filter_axis == "lidar_x":
        # LiDAR +x 为参考轴: x_l > 0, 方位角 ∈ [-half, +half]
        azimuth = np.arctan2(y_l, x_l)
        in_filter = (
            (x_l > 0.0)
            & (np.abs(azimuth) <= azimuth_half_rad)
            & (z_l >= 0.0)  # 排除圆顶反方向
        )
        ref_axis_name = "LiDAR +x"
        ref_dir_body = R_bl[:, 0]
    else:  # "dome" (default)
        # LiDAR +z (圆顶) 为参考轴: 计算每个点与圆顶方向的夹角
        norms_lidar = np.linalg.norm(pts_lidar_aligned, axis=1)
        valid = norms_lidar > 1e-8
        in_filter = np.zeros(len(pts_lidar_aligned), dtype=bool)
        if valid.any():
            dirs_lidar = np.zeros_like(pts_lidar_aligned)
            dirs_lidar[valid] = pts_lidar_aligned[valid] / norms_lidar[valid, np.newaxis]
            # cos(夹角) = dot(direction, [0,0,1]) = dirs_lidar[:, 2]
            cos_angle = dirs_lidar[:, 2]  # 与 LiDAR +z (圆顶) 的夹角余弦
            in_filter = (cos_angle >= np.cos(azimuth_half_rad)) & valid
        ref_axis_name = "dome (+z)"
        ref_dir_body = R_bl[:, 2]

    pts_fov_all = pts_body_all[in_filter]
    stats["fov_passed"] = int(np.sum(in_filter))
    stats["ref_axis"] = ref_axis_name
    stats["ref_dir_body"] = ref_dir_body.astype(np.float32)

    # ---- Step 4: ROI 过滤 (半径 + 高度) ----
    if len(pts_fov_all) > 0:
        lateral = np.linalg.norm(pts_fov_all[:, :2], axis=1)
        keep = (
            np.isfinite(pts_fov_all).all(axis=1)
            & (lateral <= roi_radius)
            & (pts_fov_all[:, 2] >= min_down)
            & (pts_fov_all[:, 2] <= max_down)
        )
        pts_roi = pts_fov_all[keep].astype(np.float32, copy=False)
    else:
        pts_roi = np.empty((0, 3), dtype=np.float32)

    stats["output_points"] = int(len(pts_roi))
    if len(pts_roi) > 0:
        stats["z_min"] = float(np.min(pts_roi[:, 2]))
        stats["z_max"] = float(np.max(pts_roi[:, 2]))
        stats["z_mean"] = float(np.mean(pts_roi[:, 2]))
        stats["x_range"] = [float(np.min(pts_roi[:, 0])), float(np.max(pts_roi[:, 0]))]
        stats["y_range"] = [float(np.min(pts_roi[:, 1])), float(np.max(pts_roi[:, 1]))]

    return pts_roi, pts_fov_all, pts_body_all, stats


# ============================================================
# ROS2 Bag 读取
# ============================================================

def _quat_to_euler(x, y, z, w):
    """四元数 → roll, pitch, yaw (rad)"""
    sinr = 2.0 * (w * x + y * z)
    cosr = 1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(sinr, cosr)
    sinp = 2.0 * (w * y - z * x)
    pitch = np.arcsin(np.clip(sinp, -1.0, 1.0)) if abs(sinp) < 1.0 else np.sign(sinp) * np.pi / 2
    siny = 2.0 * (w * z + x * y)
    cosy = 1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(siny, cosy)
    return roll, pitch, yaw


def _register_livox_types(typestore) -> None:
    """注册 livox_ros_driver2 CustomMsg / CustomPoint 消息类型."""
    from rosbags.typesys import get_types_from_msg

    msg_dirs = [
        Path.home() / "livox_ws/src/livox_ros_driver2/msg",
        Path.home() / "livox_ws/install/livox_ros_driver2/share/livox_ros_driver2/msg",
    ]
    msg_dir = None
    for d in msg_dirs:
        if d.is_dir() and (d / "CustomMsg.msg").exists():
            msg_dir = d
            break
    if msg_dir is None:
        raise FileNotFoundError("Cannot find livox_ros_driver2 msg directory")

    for msg_file in sorted(msg_dir.glob("*.msg")):
        type_name = f"livox_ros_driver2/msg/{msg_file.stem}"
        typestore.register(get_types_from_msg(msg_file.read_text(), type_name))


def read_raw_bag(bag_dir: Path) -> tuple[np.ndarray, np.ndarray | None, float] | None:
    """读取原始 Livox rosbag, 返回 (points_lidar, imu_rpy, timestamp)."""
    from rosbags.rosbag2 import Reader
    from rosbags.typesys import Stores, get_typestore

    db3_files = sorted(bag_dir.glob("*.db3"))
    if not db3_files:
        logger.error("Bag 目录 %s 中没有 .db3 文件", bag_dir)
        return None

    typestore = get_typestore(Stores.ROS2_GALACTIC)
    try:
        _register_livox_types(typestore)
    except FileNotFoundError as e:
        logger.error("无法注册 Livox 消息类型: %s", e)
        return None

    lidar_points = None
    imu_rpy = None
    timestamp = 0.0

    with Reader(str(bag_dir)) as reader:
        for conn, ts, raw_data in reader.messages():
            topic = conn.topic
            msgtype = conn.msgtype

            if topic == "/livox/lidar" and lidar_points is None:
                msg = typestore.deserialize_cdr(raw_data, msgtype)
                if msg.points and len(msg.points) > 0:
                    lidar_points = np.array(
                        [[p.x, p.y, p.z] for p in msg.points],
                        dtype=np.float32,
                    )
                    lidar_points = lidar_points[np.isfinite(lidar_points).all(axis=1)]
                    if hasattr(msg, 'header'):
                        timestamp = (
                            float(msg.header.stamp.sec)
                            + float(msg.header.stamp.nanosec) * 1e-9
                        )
                    else:
                        timestamp = float(ts) * 1e-9

            elif topic == "/livox/imu" and imu_rpy is None:
                msg = typestore.deserialize_cdr(raw_data, msgtype)
                q = msg.orientation
                if not (q.x == 0.0 and q.y == 0.0 and q.z == 0.0 and q.w == 0.0):
                    imu_rpy = np.array(
                        _quat_to_euler(q.x, q.y, q.z, q.w), dtype=np.float32
                    )

            if lidar_points is not None and imu_rpy is not None:
                break

    if lidar_points is None:
        logger.error("Bag %s: 未找到 /livox/lidar 消息", bag_dir.name)
        return None

    if imu_rpy is not None:
        logger.info(
            "[Bag] Lidar: %d pts | IMU rpy=[%.1f, %.1f, %.1f]°",
            len(lidar_points),
            np.degrees(imu_rpy[0]), np.degrees(imu_rpy[1]), np.degrees(imu_rpy[2]),
        )
    else:
        logger.info("[Bag] Lidar: %d pts | IMU unavailable (assuming level)", len(lidar_points))

    return lidar_points, imu_rpy, timestamp


# ============================================================
# 可视化
# ============================================================

def _auto_zoom_2d(ax, pts: np.ndarray, cols=(0, 1), min_span: float = 2.0):
    """根据点云数据自动缩放坐标轴, 保证至少 min_span 的范围."""
    if pts is None or len(pts) == 0:
        return
    c0, c1 = cols
    x = pts[:, c0]
    y = pts[:, c1]
    x_min, x_max = float(np.min(x)), float(np.max(x))
    y_min, y_max = float(np.min(y)), float(np.max(y))

    x_mid = (x_min + x_max) / 2.0
    y_mid = (y_min + y_max) / 2.0
    x_half = max((x_max - x_min) / 2.0, min_span / 2.0) * 1.2
    y_half = max((y_max - y_min) / 2.0, min_span / 2.0) * 1.2

    ax.set_xlim(x_mid - x_half, x_mid + x_half)
    ax.set_ylim(y_mid - y_half, y_mid + y_half)


def visualize_and_save(
    pts_roi: np.ndarray,
    pts_fov: np.ndarray,
    pts_all: np.ndarray,
    stats: dict,
    output_dir: Path,
    R_bl: np.ndarray,
    prefix: str = "ground",
):
    """生成可视化并保存到 output_dir."""
    os.makedirs(output_dir, exist_ok=True)

    # ---- 检查是否有 matplotlib ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import FancyBboxPatch
    except ImportError:
        logger.error("matplotlib 未安装, 无法可视化. 请 pip install matplotlib")
        return

    logger.info("=" * 60)
    logger.info(" 可视化地面点云")
    logger.info("=" * 60)
    logger.info("  输入点:     %d", stats.get("input_points", 0))
    logger.info("  方位角过滤: %d", stats.get("fov_passed", 0))
    logger.info("  ROI 地面点: %d", stats.get("output_points", 0))
    if "z_min" in stats:
        logger.info(
            "  高度范围:   z∈[%.2f, %.2f] mean=%.2f m",
            stats["z_min"], stats["z_max"], stats["z_mean"],
        )
    if "x_range" in stats:
        logger.info(
            "  X 范围:     [%.1f, %.1f] m", stats["x_range"][0], stats["x_range"][1],
        )
    if "y_range" in stats:
        logger.info(
            "  Y 范围:     [%.1f, %.1f] m", stats["y_range"][0], stats["y_range"][1],
        )

    # ---- 创建画布: 2 行 × 2 列 ----
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    fig.suptitle(
        f"Ground Point Cloud | LiDAR pitch={LIDAR_PITCH_DEG} deg, "
        f"pos=[{LIDAR_POS_BODY[0]:.2f},{LIDAR_POS_BODY[1]:.2f},{LIDAR_POS_BODY[2]:.2f}] m body\n"
        f"Filter: {stats.get('ref_axis', '?')} +-{AZIMUTH_HALF_DEG} deg | ROI: r={ROI_RADIUS_M}m, z=[{MIN_DOWN_M},{MAX_DOWN_M}]m | "
        f"{stats['output_points']} ground pts / {stats['input_points']} raw",
        fontsize=11,
    )

    # ---- [0,0] 俯视图 (Top-down: X-Y, 按高度着色) ----
    ax1 = axes[0, 0]
    if len(pts_roi) > 0:
        z = pts_roi[:, 2]
        z_norm = (z - z.min()) / (z.max() - z.min() + 1e-8)
        scatter1 = ax1.scatter(
            pts_roi[:, 0], pts_roi[:, 1],
            c=z, s=0.5, cmap="turbo", alpha=0.7,
            vmin=stats.get("z_min", 0), vmax=stats.get("z_max", 10),
        )
        cbar1 = plt.colorbar(scatter1, ax=ax1, label="z_down (m)")
    ax1.set_xlabel("X body: forward (m)")
    ax1.set_ylabel("Y body: right (m)")
    ax1.set_title("Top-Down View (height colormap)")
    ax1.axhline(y=0, color="gray", linestyle="--", alpha=0.3)
    ax1.axvline(x=0, color="gray", linestyle="--", alpha=0.3)
    ax1.set_aspect("equal")
    # 标记 LiDAR 位置
    ax1.plot(LIDAR_POS_BODY[0], LIDAR_POS_BODY[1], "r*", markersize=10, label="LiDAR")
    ax1.legend(fontsize=8)
    # 画 ROI 圆
    import matplotlib.patches as mpatches
    circle = mpatches.Circle(
        (LIDAR_POS_BODY[0], LIDAR_POS_BODY[1]),
        ROI_RADIUS_M, fill=False, color="red", linestyle="--", alpha=0.5,
    )
    ax1.add_patch(circle)
    # ---- 自动缩放: 以数据范围为中心, 至少 2m×2m ----
    _auto_zoom_2d(ax1, pts_roi, min_span=2.0)

    # ---- [0,1] 侧视图 (X-Z: 前向剖面) ----
    ax2 = axes[0, 1]
    if len(pts_roi) > 0:
        ax2.scatter(
            pts_roi[:, 0], pts_roi[:, 2],
            c=pts_roi[:, 1], s=0.5, cmap="coolwarm", alpha=0.7,
        )
    # 画过滤参考轴在机体 X-Z 面的投影
    ref_dir = stats.get("ref_dir_body", R_bl[:, 2])
    ref_name = stats.get("ref_axis", "dome")
    ref_len = 15.0
    ax2.annotate(
        "", xy=(LIDAR_POS_BODY[0] + ref_dir[0] * ref_len,
               LIDAR_POS_BODY[2] + ref_dir[2] * ref_len),
        xytext=(LIDAR_POS_BODY[0], LIDAR_POS_BODY[2]),
        arrowprops=dict(arrowstyle="->", color="red", lw=2),
    )
    ref_angle = np.degrees(np.arctan2(ref_dir[2], ref_dir[0]))
    ax2.text(
        LIDAR_POS_BODY[0] + ref_dir[0] * ref_len * 0.5,
        LIDAR_POS_BODY[2] + ref_dir[2] * ref_len * 0.5,
        f"{ref_name}\n{ref_angle:.0f} deg",
        color="red", fontsize=8,
    )
    # 画 ±azimuth 锥边界在 X-Z 面的投影
    az_half = np.deg2rad(AZIMUTH_HALF_DEG)
    if stats.get("filter_axis") == "dome":
        # dome 模式: 锥形, 在 LiDAR 系中绕 dome 旋转 az_half
        # 取 dome + 绕 y 轴 ±az_half (在 LiDAR x-z 面)
        for sign in [-1, 1]:
            angle_from_dome = sign * az_half
            # LiDAR 系方向: [sin(angle), 0, cos(angle)]
            dir_l = np.array([np.sin(angle_from_dome), 0.0, np.cos(angle_from_dome)])
            dir_b = dir_l @ R_bl.T
            ax2.plot(
                [LIDAR_POS_BODY[0], LIDAR_POS_BODY[0] + dir_b[0] * ref_len],
                [LIDAR_POS_BODY[2], LIDAR_POS_BODY[2] + dir_b[2] * ref_len],
                "r--", alpha=0.4, lw=1,
            )
    else:
        # lidar_x 模式: 在 LiDAR x-y 面 ±az_half
        for sign in [-1, 1]:
            dir_l = np.array([np.cos(az_half), sign * np.sin(az_half), 0.0])
            dir_b = dir_l @ R_bl.T
            ax2.plot(
                [LIDAR_POS_BODY[0], LIDAR_POS_BODY[0] + dir_b[0] * ref_len],
                [LIDAR_POS_BODY[2], LIDAR_POS_BODY[2] + dir_b[2] * ref_len],
                "r--", alpha=0.4, lw=1,
            )

    ax2.set_xlabel("X body: forward (m)")
    ax2.set_ylabel("Z body: down (m)")
    ax2.set_title("Side View (X-Z profile)")
    ax2.invert_yaxis()
    ax2.plot(LIDAR_POS_BODY[0], LIDAR_POS_BODY[2], "r*", markersize=10)
    ax2.axhline(y=MIN_DOWN_M, color="gray", linestyle=":", alpha=0.5)
    ax2.axhline(y=MAX_DOWN_M, color="gray", linestyle=":", alpha=0.5)
    ax2.grid(True, alpha=0.3)
    # ---- 自动缩放侧视图 ----
    _auto_zoom_2d(ax2, pts_roi, cols=(0, 2), min_span=1.0)

    # ---- [1,0] 原始点云俯视图 (过滤前、方位角过滤后、ROI) ----
    ax3 = axes[1, 0]
    if len(pts_all) > 0:
        n_ds = min(len(pts_all), 20000)
        idx = np.random.choice(len(pts_all), n_ds, replace=False)
        ax3.scatter(
            pts_all[idx, 0], pts_all[idx, 1],
            c="lightgray", s=0.3, alpha=0.5, label="all (pre-filter)",
        )
    if len(pts_fov) > 0:
        n_ds = min(len(pts_fov), 20000)
        idx = np.random.choice(len(pts_fov), n_ds, replace=False)
        ax3.scatter(
            pts_fov[idx, 0], pts_fov[idx, 1],
            c="orange", s=1.0, alpha=0.6, label="azimuth pass",
        )
    if len(pts_roi) > 0:
        n_ds = min(len(pts_roi), 20000)
        idx = np.random.choice(len(pts_roi), n_ds, replace=False)
        ax3.scatter(
            pts_roi[idx, 0], pts_roi[idx, 1],
            c="green", s=1.5, alpha=0.8, label="ROI (ground)",
        )
    ax3.set_xlabel("X body: forward (m)")
    ax3.set_ylabel("Y body: right (m)")
    ax3.set_title("Top-Down: All -> Azimuth -> ROI")
    ax3.legend(fontsize=7, markerscale=3)
    ax3.set_aspect("equal")
    ax3.plot(LIDAR_POS_BODY[0], LIDAR_POS_BODY[1], "r*", markersize=10)
    ax3.grid(True, alpha=0.3)
    # ---- 自动缩放: 同时考虑 ROI 和过滤后点 ----
    _auto_zoom_2d(ax3, pts_roi if len(pts_roi) > 0 else pts_fov, min_span=3.0)

    # ---- [1,1] 统计信息文本 ----
    ax4 = axes[1, 1]
    ax4.axis("off")
    lines = [
        "=== Transform Params ===",
        f"R_body_from_lidar = R_axis @ Ry({LIDAR_PITCH_DEG} deg)",
        f"t_body_from_lidar = [{LIDAR_POS_BODY[0]:.2f}, {LIDAR_POS_BODY[1]:.2f}, {LIDAR_POS_BODY[2]:.2f}] m",
        f"Filter axis = {stats.get('ref_axis', '?')}",
        f"Azimuth half-angle = {AZIMUTH_HALF_DEG} deg",
        f"ROI radius = {ROI_RADIUS_M} m",
        f"ROI z range = [{MIN_DOWN_M}, {MAX_DOWN_M}] m",
        "",
        "=== Point Stats ===",
        f"Raw LiDAR pts:   {stats.get('input_points', 0):,}",
        f"Filter pass:     {stats.get('fov_passed', 0):,} ({stats.get('fov_passed', 0) / max(stats.get('input_points', 1), 1) * 100:.1f}%)",
        f"ROI ground pts:  {stats.get('output_points', 0):,} ({stats.get('output_points', 0) / max(stats.get('input_points', 1), 1) * 100:.1f}%)",
        "",
    ]
    if "z_min" in stats:
        lines += [
            "=== Ground Height ===",
            f"z_down min:  {stats['z_min']:.2f} m",
            f"z_down max:  {stats['z_max']:.2f} m",
            f"z_down mean: {stats['z_mean']:.2f} m",
        ]
    if "x_range" in stats:
        lines += [
            "",
            "=== Spatial Extent ===",
            f"X: [{stats['x_range'][0]:.1f}, {stats['x_range'][1]:.1f}] m",
            f"Y: [{stats['y_range'][0]:.1f}, {stats['y_range'][1]:.1f}] m",
        ]
    if "imu_yaw_deg" in stats:
        lines += [
            "",
            f"IMU yaw: {stats['imu_yaw_deg']:.1f} deg",
        ]

    ax4.text(
        0.05, 0.95, "\n".join(lines),
        transform=ax4.transAxes,
        fontfamily="monospace", fontsize=9,
        verticalalignment="top",
    )

    plt.tight_layout()
    out_path = output_dir / f"{prefix}_visualization.png"
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("  可视化已保存: %s", out_path)

    # ---- 额外: 单独保存高清俯视图 ----
    fig2, ax_td = plt.subplots(figsize=(10, 10))
    if len(pts_roi) > 0:
        h = ax_td.scatter(
            pts_roi[:, 0], pts_roi[:, 1],
            c=pts_roi[:, 2], s=1.0, cmap="turbo", alpha=0.8,
        )
        cbar = plt.colorbar(h, ax=ax_td, label="z_down (m)", shrink=0.8)
    ax_td.set_xlabel("X body: forward (m)")
    ax_td.set_ylabel("Y body: right (m)")
    ax_td.set_title(f"Ground Points Top-Down | pitch={LIDAR_PITCH_DEG} deg | {stats['output_points']} pts")
    ax_td.set_aspect("equal")
    ax_td.plot(LIDAR_POS_BODY[0], LIDAR_POS_BODY[1], "r*", markersize=15, label="LiDAR")
    ax_td.legend()
    circle = mpatches.Circle(
        (LIDAR_POS_BODY[0], LIDAR_POS_BODY[1]),
        ROI_RADIUS_M, fill=False, color="white", linestyle="--", alpha=0.6,
    )
    ax_td.add_patch(circle)
    ax_td.grid(True, alpha=0.3)
    _auto_zoom_2d(ax_td, pts_roi, min_span=2.0)
    out_td = output_dir / f"{prefix}_topdown.png"
    fig2.savefig(str(out_td), dpi=150, bbox_inches="tight")
    plt.close(fig2)
    logger.info("  俯视图已保存: %s", out_td)

    # ---- 保存 ROI 点云为 .npy ----
    npy_path = output_dir / f"{prefix}_roi_points.npy"
    np.save(str(npy_path), pts_roi)
    logger.info("  ROI 点云已保存: %s (%d pts)", npy_path, len(pts_roi))


# ============================================================
# 主逻辑
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="矫正安装参数后的地面点云提取与可视化"
    )
    parser.add_argument(
        "--bag-dir",
        default="bags/rosbag2_2026_06_12-16_05_19",
        help="rosbag 目录路径",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="输出目录 (默认: bag 目录本身)",
    )
    parser.add_argument(
        "--pitch-deg",
        type=float,
        default=LIDAR_PITCH_DEG,
        help=f"LiDAR 前倾角 (默认: {LIDAR_PITCH_DEG}°)",
    )
    parser.add_argument(
        "--azimuth-half",
        type=float,
        default=AZIMUTH_HALF_DEG,
        help=f"方位角/锥角半角 (默认: {AZIMUTH_HALF_DEG}°)",
    )
    parser.add_argument(
        "--filter-axis",
        choices=["dome", "lidar_x"],
        default="dome",
        help="过滤参考轴: dome=圆顶方向(指向前下方,适合前方地面), lidar_x=LiDAR+x(指向后下方,适合近处地面)",
    )
    args = parser.parse_args()

    bag_path = Path(args.bag_dir)
    if not bag_path.is_absolute():
        bag_path = PROJECT_ROOT / bag_path
    if not bag_path.exists():
        logger.error("Bag 目录不存在: %s", bag_path)
        sys.exit(1)

    output_dir = Path(args.output_dir) if args.output_dir else bag_path
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir

    logger.info("=" * 60)
    logger.info(" 地面点云提取与可视化 (矫正安装参数)")
    logger.info("=" * 60)
    logger.info("  Bag:      %s", bag_path)
    logger.info("  Output:   %s", output_dir)
    logger.info("  Pitch:    %.0f° (dome %.0f° below horizon)", args.pitch_deg, args.pitch_deg - 90.0)
    logger.info("  Position: [%.2f, %.2f, %.2f] m body", *LIDAR_POS_BODY)
    logger.info("  Azimuth:  ±%.0f° (filter axis: %s)", args.azimuth_half, args.filter_axis)

    # ---- 构建旋转矩阵 ----
    R_bl = build_R_body_from_lidar(args.pitch_deg)
    t_bl = LIDAR_POS_BODY.copy()

    # ---- 读取 bag ----
    logger.info("-" * 40)
    logger.info(" Reading bag...")
    bag_data = read_raw_bag(bag_path)
    if bag_data is None:
        logger.error("Failed to read bag")
        sys.exit(1)

    points_lidar, imu_rpy, timestamp = bag_data

    # ---- 坐标变换 ----
    logger.info("-" * 40)
    logger.info(" Transforming LiDAR → Body (ground extraction)...")
    pts_roi, pts_fov, pts_all, stats = transform_lidar_to_body(
        points_lidar, imu_rpy,
        R_bl=R_bl, t_bl=t_bl,
        filter_axis=args.filter_axis,
        azimuth_half_deg=args.azimuth_half,
    )

    if len(pts_roi) < 10:
        logger.warning("  ROI 地面点太少 (%d), 扩大 FOV 或检查安装参数", len(pts_roi))
        # 仍然尝试可视化, 但用所有 FOV 后的点
        pts_roi = pts_fov

    # ---- 可视化 ----
    visualize_and_save(
        pts_roi, pts_fov, pts_all, stats,
        output_dir=output_dir,
        R_bl=R_bl,
        prefix=bag_path.name,
    )

    logger.info("=" * 60)
    logger.info(" 完成! 输出: %s/", output_dir)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
