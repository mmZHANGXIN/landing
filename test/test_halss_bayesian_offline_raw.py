#!/usr/bin/env python3
"""
HALSS Bayesian 离线调试脚本 (原始点云版) — 读取原始 Livox rosbag → 感知管线 → 保存结果
====================================================================================
与 test_halss_bayesian_offline.py 保持一致的感知逻辑，但输入改为原始 Livox 点云:

输入话题:
  /livox/lidar  (sensor_msgs/PointCloud2)  — 原始 Livox MID360 点云 (LiDAR 系)
  /livox/imu    (sensor_msgs/msg/Imu)       — IMU 数据 (用于姿态估计)

与 FAST-LIO 版本的关键区别:
  - 点云已在 LiDAR 坐标系 (非世界系), 无需 Odometry 位姿转换
  - LiDAR 系 (x=前, y=左, z=上) → HALSS 机体系 (x=前, y=右, z=下)
  - 使用 IMU 四元数做姿态补偿 (如有), 否则假设水平飞行
  - 其余感知管线 (HALSS Bayesian / SemanticGenerator) 完全一致

输出: 与 test_halss_bayesian_offline.py 相同
  - deskewed_cloud.npy / surface_normal.png / mean_map.png
  - variance_map.png / semantic_map.png / binary_semantic.png

用法:
  conda activate fylanding
  source /opt/ros/galactic/setup.bash
  # livox_ws 不需要 source — 使用 rosbags 库解析 CustomMsg
  python3 test_halss_bayesian_offline_raw.py --bag-dir bags/rosbag2_2026_06_12-16_05_19
  python3 test_halss_bayesian_offline_raw.py --bag-dir ... --output-dir my_results

录制原始 bag (包含点云 + IMU):
  ros2 bag record -o my_raw_bag /livox/lidar /livox/imu
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import logging
from pathlib import Path

import numpy as np
import cv2
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("HALSSOfflineRaw")

# ============================================================
# 项目路径 & 配置
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

CONFIG_PATH = PROJECT_ROOT / "config" / "experiment_config.yaml"
with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    CFG = yaml.safe_load(f)

# ---- 感知模块 (与 pipeline / nocontrol / live 一致) ----
from perception.halss_bayesian import HALSSBayesianEvaluator
from perception.halss_preprocess import _rot_zyx, _cfg_vec3
from perception.semantic_generator import SemanticGenerator

BAGS_ROOT = PROJECT_ROOT / "bags"


# ============================================================
# 命令行参数
# ============================================================
def _parse_args():
    parser = argparse.ArgumentParser(
        description="HALSS Bayesian 离线调试 (原始 Livox 点云)"
    )
    parser.add_argument(
        "--bag-dir", required=True,
        help="指定 rosbag 目录 (如 bags/rosbag2_2026_06_12-14_31_47_rawdata_100)",
    )
    parser.add_argument(
        "--output-dir", default="experiments/halss_offline_raw",
        help="输出目录 (默认: experiments/halss_offline_raw)",
    )
    return parser.parse_args()


def _semantic_generator_cfg(pcfg: dict) -> dict:
    """Build the same SemanticGenerator cfg used by the online pipeline."""
    obs_cfg = CFG.get("observation", {})
    return {
        **pcfg,
        "img_width": int(obs_cfg.get("img_width", pcfg.get("img_width", 128))),
        "img_height": int(obs_cfg.get("img_height", pcfg.get("img_height", 128))),
    }


# ============================================================
# 坐标变换: Livox LiDAR 系 → HALSS 机体系 (下视)
# ============================================================

# Livox MID360 坐标系: x=前, y=左, z=上
# HALSS 机体系:        x=前, y=右, z=下
#
# 矫正后安装参数 (用户确认 2026-06-14):
#   halss_lidar_position_body_m: [0.13, 0.0, 0.08]   # [前,右,下] 米
#   pitch=116° (圆顶相对竖直前倾, 即 26° 低于水平面, 指向前下方)
#   R_body_from_lidar = R_axis @ Ry(+116°)
#   LiDAR +x ±45° 方位角过滤 (最陡下视锥, 适合近地高度)
#   halss_lidar_yaw_offset_deg: 0.0
#   halss_lidar_fov_half_deg: 45.0


def _get_rotation_body_from_lidar(cfg: dict) -> np.ndarray:
    """构建 LiDAR 系 → 机体系 旋转矩阵 (pitch=116° 矫正版).

    R_body_from_lidar = R_axis @ Ry(116°)

    R_axis = diag(1, -1, -1):  LiDAR (x=fwd,y=left,z=up) → 机体 (x=fwd,y=right,z=down)
    Ry(116°): 圆顶绕 y 轴前倾 116° (从正上旋转到前下方 26° 低于水平)

    注意: 此脚本始终使用 pitch=116° 矫正版, 忽略 config 中的旧 26° 矩阵.
    """
    pitch_deg = 116.0  # 矫正后的实际安装角度, 覆盖 config 中的旧值
    pitch = np.deg2rad(pitch_deg)
    cp, sp = np.cos(pitch), np.sin(pitch)

    R_axis = np.array([
        [1,  0,  0],
        [0, -1,  0],
        [0,  0, -1],
    ], dtype=np.float32)

    Ry = np.array([
        [cp, 0, sp],
        [0,  1, 0],
        [-sp, 0, cp],
    ], dtype=np.float32)

    R_bl = R_axis @ Ry

    lidar_z_body = R_bl[:, 2]
    lidar_x_body = R_bl[:, 0]
    logger.info(
        "  [Transform] R_body_from_lidar = R_axis @ Ry(%.0f°)", pitch_deg,
    )
    logger.info(
        "    dome (+z) → body [%+.3f, %+.3f, %+.3f]  (%.0f° below horizon)",
        lidar_z_body[0], lidar_z_body[1], lidar_z_body[2],
        np.degrees(np.arctan2(lidar_z_body[2], lidar_z_body[0])),
    )
    logger.info(
        "    LiDAR +x → body [%+.3f, %+.3f, %+.3f]",
        lidar_x_body[0], lidar_x_body[1], lidar_x_body[2],
    )
    return R_bl


def raw_lidar_to_body_down_roi(
    points_lidar: np.ndarray,
    imu_rpy: np.ndarray | None,
    cfg: dict,
) -> tuple[np.ndarray, dict]:
    """将原始 Livox 点云 (LiDAR 系) 转换到 HALSS 机体系下视 ROI。

    处理流程:
      1. LiDAR → 机体: pts_body = pts @ R_bl.T + t_bl
         (R_bl = R_axis @ Ry(116°), t_bl = [0.13, 0, 0.08])
      2. IMU yaw-only leveling
      3. LiDAR +x 方位角 ±45° 过滤 (最陡下视锥)
      4. ROI 过滤: 半径 + 高度范围

    参数:
      points_lidar: (N,3) 原始 Livox 点云 (x=前, y=左, z=上)
      imu_rpy: (3,) [roll, pitch, yaw] 来自 IMU, None 则假设水平
      cfg: 感知配置字典

    返回:
      roi: (M,3) 机体系下视点云 (x=前, y=右, z=下)
      stats: 统计信息字典
    """
    azimuth_half_deg = float(cfg.get("halss_lidar_fov_half_deg", 45.0))
    azimuth_half_rad = np.deg2rad(azimuth_half_deg)
    t_bl = np.array(
        cfg.get("halss_lidar_position_body_m", [0.13, 0.0, 0.08]),
        dtype=np.float32,
    )

    stats = {
        "input_points": 0,
        "output_points": 0,
        "fov_passed": 0,
        "roi_radius_m": float(cfg.get("halss_roi_radius_body", cfg.get("roi_radius_world", 25.0))),
        "min_down_m": float(cfg.get("halss_min_down_m", 0.05)),
        "max_down_m": float(cfg.get("halss_max_down_m", cfg.get("halss_roi_max_down_m", 30.0))),
        "imu_available": imu_rpy is not None,
        "azimuth_half_deg": azimuth_half_deg,
    }

    if points_lidar is None:
        return np.empty((0, 3), dtype=np.float32), stats

    pts = np.asarray(points_lidar, dtype=np.float32)
    if pts.size == 0:
        return np.empty((0, 3), dtype=np.float32), stats
    if pts.ndim == 1:
        pts = pts[:3].reshape(1, 3)
    elif pts.ndim == 2 and pts.shape[1] >= 3:
        pts = pts[:, :3]

    stats["input_points"] = int(len(pts))

    # ---- Step 1: LiDAR 系 → 机体系 (旋转 + 平移) ----
    R_bl = _get_rotation_body_from_lidar(cfg)
    pts_body = pts @ R_bl.T + t_bl  # (N,3)

    # ---- Step 2: IMU yaw-only leveling ----
    if imu_rpy is not None:
        roll, pitch, yaw = float(imu_rpy[0]), float(imu_rpy[1]), float(imu_rpy[2])
        yaw_offset = np.deg2rad(float(cfg.get(
            "halss_lidar_yaw_offset_deg",
            cfg.get("lidar_yaw_offset_deg", 0.0),
        )))
        yaw = yaw + yaw_offset

        yaw_only = bool(cfg.get("halss_yaw_only", True))
        if yaw_only:
            R_level = _rot_zyx(0.0, 0.0, -yaw)
        else:
            R_level = _rot_zyx(-roll, -pitch, -yaw)
        pts_body = pts_body @ R_level.T
    else:
        R_level = np.eye(3, dtype=np.float32)

    # ---- Step 3: LiDAR +x 方位角过滤 ----
    # 逆变换回 LiDAR 系, 检查每个点是否在 LiDAR +x 半空间 ±azimuth_half 内
    R_eff = R_level @ R_bl
    pts_lidar_aligned = (pts_body - t_bl) @ R_eff  # (N,3)
    x_l, y_l, z_l = pts_lidar_aligned[:, 0], pts_lidar_aligned[:, 1], pts_lidar_aligned[:, 2]
    azimuth = np.arctan2(y_l, x_l)
    in_fov = (
        (x_l > 0.0)
        & (np.abs(azimuth) <= azimuth_half_rad)
        & (z_l >= 0.0)  # 排除圆顶反方向
    )

    pts_fov = pts_body[in_fov]
    stats["fov_passed"] = int(np.sum(in_fov))

    if len(pts_fov) == 0:
        stats["output_points"] = 0
        return np.empty((0, 3), dtype=np.float32), stats

    # ---- Step 4: ROI 过滤 (半径 + 高度) ----
    lateral = np.linalg.norm(pts_fov[:, :2], axis=1)
    radius = float(stats["roi_radius_m"])
    min_down = float(stats["min_down_m"])
    max_down = float(stats["max_down_m"])
    keep = (
        np.isfinite(pts_fov).all(axis=1)
        & (lateral <= radius)
        & (pts_fov[:, 2] >= min_down)
        & (pts_fov[:, 2] <= max_down)
    )

    roi = pts_fov[keep].astype(np.float32, copy=False)
    stats.update({
        "output_points": int(len(roi)),
        "z_min_body": float(np.min(pts_fov[:, 2])) if len(pts_fov) else float("nan"),
        "z_max_body": float(np.max(pts_fov[:, 2])) if len(pts_fov) else float("nan"),
    })
    return roi, stats


# ============================================================
# ROS2 Bag 读取 (原始 Livox)
# ============================================================

def _quat_to_euler(x, y, z, w):
    """四元数 → roll, pitch, yaw"""
    sinr = 2.0 * (w * x + y * z)
    cosr = 1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(sinr, cosr)
    sinp = 2.0 * (w * y - z * x)
    pitch = (
        np.sign(sinp) * (np.pi / 2.0)
        if abs(sinp) >= 1.0
        else np.arcsin(sinp)
    )
    siny = 2.0 * (w * z + x * y)
    cosy = 1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(siny, cosy)
    return roll, pitch, yaw


def _register_livox_types(typestore) -> None:
    """Register livox_ros_driver2 CustomMsg / CustomPoint with the type store."""
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
        logger.debug("  [typestore] registered %s", type_name)


def read_raw_bag(bag_dir: Path) -> tuple[np.ndarray, np.ndarray | None, float] | None:
    """
    读取原始 Livox rosbag，提取 LiDAR 点云 + IMU 姿态。

    使用 rosbags 库 (不依赖 rclpy type support) 解析 livox CustomMsg。

    返回: (points_lidar, imu_rpy, timestamp) 或 None
      - points_lidar: (N,3) float32  原始 Livox 点云 (LiDAR 系)
      - imu_rpy: (3,) float32 或 None  [roll, pitch, yaw] 来自 IMU
      - timestamp: float  秒
    """
    from rosbags.rosbag2 import Reader
    from rosbags.typesys import Stores, get_typestore

    db3_files = sorted(bag_dir.glob("*.db3"))
    if not db3_files:
        logger.error("Bag 目录 %s 中没有 .db3 文件", bag_dir)
        return None

    # 注册 Livox 自定义消息类型
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
            "[Bag:%s] Lidar: %d pts | IMU rpy=[%.1f,%.1f,%.1f]deg",
            bag_dir.name, len(lidar_points),
            np.degrees(imu_rpy[0]), np.degrees(imu_rpy[1]), np.degrees(imu_rpy[2]),
        )
    else:
        logger.info(
            "[Bag:%s] Lidar: %d pts | IMU unavailable (assuming level)",
            bag_dir.name, len(lidar_points),
        )

    return lidar_points, imu_rpy, timestamp


# ============================================================
# 结果保存 (与 test_halss_bayesian_offline.py 完全一致)
# ============================================================
def save_results(
    output_dir: Path,
    frame_name: str,
    sem_gen: SemanticGenerator,
    deskewed_cloud: np.ndarray,
    surf_norm_rgb: np.ndarray,
    mean_map: np.ndarray,
    var_map: np.ndarray,
    sem_map: np.ndarray,
    safety_map: np.ndarray,
):
    """保存所有中间结果到输出目录"""
    os.makedirs(output_dir, exist_ok=True)

    # 1. 点云 (机体系下视ROI) — 保存为 .npy
    np.save(str(output_dir / f"{frame_name}_deskewed_cloud.npy"), deskewed_cloud)
    logger.info("  Saved: %s_deskewed_cloud.npy (%d pts)", frame_name, len(deskewed_cloud))

    # 2. 地表法向量拟合图
    if surf_norm_rgb is not None:
        cv2.imwrite(str(output_dir / f"{frame_name}_surface_normal.png"), surf_norm_rgb)
        logger.info("  Saved: %s_surface_normal.png (%dx%d)",
                    frame_name, surf_norm_rgb.shape[1], surf_norm_rgb.shape[0])

    # 3. MC Dropout 均值图 (伪彩色)
    if mean_map is not None:
        vmin, vmax = np.nanmin(mean_map), np.nanmax(mean_map)
        if vmax - vmin > 1e-8:
            mean_vis = ((mean_map - vmin) / (vmax - vmin) * 255).astype(np.uint8)
        else:
            mean_vis = np.zeros_like(mean_map, dtype=np.uint8)
        mean_color = cv2.applyColorMap(mean_vis, cv2.COLORMAP_INFERNO)
        cv2.imwrite(str(output_dir / f"{frame_name}_mean_map.png"), mean_color)
        logger.info("  Saved: %s_mean_map.png (range [%.4f, %.4f])",
                    frame_name, vmin, vmax)

    # 4. MC Dropout 方差图 (不确定性, 伪彩色)
    if var_map is not None:
        vmin, vmax = np.nanmin(var_map), np.nanmax(var_map)
        if vmax - vmin > 1e-8:
            var_vis = ((var_map - vmin) / (vmax - vmin) * 255).astype(np.uint8)
        else:
            var_vis = np.zeros_like(var_map, dtype=np.uint8)
        var_color = cv2.applyColorMap(var_vis, cv2.COLORMAP_HOT)
        cv2.imwrite(str(output_dir / f"{frame_name}_variance_map.png"), var_color)
        logger.info("  Saved: %s_variance_map.png (range [%.6f, %.6f])",
                    frame_name, vmin, vmax)

    # 5. 语义图 (绿色=安全, 红色=危险)
    if sem_map is not None:
        sem_rgb = sem_gen.colorize(sem_map)
        safe_pct = float(np.mean(sem_map == sem_gen.safe_id)) * 100
        cv2.imwrite(str(output_dir / f"{frame_name}_semantic_map.png"), sem_rgb)
        logger.info("  Saved: %s_semantic_map.png (safe=%.1f%%)", frame_name, safe_pct)

    # 6. HALSS 二值语义图 (白色=安全, 黑色=危险)
    if safety_map is not None:
        cv2.imwrite(str(output_dir / f"{frame_name}_binary_semantic.png"), safety_map)
        safe_pct = float(safety_map.mean() / 255.0) * 100
        logger.info("  Saved: %s_binary_semantic.png (safe=%.1f%%)", frame_name, safe_pct)


# ============================================================
# 单帧处理
# ============================================================
def process_bag(
    bag_dir: Path,
    halss: HALSSBayesianEvaluator,
    sem_gen: SemanticGenerator,
    pcfg: dict,
    output_dir: Path,
):
    """处理单个原始 rosbag: 读取 → 坐标变换 → 感知管线 → 保存"""
    bag_name = bag_dir.name

    logger.info("=" * 60)
    logger.info(" Processing (raw): %s", bag_name)
    logger.info("=" * 60)

    # ---- Step 0: 读取原始 bag ----
    bag_data = read_raw_bag(bag_dir)
    if bag_data is None:
        logger.error("Failed to read raw bag: %s", bag_name)
        return False

    points_lidar, imu_rpy, timestamp = bag_data

    # ---- Step 1: LiDAR 系 → 机体系下视 ROI ----
    halss_pts, halss_stats = raw_lidar_to_body_down_roi(points_lidar, imu_rpy, pcfg)
    logger.info(
        "  ROI: %d/%d pts | radius=%.1fm | z_down=[%.2f, %.2f] | imu=%s",
        halss_stats["output_points"], halss_stats["input_points"],
        halss_stats["roi_radius_m"],
        halss_stats.get("z_min_body", float("nan")),
        halss_stats.get("z_max_body", float("nan")),
        "yes" if halss_stats["imu_available"] else "no",
    )

    if halss_stats["output_points"] < 10:
        logger.warning("  ROI 点太少 (%d), 跳过评估", halss_stats["output_points"])
        return False

    # ---- Step 2: HALSS Bayesian 评估 (与原版完全一致) ----
    t0 = time.perf_counter()
    result = halss.evaluate(halss_pts)
    dt_halss = (time.perf_counter() - t0) * 1000

    if result is None:
        logger.warning("  HALSS evaluate returned None, 跳过")
        return False

    mean_val = result["mean_map"].mean() if result["mean_map"] is not None else float("nan")
    var_val = result["variance_map"].mean() if result["variance_map"] is not None else float("nan")
    logger.info("  HALSS: %.0fms | mean=%.4f var=%.6f", dt_halss, mean_val, var_val)

    # ---- Step 3: 语义图生成 (与原版完全一致) ----
    t1 = time.perf_counter()
    bev_data = result.get("bev_data", result)
    sem_map = sem_gen.generate(bev_data)
    dt_sem = (time.perf_counter() - t1) * 1000

    safe_ratio = float(np.mean(sem_map == sem_gen.safe_id)) * 100
    logger.info("  Semantic: %.1fms | safe=%.1f%%", dt_sem, safe_ratio)

    # ---- Step 4: 提取中间结果 (与原版完全一致) ----
    surf_norm = result.get("surf_norm_rgb")
    mean_map = result.get("mean_map")
    var_map = result.get("variance_map")
    safety_map = bev_data.get("safety_map_vis", result.get("safety_map_vis"))

    # ---- Step 5: 保存 (与原版完全一致) ----
    save_results(
        output_dir, bag_name,
        sem_gen=sem_gen,
        deskewed_cloud=halss_pts,
        surf_norm_rgb=surf_norm,
        mean_map=mean_map,
        var_map=var_map,
        sem_map=sem_map,
        safety_map=safety_map,
    )

    logger.info("  Done: %s → %s/", bag_name, output_dir)
    return True


# ============================================================
# 主逻辑
# ============================================================
def main():
    args = _parse_args()

    bag_path = Path(args.bag_dir)
    if not bag_path.is_absolute():
        bag_path = PROJECT_ROOT / bag_path
    if not bag_path.exists():
        logger.error("Bag 目录不存在: %s", bag_path)
        sys.exit(1)

    logger.info("将处理: %s", bag_path)

    # ---- 初始化感知模块 (与 pipeline/nocontrol 一致) ----
    pcfg = CFG["perception"]

    logger.info("=" * 60)
    logger.info(" Initializing HALSS Bayesian evaluator (raw point cloud mode)...")
    logger.info("=" * 60)
    halss = HALSSBayesianEvaluator(pcfg)

    sem_gen = SemanticGenerator(_semantic_generator_cfg(pcfg))

    logger.info("  [HALSS] %s | grid_res=%d | mc_samples=%d | device=%s",
                "Bayesian (Unet_drop + MC Dropout)",
                halss.grid_res, halss.mc_samples, halss.device)
    logger.info("  [Semantic] safe_id=%d danger_id=%d size=%dx%d",
                sem_gen.safe_id, sem_gen.danger_id, sem_gen.img_w, sem_gen.img_h)
    logger.info("  [Input] Raw Livox (/livox/lidar + /livox/imu)")
    logger.info("  [Transform] LiDAR frame (x=fwd,y=left,z=up) → Body down (x=fwd,y=right,z=down)")

    # ---- 处理 ----
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir

    ok = process_bag(bag_path, halss, sem_gen, pcfg, output_dir)

    # ---- 汇总 ----
    logger.info("=" * 60)
    logger.info(" Done: %s", "SUCCESS" if ok else "FAILED")
    logger.info(" Output: %s/", output_dir)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
