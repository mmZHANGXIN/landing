#!/usr/bin/env python3
"""
HALSS Bayesian 离线调试脚本 — 读取 rosbag → 感知管线 → 保存结果
==================================================================
与 test_halss_bayesian_live.py / pipeline.py 保持一致的感知逻辑:
  - HALSSBayesianEvaluator (perception/halss_bayesian.py)
  - SemanticGenerator (perception/semantic_generator.py)
  - world_to_body_down_roi (perception/halss_preprocess.py)
  - 语义输出尺寸使用 observation.img_width/img_height，与 DRL 输入一致

输入: /home/orin/evelyn/landing/bags/capture_frames/frame_XXXX/ 下的 rosbag
输出: 指定目录下保存每帧的:
  - deskewed_cloud.npy         去畸变点云 (机体系下视ROI, Nx3 float32)
  - surface_normal.png         地表法向量拟合图 (RGB)
  - mean_map.png               MC Dropout 均值图
  - variance_map.png           MC Dropout 方差图
  - semantic_map.png           语义图 (绿色=安全, 红色=危险)
  - binary_semantic.png        HALSS 二值语义图 (白色=安全, 黑色=危险)

用法:
  conda activate fylanding
  source /opt/ros/galactic/setup.bash
  python3 test_halss_bayesian_offline.py                          # 处理第一个 bag
  python3 test_halss_bayesian_offline.py --bag-dir bags/capture_frames/frame_0005_20260609_165518  # 指定 bag
  python3 test_halss_bayesian_offline.py --bag-dir ... --output-dir my_results  # 指定输出目录
  python3 test_halss_bayesian_offline.py --all                                     # 处理所有 bag
  python3 test_halss_bayesian_offline.py --list                                    # 列出可用 bag
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
logger = logging.getLogger("HALSSOffline")

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
from perception.halss_preprocess import world_to_body_down_roi
from perception.semantic_generator import SemanticGenerator

BAGS_ROOT = PROJECT_ROOT / "bags" / "capture_frames"


# ============================================================
# 命令行参数
# ============================================================
def _parse_args():
    parser = argparse.ArgumentParser(
        description="HALSS Bayesian 离线调试 (从 rosbag 读取)"
    )
    parser.add_argument(
        "--bag-dir",
        help="指定 rosbag 目录 (如 bags/capture_frames/frame_0000_...)",
    )
    parser.add_argument(
        "--all", action="store_true",
        help="处理所有 bag",
    )
    parser.add_argument(
        "--output-dir", default="experiments/halss_offline",
        help="输出目录 (默认: experiments/halss_offline)",
    )
    parser.add_argument(
        "--list", action="store_true",
        help="仅列出可用 bag",
    )
    return parser.parse_args()


def _list_bags() -> list[Path]:
    """列出所有可用的 bag 目录"""
    if not BAGS_ROOT.exists():
        return []
    bags = sorted(
        d for d in BAGS_ROOT.iterdir()
        if d.is_dir() and list(d.glob("*.db3"))
    )
    return bags


def _semantic_generator_cfg(pcfg: dict) -> dict:
    """Build the same SemanticGenerator cfg used by the online pipeline."""
    obs_cfg = CFG.get("observation", {})
    return {
        **pcfg,
        "img_width": int(obs_cfg.get("img_width", pcfg.get("img_width", 128))),
        "img_height": int(obs_cfg.get("img_height", pcfg.get("img_height", 128))),
    }


# ============================================================
# ROS2 Bag 读取
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


def _pc2_to_numpy(msg) -> np.ndarray:
    """PointCloud2 → (N,3) float32 numpy"""
    field_offsets = {f.name: f.offset for f in msg.fields}
    if not all(k in field_offsets for k in ("x", "y", "z")):
        return np.empty((0, 3), dtype=np.float32)
    n = msg.width * msg.height
    endian = ">f4" if msg.is_bigendian else "<f4"
    dtype = np.dtype({
        "names": ["x", "y", "z"],
        "formats": [endian, endian, endian],
        "offsets": [
            field_offsets["x"],
            field_offsets["y"],
            field_offsets["z"],
        ],
        "itemsize": msg.point_step,
    })
    arr = np.frombuffer(msg.data, dtype=dtype, count=n)
    pts = np.column_stack((arr["x"], arr["y"], arr["z"])).astype(
        np.float32, copy=False
    )
    return pts[np.isfinite(pts).all(axis=1)]


def read_rosbag(bag_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, float] | None:
    """
    读取一个 rosbag，提取 Odometry + PointCloud2 并同步。

    返回: (pose_xyz, rpy, points_world, timestamp) 或 None
      - pose_xyz: (3,) float32  [x,y,z] 世界坐标
      - rpy: (3,) float32  [roll, pitch, yaw]
      - points_world: (N,3) float32  去畸变点云 (世界系)
      - timestamp: float  秒
    """
    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from sensor_msgs.msg import PointCloud2
    from nav_msgs.msg import Odometry

    db3_files = sorted(bag_dir.glob("*.db3"))
    if not db3_files:
        logger.error("Bag 目录 %s 中没有 .db3 文件", bag_dir)
        return None

    db3_path = str(db3_files[0])

    # 读取 metadata.yaml 确定 storage id
    metadata_path = bag_dir / "metadata.yaml"
    if metadata_path.exists():
        with open(metadata_path, "r") as f:
            meta = yaml.safe_load(f)
        storage_id = meta.get("rosbag2_bagfile_information", {}).get(
            "storage_identifier", "sqlite3"
        )
    else:
        storage_id = "sqlite3"

    reader = rosbag2_py.SequentialReader()
    storage_options = rosbag2_py.StorageOptions(
        uri=db3_path, storage_id=storage_id
    )
    converter_options = rosbag2_py.ConverterOptions(
        input_serialization_format="cdr",
        output_serialization_format="cdr",
    )
    reader.open(storage_options, converter_options)

    odometry_msg = None
    pointcloud_msg = None

    while reader.has_next():
        topic, data, t_ns = reader.read_next()
        t_sec = t_ns * 1e-9

        if topic == "/Odometry" and odometry_msg is None:
            odometry_msg = deserialize_message(data, Odometry)
        elif topic == "/cloud_registered" and pointcloud_msg is None:
            pointcloud_msg = deserialize_message(data, PointCloud2)

        if odometry_msg is not None and pointcloud_msg is not None:
            break

    if odometry_msg is None:
        logger.error("Bag %s: 未找到 /Odometry 消息", bag_dir.name)
        return None
    if pointcloud_msg is None:
        logger.error("Bag %s: 未找到 /cloud_registered 消息", bag_dir.name)
        return None

    # 提取位姿
    p = odometry_msg.pose.pose.position
    q = odometry_msg.pose.pose.orientation
    pose_xyz = np.array([p.x, p.y, p.z], dtype=np.float32)
    rpy = np.array(_quat_to_euler(q.x, q.y, q.z, q.w), dtype=np.float32)

    # 提取点云
    points_world = _pc2_to_numpy(pointcloud_msg)

    # 计算同步时间差
    t_odo = (odometry_msg.header.stamp.sec
             + odometry_msg.header.stamp.nanosec * 1e-9)
    t_pc = (pointcloud_msg.header.stamp.sec
            + pointcloud_msg.header.stamp.nanosec * 1e-9)
    sync_ms = abs(t_pc - t_odo) * 1000.0

    logger.info(
        "[Bag:%s] Odometry: pos=%s rpy=[%.1f,%.1f,%.1f]deg | "
        "Cloud: %d pts | sync=%.1fms",
        bag_dir.name,
        f"[{pose_xyz[0]:.1f},{pose_xyz[1]:.1f},{pose_xyz[2]:.1f}]",
        np.degrees(rpy[0]), np.degrees(rpy[1]), np.degrees(rpy[2]),
        len(points_world), sync_ms,
    )

    return pose_xyz, rpy, points_world, t_sec


# ============================================================
# 结果保存
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

    # 1. 去畸变点云 (机体系下视ROI) — 保存为 .npy
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
    """处理单个 rosbag: 读取 → 感知管线 → 保存"""
    bag_name = bag_dir.name

    logger.info("=" * 60)
    logger.info(" Processing: %s", bag_name)
    logger.info("=" * 60)

    # ---- Step 0: 读取 bag ----
    bag_data = read_rosbag(bag_dir)
    if bag_data is None:
        logger.error("Failed to read bag: %s", bag_name)
        return False

    pose_xyz, rpy, points_world, timestamp = bag_data

    # ---- Step 1: 机体系下视 ROI ----
    halss_pts, halss_stats = world_to_body_down_roi(points_world, pose_xyz, rpy, pcfg)
    logger.info(
        "  ROI: %d/%d pts | radius=%.1fm | z_down=[%.2f, %.2f]",
        halss_stats["output_points"], halss_stats["input_points"],
        halss_stats["roi_radius_m"],
        halss_stats.get("z_min_body", float("nan")),
        halss_stats.get("z_max_body", float("nan")),
    )

    if halss_stats["output_points"] < 10:
        logger.warning("  ROI 点太少 (%d), 跳过评估", halss_stats["output_points"])
        return False

    # ---- Step 2: HALSS Bayesian 评估 ----
    t0 = time.perf_counter()
    result = halss.evaluate(halss_pts)
    dt_halss = (time.perf_counter() - t0) * 1000

    if result is None:
        logger.warning("  HALSS evaluate returned None, 跳过")
        return False

    mean_val = result["mean_map"].mean() if result["mean_map"] is not None else float("nan")
    var_val = result["variance_map"].mean() if result["variance_map"] is not None else float("nan")
    logger.info("  HALSS: %.0fms | mean=%.4f var=%.6f", dt_halss, mean_val, var_val)

    # ---- Step 3: 语义图生成 ----
    t1 = time.perf_counter()
    bev_data = result.get("bev_data", result)
    sem_map = sem_gen.generate(bev_data)
    dt_sem = (time.perf_counter() - t1) * 1000

    safe_ratio = float(np.mean(sem_map == sem_gen.safe_id)) * 100
    logger.info("  Semantic: %.1fms | safe=%.1f%%", dt_sem, safe_ratio)

    # ---- Step 4: 提取中间结果 ----
    surf_norm = result.get("surf_norm_rgb")
    mean_map = result.get("mean_map")
    var_map = result.get("variance_map")
    safety_map = bev_data.get("safety_map_vis", result.get("safety_map_vis"))

    # ---- Step 5: 保存 ----
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

    # ---- 列出可用 bag ----
    all_bags = _list_bags()
    if not all_bags:
        logger.error("未找到任何 rosbag。请检查 %s", BAGS_ROOT)
        sys.exit(1)

    if args.list:
        print(f"\n可用 rosbag ({len(all_bags)} 个):")
        for i, bag in enumerate(all_bags):
            db3 = list(bag.glob("*.db3"))
            size_mb = sum(f.stat().st_size for f in db3) / (1024 * 1024)
            print(f"  [{i:3d}] {bag.name}  ({size_mb:.1f} MB)")
        print()
        return

    # ---- 确定要处理的 bag 列表 ----
    if args.bag_dir:
        bag_path = Path(args.bag_dir)
        if not bag_path.is_absolute():
            bag_path = PROJECT_ROOT / bag_path
        if not bag_path.exists():
            logger.error("Bag 目录不存在: %s", bag_path)
            sys.exit(1)
        bags_to_process = [bag_path]
    elif args.all:
        bags_to_process = all_bags
    else:
        # 默认: 处理第一个
        bags_to_process = [all_bags[0]]
        logger.info("未指定 --bag-dir, 使用第一个 bag: %s", all_bags[0].name)

    logger.info("将处理 %d 个 bag", len(bags_to_process))

    # ---- 初始化感知模块 (与 pipeline/nocontrol 一致) ----
    pcfg = CFG["perception"]

    logger.info("=" * 60)
    logger.info(" Initializing HALSS Bayesian evaluator...")
    logger.info("=" * 60)
    halss = HALSSBayesianEvaluator(pcfg)

    sem_gen = SemanticGenerator(_semantic_generator_cfg(pcfg))

    logger.info("  [HALSS] %s | grid_res=%d | mc_samples=%d | device=%s",
                "Bayesian (Unet_drop + MC Dropout)",
                halss.grid_res, halss.mc_samples, halss.device)
    logger.info("  [Semantic] safe_id=%d danger_id=%d size=%dx%d",
                sem_gen.safe_id, sem_gen.danger_id, sem_gen.img_w, sem_gen.img_h)

    # ---- 处理 ----
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir

    success_count = 0
    for bag_dir in bags_to_process:
        ok = process_bag(bag_dir, halss, sem_gen, pcfg, output_dir)
        if ok:
            success_count += 1

    # ---- 汇总 ----
    logger.info("=" * 60)
    logger.info(" Done: %d/%d bags processed successfully.", success_count, len(bags_to_process))
    logger.info(" Output: %s/", output_dir)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
