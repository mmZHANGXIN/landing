#!/usr/bin/env python3
"""
深度投影 & 补全离线调试 — 读取 rosbag → 深度管线 → 保存结果
=============================================================
与 pipeline.py / test_live_nocontrol.py 保持一致的深度管线逻辑:
  - world_to_body_down_roi  世界系点云→HALSS 机体系下视 ROI
  - HALSSBayesianEvaluator  生成 safe_mesh 源网格
  - DepthProjector.project_body_roi  HALSS ROI→稀疏深度
  - DepthCompletion / SparseNet (perception/sparse_depth_completion.py)  稀疏→稠密

输入: /home/orin/evelyn/landing/bags/capture_frames/frame_XXXX/ 下的 rosbag
输出: 指定目录下保存每帧的:
  - sparse_depth.npy           稀疏深度图 (128×128 float32, 米)
  - valid_mask.npy             稀疏深度有效像素 mask
  - depth_calib_frame.npz      SparseNet output_scale 标定输入
  - sparse_depth.png           稀疏深度可视化 (伪彩色)
  - dense_depth.npy            稠密深度图 (128×128 float32, 米) — 需已标定 SparseNet
  - dense_depth.png            稠密深度可视化 (伪彩色, 固定 0~dmax 色标)
  - depth_comparison.png       稀疏vs稠密对比图

用法:
  conda activate fylanding
  source /opt/ros/galactic/setup.bash
  python3 test_depth_projection_offline.py                          # 处理第一个 bag
  python3 test_depth_projection_offline.py --bag-dir bags/capture_frames/frame_0005_...  # 指定
  python3 test_depth_projection_offline.py --output-dir my_depth_results
  python3 test_depth_projection_offline.py --all
  python3 test_depth_projection_offline.py --list
  python3 test_depth_projection_offline.py --depth-output-scale <scale>  # 使用标定尺度复测
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
logger = logging.getLogger("DepthOffline")

# ============================================================
# 项目路径 & 配置
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

CONFIG_PATH = PROJECT_ROOT / "config" / "experiment_config.yaml"
with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    CFG = yaml.safe_load(f)

BAGS_ROOT = PROJECT_ROOT / "bags" / "capture_frames"


# ============================================================
# 命令行参数
# ============================================================
def _parse_args():
    parser = argparse.ArgumentParser(
        description="深度投影 & 补全离线调试 (从 rosbag 读取)"
    )
    parser.add_argument("--bag-dir", help="指定 rosbag 目录")
    parser.add_argument("--all", action="store_true", help="处理所有 bag")
    parser.add_argument("--output-dir", default="experiments/depth_offline",
                        help="输出目录 (默认: experiments/depth_offline)")
    parser.add_argument("--list", action="store_true", help="仅列出可用 bag")
    parser.add_argument("--no-completion", action="store_true",
                        help="跳过 SparseNet 深度补全, 仅保存稀疏深度")
    parser.add_argument("--depth-output-scale", type=float,
                        help="覆盖 depth_completion.output_scale, 用于标定后复测")
    parser.add_argument("--allow-uncalibrated-completion", action="store_true",
                        help="允许 output_scale=null 时运行 SparseNet, 仅用于复现/诊断错误输出")
    return parser.parse_args()


def _list_bags() -> list[Path]:
    if not BAGS_ROOT.exists():
        return []
    return sorted(d for d in BAGS_ROOT.iterdir()
                  if d.is_dir() and list(d.glob("*.db3")))


# ============================================================
# ROS2 Bag 读取 (与 test_halss_bayesian_offline.py 一致)
# ============================================================

def _quat_to_euler(x, y, z, w):
    sinr = 2.0 * (w * x + y * z)
    cosr = 1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(sinr, cosr)
    sinp = 2.0 * (w * y - z * x)
    pitch = (np.sign(sinp) * (np.pi / 2.0)
             if abs(sinp) >= 1.0 else np.arcsin(sinp))
    siny = 2.0 * (w * z + x * y)
    cosy = 1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(siny, cosy)
    return roll, pitch, yaw


def _pc2_to_numpy(msg) -> np.ndarray:
    field_offsets = {f.name: f.offset for f in msg.fields}
    if not all(k in field_offsets for k in ("x", "y", "z")):
        return np.empty((0, 3), dtype=np.float32)
    n = msg.width * msg.height
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


def read_rosbag(bag_dir: Path) -> tuple | None:
    """读取 rosbag → (pose_xyz, rpy, points_world, timestamp)"""
    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from sensor_msgs.msg import PointCloud2
    from nav_msgs.msg import Odometry

    db3_files = sorted(bag_dir.glob("*.db3"))
    if not db3_files:
        logger.error("Bag 目录 %s 中没有 .db3 文件", bag_dir)
        return None

    db3_path = str(db3_files[0])
    metadata_path = bag_dir / "metadata.yaml"
    if metadata_path.exists():
        with open(metadata_path, "r") as f:
            meta = yaml.safe_load(f)
        storage_id = meta.get("rosbag2_bagfile_information", {}).get(
            "storage_identifier", "sqlite3")
    else:
        storage_id = "sqlite3"

    reader = rosbag2_py.SequentialReader()
    storage_options = rosbag2_py.StorageOptions(uri=db3_path, storage_id=storage_id)
    converter_options = rosbag2_py.ConverterOptions(
        input_serialization_format="cdr", output_serialization_format="cdr")
    reader.open(storage_options, converter_options)

    odometry_msg = None
    pointcloud_msg = None

    while reader.has_next():
        topic, data, t_ns = reader.read_next()
        if topic == "/Odometry" and odometry_msg is None:
            odometry_msg = deserialize_message(data, Odometry)
        elif topic == "/cloud_registered" and pointcloud_msg is None:
            pointcloud_msg = deserialize_message(data, PointCloud2)
        if odometry_msg is not None and pointcloud_msg is not None:
            break

    if odometry_msg is None or pointcloud_msg is None:
        logger.error("Bag %s: 缺少 Odometry 或 PointCloud2 消息", bag_dir.name)
        return None

    p = odometry_msg.pose.pose.position
    q = odometry_msg.pose.pose.orientation
    pose_xyz = np.array([p.x, p.y, p.z], dtype=np.float32)
    rpy = np.array(_quat_to_euler(q.x, q.y, q.z, q.w), dtype=np.float32)
    points_world = _pc2_to_numpy(pointcloud_msg)

    logger.info(
        "[Bag:%s] pos=%s rpy=[%.1f,%.1f,%.1f]deg | %d pts",
        bag_dir.name,
        f"[{pose_xyz[0]:.1f},{pose_xyz[1]:.1f},{pose_xyz[2]:.1f}]",
        np.degrees(rpy[0]), np.degrees(rpy[1]), np.degrees(rpy[2]),
        len(points_world),
    )
    return pose_xyz, rpy, points_world, 0.0


# ============================================================
# 深度可视化
# ============================================================

def _depth_to_color(depth_m: np.ndarray, max_range: float = 30.0,
                    invalid_color=(64, 64, 64)) -> np.ndarray:
    """深度图 (米) → BGR 伪彩色, 固定 0~max_range 色标, 无效区域灰色"""
    vis = depth_m.astype(np.float32).copy()
    h, w = vis.shape
    bgr = np.zeros((h, w, 3), dtype=np.uint8)

    # 无效区域 (max_range 或 NaN) → 灰色
    invalid = (vis >= max_range - 0.01) | np.isnan(vis) | (vis <= 0.001)
    valid = ~invalid

    if valid.any():
        v_norm = np.clip(vis / max_range, 0.0, 1.0)
        colored = cv2.applyColorMap((v_norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
        bgr[valid] = colored[valid]

    bgr[invalid] = invalid_color
    return bgr


def _valid_depth_mask(depth_m: np.ndarray, max_range: float) -> np.ndarray:
    return np.isfinite(depth_m) & (depth_m > 0.01) & (depth_m < max_range - 0.01)


def _log_completion_overlap(
    sparse_depth: np.ndarray,
    dense_depth: np.ndarray,
    valid_mask: np.ndarray,
    max_range: float,
):
    """Log whether completion preserves observed sparse pixels well enough."""
    overlap = valid_mask.astype(bool) & _valid_depth_mask(dense_depth, max_range)
    if not overlap.any():
        logger.warning("  Dense depth sanity: no valid overlap with sparse pixels")
        return

    err = np.abs(dense_depth[overlap] - sparse_depth[overlap])
    sparse_med = float(np.median(sparse_depth[overlap]))
    dense_med = float(np.median(dense_depth[overlap]))
    err_med = float(np.median(err))
    err_p90 = float(np.percentile(err, 90))
    logger.info(
        "  Dense depth sanity @ sparse pixels: sparse_med=%.2fm dense_med=%.2fm "
        "abs_err_med=%.2fm p90=%.2fm",
        sparse_med, dense_med, err_med, err_p90,
    )

    if err_med > max(0.5, 0.2 * max(sparse_med, 0.01)):
        logger.warning(
            "  Dense depth disagrees with observed sparse depth. "
            "If output_scale is null, calibrate SparseNet before trusting dense_depth."
        )


def _put_lines(img: np.ndarray, lines: list[str], origin=(4, 18),
               line_step=18, color=(255, 255, 255)):
    x, y0 = origin
    for i, line in enumerate(lines):
        cv2.putText(
            img, line, (x, y0 + i * line_step),
            cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1,
        )


def _make_comparison(sparse_depth, dense_depth, max_range=30.0):
    """生成稀疏 vs 稠密深度对比图"""
    h, w = sparse_depth.shape

    sparse_vis = _depth_to_color(sparse_depth, max_range)
    cv2.putText(sparse_vis, "Sparse Depth", (4, 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    dense_vis = _depth_to_color(dense_depth, max_range)
    cv2.putText(dense_vis, "Dense Depth (SparseNet)", (4, 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    valid_sparse = (sparse_depth > 0.01) & (sparse_depth < max_range - 0.01)
    sparse_pct = valid_sparse.mean() * 100
    d_min = dense_depth[valid_sparse].min() if valid_sparse.any() else 0
    d_max = dense_depth[valid_sparse].max() if valid_sparse.any() else 0
    d_mean = dense_depth[valid_sparse].mean() if valid_sparse.any() else 0

    cv2.putText(sparse_vis, f"Valid={sparse_pct:.1f}%", (4, 34),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
    cv2.putText(dense_vis, f"Range=[{d_min:.1f},{d_max:.1f}]m mean={d_mean:.1f}m",
                (4, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)

    return np.hstack([sparse_vis, dense_vis])


def _make_skipped_comparison(sparse_depth, valid_mask, reason: str,
                             max_range=30.0):
    sparse_vis = _depth_to_color(sparse_depth, max_range)
    valid_sparse = valid_mask.astype(bool)
    sparse_pct = valid_sparse.mean() * 100
    sp_min = sparse_depth[valid_sparse].min() if valid_sparse.any() else float("nan")
    sp_max = sparse_depth[valid_sparse].max() if valid_sparse.any() else float("nan")
    _put_lines(
        sparse_vis,
        [
            "Sparse Depth",
            f"Valid={sparse_pct:.1f}% [{sp_min:.1f},{sp_max:.1f}]m",
        ],
    )

    h, w = sparse_depth.shape
    skipped = np.full((h, w, 3), 48, dtype=np.uint8)
    _put_lines(
        skipped,
        [
            "Dense skipped",
            reason,
            "Calibrate output_scale",
            "or pass --depth-output-scale",
        ],
        color=(220, 220, 220),
    )
    return np.hstack([sparse_vis, skipped])


# ============================================================
# 结果保存
# ============================================================
def save_depth_results(
    output_dir: Path,
    frame_name: str,
    sparse_depth: np.ndarray,
    valid_mask: np.ndarray,
    dense_depth: np.ndarray | None,
    max_range: float,
    completion_skip_reason: str | None = None,
):
    os.makedirs(output_dir, exist_ok=True)

    # 1. 稀疏深度 raw .npy
    np.save(str(output_dir / f"{frame_name}_sparse_depth.npy"), sparse_depth)
    logger.info("  Saved: %s_sparse_depth.npy (%.1f KB)",
                frame_name, sparse_depth.nbytes / 1024)

    # 2. 有效 mask + 标定帧
    np.save(str(output_dir / f"{frame_name}_valid_mask.npy"),
            valid_mask.astype(np.uint8))
    calib_arrays = {
        "sparse_depth": sparse_depth.astype(np.float32),
        "valid_mask": valid_mask.astype(np.uint8),
    }
    if dense_depth is not None:
        calib_arrays["dense_depth"] = dense_depth.astype(np.float32)
    np.savez_compressed(
        str(output_dir / f"{frame_name}_depth_calib_frame.npz"),
        **calib_arrays,
    )
    logger.info("  Saved: %s_valid_mask.npy and %s_depth_calib_frame.npz",
                frame_name, frame_name)

    # 3. 稀疏深度可视化
    sparse_vis = _depth_to_color(sparse_depth, max_range)
    valid_sparse = valid_mask.astype(bool)
    sp_pct = valid_sparse.mean() * 100
    sp_min = sparse_depth[valid_sparse].min() if valid_sparse.any() else float("nan")
    sp_max = sparse_depth[valid_sparse].max() if valid_sparse.any() else float("nan")
    cv2.putText(sparse_vis,
                f"Sparse Depth | valid={sp_pct:.1f}% range=[{sp_min:.1f},{sp_max:.1f}]m",
                (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    cv2.imwrite(str(output_dir / f"{frame_name}_sparse_depth.png"), sparse_vis)
    logger.info("  Saved: %s_sparse_depth.png (valid=%.1f%%)", frame_name, sp_pct)

    # 4. 稠密深度
    if dense_depth is not None:
        np.save(str(output_dir / f"{frame_name}_dense_depth.npy"), dense_depth)
        logger.info("  Saved: %s_dense_depth.npy (%.1f KB)",
                    frame_name, dense_depth.nbytes / 1024)

        dense_vis = _depth_to_color(dense_depth, max_range)
        d_valid = _valid_depth_mask(dense_depth, max_range)
        d_min = dense_depth[d_valid].min() if d_valid.any() else 0
        d_max = dense_depth[d_valid].max() if d_valid.any() else 0
        d_mean = dense_depth[d_valid].mean() if d_valid.any() else 0
        cv2.putText(dense_vis,
                    f"Dense Depth (SparseNet) | range=[{d_min:.1f},{d_max:.1f}]m mean={d_mean:.1f}m",
                    (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
        cv2.imwrite(str(output_dir / f"{frame_name}_dense_depth.png"), dense_vis)
        logger.info("  Saved: %s_dense_depth.png", frame_name)

        # 5. 对比图
        comparison = _make_comparison(sparse_depth, dense_depth, max_range)
        cv2.imwrite(str(output_dir / f"{frame_name}_depth_comparison.png"), comparison)
        logger.info("  Saved: %s_depth_comparison.png", frame_name)
    else:
        for suffix in ("dense_depth.npy", "dense_depth.png"):
            stale = output_dir / f"{frame_name}_{suffix}"
            if stale.exists():
                stale.unlink()
                logger.info("  Removed stale: %s", stale.name)
        reason = completion_skip_reason or "completion unavailable"
        comparison = _make_skipped_comparison(
            sparse_depth, valid_mask, reason, max_range
        )
        cv2.imwrite(str(output_dir / f"{frame_name}_depth_comparison.png"), comparison)
        logger.info("  Saved: %s_depth_comparison.png (completion skipped)", frame_name)


# ============================================================
# 单帧处理
# ============================================================
def process_bag(
    bag_dir: Path,
    halss,
    dproj,
    depth_completion,
    pcfg: dict,
    max_range: float,
    output_dir: Path,
    use_completion: bool,
    completion_skip_reason: str | None,
):
    bag_name = bag_dir.name
    logger.info("=" * 60)
    logger.info(" Processing: %s", bag_name)
    logger.info("=" * 60)

    # Step 0: 读取 bag
    bag_data = read_rosbag(bag_dir)
    if bag_data is None:
        return False

    pose_xyz, rpy, points_world, _ = bag_data

    from perception.halss_preprocess import world_to_body_down_roi

    # Step 1: 与 test_live_nocontrol.py 一致的 HALSS 预处理链路
    halss_pts, halss_stats = world_to_body_down_roi(points_world, pose_xyz, rpy, pcfg)
    logger.info(
        "  HALSS ROI: %d/%d pts | radius=%.1fm z=[%.2f, %.2f]",
        halss_stats["output_points"],
        halss_stats["input_points"],
        halss_stats["roi_radius_m"],
        halss_stats["z_min_body"],
        halss_stats["z_max_body"],
    )

    t_h0 = time.perf_counter()
    bev = halss.evaluate(halss_pts)
    dt_halss = (time.perf_counter() - t_h0) * 1000
    if bev is not None:
        source_shape = bev["safe_mesh"].shape
        safe_pct = float(np.mean(bev["safe_mesh"])) * 100.0
        logger.info(
            "  HALSS eval: %.0fms | safe_mesh=%s safe=%.1f%%",
            dt_halss, source_shape, safe_pct,
        )
    else:
        source_shape = None
        logger.warning(
            "  HALSS eval: %.0fms | no safe_mesh, depth falls back to output grid shape",
            dt_halss,
        )

    # Step 2: 深度投影 — HALSS body ROI → sparse depth
    t0 = time.perf_counter()
    sparse_depth = dproj.project_body_roi(halss_pts, source_shape=source_shape)
    dt_proj = (time.perf_counter() - t0) * 1000

    valid_mask = (sparse_depth < max_range) & (sparse_depth > 0.01)
    n_valid = valid_mask.sum()
    n_total = sparse_depth.size

    sp_min = sparse_depth[valid_mask].min() if n_valid > 0 else 0
    sp_max = sparse_depth[valid_mask].max() if n_valid > 0 else 0
    sp_mean = sparse_depth[valid_mask].mean() if n_valid > 0 else 0

    logger.info("  Sparse depth: %.0fms | valid=%d/%d (%.1f%%) | range=[%.1f,%.1f]m mean=%.1fm",
                dt_proj, n_valid, n_total, n_valid / n_total * 100,
                sp_min, sp_max, sp_mean)

    # Step 3: 深度补全 (SparseNet)
    dense_depth = None
    if use_completion and depth_completion is not None:
        if getattr(depth_completion, "output_scale", None) is None:
            logger.warning(
                "  Depth completion running WITHOUT output_scale calibration. "
                "Expected failure mode: inverse-depth output decodes near %.1fm.",
                max_range,
            )
        t1 = time.perf_counter()
        try:
            dense_depth = depth_completion.complete(sparse_depth, valid_mask)
            dt_compl = (time.perf_counter() - t1) * 1000
            d_valid = _valid_depth_mask(dense_depth, max_range)
            d_min = dense_depth[d_valid].min() if d_valid.any() else 0
            d_max = dense_depth[d_valid].max() if d_valid.any() else 0
            d_mean = dense_depth[d_valid].mean() if d_valid.any() else 0
            logger.info("  Dense depth:  %.0fms | range=[%.1f,%.1f]m mean=%.1fm",
                        dt_compl, d_min, d_max, d_mean)
            _log_completion_overlap(sparse_depth, dense_depth, valid_mask, max_range)
        except Exception as e:
            logger.warning("  Depth completion failed: %s", e)
    elif not use_completion:
        logger.info("  Depth completion: skipped (%s)",
                    completion_skip_reason or "--no-completion")
    elif depth_completion is None:
        logger.info("  Depth completion: unavailable; saved sparse depth and calibration frame only")

    # Step 4: 保存
    save_depth_results(
        output_dir, bag_name, sparse_depth, valid_mask, dense_depth, max_range,
        completion_skip_reason=completion_skip_reason,
    )
    logger.info("  Done: %s → %s/", bag_name, output_dir)
    return True


# ============================================================
# 主逻辑
# ============================================================
def main():
    args = _parse_args()
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

    # 确定要处理的 bag
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
        bags_to_process = [all_bags[0]]
        logger.info("未指定 --bag-dir, 使用第一个 bag: %s", all_bags[0].name)

    logger.info("将处理 %d 个 bag", len(bags_to_process))

    # ---- 初始化 HALSS + 深度投影 (与 pipeline/nocontrol 一致) ----
    pcfg = CFG["perception"]
    dcfg_proj = CFG["depth_projection"]
    max_range = float(dcfg_proj.get("max_range", 30.0))
    grid_cells = int(dcfg_proj.get("grid_cells", 128))

    logger.info("=" * 60)
    logger.info(" Initializing HALSS + DepthProjector...")
    logger.info("=" * 60)

    from perception.halss_bayesian import HALSSBayesianEvaluator
    from perception.depth_projection import DepthProjector

    halss = HALSSBayesianEvaluator(pcfg)
    dproj = DepthProjector(
        img_width=grid_cells,
        img_height=grid_cells,
        max_range=max_range,
        mode=dcfg_proj.get("mode", "perspective"),
        backend="numpy",  # 离线调试强制 numpy, 避免 CUDA 依赖
        fx=dcfg_proj.get("fx"),
        fy=dcfg_proj.get("fy"),
        cx=dcfg_proj.get("cx"),
        cy=dcfg_proj.get("cy"),
        R_I_to_C=dcfg_proj.get("R_I_to_C"),
    )
    logger.info(
        "  HALSS: grid_res=%d mc=%s",
        halss.grid_res,
        pcfg.get("mc_samples"),
    )
    logger.info("  DepthProjector: %dx%d mode=%s backend=%s fx=%.1f fy=%.1f max_range=%.0fm",
                dproj.out_w, dproj.out_h, dproj.mode, dproj.backend,
                dproj.fx, dproj.fy, max_range)

    # ---- 初始化深度补全 (SparseNet) ----
    use_completion = not args.no_completion
    depth_completion = None
    completion_skip_reason = "--no-completion" if args.no_completion else None

    if use_completion:
        dcfg_compl = dict(CFG["depth_completion"])
        if args.depth_output_scale is not None:
            dcfg_compl["output_scale"] = float(args.depth_output_scale)

        output_scale = dcfg_compl.get("output_scale")
        if output_scale is None and not args.allow_uncalibrated_completion:
            logger.warning(
                "  SparseNet output_scale is null; skip dense completion by default. "
                "Use --depth-output-scale <scale> after calibration, or "
                "--allow-uncalibrated-completion to reproduce the uncalibrated 29m failure."
            )
            use_completion = False
            completion_skip_reason = "output_scale is null"
        elif output_scale is None:
            logger.warning("  SparseNet output_scale is null; running uncalibrated diagnostic output.")
        else:
            logger.info("  SparseNet output_scale=%.6g", float(output_scale))

    if use_completion:
        weight_path_value = dcfg_compl.get("weight_path")
        weight_path = Path(weight_path_value) if weight_path_value else None
        if weight_path is not None and not weight_path.is_absolute():
            weight_path = PROJECT_ROOT / weight_path

        if weight_path is not None and weight_path.exists():
            try:
                from perception.sparse_depth_completion import DepthCompletion
                dcfg_compl["weight_path"] = str(weight_path)
                depth_completion = DepthCompletion(dcfg_compl)
                logger.info(
                    "  SparseNet: loaded OK | encoding=%s output_scale=%s",
                    dcfg_compl.get("input_encoding", "inverse_unit"),
                    dcfg_compl.get("output_scale"),
                )
            except Exception as e:
                logger.warning("  SparseNet init failed: %s. 仅保存稀疏深度。", e)
        else:
            logger.warning("  SparseNet weight not found: %s. 仅保存稀疏深度。",
                           weight_path or "(none)")

    # ---- 处理 ----
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir

    success = 0
    for bag_dir in bags_to_process:
        ok = process_bag(bag_dir, halss, dproj, depth_completion, pcfg, max_range,
                         output_dir, use_completion, completion_skip_reason)
        if ok:
            success += 1

    logger.info("=" * 60)
    logger.info(" Done: %d/%d bags processed. Output: %s/",
                success, len(bags_to_process), output_dir)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
