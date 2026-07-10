#!/usr/bin/env python3
"""
HALSS Bayesian 在线调试脚本 — Mid360 → FAST-LIO → HALSS Bayesian → 语义图
======================================================================
与 pipeline.py / test_live_nocontrol.py 保持一致的感知逻辑:
  - HALSSBayesianEvaluator (perception/halss_bayesian.py)
  - SemanticGenerator (perception/semantic_generator.py)

实时可视化窗口 (5 个):
  ┌──────────────┬──────────────┬──────────────┐
  │ 原始点云      │ 去畸变点云    │ 地表拟合图    │
  │ (俯视+高程)   │ (FAST-LIO)   │ (法向量RGB)   │
  ├──────────────┼──────────────┼──────────────┤
  │ 均值图        │ 方差图        │ 语义图        │
  │ (MC Dropout)  │ (不确定性)    │ (安全=绿)     │
  ├──────────────┴──────────────┴──────────────┤
  │ 二值安全图 (着陆区: 白色=安全, 黑色=危险)    │
  └─────────────────────────────────────────────┘

用法:
  source /opt/ros/galactic/setup.bash
  source ~/ros2_ws/install/setup.bash  # FAST-LIO2 workspace
  python3 test_halss_bayesian_live.py

可选参数:
  --no-raw-cloud        不订阅原始 Livox 点云 (若 topic 不可用)
  --save-frames         保存每帧可视化图片
  --save-dir DIR        保存目录 (默认 experiments/halss_debug/)
  --max-frames N        处理 N 帧后退出 (0 表示无限)
  --duration-sec SEC    运行指定秒数后退出 (0 表示无限)
  --require-gpu         强制要求 GPU, 不可用时退出
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import threading
import logging
from pathlib import Path

import numpy as np
import cv2
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("HALSSBayesianDebug")

# ============================================================
# 项目路径 & 配置
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

CONFIG_PATH = PROJECT_ROOT / "config" / "experiment_config.yaml"
with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    CFG = yaml.safe_load(f)

# ---- 感知模块 (与 pipeline / nocontrol 一致) ----
from perception.halss_bayesian import HALSSBayesianEvaluator
from perception.halss_preprocess import world_to_body_down_roi
from perception.semantic_generator import SemanticGenerator


# ============================================================
# 命令行参数
# ============================================================
def _parse_args():
    parser = argparse.ArgumentParser(
        description="HALSS Bayesian 在线调试 (Mid360 + FAST-LIO)"
    )
    parser.add_argument(
        "--no-raw-cloud", action="store_true",
        help="不订阅原始 Livox 点云",
    )
    parser.add_argument(
        "--save-frames", action="store_true",
        help="保存每帧可视化图片",
    )
    parser.add_argument(
        "--save-dir", default="experiments/halss_debug",
        help="保存目录 (默认: experiments/halss_debug)",
    )
    parser.add_argument(
        "--max-frames", type=int, default=0,
        help="处理 N 帧后退出 (0=无限)",
    )
    parser.add_argument(
        "--duration-sec", type=float, default=0.0,
        help="运行指定秒数后退出 (0=无限)",
    )
    parser.add_argument(
        "--require-gpu", action="store_true",
        help="强制要求 GPU (pipeline 行为)",
    )
    return parser.parse_args()


# ============================================================
# ROS2 数据桥接 (与 test_live_nocontrol.py 一致)
# ============================================================
try:
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import PointCloud2
    from nav_msgs.msg import Odometry
    _ROS_AVAILABLE = True
except ImportError:
    _ROS_AVAILABLE = False
    logger.error("rclpy 不可用。请 source ROS2 环境。")


class FastLIOBridge(Node):
    """订阅 FAST-LIO 输出 + 可选原始 Livox 点云"""

    def __init__(self, subscribe_raw: bool = True):
        super().__init__("halss_bayesian_debug_bridge")

        # FAST-LIO 去畸变点云 + 位姿
        self.latest_odom = None
        self.latest_cloud = None
        self._odom_lock = threading.Lock()
        self._cloud_lock = threading.Lock()

        self._odom_sub = self.create_subscription(
            Odometry, "/Odometry", self._odom_cb, 10,
        )
        self._cloud_sub = self.create_subscription(
            PointCloud2, "/cloud_registered", self._cloud_cb, 10,
        )
        logger.info("[Bridge] Subscribed /Odometry + /cloud_registered")

        # 原始 Livox 点云 (可选)
        self.latest_raw_cloud = None
        self._raw_cloud_lock = threading.Lock()
        self._has_raw = False
        if subscribe_raw:
            try:
                self._raw_cloud_sub = self.create_subscription(
                    PointCloud2, "/livox/lidar", self._raw_cloud_cb, 10,
                )
                self._has_raw = True
                logger.info("[Bridge] Subscribed /livox/lidar (raw)")
            except Exception as e:
                logger.warning("[Bridge] /livox/lidar 不可用: %s", e)

    def _odom_cb(self, msg: Odometry):
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        roll, pitch, yaw = self._quat_to_euler(q.x, q.y, q.z, q.w)
        with self._odom_lock:
            self.latest_odom = (
                msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9,
                np.array([p.x, p.y, p.z], dtype=np.float32),
                np.array([roll, pitch, yaw], dtype=np.float32),
            )

    def _cloud_cb(self, msg: PointCloud2):
        pts = self._pc2_to_numpy(msg)
        with self._cloud_lock:
            self.latest_cloud = (
                msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9,
                pts,
            )

    def _raw_cloud_cb(self, msg: PointCloud2):
        pts = self._pc2_to_numpy(msg)
        with self._raw_cloud_lock:
            self.latest_raw_cloud = pts

    @staticmethod
    def _quat_to_euler(x, y, z, w):
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

    @staticmethod
    def _pc2_to_numpy(msg: PointCloud2) -> np.ndarray:
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

    def grab_deskewed(self):
        with self._odom_lock:
            odom = self.latest_odom
        with self._cloud_lock:
            cloud = self.latest_cloud
        return odom, cloud

    def grab_raw(self):
        with self._raw_cloud_lock:
            return self.latest_raw_cloud


# ============================================================
# 可视化 (OpenCV 多窗口, 与 pipeline 显示逻辑一致)
# ============================================================
class HALSSDebugDisplay:
    """实时显示 HALSS Bayesian 各阶段中间结果"""

    def __init__(self, save_dir: str = None, save_frames: bool = False):
        self.save_dir = Path(save_dir) if save_dir else None
        self.save_frames = save_frames
        self._frame_idx = 0

        if self.save_frames and self.save_dir:
            os.makedirs(self.save_dir, exist_ok=True)

        # 窗口命名
        cv2.namedWindow("1. Raw Cloud", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("1. Raw Cloud", 420, 420)
        cv2.namedWindow("2. Deskewed Cloud", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("2. Deskewed Cloud", 420, 420)
        cv2.namedWindow("3. Surface Normal", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("3. Surface Normal", 420, 420)
        cv2.namedWindow("4. Mean Map (MC)", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("4. Mean Map (MC)", 420, 420)
        cv2.namedWindow("5. Variance Map", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("5. Variance Map", 420, 420)
        cv2.namedWindow("6. Semantic Map", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("6. Semantic Map", 420, 420)
        cv2.namedWindow("7. Binary Safety (Landing Zones)", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("7. Binary Safety (Landing Zones)", 420, 420)

    def _to_display(self, data, target_size=420, cmap=None) -> np.ndarray:
        """将任意数据转为 BGR 显示图"""
        if data is None:
            return np.zeros((target_size, target_size, 3), dtype=np.uint8)

        if data.ndim == 2:
            if data.dtype == bool:
                vis = (data.astype(np.uint8) * 255)
            else:
                vmin = np.nanmin(data)
                vmax = np.nanmax(data)
                if vmax - vmin < 1e-8:
                    vis = np.zeros_like(data, dtype=np.uint8)
                else:
                    vis = ((data - vmin) / (vmax - vmin) * 255).astype(np.uint8)
            if cmap is not None:
                vis = cv2.applyColorMap(vis, cmap)
            else:
                vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
        elif data.ndim == 3:
            if data.shape[2] == 3:
                vis = data.copy()
                if vis.dtype != np.uint8:
                    vmin = np.nanmin(vis)
                    vmax = np.nanmax(vis)
                    if vmax - vmin > 1e-8:
                        vis = ((vis - vmin) / (vmax - vmin) * 255).astype(np.uint8)
                    else:
                        vis = np.zeros_like(vis, dtype=np.uint8)
            else:
                vis = np.zeros((data.shape[0], data.shape[1], 3), dtype=np.uint8)
        else:
            vis = np.zeros((target_size, target_size, 3), dtype=np.uint8)

        h, w = vis.shape[:2]
        scale = min(target_size / h, target_size / w)
        new_w, new_h = int(w * scale), int(h * scale)
        return cv2.resize(vis, (new_w, new_h))

    def _render_cloud_bev(self, pts: np.ndarray, target_size=420) -> np.ndarray:
        """点云俯视图 (BEV), 颜色=高度"""
        canvas = np.zeros((target_size, target_size, 3), dtype=np.uint8)
        if pts is None or len(pts) == 0:
            return canvas

        margin = 2
        x_min, x_max = pts[:, 0].min(), pts[:, 0].max()
        y_min, y_max = pts[:, 1].min(), pts[:, 1].max()
        z_min, z_max = pts[:, 2].min(), pts[:, 2].max()

        if x_max - x_min < 0.1:
            x_min -= 5; x_max += 5
        if y_max - y_min < 0.1:
            y_min -= 5; y_max += 5
        if z_max - z_min < 0.1:
            z_min -= 1; z_max += 1

        x_range = x_max - x_min
        y_range = y_max - y_min
        scale_factor = min(
            (target_size - 2 * margin) / x_range,
            (target_size - 2 * margin) / y_range,
        )

        xs = ((pts[:, 0] - x_min) * scale_factor + margin).astype(np.int32)
        ys = ((pts[:, 1] - y_min) * scale_factor + margin).astype(np.int32)

        zs = ((pts[:, 2] - z_min) / (z_max - z_min) * 255).astype(np.uint8)
        colors = cv2.applyColorMap(zs, cv2.COLORMAP_JET)

        for i in range(len(pts)):
            xi, yi = xs[i], ys[i]
            if 0 <= xi < target_size and 0 <= yi < target_size:
                canvas[yi, xi] = colors[i, 0]

        return canvas

    def update(
        self,
        raw_cloud: np.ndarray | None,
        deskewed_cloud: np.ndarray | None,
        surf_norm_rgb: np.ndarray | None,
        mean_map: np.ndarray | None,
        var_map: np.ndarray | None,
        safety_map: np.ndarray | None,
        semantic_map: np.ndarray | None,
    ):
        """刷新所有窗口"""
        # 1. 原始点云
        raw_bev = self._render_cloud_bev(raw_cloud) if raw_cloud is not None else (
            np.zeros((420, 420, 3), dtype=np.uint8)
        )
        cv2.putText(
            raw_bev,
            f"Raw Cloud ({len(raw_cloud) if raw_cloud is not None else 0} pts)",
            (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1,
        )

        # 2. 去畸变点云
        deskewed_bev = self._render_cloud_bev(deskewed_cloud) if deskewed_cloud is not None else (
            np.zeros((420, 420, 3), dtype=np.uint8)
        )
        cv2.putText(
            deskewed_bev,
            f"HALSS ROI (body/down, {len(deskewed_cloud) if deskewed_cloud is not None else 0} pts)",
            (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1,
        )

        # 3. 地表法向量拟合图
        surf_vis = self._to_display(surf_norm_rgb)
        cv2.putText(
            surf_vis, "Surface Normal (Delaunay+Interp)",
            (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1,
        )

        # 4. MC Dropout 均值图
        mean_vis = self._to_display(mean_map, cmap=cv2.COLORMAP_INFERNO)
        cv2.putText(
            mean_vis, "Mean Map (MC Dropout)",
            (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1,
        )

        # 5. MC Dropout 方差图 (不确定性)
        var_vis = self._to_display(var_map, cmap=cv2.COLORMAP_HOT)
        cv2.putText(
            var_vis, "Variance Map (Uncertainty)",
            (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1,
        )

        # 7. 二值安全图
        if safety_map is not None:
            safety_bgr = cv2.cvtColor(safety_map, cv2.COLOR_GRAY2BGR)
        else:
            safety_bgr = np.zeros((420, 420, 3), dtype=np.uint8)
        cv2.putText(
            safety_bgr,
            "Binary Safety (White=Safe Landing)",
            (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1,
        )

        # 6. 语义图 (安全=绿, 危险=红)
        if semantic_map is not None:
            sem_rgb = np.zeros((*semantic_map.shape, 3), dtype=np.uint8)
            safe_id = CFG["perception"].get("safe_class_id", 1)
            danger_id = CFG["perception"].get("danger_class_id", 9)
            sem_rgb[semantic_map == safe_id] = [0, 200, 0]
            sem_rgb[semantic_map == danger_id] = [0, 0, 200]
            safe_pct = (semantic_map == safe_id).mean() * 100
            cv2.putText(
                sem_rgb,
                f"Semantic Map (Safe={safe_pct:.0f}%)",
                (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1,
            )
        else:
            sem_rgb = np.zeros((420, 420, 3), dtype=np.uint8)
        sem_rgb = cv2.resize(
            sem_rgb, (420, 420), interpolation=cv2.INTER_NEAREST,
        )

        # 显示所有窗口
        cv2.imshow("1. Raw Cloud", raw_bev)
        cv2.imshow("2. Deskewed Cloud", deskewed_bev)
        cv2.imshow("3. Surface Normal", surf_vis)
        cv2.imshow("4. Mean Map (MC)", mean_vis)
        cv2.imshow("5. Variance Map", var_vis)
        cv2.imshow("6. Semantic Map", sem_rgb)
        cv2.imshow("7. Binary Safety (Landing Zones)", safety_bgr)

        # 保存
        if self.save_frames and self.save_dir:
            idx_str = f"{self._frame_idx:06d}"
            cv2.imwrite(str(self.save_dir / f"{idx_str}_1_raw_cloud.png"), raw_bev)
            cv2.imwrite(str(self.save_dir / f"{idx_str}_2_deskewed.png"), deskewed_bev)
            cv2.imwrite(str(self.save_dir / f"{idx_str}_3_surf_normal.png"), surf_vis)
            cv2.imwrite(str(self.save_dir / f"{idx_str}_4_mean_map.png"), mean_vis)
            cv2.imwrite(str(self.save_dir / f"{idx_str}_5_var_map.png"), var_vis)
            cv2.imwrite(str(self.save_dir / f"{idx_str}_6_semantic.png"), sem_rgb)
            cv2.imwrite(str(self.save_dir / f"{idx_str}_7_safety_binary.png"), safety_bgr)
            self._frame_idx += 1


# ============================================================
# 主逻辑
# ============================================================
def main():
    args = _parse_args()

    if not _ROS_AVAILABLE:
        logger.error("ROS2 (rclpy) 不可用。请 source ROS2 环境后重试。")
        sys.exit(1)

    # ---- 配置 ----
    pcfg = CFG["perception"]

    # GPU 要求
    if args.require_gpu:
        pcfg["require_gpu"] = True

    # ---- 初始化 HALSS Bayesian (与 pipeline/nocontrol 一致) ----
    logger.info("=" * 60)
    logger.info(" Initializing HALSS Bayesian evaluator...")
    logger.info("=" * 60)
    halss = HALSSBayesianEvaluator(pcfg)

    # ---- 初始化 SemanticGenerator (与 pipeline/nocontrol 一致) ----
    sem_gen = SemanticGenerator(pcfg)

    # ---- ROS2 初始化 ----
    rclpy.init(args=None)
    bridge = FastLIOBridge(subscribe_raw=not args.no_raw_cloud)

    # ---- 可视化 ----
    display = HALSSDebugDisplay(
        save_dir=args.save_dir,
        save_frames=args.save_frames,
    )

    logger.info("=" * 60)
    logger.info(" Pipeline ready. Waiting for LiDAR data...")
    logger.info("  [HALSS] %s | grid_res=%d | mc_samples=%d",
                "Bayesian (Unet_drop + MC Dropout)", halss.grid_res, halss.mc_samples)
    logger.info("  [Semantic] safe_id=%d danger_id=%d size=%dx%d",
                sem_gen.safe_id, sem_gen.danger_id, sem_gen.img_w, sem_gen.img_h)
    logger.info("  [Raw cloud] %s",
                "enabled" if bridge._has_raw else "disabled (no /livox/lidar)")
    logger.info("  按键: q=退出, s=保存当前帧")
    logger.info("=" * 60)

    seq = 0
    start_wall = time.perf_counter()
    t_all = {"halss": [], "semantic": [], "total": []}

    try:
        while rclpy.ok():
            # 检查退出条件
            if args.duration_sec > 0.0 and (time.perf_counter() - start_wall) >= args.duration_sec:
                logger.info("Duration reached: %.1fs, frames=%d", args.duration_sec, seq)
                break
            if args.max_frames > 0 and seq >= args.max_frames:
                logger.info("Max frames reached: %d", seq)
                break

            rclpy.spin_once(bridge, timeout_sec=0.01)

            odom, cloud = bridge.grab_deskewed()
            if odom is None or cloud is None:
                time.sleep(0.05)
                continue

            t_odom, pose_xyz, rpy = odom
            t_cloud, pts = cloud
            roll, pitch, yaw = rpy
            sync_ms = abs(t_cloud - t_odom) * 1000.0

            # 丢弃不同步帧
            max_sync_ms = float(CFG.get("runtime", {}).get("max_cloud_odom_sync_ms", 100.0))
            if sync_ms > max_sync_ms:
                logger.warning("Drop stale frame: sync=%.0fms > %.0fms", sync_ms, max_sync_ms)
                continue

            seq += 1
            t0 = time.perf_counter()

            # ---- Step 1: 取机体系下视 ROI，再做 HALSS Bayesian 评估 ----
            halss_pts, halss_stats = world_to_body_down_roi(pts, pose_xyz, rpy, pcfg)
            t_h0 = time.perf_counter()
            result = halss.evaluate(halss_pts)
            t_h1 = time.perf_counter()

            # ---- Step 2: 语义图生成 (与 pipeline/nocontrol 一致) ----
            t_s0 = time.perf_counter()
            if result is not None:
                bev_data = result["bev_data"]
                sem_map = sem_gen.generate(bev_data)
                safe_ratio = float(np.mean(sem_map == pcfg["safe_class_id"]))
            else:
                sem_map = np.full(
                    (pcfg.get("img_height", 128), pcfg.get("img_width", 128)),
                    pcfg["danger_class_id"], dtype=np.uint8,
                )
                safe_ratio = 0.0
            t_s1 = time.perf_counter()

            # ---- 耗时统计 ----
            dt_halss = (t_h1 - t_h0) * 1000
            dt_semantic = (t_s1 - t_s0) * 1000
            dt_total = (time.perf_counter() - t0) * 1000
            t_all["halss"].append(dt_halss)
            t_all["semantic"].append(dt_semantic)
            t_all["total"].append(dt_total)

            # ---- 日志 ----
            if result is not None:
                mean_val = result["mean_map"].mean()
                var_val = result["variance_map"].mean()
            else:
                mean_val = var_val = float("nan")

            logger.info(
                "[%04d] HALSS=%.0fms sem=%.0fms total=%.0fms "
                "sync=%.0fms | pts=%d roi=%d safe=%.1f%% "
                "mean=%.4f var=%.6f | yaw=%.1fdeg",
                seq, dt_halss, dt_semantic, dt_total, sync_ms,
                len(pts), halss_stats["output_points"], safe_ratio * 100, mean_val, var_val,
                np.degrees(yaw),
            )

            # ---- 可视化 ----
            raw_cloud = bridge.grab_raw() if bridge._has_raw else None

            surf_norm = result["surf_norm_rgb"] if result else None
            mean_map = result["mean_map"] if result else None
            var_map = result["variance_map"] if result else None
            safety_map = result["safety_map_vis"] if result else None

            display.update(
                raw_cloud=raw_cloud,
                deskewed_cloud=halss_pts,
                surf_norm_rgb=surf_norm,
                mean_map=mean_map,
                var_map=var_map,
                safety_map=safety_map,
                semantic_map=sem_map,
            )

            # 按键处理
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                logger.info("用户按 q 退出。")
                break
            elif key == ord("s"):
                # 手动保存当前帧
                if not os.path.exists(args.save_dir):
                    os.makedirs(args.save_dir, exist_ok=True)
                display._frame_idx = seq
                display.save_frames = True
                display.update(
                    raw_cloud=raw_cloud,
                    deskewed_cloud=halss_pts,
                    surf_norm_rgb=surf_norm,
                    mean_map=mean_map,
                    var_map=var_map,
                    safety_map=safety_map,
                    semantic_map=sem_map,
                )
                display.save_frames = args.save_frames  # 恢复原设置
                logger.info("[Save] 当前帧已保存至 %s/", args.save_dir)

    except KeyboardInterrupt:
        logger.info("Interrupted.")
    finally:
        cv2.destroyAllWindows()
        rclpy.shutdown()

        # 汇总统计
        def avg(x):
            return sum(x) / len(x) if x else 0
        logger.info("=" * 60)
        logger.info(
            " Summary: frames=%d | HALSS avg=%.0fms | Semantic avg=%.1fms | Total avg=%.0fms",
            seq, avg(t_all["halss"]), avg(t_all["semantic"]), avg(t_all["total"]),
        )
        if t_all["halss"]:
            sorted_h = sorted(t_all["halss"])
            p50 = sorted_h[len(sorted_h) // 2]
            p95 = sorted_h[int(len(sorted_h) * 0.95)]
            logger.info(" HALSS P50=%.0fms P95=%.0fms max=%.0fms", p50, p95, max(t_all["halss"]))
        logger.info(" Done.")


if __name__ == "__main__":
    main()
