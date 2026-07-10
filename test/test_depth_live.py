#!/usr/bin/env python3
"""
MID360 原始点云 → 下视深度投影 实时可视化
==========================================
与 test_live_nocontrol_raw.py 一致的点云处理逻辑:
  - 订阅 /livox/lidar (PointCloud2) 原始 MID360 点云 (LiDAR 系)
  - 订阅 /livox/imu (sensor_msgs/Imu) 用于姿态估计
  - LiDAR 系 → 机体系: R_body_from_lidar = R_axis @ Ry(116°), t=[0.13,0,0.08]
  - IMU yaw-only leveling (水平校正)
  - LiDAR +x ±45° 方位角过滤
  - ROI 半径+高度范围过滤
  - 下视深度投影 (DepthProjector.project_body_roi, 与 HALSS 管线一致)
  - 无深度补全

可视化: 深度相机风格 (灰度 + JET 伪彩色双窗口)

用法:
  source /opt/ros/galactic/setup.bash
  source ~/livox_ws/install/setup.bash
  conda activate fylanding
  python3 test_depth_live.py
  python3 test_depth_live.py --colormap jet          # JET 伪彩色
  python3 test_depth_live.py --save-dir frames --save-every 10
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

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("DepthLive")

# ============================================================
# 项目路径 & 配置
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

CONFIG_PATH = PROJECT_ROOT / "config" / "experiment_config.yaml"
with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    CFG = yaml.safe_load(f)

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, Imu


# ============================================================
# 坐标变换 (与 test_live_nocontrol_raw.py 完全一致)
# ============================================================

def _quat_to_euler(x, y, z, w):
    """Quaternion → roll, pitch, yaw (rad)."""
    sinr = 2.0 * (w * x + y * z)
    cosr = 1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(sinr, cosr)
    sinp = 2.0 * (w * y - z * x)
    pitch = np.sign(sinp) * (np.pi / 2.0) if abs(sinp) >= 1.0 else np.arcsin(sinp)
    siny = 2.0 * (w * z + x * y)
    cosy = 1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(siny, cosy)
    return roll, pitch, yaw


def _rot_zyx(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Body-to-world rotation (ZYX Euler)."""
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ], dtype=np.float32)


def _get_rotation_body_from_lidar() -> np.ndarray:
    """LiDAR 系 → 机体系 旋转矩阵 (pitch=116°).

    R_body_from_lidar = R_axis @ Ry(116°)
    R_axis = diag(1, -1, -1): LiDAR (x=fwd, y=left, z=up) → 机体 (x=fwd, y=right, z=down)
    """
    pitch = np.deg2rad(116.0)
    cp, sp = np.cos(pitch), np.sin(pitch)
    R_axis = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=np.float32)
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=np.float32)
    return R_axis @ Ry


# 模块级常量
_R_BL = _get_rotation_body_from_lidar()
_T_BL = np.array([0.13, 0.0, 0.08], dtype=np.float32)


def raw_lidar_to_body_down_roi(
    points_lidar: np.ndarray,
    imu_rpy: np.ndarray | None,
    cfg: dict,
) -> tuple[np.ndarray, dict]:
    """原始 Livox 点云 (LiDAR 系) → 机体系下视 ROI (x=前, y=右, z=下).

    与 test_live_nocontrol_raw.py 完全一致的处理流程:
      1. pts_body = pts @ R_bl.T + t_bl  (LiDAR → 机体)
      2. IMU yaw-only leveling (水平校正)
      3. LiDAR +x ±45° 方位角过滤 (前向有效探测锥)
      4. ROI: 半径 + 高度范围过滤
    """
    azimuth_half_deg = float(cfg.get("halss_lidar_fov_half_deg", 45.0))
    azimuth_half_rad = np.deg2rad(azimuth_half_deg)

    stats = {
        "input_points": 0,
        "output_points": 0,
        "fov_passed": 0,
        "roi_radius_m": float(cfg.get("halss_roi_radius_body", 25.0)),
        "min_down_m": float(cfg.get("halss_min_down_m", 0.05)),
        "max_down_m": float(cfg.get("halss_max_down_m", 30.0)),
        "imu_available": imu_rpy is not None,
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

    # Step 1: LiDAR 系 → 机体系
    pts_body = pts @ _R_BL.T + _T_BL

    # Step 2: IMU yaw-only leveling
    if imu_rpy is not None:
        roll, pitch, yaw = float(imu_rpy[0]), float(imu_rpy[1]), float(imu_rpy[2])
        yaw_offset = np.deg2rad(float(cfg.get("halss_lidar_yaw_offset_deg", 0.0)))
        yaw = yaw + yaw_offset
        R_level = _rot_zyx(0.0, 0.0, -yaw)
        pts_body = pts_body @ R_level.T

    # Step 3: LiDAR +x 方位角过滤 (前向有效探测锥)
    R_eff = _R_BL.copy()
    if imu_rpy is not None:
        R_eff = _rot_zyx(0.0, 0.0, -yaw) @ _R_BL
    pts_lidar_aligned = (pts_body - _T_BL) @ R_eff
    x_l = pts_lidar_aligned[:, 0]
    y_l = pts_lidar_aligned[:, 1]
    z_l = pts_lidar_aligned[:, 2]
    azimuth = np.arctan2(y_l, x_l)
    in_fov = (
        (x_l > 0.0)
        & (np.abs(azimuth) <= azimuth_half_rad)
        & (z_l >= 0.0)
    )
    pts_fov = pts_body[in_fov]
    stats["fov_passed"] = int(np.sum(in_fov))

    if len(pts_fov) == 0:
        stats["output_points"] = 0
        return np.empty((0, 3), dtype=np.float32), stats

    # Step 4: ROI 过滤 (半径 + 高度)
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
    stats["output_points"] = int(len(roi))
    return roi, stats


# ============================================================
# ROS2 数据桥接 (原始 Livox MID360)
# ============================================================

class RawLivoxBridge(Node):
    """订阅原始 Livox MID360: /livox/lidar (PointCloud2) + /livox/imu (Imu)."""

    def __init__(self):
        super().__init__("depth_live_raw_bridge")
        self.latest_cloud = None
        self.latest_imu = None
        self._cloud_lock = threading.Lock()
        self._imu_lock = threading.Lock()

        self.create_subscription(PointCloud2, "/livox/lidar", self._cloud_cb, 10)
        self.create_subscription(Imu, "/livox/imu", self._imu_cb, 10)
        logger.info("[Bridge] Subscribed /livox/lidar (PointCloud2) + /livox/imu")

    @staticmethod
    def _pc2_to_xyz(msg: PointCloud2) -> np.ndarray:
        """PointCloud2 → (N,3) float32 (x,y,z)."""
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

    def _cloud_cb(self, msg: PointCloud2):
        pts = self._pc2_to_xyz(msg)
        if len(pts) > 0:
            ts = float(msg.header.stamp.sec) + float(msg.header.stamp.nanosec) * 1e-9
            with self._cloud_lock:
                self.latest_cloud = (ts, pts)

    def _imu_cb(self, msg: Imu):
        q = msg.orientation
        if not (q.x == 0.0 and q.y == 0.0 and q.z == 0.0 and q.w == 0.0):
            rpy = np.array(_quat_to_euler(q.x, q.y, q.z, q.w), dtype=np.float32)
            ts = float(msg.header.stamp.sec) + float(msg.header.stamp.nanosec) * 1e-9
            with self._imu_lock:
                self.latest_imu = (ts, rpy)

    def grab(self) -> tuple:
        """获取最新 cloud + imu (线程安全)."""
        with self._cloud_lock:
            cloud = self.latest_cloud
        with self._imu_lock:
            imu = self.latest_imu
        return cloud, imu


# ============================================================
# 深度相机风格可视化
# ============================================================

class DepthCameraDisplay:
    """深度相机风格双窗口显示: 灰度 + JET 伪彩色."""

    def __init__(self, width: int, height: int, max_depth_m: float,
                 window_name: str = "Downward Depth", enable_jet: bool = False):
        self.width = int(width)
        self.height = int(height)
        self.max_depth_m = float(max_depth_m)
        self.window_name = window_name
        self.enable_jet = enable_jet

        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, self.width, self.height)
        if self.enable_jet:
            self.jet_name = window_name + " (JET)"
            cv2.namedWindow(self.jet_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self.jet_name, self.width, self.height)

    def _render_gray(self, depth_m: np.ndarray) -> np.ndarray:
        """归一化深度 → 灰度图 (远=白, 近=黑)."""
        d = np.nan_to_num(
            depth_m.astype(np.float32, copy=False),
            nan=self.max_depth_m, posinf=self.max_depth_m, neginf=0.0,
        )
        d_norm = np.clip(d / self.max_depth_m, 0.0, 1.0)
        return (d_norm * 255.0).astype(np.uint8)

    def _render_jet(self, depth_m: np.ndarray) -> np.ndarray:
        """归一化深度 → JET 伪彩色 BGR (近=红, 远=蓝, 无效=灰)."""
        d = np.nan_to_num(
            depth_m.astype(np.float32, copy=False),
            nan=self.max_depth_m, posinf=self.max_depth_m, neginf=0.0,
        )
        d_norm = np.clip(d / self.max_depth_m, 0.0, 1.0)
        colored = (d_norm * 255.0).astype(np.uint8)
        jet = cv2.applyColorMap(colored, cv2.COLORMAP_JET)
        # 无效区域 (30m) 显示为灰色
        invalid = d >= (self.max_depth_m - 0.01)
        jet[invalid] = (64, 64, 64)
        return jet

    def update(self, depth_m: np.ndarray, status: str) -> int:
        gray = self._render_gray(depth_m)
        cv2.putText(gray, status, (10, 24), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, 255, 1, cv2.LINE_AA)
        cv2.imshow(self.window_name, gray)

        if self.enable_jet:
            jet = self._render_jet(depth_m)
            cv2.putText(jet, status, (10, 24), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.imshow(self.jet_name, jet)

        return cv2.waitKey(1) & 0xFF

    def close(self):
        cv2.destroyWindow(self.window_name)
        if self.enable_jet:
            cv2.destroyWindow(self.jet_name)


# ============================================================
# 命令行参数
# ============================================================

def _parse_args():
    parser = argparse.ArgumentParser(
        description="MID360 原始点云 → 下视深度投影 实时可视化"
    )
    parser.add_argument("--colormap", choices=("gray", "jet"), default="gray",
                        help="深度图配色: gray=灰度 (默认), jet=JET 伪彩色 (双窗口)")
    parser.add_argument("--max-depth-m", type=float, default=30.0,
                        help="深度图最大距离 (米), 默认 30.0")
    parser.add_argument("--roi-radius-m", type=float, default=None,
                        help="覆盖 ROI 半径 (米), 默认读取 config")
    parser.add_argument("--azimuth-half-deg", type=float, default=None,
                        help="覆盖 LiDAR +x 方位角半角 (度), 默认 45.0")
    parser.add_argument("--display-width", type=int, default=480,
                        help="显示窗口宽度, 默认 480")
    parser.add_argument("--display-height", type=int, default=480,
                        help="显示窗口高度, 默认 480")
    parser.add_argument("--save-dir", default=None,
                        help="可选保存目录, 存 *_depth.png 和 *_depth_m.npy")
    parser.add_argument("--save-every", type=int, default=0,
                        help="每 N 帧保存一次, 0=不保存")
    parser.add_argument("--max-frames", type=int, default=0,
                        help="处理 N 帧后退出, 0=无限")
    parser.add_argument("--duration-sec", type=float, default=0.0,
                        help="运行时长 (秒), 0=无限")
    return parser.parse_args()


# ============================================================
# 主逻辑
# ============================================================

def main():
    args = _parse_args()

    # ---- 配置覆盖 ----
    pcfg = dict(CFG["perception"])
    dcfg = dict(CFG["depth_projection"])
    max_range = float(args.max_depth_m)

    if args.roi_radius_m is not None:
        pcfg["halss_roi_radius_body"] = float(args.roi_radius_m)
    if args.azimuth_half_deg is not None:
        pcfg["halss_lidar_fov_half_deg"] = float(args.azimuth_half_deg)

    enable_jet = (args.colormap == "jet")
    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)

    # ---- 初始化 DepthProjector (与 pipeline 一致) ----
    from perception.depth_projection import DepthProjector

    dproj = DepthProjector(
        img_width=int(dcfg.get("grid_cells", 128)),
        img_height=int(dcfg.get("grid_cells", 128)),
        max_range=max_range,
        mode=dcfg.get("mode", "perspective"),
        backend="numpy",  # 实时调试用 numpy，避免 CUDA 初始化开销
        fx=dcfg.get("fx"),
        fy=dcfg.get("fy"),
        cx=dcfg.get("cx"),
        cy=dcfg.get("cy"),
        R_I_to_C=dcfg.get("R_I_to_C"),
    )

    display = DepthCameraDisplay(
        width=args.display_width,
        height=args.display_height,
        max_depth_m=max_range,
        window_name="Downward Depth (m)",
        enable_jet=enable_jet,
    )

    # ---- ROS2 初始化 ----
    rclpy.init()
    bridge = RawLivoxBridge()

    roi_radius = float(pcfg.get("halss_roi_radius_body", 25.0))
    azimuth_half = float(pcfg.get("halss_lidar_fov_half_deg", 45.0))

    logger.info(
        "=" * 60 + "\n"
        "  MID360 Raw → Downward Depth Projection\n"
        "  LiDAR→Body: R_axis @ Ry(116°), t=[0.13, 0, 0.08]\n"
        "  Azimuth filter: LiDAR +x ±%.0f°\n"
        "  ROI: radius=%.0fm, z=[%.2f, %.0f]m\n"
        "  Depth: %dx%d, max=%.0fm, colormap=%s\n" +
        "=" * 60,
        azimuth_half, roi_radius,
        float(pcfg.get("halss_min_down_m", 0.05)),
        float(pcfg.get("halss_max_down_m", 30.0)),
        dproj.out_w, dproj.out_h, max_range,
        args.colormap,
    )

    shown = 0
    t_start = time.perf_counter()

    try:
        while rclpy.ok():
            if args.duration_sec > 0.0 and (time.perf_counter() - t_start) >= args.duration_sec:
                logger.info("Duration reached: %.1fs, frames=%d", args.duration_sec, shown)
                break

            rclpy.spin_once(bridge, timeout_sec=0.01)
            cloud, imu = bridge.grab()
            if cloud is None:
                time.sleep(0.05)
                continue

            t_cloud, pts = cloud
            imu_rpy = imu[1] if imu is not None else None

            shown += 1
            t0 = time.perf_counter()

            # Step 1: 原始 LiDAR → 机体系下视 ROI
            body_pts, roi_stats = raw_lidar_to_body_down_roi(pts, imu_rpy, pcfg)

            # Step 2: 下视深度投影
            t_proj0 = time.perf_counter()
            depth_m = dproj.project_body_roi(body_pts, source_shape=None)
            dt_proj = (time.perf_counter() - t_proj0) * 1000

            # 统计
            valid = (depth_m > 0.01) & (depth_m < max_range - 0.01)
            n_valid = int(valid.sum())
            n_total = depth_m.size
            valid_ratio = n_valid / n_total if n_total > 0 else 0.0
            if n_valid > 0:
                d_min = float(depth_m[valid].min())
                d_mean = float(depth_m[valid].mean())
                d_max = float(depth_m[valid].max())
            else:
                d_min = d_mean = d_max = float("nan")

            dt_total = (time.perf_counter() - t0) * 1000
            imu_text = f"yaw={np.degrees(imu_rpy[2]):.0f}°" if imu_rpy is not None else "IMU:n/a"

            status = (
                f"#{shown} {imu_text} | proj={dt_proj:.0f}ms total={dt_total:.0f}ms | "
                f"pts={roi_stats['output_points']}/{roi_stats['input_points']}"
            )
            key = display.update(depth_m, status)

            logger.info(
                "[%04d] %s | roi=%d/%d fov=%d | "
                "depth valid=%d/%d(%.1f%%) [%.2f,%.2f,%.2f]m | "
                "proj=%.0fms total=%.0fms",
                shown, imu_text,
                roi_stats["output_points"], roi_stats["input_points"],
                roi_stats["fov_passed"],
                n_valid, n_total, valid_ratio * 100,
                d_min, d_mean, d_max,
                dt_proj, dt_total,
            )

            # 保存
            if args.save_dir and args.save_every > 0 and shown % args.save_every == 0:
                stem = os.path.join(args.save_dir, f"{shown:06d}")
                cv2.imwrite(f"{stem}_depth_gray.png",
                            display._render_gray(depth_m))
                if enable_jet:
                    cv2.imwrite(f"{stem}_depth_jet.png",
                                display._render_jet(depth_m))
                np.save(f"{stem}_depth_m.npy", depth_m.astype(np.float32))
                logger.info("  Saved: %s_*", stem)

            if key in (ord("q"), 27):
                break
            if args.max_frames > 0 and shown >= args.max_frames:
                break

    except KeyboardInterrupt:
        logger.info("Interrupted.")
    finally:
        display.close()
        cv2.destroyAllWindows()
        rclpy.shutdown()
        logger.info("Done. Frames processed: %d", shown)


if __name__ == "__main__":
    main()
