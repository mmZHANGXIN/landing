#!/usr/bin/env python3
"""
HALSS Bayesian 实时调试脚本 (原始 Livox 点云) — 订阅 /livox/lidar → 感知 → 实时可视化
====================================================================================
与 test_halss_bayesian_offline_raw.py 共用同一套坐标变换和感知管线,
但改为订阅实时 ROS2 话题, 并在 OpenCV 窗口中实时显示二值安全语义图。

实时显示:
  - 左: 语义图 (绿色=安全, 红色=危险)
  - 右: HALSS 二值语义图 (白色=安全, 黑色=危险)
  - 状态栏: FPS, 安全比例, 点云数量

用法:
  conda activate fylanding
  source /opt/ros/galactic/setup.bash
  python3 test_halss_bayesian_live_raw.py

  # 可选参数:
  python3 test_halss_bayesian_live_raw.py --save-dir experiments/halss_live_raw
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import threading
import logging
from pathlib import Path
from collections import deque

import numpy as np
import cv2
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("HALSSLiveRaw")

# ============================================================
# 项目路径 & 配置
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

CONFIG_PATH = PROJECT_ROOT / "config" / "experiment_config.yaml"
with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    CFG = yaml.safe_load(f)

from perception.halss_bayesian import HALSSBayesianEvaluator
from perception.halss_preprocess import _rot_zyx
from perception.semantic_generator import SemanticGenerator


# ============================================================
# 命令行参数
# ============================================================
def _parse_args():
    parser = argparse.ArgumentParser(
        description="HALSS Bayesian 实时调试 (原始 Livox 点云 + 可视化)"
    )
    parser.add_argument(
        "--save-dir", default=None,
        help="可选: 保存每帧结果到指定目录",
    )
    parser.add_argument(
        "--no-display", action="store_true",
        help="禁用 OpenCV 显示 (仅处理, 不显示)",
    )
    return parser.parse_args()


def _semantic_generator_cfg(pcfg: dict) -> dict:
    obs_cfg = CFG.get("observation", {})
    return {
        **pcfg,
        "img_width": int(obs_cfg.get("img_width", pcfg.get("img_width", 128))),
        "img_height": int(obs_cfg.get("img_height", pcfg.get("img_height", 128))),
    }


# ============================================================
# 坐标变换 (与 test_halss_bayesian_offline_raw.py 完全一致)
# ============================================================

def _get_rotation_body_from_lidar(cfg: dict) -> np.ndarray:
    """从配置读取 R_body_from_lidar, 若不存在则用简单翻转 + pitch 构造."""
    matrix_cfg = cfg.get("halss_lidar_rotation_body_from_lidar")
    if matrix_cfg is not None:
        return np.array(matrix_cfg, dtype=np.float32)

    pitch_deg = float(cfg.get("halss_lidar_pitch_down_deg", 26.0))
    pitch = np.deg2rad(pitch_deg)
    cp, sp = np.cos(pitch), np.sin(pitch)
    R_pitch = np.array([
        [cp, 0, sp],
        [0,  1, 0],
        [-sp, 0, cp],
    ], dtype=np.float32)
    R_flip = np.array([
        [1,  0,  0],
        [0, -1,  0],
        [0,  0, -1],
    ], dtype=np.float32)
    return R_flip @ R_pitch


def raw_lidar_to_body_down_roi(
    points_lidar: np.ndarray,
    imu_rpy: np.ndarray | None,
    cfg: dict,
) -> tuple[np.ndarray, dict]:
    """将原始 Livox 点云 (LiDAR 系) 转换到 HALSS 机体系下视 ROI。

    处理流程:
      1. 完整安装旋转 R_body_from_lidar
      2. IMU 姿态补偿 (leveling)
      3. FOV 过滤: 保留射向地面的点
      4. ROI 过滤: 半径 + 高度范围
    """
    fov_half_deg = float(cfg.get("halss_lidar_fov_half_deg", 45.0))
    fov_half_rad = np.deg2rad(fov_half_deg)

    stats = {
        "input_points": 0,
        "output_points": 0,
        "fov_passed": 0,
        "roi_radius_m": float(cfg.get("halss_roi_radius_body", cfg.get("roi_radius_world", 25.0))),
        "min_down_m": float(cfg.get("halss_min_down_m", 0.05)),
        "max_down_m": float(cfg.get("halss_max_down_m", cfg.get("halss_roi_max_down_m", 30.0))),
        "imu_available": imu_rpy is not None,
        "fov_half_deg": fov_half_deg,
    }

    if points_lidar is None:
        return np.empty((0, 3), dtype=np.float32), stats

    pts = np.asarray(points_lidar, dtype=np.float32)
    if pts.size == 0:
        return np.empty((0, 3), dtype=np.float32), stats
    if pts.ndim == 2 and pts.shape[1] >= 3:
        pts = pts[:, :3]

    stats["input_points"] = int(len(pts))

    # Step 1: LiDAR 系 → 机体系
    R_bl = _get_rotation_body_from_lidar(cfg)
    pts_body = pts @ R_bl.T
    lidar_z_body = R_bl[:, 2]

    # Step 2: IMU 姿态补偿
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
        lidar_z_body = R_level @ lidar_z_body

    # Step 3: FOV 过滤
    norms = np.linalg.norm(pts_body, axis=1)
    valid_norm = norms > 1e-8
    if valid_norm.any():
        dirs = np.zeros_like(pts_body)
        dirs[valid_norm] = pts_body[valid_norm] / norms[valid_norm, np.newaxis]
        body_down_vector = np.array([0, 0, 1], dtype=np.float32)  # 机体下方向
        # cos_angle = dirs @ lidar_z_body  # cos(点方向与主轴夹角)
        cos_angle = dirs @ body_down_vector  # cos(点方向与机体下夹角)
        in_fov = (cos_angle >= np.cos(fov_half_rad)) & valid_norm
    else:
        in_fov = np.zeros(len(pts_body), dtype=bool)

    pts_fov = pts_body[in_fov]
    stats["fov_passed"] = int(np.sum(in_fov))

    if len(pts_fov) == 0:
        stats["output_points"] = 0
        return np.empty((0, 3), dtype=np.float32), stats

    # Step 4: ROI 过滤
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
    })
    return roi, stats


# ============================================================
# ROS2 实时数据桥接
# ============================================================
try:
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import PointCloud2, Imu
    _ROS_AVAILABLE = True
except ImportError:
    _ROS_AVAILABLE = False
    logger.error("rclpy 不可用。请 source /opt/ros/galactic/setup.bash")


class RawLivoxBridge(Node):
    """订阅原始 Livox 点云 + IMU, 缓存最新帧."""

    def __init__(self):
        super().__init__("halss_live_raw_bridge")

        self.latest_lidar = None    # (N,3) np.ndarray
        self.latest_imu_rpy = None  # (3,) np.ndarray
        self._lidar_lock = threading.Lock()
        self._imu_lock = threading.Lock()
        self._new_lidar = threading.Event()

        # 订阅 /livox/lidar (live 模式下是 PointCloud2)
        self._lidar_sub = self.create_subscription(
            PointCloud2, "/livox/lidar", self._lidar_cb, 10,
        )
        # 订阅 /livox/imu
        self._imu_sub = self.create_subscription(
            Imu, "/livox/imu", self._imu_cb, 10,
        )
        logger.info("[Bridge] Subscribed /livox/lidar + /livox/imu")

    def _lidar_cb(self, msg: PointCloud2):
        pts = self._pc2_to_numpy(msg)
        with self._lidar_lock:
            self.latest_lidar = pts
        self._new_lidar.set()

    def _imu_cb(self, msg: Imu):
        q = msg.orientation
        if not (q.x == 0.0 and q.y == 0.0 and q.z == 0.0 and q.w == 0.0):
            rpy = np.array(self._quat_to_euler(q.x, q.y, q.z, q.w), dtype=np.float32)
            with self._imu_lock:
                self.latest_imu_rpy = rpy

    def grab(self) -> tuple[np.ndarray | None, np.ndarray | None]:
        """线程安全地获取最新的 lidar + imu 快照."""
        with self._lidar_lock:
            lidar = self.latest_lidar
        with self._imu_lock:
            imu = self.latest_imu_rpy
        return lidar, imu

    def wait_for_lidar(self, timeout: float = 1.0) -> bool:
        """等待新的 lidar 帧到达."""
        return self._new_lidar.wait(timeout)

    def clear_new_lidar(self):
        self._new_lidar.clear()

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
        pts = np.column_stack((arr["x"], arr["y"], arr["z"])).astype(np.float32, copy=False)
        return pts[np.isfinite(pts).all(axis=1)]


# ============================================================
# 可视化
# ============================================================

def make_display_frame(
    sem_map: np.ndarray,
    safety_map: np.ndarray,
    sem_gen: SemanticGenerator,
    safe_pct: float,
    dt_ms: float,
    fps: float,
    lidar_n: int,
    roi_n: int,
    imu_ok: bool,
) -> np.ndarray:
    """构造并排显示的 OpenCV 帧.

    左: 语义图 (colorized)
    右: 二值安全图 (白色=安全)
    底栏: 状态文字
    """
    h, w = sem_map.shape[:2]

    # 左: 语义图
    sem_rgb = sem_gen.colorize(sem_map)
    h_sem, w_sem = sem_rgb.shape[:2]

    # 右: HALSS 二值安全图 (白色=安全, 黑色=危险), 统一尺寸
    if safety_map is not None:
        if safety_map.shape[:2] != (h_sem, w_sem):
            safety_map = cv2.resize(
                safety_map, (w_sem, h_sem), interpolation=cv2.INTER_NEAREST,
            )
        halss_rgb = cv2.cvtColor(safety_map, cv2.COLOR_GRAY2BGR)
    else:
        halss_rgb = np.zeros((h_sem, w_sem, 3), dtype=np.uint8)

    # 并排
    display = np.hstack([sem_rgb, halss_rgb])

    # 放大以便观看 (x3)
    display = cv2.resize(display, None, fx=3, fy=3, interpolation=cv2.INTER_NEAREST)

    # 底栏
    dh, dw = display.shape[:2]
    bar = np.zeros((40, dw, 3), dtype=np.uint8)
    display = np.vstack([display, bar])

    text_lines = [
        f"Safe: {safe_pct:.0f}% | {dt_ms:.0f}ms | FPS: {fps:.1f}",
        f"LiDAR: {lidar_n} pts | ROI: {roi_n} pts | IMU: {'OK' if imu_ok else 'NO'}",
        "Semantic              HALSS Binary",
    ]
    y0 = dh + 14
    for i, txt in enumerate(text_lines):
        cv2.putText(display, txt, (5, y0 + i * 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)

    return display


# ============================================================
# 主循环
# ============================================================
def main():
    args = _parse_args()

    if not _ROS_AVAILABLE:
        logger.error("rclpy 不可用, 请先 source ROS2 环境.")
        sys.exit(1)

    # ---- 初始化 ROS2 ----
    rclpy.init()
    bridge = RawLivoxBridge()

    # ---- 初始化感知模块 ----
    pcfg = CFG["perception"]

    logger.info("=" * 60)
    logger.info(" Initializing HALSS Bayesian evaluator (live raw mode)...")
    logger.info("=" * 60)
    halss = HALSSBayesianEvaluator(pcfg)
    sem_gen = SemanticGenerator(_semantic_generator_cfg(pcfg))

    logger.info("  [HALSS] %s | grid_res=%d | mc_samples=%d | device=%s",
                "Bayesian (Unet_drop + MC Dropout)",
                halss.grid_res, halss.mc_samples, halss.device)
    logger.info("  [Semantic] safe_id=%d danger_id=%d size=%dx%d",
                sem_gen.safe_id, sem_gen.danger_id, sem_gen.img_w, sem_gen.img_h)

    # ---- 可选: 保存目录 ----
    save_dir = None
    if args.save_dir:
        save_dir = Path(args.save_dir)
        if not save_dir.is_absolute():
            save_dir = PROJECT_ROOT / save_dir
        os.makedirs(save_dir, exist_ok=True)
        logger.info("  Saving frames to: %s/", save_dir)

    # ---- FPS 追踪 ----
    dt_window = deque(maxlen=30)
    frame_idx = 0
    show = not args.no_display

    if show:
        cv2.namedWindow("HALSS Live Raw", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("HALSS Live Raw", 900, 500)

    logger.info("=" * 60)
    logger.info(" Running. 按 'q' 退出, 's' 保存当前帧.")
    logger.info("=" * 60)

    executor = rclpy.executors.SingleThreadedExecutor()
    executor.add_node(bridge)

    try:
        while rclpy.ok():
            # Spin ROS2 (非阻塞)
            executor.spin_once(timeout_sec=0.01)

            # 等待新 lidar 帧
            if not bridge.wait_for_lidar(timeout=5.0):
                logger.warning("超时: 超过 5 秒未收到 /livox/lidar 数据")
                continue
            bridge.clear_new_lidar()

            lidar_pts, imu_rpy = bridge.grab()
            if lidar_pts is None or len(lidar_pts) == 0:
                continue

            t0 = time.perf_counter()

            # Step 1: LiDAR → 机体 ROI
            halss_pts, stats = raw_lidar_to_body_down_roi(lidar_pts, imu_rpy, pcfg)
            roi_n = stats["output_points"]
            if roi_n < 10:
                logger.debug("ROI 点太少 (%d), 跳过", roi_n)
                continue

            # Step 2: HALSS Bayesian
            result = halss.evaluate(halss_pts)
            if result is None:
                continue

            # Step 3: 语义图
            bev_data = result.get("bev_data", result)
            sem_map = sem_gen.generate(bev_data)

            safe_pct = float(np.mean(sem_map == sem_gen.safe_id)) * 100
            dt_ms = (time.perf_counter() - t0) * 1000

            # FPS
            dt_window.append(dt_ms)
            avg_ms = np.mean(dt_window) if dt_window else dt_ms
            fps = 1000.0 / avg_ms if avg_ms > 0 else 0

            # 提取二值安全图
            safety_map = bev_data.get("safety_map_vis", result.get("safety_map_vis"))

            logger.debug(
                "Frame %d: %d→%d pts | safe=%.1f%% | %.0fms | fps=%.1f",
                frame_idx, len(lidar_pts), roi_n, safe_pct, dt_ms, fps,
            )

            # ---- 显示 ----
            if show:
                display = make_display_frame(
                    sem_map, safety_map, sem_gen,
                    safe_pct, dt_ms, fps,
                    len(lidar_pts), roi_n,
                    imu_rpy is not None,
                )
                cv2.imshow("HALSS Live Raw", display)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    logger.info("用户按 'q' 退出.")
                    break
                elif key == ord('s') and save_dir is not None:
                    # 保存当前帧
                    fname = f"frame_{frame_idx:04d}"
                    cv2.imwrite(str(save_dir / f"{fname}_semantic.png"),
                                sem_gen.colorize(sem_map))
                    if safety_map is not None:
                        cv2.imwrite(str(save_dir / f"{fname}_binary.png"), safety_map)
                    np.save(str(save_dir / f"{fname}_roi_cloud.npy"), halss_pts)
                    logger.info("  Saved frame %d to %s/", frame_idx, save_dir)

            # ---- 自动保存 ----
            if save_dir is not None and frame_idx == 0:
                fname = f"frame_{frame_idx:04d}"
                cv2.imwrite(str(save_dir / f"{fname}_semantic.png"),
                            sem_gen.colorize(sem_map))
                if safety_map is not None:
                    cv2.imwrite(str(save_dir / f"{fname}_binary.png"), safety_map)
                np.save(str(save_dir / f"{fname}_roi_cloud.npy"), halss_pts)

            frame_idx += 1

    except KeyboardInterrupt:
        logger.info("Interrupted by user.")
    finally:
        if show:
            cv2.destroyAllWindows()
        bridge.destroy_node()
        rclpy.shutdown()
        logger.info("Done. Processed %d frames.", frame_idx)


if __name__ == "__main__":
    main()
