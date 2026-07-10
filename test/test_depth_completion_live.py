#!/usr/bin/env python3
"""
深度补全 + 语义三窗口实时调试
=================================
Mid360 → HALSS语义 + 稀疏深度 → SparseNet(PyTorch) → 稠密深度 → 三窗口 + ZMQ

视窗: 左=稀疏深度(点云z-buffer) | 中=SparseNet稠密补全 | 右=HALSS语义(安全/危险)

与 test_live_nocontrol_raw.py 感知管线完全一致, 仅显示改为三窗口 + 去飞控。

用法:
  source /opt/ros/galactic/setup.bash
  source ~/livox_ws/install/setup.bash
  conda activate fylanding
  LD_LIBRARY_PATH=/opt/ros/galactic/lib:/opt/ros/galactic/lib/aarch64-linux-gnu:/opt/ros/galactic/opt/rcl_logging_spdlog/lib:$LD_LIBRARY_PATH \
  python scripts/test_depth_completion_live.py [--no-zmq] [--no-display]
"""

import argparse, sys, os, time, threading, logging
import numpy as np
import cv2
import yaml
from typing import Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("DepthCompLive")

PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CFG_PATH = os.path.join(PROJ_ROOT, "config", "experiment_config.yaml")
with open(CFG_PATH) as f:
    CFG = yaml.safe_load(f)
sys.path.insert(0, PROJ_ROOT)

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, Imu
from perception.halss_preprocess import _rot_zyx
from perception.halss_bayesian import HALSSBayesianEvaluator
from perception.semantic_generator import SemanticGenerator
from perception.sparse_depth_completion import DepthCompletion

try:
    import zmq
    HAS_ZMQ = True
except ImportError:
    HAS_ZMQ = False


# ============================================================
# LiDAR → 机体
# ============================================================

def _get_rotation_body_from_lidar():
    pitch = np.deg2rad(116.0)
    cp, sp = np.cos(pitch), np.sin(pitch)
    R_axis = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=np.float32)
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=np.float32)
    return R_axis @ Ry

_R_BL = _get_rotation_body_from_lidar()
_T_BL = np.array([0.13, 0.0, 0.08], dtype=np.float32)


def lidar_to_body_roi(pts_lidar, imu_rpy, cfg):
    """LiDAR → 机体 (仅 FOV 方位角过滤, 不裁剪 ROI 半径)."""
    az_half = np.deg2rad(float(cfg.get("halss_lidar_fov_half_deg", 45.0)))
    min_d = float(cfg.get("halss_min_down_m", 0.05))
    max_d = float(cfg.get("halss_max_down_m", 30.0))

    pts = np.asarray(pts_lidar, dtype=np.float32)
    if pts.size == 0: return np.empty((0, 3), dtype=np.float32)
    if pts.ndim == 1: pts = pts[:3].reshape(1, 3)
    elif pts.ndim == 2 and pts.shape[1] >= 3: pts = pts[:, :3]

    pts_body = pts @ _R_BL.T + _T_BL
    yaw = 0.0
    if imu_rpy is not None:
        roll, pitch, yaw = float(imu_rpy[0]), float(imu_rpy[1]), float(imu_rpy[2])
        yaw += np.deg2rad(float(cfg.get("halss_lidar_yaw_offset_deg", 0.0)))
        R_level = _rot_zyx(0.0, 0.0, -yaw)
        pts_body = pts_body @ R_level.T

    # 仅方位角 + 高度过滤 (无半径裁剪)
    R_eff = _rot_zyx(0.0, 0.0, -yaw) @ _R_BL
    pts_la = (pts_body - _T_BL) @ R_eff
    az = np.arctan2(pts_la[:, 1], pts_la[:, 0])
    in_fov = (pts_la[:, 0] > 0.0) & (np.abs(az) <= az_half) & (pts_la[:, 2] >= 0.0)
    pts_fov = pts_body[in_fov]
    if len(pts_fov) == 0: return np.empty((0, 3), dtype=np.float32)

    keep = (np.isfinite(pts_fov).all(axis=1)
            & (pts_fov[:, 2] >= min_d) & (pts_fov[:, 2] <= max_d))
    return pts_fov[keep].astype(np.float32, copy=False)


# ============================================================
# 稀疏深度投影
# ============================================================

# ============================================================
# BEV 深度投影 (FOV 点云自适应铺满画布)
# ============================================================

def project_bev_depth(pts_body, grid_res=64, out_size=128, max_range=30.0):
    """机体系 FOV 点云 → BEV 深度图.

    使用当前 FOV 内点云的 XY 包围盒作为整张画布范围, 与 HALSS 语义图的
    surface-normal 输入保持同样的自适应铺满效果；不再做 ROI 半径裁剪。
    """
    empty = np.full((out_size, out_size), max_range, dtype=np.float32)
    if pts_body is None or len(pts_body) == 0:
        return empty

    pts = np.asarray(pts_body, dtype=np.float32)
    valid_pts = np.isfinite(pts).all(axis=1)
    pts = pts[valid_pts]
    if len(pts) == 0:
        return empty

    z_all = pts[:, 2]
    valid_z = (z_all > 0.01) & (z_all < max_range)
    pts = pts[valid_z]
    if len(pts) == 0:
        return empty

    x_min, x_max = float(np.min(pts[:, 0])), float(np.max(pts[:, 0]))
    y_min, y_max = float(np.min(pts[:, 1])), float(np.max(pts[:, 1]))
    x_span = x_max - x_min
    y_span = y_max - y_min
    if x_span < 1e-6:
        x_min -= 0.5
        x_max += 0.5
        x_span = x_max - x_min
    if y_span < 1e-6:
        y_min -= 0.5
        y_max += 0.5
        y_span = y_max - y_min

    col_idx = np.rint((pts[:, 0] - x_min) / x_span * (grid_res - 1)).astype(np.int32)
    row_unflipped = np.rint((pts[:, 1] - y_min) / y_span * (grid_res - 1)).astype(np.int32)
    row_idx = (grid_res - 1) - row_unflipped

    valid = (
        (row_idx >= 0) & (row_idx < grid_res)
        & (col_idx >= 0) & (col_idx < grid_res)
    )
    row_idx, col_idx = row_idx[valid], col_idx[valid]
    z_vals = pts[valid, 2]
    if len(z_vals) == 0:
        return empty

    accum = np.zeros((grid_res, grid_res), dtype=np.float32)
    count = np.zeros((grid_res, grid_res), dtype=np.int32)
    np.add.at(accum, (row_idx, col_idx), z_vals)
    np.add.at(count, (row_idx, col_idx), 1)
    mask = count > 0
    grid = np.full((grid_res, grid_res), np.nan, dtype=np.float32)
    grid[mask] = accum[mask] / count[mask]

    if grid_res != out_size:
        grid = cv2.resize(grid, (out_size, out_size), interpolation=cv2.INTER_NEAREST)
        nan_mask = np.isnan(grid) | (grid <= 0)
        grid[nan_mask] = max_range
    else:
        grid = np.where(np.isnan(grid), max_range, grid)
    return grid.astype(np.float32)


# ============================================================
# 可视化
# ============================================================

class LiveDisplay:
    def __init__(self, dmax=30.0):
        import matplotlib
        matplotlib.use("TkAgg")
        import matplotlib.pyplot as plt
        plt.ion()
        self.fig, (self.ax_sp, self.ax_dn, self.ax_sm) = plt.subplots(1, 3, figsize=(16, 5))
        self.fig.canvas.manager.set_window_title("Depth + Semantic (TF SparseNet + HALSS)")
        plt.show(block=False)
        self.im_sp = self.im_dn = self.im_sm = None
        self.dmax = dmax
        self._ready = False

    def update(self, sparse, dense, semantic=None):
        if not self._ready:
            self.im_sp = self.ax_sp.imshow(sparse, cmap="inferno", vmin=0, vmax=self.dmax)
            self.ax_sp.set_title("Sparse Depth")
            self.fig.colorbar(self.im_sp, ax=self.ax_sp, fraction=0.046, label="m")
            self.im_dn = self.ax_dn.imshow(dense, cmap="inferno", vmin=0, vmax=self.dmax)
            self.ax_dn.set_title("Dense Depth (SparseNet)")
            self.fig.colorbar(self.im_dn, ax=self.ax_dn, fraction=0.046, label="m")
            if semantic is not None:
                # 配色: 安全=绿, 危险=红
                sem_rgb = np.zeros((semantic.shape[0], semantic.shape[1], 3), dtype=np.uint8)
                sem_rgb[semantic == 1] = [0, 255, 0]    # 安全=绿
                sem_rgb[semantic == 9] = [255, 0, 0]    # 危险=红
                sem_rgb[(semantic != 1) & (semantic != 9)] = [128, 128, 128]  # 未知=灰
                self.im_sm = self.ax_sm.imshow(sem_rgb)
                self.ax_sm.set_title("Semantic (HALSS)")
            else:
                self.ax_sm.text(0.5, 0.5, "Semantic N/A", ha="center", va="center",
                               transform=self.ax_sm.transAxes, fontsize=12, color="gray")
            self._ready = True
        else:
            self.im_sp.set_data(sparse)
            self.im_dn.set_data(dense)
            if semantic is not None and self.im_sm is not None:
                sem_rgb = np.zeros((semantic.shape[0], semantic.shape[1], 3), dtype=np.uint8)
                sem_rgb[semantic == 1] = [0, 255, 0]
                sem_rgb[semantic == 9] = [255, 0, 0]
                sem_rgb[(semantic != 1) & (semantic != 9)] = [128, 128, 128]
                self.im_sm.set_data(sem_rgb)
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()


# ============================================================
# ROS2 桥接
# ============================================================

class LivoxBridge(Node):
    def __init__(self):
        super().__init__("depth_comp_bridge")
        self.latest_cloud = None
        self.latest_imu = None
        self._c_lock = threading.Lock()
        self._i_lock = threading.Lock()
        self.create_subscription(PointCloud2, "/livox/lidar", self._cb_cloud, 10)
        self.create_subscription(Imu, "/livox/imu", self._cb_imu, 10)
        logger.info("[Bridge] /livox/lidar + /livox/imu")

    def _pc2np(self, msg):
        fos = {f.name: f.offset for f in msg.fields}
        if not all(k in fos for k in ("x","y","z")): return np.empty((0,3), dtype=np.float32)
        n = msg.width * msg.height
        e = ">f4" if msg.is_bigendian else "<f4"
        dt = np.dtype({"names":["x","y","z"],"formats":[e]*3,
                       "offsets":[fos[k] for k in ("x","y","z")],"itemsize":msg.point_step})
        arr = np.frombuffer(msg.data, dtype=dt, count=n)
        pts = np.column_stack((arr["x"],arr["y"],arr["z"])).astype(np.float32)
        return pts[np.isfinite(pts).all(axis=1)]

    def _cb_cloud(self, msg):
        pts = self._pc2np(msg)
        if len(pts):
            ts = msg.header.stamp.sec + msg.header.stamp.nanosec*1e-9
            with self._c_lock: self.latest_cloud = (ts, pts)

    def _cb_imu(self, msg):
        q = msg.orientation
        if q.x or q.y or q.z or q.w:
            sinr, cosr = 2*(q.w*q.x+q.y*q.z), 1-2*(q.x*q.x+q.y*q.y)
            roll = np.arctan2(sinr, cosr)
            sinp = 2*(q.w*q.y-q.z*q.x)
            pitch = np.arcsin(np.clip(sinp,-1,1)) if abs(sinp)<1 else np.sign(sinp)*np.pi/2
            siny, cosy = 2*(q.w*q.z+q.x*q.y), 1-2*(q.y*q.y+q.z*q.z)
            yaw = np.arctan2(siny, cosy)
            ts = msg.header.stamp.sec+msg.header.stamp.nanosec*1e-9
            with self._i_lock: self.latest_imu = (ts, np.array([roll,pitch,yaw],dtype=np.float32))

    def grab(self):
        with self._c_lock: c = self.latest_cloud
        with self._i_lock: i = self.latest_imu
        return c, i


# ============================================================
# 主逻辑
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-zmq", action="store_true")
    parser.add_argument("--no-display", action="store_true")
    parser.add_argument("--input-size", type=int, default=128)
    parser.add_argument("--dmax", type=float, default=30.0)
    args = parser.parse_args()

    dmax = args.dmax
    sz = args.input_size
    pcfg = CFG["perception"]
    dcfg = CFG["depth_completion"]

    # ---- PyTorch SparseNet 深度补全 (与 test_live_nocontrol_raw.py 一致) ----
    depth_comp = None
    if dcfg.get("weight_path") and os.path.exists(dcfg["weight_path"]):
        depth_comp = DepthCompletion(dcfg)
        logger.info("SparseNet (PyTorch) loaded OK")
    else:
        logger.warning("No SparseNet weight found, using raw sparse depth")

    # ---- HALSS 语义 ----
    halss = HALSSBayesianEvaluator(pcfg)
    sem_gen = SemanticGenerator(pcfg)
    logger.info("HALSS + SemanticGenerator loaded OK")

    # ---- ZMQ ----
    zmq_ctx, zmq_pub = None, None
    if not args.no_zmq and HAS_ZMQ:
        zmq_ctx = zmq.Context()
        zmq_pub = zmq_ctx.socket(zmq.PUB)
        zmq_pub.bind("tcp://127.0.0.1:5556")
        logger.info("ZMQ PUB → tcp://127.0.0.1:5556")

    # ---- Display ----
    display = None if args.no_display else LiveDisplay(dmax)

    # ---- ROS2 ----
    rclpy.init()
    bridge = LivoxBridge()
    logger.info("Running depth completion loop...")
    logger.info(f"  SparseNet: {'OK' if depth_comp else 'N/A (raw sparse)'}")
    logger.info(f"  HALSS: OK")

    frame_id = 0
    last_print = time.time()

    try:
        while rclpy.ok():
            rclpy.spin_once(bridge, timeout_sec=0.01)
            cloud, imu = bridge.grab()
            if cloud is None: time.sleep(0.01); continue

            ts_c, pts_lidar = cloud
            imu_rpy = imu[1] if imu else None
            t0 = time.time()

            # 1. LiDAR → 机体 ROI
            pts_body = lidar_to_body_roi(pts_lidar, imu_rpy, pcfg)

            # 2. HALSS 语义
            bev = halss.evaluate(pts_body)
            if bev is not None:
                sem_map = sem_gen.generate(bev)
            else:
                sem_map = np.full((sz, sz), pcfg["danger_class_id"], dtype=np.uint8)

            # 3. BEV 稀疏深度 (FOV 点云自适应铺满画布, 不做 ROI 裁剪)
            sparse_depth = project_bev_depth(
                pts_body,
                grid_res=pcfg.get("halss_grid_res", 64),
                out_size=sz, max_range=dmax)
            valid_mask = (sparse_depth < dmax) & (sparse_depth > 0.01)

            # 4. 深度补全
            if depth_comp is not None and valid_mask.sum() > 5:
                dense_depth = depth_comp.complete(sparse_depth, valid_mask)
            else:
                dense_depth = np.where(valid_mask, sparse_depth, dmax)

            # 5. 可视化
            if display:
                display.update(sparse_depth, dense_depth, sem_map)

            # 6. ZMQ 发布
            if zmq_pub:
                from control.zmq_protocol import serialize_dense_depth_frame
                h, p = serialize_dense_depth_frame(frame_id, dense_depth, sem_map)
                zmq_pub.send(h + p)

            dt_ms = (time.time() - t0) * 1000
            frame_id += 1
            if time.time() - last_print >= 2.0:
                sd_valid = sparse_depth[valid_mask]
                sp_min = float(sd_valid.min()) if len(sd_valid) else float("nan")
                sp_max = float(sd_valid.max()) if len(sd_valid) else float("nan")
                sem_safe = int((sem_map == pcfg["safe_class_id"]).sum())
                sem_danger = int((sem_map == pcfg["danger_class_id"]).sum())
                logger.info(
                    f"[{frame_id:5d}] pts_in={len(pts_lidar):5d} body={len(pts_body):4d} "
                    f"sp=[{sp_min:.1f},{sp_max:.1f}]m "
                    f"dn=[{dense_depth.min():.1f},{dense_depth.max():.1f}]m "
                    f"sem=safe:{sem_safe} danger:{sem_danger} "
                    f"lat={dt_ms:.0f}ms")
                last_print = time.time()

    except KeyboardInterrupt:
        logger.info("Interrupted")
    finally:
        if zmq_pub: zmq_pub.close()
        if zmq_ctx: zmq_ctx.term()
        rclpy.shutdown()
        logger.info("Stopped")

if __name__ == "__main__":
    main()
