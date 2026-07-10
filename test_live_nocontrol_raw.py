#!/usr/bin/env python3
"""
无飞控在线测试 (NN-fill 深度 + ONNX DRL) — ROS1 Noetic 版
============================================================
MID360 原始点云 → HALSS语义 + 最近邻深度渲染 → ONNX DRL 推理

深度: 几何最近邻填洞 + 平滑 (不依赖 SparseNet/TF)
DRL: ONNX 推理 (零 TensorFlow 依赖)
可视化: 语义=白色安全/黑色危险, 深度=深度相机透视风格

不接飞控, 终端打印推理动作。

依赖:
  - /livox/lidar  (sensor_msgs/PointCloud2, livox_ros_driver2 rviz_MID360.launch, xfer_format=0/2)
  - /livox/imu    (sensor_msgs/Imu)

用法:
  source /opt/ros/noetic/setup.bash
  source ~/livox_ws/devel/setup.bash
  conda activate fylanding
  python test_live_nocontrol_raw.py --no-display --onnx-model weights/ppo2_policy.onnx
"""

import argparse, sys, os, time, threading, logging
import numpy as np
import cv2
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("LiveNoControl")

PROJ_ROOT = os.path.dirname(os.path.abspath(__file__))
CFG_PATH = os.path.join(PROJ_ROOT, "config", "experiment_config.yaml")
with open(CFG_PATH) as f:
    CFG = yaml.safe_load(f)
sys.path.insert(0, PROJ_ROOT)

# ---- ROS1 ----
import rospy
from sensor_msgs.msg import PointCloud2, Imu

# ---- 感知 ----
from perception.halss_bayesian import HALSSBayesianEvaluator
from perception.halss_preprocess import _rot_zyx
from perception.semantic_generator import SemanticGenerator

# ---- ONNX DRL ----
try:
    import onnxruntime as ort
    HAS_ONNX = True
except ImportError:
    ort = None
    HAS_ONNX = False

# 与正式 zmq_protocol / DeepRL 训练一致的语义灰度映射
CLASS_TO_GRAY = {
    -1: 0,     0: 10,    1: 30,    2: 60,    3: 70,    4: 20,
     5: 40,    6: 80,    7: 90,    8: 50,    9: 250,
}

# ---- 从 ActionDecomposer 获取正确动作名 (sign=-1 = 原始 DeepRL 符号) ----
from control.action_decomposer import ActionDecomposer
_DECOMP = ActionDecomposer(CFG.get("uav", {}))
ACTION_NAMES_DECOMP = _DECOMP.action_names
ACTION_SIGN = _DECOMP.action_lateral_sign
logger.info(f"ActionDecomposer (sign={ACTION_SIGN}): {ACTION_NAMES_DECOMP}")


# ============================================================
# LiDAR → 机体 坐标变换
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
    """LiDAR → 机体 (FOV 方位角 + 高度过滤)."""
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
# BEV 稀疏深度投影 (与 HALSS 语义图对齐)
# ============================================================

def project_bev_depth(pts_body, grid_res=64, out_size=128, max_range=30.0):
    """机体系 FOV 点云 → BEV 深度图.

    Returns:
        depth: (out_size, out_size) float32 深度图
        bounds: dict with x_min, x_max, y_min, y_max (用于点云画布对齐)
    """
    empty = np.full((out_size, out_size), max_range, dtype=np.float32)
    empty_bounds = {"x_min": -1.0, "x_max": 1.0, "y_min": -1.0, "y_max": 1.0}
    if pts_body is None or len(pts_body) == 0:
        return empty, empty_bounds

    pts = np.asarray(pts_body, dtype=np.float32)
    valid_pts = np.isfinite(pts).all(axis=1)
    pts = pts[valid_pts]
    if len(pts) == 0:
        return empty, empty_bounds

    z_all = pts[:, 2]
    valid_z = (z_all > 0.01) & (z_all < max_range)
    pts = pts[valid_z]
    if len(pts) == 0:
        return empty, empty_bounds

    x_min, x_max = float(np.min(pts[:, 0])), float(np.max(pts[:, 0]))
    y_min, y_max = float(np.min(pts[:, 1])), float(np.max(pts[:, 1]))
    x_span = x_max - x_min
    y_span = y_max - y_min
    if x_span < 1e-6:
        x_min -= 0.5; x_max += 0.5; x_span = x_max - x_min
    if y_span < 1e-6:
        y_min -= 0.5; y_max += 0.5; y_span = y_max - y_min

    bounds = {"x_min": x_min, "x_max": x_max, "y_min": y_min, "y_max": y_max}

    col_idx = np.rint((pts[:, 0] - x_min) / x_span * (grid_res - 1)).astype(np.int32)
    row_unflipped = np.rint((pts[:, 1] - y_min) / y_span * (grid_res - 1)).astype(np.int32)
    row_idx = (grid_res - 1) - row_unflipped

    valid = (row_idx >= 0) & (row_idx < grid_res) & (col_idx >= 0) & (col_idx < grid_res)
    row_idx, col_idx = row_idx[valid], col_idx[valid]
    z_vals = pts[valid, 2]
    if len(z_vals) == 0:
        return empty, empty_bounds

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
    return grid.astype(np.float32), bounds


def project_pointcloud_canvas(pts_body, bounds, out_size=128, dmax=30.0):
    """点云 → 与深度/语义同画布的 128x128 栅格.

    使用与 project_bev_depth 完全相同的 bbox 映射:
      col = (x - x_min) / x_span * (out_size - 1)
      row = (out_size - 1) - (y - y_min) / y_span * (out_size - 1)
    每像素取平均 z_down.
    """
    empty = np.full((out_size, out_size), dmax, dtype=np.float32)
    if pts_body is None or len(pts_body) == 0:
        return empty

    pts = np.asarray(pts_body, dtype=np.float32)
    x_min, x_max = bounds["x_min"], bounds["x_max"]
    y_min, y_max = bounds["y_min"], bounds["y_max"]
    x_span = x_max - x_min
    y_span = y_max - y_min
    if x_span < 1e-6 or y_span < 1e-6:
        return empty

    col_idx = np.rint((pts[:, 0] - x_min) / x_span * (out_size - 1)).astype(np.int32)
    row_unflipped = np.rint((pts[:, 1] - y_min) / y_span * (out_size - 1)).astype(np.int32)
    row_idx = (out_size - 1) - row_unflipped

    valid = (row_idx >= 0) & (row_idx < out_size) & (col_idx >= 0) & (col_idx < out_size)
    row_idx, col_idx = row_idx[valid], col_idx[valid]
    z_vals = pts[valid, 2]
    if len(z_vals) == 0:
        return empty

    accum = np.zeros((out_size, out_size), dtype=np.float32)
    count = np.zeros((out_size, out_size), dtype=np.int32)
    np.add.at(accum, (row_idx, col_idx), z_vals)
    np.add.at(count, (row_idx, col_idx), 1)
    mask = count > 0
    canvas = np.full((out_size, out_size), dmax, dtype=np.float32)
    canvas[mask] = accum[mask] / count[mask]
    return canvas


# ============================================================
# 稀疏深度渲染 (最近邻填洞 + 平滑)
# ============================================================

def render_sparse_depth(sparse_depth, valid_mask, dmax, min_valid=5, median_ksize=5):
    """最近有效稀疏点填洞 + 平滑."""
    if valid_mask.sum() < min_valid:
        return np.where(valid_mask, sparse_depth, dmax).astype(np.float32)

    invalid = ~valid_mask
    mask_u8 = invalid.astype(np.uint8)
    _, labels = cv2.distanceTransformWithLabels(
        mask_u8, distanceType=cv2.DIST_L2, maskSize=5,
        labelType=cv2.DIST_LABEL_PIXEL)

    valid_coords = np.column_stack(np.where(valid_mask))
    label_vals = labels[invalid]
    nearest_idx = np.clip(label_vals - 1, 0, len(valid_coords) - 1)

    filled = sparse_depth.copy()
    filled[invalid] = sparse_depth[
        valid_coords[nearest_idx, 0],
        valid_coords[nearest_idx, 1]]

    if median_ksize >= 3:
        smoothed = cv2.medianBlur(filled.astype(np.float32), median_ksize)
    else:
        smoothed = filled.astype(np.float32)

    rendered = np.where(valid_mask, sparse_depth, smoothed)
    return np.clip(rendered, 0.0, dmax).astype(np.float32)


# ============================================================
# ONNX DRL 推理器
# ============================================================

class ONNXDRL:
    """轻量 ONNX DRL 推理 (从 TF1 PPO2 导出).

    ONNX 图内部含 input/truediv (来自 scale=True), 会自动 /255.
    因此外部输入应为 raw observation 尺度:
      depth channel:   米值 float32, 典型 0..30
      semantic channel: CLASS_TO_GRAY 灰度值 float32, Terrain=30, Others=250
    """

    def __init__(self, onnx_path, obs_h=128, obs_w=128, dmax=30.0):
        if not HAS_ONNX:
            raise ImportError("pip install onnxruntime")
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        opts.inter_op_num_threads = 1
        self.session = ort.InferenceSession(onnx_path, opts)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        in_shape = self.session.get_inputs()[0].shape
        logger.info(f"[ONNX] input={self.input_name} shape={in_shape}")

        if len(in_shape) == 4 and in_shape[1] in (2, 3):
            self.layout = "chw"
        else:
            self.layout = "hwc"
        self.obs_h, self.obs_w = obs_h, obs_w
        self.dmax = dmax

        # 预热
        dummy = np.zeros((1, obs_h, obs_w, 2), dtype=np.float32)
        self._forward(dummy)
        logger.info("[ONNX] warmup OK")

    def _forward(self, obs_raw):
        if self.layout == "chw":
            inp = np.transpose(obs_raw, (0, 3, 1, 2)).astype(np.float32)
        else:
            inp = obs_raw.astype(np.float32)
        return self.session.run([self.output_name], {self.input_name: inp})[0]

    def predict(self, depth_map, sem_map, flip_lr=False, flip_ud=False):
        """深度图 + 语义图 -> 动作索引 0-9 + 诊断信息.

        编码: ONNX 内部含 input/truediv (scale=True), 外部输入保持 raw 尺度.
          depth channel:   米值 float32 (clip 0..dmax), 典型 0..30
          semantic channel: CLASS_TO_GRAY 灰度值 float32, Terrain=30, Others=250
        不再提前 /255, 不再映射到 0..255 再除.
        """
        # Depth channel: clip 米值, 保持为 raw float32 米值
        depth_clipped = np.clip(
            np.nan_to_num(depth_map, nan=self.dmax, posinf=self.dmax, neginf=0.0),
            0.0, self.dmax)
        depth_ch = depth_clipped.astype(np.float32)

        # Semantic channel: CLASS_TO_GRAY 灰度值 float32
        sem_int = np.clip(sem_map, -1, 9).astype(np.int16)
        sem_ch = np.zeros_like(sem_int, dtype=np.float32)
        for class_id, gray_val in CLASS_TO_GRAY.items():
            sem_ch[sem_int == class_id] = float(gray_val)

        # 诊断翻转变换 (作用在 raw 尺度上)
        if flip_lr:
            depth_ch = np.fliplr(depth_ch)
            sem_ch = np.fliplr(sem_ch)
        if flip_ud:
            depth_ch = np.flipud(depth_ch)
            sem_ch = np.flipud(sem_ch)

        obs = np.stack([depth_ch, sem_ch], axis=-1)
        obs = np.expand_dims(obs, axis=0)

        logits = self._forward(obs)
        action = int(np.argmax(logits[0]))

        # softmax probs
        l = logits[0]
        e = np.exp(l - l.max())
        probs = e / e.sum()

        info = {
            "depth_raw_median": float(np.median(depth_map)),
            "depth_input_mean": float(depth_ch.mean()),
            "depth_input_min": float(depth_ch.min()),
            "depth_input_max": float(depth_ch.max()),
            "sem_input_mean": float(sem_ch.mean()),
            "sem_input_min": float(sem_ch.min()),
            "sem_input_max": float(sem_ch.max()),
            "sem_input_unique": sorted(np.unique(sem_ch).astype(int).tolist()),
            "obs_raw_min": float(obs.min()),
            "obs_raw_max": float(obs.max()),
            "softmax_probs": probs.astype(float).tolist(),
        }
        return action, logits[0], info

    def action_name(self, action_id):
        return ACTION_NAMES_DECOMP[action_id] if 0 <= action_id < len(ACTION_NAMES_DECOMP) else "?"


# ============================================================
# 可视化
# ============================================================

class LiveDisplay:
    """三窗口: 左=点云画布, 中=语义(白安全/黑危险), 右=渲染深度(inferno). 三个窗口画布对齐."""

    def __init__(self, sz=300, dmax=30.0):
        import matplotlib
        matplotlib.use("TkAgg")
        import matplotlib.pyplot as plt
        plt.ion()
        self.fig, (self.ax_pc, self.ax_sem, self.ax_depth) = plt.subplots(
            1, 3, figsize=(15, 5), constrained_layout=True)
        self.fig.canvas.manager.set_window_title("PointCloud Canvas + Semantic + Depth (NN-fill + ONNX DRL)")
        plt.show(block=False)
        self.sz = sz
        self.dmax = dmax
        self.im_pc = None
        self.im_sem = None
        self.im_depth = None
        self.pc_cbar = None
        self.depth_cbar = None
        self._ready = False

    def _resize(self, img):
        h, w = img.shape[:2]
        if h != self.sz or w != self.sz:
            return cv2.resize(img, (self.sz, self.sz), interpolation=cv2.INTER_NEAREST)
        return img

    def update(self, sem_map, depth_map, pc_canvas=None):
        # 语义: 白色=安全(1), 黑色=危险(9), 灰色=未知
        sem_vis = np.full(sem_map.shape, 128, dtype=np.uint8)
        sem_vis[sem_map == 1] = 255
        sem_vis[sem_map == 9] = 0

        # 点云画布: 与深度图相同 inferno 色标
        if pc_canvas is None:
            pc_canvas = np.full_like(depth_map, self.dmax)

        sem_disp = self._resize(sem_vis)
        depth_disp = self._resize(depth_map)
        pc_disp = self._resize(pc_canvas)

        if not self._ready:
            # 点云画布
            self.im_pc = self.ax_pc.imshow(
                pc_disp, cmap="inferno", vmin=0, vmax=self.dmax, interpolation="nearest")
            self.ax_pc.set_title("Point Cloud Canvas")
            self.pc_cbar = self.fig.colorbar(
                self.im_pc, ax=self.ax_pc, fraction=0.046, pad=0.04, label="z (m)")

            # 语义
            self.im_sem = self.ax_sem.imshow(
                sem_disp, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
            self.ax_sem.set_title("Semantic (white=safe, black=danger)")

            # 渲染深度
            self.im_depth = self.ax_depth.imshow(
                depth_disp, cmap="inferno", vmin=0, vmax=self.dmax, interpolation="nearest")
            self.ax_depth.set_title("Rendered Depth (NN Fill)")
            self.depth_cbar = self.fig.colorbar(
                self.im_depth, ax=self.ax_depth, fraction=0.046, pad=0.04, label="m")
            self._ready = True
        else:
            self.im_pc.set_data(pc_disp)
            self.im_sem.set_data(sem_disp)
            self.im_depth.set_data(depth_disp)
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()


# ============================================================
# ROS1 数据桥接
# ============================================================

class LivoxBridge:
    def __init__(self):
        rospy.init_node("live_nocontrol_bridge")
        self.latest_cloud = None
        self.latest_imu = None
        self._c_lock = threading.Lock()
        self._i_lock = threading.Lock()
        self._cloud_seq = 0
        self._imu_seq = 0
        rospy.Subscriber("/livox/lidar", PointCloud2, self._cb_cloud)
        rospy.Subscriber("/livox/imu", Imu, self._cb_imu)
        logger.info("[Bridge] /livox/lidar + /livox/imu")

    def _pc2np(self, msg):
        fos = {f.name: f.offset for f in msg.fields}
        if not all(k in fos for k in ("x","y","z")):
            return np.empty((0, 3), dtype=np.float32)
        n = msg.width * msg.height
        e = ">f4" if msg.is_bigendian else "<f4"
        dt = np.dtype({"names": ["x","y","z"], "formats": [e]*3,
                       "offsets": [fos[k] for k in ("x","y","z")],
                       "itemsize": msg.point_step})
        arr = np.frombuffer(msg.data, dtype=dt, count=n)
        pts = np.column_stack((arr["x"], arr["y"], arr["z"])).astype(np.float32)
        return pts[np.isfinite(pts).all(axis=1)]

    def _cb_cloud(self, msg):
        pts = self._pc2np(msg)
        if len(pts):
            ts = msg.header.stamp.secs + msg.header.stamp.nsecs * 1e-9
            with self._c_lock: self.latest_cloud = (ts, pts)
            self._cloud_seq += 1
            if self._cloud_seq == 1:
                print(f"[Bridge] first cloud received: {len(pts)} pts, ts={ts:.3f}", flush=True)

    def _cb_imu(self, msg):
        q = msg.orientation
        if q.x or q.y or q.z or q.w:
            sinr, cosr = 2*(q.w*q.x+q.y*q.z), 1-2*(q.x*q.x+q.y*q.y)
            roll = np.arctan2(sinr, cosr)
            sinp = 2*(q.w*q.y-q.z*q.x)
            pitch = np.arcsin(np.clip(sinp, -1, 1)) if abs(sinp) < 1 else np.sign(sinp)*np.pi/2
            siny, cosy = 2*(q.w*q.z+q.x*q.y), 1-2*(q.y*q.y+q.z*q.z)
            yaw = np.arctan2(siny, cosy)
            ts = msg.header.stamp.secs + msg.header.stamp.nsecs * 1e-9
            with self._i_lock: self.latest_imu = (ts, np.array([roll, pitch, yaw], dtype=np.float32))
            self._imu_seq += 1
            if self._imu_seq == 1:
                print(f"[Bridge] first imu received: rpy=({np.degrees(roll):.1f},{np.degrees(pitch):.1f},{np.degrees(yaw):.1f})°", flush=True)

    def grab(self):
        with self._c_lock: c = self.latest_cloud
        with self._i_lock: i = self.latest_imu
        return c, i


# ============================================================
# 主逻辑
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="NN-fill depth + ONNX DRL live test with diagnostics")
    parser.add_argument("--no-display", action="store_true")
    parser.add_argument("--onnx-model", default="weights/ppo2_policy.onnx",
                        help="Path to ONNX model")
    parser.add_argument("--dmax", type=float, default=30.0)

    # 诊断开关
    parser.add_argument("--diagnose-drl", action="store_true",
                        help="Enable full DRL diagnosis logging")
    parser.add_argument("--flip-lr", action="store_true",
                        help="Flip DRL input left-right")
    parser.add_argument("--flip-ud", action="store_true",
                        help="Flip DRL input up-down")
    parser.add_argument("--semantic-override",
                        choices=["none", "all_safe", "all_danger", "center_safe"],
                        default="none",
                        help="Semantic map override for diagnosis")
    parser.add_argument("--depth-override",
                        choices=["none", "constant_near", "constant_mid", "constant_far"],
                        default="none",
                        help="Depth map override for diagnosis")
    args = parser.parse_args()

    dmax = args.dmax
    pcfg = CFG["perception"]
    sz = CFG["observation"].get("img_width", 128)
    override_str = f"sem={args.semantic_override} depth={args.depth_override}"
    if args.flip_lr: override_str += " flip-lr"
    if args.flip_ud: override_str += " flip-ud"

    # ---- HALSS 语义 ----
    halss = HALSSBayesianEvaluator(pcfg)
    sem_gen = SemanticGenerator(pcfg)
    logger.info("HALSS + SemanticGenerator OK")

    # ---- ONNX DRL ----
    onnx_path = args.onnx_model
    if not os.path.exists(onnx_path):
        logger.error(f"ONNX model not found: {onnx_path}")
        logger.error("Run: python scripts/export_ppo2_to_onnx.py first")
        sys.exit(1)
    drl = ONNXDRL(onnx_path, obs_h=sz, obs_w=sz, dmax=dmax)
    logger.info(f"ONNX DRL loaded: {onnx_path}")

    # ---- Display ----
    display = None if args.no_display else LiveDisplay(sz=300, dmax=dmax)

    # ---- ROS1 ----
    print("Initializing ROS bridge...", flush=True)
    bridge = LivoxBridge()  # rospy.init_node() called inside
    print("ROS bridge OK.", flush=True)
    print(f"Pipeline ready. Waiting for /livox/lidar...", flush=True)
    logger.info(f"  Depth backend: NN-fill (geometry)")
    logger.info(f"  DRL backend:   ONNX (raw input: depth=meters, sem=gray; ONNX internal /255)")
    logger.info(f"  Control:       NONE (print only)")
    logger.info(f"  Overrides:     {override_str}")
    logger.info(f"  Action names:  from ActionDecomposer sign={ACTION_SIGN}: {ACTION_NAMES_DECOMP}")

    frame_id = 0
    last_print = time.time()
    action_names = ACTION_NAMES_DECOMP
    last_wait_log = 0.0  # 周期性 "waiting" 日志

    # 诊断: 动作计数器
    action_counts = [0] * 10

    try:
        print("Entering main loop...", flush=True)
        while not rospy.is_shutdown():
            cloud, imu = bridge.grab()
            if cloud is None:
                rospy.sleep(0.01)
                now = time.time()
                if now - last_wait_log > 3.0:
                    print(f"  Waiting for /livox/lidar data... (is MID360 driver running?)", flush=True)
                    last_wait_log = now
                continue

            ts_c, pts_lidar = cloud
            imu_rpy = imu[1] if imu else None
            t0 = time.time()

            # 1. LiDAR -> 机体 ROI
            pts_body = lidar_to_body_roi(pts_lidar, imu_rpy, pcfg)

            # 2. HALSS 语义
            bev = halss.evaluate(pts_body)
            if bev is not None:
                sem_map_raw = sem_gen.generate(bev)
            else:
                sem_map_raw = np.full((sz, sz), pcfg["danger_class_id"], dtype=np.uint8)

            # 3. BEV 稀疏深度 + 共享 bbox
            sparse_depth, bounds = project_bev_depth(
                pts_body,
                grid_res=pcfg.get("halss_grid_res", 64),
                out_size=sz, max_range=dmax)
            valid_mask = (sparse_depth < dmax) & (sparse_depth > 0.01)

            # 3b. 点云画布 (与深度图共用 bbox)
            pc_canvas = project_pointcloud_canvas(pts_body, bounds, out_size=sz, dmax=dmax)

            # 4. NN-fill 深度渲染
            rendered_depth = render_sparse_depth(sparse_depth, valid_mask, dmax)

            # ---- 诊断: semantic override ----
            if args.semantic_override == "all_safe":
                sem_map_drl = np.full((sz, sz), pcfg["safe_class_id"], dtype=np.uint8)
            elif args.semantic_override == "all_danger":
                sem_map_drl = np.full((sz, sz), pcfg["danger_class_id"], dtype=np.uint8)
            elif args.semantic_override == "center_safe":
                sem_map_drl = np.full((sz, sz), pcfg["danger_class_id"], dtype=np.uint8)
                c = sz // 2
                r = sz // 4
                sem_map_drl[c-r:c+r, c-r:c+r] = pcfg["safe_class_id"]
            else:
                sem_map_drl = sem_map_raw

            # ---- 诊断: depth override ----
            if args.depth_override == "constant_near":
                depth_drl = np.full((sz, sz), 1.0, dtype=np.float32)
            elif args.depth_override == "constant_mid":
                depth_drl = np.full((sz, sz), 10.0, dtype=np.float32)
            elif args.depth_override == "constant_far":
                depth_drl = np.full((sz, sz), float(dmax), dtype=np.float32)
            else:
                depth_drl = rendered_depth

            # 5. ONNX DRL 推理
            action_id, logits, drl_info = drl.predict(
                depth_drl, sem_map_drl,
                flip_lr=args.flip_lr, flip_ud=args.flip_ud)
            action_name = drl.action_name(action_id)
            action_counts[action_id] += 1

            dt_ms = (time.time() - t0) * 1000
            frame_id += 1

            # ---- 每帧打印动作 (print 绕过日志缓冲) ----
            top3_idx = np.argsort(logits)[-3:][::-1]
            sp = drl_info["softmax_probs"]
            top3_str = " ".join(f"{action_names[i]}:p={sp[i]:.2f}" for i in top3_idx)
            print(f"\r[frame {frame_id:04d}] ACTION={action_id}({action_name})  "
                  f"top3: {top3_str}  lat={dt_ms:.0f}ms  pts={len(pts_body)}", end="")

            # 终端打印 (每 0.5s 带详细诊断)
            if time.time() - last_print >= 0.5:
                sp_valid = sparse_depth[valid_mask]
                sp_med = float(np.median(sp_valid)) if len(sp_valid) else float("nan")
                rd_med = float(np.median(rendered_depth))
                sem_safe_raw = int((sem_map_raw == pcfg["safe_class_id"]).sum())
                sem_danger_raw = int((sem_map_raw == pcfg["danger_class_id"]).sum())
                sem_safe_drl = int((sem_map_drl == pcfg["safe_class_id"]).sum())
                sem_danger_drl = int((sem_map_drl == pcfg["danger_class_id"]).sum())

                top3_idx = np.argsort(logits)[-3:][::-1]
                sp = drl_info["softmax_probs"]
                top3_str = " ".join(
                    f"{action_names[i]}:logit={logits[i]:.1f} p={sp[i]:.2f}" for i in top3_idx)

                # 观测统计 (raw 尺度: depth=米值, sem=CLASS_TO_GRAY 灰度值, ONNX 内部 /255)
                obs_str = (
                    f"depth_in={drl_info['depth_input_mean']:.1f}m "
                    f"[{drl_info['depth_input_min']:.1f},{drl_info['depth_input_max']:.1f}] "
                    f"sem_in={drl_info['sem_input_mean']:.0f} "
                    f"[{drl_info['sem_input_min']:.0f},{drl_info['sem_input_max']:.0f}] "
                    f"sem_uniq={drl_info['sem_input_unique']} "
                    f"obs_raw=[{drl_info['obs_raw_min']:.1f},{drl_info['obs_raw_max']:.1f}]"
                )

                # 动作分布摘要
                total_acts = sum(action_counts)
                if total_acts > 0:
                    act_dist = " ".join(
                        f"{action_names[i]}:{action_counts[i]}" for i in range(10)
                        if action_counts[i] > 0)
                else:
                    act_dist = "none"

                logger.info(
                    f"[{frame_id:5d}] pts={len(pts_lidar):4d}->body={len(pts_body):3d} "
                    f"sp_med={sp_med:.1f}m rd_med={rd_med:.1f}m "
                    f"sem_raw safe={sem_safe_raw} danger={sem_danger_raw} "
                    f"sem_drl safe={sem_safe_drl} danger={sem_danger_drl} "
                    f"lat={dt_ms:.0f}ms")
                logger.info(f"  {obs_str}")
                logger.info(
                    f"  >>> ACTION: {action_id} ({action_name})  "
                    f"top3: {top3_str}")

                # 画布方向 -> 机体系动作 -> NED 世界方向 闭环
                yaw_rad = float(imu_rpy[2]) if imu_rpy is not None else 0.0
                v_body, v_ned, yr = _DECOMP.decompose(action_id, yaw_rad)
                yaw_deg = np.rad2deg(yaw_rad)
                logger.info(
                    f"  yaw={yaw_deg:+.0f}°  action={action_id}({action_name})  "
                    f"v_body=[{v_body[0]:+.1f},{v_body[1]:+.1f},{v_body[2]:+.1f}]  "
                    f"v_ned=[{v_ned[0]:+.1f},{v_ned[1]:+.1f},{v_ned[2]:+.1f}]  "
                    f"yr={yr:+.2f}rad/s")

                logger.info(f"  act_dist: {act_dist}")

                if args.diagnose_drl:
                    logger.info(f"  logits: {' '.join(f'{action_names[i]}={logits[i]:.2f}' for i in range(10))}")
                    logger.info(f"  probs:  {' '.join(f'{action_names[i]}={sp[i]:.3f}' for i in range(10))}")
                    logger.info(f"  depth_input: mean={drl_info['depth_input_mean']:.1f}m min={drl_info['depth_input_min']:.1f} max={drl_info['depth_input_max']:.1f}")
                    logger.info(f"  sem_input:   mean={drl_info['sem_input_mean']:.0f} unique={drl_info['sem_input_unique']}")
                    logger.info(f"  obs_raw range: [{drl_info['obs_raw_min']:.1f}, {drl_info['obs_raw_max']:.1f}] (ONNX 内部 /255)")

                # rd vs sp 偏离告警
                if not np.isnan(sp_med) and abs(rd_med - sp_med) > 3.0:
                    logger.warning(
                        f"  rd_med({rd_med:.1f}) deviates from sp_med({sp_med:.1f}) by >3m")

                last_print = time.time()

            # 可视化 (始终显示原始语义和渲染深度)
            if display:
                display.update(sem_map_raw, rendered_depth, pc_canvas)

    except KeyboardInterrupt:
        logger.info("Interrupted")
    finally:
        total = sum(action_counts)
        if total > 0:
            logger.info("=== Final Action Distribution ===")
            for i in range(10):
                logger.info(f"  {action_names[i]}: {action_counts[i]} ({100*action_counts[i]/total:.1f}%)")
        rospy.signal_shutdown("interrupted")
        logger.info("Stopped")


if __name__ == "__main__":
    main()
