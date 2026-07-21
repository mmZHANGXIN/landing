#!/usr/bin/env python3
"""
无飞控在线测试 — FAST-LIO 去畸变点云 + NN-fill 深度 + ONNX DRL
===============================================================
订阅 /Odometry 和 /cloud_registered, 跑 HALSS + NN-fill + ONNX DRL, 实时显示, 终端打印动作和耗时。
不发送 MAVSDK 控制指令。

用法:
  source /opt/ros/noetic/setup.bash
  source ~/livox_ws/devel/setup.bash
  source ~/fast_lio_ws/devel/setup.bash
  conda activate fylanding
  python test_live_nocontrol.py --no-display --onnx-model weights/ppo2_policy.onnx
"""

import argparse
import sys, os, time, logging
import numpy as np
import cv2

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("LiveTest")


def _fmt_vec(v: np.ndarray) -> str:
    return f"[{v[0]:.1f},{v[1]:.1f},{v[2]:.1f}]"


def _top_probs(probs, action_names=None, k: int = 3) -> str:
    if probs is None:
        return "p=n/a"
    pairs = sorted(enumerate(probs), key=lambda item: item[1], reverse=True)[:k]
    if action_names is None:
        return "p=" + ",".join(f"{idx}:{prob:.2f}" for idx, prob in pairs)
    return "p=" + ",".join(f"{idx}:{action_names[idx]}:{prob:.2f}" for idx, prob in pairs)

# ------------------------------ config ------------------------------
import yaml
_EARLY_PARSER = argparse.ArgumentParser(add_help=False)
_EARLY_PARSER.add_argument(
    "--config",
    default=os.path.join(os.path.dirname(__file__), "config", "experiment_config.yaml"),
)
_EARLY_ARGS, _ = _EARLY_PARSER.parse_known_args()


def _load_profile(path):
    path = os.path.abspath(path)
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    parent = cfg.pop("extends", None)
    if not parent:
        return cfg
    base = _load_profile(os.path.join(os.path.dirname(path), parent))

    def merge(dst, src):
        for key, value in src.items():
            if isinstance(value, dict) and isinstance(dst.get(key), dict):
                merge(dst[key], value)
            else:
                dst[key] = value
        return dst

    return merge(base, cfg)


CFG = _load_profile(_EARLY_ARGS.config)

sys.path.insert(0, os.path.dirname(__file__))

import rospy

# ---- 感知模块 ----
from perception.halss_bayesian import HALSSBayesianEvaluator
from perception.halss_preprocess import world_to_level_body_roi
from perception.semantic_generator import SemanticGenerator
from perception.training_camera_projection import (
    TrainingCameraModel,
    project_training_camera,
)

# ---- 里程计 ----
from odometry import FastLIOInterface

# ---- 控制 ----
from control.action_decomposer import ActionDecomposer

# ---- ONNX DRL ----
try:
    import onnxruntime as ort
    HAS_ONNX = True
except ImportError:
    ort = None
    HAS_ONNX = False

# 与 raw 版 / DeepRL 训练一致的语义灰度映射
CLASS_TO_GRAY = {
    -1: 0,     0: 10,    1: 30,    2: 60,    3: 70,    4: 20,
     5: 40,    6: 80,    7: 90,    8: 50,    9: 250,
}

_DECOMP = ActionDecomposer(CFG.get("uav", {}))
ACTION_NAMES_DECOMP = _DECOMP.action_names
ACTION_SIGN = _DECOMP.action_lateral_sign
logger.info(f"ActionDecomposer (sign={ACTION_SIGN}): {ACTION_NAMES_DECOMP}")


def _parse_args():
    parser = argparse.ArgumentParser(description="No-control live Orin landing pipeline test")
    parser.add_argument("--config", default=_EARLY_ARGS.config,
                        help="Experiment profile used by the perception pipeline")
    parser.add_argument("--yaw-rate-rad-s", type=float, default=None,
                        help="Override uav.yaw_rate_rad_s for this no-control run")
    parser.add_argument("--save-raw-arrays", action="store_true",
                        help="Save *_calib_frame.npz for ONNX DRL diagnosis")
    parser.add_argument("--save-frames", action="store_true",
                        help="Save displayed binary semantic and depth frames")
    parser.add_argument("--save-dir", default=None,
                        help="Override visualization.save_dir")
    parser.add_argument("--require-yaw-rate", action="store_true",
                        help="Fail if the configured yaw-fault rate is zero")
    parser.add_argument("--max-frames", type=int, default=0,
                        help="Stop after N processed frames (0 = run until interrupted)")
    parser.add_argument("--duration-sec", type=float, default=0.0,
                        help="Stop after this many seconds of wall time (0 = run until interrupted)")
    parser.add_argument("--no-display", action="store_true",
                        help="Disable matplotlib live display")
    parser.add_argument("--onnx-model", default="weights/ppo2_policy.onnx",
                        help="Path to ONNX DRL model")
    parser.add_argument("--dmax", type=float, default=30.0,
                        help="Depth max range in meters")
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
    return parser.parse_args()


def _apply_cli_overrides(cfg: dict, args):
    if args.yaw_rate_rad_s is not None:
        cfg.setdefault("uav", {})["yaw_rate_rad_s"] = float(args.yaw_rate_rad_s)
    vis_cfg = cfg.setdefault("visualization", {})
    if args.save_raw_arrays:
        vis_cfg["save_raw_arrays"] = True
    if args.save_frames:
        vis_cfg["save_frames"] = True
    if args.save_dir:
        vis_cfg["save_dir"] = args.save_dir


# ============================================================
# ROS1 数据桥接 (对齐 pipeline.py 的 FastLIOInterface)
# ============================================================

def _init_ros_bridge(node_name: str = "live_fastlio_nocontrol_bridge") -> FastLIOInterface:
    """初始化 ROS 节点并订阅 FAST-LIO 话题，与 pipeline.py 完全对齐。

    话题名从 config/experiment_config.yaml → localization 段读取，
    与 pipeline.py 中 OrinLandingPipeline.init_ros_node() 使用相同配置项。
    """
    from nav_msgs.msg import Odometry
    from sensor_msgs.msg import PointCloud2

    if not rospy.core.is_initialized():
        rospy.init_node(node_name, anonymous=False)

    loc_cfg = CFG.get("localization", {})
    odom_topic = loc_cfg.get("fastlio_odom_topic", "/ali_odom")
    cloud_topic = loc_cfg.get("world_cloud_topic", "/ali_cloud")

    fastlio = FastLIOInterface(use_ros=True)

    odom_sub = rospy.Subscriber(
        odom_topic, Odometry, fastlio.odometry_callback, queue_size=10)
    cloud_sub = rospy.Subscriber(
        cloud_topic, PointCloud2, fastlio.pointcloud_callback, queue_size=10)

    logger.info("[Bridge] Subscribed to %s + %s (FastLIOInterface)", odom_topic, cloud_topic)

    # 保存引用防止被 GC
    fastlio._odom_sub = odom_sub
    fastlio._cloud_sub = cloud_sub
    return fastlio


def _grab_latest_snapshot(fastlio: FastLIOInterface):
    """获取最新 FAST-LIO 快照，与 pipeline.py _grab_latest_snapshot 完全一致。

    Returns:
        (pts, pose, cloud_seq, pose_seq, sync_ms) 或 (None, None, -1, -1, None)
    """
    pts = fastlio.points
    pose = fastlio.pose
    cloud_seq = fastlio.points_seq
    pose_seq = fastlio.pose_seq
    sync_ms = fastlio.sync_delta_ms
    if pts is None or pose is None:
        return None, None, -1, -1, None
    return pts.copy(), pose.copy(), cloud_seq, pose_seq, sync_ms


# ============================================================
# BEV 稀疏深度投影 (与 HALSS 语义图对齐)
# ============================================================

def project_bev_depth(pts_body, grid_res=64, out_size=128, max_range=30.0,
                     half_x=5.0, half_y=5.0):
    """机体系下视 ROI 点云 -> BEV 稀疏深度图 (固定 bounds, 可配置半宽)."""
    empty = np.full((out_size, out_size), max_range, dtype=np.float32)
    bounds = {"x_min": -half_x, "x_max": half_x,
              "y_min": -half_y, "y_max": half_y}
    if pts_body is None or len(pts_body) == 0:
        return empty, bounds

    pts = np.asarray(pts_body, dtype=np.float32)
    valid_pts = np.isfinite(pts).all(axis=1)
    pts = pts[valid_pts]
    if len(pts) == 0:
        return empty, bounds

    z_all = pts[:, 2]
    valid_z = (z_all > 0.01) & (z_all < max_range)
    pts = pts[valid_z]
    if len(pts) == 0:
        return empty, bounds

    x_span = 2.0 * half_x
    y_span = 2.0 * half_y
    x_min, y_min = -half_x, -half_y

    col_idx = np.rint((pts[:, 0] - x_min) / x_span * (grid_res - 1)).astype(np.int32)
    row_unflipped = np.rint((pts[:, 1] - y_min) / y_span * (grid_res - 1)).astype(np.int32)
    row_idx = (grid_res - 1) - row_unflipped

    valid = (row_idx >= 0) & (row_idx < grid_res) & (col_idx >= 0) & (col_idx < grid_res)
    row_idx, col_idx = row_idx[valid], col_idx[valid]
    z_vals = pts[valid, 2]
    if len(z_vals) == 0:
        return empty, bounds

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
    """点云 -> 与深度/语义同画布的栅格."""
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


def make_binary_semantic_vis(sem_map, safe_id=1, danger_id=9):
    """Semantic class map -> uint8 visualization: safe=white, danger=black, unknown=gray."""
    sem_vis = np.full(sem_map.shape, 128, dtype=np.uint8)
    sem_vis[sem_map == safe_id] = 255
    sem_vis[sem_map == danger_id] = 0
    return sem_vis


def compute_roi_half_from_height(pose_z_world, ground_z_world, projection_mode,
                                 training_camera, roi_fov_half_rad,
                                 roi_min_half, roi_max_half,
                                 halss_points=None, roi_height_source="pose_z"):
    """Compute dynamic ROI half-extent from current height above ground.

    For training_camera mode, uses the camera's rectangular FOV to compute
    separate forward/lateral half-extents.  For level_body_bev, uses a
    symmetric square FOV cone.

    Returns (half_x, half_y, height_m).
    """
    if roi_height_source == "pointcloud_median":
        if halss_points is not None and len(halss_points) > 10:
            H = float(np.median(np.asarray(halss_points, dtype=np.float32)[:, 2]))
        else:
            H = abs(float(pose_z_world) - ground_z_world)
    else:
        H = abs(float(pose_z_world) - ground_z_world)
    H = max(H, 0.1)
    if projection_mode == "training_camera":
        half_x, half_y = training_camera.ground_half_extents(H)
        half_x = max(roi_min_half, min(roi_max_half, half_x))
        half_y = max(roi_min_half, min(roi_max_half, half_y))
        return half_x, half_y, H
    half = H * np.tan(roi_fov_half_rad)
    half_clamped = max(roi_min_half, min(roi_max_half, half))
    return half_clamped, half_clamped, H


# ============================================================
# ONNX DRL 推理器
# ============================================================

class ONNXDRL:
    """轻量 ONNX DRL 推理 (从 TF1 PPO2 导出), 与 pipeline.py 对齐."""

    def __init__(self, onnx_path, obs_h=128, obs_w=128, dmax=30.0,
                 depth_norm_mode="raw_meters_graph_scaled",
                 semantic_norm_mode="raw_gray_graph_scaled"):
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
        self.depth_norm_mode = depth_norm_mode
        self.semantic_norm_mode = semantic_norm_mode
        if self.depth_norm_mode != "raw_meters_graph_scaled":
            raise ValueError(
                "ONNX graph already contains input/truediv; use "
                "observation.depth_norm_mode=raw_meters_graph_scaled"
            )
        if self.semantic_norm_mode != "raw_gray_graph_scaled":
            raise ValueError(
                "ONNX graph already contains input/truediv; use "
                "observation.semantic_norm_mode=raw_gray_graph_scaled"
            )

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
        """深度图 + 语义图 -> 动作索引 0-9 + 诊断信息 (与 pipeline.py 对齐)."""
        depth_clipped = np.clip(
            np.nan_to_num(depth_map, nan=self.dmax, posinf=self.dmax, neginf=0.0),
            0.0, self.dmax)
        # The exported graph contains SB2 policy scale=True as input/truediv.
        # Feed the same raw Box(0,255) values used by training exactly once.
        depth_ch = depth_clipped.astype(np.float32)

        sem_int = np.clip(sem_map, -1, 9).astype(np.int16)
        sem_ch = np.zeros_like(sem_int, dtype=np.float32)
        for class_id, gray_val in CLASS_TO_GRAY.items():
            sem_ch[sem_int == class_id] = float(gray_val)

        if flip_lr:
            depth_ch = np.fliplr(depth_ch)
            sem_ch = np.fliplr(sem_ch)
        if flip_ud:
            depth_ch = np.flipud(depth_ch)
            sem_ch = np.flipud(sem_ch)

        obs = np.stack([depth_ch, sem_ch], axis=-1)
        obs = np.expand_dims(obs, axis=0)

        logits = self._forward(obs)
        logits = np.asarray(logits[0], dtype=np.float32)
        action = int(np.argmax(logits))
        exps = np.exp(logits - float(np.max(logits)))
        probs = exps / max(float(np.sum(exps)), 1e-12)

        depth_after_graph_scale = depth_ch / 255.0
        sem_after_graph_scale = sem_ch / 255.0

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
            "action_probs": probs.astype(float).tolist(),
            "confidence": float(np.max(probs)),
            "depth_norm_min": float(depth_after_graph_scale.min()),
            "depth_norm_mean": float(depth_after_graph_scale.mean()),
            "depth_norm_max": float(depth_after_graph_scale.max()),
            "sem_norm_min": float(sem_after_graph_scale.min()),
            "sem_norm_mean": float(sem_after_graph_scale.mean()),
            "sem_norm_max": float(sem_after_graph_scale.max()),
            "logits": logits.astype(float).tolist(),
        }
        return action, logits, info

    def action_name(self, action_id):
        return ACTION_NAMES_DECOMP[action_id] if 0 <= action_id < len(ACTION_NAMES_DECOMP) else "?"


# ============================================================
# 可视化
# ============================================================

class LiveDisplay:
    """三窗口: 左=点云画布, 中=语义, 右=渲染深度."""

    def __init__(self, sz=300, dmax=30.0, cfg: dict = None):
        import matplotlib
        matplotlib.use("TkAgg")
        import matplotlib.pyplot as plt
        plt.ion()
        cfg = cfg or {}
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
        self.save_frames = bool(cfg.get("save_frames", False))
        self.save_raw_arrays = bool(cfg.get("save_raw_arrays", False))
        self.save_dir = cfg.get("save_dir", "experiments/frames")
        self.depth_vmax_m = float(cfg.get("depth_vmax_m", dmax))
        self._frame_idx = 0
        if self.save_frames or self.save_raw_arrays:
            os.makedirs(self.save_dir, exist_ok=True)

    def _resize(self, img):
        h, w = img.shape[:2]
        if h != self.sz or w != self.sz:
            return cv2.resize(img, (self.sz, self.sz), interpolation=cv2.INTER_NEAREST)
        return img

    def _depth_camera_u8(self, depth_map: np.ndarray) -> np.ndarray:
        """Metric depth camera-style grayscale: 0m=black, vmax=white."""
        depth_m = np.nan_to_num(
            depth_map.astype(np.float32, copy=False),
            nan=self.depth_vmax_m,
            posinf=self.depth_vmax_m,
            neginf=0.0,
        )
        depth_norm = np.clip(depth_m / self.depth_vmax_m, 0.0, 1.0)
        return (depth_norm * 255.0).astype(np.uint8)

    def update(self, sem_map: np.ndarray, depth_map: np.ndarray, pc_canvas=None, raw_arrays: dict = None):
        sem_vis = make_binary_semantic_vis(sem_map)
        if pc_canvas is None:
            pc_canvas = np.full_like(depth_map, self.dmax)

        sem_disp = self._resize(sem_vis)
        depth_disp = self._resize(depth_map)
        pc_disp = self._resize(pc_canvas)

        if not self._ready:
            self.im_pc = self.ax_pc.imshow(
                pc_disp, cmap="inferno", vmin=0, vmax=self.dmax, interpolation="nearest")
            self.ax_pc.set_title("Point Cloud Canvas")
            self.pc_cbar = self.fig.colorbar(
                self.im_pc, ax=self.ax_pc, fraction=0.046, pad=0.04, label="z (m)")

            self.im_sem = self.ax_sem.imshow(
                sem_disp, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
            self.ax_sem.set_title("Semantic (white=safe, black=danger)")

            self.im_depth = self.ax_depth.imshow(
                depth_disp, cmap="inferno", vmin=0, vmax=self.dmax, interpolation="nearest")
            self.ax_depth.set_title("Rendered Depth (NN Fill)")
            self.depth_cbar = self.fig.colorbar(
                self.im_depth,
                ax=self.ax_depth,
                fraction=0.046,
                pad=0.04,
                label="m",
            )
            self.depth_cbar.set_ticks(np.linspace(0.0, self.depth_vmax_m, 4))
            for ax in [self.ax_pc, self.ax_sem, self.ax_depth]:
                ax.axis("off")
            self._ready = True
        else:
            self.im_pc.set_data(pc_disp)
            self.im_sem.set_data(sem_disp)
            self.im_depth.set_data(depth_disp)
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
        if self.save_frames:
            sem_bgr = sem_vis
            depth_vis = self._depth_camera_u8(depth_map)
            cv2.imwrite(os.path.join(self.save_dir, f"{self._frame_idx:06d}_binary_semantic.png"), sem_bgr)
            cv2.imwrite(os.path.join(self.save_dir, f"{self._frame_idx:06d}_depth.png"), depth_vis)
        if self.save_raw_arrays and raw_arrays:
            np.savez_compressed(
                os.path.join(self.save_dir, f"{self._frame_idx:06d}_calib_frame.npz"),
                **raw_arrays,
            )
        if self.save_frames or self.save_raw_arrays:
            self._frame_idx += 1


# ============================================================
# 主逻辑
# ============================================================

def main():
    args = _parse_args()
    _apply_cli_overrides(CFG, args)
    dmax = float(args.dmax)
    sz = int(CFG["observation"].get("img_width", 128))
    override_str = f"sem={args.semantic_override} depth={args.depth_override}"
    if args.flip_lr:
        override_str += " flip-lr"
    if args.flip_ud:
        override_str += " flip-ud"
    logger.info(
        "No-control config: yaw_rate_rad_s=%.3f save_raw_arrays=%s save_frames=%s save_dir=%s max_cloud_odom_sync_ms=%.0f",
        float(CFG["uav"].get("yaw_rate_rad_s", 0.0)),
        CFG.get("visualization", {}).get("save_raw_arrays", False),
        CFG.get("visualization", {}).get("save_frames", False),
        CFG.get("visualization", {}).get("save_dir", "experiments/frames"),
        float(CFG.get("runtime", {}).get("max_cloud_odom_sync_ms", 100.0)),
    )
    if args.require_yaw_rate and abs(float(CFG["uav"].get("yaw_rate_rad_s", 0.0))) < 1e-6:
        logger.error("yaw_rate_rad_s is zero and --require-yaw-rate is set.")
        sys.exit(2)

    print("Initializing ROS FAST-LIO bridge...", flush=True)
    fastlio = _init_ros_bridge()
    print("ROS FAST-LIO bridge OK.", flush=True)

    # 等待 FAST-LIO 数据就绪 (与 pipeline.py 一致)
    logger.info("[Bridge] Waiting for FAST-LIO data...")
    while not fastlio.initialized:
        rospy.sleep(0.02)
    logger.info("[Bridge] FAST-LIO ready.")

    # 初始化模块
    pcfg = CFG["perception"]
    halss = HALSSBayesianEvaluator(pcfg)
    sem_gen = SemanticGenerator(pcfg)
    logger.info("HALSS + SemanticGenerator OK")

    # ---- 深度投影模式 (与 pipeline.py 对齐) ----
    depth_cfg = CFG.get("depth_projection", {})
    projection_mode = str(depth_cfg.get("mode", "training_camera")).lower()
    if projection_mode not in ("training_camera", "level_body_bev"):
        logger.warning(
            "Unknown depth_projection.mode=%s, falling back to training_camera",
            projection_mode,
        )
        projection_mode = "training_camera"
    training_camera = TrainingCameraModel.from_config(
        depth_cfg.get("training_camera", {}),
        output_width=sz,
        output_height=sz,
        far_m=dmax,
    )
    danger_id = int(pcfg.get("danger_class_id", 9))
    safe_id = int(pcfg.get("safe_class_id", 1))
    logger.info("Projection mode: %s", projection_mode)
    if projection_mode == "training_camera":
        logger.info(
            "  TrainingCamera: FOV=%.1fx%.1fdeg "
            "scaled_intrinsics=fx=%.3f fy=%.3f cx=%.3f cy=%.3f",
            training_camera.horizontal_fov_deg,
            training_camera.vertical_fov_deg,
            training_camera.fx, training_camera.fy,
            training_camera.cx, training_camera.cy,
        )

    onnx_path = args.onnx_model
    if not os.path.exists(onnx_path):
        logger.error(f"ONNX model not found: {onnx_path}")
        logger.error("Run: python scripts/export_ppo2_to_onnx.py first")
        rospy.signal_shutdown("onnx model missing")
        sys.exit(1)
    drl = ONNXDRL(onnx_path, obs_h=sz, obs_w=sz, dmax=dmax)
    logger.info(f"ONNX DRL loaded: {onnx_path}")

    decomposer = ActionDecomposer(CFG["uav"])
    logger.info(
        "Action mapping: frame=%s lateral_sign=%d act3=%s",
        decomposer.action_frame,
        decomposer.action_lateral_sign,
        decomposer.action_id_to_name(3),
    )
    display = None if args.no_display else LiveDisplay(
        sz=300, dmax=dmax, cfg=CFG.get("visualization", {}))

    print("Pipeline ready. Waiting for /cloud_registered + /Odometry...", flush=True)
    logger.info("Pipeline ready. Waiting for FAST-LIO data...")
    logger.info("  Depth backend: %s + NN-fill", projection_mode)
    logger.info("  DRL backend:   ONNX (raw input: depth=meters, sem=gray; ONNX internal /255)")
    logger.info("  Control:       NONE (print only)")
    logger.info("  Overrides:     %s", override_str)
    logger.info("  Action names:  from ActionDecomposer sign=%s: %s", ACTION_SIGN, ACTION_NAMES_DECOMP)
    seq = 0
    last_cloud_seq = -1
    frame_id = 0
    last_print = time.time()
    action_names = ACTION_NAMES_DECOMP
    action_counts = [0] * 10
    start_wall = time.perf_counter()
    max_sync_ms = float(CFG.get("runtime", {}).get("max_cloud_odom_sync_ms", 100.0))
    last_wait_log = 0.0
    half_x = float(CFG.get("perception", {}).get("halss_roi_half_x_m", 5.0))
    half_y = float(CFG.get("perception", {}).get("halss_roi_half_y_m", 5.0))
    logger.info("  ROI: x∈[-%.1f,%.1f]m y∈[-%.1f,%.1f]m (mode=%s)",
                half_x, half_x, half_y, half_y, projection_mode)

    # ---- 动态 FOV ROI 参数 ----
    roi_dynamic = bool(CFG.get("perception", {}).get("halss_roi_dynamic_enabled", True))
    roi_fov_half_rad = np.radians(float(CFG.get("perception", {}).get("halss_roi_fov_half_deg", 45.0)))
    roi_min_half = float(CFG.get("perception", {}).get("halss_roi_min_half_m", 0.5))
    roi_max_half = float(CFG.get("perception", {}).get("halss_roi_max_half_m", 15.0))
    roi_height_src = str(CFG.get("perception", {}).get("halss_roi_height_source", "pose_z"))
    # 记录起飞点 Fast-LIO z 作为地面参考
    ground_z_world = None
    if roi_dynamic:
        logger.info("  ROI mode: dynamic FOV=%.0f° min=%.1fm max=%.1fm height_src=%s",
                     np.degrees(roi_fov_half_rad) * 2, roi_min_half, roi_max_half, roi_height_src)

    try:
        while not rospy.is_shutdown():
            if args.duration_sec > 0.0 and (time.perf_counter() - start_wall) >= args.duration_sec:
                logger.info(
                    "No-control duration reached: %.1fs, processed_frames=%d",
                    args.duration_sec, seq,
                )
                break

            # ---- 获取最新 FAST-LIO 快照 (对齐 pipeline.py _grab_latest_snapshot) ----
            frame_pts, frame_pose, cloud_seq, pose_seq, sync_ms = _grab_latest_snapshot(fastlio)
            if frame_pts is None or frame_pose is None:
                rospy.sleep(0.05)
                now = time.time()
                if now - last_wait_log > 3.0:
                    print(
                        "  Waiting for FAST-LIO data... "
                        "(need /cloud_registered + /Odometry)",
                        flush=True,
                    )
                    last_wait_log = now
                continue

            if cloud_seq <= last_cloud_seq:
                rospy.sleep(0.005)
                continue
            last_cloud_seq = cloud_seq

            # 时间戳缺失或同步超差 → 跳过，与 pipeline.py 一致
            if sync_ms is None:
                logger.warning(
                    "Missing FAST-LIO header timestamps; skip cloud_seq=%d pose_seq=%d",
                    cloud_seq, pose_seq,
                )
                continue

            if sync_ms > max_sync_ms:
                logger.warning(
                    "Drop stale cloud/odom pair: sync=%.0fms > %.0fms cloud_seq=%d pose_seq=%d",
                    sync_ms, max_sync_ms, cloud_seq, pose_seq,
                )
                continue

            pose_xyz = frame_pose[:3]
            rpy = frame_pose[3:]
            roll, pitch, yaw = float(rpy[0]), float(rpy[1]), float(rpy[2])
            pts = frame_pts

            seq += 1
            t0 = time.time()

            # ---- 动态 FOV ROI: 根据当前对地高度计算半宽 ----
            if ground_z_world is None:
                ground_z_world = float(pose_xyz[2])
            if roi_dynamic:
                cur_half_x, cur_half_y, _dyn_h = compute_roi_half_from_height(
                    float(pose_xyz[2]), ground_z_world, projection_mode,
                    training_camera, roi_fov_half_rad,
                    roi_min_half, roi_max_half,
                    halss_points=None, roi_height_source=roi_height_src,
                )
            else:
                cur_half_x, cur_half_y = half_x, half_y

            # 1. HALSS 贝叶斯安全语义: 世界系点云 -> 水平机体动态 ROI
            halss_pts, halss_stats = world_to_level_body_roi(
                pts, pose_xyz, rpy, pcfg,
                half_x=cur_half_x, half_y=cur_half_y,
            )
            roi_bounds = {
                "x_min": -cur_half_x, "x_max": cur_half_x,
                "y_min": -cur_half_y, "y_max": cur_half_y,
            }
            bev = halss.evaluate(halss_pts, fixed_bounds=roi_bounds)

            # 2. 语义图
            if bev is not None:
                bev_data = bev.get("bev_data", bev) if isinstance(bev, dict) else bev
                sem_map = sem_gen.generate(bev_data)
            else:
                sem_map = np.full((sz, sz), danger_id, dtype=np.uint8)

            # 3. 深度投影 + 语义投影 (与 pipeline.py 对齐: 支持 training_camera / level_body_bev)
            semantic_valid_mask = np.ones_like(sem_map, dtype=bool)
            if projection_mode == "training_camera":
                sparse_depth, valid_mask, sem_map, semantic_valid_mask = (
                    project_training_camera(
                        halss_pts, sem_map, roi_bounds, training_camera,
                        danger_id=danger_id,
                    )
                )
                # 点云画布 (用 roi_bounds)
                pc_canvas = project_pointcloud_canvas(
                    halss_pts, roi_bounds, out_size=sz, dmax=dmax,
                )
            else:
                sparse_depth, bounds = project_bev_depth(
                    halss_pts,
                    grid_res=pcfg.get("halss_grid_res", 64),
                    out_size=sz,
                    max_range=dmax,
                    half_x=cur_half_x,
                    half_y=cur_half_y,
                )
                valid_mask = (sparse_depth < dmax) & (sparse_depth > 0.01)
                pc_canvas = project_pointcloud_canvas(
                    halss_pts, bounds, out_size=sz, dmax=dmax,
                )

            # 4. NN-fill 深度渲染
            rendered_depth = render_sparse_depth(sparse_depth, valid_mask, dmax)

            # ---- 诊断: semantic override ----
            if args.semantic_override == "all_safe":
                sem_map_drl = np.full((sz, sz), safe_id, dtype=np.uint8)
            elif args.semantic_override == "all_danger":
                sem_map_drl = np.full((sz, sz), danger_id, dtype=np.uint8)
            elif args.semantic_override == "center_safe":
                sem_map_drl = np.full((sz, sz), danger_id, dtype=np.uint8)
                c = sz // 2
                r = sz // 4
                sem_map_drl[c-r:c+r, c-r:c+r] = safe_id
            else:
                sem_map_drl = sem_map

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
            action_counts[action_id] += 1

            # 6. 动作解算
            action_name = drl.action_name(action_id)
            v_body, v_ned, yaw_rate = decomposer.decompose(action_id, yaw)

            dt_ms = (time.time() - t0) * 1000
            frame_id += 1

            # ---- 每帧打印动作 (print 绕过日志缓冲) ----
            top3_idx = np.argsort(logits)[-3:][::-1]
            sp = drl_info["softmax_probs"]
            top3_str = " ".join(f"{action_names[i]}:p={sp[i]:.2f}" for i in top3_idx)
            print(
                f"\r[frame {frame_id:04d}] ACTION={action_id}({action_name})  "
                f"top3: {top3_str}  lat={dt_ms:.0f}ms  "
                f"pts={len(halss_pts)} sync={sync_ms:.0f}ms",
                end="",
            )

            sem_safe_raw = int((sem_map == safe_id).sum())
            sem_danger_raw = int((sem_map == danger_id).sum())
            sem_safe_drl = int((sem_map_drl == safe_id).sum())
            sem_danger_drl = int((sem_map_drl == danger_id).sum())
            binary_semantic_vis = make_binary_semantic_vis(
                sem_map,
                safe_id=safe_id,
                danger_id=danger_id,
            )
            # 训练相机模式下, 标记投影覆盖范围外的像素为未知 (灰色)
            if projection_mode == "training_camera":
                binary_semantic_vis[~semantic_valid_mask] = 128
            raw_arrays = {
                "sparse_depth": sparse_depth.astype(np.float32),
                "valid_mask": valid_mask.astype(np.uint8),
                "semantic_valid_mask": semantic_valid_mask.astype(np.uint8),
                "dense_depth": rendered_depth.astype(np.float32),
                "sem_map": sem_map.astype(np.uint8),
                "binary_semantic_vis": binary_semantic_vis.astype(np.uint8),
                "yaw_rad": np.array(yaw, dtype=np.float32),
                "cloud_odom_sync_ms": np.array(sync_ms, dtype=np.float32),
                "pose": frame_pose.astype(np.float32),
                "halss_points_body": halss_pts.astype(np.float32),
                "halss_lidar_position_body_m": halss_stats["lidar_position_body_m"],
                "action_id": np.array(action_id, dtype=np.int32),
                "v_body": v_body.astype(np.float32),
                "v_ned": v_ned.astype(np.float32),
                "cloud_seq": np.array(cloud_seq, dtype=np.int32),
                "pose_seq": np.array(pose_seq, dtype=np.int32),
            }

            # 终端打印 (每 0.5s 带详细诊断)
            if time.time() - last_print >= 0.5:
                sp_valid = sparse_depth[valid_mask]
                sp_med = float(np.median(sp_valid)) if len(sp_valid) else float("nan")
                rd_med = float(np.median(rendered_depth))
                top3_str = " ".join(
                    f"{action_names[i]}:logit={logits[i]:.1f} p={sp[i]:.2f}" for i in top3_idx)
                obs_str = (
                    f"depth_in={drl_info['depth_input_mean']:.1f}m "
                    f"[{drl_info['depth_input_min']:.1f},{drl_info['depth_input_max']:.1f}] "
                    f"sem_in={drl_info['sem_input_mean']:.0f} "
                    f"[{drl_info['sem_input_min']:.0f},{drl_info['sem_input_max']:.0f}] "
                    f"sem_uniq={drl_info['sem_input_unique']} "
                    f"obs_raw=[{drl_info['obs_raw_min']:.1f},{drl_info['obs_raw_max']:.1f}]"
                )
                total_acts = sum(action_counts)
                act_dist = " ".join(
                    f"{action_names[i]}:{action_counts[i]}" for i in range(10)
                    if action_counts[i] > 0) if total_acts > 0 else "none"

                logger.info(
                    f"[{frame_id:5d}] pts={len(pts):4d}->body={len(halss_pts):3d} "
                    f"sp_med={sp_med:.1f}m rd_med={rd_med:.1f}m "
                    f"sem_raw safe={sem_safe_raw} danger={sem_danger_raw} "
                    f"sem_drl safe={sem_safe_drl} danger={sem_danger_drl} "
                    f"lat={dt_ms:.0f}ms sync={sync_ms:.0f}ms "
                    f"cloud_seq={cloud_seq} pose_seq={pose_seq}")
                logger.info(f"  {obs_str}")
                logger.info(
                    f"  >>> ACTION: {action_id} ({action_name})  "
                    f"top3: {top3_str}")
                logger.info(
                    f"  yaw={np.degrees(yaw):+.0f}deg  action={action_id}({action_name})  "
                    f"v_body=[{v_body[0]:+.1f},{v_body[1]:+.1f},{v_body[2]:+.1f}]  "
                    f"v_ned=[{v_ned[0]:+.1f},{v_ned[1]:+.1f},{v_ned[2]:+.1f}]  "
                    f"yr={yaw_rate:+.2f}rad/s")
                logger.info(f"  act_dist: {act_dist}")

                if args.diagnose_drl:
                    logger.info(f"  logits: {' '.join(f'{action_names[i]}={logits[i]:.2f}' for i in range(10))}")
                    logger.info(f"  probs:  {' '.join(f'{action_names[i]}={sp[i]:.3f}' for i in range(10))}")
                    logger.info(f"  depth_input: mean={drl_info['depth_input_mean']:.1f}m min={drl_info['depth_input_min']:.1f} max={drl_info['depth_input_max']:.1f}")
                    logger.info(f"  sem_input:   mean={drl_info['sem_input_mean']:.0f} unique={drl_info['sem_input_unique']}")
                    logger.info(f"  obs_raw range: [{drl_info['obs_raw_min']:.1f}, {drl_info['obs_raw_max']:.1f}] (ONNX internal /255)")

                if not np.isnan(sp_med) and abs(rd_med - sp_med) > 3.0:
                    logger.warning(
                        f"  rd_med({rd_med:.1f}) deviates from sp_med({sp_med:.1f}) by >3m")

                last_print = time.time()

            if display is not None:
                display.update(sem_map, rendered_depth, pc_canvas, raw_arrays=raw_arrays)
            if args.max_frames > 0 and seq >= args.max_frames:
                logger.info("No-control max frames reached: %d", seq)
                break

    except KeyboardInterrupt:
        logger.info("\nInterrupted.")
    finally:
        total = sum(action_counts)
        if total > 0:
            logger.info("=== Final Action Distribution ===")
            for i in range(10):
                logger.info(f"  {action_names[i]}: {action_counts[i]} ({100*action_counts[i]/total:.1f}%)")
        rospy.signal_shutdown("test_live_nocontrol stopped")
        logger.info("Done.")


if __name__ == "__main__":
    main()
