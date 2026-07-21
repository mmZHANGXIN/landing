#!/usr/bin/env python3
"""
Orin Landing — 离线 rosbag 回放感知 + DRL 推理管线
====================================================
与 pipeline.py 高度一致的感知决策管线，从 rosbag 读取 FAST-LIO 去畸变点云和位姿，
跑完整的 HALSS 贝叶斯语义 + training-camera 深度投影 + NN-fill + ONNX PPO 推理，
实时 OpenCV 可视化并打印每帧动作。

支持:
  - 室内 world-cloud 模式: /ali_cloud + /ali_odom
  - 室外 body-cloud 模式: /cloud_registered_body + /mavros/local_position/odom

用法:
  source /opt/ros/noetic/setup.bash
  python scripts/replay_bag_offline.py \\
      --bag experiments/runs/20260720_174909_orin_landing/input.bag \\
      --config ./config/experiment_outdoor_gps.yaml \\
      --onnx-model weights/ppo2_policy.onnx
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
import cv2

# 将项目根目录添加到 sys.path (与 test_live_nocontrol.py 一致)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("ReplayBag")

# ──────────────────────────────────────────────
# 解析 PointCloud2 → numpy (与 pipeline.py 一致)
# ──────────────────────────────────────────────
def _pc2_to_numpy(msg) -> np.ndarray:
    """sensor_msgs/PointCloud2 → (N, 3) float32, 滤除 NaN."""
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


def _quat_to_euler(x, y, z, w):
    """四元数 → 欧拉角 (roll, pitch, yaw) rad."""
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)
    sinp = 2.0 * (w * y - z * x)
    pitch = math.asin(max(-1.0, min(1.0, sinp))) if abs(sinp) < 1 else math.copysign(math.pi / 2, sinp)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return roll, pitch, yaw


def _stamp_to_sec(stamp) -> float:
    return float(stamp.secs) + float(stamp.nsecs) * 1e-9


# ──────────────────────────────────────────────
# Config 加载 (与 pipeline.py 完全一致)
# ──────────────────────────────────────────────
def _load_config(path: str) -> dict:
    config_path = Path(path).resolve()
    import yaml
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    parent = cfg.pop("extends", None)
    if parent:
        parent_path = Path(parent)
        if not parent_path.is_absolute():
            parent_path = config_path.parent / parent_path
        base = _load_config(str(parent_path))
        return _merge_config_overrides(base, cfg)
    return cfg


def _merge_config_overrides(cfg: dict, overrides: dict) -> dict:
    for section, values in overrides.items():
        if isinstance(values, dict) and isinstance(cfg.get(section), dict):
            cfg.setdefault(section, {})
            _merge_config_overrides(cfg[section], values)
        else:
            cfg[section] = values
    return cfg


# ──────────────────────────────────────────────
# 与 pipeline.py 共享的辅助函数
# ──────────────────────────────────────────────
CLASS_TO_GRAY = {
    -1: 0, 0: 10, 1: 30, 2: 60, 3: 70, 4: 20,
     5: 40, 6: 80, 7: 90, 8: 50, 9: 250,
}


def make_binary_semantic_vis(sem_map, safe_id=1, danger_id=9):
    sem_vis = np.full(sem_map.shape, 128, dtype=np.uint8)
    sem_vis[sem_map == safe_id] = 255
    sem_vis[sem_map == danger_id] = 0
    return sem_vis


def _limited_xy_velocity(dx, dy, max_speed_mps, kp_s=1.0):
    distance = math.hypot(float(dx), float(dy))
    if distance <= 1e-9:
        return 0.0, 0.0
    speed = min(float(max_speed_mps), float(kp_s) * distance)
    scale = speed / distance
    return float(dx) * scale, float(dy) * scale


def _limited_axis_velocity(error, max_speed_mps, kp_s=1.0):
    return max(-float(max_speed_mps), min(float(max_speed_mps), float(kp_s) * float(error)))


def _top_probs(probs, action_names=None, k=3):
    if probs is None:
        return "p=n/a"
    pairs = sorted(enumerate(probs), key=lambda item: item[1], reverse=True)[:k]
    if action_names is None:
        return "p=" + ",".join(f"{idx}:{prob:.2f}" for idx, prob in pairs)
    return "p=" + ",".join(f"{idx}:{action_names[idx]}:{prob:.2f}" for idx, prob in pairs)


# ──────────────────────────────────────────────
# BEV 深度投影 (与 pipeline.py 共享)
# ──────────────────────────────────────────────
def project_bev_depth(pts_body, grid_res=64, out_size=128, max_range=30.0,
                      half_x=5.0, half_y=5.0):
    empty = np.full((out_size, out_size), max_range, dtype=np.float32)
    bounds = {"x_min": -half_x, "x_max": half_x,
              "y_min": -half_y, "y_max": half_y}
    if pts_body is None or len(pts_body) == 0:
        return empty, bounds
    pts = np.asarray(pts_body, dtype=np.float32)
    pts = pts[np.isfinite(pts).all(axis=1)]
    if len(pts) == 0:
        return empty, bounds
    z_all = pts[:, 2]
    pts = pts[(z_all > 0.01) & (z_all < max_range)]
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
        grid[np.isnan(grid) | (grid <= 0)] = max_range
    else:
        grid = np.where(np.isnan(grid), max_range, grid)
    return grid.astype(np.float32), bounds


def render_sparse_depth(sparse_depth, valid_mask, dmax, min_valid=5, median_ksize=5):
    if valid_mask.sum() < min_valid:
        return np.where(valid_mask, sparse_depth, dmax).astype(np.float32)
    invalid = ~valid_mask
    _, labels = cv2.distanceTransformWithLabels(
        invalid.astype(np.uint8),
        distanceType=cv2.DIST_L2, maskSize=5,
        labelType=cv2.DIST_LABEL_PIXEL,
    )
    valid_coords = np.column_stack(np.where(valid_mask))
    label_vals = labels[invalid]
    nearest_idx = np.clip(label_vals - 1, 0, len(valid_coords) - 1)
    filled = sparse_depth.copy()
    filled[invalid] = sparse_depth[
        valid_coords[nearest_idx, 0],
        valid_coords[nearest_idx, 1],
    ]
    if median_ksize >= 3:
        smoothed = cv2.medianBlur(filled.astype(np.float32), median_ksize)
    else:
        smoothed = filled.astype(np.float32)
    rendered = np.where(valid_mask, sparse_depth, smoothed)
    return np.clip(rendered, 0.0, dmax).astype(np.float32)


# ──────────────────────────────────────────────
# ONNX DRL (与 pipeline.py 完全一致)
# ──────────────────────────────────────────────
class ONNXDRL:
    def __init__(self, onnx_path: str, obs_h=128, obs_w=128, dmax=30.0,
                 depth_norm_mode="raw_meters_graph_scaled",
                 semantic_norm_mode="raw_gray_graph_scaled"):
        import onnxruntime as ort
        if not Path(onnx_path).is_file():
            raise FileNotFoundError(f"ONNX model not found: {onnx_path}")
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        opts.inter_op_num_threads = 1
        self.session = ort.InferenceSession(onnx_path, opts)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        in_shape = self.session.get_inputs()[0].shape
        self.layout = "chw" if len(in_shape) == 4 and in_shape[1] in (2, 3) else "hwc"
        self.obs_h, self.obs_w = obs_h, obs_w
        self.dmax = dmax
        if depth_norm_mode != "raw_meters_graph_scaled":
            raise ValueError("ONNX graph already contains input/truediv")
        if semantic_norm_mode != "raw_gray_graph_scaled":
            raise ValueError("ONNX graph already contains input/truediv")
        dummy = np.zeros((1, obs_h, obs_w, 2), dtype=np.float32)
        self._forward(dummy)
        logger.info("[ONNX] model=%s layout=%s warmup=OK", onnx_path, self.layout)

    def _forward(self, obs_raw):
        if self.layout == "chw":
            inp = np.transpose(obs_raw, (0, 3, 1, 2)).astype(np.float32)
        else:
            inp = obs_raw.astype(np.float32)
        return self.session.run([self.output_name], {self.input_name: inp})[0]

    def predict(self, depth_map, sem_map):
        depth_clipped = np.clip(
            np.nan_to_num(depth_map, nan=self.dmax, posinf=self.dmax, neginf=0.0),
            0.0, self.dmax,
        )
        depth_ch = depth_clipped.astype(np.float32)
        sem_int = np.clip(sem_map, -1, 9).astype(np.int16)
        sem_ch = np.zeros_like(sem_int, dtype=np.float32)
        for class_id, gray_val in CLASS_TO_GRAY.items():
            sem_ch[sem_int == class_id] = float(gray_val)
        obs = np.expand_dims(np.stack([depth_ch, sem_ch], axis=-1), axis=0)
        logits = self._forward(obs)[0]
        logits = np.asarray(logits, dtype=np.float32)
        exps = np.exp(logits - float(np.max(logits)))
        probs = exps / max(float(np.sum(exps)), 1e-12)
        action = int(np.argmax(logits))
        depth_after = depth_ch / 255.0
        sem_after = sem_ch / 255.0
        info = {
            "depth_raw_median": float(np.median(depth_map)),
            "depth_input_mean": float(depth_ch.mean()),
            "depth_input_min": float(depth_ch.min()),
            "depth_input_max": float(depth_ch.max()),
            "sem_input_mean": float(sem_ch.mean()),
            "sem_input_min": float(sem_ch.min()),
            "sem_input_max": float(sem_ch.max()),
            "sem_input_unique": [round(float(x), 6) for x in sorted(np.unique(sem_ch))],
            "obs_raw_min": float(obs.min()),
            "obs_raw_max": float(obs.max()),
            "softmax_probs": probs.astype(float).tolist(),
            "action_probs": probs.astype(float).tolist(),
            "confidence": float(np.max(probs)),
            "depth_norm_min": float(depth_after.min()),
            "depth_norm_mean": float(depth_after.mean()),
            "depth_norm_max": float(depth_after.max()),
            "sem_norm_min": float(sem_after.min()),
            "sem_norm_mean": float(sem_after.mean()),
            "sem_norm_max": float(sem_after.max()),
            "logits": logits.astype(float).tolist(),
        }
        return action, info


# ──────────────────────────────────────────────
# OpenCV 可视化 (与 pipeline.py RealtimeVisualizer 完全对齐)
# ──────────────────────────────────────────────
class ReplayVisualizer:
    """双窗口: 二值安全语义图 + 推断深度图 (inferno色表+色条)."""

    def __init__(self, dmax=30.0, depth_vmax_m=30.0, display_width=300):
        self.dmax = dmax
        self.depth_vmax_m = depth_vmax_m
        self.display_width = int(display_width)
        self._windows_ready = False
        # 深度图显示高基于源图高宽比，不固定为正方形
        self._disp_h = 300

    def _init_windows(self):
        cv2.namedWindow("1.Binary Semantic Map", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("1.Binary Semantic Map", self.display_width, self._disp_h)
        cv2.moveWindow("1.Binary Semantic Map", 20, 50)

        cv2.namedWindow("2.Depth Map", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("2.Depth Map", self.display_width, self._disp_h)
        cv2.moveWindow("2.Depth Map", 20 + self.display_width + 10, 50)

        self._windows_ready = True

    def update(self, depth_map, sem_map, binary_semantic_vis):
        """更新双窗口。语义图自适应裁剪填满窗口，深度图固定 inferno 色表+色条。

        DRL 输入始终是 128x128 完整图像（含外围 unknown 区域）。
        可视化只显示有效语义区域的裁剪放大视图。
        """
        h, w = depth_map.shape[:2]
        disp_w = self.display_width
        disp_h = self._disp_h

        if not self._windows_ready:
            self._init_windows()

        # ---- 1. 二值语义图: 自适应裁剪非灰色区域后 resize 填满窗口 ----
        if binary_semantic_vis is not None:
            sem_disp = binary_semantic_vis
            if sem_disp.ndim == 3:
                sem_disp = cv2.cvtColor(sem_disp.astype(np.uint8), cv2.COLOR_BGR2GRAY)
        else:
            sem_disp = make_binary_semantic_vis(sem_map)

        # 自适应裁剪: 找到非灰色 (≠128) 像素的包围盒，裁掉纯灰色边框
        valid_mask = sem_disp != 128
        if valid_mask.any():
            rows = np.any(valid_mask, axis=1)
            cols = np.any(valid_mask, axis=0)
            r_min, r_max = np.where(rows)[0][[0, -1]]
            c_min, c_max = np.where(cols)[0][[0, -1]]
            # 加一点 padding 避免边缘贴死
            pad = 2
            r_min = max(0, r_min - pad)
            r_max = min(h - 1, r_max + pad)
            c_min = max(0, c_min - pad)
            c_max = min(w - 1, c_max + pad)
            sem_cropped = sem_disp[r_min:r_max + 1, c_min:c_max + 1]
        else:
            sem_cropped = sem_disp  # 全是灰色时保留原图

        sem_bgr = cv2.cvtColor(sem_cropped.astype(np.uint8), cv2.COLOR_GRAY2BGR)
        sem_bgr = cv2.resize(sem_bgr, (disp_w, disp_h), interpolation=cv2.INTER_NEAREST)
        cv2.imshow("1.Binary Semantic Map", sem_bgr)

        # ---- 2. 深度图: 同样自适应裁剪后 inferno 色表 + 右侧色条 ----
        depth_m = np.nan_to_num(
            depth_map.astype(np.float32, copy=False),
            nan=self.depth_vmax_m, posinf=self.depth_vmax_m, neginf=0.0,
        )
        depth_norm = np.clip(depth_m / self.depth_vmax_m, 0.0, 1.0)
        depth_u8 = (depth_norm * 255.0).astype(np.uint8)

        # 深度图也用同样的 bounding box 裁剪（与语义图空间对齐）
        if valid_mask.any():
            depth_cropped = depth_u8[r_min:r_max + 1, c_min:c_max + 1]
        else:
            depth_cropped = depth_u8

        depth_resized = cv2.resize(depth_cropped, (disp_w, disp_h), interpolation=cv2.INTER_NEAREST)
        colored = cv2.applyColorMap(depth_resized, cv2.COLORMAP_INFERNO)

        bar_w = 40
        with_bar = np.zeros((disp_h, disp_w + bar_w + 10, 3), dtype=np.uint8)
        with_bar[:, :disp_w] = colored
        for row in range(disp_h):
            val = 255 - int(row / max(disp_h - 1, 1) * 255)
            with_bar[row, disp_w + 5:disp_w + bar_w + 5] = cv2.applyColorMap(
                np.array([[val]], dtype=np.uint8), cv2.COLORMAP_INFERNO)[0, 0]
        bar_x = disp_w + 5
        cv2.putText(with_bar, "0m", (bar_x - 5, disp_h - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(with_bar, f"{int(self.depth_vmax_m)}m", (bar_x - 5, 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.imshow("2.Depth Map", with_bar)

        cv2.waitKey(1)

    def close(self):
        cv2.destroyAllWindows()


# ──────────────────────────────────────────────
# 主回放逻辑
# ──────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="离线 rosbag 回放 — 感知 + ONNX DRL 推理 (与 pipeline.py 一致)"
    )
    parser.add_argument("--bag", type=str, required=True,
                        help="输入的 rosbag 路径 (experiments/runs/*/input.bag)")
    parser.add_argument("--config", type=str, required=True,
                        help="实验配置路径 (experiment_outdoor_gps.yaml)")
    parser.add_argument("--onnx-model", type=str, default="weights/ppo2_policy.onnx",
                        help="ONNX DRL 策略模型路径")
    parser.add_argument("--dmax", type=float, default=30.0,
                        help="深度最大距离 (米)")
    parser.add_argument("--ground-z", type=float, default=None,
                        help="手动指定地面 Z (米); 默认取第一帧 pose.z")
    parser.add_argument("--max-frames", type=int, default=0,
                        help="最多处理 N 帧 (0=处理全部)")
    parser.add_argument("--no-display", action="store_true",
                        help="关闭可视化窗口")
    parser.add_argument("--rate", type=float, default=0.0,
                        help="回放速率倍率 (0=尽可能快, 1=实时)")
    parser.add_argument("--world-cloud", action="store_true",
                        help="强制使用 world-cloud 模式 (/ali_cloud + /ali_odom)")
    parser.add_argument("--body-cloud", action="store_true",
                        help="强制使用 body-cloud 模式 (/cloud_registered_body + /mavros/local_position/odom)")
    args = parser.parse_args()

    # ── 加载配置 ──
    cfg = _load_config(args.config)
    logger.info("Config loaded: %s", args.config)

    # ── 确定话题和模式 ──
    loc_cfg = cfg.get("localization", {})
    use_body_cloud = args.body_cloud or (
        not args.world_cloud
        and bool(loc_cfg.get("use_body_cloud",
                             loc_cfg.get("mode", "") == "gps_px4_fastlio_perception"))
    )

    if use_body_cloud:
        cloud_topic = str(loc_cfg.get("body_cloud_topic", "/cloud_registered_body"))
        px4_odom_topic = "/mavros/local_position/odom"
        fastlio_odom_topic = str(loc_cfg.get("fastlio_odom_topic", "/ali_odom"))
        logger.info("Mode: body-cloud  cloud=%s  px4_odom=%s  fastlio_odom=%s",
                     cloud_topic, px4_odom_topic, fastlio_odom_topic)
    else:
        cloud_topic = str(loc_cfg.get("world_cloud_topic", "/ali_cloud"))
        fastlio_odom_topic = str(loc_cfg.get("fastlio_odom_topic", "/ali_odom"))
        px4_odom_topic = None
        logger.info("Mode: world-cloud  cloud=%s  odom=%s",
                     cloud_topic, fastlio_odom_topic)

    # ── 检查 rosbag ──
    bag_path = Path(args.bag)
    if not bag_path.is_file():
        logger.error("Bag not found: %s", bag_path)
        sys.exit(1)

    # ── 初始化感知模块 ──
    perc_cfg = cfg["perception"]
    depth_cfg = cfg["depth_projection"]
    obs_cfg = cfg["observation"]
    uav_cfg = cfg.get("uav", {})
    runtime_cfg = cfg.get("runtime", {})

    obs_h = int(obs_cfg.get("img_height", 128))
    obs_w = int(obs_cfg.get("img_width", 128))
    depth_max = args.dmax
    safe_id = int(perc_cfg.get("safe_class_id", 1))
    danger_id = int(perc_cfg.get("danger_class_id", 9))
    max_sync_ms = float(runtime_cfg.get("max_cloud_odom_sync_ms", 100.0))
    projection_mode = str(depth_cfg.get("mode", "training_camera")).lower()
    sim_dt = float(uav_cfg.get("sim_dt", 0.25))

    # ROI 参数
    roi_half_x = float(perc_cfg.get("halss_roi_half_x_m", 5.0))
    roi_half_y = float(perc_cfg.get("halss_roi_half_y_m", 5.0))
    roi_dynamic = bool(perc_cfg.get("halss_roi_dynamic_enabled", True))
    roi_fov_half_rad = math.radians(float(perc_cfg.get("halss_roi_fov_half_deg", 45.0)))
    roi_min_half = float(perc_cfg.get("halss_roi_min_half_m", 0.5))
    roi_max_half = float(perc_cfg.get("halss_roi_max_half_m", 15.0))
    roi_height_src = str(perc_cfg.get("halss_roi_height_source", "pose_z"))
    halss_ray_sampling = bool(perc_cfg.get("halss_pinhole_ray_sampling_enabled", False))
    halss_ray_grid = int(perc_cfg.get("halss_pinhole_ray_grid_res", 64))

    # 延时导入感知模块 (与 pipeline.py _import_runtime_deps 一致)
    from perception.halss_bayesian import HALSSBayesianEvaluator
    from perception.halss_preprocess import (
        world_to_level_body_roi,
        body_cloud_to_level_body_roi,
    )
    from perception.semantic_generator import SemanticGenerator
    from perception.training_camera_projection import (
        TrainingCameraModel,
        project_training_camera,
        sample_nearest_points_by_camera_rays,
    )
    from control.action_decomposer import ActionDecomposer

    # ── 初始化各模块 ──
    logger.info("[Init] HALSS Bayesian evaluator...")
    halss = HALSSBayesianEvaluator(perc_cfg)

    logger.info("[Init] Semantic generator...")
    sem_gen = SemanticGenerator({**perc_cfg, "img_width": obs_w, "img_height": obs_h})

    logger.info("[Init] Training camera model...")
    training_camera = TrainingCameraModel.from_config(
        depth_cfg.get("training_camera", {}),
        output_width=obs_w, output_height=obs_h, far_m=depth_max,
    )
    logger.info("  FOV=%.1fx%.1fdeg scaled_fx=%.3f fy=%.3f",
                training_camera.horizontal_fov_deg, training_camera.vertical_fov_deg,
                training_camera.fx, training_camera.fy)

    logger.info("[Init] ONNX DRL...")
    drl = ONNXDRL(
        args.onnx_model,
        obs_h=obs_h, obs_w=obs_w, dmax=depth_max,
        depth_norm_mode=str(obs_cfg.get("depth_norm_mode", "raw_meters_graph_scaled")),
        semantic_norm_mode=str(obs_cfg.get("semantic_norm_mode", "raw_gray_graph_scaled")),
    )

    logger.info("[Init] Action decomposer...")
    decomposer = ActionDecomposer(uav_cfg)
    action_names = decomposer.action_names
    logger.info("  %d actions, frame=%s sign=%d",
                len(action_names), decomposer.action_frame, decomposer.action_lateral_sign)

    # ── 可视化 ──
    vis_cfg = cfg.get("visualization", {})
    display = None if args.no_display else ReplayVisualizer(
        dmax=depth_max,
        depth_vmax_m=float(vis_cfg.get("depth_vmax_m", depth_max)),
        display_width=int(vis_cfg.get("display_width", 300)),
    )

    # ──────────────────────────────────────────
    # 开始读取 rosbag
    # ──────────────────────────────────────────
    import rosbag
    from sensor_msgs.msg import PointCloud2
    from nav_msgs.msg import Odometry

    logger.info("[Bag] Opening %s ...", bag_path)
    bag = rosbag.Bag(str(bag_path), "r")

    # 统计 bag 中的话题
    topics_info = bag.get_type_and_topic_info()
    logger.info("[Bag] Topics in bag: %s",
                ", ".join(sorted(topics_info.topics.keys())))

    # ── 扫描 bag 获取 cloud + odom 消息 ──
    # 策略: 遍历 bag, 将 cloud 消息放入队列; 同时维护 odom 滑动窗口,
    # 对每个 cloud 找到最近的 odom 并处理.
    # 这样可以避免将整个 bag 读入内存, 同时保证帧顺序.
    import rosbag  # noqa
    from sensor_msgs.msg import PointCloud2  # noqa
    from nav_msgs.msg import Odometry  # noqa

    # body-cloud 需要 PX4 odom 获取 roll/pitch
    if use_body_cloud:
        primary_cloud_topic = cloud_topic
        primary_odom_topic = px4_odom_topic
        secondary_odom_topic = fastlio_odom_topic
        read_topics = [primary_cloud_topic, primary_odom_topic, secondary_odom_topic]
    else:
        primary_cloud_topic = cloud_topic
        primary_odom_topic = fastlio_odom_topic
        read_topics = [primary_cloud_topic, primary_odom_topic]

    # 验证 bag 中是否包含必要话题
    available = set(topics_info.topics.keys())
    for t in read_topics:
        if t not in available:
            logger.warning("[Bag] Topic %s not found in bag", t)

    logger.info("[Bag] Reading topics: %s", read_topics)

    # ── 状态变量 ──
    frame_count = 0
    action_counts = [0] * 10
    ground_z_world = args.ground_z
    last_cloud_stamp = None
    last_cloud_pts = None
    last_odom_stamp = None
    last_odom_msg = None
    last_fastlio_stamp = None
    last_fastlio_pose = None

    # PX4 odom 滑动窗口 (body-cloud 模式): 存储 (stamp, msg) 元组
    px4_odom_window = deque(maxlen=50)

    # 用于统计
    start_wall = time.perf_counter()
    total_processed = 0

    try:
        for topic_name, msg, ros_stamp in bag.read_messages(topics=read_topics):
            stamp_sec = _stamp_to_sec(msg.header.stamp) if hasattr(msg, 'header') else _stamp_to_sec(ros_stamp)

            # ── 如果是 PX4 odom (body-cloud 模式), 放入滑动窗口 ──
            if use_body_cloud and topic_name == px4_odom_topic:
                px4_odom_window.append((stamp_sec, msg))
                continue

            # ── 如果是 FAST-LIO odom (world-cloud 模式或诊断用) ──
            if topic_name == fastlio_odom_topic:
                if use_body_cloud:
                    # 仅诊断记录, 不用于感知
                    p = msg.pose.pose.position
                    q = msg.pose.pose.orientation
                    r, pch, y = _quat_to_euler(q.x, q.y, q.z, q.w)
                    last_fastlio_pose = np.array([p.x, p.y, p.z, r, pch, y], dtype=np.float32)
                    last_fastlio_stamp = stamp_sec
                else:
                    # world-cloud 模式: 直接作为位姿源
                    p = msg.pose.pose.position
                    q = msg.pose.pose.orientation
                    r, pch, y = _quat_to_euler(q.x, q.y, q.z, q.w)
                    last_odom_msg = np.array([p.x, p.y, p.z, r, pch, y], dtype=np.float32)
                    last_odom_stamp = stamp_sec
                    last_odom_raw = msg
                continue

            # ── 如果是点云 ──
            if topic_name == primary_cloud_topic:
                cloud_pts = _pc2_to_numpy(msg)
                cloud_stamp = stamp_sec

                if len(cloud_pts) == 0:
                    logger.warning("[Frame] Empty cloud at t=%.3f, skipping", cloud_stamp)
                    continue

                # 获取匹配的位姿
                if use_body_cloud:
                    # 从 PX4 odom 窗口找到最近的
                    if len(px4_odom_window) == 0:
                        logger.warning("[Frame] No PX4 odom available at t=%.3f", cloud_stamp)
                        continue
                    # 找时间最近的 PX4 odom
                    odom_times = np.array([t for t, _ in px4_odom_window])
                    nearest_idx = np.argmin(np.abs(odom_times - cloud_stamp))
                    nearest_stamp = float(odom_times[nearest_idx])
                    sync_ms = abs(nearest_stamp - cloud_stamp) * 1000.0
                    if sync_ms > max_sync_ms:
                        logger.warning("[Frame] Sync %.0fms > %.0fms at t=%.3f, skip",
                                       sync_ms, max_sync_ms, cloud_stamp)
                        continue
                    _, px4_msg = px4_odom_window[nearest_idx]
                    q = px4_msg.pose.pose.orientation
                    roll, pitch, _ = _quat_to_euler(q.x, q.y, q.z, q.w)
                    # 获取 PX4 位置和 yaw (用于动作解算)
                    p = px4_msg.pose.pose.position
                    px4_pos = np.array([p.x, p.y, p.z], dtype=np.float32)
                    _, _, px4_yaw = _quat_to_euler(q.x, q.y, q.z, q.w)

                    # 组合 frame_pose: [x, y, z, roll, pitch, yaw] (ENU yaw)
                    frame_pose = np.array([
                        px4_pos[0], px4_pos[1], px4_pos[2],
                        roll, pitch, px4_yaw,
                    ], dtype=np.float32)

                    # body_cloud_to_level_body_roi: 只需要 roll, pitch
                    try:
                        projection_pts, halss_stats = body_cloud_to_level_body_roi(
                            cloud_pts, roll, pitch, perc_cfg,
                            half_x=roi_half_x, half_y=roi_half_y,
                        )
                    except Exception as e:
                        logger.warning("[Frame] body_cloud_to_level_body_roi error: %s", e)
                        continue

                    # Pinhole ray sampling (如果启用)
                    if halss_ray_sampling:
                        halss_pts, ray_stats = sample_nearest_points_by_camera_rays(
                            projection_pts, training_camera,
                            ray_width=halss_ray_grid, ray_height=halss_ray_grid,
                        )
                        halss_stats["pre_ray_points"] = int(len(projection_pts))
                        halss_stats["frustum_points"] = int(ray_stats["frustum_points"])
                        halss_stats["ray_grid"] = [halss_ray_grid, halss_ray_grid]
                        halss_stats["output_points"] = int(len(halss_pts))
                    else:
                        halss_pts = projection_pts
                        halss_stats["output_points"] = int(len(halss_pts))

                else:
                    # world-cloud 模式
                    if last_odom_msg is None:
                        logger.warning("[Frame] No odom available at t=%.3f", cloud_stamp)
                        continue
                    sync_ms = abs(last_odom_stamp - cloud_stamp) * 1000.0
                    if sync_ms > max_sync_ms:
                        logger.warning("[Frame] Sync %.0fms > %.0fms at t=%.3f, skip",
                                       sync_ms, max_sync_ms, cloud_stamp)
                        continue
                    frame_pose = last_odom_msg.copy()

                    try:
                        halss_pts, halss_stats = world_to_level_body_roi(
                            cloud_pts, frame_pose[:3], frame_pose[3:],
                            perc_cfg, half_x=roi_half_x, half_y=roi_half_y,
                        )
                    except Exception as e:
                        logger.warning("[Frame] world_to_level_body_roi error: %s", e)
                        continue
                    projection_pts = halss_pts

                # ── 检查 ROI 点数 ──
                if halss_stats.get("output_points", 0) < 10:
                    logger.debug("[Frame] Sparse ROI: %d points, skip",
                                 halss_stats.get("output_points", 0))
                    continue

                # ── 记录地面 Z (第一帧) ──
                if ground_z_world is None:
                    ground_z_world = float(frame_pose[2])
                    logger.info("[Ground] First frame pose_z=%.2f set as ground_z", ground_z_world)

                # ── 动态 FOV ROI ──
                if roi_dynamic:
                    H = abs(float(frame_pose[2]) - ground_z_world)
                    H = max(H, 0.1)
                    if projection_mode == "training_camera":
                        cur_half_x, cur_half_y = training_camera.ground_half_extents(H)
                        cur_half_x = max(roi_min_half, min(roi_max_half, cur_half_x))
                        cur_half_y = max(roi_min_half, min(roi_max_half, cur_half_y))
                    else:
                        half = H * math.tan(roi_fov_half_rad)
                        cur_half_x = cur_half_y = max(roi_min_half, min(roi_max_half, half))
                else:
                    cur_half_x, cur_half_y = roi_half_x, roi_half_y

                # ── ROI bounds ──
                roi_bounds = {
                    "x_min": -cur_half_x, "x_max": cur_half_x,
                    "y_min": -cur_half_y, "y_max": cur_half_y,
                }

                # ── 1. HALSS 语义 ──
                t0 = time.perf_counter()
                try:
                    halss_result = halss.evaluate(halss_pts, fixed_bounds=roi_bounds)
                except Exception as e:
                    logger.warning("[Frame] HALSS error: %s", e)
                    halss_result = None

                if halss_result is not None:
                    bev_data = halss_result.get("bev_data", halss_result)
                    sem_map = sem_gen.generate(bev_data)
                else:
                    sem_map = np.full((obs_h, obs_w), danger_id, dtype=np.uint8)
                t_halss = time.perf_counter() - t0

                # ── 2. 深度投影 ──
                t1 = time.perf_counter()
                semantic_valid_mask = np.ones_like(sem_map, dtype=bool)
                if projection_mode == "training_camera":
                    sparse_depth, valid_mask, sem_map, semantic_valid_mask = \
                        project_training_camera(
                            projection_pts, sem_map, roi_bounds, training_camera,
                            danger_id=danger_id,
                        )
                    binary_semantic_vis = make_binary_semantic_vis(
                        sem_map, safe_id=safe_id, danger_id=danger_id,
                    )
                    binary_semantic_vis[~semantic_valid_mask] = 128
                else:
                    sparse_depth, _ = project_bev_depth(
                        halss_pts,
                        grid_res=int(perc_cfg.get("halss_grid_res", 64)),
                        out_size=obs_w, max_range=depth_max,
                        half_x=cur_half_x, half_y=cur_half_y,
                    )
                    valid_mask = (sparse_depth < depth_max) & (sparse_depth > 0.01)
                    binary_semantic_vis = make_binary_semantic_vis(
                        sem_map, safe_id=safe_id, danger_id=danger_id,
                    )
                t_depth = time.perf_counter() - t1

                # ── 3. NN-fill 深度渲染 ──
                t2 = time.perf_counter()
                rendered_depth = render_sparse_depth(sparse_depth, valid_mask, depth_max)
                t_completion = time.perf_counter() - t2

                # ── 4. ONNX DRL 推理 ──
                t3 = time.perf_counter()
                action_id, rl_info = drl.predict(rendered_depth, sem_map)
                rl_info["semantic_valid_ratio"] = float(np.mean(semantic_valid_mask))
                t_drl = time.perf_counter() - t3

                # ── 5. 动作解算 ──
                action_name = decomposer.action_id_to_name(action_id)
                execution_yaw = float(frame_pose[5])  # ENU yaw for logging
                v_body, v_ned, action_yaw_rate = decomposer.decompose(
                    action_id, execution_yaw,
                )
                action_counts[action_id] += 1

                total_ms = (time.perf_counter() - t0) * 1000
                frame_count += 1
                total_processed += 1

                # ── 打印每帧结果 ──
                safe_ratio = float(np.mean(sem_map == safe_id))
                danger_ratio = float(np.mean(sem_map == danger_id))
                conf = rl_info.get("confidence", 0.0)
                top3_str = _top_probs(rl_info.get("action_probs"), action_names, k=3)

                print(
                    f"\r[{frame_count:04d}] ACT={action_id}({action_name})  "
                    f"pts={halss_stats.get('output_points', 0):4d}  "
                    f"sem_safe={safe_ratio:.2f} sem_danger={danger_ratio:.2f}  "
                    f"depth={float(np.mean(rendered_depth)):.1f}m  "
                    f"conf={conf:.2f}  {top3_str}  "
                    f"lat={total_ms:.0f}ms  "
                    f"(H={t_halss*1000:.0f} D={t_depth*1000:.0f} C={t_completion*1000:.0f} RL={t_drl*1000:.0f})  "
                    f"v_ned=({v_ned[0]:+.2f},{v_ned[1]:+.2f},{v_ned[2]:+.2f})  "
                    f"yaw_rate={action_yaw_rate:+.2f}",
                    end="", flush=True,
                )

                # ── 日志 (每 5 帧) ──
                if frame_count % 5 == 0:
                    logger.info(
                        "[%04d] act=%d(%s) pts=%d sem_safe=%.2f sem_danger=%.2f "
                        "depth_mean=%.1fm conf=%.2f %s lat=%.0fms "
                        "v_ned=(%.2f,%.2f,%.2f) yr=%.2f",
                        frame_count, action_id, action_name,
                        halss_stats.get('output_points', 0),
                        safe_ratio, danger_ratio,
                        float(np.mean(rendered_depth)), conf, top3_str,
                        total_ms,
                        v_ned[0], v_ned[1], v_ned[2], action_yaw_rate,
                    )
                    logger.info(
                        "  depth_in=[%.1f,%.1f,%.1f] sem_in=[%.0f,%.0f,%.0f] "
                        "obs_raw=[%.1f,%.1f]",
                        rl_info["depth_input_min"], rl_info["depth_input_mean"],
                        rl_info["depth_input_max"],
                        rl_info["sem_input_min"], rl_info["sem_input_mean"],
                        rl_info["sem_input_max"],
                        rl_info["obs_raw_min"], rl_info["obs_raw_max"],
                    )

                # ── 可视化 ──
                if display is not None:
                    display.update(rendered_depth, sem_map, binary_semantic_vis)

                # ── 速率控制 ──
                if args.rate > 0:
                    elapsed = time.perf_counter() - t0
                    sleep = max(0.0, (1.0 / args.rate) - elapsed)
                    if sleep > 0:
                        time.sleep(sleep)

                # ── 最大帧数限制 ──
                if args.max_frames > 0 and frame_count >= args.max_frames:
                    logger.info("[Replay] Max frames reached: %d", args.max_frames)
                    break

    except KeyboardInterrupt:
        logger.info("[Replay] Interrupted by user.")
    finally:
        bag.close()
        elapsed = time.perf_counter() - start_wall
        logger.info("=" * 60)
        logger.info("Replay summary:")
        logger.info("  Total frames: %d", total_processed)
        logger.info("  Elapsed: %.1fs", elapsed)
        if total_processed > 0:
            logger.info("  Avg fps: %.1f", total_processed / max(elapsed, 0.01))
        total_acts = sum(action_counts)
        if total_acts > 0:
            logger.info("  Action distribution:")
            for i, cnt in enumerate(action_counts):
                if cnt > 0:
                    pct = 100.0 * cnt / total_acts
                    logger.info("    %s: %d (%.1f%%)", action_names[i], cnt, pct)
        logger.info("=" * 60)

        if display is not None:
            display.close()


if __name__ == "__main__":
    main()
