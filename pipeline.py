#!/usr/bin/env python3
"""
Orin Landing - 真机实时感知-决策-控制主管线
==========================================

链路:
  GIS 九宫格全局安全区 → MAVROS/PX4 位置引导
  Mid360 → FAST-LIO frontend IMU 去畸变点云（室外不运行 SLAM 后端）
  去畸变点云 → Mid360 安装外参补偿 → PX4 重力对齐 → 固定针孔射线采样
  点云 → HALSS Bayesian 二值安全语义图
  点云 → 原训练下视相机几何稀疏深度 → NN-fill 渲染深度
  [rendered_depth, binary_semantic] → ONNX PPO → 离散动作
  离散动作 + 同帧 Fast-LIO yaw → NED 速度 + yaw setpoint → PX4
  python pipeline.py --config ./config/experiment_outdoor_gps.yaml --mode ros --allow-high-yaw-rate-test
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import math
import os
import sys
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("OrinLanding")

from diagnostics.flight_ready import flight_ready_failures

np = None
cv2 = None
ort = None
_RUNTIME_DEPS_READY = False


def _import_runtime_deps():
    """Import CUDA/ROS/model dependencies only after strict config gates pass."""
    global np, cv2, ort
    global MAVROSController, PoseSourceManager, ned_to_enu_velocity
    global ActionDecomposer, ActionCollapseMonitor
    global FastLIOInterface, HALSSBayesianEvaluator, world_to_level_body_roi
    global body_cloud_to_level_body_roi
    global TrainingCameraModel, project_training_camera
    global sample_nearest_points_by_camera_rays
    global MissionStateManager, StateInputs
    global SemanticGenerator, GlobalSafetyPrior
    global RealtimeVisualizer, _RUNTIME_DEPS_READY
    global MissionState

    if _RUNTIME_DEPS_READY:
        return

    import numpy as _np
    import cv2 as _cv2
    import onnxruntime as _ort
    from control.mavros_controller import MAVROSController as _MAVROSController, ned_to_enu_velocity as _ned_to_enu_velocity
    from control.pose_source_manager import PoseSourceManager as _PoseSourceManager
    from control.action_decomposer import ActionDecomposer as _ActionDecomposer
    from diagnostics.action_monitor import ActionCollapseMonitor as _ActionCollapseMonitor
    from odometry import FastLIOInterface as _FastLIOInterface
    from perception.halss_bayesian import HALSSBayesianEvaluator as _HALSSBayesianEvaluator
    from perception.halss_preprocess import (
        world_to_level_body_roi as _world_to_level_body_roi,
        body_cloud_to_level_body_roi as _body_cloud_to_level_body_roi,
    )
    from perception.training_camera_projection import (
        TrainingCameraModel as _TrainingCameraModel,
        project_training_camera as _project_training_camera,
        sample_nearest_points_by_camera_rays as _sample_nearest_points_by_camera_rays,
    )
    from control.mission_state_manager import MissionStateManager as _MissionStateManager
    from control.mission_state_manager import StateInputs as _StateInputs
    from control.mission_state_manager import MissionState as _MissionState
    from perception.semantic_generator import SemanticGenerator as _SemanticGenerator
    from preprocessing.global_safety_prior import GlobalSafetyPrior as _GlobalSafetyPrior
    from visualization import RealtimeVisualizer as _RealtimeVisualizer

    np = _np
    cv2 = _cv2
    ort = _ort
    MAVROSController = _MAVROSController
    PoseSourceManager = _PoseSourceManager
    ned_to_enu_velocity = _ned_to_enu_velocity
    ActionDecomposer = _ActionDecomposer
    ActionCollapseMonitor = _ActionCollapseMonitor
    FastLIOInterface = _FastLIOInterface
    HALSSBayesianEvaluator = _HALSSBayesianEvaluator
    world_to_level_body_roi = _world_to_level_body_roi
    body_cloud_to_level_body_roi = _body_cloud_to_level_body_roi
    TrainingCameraModel = _TrainingCameraModel
    project_training_camera = _project_training_camera
    sample_nearest_points_by_camera_rays = _sample_nearest_points_by_camera_rays
    MissionStateManager = _MissionStateManager
    StateInputs = _StateInputs
    MissionState = _MissionState
    SemanticGenerator = _SemanticGenerator
    GlobalSafetyPrior = _GlobalSafetyPrior
    RealtimeVisualizer = _RealtimeVisualizer
    _RUNTIME_DEPS_READY = True


def _fmt_vec(v: np.ndarray) -> str:
    return f"[{v[0]:.1f},{v[1]:.1f},{v[2]:.1f}]"


def _top_probs(probs, action_names=None, k: int = 3) -> str:
    if probs is None:
        return "p=n/a"
    pairs = sorted(enumerate(probs), key=lambda item: item[1], reverse=True)[:k]
    if action_names is None:
        return "p=" + ",".join(f"{idx}:{prob:.2f}" for idx, prob in pairs)
    return "p=" + ",".join(f"{idx}:{action_names[idx]}:{prob:.2f}" for idx, prob in pairs)


def _wrap_pi(angle_rad: float) -> float:
    return (angle_rad + math.pi) % (2.0 * math.pi) - math.pi


def _limited_xy_velocity(dx: float, dy: float, max_speed_mps: float,
                         kp_s: float = 1.0) -> tuple:
    """Return a target-directed XY velocity with proportional slowdown and a hard cap."""
    distance = math.hypot(float(dx), float(dy))
    if distance <= 1e-9:
        return 0.0, 0.0
    speed = min(float(max_speed_mps), float(kp_s) * distance)
    scale = speed / distance
    return float(dx) * scale, float(dy) * scale


def _limited_axis_velocity(error: float, max_speed_mps: float, kp_s: float = 1.0) -> float:
    """Proportional single-axis velocity with a symmetric hard limit."""
    return max(-float(max_speed_mps), min(float(max_speed_mps), float(kp_s) * float(error)))


CLASS_TO_GRAY = {
    -1: 0,
    0: 10,
    1: 30,
    2: 60,
    3: 70,
    4: 20,
    5: 40,
    6: 80,
    7: 90,
    8: 50,
    9: 250,
}


def project_bev_depth(pts_body, grid_res=64, out_size=128, max_range=30.0,
                     half_x=5.0, half_y=5.0):
    """Project body-frame down-looking ROI points to a sparse BEV depth map.

    Uses fixed bounds aligned with world_to_level_body_roi:
      x (forward) ∈ [-half_x, half_x] m  →  col axis
      y (lateral) ∈ [-half_y, half_y] m  →  row axis (flipped so +y maps to top)
    """
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
    row_idx = row_idx[valid]
    col_idx = col_idx[valid]
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
    """Nearest-neighbor fill sparse BEV depth, then lightly smooth holes."""
    if valid_mask.sum() < min_valid:
        return np.where(valid_mask, sparse_depth, dmax).astype(np.float32)

    invalid = ~valid_mask
    _, labels = cv2.distanceTransformWithLabels(
        invalid.astype(np.uint8),
        distanceType=cv2.DIST_L2,
        maskSize=5,
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


def make_binary_semantic_vis(sem_map, safe_id=1, danger_id=9):
    """Semantic class map -> uint8 visualization: safe white, danger black, unknown gray."""
    sem_vis = np.full(sem_map.shape, 128, dtype=np.uint8)
    sem_vis[sem_map == safe_id] = 255
    sem_vis[sem_map == danger_id] = 0
    return sem_vis


class ONNXDRL:
    """ONNX PPO policy used by the live MAVROS pipeline."""

    def __init__(self, onnx_path: str, obs_h: int = 128, obs_w: int = 128,
                 dmax: float = 30.0,
                 depth_norm_mode: str = "raw_meters_graph_scaled",
                 semantic_norm_mode: str = "raw_gray_graph_scaled"):
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
        self.obs_h = obs_h
        self.obs_w = obs_w
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
        logger.info(
            "[ONNX] model=%s input=%s shape=%s layout=%s "
            "external_encoding=(raw_depth_m,raw_gray), graph_scale=/255 warmup=OK",
            onnx_path, self.input_name, in_shape, self.layout,
        )

    def _forward(self, obs_raw):
        if self.layout == "chw":
            inp = np.transpose(obs_raw, (0, 3, 1, 2)).astype(np.float32)
        else:
            inp = obs_raw.astype(np.float32)
        return self.session.run([self.output_name], {self.input_name: inp})[0]

    def predict(self, depth_map, sem_map):
        depth_clipped = np.clip(
            np.nan_to_num(depth_map, nan=self.dmax, posinf=self.dmax, neginf=0.0),
            0.0,
            self.dmax,
        )
        # The exported graph contains SB2 policy scale=True as input/truediv.
        # Feed the same raw Box(0,255) values used by training exactly once.
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
            "sem_input_unique": [round(float(x), 6) for x in sorted(np.unique(sem_ch))],
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
        return action, info


def _parse_bounds(value):
    if value is None:
        return None
    if isinstance(value, str):
        parts = [float(x.strip()) for x in value.split(",")]
    else:
        parts = [float(x) for x in value]
    if len(parts) != 4:
        raise ValueError("GIS bounds must be lon_left,lat_bottom,lon_right,lat_top")
    return tuple(parts)


def _validate_global_guidance_override(args):
    failures = []
    if getattr(args, "safe_point", None):
        try:
            lat, lon = _parse_safe_point(args.safe_point)
            if not (-90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0):
                raise ValueError("latitude/longitude outside valid ranges")
        except ValueError as exc:
            failures.append(f"flight-ready gate: invalid --safe-point ({exc})")
        if getattr(args, "safe_point_source", "manual") != "gis":
            failures.append(
                "flight-ready gate: --safe-point requires --safe-point-source gis "
                "to prove it came from offline GIS nine-grid risk assessment"
            )
        return not failures, failures

    if getattr(args, "gis_image", None) or getattr(args, "gis_bounds", None):
        image_path = getattr(args, "gis_image", None)
        bounds = getattr(args, "gis_bounds", None)
        if not image_path:
            failures.append("flight-ready gate: --gis-bounds requires --gis-image")
        elif not Path(image_path).is_file():
            failures.append(f"flight-ready gate: --gis-image missing: {image_path}")
        if not bounds:
            failures.append("flight-ready gate: --gis-image requires --gis-bounds")
        else:
            try:
                _parse_bounds(bounds)
            except ValueError as exc:
                failures.append(f"flight-ready gate: invalid --gis-bounds ({exc})")
        mask_path = getattr(args, "gis_mask", None)
        if mask_path and not Path(mask_path).is_file():
            failures.append(f"flight-ready gate: --gis-mask missing: {mask_path}")
        return not failures, failures

    return None, failures


def _config_has_global_guidance(cfg, args) -> bool:
    override_ready, _ = _validate_global_guidance_override(args)
    if override_ready is not None:
        return override_ready
    gp = cfg.get("global_prior", {})
    if (
        gp.get("target_lat") is not None
        and gp.get("target_lon") is not None
        and gp.get("target_source") == "gis"
    ):
        return True
    return bool(gp.get("enabled", False) and gp.get("image_path") and gp.get("bounds"))


def _parse_safe_point(value: str):
    try:
        lat_str, lon_str = value.split(",")
        return float(lat_str.strip()), float(lon_str.strip())
    except ValueError as exc:
        raise ValueError("expected lat,lon") from exc


def _load_config(path: str) -> dict:
    config_path = Path(path).resolve()
    try:
        import yaml
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
    except ModuleNotFoundError as exc:
        if exc.name != "yaml":
            raise
        from preflight_check import _load_simple_yaml
        cfg = _load_simple_yaml(config_path)

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


def _build_config_overrides(args) -> dict:
    overrides = {}
    if getattr(args, "yaw_rate_rad_s", None) is not None:
        overrides.setdefault("uav", {})["yaw_rate_rad_s"] = float(args.yaw_rate_rad_s)
    if getattr(args, "allow_high_yaw_rate_test", False):
        overrides.setdefault("uav", {})["high_yaw_rate_test_enabled"] = True
    if getattr(args, "dmax", None) is not None:
        overrides.setdefault("depth_projection", {})["max_range"] = float(args.dmax)
    if getattr(args, "onnx_model", None):
        overrides.setdefault("decision", {})["onnx_model_path"] = args.onnx_model
    return overrides


class OrinLandingPipeline:
    """Orin 真机着陆主管线。"""

    def __init__(
        self,
        config_path: str,
        config_overrides: dict = None,
        onnx_model_path: str = None,
    ):
        _import_runtime_deps()
        self.cfg = _load_config(config_path)
        if config_overrides:
            self.cfg = _merge_config_overrides(self.cfg, config_overrides)

        self.config_path = config_path
        self._rospy = None
        self._ros_node = None

        # --- 飞控后端抽象 ---
        fc_cfg = self.cfg.get("flight_controller", {})
        fc_backend = str(fc_cfg.get("backend", "mavros")).lower()
        if fc_backend != "mavros":
            raise ValueError("The live pipeline supports flight_controller.backend=mavros only")
        self._fc_backend = "mavros"
        self._mavros_ns = str(fc_cfg.get("mavros_ns", "/mavros"))
        self._setpoint_rate_hz = float(fc_cfg.get("setpoint_rate_hz", 20))
        self._offboard_warmup_s = float(fc_cfg.get("offboard_warmup_s", 2.0))

        obs_cfg = self.cfg["observation"]
        perc_cfg = self.cfg["perception"]
        depth_cfg = self.cfg["depth_projection"]
        uav_cfg = self.cfg["uav"]
        drl_cfg = self.cfg.get("decision", {})

        self.depth_max = float(depth_cfg.get("max_range", 30.0))
        self.obs_h = int(obs_cfg.get("img_height", 128))
        self.obs_w = int(obs_cfg.get("img_width", 128))
        self.safe_id = int(perc_cfg.get("safe_class_id", 1))
        self.danger_id = int(perc_cfg.get("danger_class_id", 9))
        self.sim_dt = float(uav_cfg.get("sim_dt", 0.25))
        self.max_steps = int(float(uav_cfg.get("max_t", 90.0)) / self.sim_dt)
        self.yaw_rate_cmd = float(uav_cfg.get("yaw_rate_rad_s", 0.0))
        self.max_roll_pitch_rad = math.radians(float(uav_cfg.get("max_roll_pitch_deg", 30.0)))
        self.execution_yaw_source = str(
            uav_cfg.get("execution_yaw_source", "px4_ekf")
        ).lower()
        if self.execution_yaw_source not in ("px4_ekf", "fastlio"):
            raise ValueError("uav.execution_yaw_source must be px4_ekf or fastlio")
        runtime_cfg = self.cfg.get("runtime", {})
        self.max_inference_result_age_ms = float(
            runtime_cfg.get("max_inference_result_age_ms", 500.0)
        )
        self.drop_stale_frames = bool(runtime_cfg.get("drop_stale_frames", True))
        self.drop_slow_frames = bool(runtime_cfg.get("drop_slow_frames", True))
        self.max_cloud_odom_sync_ms = float(runtime_cfg.get("max_cloud_odom_sync_ms", 100.0))
        localization_cfg = self.cfg.get("localization", {})
        self.localization_mode = str(
            localization_cfg.get("mode", "gps_px4_fastlio_perception")
        ).lower()
        self.px4_position_source = str(
            localization_cfg.get("px4_position_source", "gps")
        ).lower()
        self.external_pose_topic = str(
            localization_cfg.get("external_pose_topic", "/mavros/vision_pose/pose")
        )
        self.fastlio_odom_topic = localization_cfg.get("fastlio_odom_topic", "/Odometry")
        self.fastlio_pose_required = bool(
            localization_cfg.get("fastlio_pose_required", True)
        )
        self._use_body_cloud = bool(
            localization_cfg.get(
                "use_body_cloud",
                self.localization_mode == "gps_px4_fastlio_perception",
            )
        )
        self.fastlio_cloud_topic = (
            localization_cfg.get("body_cloud_topic", "/cloud_registered_body")
            if self._use_body_cloud
            else localization_cfg.get("world_cloud_topic", "/cloud_registered")
        )
        self._home_enu = np.zeros(3, dtype=np.float32)  # ENU home (takeoff point)
        self._takeoff_altitude_m = 0.0  # set during run()
        self._pre_goto_yaw_enu_deg = 0.0

        logger.info("=" * 60)
        logger.info(" Initializing Orin Landing Pipeline...")
        logger.info("=" * 60)

        logger.info("[Init] HALSS Bayesian evaluator...")
        self.halss = HALSSBayesianEvaluator(perc_cfg)

        logger.info("[Init] Semantic generator...")
        self.sem_gen = SemanticGenerator(
            {**perc_cfg, "img_width": self.obs_w, "img_height": self.obs_h}
        )

        self.onnx_model_path = (
            onnx_model_path
            or drl_cfg.get("onnx_model_path")
            or "weights/ppo2_policy.onnx"
        )
        logger.info("[Init] ONNX DRL policy...")
        self.drl = ONNXDRL(
            self.onnx_model_path,
            obs_h=self.obs_h,
            obs_w=self.obs_w,
            dmax=self.depth_max,
            depth_norm_mode=str(obs_cfg.get("depth_norm_mode", "raw_meters_graph_scaled")),
            semantic_norm_mode=str(obs_cfg.get("semantic_norm_mode", "raw_gray_graph_scaled")),
        )
        logger.info(
            "[Init] Perception route: HALSS + training-camera NN-fill depth + ONNX DRL "
            "(dmax=%.1fm, obs=%dx%d)",
            self.depth_max,
            self.obs_w,
            self.obs_h,
        )

        logger.info("[Init] Action decomposer...")
        self.decomposer = ActionDecomposer(uav_cfg)
        logger.info(
            "[Init] Action mapping frame=%s lateral_sign=%d (act3=%s)",
            self.decomposer.action_frame,
            self.decomposer.action_lateral_sign,
            self.decomposer.action_id_to_name(3),
        )
        logger.info("[Init] Body-action execution yaw source=%s", self.execution_yaw_source)
        logger.info(
            "[Init] Localization mode=%s px4_position_source=%s global_prior=%s "
            "yaw_rate=%.3frad/s setpoint_rate=%.1fHz",
            self.localization_mode,
            self.px4_position_source,
            self.cfg.get("global_prior", {}).get("mode", "gps"),
            self.yaw_rate_cmd,
            self._setpoint_rate_hz,
        )

        logger.info(
            "[Init] Fast-LIO interface (cloud=%s body_mode=%s pose_required=%s)...",
            self.fastlio_cloud_topic,
            self._use_body_cloud,
            self.fastlio_pose_required,
        )
        self.fastlio = FastLIOInterface(use_ros=True)

        logger.info("[Init] Pose source manager...")
        self.pose_source_mgr = PoseSourceManager(self.cfg)

        logger.info("[Init] Visualizer...")
        self.visualizer = RealtimeVisualizer(self.cfg["visualization"])
        self.action_monitor = ActionCollapseMonitor(self.cfg.get("visualization", {}), logger)

        self.fc = None  # MAVROS flight-controller adapter
        self.step_count = 0

        self._safe_lat = None
        self._safe_lon = None
        self._safe_ned = None
        self._safe_ned_target = None  # indoor local body-offset NED target (3D)
        self._home_ned = np.zeros(3, dtype=np.float32)
        self._home_lat = None
        self._home_lon = None
        self._direct_land_enu_xy = None
        self._horizontal_hold_enu_xy = None
        self._goto_tolerance_xy = float(self.cfg.get("global_prior", {}).get("goto_tolerance_xy_m", 0.2))
        self._goto_tolerance_z = float(self.cfg.get("global_prior", {}).get("goto_tolerance_z_m", 0.15))
        self._goto_max_time_s = float(self.cfg.get("global_prior", {}).get("goto_max_time_s", 30.0))
        self._goto_timeout_action = str(self.cfg.get("global_prior", {}).get("goto_timeout_action", "direct_land")).lower()
        goto_speed = self.cfg.get("global_prior", {}).get("goto_max_horizontal_speed_mps")
        self._goto_max_horizontal_speed_mps = (
            None
            if goto_speed is None or self.localization_mode != "gps_px4_fastlio_perception"
            else float(goto_speed)
        )
        self._goto_horizontal_kp_s = float(
            self.cfg.get("global_prior", {}).get("goto_horizontal_kp_s", 1.0)
        )
        self._goto_max_vertical_speed_mps = float(
            self.cfg.get("global_prior", {}).get("goto_max_vertical_speed_mps", 1.0)
        )
        if (
            self._goto_max_horizontal_speed_mps is not None
            and self._goto_max_horizontal_speed_mps <= 0.0
        ):
            raise ValueError("global_prior.goto_max_horizontal_speed_mps must be > 0")
        if self._goto_horizontal_kp_s <= 0.0:
            raise ValueError("global_prior.goto_horizontal_kp_s must be > 0")
        if self._goto_max_vertical_speed_mps <= 0.0:
            raise ValueError("global_prior.goto_max_vertical_speed_mps must be > 0")
        self._use_global_guidance = False
        self._indoor_local_mode = False
        self._local_body_offset_m = None
        self._safe_altitude_m = None  # GPS mode target altitude
        self._configure_global_prior_from_config()
        mission_cfg = dict(self.cfg.get("mission_state", {}))
        # global_prior.mode is the single public source of truth. The state
        # manager keeps its legacy internal key for compatibility.
        mission_cfg["global_prior_mode"] = str(
            self.cfg.get("global_prior", {}).get("mode", "gps")
        ).lower()
        mission_cfg.setdefault("max_cloud_odom_sync_ms", self.max_cloud_odom_sync_ms)
        self.state_manager = MissionStateManager(mission_cfg)
        self.mission_state = self.state_manager.state.value

        self._timing = {"halss": [], "depth": [], "completion": [], "rl": [], "control": [], "total": []}

        # --- ROI bounds (shared by depth projection and HALSS semantic grid) ---
        perc_cfg_roi = self.cfg.get("perception", {})
        self._roi_half_x = float(perc_cfg_roi.get("halss_roi_half_x_m", 5.0))
        self._roi_half_y = float(perc_cfg_roi.get("halss_roi_half_y_m", 5.0))
        self._roi_dynamic = bool(perc_cfg_roi.get("halss_roi_dynamic_enabled", True))
        self._roi_fov_half_rad = math.radians(float(perc_cfg_roi.get("halss_roi_fov_half_deg", 45.0)))
        self._roi_min_half_m = float(perc_cfg_roi.get("halss_roi_min_half_m", 0.5))
        self._roi_max_half_m = float(perc_cfg_roi.get("halss_roi_max_half_m", 15.0))
        self._roi_height_source = str(perc_cfg_roi.get("halss_roi_height_source", "pose_z"))
        self._ground_z_world = 0.0  # FAST-LIO world z indoors; PX4 launch ENU z outdoors
        self._projection_mode = str(depth_cfg.get("mode", "training_camera")).lower()
        self._training_camera = TrainingCameraModel.from_config(
            depth_cfg.get("training_camera", {}),
            output_width=self.obs_w,
            output_height=self.obs_h,
            far_m=self.depth_max,
        )
        self._halss_pinhole_ray_sampling = bool(
            perc_cfg_roi.get("halss_pinhole_ray_sampling_enabled", False)
        )
        self._halss_ray_grid_res = int(
            perc_cfg_roi.get("halss_pinhole_ray_grid_res", 64)
        )
        if self._halss_ray_grid_res <= 0:
            raise ValueError("perception.halss_pinhole_ray_grid_res must be positive")
        if self._projection_mode not in ("training_camera", "level_body_bev"):
            raise ValueError("depth_projection.mode must be training_camera or level_body_bev")
        logger.info(
            "[Init] Projection=%s FOV=%.1fx%.1fdeg scaled_intrinsics="
            "fx=%.3f fy=%.3f cx=%.3f cy=%.3f",
            self._projection_mode,
            self._training_camera.horizontal_fov_deg,
            self._training_camera.vertical_fov_deg,
            self._training_camera.fx,
            self._training_camera.fy,
            self._training_camera.cx,
            self._training_camera.cy,
        )
        if self._use_body_cloud:
            logger.info(
                "[Init] HALSS body-cloud sampling=%s ray_grid=%dx%d max_points=%d",
                "pinhole_nearest" if self._halss_pinhole_ray_sampling else "disabled",
                self._halss_ray_grid_res,
                self._halss_ray_grid_res,
                self._halss_ray_grid_res ** 2,
            )

        # --- velocity command CSV logger ---
        self._vel_log_path = None
        self._vel_log_file = None
        self._vel_log_writer = None
        self._init_velocity_log()

        # --- DRL action log CSV ---
        self._drl_log_path = None
        self._drl_log_file = None
        self._drl_log_writer = None

        # --- mission timeline + per-frame perception timing CSV ---
        self._event_log_path = None
        self._event_log_file = None
        self._event_log_writer = None
        self._recorded_mission_events = set()
        self._frame_timing_path = None
        self._frame_timing_file = None
        self._frame_timing_writer = None
        self._perception_gate_path = None
        self._perception_gate_file = None
        self._perception_gate_writer = None

        # --- rosbag recording ---
        self._record_bag = False
        self._bag_process = None
        self._run_dir = None  # experiments/runs/YYYYMMDD_HHMMSS_orin_landing/
        self._config_overrides = config_overrides or {}

        # Initialize run directory and log files
        self._setup_run_dir()

        logger.info("=" * 60)
        logger.info(" Pipeline initialized. Ready.")
        logger.info("=" * 60)

    # ------------------------------------------------------------------
    # Global prior
    # ------------------------------------------------------------------

    def _configure_global_prior_from_config(self):
        cfg = self.cfg.get("global_prior", {})
        prior_mode = str(cfg.get("mode", "gps")).lower()

        # --- Indoor local body offset mode (3D) ---
        if prior_mode == "local_body_offset":
            offset = cfg.get("local_body_offset_m", [0.0, 0.0, 2.0])
            if offset is None or len(offset) < 3:
                logger.warning("[GlobalPrior] local_body_offset_m must be 3D [forward, right, up]; "
                               "falling back to [0, 0, 2.0]")
                offset = [0.0, 0.0, 2.0]
            self._local_body_offset_m = (
                float(offset[0]), float(offset[1]), float(offset[2])
            )
            self._indoor_local_mode = True
            self._use_global_guidance = True
            logger.info(
                "[GlobalPrior] Indoor 3D body-offset mode: forward=%.2fm right=%.2fm up=%.2fm",
                self._local_body_offset_m[0],
                self._local_body_offset_m[1],
                self._local_body_offset_m[2],
            )
            return

        target_lat = cfg.get("target_lat")
        target_lon = cfg.get("target_lon")
        if target_lat is not None and target_lon is not None:
            self.set_safe_point(float(target_lat), float(target_lon), source="config")
            self._safe_altitude_m = float(cfg.get("target_altitude_m", 2.0))
            logger.info("[GlobalPrior] GPS target altitude: %.2fm", self._safe_altitude_m)
            return

        if not cfg.get("enabled", False):
            return

        image_path = cfg.get("image_path") or cfg.get("satellite_image_path")
        sem_mask_path = cfg.get("sem_mask_path") or cfg.get("mask_path")
        bounds = _parse_bounds(cfg.get("bounds"))
        if image_path is None:
            logger.warning("[GlobalPrior] enabled=true but no image_path configured.")
            return
        self.configure_global_prior_from_gis(image_path, sem_mask_path, bounds)

    def configure_global_prior_from_gis(self, image_path: str, sem_mask_path: str = None,
                                        bounds=None):
        bounds = _parse_bounds(bounds)
        gsp_cfg = dict(self.cfg.get("global_prior", {}))
        if sem_mask_path:
            gsp_cfg.pop("model_arch", None)
            gsp_cfg.pop("model_path", None)
        gsp = GlobalSafetyPrior(gsp_cfg)
        result = gsp.assess_from_file(image_path, sem_mask_path, bounds=bounds)
        save_dir = self.cfg.get("global_prior", {}).get("save_dir")
        if save_dir:
            gsp.save_results(result, save_dir)
        gps = result.get("best_center_gps")
        if gps is None:
            logger.warning("[GlobalPrior] GIS result has no GPS target; global guidance disabled.")
            return
        self.set_safe_point(float(gps[0]), float(gps[1]), source="GIS")
        logger.info("[GlobalPrior] Risk grid:\n%s", result.get("risk_grid"))

    def set_safe_point(self, lat: float, lon: float, source: str = "manual"):
        self._safe_lat = lat
        self._safe_lon = lon
        self._safe_ned = None
        self._use_global_guidance = True
        logger.info("[GlobalPrior] Safe point from %s: lat=%.7f lon=%.7f", source, lat, lon)

    def enforce_flight_ready(self):
        """Fail before ROS/MAVROS startup if full-experiment gates are not met."""
        failures = flight_ready_failures(
            self.cfg,
            global_guidance_ready=bool(self._use_global_guidance),
        )
        if failures:
            for failure in failures:
                logger.error("[FlightReady] %s", failure)
            raise RuntimeError(
                "Flight-ready checks failed. Run preflight_check.py --flight-ready "
                "and fix the reported gates before connecting the control loop."
            )
        logger.info("[FlightReady] Strict experiment gates passed.")

    # ------------------------------------------------------------------
    # ROS
    # ------------------------------------------------------------------

    def init_ros_node(self, node_name: str = "orin_landing") -> bool:
        try:
            import rospy
            from nav_msgs.msg import Odometry
            from sensor_msgs.msg import PointCloud2

            if not rospy.core.is_initialized():
                rospy.init_node(node_name, anonymous=False)
            self._rospy = rospy
            self._ros_node = node_name
            self._odom_sub = None
            if self.fastlio_pose_required:
                self._odom_sub = rospy.Subscriber(
                    self.fastlio_odom_topic,
                    Odometry,
                    self.fastlio.odometry_callback,
                    queue_size=10,
                )
            self._cloud_sub = rospy.Subscriber(
                self.fastlio_cloud_topic, PointCloud2, self.fastlio.pointcloud_callback, queue_size=10
            )
            logger.info(
                "[ROS1] Node created, subscribed to cloud=%s odometry=%s.",
                self.fastlio_cloud_topic,
                self.fastlio_odom_topic if self.fastlio_pose_required else "disabled",
            )
            return True
        except Exception as e:
            logger.error("[ROS1] Failed to init: %s", e)
            return False

    async def _verify_localization_profile_online(self) -> None:
        """Reject a mismatched FAST-LIO/PX4 localization setup before arming."""
        if self.localization_mode == "fastlio_external_vision":
            from geometry_msgs.msg import PoseStamped
            try:
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(
                    None,
                    lambda: self._rospy.wait_for_message(
                        self.external_pose_topic, PoseStamped, timeout=5.0
                    ),
                )
            except Exception as exc:
                raise RuntimeError(
                    f"Indoor profile requires live external vision on {self.external_pose_topic}; "
                    "launch FAST-LIO with external_vision:=true"
                ) from exc
            if not getattr(self.fc, "_enu_ready", False):
                raise RuntimeError("Indoor profile requires /mavros/local_position/odom")
            logger.info(
                "[Localization] Indoor gate passed: external vision + MAVROS local odom live"
            )
            return

        deadline = time.perf_counter() + 10.0
        while not getattr(self.fc, "_gps_ready", False) and time.perf_counter() < deadline:
            await asyncio.sleep(0.05)
        if not getattr(self.fc, "_gps_ready", False) or not getattr(self.fc, "_enu_ready", False):
            raise RuntimeError("Outdoor profile requires valid GPS and MAVROS local odometry")
        localization_cfg = self.cfg.get("localization", {})
        gps_ok, gps_reason = self.fc.gps_health(
            max_age_s=float(localization_cfg.get("max_gps_age_s", 2.0)),
            max_horizontal_accuracy_m=float(
                localization_cfg.get("max_gps_horizontal_accuracy_m", 5.0)
            ),
        )
        if not gps_ok:
            raise RuntimeError(f"Outdoor PX4 GPS health gate failed: {gps_reason}")
        if gps_reason == "ok_covariance_unknown":
            logger.warning(
                "[Localization] GPS fix is valid but NavSatFix covariance is unknown"
            )
        # FAST-LIO advertises the topic in both modes, so publisher presence is
        # not sufficient. Outdoor mode rejects actual messages in a quiet
        # observation window.
        from geometry_msgs.msg import PoseStamped
        vision_message_seen = False
        try:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(
                None,
                lambda: self._rospy.wait_for_message(
                    self.external_pose_topic, PoseStamped, timeout=1.0
                ),
            )
            vision_message_seen = True
        except Exception:
            pass
        if vision_message_seen:
            raise RuntimeError(
                f"Outdoor profile forbids FAST-LIO injection on {self.external_pose_topic}; "
                "launch FAST-LIO with external_vision:=false"
            )
        logger.info(
            "[Localization] Outdoor gate passed: GPS/local odom live (%s), vision injection absent",
            gps_reason,
        )

    async def _verify_fastlio_static_initialization(self) -> None:
        """Fail before arming if the outdoor FAST-LIO process is already divergent."""
        cfg = self.cfg.get("fastlio_health", {})
        if not bool(cfg.get("require_static_initialization", True)):
            return
        duration = float(cfg.get("static_initialization_s", 5.0))
        max_drift = float(cfg.get("static_max_drift_m", 0.2))
        max_norm = float(cfg.get("initial_position_norm_max_m", 50.0))

        try:
            import rosgraph
            master = rosgraph.Master(self._ros_node or "/orin_landing")
            publishers, _, _ = master.getSystemState()
            count = len(next((nodes for topic, nodes in publishers
                              if topic == self.fastlio_odom_topic), []))
            if count != 1:
                raise RuntimeError(
                    f"{self.fastlio_odom_topic} must have exactly one publisher; got {count}"
                )
        except RuntimeError:
            raise
        except Exception as exc:
            raise RuntimeError(f"cannot verify FAST-LIO publisher count: {exc}") from exc

        samples = []
        start = time.perf_counter()
        start_cloud_seq = self.fastlio.points_seq
        while time.perf_counter() - start < duration:
            pose = self.fastlio.pose
            if pose is not None and np.isfinite(pose).all():
                samples.append(np.asarray(pose, dtype=np.float64).copy())
            await asyncio.sleep(0.05)
        if len(samples) < max(10, int(duration * 5)):
            raise RuntimeError("FAST-LIO static gate: insufficient finite /ali_odom samples")
        if self.fastlio.points_seq <= start_cloud_seq:
            raise RuntimeError("FAST-LIO static gate: body cloud did not advance")
        xyz = np.asarray([sample[:3] for sample in samples])
        drift = float(np.max(np.linalg.norm(xyz - xyz[0], axis=1)))
        initial_norm = float(np.linalg.norm(xyz[0]))
        quat_norm = self.fastlio.quaternion_norm
        covariance = self.fastlio.pose_covariance
        if initial_norm > max_norm:
            raise RuntimeError(
                f"FAST-LIO static gate: initial position norm {initial_norm:.1f}m > {max_norm:.1f}m; restart FAST-LIO"
            )
        if drift > max_drift:
            raise RuntimeError(
                f"FAST-LIO static gate: drift {drift:.3f}m/{duration:.1f}s > {max_drift:.3f}m"
            )
        if quat_norm is None or not math.isfinite(quat_norm) or abs(quat_norm - 1.0) > 0.01:
            raise RuntimeError(f"FAST-LIO static gate: invalid quaternion norm {quat_norm}")
        if covariance is None or not np.isfinite(covariance).all():
            raise RuntimeError("FAST-LIO static gate: covariance is missing or non-finite")
        logger.info(
            "[FastLIOStatic] Passed: publishers=1 samples=%d initial_norm=%.3fm drift=%.3fm quat=%.6f",
            len(samples), initial_norm, drift, quat_norm,
        )

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    async def run(self):
        self.fc = MAVROSController(
            mavros_ns=self._mavros_ns,
            setpoint_rate_hz=self._setpoint_rate_hz,
            offboard_warmup_s=self._offboard_warmup_s,
        )
        logger.info("[Pipeline] Using MAVROS flight controller (ns=%s)", self._mavros_ns)

        await self.fc.connect()

        logger.info("[Pipeline] Waiting for FAST-LIO data...")
        while not self.fastlio.initialized:
            await asyncio.sleep(0.02)
        logger.info("[Pipeline] FAST-LIO ready.")

        await self._verify_localization_profile_online()

        # --- 启动录包 (若启用) ---
        recording_ok = await self._start_recording()
        record_cfg = self.cfg.get("experiment_recording", {})
        if self._record_bag and record_cfg.get("required", False) and not recording_ok:
            raise RuntimeError("Required rosbag recording failed before arming")
        if self._use_body_cloud:
            try:
                await self._verify_fastlio_static_initialization()
            except Exception:
                await self._stop_recording()
                raise

        # Capture the launch reference while the aircraft is still on the
        # ground.  Manual climb before OFFBOARD must never redefine zero height.
        launch_ned, launch_attitude = await self.fc.wait_for_local_pose(timeout_s=10.0)
        self._home_ned = launch_ned.copy()
        self._home_enu = self.fc.uavPosENU.copy()
        self._pre_goto_yaw_enu_deg = math.degrees(float(self.fc.uavYawENU))
        self._launch_attitude_ned = launch_attitude.copy()
        if self._use_global_guidance and not self._indoor_local_mode:
            _, launch_lat_lon, _ = await self.fc.wait_for_home(timeout_s=10.0)
            self._home_lat, self._home_lon = launch_lat_lon
        if self._use_body_cloud:
            self._ground_z_world = float(self._home_enu[2])
            self.state_manager.ground_z_ref_m = self._ground_z_world
            self.state_manager.height_source = "px4_enu_z"
            self.state_manager.height_axis = "pos_z"
        elif self.fastlio.pose is not None:
            self._ground_z_world = float(self.fastlio.pose[2])
            if self.state_manager.auto_ground_z_ref:
                self.state_manager.ground_z_ref_m = self._ground_z_world
        self._log_mission_event("LAUNCH_REFERENCE", reason="ground_reference_captured")
        logger.info(
            "[Pipeline] Ground launch reference: NED=%s ENU=%s gps=%s ground_z=%.2f",
            _fmt_vec(self._home_ned), _fmt_vec(self._home_enu),
            "n/a" if self._home_lat is None else f"{self._home_lat:.7f},{self._home_lon:.7f}",
            self._ground_z_world,
        )

        # --- 新流程: 持续发送当前位置 hold, 等待遥控器 OFFBOARD+解锁 ---
        self.fc.start_hold_stream()
        logger.info("[Pipeline] Hold stream started. Waiting for RC OFFBOARD+arm...")
        rc_ok = await self.fc.wait_for_manual_offboard_and_arm(
            timeout_s=120.0,
            # Keep yaw fixed while recording home and during vertical takeoff.
            # The planned yaw fault starts only when GOTO_SAFE begins.
            post_arm_yaw_rate_rad_s=None,
        )
        if not rc_ok:
            logger.error("[Pipeline] RC OFFBOARD+arm timeout. Aborting.")
            await self._stop_recording()
            await self._shutdown()
            return
        logger.info("[Pipeline] RC OFFBOARD+arm confirmed. Proceeding to mission.")
        self._log_mission_event("ARMED", reason="manual_offboard_and_arm_confirmed")

        # The handoff point is diagnostic only; the ground launch reference
        # captured above remains authoritative.
        local_ned, local_attitude = await self.fc.wait_for_local_pose(timeout_s=10.0)
        self._handoff_enu = self.fc.uavPosENU.copy()
        self._log_mission_event("OFFBOARD_HANDOFF", reason="manual_climb_complete")
        logger.info(
            "[Pipeline] Home: ned=%s enu=%s ground_z=%.2f",
            _fmt_vec(self._home_ned),
            _fmt_vec(self._home_enu),
            self._ground_z_world,
        )

        if self._indoor_local_mode:
            self._safe_ned_target = self._compute_local_ned_target_3d(
                self._home_ned, float(local_attitude[2]),
            )
            self._takeoff_altitude_m = float(self._local_body_offset_m[2])
            logger.info(
                "[Pipeline] Indoor 3D safe point: NED=%s offset=(%.2f,%.2f,%.2f)m takeoff_alt=%.1fm",
                _fmt_vec(self._safe_ned_target),
                self._local_body_offset_m[0], self._local_body_offset_m[1],
                self._local_body_offset_m[2],
                self._takeoff_altitude_m,
            )
            self._apply_state_decision(self.state_manager.start_after_takeoff(True))

        elif self._use_global_guidance:
            self._takeoff_altitude_m = float(self._safe_altitude_m)
            logger.info(
                "[Pipeline] Home telemetry: lat=%.7f lon=%.7f ned=%s takeoff_alt=%.1fm",
                self._home_lat, self._home_lon, _fmt_vec(self._home_ned),
                self._takeoff_altitude_m,
            )

            if self._safe_ned is None:
                self._safe_ned = self._gps_to_ned_offset_3d(
                    self._safe_lat, self._safe_lon, self._safe_altitude_m,
                )
            logger.info(
                "[Pipeline] GPS 3D target: lat=%.7f lon=%.7f alt=%.1fm → NED=%s",
                self._safe_lat, self._safe_lon, self._safe_altitude_m,
                _fmt_vec(self._safe_ned),
            )

            dist_xy = float(np.linalg.norm(self._safe_ned[:2] - self._home_ned[:2]))
            dist_z = abs(self._safe_ned[2] - self._home_ned[2])
            if dist_xy <= 1e-3 and dist_z <= 0.5:
                raise RuntimeError(
                    "[GlobalPrior] Safe point too close to home; "
                    "home GPS may be unavailable or target equals takeoff position."
                )
            self._apply_state_decision(self.state_manager.start_after_takeoff(True))
        else:
            self._apply_state_decision(self.state_manager.start_after_takeoff(False))
            logger.info("[Pipeline] No global prior. Starting DRL descent.")

        # --- Phase 1: vertical takeoff to safe altitude (if global guidance) ---
        if self._indoor_local_mode or self._use_global_guidance:
            print(f"\n>>> Phase 1: Taking off to {self._takeoff_altitude_m:.1f}m ...", flush=True)
            self._log_mission_event(
                "TAKEOFF_STARTED", reason=f"target_altitude_m={self._takeoff_altitude_m:.3f}"
            )
            reached_high_altitude = await self._vertical_takeoff(self._takeoff_altitude_m)
            if reached_high_altitude:
                self._log_mission_event(
                    "HIGH_ALTITUDE_REACHED",
                    reason=f"target_altitude_m={self._takeoff_altitude_m:.3f}",
                )
            else:
                logger.warning(
                    "[MissionEvent] HIGH_ALTITUDE_REACHED not recorded: target tolerance not met"
                )
                self._apply_state_decision(self.state_manager.reset(
                    MissionState.HOLD_FOR_MANUAL,
                    "takeoff_staging_timeout",
                ))
                self._log_mission_event(
                    "MANUAL_TAKEOVER_REQUESTED",
                    reason="takeoff_staging_timeout",
                )
            print(f">>> Phase 1 complete. mission_state={self.mission_state}", flush=True)

        if self.mission_state == "GOTO_SAFE":
            print(f"\n>>> Phase 2: Moving to safe point ...", flush=True)
            self._log_mission_event("GOTO_STARTED", reason="goto_safe_phase_started")
            await self._goto_safe_point()
            print(f">>> Phase 2 complete. mission_state={self.mission_state}", flush=True)

        if self.mission_state == "HOLD_FOR_MANUAL":
            await self._hold_for_manual_takeover()
            await self._shutdown()
            return
        if self.mission_state in ("ABORT", "IDLE"):
            await self._shutdown()
            return

        # --- GPU warmup: 在安全点运行一帧 HALSS + depth + DRL, 预热 CUDA ---
        print(f"\n>>> GPU warmup at safe point ...", flush=True)
        warmup_ok = await self._warmup_perception()
        if not warmup_ok:
            self._apply_state_decision(self.state_manager.reset(
                MissionState.HOLD_FOR_MANUAL,
                "perception_warmup_failed",
            ))
            self._log_mission_event(
                "MANUAL_TAKEOVER_REQUESTED",
                reason="perception_warmup_failed",
            )
            await self._hold_for_manual_takeover()
            await self._shutdown()
            return
        print(f">>> GPU warmup complete.", flush=True)

        print(f"\n>>> Starting DRL descent loop ...", flush=True)
        await self._run_descent_loop()
        await self._shutdown()

    # The former MAVSDK arm/offboard branch was intentionally removed from the
    # live entry point. Historical MAVSDK adapters remain in control/ only for
    # old offline tools; real flights use MAVROS and manual RC OFFBOARD + arm.

    async def _run_descent_loop(self):
        logger.info("[Pipeline] Starting DRL descent control loop...")
        print(f"[DescentLoop] Entered. mission_state={self.mission_state}", flush=True)
        last_processed_seq = -1
        last_wait_log = 0.0
        first_frame = True
        perception_gate_failure_since = None

        try:
            while self.mission_state not in ("LANDED", "ABORT", "EMERGENCY_STOP"):
                loop_start = time.perf_counter()

                # ── DEBUG: loop entry every 20 iterations ──
                if self.step_count % 20 == 0:
                    pts_now = self.fastlio.points
                    stamp_now = self.fastlio.points_stamp
                    logger.info(
                        "[DEBUG_LOOP] state=%s cloud_seq=%d pose_seq=%d "
                        "last_processed=%d points=%d stamp=%s",
                        self.mission_state,
                        self.fastlio.points_seq,
                        self.fastlio.pose_seq,
                        last_processed_seq,
                        len(pts_now) if pts_now is not None else 0,
                        f"{stamp_now:.3f}" if stamp_now is not None else "None",
                    )

                # Flight-controller state is authoritative and must be checked
                # before waiting on perception.  Leaving OFFBOARD is an
                # intentional manual takeover, not an autonomous emergency.
                if self.fc is not None and getattr(self.fc, "isOffboard", None) is False:
                    logger.warning("[Pipeline] OFFBOARD exited; yielding to manual control")
                    self._log_mission_event("MANUAL_TAKEOVER", reason="offboard_exited")
                    self.mission_state = "IDLE"
                    break
                if self.fc is not None and getattr(self.fc, "isArmed", None) is False:
                    logger.info("[Pipeline] Vehicle disarmed; ending control loop")
                    self._log_mission_event("DISARMED", reason="px4_disarmed")
                    self.mission_state = "LANDED"
                    break

                (
                    frame_points,
                    frame_pose,
                    cloud_seq,
                    pose_seq,
                    sync_ms,
                ) = self._grab_latest_snapshot()
                cloud_stamp_ros_s = self.fastlio.points_stamp

                # ── DEBUG: snapshot sync status ──
                logger.debug(
                    "[DEBUG_SNAPSHOT] pts=%s pose=%s cloud_seq=%d pose_seq=%d "
                    "sync_ms=%s stamp=%s",
                    "None" if frame_points is None else str(len(frame_points)),
                    "None" if frame_pose is None else f"[{frame_pose[0]:.2f},{frame_pose[1]:.2f},{frame_pose[2]:.2f}]",
                    cloud_seq, pose_seq,
                    f"{sync_ms:.1f}" if sync_ms is not None else "None",
                    f"{cloud_stamp_ros_s:.3f}" if cloud_stamp_ros_s is not None else "None",
                )
                if frame_points is None or frame_pose is None:
                    now = time.perf_counter()
                    if perception_gate_failure_since is None:
                        perception_gate_failure_since = now
                    if now - last_wait_log > 3.0:
                        logger.info(
                            "[Pipeline] Waiting for cloud=%s + pose=%s...",
                            self.fastlio_cloud_topic,
                            "PX4 timestamp match" if self._use_body_cloud else self.fastlio_odom_topic,
                        )
                        last_wait_log = now
                    if self._use_body_cloud and frame_points is not None:
                        latest_px4 = np.array([
                            *np.asarray(self.fc.uavPosENU, dtype=np.float32),
                            float(getattr(self.fc, "uavRollENU", 0.0)),
                            float(getattr(self.fc, "uavPitchENU", 0.0)),
                            float(getattr(self.fc, "uavYawENU", 0.0)),
                        ], dtype=np.float32)
                        self._log_perception_gate(
                            cloud_seq, pose_seq, None, latest_px4,
                            "px4_cloud_sync", False, "no_px4_odom_within_sync_limit",
                            frame_points,
                        )
                    await self._send_zero_velocity(0.0 if self._use_body_cloud else self.yaw_rate_cmd, now)
                    if self._use_body_cloud and now - perception_gate_failure_since > 0.2:
                        self._apply_state_decision(self.state_manager.reset(
                            MissionState.HOLD_FOR_MANUAL,
                            "body_cloud_or_px4_sync_timeout",
                        ))
                        self._log_mission_event(
                            "MANUAL_TAKEOVER_REQUESTED",
                            reason="body_cloud_or_px4_sync_timeout",
                        )
                        await self._hold_for_manual_takeover()
                        break
                    await asyncio.sleep(0.02)
                    continue

                if cloud_seq <= last_processed_seq:
                    await asyncio.sleep(0.005)
                    continue

                if sync_ms is None:
                    logger.warning(
                        "[Pipeline] Missing FAST-LIO header timestamps; zero velocity cloud_seq=%d pose_seq=%d",
                        cloud_seq,
                        pose_seq,
                    )
                    now = time.perf_counter()
                    if perception_gate_failure_since is None:
                        perception_gate_failure_since = now
                    await self._send_zero_velocity(0.0 if self._use_body_cloud else self.yaw_rate_cmd, now)
                    if self._use_body_cloud and now - perception_gate_failure_since > 0.2:
                        self._apply_state_decision(self.state_manager.reset(
                            MissionState.HOLD_FOR_MANUAL,
                            "cloud_timestamp_missing",
                        ))
                        self._log_mission_event(
                            "MANUAL_TAKEOVER_REQUESTED",
                            reason="cloud_timestamp_missing",
                        )
                        await self._hold_for_manual_takeover()
                        break
                    await asyncio.sleep(0.005)
                    continue

                if sync_ms > self.max_cloud_odom_sync_ms:
                    logger.warning(
                        "[Pipeline] FAST-LIO sync %.0fms > %.0fms; zero velocity cloud_seq=%d pose_seq=%d",
                        sync_ms,
                        self.max_cloud_odom_sync_ms,
                        cloud_seq,
                        pose_seq,
                    )
                    now = time.perf_counter()
                    if perception_gate_failure_since is None:
                        perception_gate_failure_since = now
                    self._log_perception_gate(
                        cloud_seq, pose_seq, sync_ms, frame_pose,
                        "px4_cloud_sync", False, "sync_limit_exceeded",
                        frame_points,
                    )
                    await self._send_zero_velocity(0.0 if self._use_body_cloud else self.yaw_rate_cmd, now)
                    if self._use_body_cloud and now - perception_gate_failure_since > 0.2:
                        self._apply_state_decision(self.state_manager.reset(
                            MissionState.HOLD_FOR_MANUAL,
                            "px4_cloud_sync_timeout",
                        ))
                        self._log_mission_event(
                            "MANUAL_TAKEOVER_REQUESTED",
                            reason=f"px4_cloud_sync_timeout;sync_ms={sync_ms:.1f}",
                        )
                        await self._hold_for_manual_takeover()
                        break
                    await asyncio.sleep(0.005)
                    continue

                last_processed_seq = cloud_seq
                perception_gate_failure_since = None

                # --- 定位源健康评估 ---
                # FAST-LIO stamps use the ROS clock. Never subtract them from
                # time.perf_counter(), which has an unrelated process-local epoch.
                now_ts = self._rospy.Time.now().to_sec() if self._rospy is not None else time.time()
                health = self.pose_source_mgr.evaluate(
                    # In outdoor body-cloud mode /ali_odom is diagnostic only.
                    fastlio_pose=(
                        self.fastlio.pose
                        if self.fastlio_pose_required
                        else None
                    ) if self._use_body_cloud else frame_pose,
                    fastlio_pose_stamp=(
                        self.fastlio.pose_stamp if self.fastlio_pose_required else None
                    ),
                    fastlio_points=frame_points,
                    fastlio_points_stamp=self.fastlio.points_stamp,
                    now=now_ts,
                )

                # Outdoor body-cloud mode is PX4-authoritative by construction.
                if self._use_body_cloud:
                    control_pose = frame_pose
                elif health.control_pose_source.value == "gps_fallback":
                    control_pose = np.array([
                        self.fc.uavPosNED[0], self.fc.uavPosNED[1], self.fc.uavPosNED[2],
                        self.fc.uavAngEular[0], self.fc.uavAngEular[1], self.fc.uavAngEular[2],
                    ], dtype=np.float32)
                    logger.info(
                        "[PoseSrc] GPS FALLBACK ACTIVE: control_pose=gps_fallback "
                        "perception_cloud=%s pose_healthy=%s cloud_healthy=%s reason=%s",
                        health.perception_cloud_source.value,
                        health.fastlio_pose_healthy,
                        health.fastlio_cloud_healthy,
                        health.degraded_reason or "n/a",
                    )
                else:
                    control_pose = frame_pose

                if self._use_body_cloud and not health.fastlio_cloud_healthy:
                    await self._send_zero_velocity_logged(
                        0.0, time.perf_counter(), state=self.mission_state,
                        reason="body_cloud_unhealthy", pose_xyz=control_pose,
                        sync_ms=sync_ms, fallback_reason=health.degraded_reason or "cloud_health",
                    )
                    self._log_perception_gate(
                        cloud_seq, pose_seq, sync_ms, control_pose,
                        "body_cloud_health", False, health.degraded_reason or "unhealthy",
                        frame_points,
                    )
                    self._apply_state_decision(self.state_manager.reset(
                        MissionState.HOLD_FOR_MANUAL,
                        "body_cloud_unhealthy",
                    ))
                    self._log_mission_event(
                        "MANUAL_TAKEOVER_REQUESTED",
                        reason=health.degraded_reason or "body_cloud_unhealthy",
                    )
                    await self._hold_for_manual_takeover()
                    break

                # A large PX4 roll/pitch can invalidate the level-body ROI.
                # down-looking ROI. Do not let such a geometrically invalid
                # observation reach the policy; hold until the vehicle levels.
                if max(abs(float(control_pose[3])), abs(float(control_pose[4]))) > self.max_roll_pitch_rad:
                    await self._send_zero_velocity_logged(
                        self.yaw_rate_cmd,
                        time.perf_counter(),
                        state=self.mission_state,
                        reason="attitude_exceeds_perception_limit",
                        pose_xyz=control_pose,
                        sync_ms=sync_ms,
                        fallback_reason="attitude_hold",
                    )
                    logger.warning(
                        "[Pipeline] Attitude hold: roll=%.1fdeg pitch=%.1fdeg limit=%.1fdeg",
                        math.degrees(float(control_pose[3])),
                        math.degrees(float(control_pose[4])),
                        math.degrees(self.max_roll_pitch_rad),
                    )
                    self.step_count += 1
                    self._log_perception_gate(
                        cloud_seq, pose_seq, sync_ms, control_pose,
                        "px4_attitude", False, "attitude_exceeds_perception_limit",
                        frame_points,
                    )
                    await asyncio.sleep(max(0.0, self.sim_dt - (time.perf_counter() - loop_start)))
                    continue

                # ---- 动态 FOV ROI: 根据当前对地高度计算半宽 ---- 
                dyn_half_x, dyn_half_y, dyn_height = self._compute_roi_half_from_height(
                    float(control_pose[2]), None,
                )
                self._roi_half_x = dyn_half_x
                self._roi_half_y = dyn_half_y

                # Outdoor uses the deskewed IMU/body cloud and PX4 roll/pitch;
                # indoor retains the matched FAST-LIO world cloud/pose path.
                t_p0 = time.perf_counter()
                halss_points, projection_points, halss_stats = self._prepare_halss_points(
                    frame_points, frame_pose, control_pose, dyn_half_x, dyn_half_y,
                )
                t_p1 = time.perf_counter()
                self._log_perception_gate(
                    cloud_seq, pose_seq, sync_ms, control_pose,
                    "ready_for_inference", halss_stats["output_points"] >= 10,
                    "ok" if halss_stats["output_points"] >= 10 else "roi_sparse",
                    frame_points, halss_stats,
                )

                ground_p05 = self._pointcloud_ground_clearance(projection_points, 5.0)
                ground_min = self._pointcloud_ground_clearance(projection_points, 0.0)
                pose_xyz = tuple(float(x) for x in control_pose[:3])
                pose_yaw = float(control_pose[5])
                pose_height_m = self.state_manager.height_from_pose(pose_xyz)
                landed_state_on_ground = bool(
                    getattr(self.fc, "landed_state_on_ground", False)
                ) and (
                    pose_height_m is None
                    or pose_height_m <= max(0.3, self.state_manager.landed_height_m + 0.15)
                )
                if landed_state_on_ground:
                    self._log_mission_event(
                        "PX4_ON_GROUND", reason="mavros_extended_state_on_ground"
                    )
                state_inputs = StateInputs(
                    now=loop_start,
                    pose_xyz=pose_xyz,
                    yaw_rad=pose_yaw,
                    velocity_xyz=tuple(float(x) for x in getattr(
                        self.fc,
                        "uavVelENU" if self._use_body_cloud else "uavVelNED",
                        np.zeros(3),
                    )[:3]),
                    cloud_odom_sync_ms=sync_ms,
                    perception_ok=halss_stats["output_points"] >= 10,
                    landed_state_on_ground=landed_state_on_ground,
                    offboard_active=getattr(self.fc, "isOffboard", None),
                    armed=getattr(self.fc, "isArmed", None),
                    step_count=self.step_count,
                    max_steps=self.max_steps,
                    ground_clearance_p05_m=ground_p05,
                    ground_clearance_min_m=ground_min,
                )
                state_decision = self.state_manager.update(state_inputs)
                self._apply_state_decision(state_decision)

                # ── DEBUG: state manager gate ──
                logger.info(
                    "[DEBUG_GATE] cloud_seq=%d sync_ms=%s roi_points=%d "
                    "state=%s allow_drl=%s direct_land=%s abort=%s "
                    "reason=%s",
                    cloud_seq,
                    f"{sync_ms:.1f}" if sync_ms is not None else "None",
                    halss_stats.get("output_points", 0),
                    state_decision.state.value,
                    state_decision.allow_drl,
                    state_decision.direct_land,
                    state_decision.abort,
                    state_decision.reason,
                )

                if first_frame:
                    if self._use_body_cloud and self._halss_pinhole_ray_sampling:
                        print(
                            "[DescentLoop] Pinhole sampling: "
                            f"cloud={halss_stats.get('input_points', 0)} "
                            f"rect_roi={halss_stats.get('pre_ray_points', 0)} "
                            f"frustum={halss_stats.get('frustum_points', 0)} "
                            f"rays={halss_stats.get('output_points', 0)}/"
                            f"{self._halss_ray_grid_res ** 2}",
                            flush=True,
                        )
                    print(f"[DescentLoop] First frame: pose_z={pose_xyz[2]:.2f} height_m={pose_height_m} "
                          f"state={state_decision.state.value} reason={state_decision.reason} "
                          f"allow_drl={state_decision.allow_drl} direct_land={state_decision.direct_land} "
                          f"landed={state_decision.landed} abort={state_decision.abort}",
                          flush=True)
                    first_frame = False

                # --- MAVROS 安全 fallback: OFFBOARD 丢失或 disarm → ABORT ---
                if self.fc is not None and self.fc.safety_fallback:
                    logger.error(
                        "[Pipeline] MAVROS safety fallback: %s → ABORT",
                        self.fc.safety_fallback_reason,
                    )
                    self.mission_state = "ABORT"
                    await self._stop_recording()
                    await self._emergency_stop()
                    break

                # --- FAST-LIO 健康退化动作 ---
                if health.degraded_action is not None and not self._use_body_cloud:
                    degraded = health.degraded_action.value
                    if degraded == "abort":
                        logger.error(
                            "[Pipeline] FAST-LIO health degraded → ABORT: reason=%s "
                            "pose_healthy=%s cloud_healthy=%s",
                            health.degraded_reason,
                            health.fastlio_pose_healthy,
                            health.fastlio_cloud_healthy,
                        )
                        self.mission_state = "ABORT"
                        await self._emergency_stop()
                        break
                    elif degraded == "direct_land":
                        logger.warning(
                            "[Pipeline] FAST-LIO health degraded → DIRECT_LAND: reason=%s "
                            "pose_healthy=%s cloud_healthy=%s",
                            health.degraded_reason,
                            health.fastlio_pose_healthy,
                            health.fastlio_cloud_healthy,
                        )
                        # 强制进入 direct_land
                        if self.mission_state not in ("DIRECT_LAND", "LANDED", "ABORT"):
                            self._apply_state_decision(self.state_manager.reset(
                                state=MissionState.DIRECT_LAND,
                                reason=f"fastlio_health_degraded: {health.degraded_reason}",
                            ))
                    # "use_gps_fallback" 已在上面通过 control_pose 处理, 无需额外动作

                if state_decision.abort:
                    logger.warning("[Pipeline] State manager requested abort: %s", state_decision.reason)
                    self.mission_state = "ABORT"
                    await self._emergency_stop()
                    break
                if state_decision.landed:
                    logger.info(
                        "[Pipeline] Target altitude reached: %.2fm",
                        float("nan") if state_decision.height_m is None else state_decision.height_m,
                    )
                    if state_decision.reason == "px4_landed_state":
                        self._log_mission_event(
                            "PX4_ON_GROUND", reason="mavros_extended_state_on_ground"
                        )
                    await self.fc.send_velocity_enu_yaw_rate(0.0, 0.0, 0.0, 0.0)
                    await self.fc.disarm()
                    for _ in range(40):
                        if not bool(getattr(self.fc, "isArmed", True)):
                            self._log_mission_event("DISARMED", reason="px4_disarm_confirmed")
                            break
                        await asyncio.sleep(0.05)
                    else:
                        logger.warning(
                            "[MissionEvent] DISARMED not recorded: PX4 armed state did not clear"
                        )
                    break

                if not state_decision.allow_drl and not state_decision.direct_land:
                    if state_decision.state.value == "IDLE":
                        logger.warning(
                            "[Pipeline] Control loop left active flight state: reason=%s",
                            state_decision.reason,
                        )
                        break
                    await self._send_zero_velocity_logged(
                        self.yaw_rate_cmd, time.perf_counter(),
                        state=state_decision.state.value,
                        reason=state_decision.reason,
                        pose_xyz=frame_pose,
                        sync_ms=sync_ms,
                        fallback_reason="fsm_hold",
                    )
                    logger.info(
                        "[%04d] HOLD state=%s reason=%s yaw_rate_sp=%.2f",
                        self.step_count,
                        state_decision.state.value,
                        state_decision.reason,
                        self.yaw_rate_cmd,
                    )
                    self.step_count += 1
                    await asyncio.sleep(max(0.0, self.sim_dt - (time.perf_counter() - loop_start)))
                    continue

                if state_decision.direct_land:
                    t_h0 = t_h1 = time.perf_counter()
                    sem_map = np.full((self.obs_h, self.obs_w), self.danger_id, dtype=np.uint8)
                    binary_semantic_vis = make_binary_semantic_vis(
                        sem_map,
                        safe_id=self.safe_id,
                        danger_id=self.danger_id,
                    )
                    t_d0 = t_d1 = time.perf_counter()
                    sparse_depth = np.full((self.obs_h, self.obs_w), self.depth_max, dtype=np.float32)
                    valid_mask = np.zeros((self.obs_h, self.obs_w), dtype=bool)
                    semantic_valid_mask = np.zeros((self.obs_h, self.obs_w), dtype=bool)
                    t_c0 = t_c1 = time.perf_counter()
                    rendered_depth = sparse_depth.copy()
                    action_id = 9
                    action_name = "DIRECT_LAND"
                    t_r0 = time.perf_counter()
                    rl_info = self._direct_land_rl_info(rendered_depth, sem_map)
                    v_body = np.array([0.0, 0.0, state_decision.direct_land_vz_mps], dtype=np.float32)
                    v_ned = v_body.copy()
                    action_yaw_rate = self.yaw_rate_cmd if state_decision.continue_yaw_rate else 0.0
                    t_r1 = time.perf_counter()
                else:
                    t_h0 = time.perf_counter()
                    roi_bounds = self._roi_bounds()
                    halss_result = self.halss.evaluate(
                        halss_points, fixed_bounds=roi_bounds,
                    )
                    t_h1 = time.perf_counter()
                    if halss_stats["output_points"] < 10:
                        logger.warning(
                            "[Pipeline] HALSS level-body ROI sparse: roi=%d/%d "
                            "half=(%.1f,%.1f)m height=%.1fm z=[%.2f, %.2f]",
                            halss_stats["output_points"],
                            halss_stats["input_points"],
                            dyn_half_x,
                            dyn_half_y,
                            dyn_height,
                            halss_stats["z_min_body"],
                            halss_stats["z_max_body"],
                        )

                    if halss_result is not None:
                        bev_data = halss_result.get("bev_data", halss_result)
                        sem_map = self.sem_gen.generate(bev_data)
                    else:
                        sem_map = np.full((self.obs_h, self.obs_w), self.danger_id, dtype=np.uint8)
                    binary_semantic_vis = make_binary_semantic_vis(
                        sem_map,
                        safe_id=self.safe_id,
                        danger_id=self.danger_id,
                    )

                    t_d0 = time.perf_counter()
                    semantic_valid_mask = np.ones_like(sem_map, dtype=bool)
                    if self._projection_mode == "training_camera":
                        sparse_depth, valid_mask, sem_map, semantic_valid_mask = (
                            project_training_camera(
                                projection_points,
                                sem_map,
                                roi_bounds,
                                self._training_camera,
                                danger_id=self.danger_id,
                            )
                        )
                        binary_semantic_vis = make_binary_semantic_vis(
                            sem_map,
                            safe_id=self.safe_id,
                            danger_id=self.danger_id,
                        )
                        binary_semantic_vis[~semantic_valid_mask] = 128
                    else:
                        sparse_depth, _ = project_bev_depth(
                            halss_points,
                            grid_res=int(self.cfg["perception"].get("halss_grid_res", 64)),
                            out_size=self.obs_w,
                            max_range=self.depth_max,
                            half_x=self._roi_half_x,
                            half_y=self._roi_half_y,
                        )
                        valid_mask = (sparse_depth < self.depth_max) & (sparse_depth > 0.01)
                    t_d1 = time.perf_counter()

                    t_c0 = time.perf_counter()
                    rendered_depth = render_sparse_depth(sparse_depth, valid_mask, self.depth_max)
                    t_c1 = time.perf_counter()

                    t_r0 = time.perf_counter()
                    # ── DEBUG: DRL input ──
                    logger.debug(
                        "[DEBUG_DRL_INPUT] depth_valid=%d/%d semantic_valid=%d/%d "
                        "depth_range=[%.2f,%.2f] sem_unique=%s",
                        int(np.sum(valid_mask)), valid_mask.size,
                        int(np.sum(sem_map != self.danger_id)), sem_map.size,
                        float(np.min(rendered_depth)), float(np.max(rendered_depth)),
                        sorted(np.unique(sem_map).tolist()),
                    )
                    action_id, rl_info = self.drl.predict(rendered_depth, sem_map)
                    rl_info["semantic_valid_ratio"] = float(np.mean(semantic_valid_mask))
                    t_r1 = time.perf_counter()
                    # ── DEBUG: DRL output ──
                    logger.info(
                        "[DEBUG_DRL_OUTPUT] action_id=%d action_name=%s "
                        "confidence=%.3f probs=%s",
                        action_id,
                        self.decomposer.action_id_to_name(action_id),
                        float(rl_info.get("confidence", -1)),
                        _top_probs(rl_info.get("action_probs")),
                    )
                    print(f"  ONNX {int((t_r1 - t_r0) * 1000)}ms act={action_id}", flush=True)

                    action_name = self.decomposer.action_id_to_name(action_id)
                    execution_yaw_ned = (
                        float(self.fc.uavAngEular[2])
                        if self.execution_yaw_source == "px4_ekf"
                        else float(frame_pose[5])
                    )
                    v_body, v_ned, action_yaw_rate = self.decomposer.decompose(
                        action_id,
                        execution_yaw_ned,
                    )

                if (
                    state_decision.reason.endswith("crosscheck_warn")
                    and state_decision.height_m is not None
                    and state_decision.ground_clearance_p05_m is not None
                ):
                    logger.warning(
                        "[State] height/ground crosscheck mismatch: height=%.2fm ground_p05=%.2fm threshold=%.2fm",
                        state_decision.height_m,
                        state_decision.ground_clearance_p05_m,
                        self.state_manager.ground_crosscheck_max_error_m,
                    )

                pre_control_ms = (time.perf_counter() - loop_start) * 1000.0
                source_age_ms = max(0.0, float(health.cloud_age_ms or 0.0))
                result_age_ms = source_age_ms + pre_control_ms
                fresh_cloud_seq = self.fastlio.points_seq
                newer_frames = max(0, fresh_cloud_seq - cloud_seq)
                if newer_frames:
                    logger.debug(
                        "[%04d] inference result accepted with %d newer cloud frame(s): "
                        "source=%d latest=%d age=%.0fms",
                        self.step_count, newer_frames, cloud_seq, fresh_cloud_seq, result_age_ms,
                    )
                result_too_old = result_age_ms > self.max_inference_result_age_ms
                if (result_too_old and self.drop_stale_frames and self.drop_slow_frames
                        and not state_decision.direct_land):
                    await self._send_zero_velocity_logged(
                        action_yaw_rate, time.perf_counter(),
                        state=state_decision.state.value,
                        reason=state_decision.reason,
                        pose_xyz=frame_pose,
                        sync_ms=sync_ms,
                        fallback_reason="inference_result_too_old",
                    )
                    self._record_timing(
                        t_h1 - t_h0, t_d1 - t_d0, t_c1 - t_c0,
                        t_r1 - t_r0, 0.0, time.perf_counter() - loop_start,
                        cloud_stamp_ros_s=cloud_stamp_ros_s,
                        cloud_seq=cloud_seq,
                        pose_seq=pose_seq,
                        state=state_decision.state.value,
                        sync_ms=sync_ms,
                        source_age_ms=source_age_ms,
                        result_age_ms=result_age_ms,
                        newer_frames=newer_frames,
                        accepted=False,
                        fallback_reason="inference_result_too_old",
                        perception_executed=True,
                        pointcloud_preprocess_s=t_p1 - t_p0,
                    )
                    logger.warning(
                        "[%04d] INFERENCE RESULT TOO OLD age=%.0fms limit=%.0fms "
                        "cloud_seq=%d pose_seq=%d yaw=%.1fdeg yaw_rate_sp=%.2f act=%d(%s) v_ned=%s",
                        self.step_count, result_age_ms, self.max_inference_result_age_ms,
                        cloud_seq, pose_seq, math.degrees(float(frame_pose[5])), action_yaw_rate,
                        action_id, action_name, _fmt_vec(v_ned),
                    )
                    if not state_decision.direct_land:
                        self._monitor_action_collapse(
                            action_id, action_name, rl_info, sparse_depth, valid_mask,
                            rendered_depth, sem_map, binary_semantic_vis, frame_pose,
                            sync_ms, v_body, v_ned, cloud_seq, pose_seq,
                            semantic_valid_mask=semantic_valid_mask,
                        )
                    try:
                        self.visualizer.update(
                            sem_map=sem_map,
                            depth_map=rendered_depth,
                            binary_semantic_vis=binary_semantic_vis,
                        )
                    except Exception:
                        pass
                    self.step_count += 1
                    await asyncio.sleep(max(0.0, self.sim_dt - (time.perf_counter() - loop_start)))
                    continue

                t_ctrl = time.perf_counter()
                # ENU velocity + yaw_rate control
                enu_vx, enu_vy, enu_vz = ned_to_enu_velocity(
                    float(v_ned[0]), float(v_ned[1]), float(v_ned[2])
                )
                v_mavros_sp = np.zeros(3, dtype=np.float32)
                if state_decision.direct_land:
                    # FAST-LIO world XY and MAVROS ENU XY are not guaranteed to
                    # share an origin or axes. Lock landing XY in PX4's own ENU
                    # frame at the moment DIRECT_LAND becomes active.
                    if self._direct_land_enu_xy is None:
                        px4_enu = np.asarray(self.fc.uavPosENU, dtype=np.float32)
                        self._direct_land_enu_xy = (float(px4_enu[0]), float(px4_enu[1]))
                    lx, ly = self._direct_land_enu_xy
                    await self.fc.send_landing_enu_yaw_rate(lx, ly, enu_vz, action_yaw_rate)
                else:
                    self._direct_land_enu_xy = None
                    await self._send_velocity_with_horizontal_hold(
                        enu_vx, enu_vy, enu_vz, action_yaw_rate
                    )
                v_mavros_sp = np.array([enu_vx, enu_vy, enu_vz], dtype=np.float32)
                t_ctrl_end = time.perf_counter()

                # ---- real-time per-frame output (like test_live_nocontrol.py) ----
                pos_str = f"pos=({control_pose[0]:.1f},{control_pose[1]:.1f},{control_pose[2]:.2f})"
                v_str = f"v_enu=({enu_vx:+.2f},{enu_vy:+.2f},{enu_vz:+.2f})"
                yaw_str = f"yaw={math.degrees(float(control_pose[5])):.0f}deg yr={action_yaw_rate:+.2f}"
                lat_str = f"lat={(t_ctrl_end - loop_start)*1000:.0f}ms"
                land_str = " [DIRECT_LAND]" if state_decision.direct_land else ""
                print(
                    f"\r[{self.step_count:04d}] ACT={action_id}({action_name}) {pos_str} {v_str} {yaw_str} {lat_str}{land_str}",
                    end="", flush=True,
                )

                # --- velocity command CSV log ---
                _fc_vel = getattr(self.fc, "uavVelNED", np.zeros(3, dtype=np.float32))
                _att = getattr(self.fc, "uavAngEular", np.zeros(3, dtype=np.float32))
                self._log_velocity_command(
                    step=self.step_count,
                    timestamp_s=loop_start,
                    state=state_decision.state.value,
                    reason=state_decision.reason,
                    action_id=action_id,
                    action_name=action_name,
                    v_body=v_body,
                    v_ned=v_ned,
                    v_mavros_sp=v_mavros_sp,
                    fc_vel=_fc_vel,
                    roll_deg=math.degrees(float(getattr(self.fc, "uavRollENU", _att[0]))),
                    pitch_deg=math.degrees(float(getattr(self.fc, "uavPitchENU", _att[1]))),
                    yaw_deg=math.degrees(float(getattr(self.fc, "uavYawENU", 0.0))),
                    yaw_rate_setpoint_deg_s=math.degrees(action_yaw_rate),
                    height_m=float("nan") if state_decision.height_m is None else state_decision.height_m,
                    sync_ms=sync_ms,
                    direct_land=state_decision.direct_land,
                    fallback_reason="",
                    health_ctrl="px4_ekf" if self._use_body_cloud else health.control_pose_source.value,
                    health_cloud_src=("fastlio_deskewed_body" if self._use_body_cloud
                                      else health.perception_cloud_source.value),
                    health_pose_ok=("diagnostic" if self._use_body_cloud else
                                    ("ok" if health.fastlio_pose_healthy else "degraded")),
                    health_cloud_ok="ok" if health.fastlio_cloud_healthy else "degraded",
                )

                # --- DRL action log CSV ---
                self._log_drl_action(
                    step=self.step_count,
                    cloud_seq=cloud_seq,
                    pose_seq=pose_seq,
                    sync_ms=sync_ms,
                    pose_xyz=control_pose[:3],
                    yaw_rad=float(control_pose[5]),
                    roi_points=halss_stats.get("output_points", 0),
                    roi_z_min=halss_stats.get("z_min_body", float("nan")),
                    roi_z_max=halss_stats.get("z_max_body", float("nan")),
                    depth_min=float(np.min(rendered_depth)),
                    depth_mean=float(np.mean(rendered_depth)),
                    depth_max=float(np.max(rendered_depth)),
                    sem_safe_ratio=float(np.mean(sem_map == self.safe_id)),
                    sem_danger_ratio=float(np.mean(sem_map == self.danger_id)),
                    action_id=action_id,
                    action_name=action_name,
                    action_probs=rl_info.get("action_probs"),
                    v_body=v_body,
                    v_ned=v_ned,
                    v_mavros_sp=v_mavros_sp,
                )

                self._record_timing(
                    t_h1 - t_h0, t_d1 - t_d0, t_c1 - t_c0,
                    t_r1 - t_r0, t_ctrl_end - t_ctrl, t_ctrl_end - loop_start,
                    cloud_stamp_ros_s=cloud_stamp_ros_s,
                    cloud_seq=cloud_seq,
                    pose_seq=pose_seq,
                    state=state_decision.state.value,
                    sync_ms=sync_ms,
                    source_age_ms=source_age_ms,
                    result_age_ms=result_age_ms,
                    newer_frames=newer_frames,
                    accepted=True,
                    fallback_reason="",
                    perception_executed=not state_decision.direct_land,
                    pointcloud_preprocess_s=t_p1 - t_p0,
                )

                try:
                    self.visualizer.update(
                        sem_map=sem_map,
                        depth_map=rendered_depth,
                        binary_semantic_vis=binary_semantic_vis,
                    )
                except Exception:
                    pass

                self._log_frame(
                    action_id, action_name, frame_pose[5],
                    math.degrees(float(frame_pose[5])), action_yaw_rate, sync_ms,
                    v_body, v_ned, rendered_depth, valid_mask, sem_map, rl_info,
                    cloud_seq, pose_seq, state_decision, health,
                )
                if not state_decision.direct_land:
                    self._monitor_action_collapse(
                        action_id, action_name, rl_info, sparse_depth, valid_mask,
                        rendered_depth, sem_map, binary_semantic_vis, frame_pose,
                        sync_ms, v_body, v_ned, cloud_seq, pose_seq,
                        semantic_valid_mask=semantic_valid_mask,
                    )

                self.step_count += 1

                await asyncio.sleep(max(0.0, self.sim_dt - (time.perf_counter() - loop_start)))

        except KeyboardInterrupt:
            logger.info("[Pipeline] Interrupted by user.")
            self.mission_state = "ABORT"
        except Exception as e:
            logger.error("[Pipeline] Fatal error: %s", e)
            import traceback
            traceback.print_exc()
            self.mission_state = "ABORT"
            await self._emergency_stop()

    # ------------------------------------------------------------------
    # Control helpers
    # ------------------------------------------------------------------

    async def _send_zero_velocity(self, yaw_rate_rad_s: float, now: float) -> None:
        """Hold a fixed PX4 ENU XY point instead of relying on zero velocity."""
        await self._send_velocity_with_horizontal_hold(0.0, 0.0, 0.0, yaw_rate_rad_s)

    async def _send_velocity_with_horizontal_hold(
        self, vx: float, vy: float, vz: float, yaw_rate_rad_s: float
    ) -> None:
        """Use position-XY feedback whenever the requested lateral speed is zero."""
        if self.fc is None:
            return
        if math.hypot(float(vx), float(vy)) <= 1e-4:
            if self._horizontal_hold_enu_xy is None:
                pos = np.asarray(self.fc.uavPosENU, dtype=np.float32)
                self._horizontal_hold_enu_xy = (float(pos[0]), float(pos[1]))
            hold_x, hold_y = self._horizontal_hold_enu_xy
            await self.fc.send_landing_enu_yaw_rate(
                hold_x, hold_y, float(vz), float(yaw_rate_rad_s)
            )
        else:
            self._horizontal_hold_enu_xy = None
            await self.fc.send_velocity_enu_yaw_rate(
                float(vx), float(vy), float(vz), float(yaw_rate_rad_s)
            )

    async def _send_zero_velocity_logged(
        self, yaw_rate_rad_s: float, now: float,
        state: str = "", reason: str = "",
        pose_xyz=None, sync_ms=None,
        fallback_reason: str = ""
    ) -> None:
        """Send fixed-XY hold + yaw_rate with CSV logging."""
        await self._send_velocity_with_horizontal_hold(0.0, 0.0, 0.0, yaw_rate_rad_s)
        # Log the fallback as zero commanded velocity; XY position fields are
        # maintained internally by the MAVROS mixed position/velocity setpoint.
        _fc_vel = getattr(self.fc, "uavVelNED", np.zeros(3, dtype=np.float32)) if self.fc else np.zeros(3)
        _att = getattr(self.fc, "uavAngEular", np.zeros(3, dtype=np.float32)) if self.fc else np.zeros(3)
        _yaw = float(pose_xyz[5]) if pose_xyz is not None and len(pose_xyz) > 5 else float(_att[2])
        if self._use_body_cloud and self.fc is not None:
            height_pose = tuple(float(v) for v in self.fc.uavPosENU)
        elif pose_xyz is not None:
            height_pose = tuple(float(v) for v in pose_xyz[:3])
        else:
            height_pose = None
        height_m = self.state_manager.height_from_pose(height_pose)
        report = self.pose_source_mgr.get_current_report()
        self._log_velocity_command(
            step=self.step_count,
            timestamp_s=now,
            state=state,
            reason=reason,
            action_id=-1,
            action_name="XY_HOLD_FALLBACK",
            v_body=np.zeros(3, dtype=np.float32),
            v_ned=np.zeros(3, dtype=np.float32),
            v_mavros_sp=np.zeros(3, dtype=np.float32),
            fc_vel=_fc_vel,
            roll_deg=math.degrees(float(getattr(self.fc, "uavRollENU", _att[0]))),
            pitch_deg=math.degrees(float(getattr(self.fc, "uavPitchENU", _att[1]))),
            yaw_deg=math.degrees(float(getattr(self.fc, "uavYawENU", _yaw))),
            yaw_rate_setpoint_deg_s=math.degrees(yaw_rate_rad_s),
            height_m=float("nan") if height_m is None else height_m,
            sync_ms=sync_ms,
            direct_land=False,
            fallback_reason=fallback_reason,
            health_ctrl="px4_ekf" if self._use_body_cloud else report.control_pose_source.value,
            health_cloud_src=("fastlio_deskewed_body" if self._use_body_cloud
                              else report.perception_cloud_source.value),
            health_pose_ok="diagnostic" if self._use_body_cloud else (
                "ok" if report.fastlio_pose_healthy else "degraded"
            ),
            health_cloud_ok="ok" if report.fastlio_cloud_healthy else "degraded",
        )

    async def _warmup_perception(self) -> bool:
        """Warm CUDA, then verify one fresh full inference meets the age budget.

        The drone stays in position hold (last Phase 2 setpoint still active in background thread).
        The cold pass is discarded.  A second, newer cloud must complete within
        max_inference_result_age_ms or descent remains blocked.
        """
        warmup_start = time.perf_counter()
        timeout_s = 15.0
        cold_pass_cloud_seq = None

        while time.perf_counter() - warmup_start < timeout_s:
            frame_pts, frame_pose, cloud_seq, pose_seq, sync_ms = self._grab_latest_snapshot()
            if self.fc is not None and (
                getattr(self.fc, "isOffboard", None) is False
                or getattr(self.fc, "isArmed", None) is False
            ):
                print("  Warmup cancelled: vehicle is not armed in OFFBOARD", flush=True)
                return False
            if frame_pts is None or frame_pose is None:
                await asyncio.sleep(0.1)
                continue
            if sync_ms is None or sync_ms > self.max_cloud_odom_sync_ms:
                await asyncio.sleep(0.05)
                continue
            if cold_pass_cloud_seq is not None and cloud_seq <= cold_pass_cloud_seq:
                await asyncio.sleep(0.01)
                continue

            t0 = time.perf_counter()
            try:
                # Dynamic FOV ROI
                dyn_half_x, dyn_half_y, _ = self._compute_roi_half_from_height(
                    float(frame_pose[2]), None
                )
                roi_bounds = {
                    "x_min": -dyn_half_x, "x_max": dyn_half_x,
                    "y_min": -dyn_half_y, "y_max": dyn_half_y,
                }

                # HALSS
                halss_pts, projection_pts, halss_stats = self._prepare_halss_points(
                    frame_pts, frame_pose, frame_pose, dyn_half_x, dyn_half_y,
                )
                if int(halss_stats.get("output_points", 0)) < 10:
                    await asyncio.sleep(0.05)
                    continue
                halss_result = self.halss.evaluate(halss_pts, fixed_bounds=roi_bounds)

                # Semantic
                if halss_result is not None:
                    bev_data = halss_result.get("bev_data", halss_result)
                    sem_map = self.sem_gen.generate(bev_data)
                else:
                    sem_map = np.full((self.obs_h, self.obs_w), self.danger_id, dtype=np.uint8)

                # Depth
                if self._projection_mode == "training_camera":
                    sparse_depth, valid_mask, sem_map, _ = project_training_camera(
                        projection_pts, sem_map, roi_bounds, self._training_camera,
                        danger_id=self.danger_id,
                    )
                else:
                    sparse_depth, _ = project_bev_depth(
                        halss_pts,
                        grid_res=int(self.cfg["perception"].get("halss_grid_res", 64)),
                        out_size=self.obs_w, max_range=self.depth_max,
                        half_x=dyn_half_x, half_y=dyn_half_y,
                    )
                    valid_mask = (sparse_depth < self.depth_max) & (sparse_depth > 0.01)
                rendered_depth = render_sparse_depth(sparse_depth, valid_mask, self.depth_max)

                # DRL inference (the main warmup target)
                action_id, _ = self.drl.predict(rendered_depth, sem_map)
                action_name = self.decomposer.action_id_to_name(action_id)

                dt = (time.perf_counter() - t0) * 1000
                cloud_stamp = self.fastlio.points_stamp
                result_age_ms = dt
                if cloud_stamp is not None:
                    result_age_ms = max(
                        result_age_ms,
                        (self._ros_time_now_s() - float(cloud_stamp)) * 1000.0,
                    )
                if cold_pass_cloud_seq is None:
                    cold_pass_cloud_seq = cloud_seq
                    print(
                        f"  Warmup cold pass: act={action_id}({action_name}) "
                        f"dt={dt:.0f}ms; validating on a fresh frame...",
                        flush=True,
                    )
                    await asyncio.sleep(0.01)
                    continue
                if result_age_ms > self.max_inference_result_age_ms:
                    print(
                        f"  Warmup rejected: dt={dt:.0f}ms result_age={result_age_ms:.0f}ms exceeds "
                        f"max_inference_result_age_ms={self.max_inference_result_age_ms:.0f}ms",
                        flush=True,
                    )
                    return False
                print(
                    f"  Warmup validated: act={action_id}({action_name}) "
                    f"dt={dt:.0f}ms result_age={result_age_ms:.0f}ms "
                    f"<= {self.max_inference_result_age_ms:.0f}ms",
                    flush=True,
                )
                return True
            except Exception as e:
                print(f"  Warmup error: {e}, retrying...", flush=True)
                await asyncio.sleep(0.2)

        print(f"  Warmup timeout after {timeout_s}s — requesting manual takeover", flush=True)
        return False

    async def _vertical_takeoff(self, target_altitude_m: float) -> bool:
        """Move from the manual handoff point to launch-origin staging altitude.

        1a. Velocity climb to target altitude
        1b. Position hold + fixed-yaw stabilization
        """
        home_enu = self._home_enu
        target_z = float(home_enu[2]) + target_altitude_m
        # Do not reuse the policy's 10 m/s DESCEND action magnitude for takeoff.
        # Mission guidance has its own conservative vertical-speed setting.
        climb_speed = float(
            self.cfg.get("global_prior", {}).get("takeoff_speed_z_ms", 1.0)
        )
        climb_timeout_s = float(self._goto_max_time_s)

        logger.info("[Takeoff] Phase 1: climbing to %.1fm @ %.1f m/s (timeout=%.0fs)...",
                   target_altitude_m, climb_speed, climb_timeout_s)
        climb_start = time.perf_counter()
        last_log = 0.0

        # 1a. Proportional 3-D velocity, returning over the ground launch point.
        while True:
            current = np.asarray(self.fc.uavPosENU, dtype=np.float32)
            dx = float(home_enu[0] - current[0])
            dy = float(home_enu[1] - current[1])
            dz = float(target_z - current[2])
            dist_xy = math.hypot(dx, dy)
            if dist_xy <= self._goto_tolerance_xy and abs(dz) <= self._goto_tolerance_z:
                logger.info("[Takeoff] Reached target altitude %.2fm in %.1fs",
                           current[2] - float(home_enu[2]), time.perf_counter() - climb_start)
                break
            if time.perf_counter() - climb_start > climb_timeout_s:
                logger.error("[Takeoff] Timeout %.0fs at height %.2fm XY error %.2fm",
                             climb_timeout_s, current[2] - float(home_enu[2]), dist_xy)
                break
            vx, vy = _limited_xy_velocity(
                dx, dy, self._goto_max_horizontal_speed_mps or 2.0,
                self._goto_horizontal_kp_s,
            )
            vz = _limited_axis_velocity(dz, min(climb_speed, self._goto_max_vertical_speed_mps))
            await self.fc.send_velocity_enu_yaw(vx, vy, vz, self._pre_goto_yaw_enu_deg)
            now = time.perf_counter()
            if now - last_log >= 1.0:
                logger.info("[Takeoff] height=%.2fm target=%.1fm xy_error=%.2fm v=(%.2f,%.2f,%.2f) elapsed=%.1fs",
                           current[2] - float(home_enu[2]), target_altitude_m, dist_xy,
                           vx, vy, vz, now - climb_start)
                last_log = now
            await asyncio.sleep(0.05)

        # 1b. Position hold + fixed-yaw stabilization
        logger.info("[Takeoff] Stabilizing at %.1fm...", target_altitude_m)
        for _ in range(20):  # ~2s
            await self.fc.send_position_enu_yaw(
                float(home_enu[0]), float(home_enu[1]), target_z,
                self._pre_goto_yaw_enu_deg,
            )
            await asyncio.sleep(0.1)
        logger.info("[Takeoff] Phase 1 complete.")
        final = np.asarray(
            self.fc.uavPosENU if hasattr(self.fc, "uavPosENU") else np.full(3, np.nan),
            dtype=np.float64,
        )
        final_xy_error = float(np.linalg.norm(final[:2] - np.asarray(home_enu[:2])))
        final_z_error = abs(float(final[2]) - target_z)
        return bool(
            np.isfinite(final).all()
            and final_xy_error <= self._goto_tolerance_xy
            and final_z_error <= self._goto_tolerance_z
        )

    async def _goto_safe_point(self):
        """Phase 2: Move to global safe point with position + yaw_rate control.

        Outdoor profile uses capped ENU velocity XY plus position Z, so the
        horizontal setpoint cannot exceed the configured limit. Profiles with
        no limit retain the legacy full-position setpoint.
        """
        # Convert safe NED target to ENU
        if self._indoor_local_mode:
            target_ned = self._safe_ned_target
        else:
            if self._safe_ned is None:
                self._safe_ned = self._gps_to_ned_offset_3d(
                    self._safe_lat, self._safe_lon, self._safe_altitude_m,
                )
            target_ned = self._safe_ned

        # NED → ENU: (north, east, down) → (east, north, up)
        target_enu_x = float(target_ned[1])  # NED east → ENU x
        target_enu_y = float(target_ned[0])  # NED north → ENU y
        target_enu_z = float(-target_ned[2])  # NED down → ENU up (flip sign)

        speed_text = (
            "PX4 position control"
            if self._goto_max_horizontal_speed_mps is None
            else f"capped velocity XY <= {self._goto_max_horizontal_speed_mps:.2f}m/s"
        )
        logger.info(
            "[GOTO_SAFE] Phase 2: ENU target (%.2f, %.2f, %.2f) "
            "tol_xy=%.2fm tol_z=%.2fm mode=%s",
            target_enu_x, target_enu_y, target_enu_z,
            self._goto_tolerance_xy, self._goto_tolerance_z, speed_text,
        )

        # GOTO_SAFE is the deliberate yaw-fault start boundary. Seed the
        # heartbeat before checking arrival.
        current_enu = self.fc.uavPosENU if hasattr(self.fc, "uavPosENU") else np.zeros(3)
        goto_vx = goto_vy = goto_vz = 0.0
        if self._goto_max_horizontal_speed_mps is None:
            await self.fc.send_position_enu_yaw_rate(
                target_enu_x, target_enu_y, target_enu_z, self.yaw_rate_cmd
            )
        else:
            goto_vx, goto_vy = _limited_xy_velocity(
                target_enu_x - float(current_enu[0]),
                target_enu_y - float(current_enu[1]),
                self._goto_max_horizontal_speed_mps,
                self._goto_horizontal_kp_s,
            )
            goto_vz = _limited_axis_velocity(
                target_enu_z - float(current_enu[2]), self._goto_max_vertical_speed_mps,
                self._goto_horizontal_kp_s,
            )
            await self.fc.send_velocity_enu_yaw_rate(goto_vx, goto_vy, goto_vz, self.yaw_rate_cmd)

        goto_start = time.perf_counter()
        while self.mission_state == "GOTO_SAFE":
            current_enu = self.fc.uavPosENU if hasattr(self.fc, 'uavPosENU') else np.zeros(3)
            dist_xy = math.sqrt(
                (target_enu_x - current_enu[0])**2 + (target_enu_y - current_enu[1])**2
            )
            dist_z = abs(target_enu_z - float(current_enu[2]))

            if self.step_count % 5 == 0:  # log every 0.5s
                print(f"\r  GOTO: dist_xy={dist_xy:.2f}m dist_z={dist_z:.2f}m cur=({current_enu[0]:.1f},{current_enu[1]:.1f},{current_enu[2]:.2f})", end="", flush=True)

            if dist_xy <= self._goto_tolerance_xy and dist_z <= self._goto_tolerance_z:
                elapsed = time.perf_counter() - goto_start
                logger.info(
                    "[GOTO_SAFE] Arrived. XY error=%.2fm Z error=%.2fm time=%.1fs",
                    dist_xy, dist_z, elapsed,
                )
                # Replace the last velocity setpoint immediately. GPU warmup
                # follows GOTO and must hold the reached safe point.
                await self.fc.send_position_enu_yaw_rate(
                    target_enu_x, target_enu_y, target_enu_z, self.yaw_rate_cmd
                )
                self._apply_state_decision(self.state_manager.mark_goto_arrived(
                    reason="goto_safe_arrived_local" if self._indoor_local_mode else "goto_safe_arrived"
                ))
                self._log_mission_event(
                    "GOTO_ARRIVED",
                    reason=f"xy_error_m={dist_xy:.3f};z_error_m={dist_z:.3f}",
                )
                break

            elapsed = time.perf_counter() - goto_start
            if elapsed > self._goto_max_time_s:
                logger.warning(
                    "[GOTO_SAFE] Timeout %.1fs > %.1fs. XY=%.2fm Z=%.2fm",
                    elapsed, self._goto_max_time_s, dist_xy, dist_z,
                )
                # Never leave a nonzero GOTO velocity active during timeout
                # handling or the subsequent warmup/direct-land transition.
                await self.fc.send_position_enu_yaw_rate(
                    float(current_enu[0]), float(current_enu[1]), float(current_enu[2]),
                    0.0,
                )
                if self._goto_timeout_action == "abort":
                    self.mission_state = "ABORT"
                    await self._emergency_stop()
                else:
                    self._apply_state_decision(self.state_manager.reset(
                        state=MissionState.HOLD_FOR_MANUAL,
                        reason="goto_timeout_wait_manual",
                    ))
                    self._log_mission_event(
                        "MANUAL_TAKEOVER_REQUESTED",
                        reason=f"goto_timeout;xy_error_m={dist_xy:.3f};z_error_m={dist_z:.3f}",
                    )
                break

            if self._goto_max_horizontal_speed_mps is None:
                await self.fc.send_position_enu_yaw_rate(
                    target_enu_x, target_enu_y, target_enu_z, self.yaw_rate_cmd
                )
            else:
                goto_vx, goto_vy = _limited_xy_velocity(
                    target_enu_x - float(current_enu[0]),
                    target_enu_y - float(current_enu[1]),
                    self._goto_max_horizontal_speed_mps,
                    self._goto_horizontal_kp_s,
                )
                goto_vz = _limited_axis_velocity(
                    target_enu_z - float(current_enu[2]), self._goto_max_vertical_speed_mps,
                    self._goto_horizontal_kp_s,
                )
                await self.fc.send_velocity_enu_yaw_rate(
                    goto_vx, goto_vy, goto_vz, self.yaw_rate_cmd
                )

            if self.step_count % 10 == 0:
                logger.info(
                    "[GOTO_SAFE] XY=%.2fm Z=%.2fm target=(%.2f,%.2f,%.2f) "
                    "vel_sp=(%.2f,%.2f,%.2f)m/s yaw_rate=%.2f elapsed=%.1fs",
                    dist_xy, dist_z, target_enu_x, target_enu_y, target_enu_z,
                    goto_vx, goto_vy, goto_vz, self.yaw_rate_cmd, elapsed,
                )
            self.step_count += 1
            await asyncio.sleep(0.1)

    async def _hold_for_manual_takeover(self):
        """Hold the current PX4 pose until the pilot exits OFFBOARD or disarms."""
        hold = np.asarray(self.fc.uavPosENU, dtype=np.float32).copy()
        yaw_deg = math.degrees(float(self.fc.uavYawENU))
        logger.warning(
            "[ManualHold] Holding ENU=%s yaw=%.1fdeg; pilot must switch out of OFFBOARD",
            _fmt_vec(hold), yaw_deg,
        )
        last_notice = 0.0
        while bool(getattr(self.fc, "isArmed", False)) and bool(getattr(self.fc, "isOffboard", False)):
            await self.fc.send_position_enu_yaw(
                float(hold[0]), float(hold[1]), float(hold[2]), yaw_deg,
            )
            now = time.perf_counter()
            if now - last_notice >= 1.0:
                logger.warning("[ManualHold] Waiting for pilot takeover; OFFBOARD remains active")
                last_notice = now
            await asyncio.sleep(0.1)
        if not bool(getattr(self.fc, "isOffboard", False)):
            self.mission_state = "IDLE"
            self._log_mission_event("MANUAL_TAKEOVER", reason="pilot_exited_offboard")
        else:
            self.mission_state = "LANDED"
            self._log_mission_event("DISARMED", reason="disarmed_during_manual_hold")

    def _compute_local_ned_target_3d(self, home_ned: np.ndarray, home_yaw_rad: float) -> np.ndarray:
        """Convert 3D body-frame offset to local NED target.

        N_target = N0 + forward*cos(yaw0) - right*sin(yaw0)
        E_target = E0 + forward*sin(yaw0) + right*cos(yaw0)
        D_target = D0 - up  (NED down为正, up向上为负)
        """
        forward_m, right_m, up_m = self._local_body_offset_m
        cos_yaw = math.cos(home_yaw_rad)
        sin_yaw = math.sin(home_yaw_rad)
        dn = forward_m * cos_yaw - right_m * sin_yaw
        de = forward_m * sin_yaw + right_m * cos_yaw
        return np.array([
            float(home_ned[0]) + dn,
            float(home_ned[1]) + de,
            float(home_ned[2]) - up_m,  # NED down为正, up向上为-d
        ], dtype=np.float32)

    def _gps_to_ned_offset_3d(self, lat: float, lon: float, alt_m: float) -> np.ndarray:
        """GPS (lat, lon) + 相对高度 → 3D NED offset."""
        if self._home_lat is None or self._home_lon is None:
            raise RuntimeError("[GlobalPrior] Home GPS is unavailable; cannot convert safe point to NED.")
        home_lat = self._home_lat
        home_lon = self._home_lon
        meters_per_deg_lat = 111320.0
        meters_per_deg_lon = 111320.0 * math.cos(math.radians(home_lat))
        dn = (lat - home_lat) * meters_per_deg_lat
        de = (lon - home_lon) * meters_per_deg_lon
        dd = self._home_ned[2] - alt_m  # NED down: alt_m 高于起飞点 → dd < home_ned[2]
        return np.array([
            float(self._home_ned[0]) + dn,
            float(self._home_ned[1]) + de,
            dd,
        ], dtype=np.float32)

    def _compute_local_ned_target(self, home_ned: np.ndarray, home_yaw_rad: float) -> np.ndarray:
        """[DEPRECATED] 2D body-frame offset → NED target. Use _compute_local_ned_target_3d instead."""
        return self._compute_local_ned_target_3d(home_ned, home_yaw_rad)

    def _gps_to_ned_offset(self, lat: float, lon: float) -> np.ndarray:
        """[DEPRECATED] Use _gps_to_ned_offset_3d instead."""
        return self._gps_to_ned_offset_3d(lat, lon, self._safe_altitude_m or 0.0)

    def _gps_to_ned_distance(self, lat: float, lon: float) -> float:
        target = self._gps_to_ned_offset_3d(lat, lon, self._safe_altitude_m or 0.0)
        return float(np.linalg.norm(target[:2]))

    # ------------------------------------------------------------------
    # Data/logging
    # ------------------------------------------------------------------

    def _grab_latest_snapshot(self):
        """Return one immutable FAST-LIO cloud/pose snapshot for one control decision."""
        pts = self.fastlio.points
        cloud_seq = self.fastlio.points_seq
        pose_seq = self.fastlio.pose_seq
        if pts is None:
            return None, None, -1, -1, None
        if self._use_body_cloud:
            if self.fc is None or self.fastlio.points_stamp is None:
                return pts.copy(), None, cloud_seq, pose_seq, None
            sample = self.fc.get_odom_nearest(
                self.fastlio.points_stamp,
                max_delta_ms=self.max_cloud_odom_sync_ms,
            )
            if sample is None:
                return pts.copy(), None, cloud_seq, pose_seq, None
            pos = sample["position_enu"]
            pose = np.array([
                pos[0], pos[1], pos[2],
                sample["roll"], sample["pitch"],
                # Action decomposition explicitly uses PX4 NED yaw later; this
                # field remains ENU yaw for logging and local geometry only.
                sample["yaw_enu"],
            ], dtype=np.float32)
            return pts.copy(), pose, cloud_seq, pose_seq, float(sample["sync_ms"])
        pose = self.fastlio.pose
        sync_ms = self.fastlio.sync_delta_ms
        if pose is None:
            return None, None, -1, -1, None
        return pts.copy(), pose.copy(), cloud_seq, pose_seq, sync_ms

    def _record_timing(
        self, halss, depth, completion, rl, control, total, *,
        cloud_stamp_ros_s=None, cloud_seq=-1, pose_seq=-1, state="",
        sync_ms=None, source_age_ms=None, result_age_ms=None, newer_frames=0,
        accepted=True, fallback_reason="", perception_executed=False,
        pointcloud_preprocess_s=0.0,
    ):
        self._timing["halss"].append(halss * 1000.0)
        self._timing["depth"].append(depth * 1000.0)
        self._timing["completion"].append(completion * 1000.0)
        self._timing["rl"].append(rl * 1000.0)
        self._timing["control"].append(control * 1000.0)
        self._timing["total"].append(total * 1000.0)
        if perception_executed:
            self._log_frame_timing(
                cloud_stamp_ros_s=cloud_stamp_ros_s,
                cloud_seq=cloud_seq,
                pose_seq=pose_seq,
                state=state,
                sync_ms=sync_ms,
                pointcloud_preprocess_ms=pointcloud_preprocess_s * 1000.0,
                halss_ms=halss * 1000.0,
                depth_ms=depth * 1000.0,
                completion_ms=completion * 1000.0,
                onnx_ms=rl * 1000.0,
                control_ms=control * 1000.0,
                pipeline_total_ms=total * 1000.0,
                source_age_ms=source_age_ms,
                result_age_ms=result_age_ms,
                newer_frames=newer_frames,
                accepted=accepted,
                fallback_reason=fallback_reason,
            )

    def _apply_state_decision(self, decision):
        self.mission_state = decision.state.value
        if decision.transition:
            logger.info(
                "[State] %s -> %s reason=%s height=%s ground_p05=%s",
                decision.previous_state.value,
                decision.state.value,
                decision.reason,
                "n/a" if decision.height_m is None else f"{decision.height_m:.2f}m",
                (
                    "n/a"
                    if decision.ground_clearance_p05_m is None
                    else f"{decision.ground_clearance_p05_m:.2f}m"
                ),
            )
            if decision.state.value == "DRL_DESCENT":
                self._log_mission_event("DRL_DESCENT_STARTED", reason=decision.reason)
            elif decision.state.value == "DIRECT_LAND":
                self._log_mission_event("DIRECT_LAND_STARTED", reason=decision.reason)

    def _prepare_halss_points(self, frame_points, frame_pose, control_pose,
                              half_x: float, half_y: float):
        """Prepare bounded HALSS input while retaining full points for depth.

        Outdoor body-cloud mode first performs the existing rigid transform and
        level-body ROI, then optionally keeps the nearest observed point in each
        fixed pinhole ray.  When pinhole-ray sampling is enabled, both HALSS
        and Depth receive the same ray-sampled point set (shared sampling).
        Indoor world-cloud behavior is unchanged.
        """
        if not self._use_body_cloud:
            points, stats = world_to_level_body_roi(
                frame_points, frame_pose[:3], frame_pose[3:],
                self.cfg["perception"], half_x=half_x, half_y=half_y,
            )
            return points, points, stats

        projection_points, stats = body_cloud_to_level_body_roi(
            frame_points,
            float(control_pose[3]), float(control_pose[4]),
            self.cfg["perception"], half_x=half_x, half_y=half_y,
        )
        if not self._halss_pinhole_ray_sampling:
            return projection_points, projection_points, stats

        # Shared nearest-point-per-ray sampling: one call, both HALSS
        # and depth projection consume the identical sparse point set.
        sampled_points, ray_stats = sample_nearest_points_by_camera_rays(
            projection_points,
            self._training_camera,
            ray_width=self._halss_ray_grid_res,
            ray_height=self._halss_ray_grid_res,
        )
        stats["pre_ray_points"] = int(len(projection_points))
        stats["frustum_points"] = int(ray_stats["frustum_points"])
        stats["ray_grid"] = [self._halss_ray_grid_res, self._halss_ray_grid_res]
        stats["output_points"] = int(len(sampled_points))
        return sampled_points, sampled_points, stats

    def _roi_bounds(self) -> dict:
        """Return the current (possibly dynamic) ROI bounds for HALSS / depth."""
        return {
            "x_min": -self._roi_half_x, "x_max": self._roi_half_x,
            "y_min": -self._roi_half_y, "y_max": self._roi_half_y,
        }

    def _compute_roi_half_from_height(self, pose_z_world: float,
                                       halss_points: np.ndarray = None) -> tuple:
        """Compute dynamic ROI half-extent from current height above ground.

        training_camera uses the original renderer's rectangular FOV. Legacy
        BEV uses half = H * tan(fov_half).
        Returns (half_x, half_y, height_m).
        """
        if not self._roi_dynamic:
            return self._roi_half_x, self._roi_half_y, float("nan")

        if self._roi_height_source == "pointcloud_median":
            if halss_points is not None and len(halss_points) > 10:
                H = float(np.median(np.asarray(halss_points, dtype=np.float32)[:, 2]))
            else:
                H = abs(float(pose_z_world) - self._ground_z_world)
        else:  # "pose_z"
            H = abs(float(pose_z_world) - self._ground_z_world)

        H = max(H, 0.1)
        if self._projection_mode == "training_camera":
            half_x, half_y = self._training_camera.ground_half_extents(H)
            half_x = max(self._roi_min_half_m, min(self._roi_max_half_m, half_x))
            half_y = max(self._roi_min_half_m, min(self._roi_max_half_m, half_y))
            return half_x, half_y, H
        half = H * math.tan(self._roi_fov_half_rad)
        half_clamped = max(self._roi_min_half_m, min(self._roi_max_half_m, half))
        return half_clamped, half_clamped, H

    @staticmethod
    def _pointcloud_ground_clearance(points_body, percentile: float):
        if points_body is None:
            return None
        pts = np.asarray(points_body, dtype=np.float32)
        if pts.size == 0 or pts.ndim != 2 or pts.shape[1] < 3:
            return None
        z = pts[:, 2]
        z = z[np.isfinite(z) & (z > 0.01)]
        if z.size == 0:
            return None
        percentile = float(np.clip(percentile, 0.0, 100.0))
        return float(np.percentile(z, percentile))

    def _direct_land_rl_info(self, depth_map, sem_map):
        depth = np.nan_to_num(depth_map, nan=self.depth_max, posinf=self.depth_max, neginf=0.0)
        sem_safe = sem_map == self.safe_id
        depth_norm = np.clip(depth, 0.0, self.depth_max) / 255.0
        sem_norm = np.where(sem_safe, 30.0, 250.0) / 255.0
        probs = [0.0] * len(self.decomposer.action_names)
        if len(probs) > 9:
            probs[9] = 1.0
        return {
            "confidence": 1.0,
            "action_probs": probs,
            "depth_norm_min": float(np.min(depth_norm)),
            "depth_norm_mean": float(np.mean(depth_norm)),
            "depth_norm_max": float(np.max(depth_norm)),
            "sem_norm_min": float(np.min(sem_norm)),
            "sem_norm_mean": float(np.mean(sem_norm)),
            "sem_norm_max": float(np.max(sem_norm)),
            "direct_land_override": True,
        }

    def _log_frame(self, action_id, action_name, yaw_rad, yaw_deg, yaw_rate, sync_ms,
                   v_body, v_ned, dense_depth, valid_mask, sem_map, rl_info,
                   cloud_seq, pose_seq, state_decision=None, health=None):
        safe_ratio = float(np.mean(sem_map == self.safe_id))
        danger_ratio = float(np.mean(sem_map == self.danger_id))
        valid_ratio = float(np.mean(valid_mask))
        conf = rl_info.get("confidence")
        conf_str = "n/a" if conf is None else f"{conf:.2f}"
        state_text = "n/a" if state_decision is None else state_decision.state.value
        reason_text = "n/a" if state_decision is None else state_decision.reason
        height_text = float("nan") if state_decision is None or state_decision.height_m is None else state_decision.height_m
        ground_p05_text = (
            float("nan")
            if state_decision is None or state_decision.ground_clearance_p05_m is None
            else state_decision.ground_clearance_p05_m
        )
        ground_min_text = (
            float("nan")
            if state_decision is None or state_decision.ground_clearance_min_m is None
            else state_decision.ground_clearance_min_m
        )
        # 健康状态
        health_pose = "n/a"
        health_cloud = "n/a"
        health_ctrl = "n/a"
        health_cloud_src = "n/a"
        if health is not None:
            health_pose = "ok" if health.fastlio_pose_healthy else "DEGRADED"
            health_cloud = "ok" if health.fastlio_cloud_healthy else "DEGRADED"
            health_ctrl = health.control_pose_source.value
            health_cloud_src = health.perception_cloud_source.value
        logger.info(
            "[%04d] cloud_seq=%d pose_seq=%d act=%d(%s) "
            "H=%.0fms D=%.0fms C=%.0fms RL=%.0fms total=%.0fms "
            "| yaw=%.0fdeg yaw_sp=%.0fdeg yr=%.2f sync=%.0fms v_body=%s v_ned=%s "
            "| depth=%.1f/%.1f/%.1fm obsD=%.2f/%.2f/%.2f obsS=%.2f/%.2f/%.2f "
            "valid=%.2f sem_safe=%.2f sem_danger=%.2f conf=%s %s "
            "| state=%s reason=%s height=%.2fm ground_p05=%.2fm ground_min=%.2fm "
            "| health pose=%s cloud=%s ctrl=%s cloud_src=%s",
            self.step_count, cloud_seq, pose_seq, action_id, action_name,
            self._timing["halss"][-1], self._timing["depth"][-1],
            self._timing["completion"][-1], self._timing["rl"][-1],
            self._timing["total"][-1], math.degrees(yaw_rad), yaw_deg, yaw_rate, sync_ms,
            _fmt_vec(v_body), _fmt_vec(v_ned),
            float(np.min(dense_depth)), float(np.mean(dense_depth)), float(np.max(dense_depth)),
            rl_info.get("depth_norm_min", float("nan")),
            rl_info.get("depth_norm_mean", float("nan")),
            rl_info.get("depth_norm_max", float("nan")),
            rl_info.get("sem_norm_min", float("nan")),
            rl_info.get("sem_norm_mean", float("nan")),
            rl_info.get("sem_norm_max", float("nan")),
            valid_ratio, safe_ratio, danger_ratio, conf_str,
            _top_probs(rl_info.get("action_probs"), self.decomposer.action_names),
            state_text, reason_text, height_text, ground_p05_text, ground_min_text,
            health_pose, health_cloud, health_ctrl, health_cloud_src,
        )

    def _monitor_action_collapse(self, action_id, action_name, rl_info, sparse_depth,
                                 valid_mask, dense_depth, sem_map, binary_semantic_vis,
                                 pose, sync_ms, v_body, v_ned, cloud_seq, pose_seq,
                                 semantic_valid_mask=None):
        if semantic_valid_mask is None:
            semantic_valid_mask = np.ones_like(sem_map, dtype=bool)
        self.action_monitor.observe(
            self.step_count,
            action_id,
            action_name,
            rl_info,
            {
                "sparse_depth": sparse_depth.astype(np.float32),
                "valid_mask": valid_mask.astype(np.uint8),
                "semantic_valid_mask": semantic_valid_mask.astype(np.uint8),
                "dense_depth": dense_depth.astype(np.float32),
                "sem_map": sem_map.astype(np.uint8),
                "binary_semantic_vis": (
                    binary_semantic_vis.astype(np.uint8)
                    if binary_semantic_vis is not None
                    else np.zeros_like(sem_map, dtype=np.uint8)
                ),
                "pose": pose.astype(np.float32),
                "yaw_rad": np.array(pose[5], dtype=np.float32),
                "cloud_odom_sync_ms": np.array(sync_ms, dtype=np.float32),
                "cloud_seq": np.array(cloud_seq, dtype=np.int32),
                "pose_seq": np.array(pose_seq, dtype=np.int32),
                "action_id": np.array(action_id, dtype=np.int32),
                "v_body": v_body.astype(np.float32),
                "v_ned": v_ned.astype(np.float32),
            },
            action_names=self.decomposer.action_names,
        )

    def _log_timing(self):
        if not self._timing["total"]:
            return
        avg = lambda key: sum(self._timing[key][-30:]) / len(self._timing[key][-30:])
        logger.info(
            "[Timing] H=%.1fms D=%.1fms C=%.1fms RL=%.1fms CTRL=%.1fms total=%.1fms (~%.1fHz)",
            avg("halss"), avg("depth"), avg("completion"), avg("rl"), avg("control"),
            avg("total"), 1000.0 / max(avg("total"), 1e-3),
        )

    async def _shutdown(self):
        logger.info("[Pipeline] Shutting down...")
        try:
            if self.fc is not None:
                await self.fc.send_velocity_enu_yaw_rate(0.0, 0.0, 0.0, 0.0)
        except Exception:
            pass
        self.visualizer.close()
        await self._stop_recording()
        self._close_velocity_log()
        self._close_drl_action_log()
        self._close_experiment_logs()
        if self._rospy is not None:
            self._rospy.signal_shutdown("orin landing pipeline stopped")
        logger.info("[Pipeline] Shutdown complete.")

    # ------------------------------------------------------------------
    # Velocity command CSV logger
    # ------------------------------------------------------------------

    def _init_velocity_log(self):
        """Initialize velocity command CSV log. Deferred until _setup_run_dir is called."""
        pass  # actual init deferred to _setup_run_dir

    def _setup_run_dir(self):
        """Create experiment run directory and initialize log files."""
        if self._run_dir is not None:
            logger.debug("[RunDir] Reusing existing run directory %s", self._run_dir)
            return
        import csv
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        self._run_dir = Path("experiments/runs") / f"{timestamp}_orin_landing"
        self._run_dir.mkdir(parents=True, exist_ok=True)
        logger.info("[RunDir] Created %s", self._run_dir)

        # Velocity CSV
        self._vel_log_path = self._run_dir / "velocity_commands.csv"
        self._vel_log_file = open(str(self._vel_log_path), "w", newline="")
        self._vel_log_writer = csv.writer(self._vel_log_file)
        self._vel_log_writer.writerow([
            "step", "timestamp_ros_s", "state", "reason",
            "action_id", "action_name",
            "enu_x", "enu_y", "enu_z",
            "v_body_x", "v_body_y", "v_body_z",
            "v_ned_x", "v_ned_y", "v_ned_z",
            "v_mavros_x", "v_mavros_y", "v_mavros_z",
            "fc_vel_x", "fc_vel_y", "fc_vel_z",
            "roll_deg", "pitch_deg", "yaw_deg",
            "yaw_rate_cmd_rad_s", "yaw_rate_setpoint_deg_s",
            "yaw_rate_measured_rad_s", "yaw_rate_measured_source",
            "hold_enu_x", "hold_enu_y", "xy_hold_active",
            "setpoint_type_mask", "setpoint_age_ms", "setpoint_publish_rate_hz",
            "height_m",
            "sync_ms", "direct_land", "fallback_reason",
            "control_pose_source", "perception_cloud_source",
            "fastlio_pose_healthy", "fastlio_cloud_healthy",
            "fastlio_x", "fastlio_y", "fastlio_z",
            "fastlio_roll_deg", "fastlio_pitch_deg", "fastlio_yaw_deg",
            "height_source",
        ])
        self._vel_log_file.flush()

        # DRL action CSV
        self._drl_log_path = self._run_dir / "drl_action_log.csv"
        self._drl_log_file = open(str(self._drl_log_path), "w", newline="")
        self._drl_log_writer = csv.writer(self._drl_log_file)
        self._drl_log_writer.writerow([
            "step", "timestamp_ros_s", "cloud_seq", "pose_seq", "sync_ms",
            "pose_x", "pose_y", "pose_z", "yaw_rad",
            "roi_points", "roi_z_min", "roi_z_max",
            "depth_min", "depth_mean", "depth_max",
            "sem_safe_ratio", "sem_danger_ratio",
            "action_id", "action_name", "action_probs",
            "v_body_x", "v_body_y", "v_body_z",
            "v_ned_x", "v_ned_y", "v_ned_z",
            "v_mavros_x", "v_mavros_y", "v_mavros_z",
        ])
        self._drl_log_file.flush()
        logger.info("[RunDir] Logs: %s, %s", self._vel_log_path, self._drl_log_path)

        # Mission phase boundaries. ROS time is used for alignment with rosbag;
        # monotonic time is retained for robust same-process duration checks.
        self._event_log_path = self._run_dir / "mission_events.csv"
        self._event_log_file = open(str(self._event_log_path), "w", newline="")
        self._event_log_writer = csv.writer(self._event_log_file)
        self._event_log_writer.writerow([
            "event", "timestamp_ros_s", "monotonic_s", "mission_state", "reason",
            "enu_x", "enu_y", "enu_z", "fastlio_x", "fastlio_y", "fastlio_z",
            "height_m", "armed", "offboard", "px4_on_ground",
        ])
        self._event_log_file.flush()

        # One row per actual cloud perception/inference attempt. Warmup and
        # DIRECT_LAND placeholder frames are deliberately excluded.
        self._frame_timing_path = self._run_dir / "frame_timing.csv"
        self._frame_timing_file = open(str(self._frame_timing_path), "w", newline="")
        self._frame_timing_writer = csv.writer(self._frame_timing_file)
        self._frame_timing_writer.writerow([
            "timestamp_ros_s", "cloud_stamp_ros_s", "cloud_seq", "pose_seq", "state",
            "sync_ms", "pointcloud_preprocess_ms", "halss_ms", "depth_projection_ms",
            "depth_completion_ms", "onnx_ms", "perception_inference_ms", "control_ms",
            "pipeline_total_ms",
            "source_age_ms", "result_age_ms", "newer_frames", "accepted",
            "fallback_reason",
        ])
        self._frame_timing_file.flush()

        self._perception_gate_path = self._run_dir / "perception_gate_log.csv"
        self._perception_gate_file = open(str(self._perception_gate_path), "w", newline="")
        self._perception_gate_writer = csv.writer(self._perception_gate_file)
        self._perception_gate_writer.writerow([
            "timestamp_ros_s", "cloud_stamp_ros_s", "cloud_seq", "pose_seq",
            "cloud_px4_sync_ms", "cloud_points", "finite_ratio", "roi_points",
            "px4_x", "px4_y", "px4_z", "px4_roll_deg", "px4_pitch_deg", "px4_yaw_deg",
            "fastlio_x", "fastlio_y", "fastlio_z",
            "fastlio_roll_deg", "fastlio_pitch_deg", "fastlio_yaw_deg",
            "gate_stage", "accepted", "reason", "cloud_source",
        ])
        self._perception_gate_file.flush()
        logger.info(
            "[RunDir] Timeline: %s; frame timing: %s",
            self._event_log_path,
            self._frame_timing_path,
        )

        # Config snapshot
        self._save_config_snapshot()
        # Metadata
        self._save_run_metadata()

    def _save_config_snapshot(self):
        """Save a snapshot of the current config to the run directory."""
        import yaml
        try:
            snapshot_path = self._run_dir / "experiment_config_snapshot.yaml"
            with open(snapshot_path, "w", encoding="utf-8") as f:
                yaml.safe_dump(self.cfg, f, default_flow_style=False)
            logger.info("[RunDir] Config snapshot: %s", snapshot_path)
        except Exception as e:
            logger.warning("[RunDir] Config snapshot failed: %s", e)

    def _save_run_metadata(self):
        """Save run_metadata.json with model/hash/ROI/git info."""
        import json, hashlib
        metadata = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "config_path": self.config_path,
            "onnx_model_path": self.onnx_model_path,
            "onnx_model_sha256": self._file_sha256(self.onnx_model_path),
            "halss_weight_path": self.cfg.get("perception", {}).get("halss_weight_path"),
            "halss_weight_sha256": self._file_sha256(
                self.cfg.get("perception", {}).get("halss_weight_path", "")
            ),
            "roi_type": "level_body_dynamic_fov" if self._roi_dynamic else "level_body_fixed",
            "roi_fov_half_deg": math.degrees(self._roi_fov_half_rad),
            "roi_half_min_m": self._roi_min_half_m,
            "roi_half_max_m": self._roi_max_half_m,
            "dmax": self.depth_max,
            "obs_size": [self.obs_h, self.obs_w],
            "yaw_rate_rad_s": self.yaw_rate_cmd,
            "sim_dt": self.sim_dt,
            "fc_backend": self._fc_backend,
            "fc_mavros_ns": self._mavros_ns,
            "localization_mode": self.localization_mode,
            "px4_position_source": self.px4_position_source,
            "external_pose_topic": self.external_pose_topic,
            "perception_cloud_topic": self.fastlio_cloud_topic,
            "perception_cloud_frame": "lidar_imu_body" if self._use_body_cloud else "fastlio_world",
            "control_pose_source": "px4_ekf" if self._use_body_cloud else "fastlio_or_fallback",
            "height_source": self.state_manager.height_source,
            "goto_max_horizontal_speed_mps": self._goto_max_horizontal_speed_mps,
            "goto_max_vertical_speed_mps": self._goto_max_vertical_speed_mps,
            "rosbag_topics": self.cfg.get("experiment_recording", {}).get("bag_topics", []),
            "git_summary": self._git_summary(),
        }
        # Add CLI overrides from config_overrides if available
        overrides = getattr(self, '_config_overrides', None)
        if overrides:
            metadata["config_overrides"] = overrides

        try:
            meta_path = self._run_dir / "run_metadata.json"
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2, default=str)
            logger.info("[RunDir] Metadata: %s", meta_path)
        except Exception as e:
            logger.warning("[RunDir] Metadata failed: %s", e)

    @staticmethod
    def _file_sha256(path: str) -> str:
        """Return SHA256 hex digest of a file, or 'n/a' if missing."""
        import hashlib
        p = Path(path) if path else None
        if not p or not p.is_file():
            return "n/a"
        try:
            h = hashlib.sha256()
            with open(p, "rb") as f:
                for chunk in iter(lambda: f.read(65536), b""):
                    h.update(chunk)
            return h.hexdigest()
        except Exception:
            return "n/a"

    @staticmethod
    def _git_summary() -> str:
        """Return current git branch and short commit hash, or 'n/a'."""
        import subprocess
        try:
            branch = subprocess.check_output(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                cwd=Path(__file__).parent,
                stderr=subprocess.DEVNULL,
            ).decode().strip()
            commit = subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"],
                cwd=Path(__file__).parent,
                stderr=subprocess.DEVNULL,
            ).decode().strip()
            return f"{branch}@{commit}"
        except Exception:
            return "n/a"

    def _log_velocity_command(
        self, step: int, timestamp_s: float, state: str, reason: str,
        action_id: int, action_name: str,
        v_body, v_ned, v_mavros_sp, fc_vel,
        roll_deg: float, pitch_deg: float, yaw_deg: float,
        yaw_rate_setpoint_deg_s: float, height_m: float,
        sync_ms: float, direct_land: bool, fallback_reason: str = "",
        health_ctrl: str = "", health_cloud_src: str = "",
        health_pose_ok: str = "", health_cloud_ok: str = "",
    ):
        if self._vel_log_writer is None:
            return
        try:
            timestamp_s = (
                self._rospy.Time.now().to_sec()
                if self._rospy is not None else time.time()
            )
            v_body = np.asarray(v_body, dtype=np.float32).ravel()
            v_ned = np.asarray(v_ned, dtype=np.float32).ravel()
            v_mavros_sp = np.asarray(v_mavros_sp, dtype=np.float32).ravel()
            fc_vel = np.asarray(fc_vel, dtype=np.float32).ravel()
            enu_pos = np.asarray(
                getattr(self.fc, "uavPosENU", np.zeros(3)), dtype=np.float32
            ).ravel()
            yaw_rate_measured = float(getattr(self.fc, "uavYawRateENU", float("nan")))
            sp_status = self.fc.get_setpoint_status() if self.fc is not None else {}
            updated = sp_status.get("updated_monotonic_s")
            sp_age_ms = (
                float("nan") if updated is None
                else max(0.0, (time.perf_counter() - float(updated)) * 1000.0)
            )
            hold = self._horizontal_hold_enu_xy
            fastlio_pose = np.asarray(
                getattr(self.fastlio, "pose", np.full(6, np.nan)), dtype=np.float64
            ).ravel()
            fastlio_pose = np.pad(
                fastlio_pose, (0, max(0, 6 - fastlio_pose.size)), constant_values=np.nan
            )
            fastlio_rpy_deg = np.degrees(fastlio_pose[3:6])
            self._vel_log_writer.writerow([
                step, f"{timestamp_s:.6f}", state, reason,
                action_id, action_name,
                f"{enu_pos[0]:.4f}", f"{enu_pos[1]:.4f}", f"{enu_pos[2]:.4f}",
                f"{v_body[0]:.4f}", f"{v_body[1]:.4f}", f"{v_body[2]:.4f}",
                f"{v_ned[0]:.4f}", f"{v_ned[1]:.4f}", f"{v_ned[2]:.4f}",
                f"{v_mavros_sp[0]:.4f}", f"{v_mavros_sp[1]:.4f}", f"{v_mavros_sp[2]:.4f}",
                f"{fc_vel[0]:.4f}", f"{fc_vel[1]:.4f}", f"{fc_vel[2]:.4f}",
                f"{roll_deg:.2f}", f"{pitch_deg:.2f}", f"{yaw_deg:.2f}",
                f"{math.radians(yaw_rate_setpoint_deg_s):.6f}",
                f"{yaw_rate_setpoint_deg_s:.2f}",
                f"{yaw_rate_measured:.6f}", "mavros_local_odom.twist.angular.z",
                "n/a" if hold is None else f"{hold[0]:.4f}",
                "n/a" if hold is None else f"{hold[1]:.4f}",
                "0" if hold is None else "1",
                sp_status.get("type_mask", "n/a"),
                "n/a" if not np.isfinite(sp_age_ms) else f"{sp_age_ms:.1f}",
                f"{float(sp_status.get('publish_rate_hz', 0.0)):.2f}",
                f"{height_m:.3f}",
                f"{sync_ms:.1f}" if sync_ms is not None else "n/a",
                "1" if direct_land else "0",
                fallback_reason,
                health_ctrl, health_cloud_src,
                health_pose_ok, health_cloud_ok,
                *(self._csv_float(v, 4) for v in fastlio_pose[:3]),
                *(self._csv_float(v, 2) for v in fastlio_rpy_deg),
                self.state_manager.height_source,
            ])
            if step % 10 == 0:
                self._vel_log_file.flush()
        except Exception as e:
            logger.warning("[VelLog] Write error: %s", e)

    def _close_velocity_log(self):
        if self._vel_log_file is not None:
            try:
                self._vel_log_file.flush()
                self._vel_log_file.close()
                logger.info("[VelLog] Closed %s", self._vel_log_path)
            except Exception:
                pass
            self._vel_log_file = None
            self._vel_log_writer = None

    # ------------------------------------------------------------------
    # DRL action log CSV
    # ------------------------------------------------------------------

    def _log_drl_action(
        self, step: int, cloud_seq: int, pose_seq: int, sync_ms: float,
        pose_xyz, yaw_rad: float,
        roi_points: int, roi_z_min: float, roi_z_max: float,
        depth_min: float, depth_mean: float, depth_max: float,
        sem_safe_ratio: float, sem_danger_ratio: float,
        action_id: int, action_name: str, action_probs,
        v_body, v_ned, v_mavros_sp,
    ):
        if self._drl_log_writer is None:
            return
        try:
            v_body = np.asarray(v_body, dtype=np.float32).ravel()
            v_ned = np.asarray(v_ned, dtype=np.float32).ravel()
            v_mavros_sp = np.asarray(v_mavros_sp, dtype=np.float32).ravel()
            probs_str = ",".join(f"{p:.4f}" for p in (action_probs or []))
            self._drl_log_writer.writerow([
                step,
                f"{(self._rospy.Time.now().to_sec() if self._rospy is not None else time.time()):.6f}",
                cloud_seq, pose_seq,
                f"{sync_ms:.1f}" if sync_ms is not None else "n/a",
                f"{pose_xyz[0]:.3f}", f"{pose_xyz[1]:.3f}", f"{pose_xyz[2]:.3f}",
                f"{yaw_rad:.4f}",
                roi_points,
                f"{roi_z_min:.3f}" if np.isfinite(roi_z_min) else "n/a",
                f"{roi_z_max:.3f}" if np.isfinite(roi_z_max) else "n/a",
                f"{depth_min:.3f}", f"{depth_mean:.3f}", f"{depth_max:.3f}",
                f"{sem_safe_ratio:.4f}", f"{sem_danger_ratio:.4f}",
                action_id, action_name, probs_str,
                f"{v_body[0]:.4f}", f"{v_body[1]:.4f}", f"{v_body[2]:.4f}",
                f"{v_ned[0]:.4f}", f"{v_ned[1]:.4f}", f"{v_ned[2]:.4f}",
                f"{v_mavros_sp[0]:.4f}", f"{v_mavros_sp[1]:.4f}", f"{v_mavros_sp[2]:.4f}",
            ])
            if step % 10 == 0:
                self._drl_log_file.flush()
        except Exception as e:
            logger.warning("[DrlLog] Write error: %s", e)

    def _close_drl_action_log(self):
        if self._drl_log_file is not None:
            try:
                self._drl_log_file.flush()
                self._drl_log_file.close()
                logger.info("[DrlLog] Closed %s", self._drl_log_path)
            except Exception:
                pass
            self._drl_log_file = None
            self._drl_log_writer = None

    def _ros_time_now_s(self) -> float:
        return (
            self._rospy.Time.now().to_sec()
            if self._rospy is not None else time.time()
        )

    @staticmethod
    def _csv_float(value, digits: int = 3):
        try:
            number = float(value)
        except (TypeError, ValueError):
            return "n/a"
        return f"{number:.{digits}f}" if math.isfinite(number) else "n/a"

    def _log_mission_event(self, event: str, reason: str = "") -> bool:
        """Write one deduplicated mission boundary aligned to ROS/rosbag time."""
        if self._event_log_writer is None or event in self._recorded_mission_events:
            return False
        fc = self.fc
        enu = np.asarray(
            getattr(fc, "uavPosENU", np.full(3, np.nan)), dtype=np.float64
        ).ravel()
        fastlio_pose = np.asarray(
            getattr(self.fastlio, "pose", np.full(6, np.nan)), dtype=np.float64
        ).ravel()
        enu = np.pad(enu, (0, max(0, 3 - enu.size)), constant_values=np.nan)
        fastlio_pose = np.pad(
            fastlio_pose, (0, max(0, 3 - fastlio_pose.size)), constant_values=np.nan
        )
        height_pose = enu if self._use_body_cloud else fastlio_pose[:3]
        height_m = self.state_manager.height_from_pose(tuple(height_pose))
        self._event_log_writer.writerow([
            event,
            f"{self._ros_time_now_s():.6f}",
            f"{time.perf_counter():.6f}",
            self.mission_state,
            reason,
            *(self._csv_float(v, 4) for v in enu[:3]),
            *(self._csv_float(v, 4) for v in fastlio_pose[:3]),
            self._csv_float(height_m, 3),
            "1" if bool(getattr(fc, "isArmed", False)) else "0",
            "1" if bool(getattr(fc, "isOffboard", False)) else "0",
            "1" if bool(getattr(fc, "landed_state_on_ground", False)) else "0",
        ])
        self._event_log_file.flush()
        self._recorded_mission_events.add(event)
        logger.info("[MissionEvent] %s state=%s reason=%s", event, self.mission_state, reason)
        return True

    def _log_frame_timing(
        self, *, cloud_stamp_ros_s, cloud_seq, pose_seq, state, sync_ms,
        pointcloud_preprocess_ms, halss_ms, depth_ms, completion_ms, onnx_ms, control_ms,
        pipeline_total_ms, source_age_ms, result_age_ms, newer_frames,
        accepted, fallback_reason,
    ) -> None:
        if self._frame_timing_writer is None:
            return
        perception_inference_ms = (
            pointcloud_preprocess_ms + halss_ms + depth_ms + completion_ms + onnx_ms
        )
        self._frame_timing_writer.writerow([
            f"{self._ros_time_now_s():.6f}",
            self._csv_float(cloud_stamp_ros_s, 6),
            cloud_seq,
            pose_seq,
            state,
            self._csv_float(sync_ms, 3),
            self._csv_float(pointcloud_preprocess_ms, 3),
            self._csv_float(halss_ms, 3),
            self._csv_float(depth_ms, 3),
            self._csv_float(completion_ms, 3),
            self._csv_float(onnx_ms, 3),
            self._csv_float(perception_inference_ms, 3),
            self._csv_float(control_ms, 3),
            self._csv_float(pipeline_total_ms, 3),
            self._csv_float(source_age_ms, 3),
            self._csv_float(result_age_ms, 3),
            int(newer_frames),
            "1" if accepted else "0",
            fallback_reason,
        ])
        if int(cloud_seq) % 10 == 0:
            self._frame_timing_file.flush()

    def _log_perception_gate(
        self, cloud_seq, pose_seq, sync_ms, px4_pose,
        stage, accepted, reason, points, stats=None,
    ) -> None:
        if self._perception_gate_writer is None:
            return
        stats = stats or {}
        pts = np.asarray(points, dtype=np.float32) if points is not None else np.empty((0, 3))
        finite_ratio = (
            float(np.mean(np.isfinite(pts[:, :3]).all(axis=1)))
            if pts.ndim == 2 and pts.shape[0] and pts.shape[1] >= 3 else 0.0
        )
        px4 = np.asarray(px4_pose, dtype=np.float64).ravel()
        px4 = np.pad(px4, (0, max(0, 6 - px4.size)), constant_values=np.nan)
        fastlio = np.asarray(
            getattr(self.fastlio, "pose", np.full(6, np.nan)), dtype=np.float64
        ).ravel()
        fastlio = np.pad(fastlio, (0, max(0, 6 - fastlio.size)), constant_values=np.nan)
        self._perception_gate_writer.writerow([
            self._csv_float(self._ros_time_now_s(), 6),
            self._csv_float(self.fastlio.points_stamp, 6),
            int(cloud_seq), int(pose_seq), self._csv_float(sync_ms, 3),
            int(len(pts)), self._csv_float(stats.get("finite_ratio", finite_ratio), 4),
            int(stats.get("output_points", 0)),
            *(self._csv_float(v, 4) for v in px4[:3]),
            *(self._csv_float(v, 2) for v in np.degrees(px4[3:6])),
            *(self._csv_float(v, 4) for v in fastlio[:3]),
            *(self._csv_float(v, 2) for v in np.degrees(fastlio[3:6])),
            stage, "1" if accepted else "0", reason,
            "fastlio_deskewed_body" if self._use_body_cloud else "fastlio_world",
        ])
        if int(cloud_seq) % 10 == 0:
            self._perception_gate_file.flush()

    def _close_experiment_logs(self) -> None:
        for kind in ("event_log", "frame_timing", "perception_gate"):
            file_obj = getattr(self, f"_{kind}_file", None)
            if file_obj is not None:
                try:
                    file_obj.flush()
                    file_obj.close()
                except Exception:
                    pass
                setattr(self, f"_{kind}_file", None)
                setattr(self, f"_{kind}_writer", None)

    # ------------------------------------------------------------------
    # Rosbag recording
    # ------------------------------------------------------------------

    async def _start_recording(self):
        """Start rosbag recording if enabled."""
        if not self._record_bag:
            return False
        try:
            bag_path = self._run_dir / "input.bag"
            topics = list(self.cfg.get("experiment_recording", {}).get(
                "bag_topics", [self.fastlio_cloud_topic, self.fastlio_odom_topic]
            ))
            cmd = ["rosbag", "record", "-O", str(bag_path)] + topics
            logger.info("[Rosbag] Starting: %s", " ".join(cmd))
            self._bag_process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.PIPE,
            )
            # Give rosbag a moment to start
            await asyncio.sleep(1.0)
            if self._bag_process.returncode is not None:
                stderr = await self._bag_process.stderr.read()
                logger.error("[Rosbag] Failed to start: %s", stderr.decode() if stderr else "unknown")
                self._bag_process = None
                return False
            else:
                logger.info("[Rosbag] Recording to %s", bag_path)
                return True
        except FileNotFoundError:
            logger.warning("[Rosbag] rosbag command not found; skipping recording")
            self._bag_process = None
            return False
        except Exception as e:
            logger.warning("[Rosbag] Failed to start recording: %s", e)
            self._bag_process = None
            return False

    async def _stop_recording(self):
        """Stop rosbag recording if active."""
        if self._bag_process is None:
            return
        try:
            logger.info("[Rosbag] Stopping recording...")
            self._bag_process.terminate()
            try:
                await asyncio.wait_for(self._bag_process.wait(), timeout=5.0)
            except asyncio.TimeoutError:
                self._bag_process.kill()
                await self._bag_process.wait()
            logger.info("[Rosbag] Recording stopped.")
        except Exception as e:
            logger.warning("[Rosbag] Error stopping recording: %s", e)
        finally:
            self._bag_process = None

    def enable_recording(self, enabled: bool = True):
        """Enable/disable rosbag recording (call before run())."""
        self._record_bag = enabled
        if enabled:
            topics = self.cfg.get("experiment_recording", {}).get("bag_topics", [])
            logger.info("[Rosbag] Recording enabled: %d configured topics", len(topics))

    async def _emergency_stop(self):
        logger.error("[Pipeline] EMERGENCY STOP!")
        try:
            if self.fc is not None:
                await self.fc.send_velocity_enu_yaw_rate(0.0, 0.0, 0.0, 0.0)
            if self.fc is not None:
                await self.fc.disarm()
        except Exception:
            pass


def main():
    parser = argparse.ArgumentParser(description="Orin Landing Pipeline")
    parser.add_argument("--config", type=str, default="./config/experiment_config.yaml")
    parser.add_argument("--mode", type=str, default="ros", choices=["ros", "replay"])
    parser.add_argument("--safe-point", type=str, default=None,
                        help="Override target GPS: lat,lon")
    parser.add_argument("--safe-point-source", type=str, default="manual",
                        choices=["manual", "gis"],
                        help="Source of --safe-point. Strict flight gates require 'gis'.")
    parser.add_argument("--gis-image", type=str, default=None,
                        help="GIS satellite image path")
    parser.add_argument("--gis-mask", type=str, default=None,
                        help="Precomputed GIS semantic mask path")
    parser.add_argument("--gis-bounds", type=str, default=None,
                        help="lon_left,lat_bottom,lon_right,lat_top")
    parser.add_argument("--yaw-rate-rad-s", type=float, default=None,
                        help="Override uav.yaw_rate_rad_s for this experiment run")
    parser.add_argument("--allow-high-yaw-rate-test", action="store_true",
                        help="Explicitly allow body-cloud yaw-rate tests above 1 and up to 5rad/s")
    parser.add_argument("--onnx-model", type=str, default=None,
                        help="ONNX DRL policy path (default: decision.onnx_model_path)")
    parser.add_argument("--dmax", type=float, default=None,
                        help="Override depth_projection.max_range for BEV NN-fill depth and ONNX inference")
    parser.add_argument("--flight-ready-check-only", action="store_true",
                        help="Evaluate strict gates with CLI overrides, then exit before model/ROS init")
    parser.add_argument("--allow-incomplete-experiment", action="store_true",
                        help="Bypass strict flight-ready gates for bench debugging only")
    parser.add_argument("--record-bag", action="store_true", default=None,
                        help="Enable configured rosbag recording (FAST-LIO + MAVROS localization)")
    parser.add_argument("--no-record-bag", action="store_true", default=None,
                        help="Disable rosbag recording")
    args = parser.parse_args()
    config_overrides = _build_config_overrides(args)

    if args.flight_ready_check_only and args.allow_incomplete_experiment:
        logger.error("[FlightReady] --flight-ready-check-only cannot be combined with --allow-incomplete-experiment")
        sys.exit(2)

    if args.mode == "ros":
        if args.allow_incomplete_experiment:
            logger.warning("[FlightReady] Strict gates bypassed before model initialization")
        else:
            try:
                cfg_preview = _load_config(args.config)
                cfg_preview = _merge_config_overrides(cfg_preview, dict(config_overrides))
                override_ready, override_failures = _validate_global_guidance_override(args)
                failures = flight_ready_failures(
                    cfg_preview,
                    global_guidance_ready=(
                        True if override_ready is True and not override_failures else None
                    ),
                )
                failures = override_failures + failures
            except Exception as exc:
                logger.error("[FlightReady] Config preview failed: %s", exc)
                sys.exit(2)
            if failures:
                for failure in failures:
                    logger.error("[FlightReady] %s", failure)
                logger.error(
                    "Flight-ready checks failed before model initialization. "
                    "Run preflight_check.py --flight-ready and fix the reported gates."
                )
                sys.exit(2)
            logger.info("[FlightReady] Preview gates passed before model initialization")

    if args.flight_ready_check_only:
        logger.info("[FlightReady] Check-only mode complete.")
        return

    pipeline = OrinLandingPipeline(
        args.config,
        config_overrides=config_overrides,
        onnx_model_path=args.onnx_model,
    )

    # --- 录包开关 ---
    record_cfg = pipeline.cfg.get("experiment_recording", {})
    if args.record_bag:
        pipeline.enable_recording(True)
    elif args.no_record_bag:
        pipeline.enable_recording(False)
    elif record_cfg.get("enabled", False):
        pipeline.enable_recording(True)
    # else: default disabled

    if args.gis_image:
        pipeline.configure_global_prior_from_gis(args.gis_image, args.gis_mask, args.gis_bounds)

    if args.safe_point:
        try:
            lat, lon = _parse_safe_point(args.safe_point)
            pipeline.set_safe_point(lat, lon, source=f"CLI:{args.safe_point_source}")
        except ValueError:
            logger.error("[Main] Invalid --safe-point format: %s", args.safe_point)
            sys.exit(1)

    if args.mode == "ros":
        if args.allow_incomplete_experiment:
            logger.warning("[FlightReady] Strict gates bypassed by --allow-incomplete-experiment")
        else:
            try:
                pipeline.enforce_flight_ready()
            except RuntimeError as exc:
                logger.error("%s", exc)
                sys.exit(2)
        if not pipeline.init_ros_node():
            sys.exit(1)
        asyncio.run(pipeline.run())
    else:
        logger.info("[Replay] Not implemented. Use test_live_nocontrol.py for no-control ROS testing.")


if __name__ == "__main__":
    main()
