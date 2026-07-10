#!/usr/bin/env python3
"""
Orin Landing - 真机实时感知-决策-控制主管线
==========================================

链路:
  GIS 九宫格全局安全区 → MAVSDK 位置引导
  Mid360 → FAST-LIO 去畸变点云/位姿
  点云 → HALSS Bayesian 二值安全语义图
  点云 → HALSS 对齐 BEV 稀疏深度 → NN-fill 渲染深度
  [rendered_depth, binary_semantic] → ONNX PPO → 离散动作
  离散动作 + 同帧 Fast-LIO yaw → NED 速度 + yaw setpoint → PX4
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
    global MAVSDKController, MAVROSController, PoseSourceManager
    global ActionDecomposer, ActionCollapseMonitor
    global FastLIOInterface, HALSSBayesianEvaluator, world_to_level_body_roi
    global MissionStateManager, StateInputs
    global SemanticGenerator, GlobalSafetyPrior
    global RealtimeVisualizer, _RUNTIME_DEPS_READY
    global MissionState

    if _RUNTIME_DEPS_READY:
        return

    import numpy as _np
    import cv2 as _cv2
    import onnxruntime as _ort
    from control import MAVSDKController as _MAVSDKController
    from control import MAVROSController as _MAVROSController
    from control import PoseSourceManager as _PoseSourceManager
    from control.action_decomposer import ActionDecomposer as _ActionDecomposer
    from diagnostics.action_monitor import ActionCollapseMonitor as _ActionCollapseMonitor
    from odometry import FastLIOInterface as _FastLIOInterface
    from perception.halss_bayesian import HALSSBayesianEvaluator as _HALSSBayesianEvaluator
    from perception.halss_preprocess import world_to_level_body_roi as _world_to_level_body_roi
    from control.mission_state_manager import MissionStateManager as _MissionStateManager
    from control.mission_state_manager import StateInputs as _StateInputs
    from control.mission_state_manager import MissionState as _MissionState
    from perception.semantic_generator import SemanticGenerator as _SemanticGenerator
    from preprocessing.global_safety_prior import GlobalSafetyPrior as _GlobalSafetyPrior
    from visualization import RealtimeVisualizer as _RealtimeVisualizer

    np = _np
    cv2 = _cv2
    ort = _ort
    MAVSDKController = _MAVSDKController
    MAVROSController = _MAVROSController
    PoseSourceManager = _PoseSourceManager
    ActionDecomposer = _ActionDecomposer
    ActionCollapseMonitor = _ActionCollapseMonitor
    FastLIOInterface = _FastLIOInterface
    HALSSBayesianEvaluator = _HALSSBayesianEvaluator
    world_to_level_body_roi = _world_to_level_body_roi
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
    """ONNX PPO policy used by the live no-control pipeline."""

    def __init__(self, onnx_path: str, obs_h: int = 128, obs_w: int = 128, dmax: float = 30.0):
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

        dummy = np.zeros((1, obs_h, obs_w, 2), dtype=np.float32)
        self._forward(dummy)
        logger.info("[ONNX] model=%s input=%s shape=%s layout=%s warmup=OK",
                    onnx_path, self.input_name, in_shape, self.layout)

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
            "depth_norm_min": float(depth_ch.min()),
            "depth_norm_mean": float(depth_ch.mean()),
            "depth_norm_max": float(depth_ch.max()),
            "sem_norm_min": float(sem_ch.min()),
            "sem_norm_mean": float(sem_ch.mean()),
            "sem_norm_max": float(sem_ch.max()),
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
            _parse_safe_point(args.safe_point)
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
    try:
        import yaml
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except ModuleNotFoundError as exc:
        if exc.name != "yaml":
            raise
        from preflight_check import _load_simple_yaml
        return _load_simple_yaml(Path(path))


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
    if getattr(args, "depth_output_scale", None) is not None:
        overrides.setdefault("depth_completion", {})["output_scale"] = float(args.depth_output_scale)
    if getattr(args, "dmax", None) is not None:
        overrides.setdefault("depth_projection", {})["max_range"] = float(args.dmax)
        overrides.setdefault("zmq_pipeline", {}).setdefault("drl_control", {})["dmax"] = float(args.dmax)
    if getattr(args, "onnx_model", None):
        overrides.setdefault("zmq_pipeline", {}).setdefault("drl_control", {})[
            "onnx_model_path"
        ] = args.onnx_model
    return overrides


class OrinLandingPipeline:
    """Orin 真机着陆主管线。"""

    def __init__(
        self,
        config_path: str,
        mavsdk_address: str = None,
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
        self._fc_backend = fc_backend
        if fc_backend == "mavros":
            self._mavsdk_address = None
            self._mavros_ns = str(fc_cfg.get("mavros_ns", "/mavros"))
            self._setpoint_rate_hz = float(fc_cfg.get("setpoint_rate_hz", 20))
            self._offboard_warmup_s = float(fc_cfg.get("offboard_warmup_s", 2.0))
        else:
            self._mavsdk_address = mavsdk_address or self.cfg.get("uav", {}).get(
                "mavsdk_address", "udp://:14540"
            )
            self._mavros_ns = None

        obs_cfg = self.cfg["observation"]
        perc_cfg = self.cfg["perception"]
        depth_cfg = self.cfg["depth_projection"]
        uav_cfg = self.cfg["uav"]
        drl_cfg = self.cfg.get("zmq_pipeline", {}).get("drl_control", {})

        self.depth_max = float(depth_cfg.get("max_range", 30.0))
        self.obs_h = int(obs_cfg.get("img_height", 128))
        self.obs_w = int(obs_cfg.get("img_width", 128))
        self.safe_id = int(perc_cfg.get("safe_class_id", 1))
        self.danger_id = int(perc_cfg.get("danger_class_id", 9))
        self.sim_dt = float(uav_cfg.get("sim_dt", 0.25))
        self.target_altitude = float(uav_cfg.get("target_altitude", 0.5))
        self.max_steps = int(float(uav_cfg.get("max_t", 90.0)) / self.sim_dt)
        self.yaw_rate_cmd = float(uav_cfg.get("yaw_rate_rad_s", 0.0))
        runtime_cfg = self.cfg.get("runtime", {})
        self.max_frame_latency_ms = float(runtime_cfg.get("max_frame_latency_ms", 100.0))
        self.drop_slow_frames = bool(runtime_cfg.get("drop_slow_frames", True))
        self.max_cloud_odom_sync_ms = float(runtime_cfg.get("max_cloud_odom_sync_ms", 100.0))
        localization_cfg = self.cfg.get("localization", {})
        self.fastlio_odom_topic = localization_cfg.get("fastlio_odom_topic", "/Odometry")
        self.fastlio_cloud_topic = localization_cfg.get("world_cloud_topic", "/cloud_registered")
        self._yaw_setpoint_rad = None
        self._last_yaw_update_t = None

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
        )
        logger.info(
            "[Init] Perception route: HALSS + BEV NN-fill depth + ONNX DRL "
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

        logger.info("[Init] Fast-LIO interface...")
        self.fastlio = FastLIOInterface(use_ros=True)

        logger.info("[Init] Pose source manager...")
        self.pose_source_mgr = PoseSourceManager(self.cfg)

        logger.info("[Init] Visualizer...")
        self.visualizer = RealtimeVisualizer(self.cfg["visualization"])
        self.action_monitor = ActionCollapseMonitor(self.cfg.get("visualization", {}), logger)

        self.fc = None  # 统一飞控抽象 (MAVROSController 或 MAVSDKController)
        self.step_count = 0

        self._safe_lat = None
        self._safe_lon = None
        self._safe_ned = None
        self._safe_ned_target = None  # indoor local body-offset NED target (3D)
        self._home_ned = np.zeros(3, dtype=np.float32)
        self._home_lat = None
        self._home_lon = None
        self._goto_tolerance_xy = float(self.cfg.get("global_prior", {}).get("goto_tolerance_xy_m", 0.2))
        self._goto_tolerance_z = float(self.cfg.get("global_prior", {}).get("goto_tolerance_z_m", 0.15))
        self._goto_max_time_s = float(self.cfg.get("global_prior", {}).get("goto_max_time_s", 30.0))
        self._goto_timeout_action = str(self.cfg.get("global_prior", {}).get("goto_timeout_action", "direct_land")).lower()
        self._goto_speed_xy = float(self.cfg.get("global_prior", {}).get("goto_speed_xy_ms", 0.2))
        self._goto_speed_z = float(self.cfg.get("global_prior", {}).get("goto_speed_z_ms", 0.25))
        self._use_global_guidance = False
        self._indoor_local_mode = False
        self._local_body_offset_m = None
        self._safe_altitude_m = None  # GPS mode target altitude
        self._configure_global_prior_from_config()
        mission_cfg = dict(self.cfg.get("mission_state", {}))
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
        self._ground_z_world = 0.0  # set during run() after takeoff

        # --- velocity command CSV logger ---
        self._vel_log_path = None
        self._vel_log_file = None
        self._vel_log_writer = None
        self._init_velocity_log()

        # --- DRL action log CSV ---
        self._drl_log_path = None
        self._drl_log_file = None
        self._drl_log_writer = None

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
        """Fail before ROS/MAVSDK startup if full-experiment gates are not met."""
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
            self._odom_sub = rospy.Subscriber(
                self.fastlio_odom_topic, Odometry, self.fastlio.odometry_callback, queue_size=10
            )
            self._cloud_sub = rospy.Subscriber(
                self.fastlio_cloud_topic, PointCloud2, self.fastlio.pointcloud_callback, queue_size=10
            )
            logger.info(
                "[ROS1] Node created, subscribed to %s + %s.",
                self.fastlio_odom_topic,
                self.fastlio_cloud_topic,
            )
            return True
        except Exception as e:
            logger.error("[ROS1] Failed to init: %s", e)
            return False

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    async def run(self):
        # --- 创建飞控控制器 ---
        if self._fc_backend == "mavros":
            self.fc = MAVROSController(
                mavros_ns=self._mavros_ns,
                setpoint_rate_hz=self._setpoint_rate_hz,
                offboard_warmup_s=self._offboard_warmup_s,
            )
            logger.info("[Pipeline] Using MAVROS flight controller (ns=%s)", self._mavros_ns)
        else:
            self.fc = MAVSDKController(system_address=self._mavsdk_address)
            logger.info("[Pipeline] Using MAVSDK flight controller (addr=%s)", self._mavsdk_address)

        await self.fc.connect()

        logger.info("[Pipeline] Waiting for FAST-LIO data...")
        while not self.fastlio.initialized:
            await asyncio.sleep(0.02)
        logger.info("[Pipeline] FAST-LIO ready.")

        # --- 启动录包 (若启用) ---
        await self._start_recording()

        # --- 新流程: 持续发送当前位置 hold, 等待遥控器 OFFBOARD+解锁 ---
        if self._fc_backend == "mavros":
            self.fc.start_hold_stream()
            logger.info("[Pipeline] Hold stream started. Waiting for RC OFFBOARD+arm...")
            rc_ok = await self.fc.wait_for_manual_offboard_and_arm(timeout_s=120.0)
            if not rc_ok:
                logger.error("[Pipeline] RC OFFBOARD+arm timeout. Aborting.")
                await self._stop_recording()
                await self._shutdown()
                return
            logger.info("[Pipeline] RC OFFBOARD+arm confirmed. Proceeding to mission.")
        else:
            # MAVSDK: 保持原有 arm + init_offboard 流程
            await self._takeoff_legacy()

        # --- 记录 home 点 ---
        local_ned, local_attitude = await self.fc.wait_for_local_pose(timeout_s=10.0)
        self._home_ned = local_ned
        self._yaw_setpoint_rad = float(local_attitude[2])
        self._last_yaw_update_t = time.perf_counter()
        # 记录起飞点 Fast-LIO 世界 z 作为地面参考 (用于动态 FOV ROI)
        if self.fastlio.pose is not None:
            self._ground_z_world = float(self.fastlio.pose[2])
        logger.info(
            "[Pipeline] Home: ned=%s yaw=%.1fdeg ground_z=%.2f",
            _fmt_vec(self._home_ned),
            math.degrees(self._yaw_setpoint_rad),
            self._ground_z_world,
        )
        logger.info(
            "[Pipeline] Local telemetry: ned=%s yaw=%.1fdeg",
            _fmt_vec(self._home_ned),
            math.degrees(self._yaw_setpoint_rad),
        )

        if self._indoor_local_mode:
            self._safe_ned_target = self._compute_local_ned_target_3d(
                self._home_ned, self._yaw_setpoint_rad,
            )
            logger.info(
                "[Pipeline] Indoor 3D safe point: NED=%s offset=(%.2f,%.2f,%.2f)m yaw0=%.1fdeg",
                _fmt_vec(self._safe_ned_target),
                self._local_body_offset_m[0], self._local_body_offset_m[1],
                self._local_body_offset_m[2],
                math.degrees(self._yaw_setpoint_rad),
            )
            self._apply_state_decision(self.state_manager.start_after_takeoff(True))

        elif self._use_global_guidance:
            home_ned, home_lat_lon, home_attitude = await self.fc.wait_for_home(timeout_s=10.0)
            self._home_ned = home_ned
            self._home_lat, self._home_lon = home_lat_lon
            self._yaw_setpoint_rad = float(home_attitude[2])
            self._last_yaw_update_t = time.perf_counter()
            logger.info(
                "[Pipeline] Home telemetry: lat=%.7f lon=%.7f ned=%s yaw=%.1fdeg",
                self._home_lat, self._home_lon, _fmt_vec(self._home_ned),
                math.degrees(self._yaw_setpoint_rad),
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

        if self.mission_state == "GOTO_SAFE":
            await self._goto_safe_point()

        await self._run_descent_loop()
        await self._shutdown()

    async def _takeoff_legacy(self):
        """[MAVSDK only] Legacy arm + init_offboard takeoff flow."""
        logger.info("[Pipeline] Arming and entering offboard...")
        await self.fc.arm()
        await asyncio.sleep(1.0)
        await self.fc.init_offboard()
        await asyncio.sleep(0.5)

        local_ned, local_attitude = await self.fc.wait_for_local_pose(timeout_s=10.0)
        self._home_ned = local_ned
        self._yaw_setpoint_rad = float(local_attitude[2])
        self._last_yaw_update_t = time.perf_counter()
        logger.info(
            "[Pipeline] Local telemetry: ned=%s yaw=%.1fdeg",
            _fmt_vec(self._home_ned),
            math.degrees(self._yaw_setpoint_rad),
        )

        if self._indoor_local_mode:
            self._safe_ned_target = self._compute_local_ned_target_3d(
                self._home_ned, self._yaw_setpoint_rad,
            )
            logger.info(
                "[Pipeline] Indoor 3D safe point: NED=%s offset=(%.2f,%.2f,%.2f)m yaw0=%.1fdeg",
                _fmt_vec(self._safe_ned_target),
                self._local_body_offset_m[0], self._local_body_offset_m[1],
                self._local_body_offset_m[2],
                math.degrees(self._yaw_setpoint_rad),
            )
            self._apply_state_decision(self.state_manager.start_after_takeoff(True))

        elif self._use_global_guidance:
            home_ned, home_lat_lon, home_attitude = await self.fc.wait_for_home(timeout_s=10.0)
            self._home_ned = home_ned
            self._home_lat, self._home_lon = home_lat_lon
            self._yaw_setpoint_rad = float(home_attitude[2])
            self._last_yaw_update_t = time.perf_counter()
            logger.info(
                "[Pipeline] Home telemetry: lat=%.7f lon=%.7f ned=%s yaw=%.1fdeg",
                self._home_lat, self._home_lon, _fmt_vec(self._home_ned),
                math.degrees(self._yaw_setpoint_rad),
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

    async def _run_descent_loop(self):
        logger.info("[Pipeline] Starting DRL descent control loop...")
        last_processed_seq = -1
        skipped_frames = 0
        last_wait_log = 0.0

        try:
            while self.mission_state not in ("LANDED", "ABORT", "EMERGENCY_STOP"):
                loop_start = time.perf_counter()

                (
                    frame_points,
                    frame_pose,
                    cloud_seq,
                    pose_seq,
                    sync_ms,
                ) = self._grab_latest_snapshot()
                if frame_points is None or frame_pose is None:
                    now = time.perf_counter()
                    if now - last_wait_log > 3.0:
                        logger.info(
                            "[Pipeline] Waiting for %s + %s...",
                            self.fastlio_cloud_topic,
                            self.fastlio_odom_topic,
                        )
                        last_wait_log = now
                    await self._send_zero_velocity(self.yaw_rate_cmd, now)
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
                    await self._send_zero_velocity(self.yaw_rate_cmd, time.perf_counter())
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
                    await self._send_zero_velocity(self.yaw_rate_cmd, time.perf_counter())
                    await asyncio.sleep(0.005)
                    continue

                last_processed_seq = cloud_seq

                # --- 定位源健康评估 ---
                now_ts = time.perf_counter()
                health = self.pose_source_mgr.evaluate(
                    fastlio_pose=frame_pose,
                    fastlio_pose_stamp=self.fastlio.pose_stamp,
                    fastlio_points=frame_points,
                    fastlio_points_stamp=self.fastlio.points_stamp,
                    now=now_ts,
                )

                # 获取控制位姿 (可能从 FAST-LIO 或 MAVROS GPS fallback)
                if health.control_pose_source.value == "gps_fallback":
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

                # ---- 动态 FOV ROI: 根据当前对地高度计算半宽 ---- 
                dyn_half, _, dyn_height = self._compute_roi_half_from_height(
                    float(control_pose[2]), None,
                )
                self._roi_half_x = dyn_half
                self._roi_half_y = dyn_half

                # 感知输入始终使用 FAST-LIO 去畸变点云 (水平机体, 动态 ROI)
                halss_points, halss_stats = world_to_level_body_roi(
                    frame_points,
                    control_pose[:3],
                    control_pose[3:],
                    self.cfg["perception"],
                    half_x=dyn_half,
                    half_y=dyn_half,
                    )

                ground_p05 = self._pointcloud_ground_clearance(halss_points, 5.0)
                ground_min = self._pointcloud_ground_clearance(halss_points, 0.0)
                pose_xyz = tuple(float(x) for x in control_pose[:3])
                pose_yaw = float(control_pose[5])
                pose_height_m = self.state_manager.height_from_pose(pose_xyz)
                landed_state_on_ground = bool(
                    getattr(self.fc, "landed_state_on_ground", False)
                ) and (
                    pose_height_m is None
                    or pose_height_m <= max(0.3, self.state_manager.landed_height_m + 0.15)
                )
                state_inputs = StateInputs(
                    now=loop_start,
                    pose_xyz=pose_xyz,
                    yaw_rad=pose_yaw,
                    velocity_xyz=tuple(float(x) for x in getattr(self.fc, "uavVelNED", np.zeros(3))[:3]),
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

                # --- MAVROS 安全 fallback: OFFBOARD 丢失或 disarm → ABORT ---
                if self._fc_backend == "mavros" and self.fc is not None and self.fc.safety_fallback:
                    logger.error(
                        "[Pipeline] MAVROS safety fallback: %s → ABORT",
                        self.fc.safety_fallback_reason,
                    )
                    self.mission_state = "ABORT"
                    await self._stop_recording()
                    await self._emergency_stop()
                    break

                # --- FAST-LIO 健康退化动作 ---
                if health.degraded_action is not None:
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
                    yaw_deg = math.degrees(self._yaw_setpoint_rad or float(frame_pose[5]))
                    await self.fc.send_velocity_ned_yaw(0.0, 0.0, 0.0, yaw_deg)
                    await self.fc.disarm()
                    break

                if not state_decision.allow_drl and not state_decision.direct_land:
                    if state_decision.state.value == "IDLE":
                        logger.warning(
                            "[Pipeline] Control loop left active flight state: reason=%s",
                            state_decision.reason,
                        )
                        break
                    yaw_deg = await self._send_zero_velocity_logged(
                        self.yaw_rate_cmd, time.perf_counter(),
                        state=state_decision.state.value,
                        reason=state_decision.reason,
                        pose_xyz=frame_pose,
                        sync_ms=sync_ms,
                        fallback_reason="fsm_hold",
                    )
                    logger.info(
                        "[%04d] HOLD state=%s reason=%s yaw_sp=%.1fdeg",
                        self.step_count,
                        state_decision.state.value,
                        state_decision.reason,
                        yaw_deg,
                    )
                    self.step_count += 1
                    await asyncio.sleep(max(0.0, self.sim_dt - (time.perf_counter() - loop_start)))
                    continue

                if state_decision.direct_land:
                    t_h0 = t_h1 = time.perf_counter()
                    sem_map = np.full((self.obs_h, self.obs_w), self.danger_id, dtype=np.uint8)
                    safety_bev = None
                    binary_semantic_vis = make_binary_semantic_vis(
                        sem_map,
                        safe_id=self.safe_id,
                        danger_id=self.danger_id,
                    )
                    t_d0 = t_d1 = time.perf_counter()
                    sparse_depth = np.full((self.obs_h, self.obs_w), self.depth_max, dtype=np.float32)
                    valid_mask = np.zeros((self.obs_h, self.obs_w), dtype=bool)
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
                            "half=%.1fm height=%.1fm z=[%.2f, %.2f]",
                            halss_stats["output_points"],
                            halss_stats["input_points"],
                            dyn_half,
                            dyn_height,
                            halss_stats["z_min_body"],
                            halss_stats["z_max_body"],
                        )

                    if halss_result is not None:
                        bev_data = halss_result.get("bev_data", halss_result)
                        sem_map = self.sem_gen.generate(bev_data)
                        safety_bev = bev_data.get("safe_mesh")
                    else:
                        sem_map = np.full((self.obs_h, self.obs_w), self.danger_id, dtype=np.uint8)
                        safety_bev = None
                    binary_semantic_vis = make_binary_semantic_vis(
                        sem_map,
                        safe_id=self.safe_id,
                        danger_id=self.danger_id,
                    )

                    t_d0 = time.perf_counter()
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
                    action_id, rl_info = self.drl.predict(rendered_depth, sem_map)
                    t_r1 = time.perf_counter()

                    action_name = self.decomposer.action_id_to_name(action_id)
                    v_body, v_ned, action_yaw_rate = self.decomposer.decompose(
                        action_id,
                        float(frame_pose[5]),
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

                fresh_cloud_seq = self.fastlio.points_seq
                if not state_decision.direct_land and fresh_cloud_seq > cloud_seq:
                    skipped_frames += max(1, fresh_cloud_seq - cloud_seq)
                    yaw_deg = await self._send_zero_velocity_logged(
                        action_yaw_rate, time.perf_counter(),
                        state=state_decision.state.value,
                        reason=state_decision.reason,
                        pose_xyz=frame_pose,
                        sync_ms=sync_ms,
                        fallback_reason="stale_inference_dropped",
                    )
                    if skipped_frames % 10 == 0:
                        logger.info("[Pipeline] Dropped %d stale inferred frames total", skipped_frames)
                    logger.info(
                        "[%04d] STALE INFERENCE DROPPED cloud_seq=%d->%d pose_seq=%d "
                        "yaw=%.1fdeg yaw_sp=%.1fdeg act=%d(%s)",
                        self.step_count,
                        cloud_seq,
                        fresh_cloud_seq,
                        pose_seq,
                        math.degrees(float(frame_pose[5])),
                        yaw_deg,
                        action_id,
                        action_name,
                    )
                    self.step_count += 1
                    await asyncio.sleep(max(0.0, self.sim_dt - (time.perf_counter() - loop_start)))
                    continue

                pre_control_ms = (time.perf_counter() - loop_start) * 1000.0
                slow_frame = pre_control_ms > self.max_frame_latency_ms
                if slow_frame and self.drop_slow_frames and not state_decision.direct_land:
                    yaw_deg = await self._send_zero_velocity_logged(
                        action_yaw_rate, time.perf_counter(),
                        state=state_decision.state.value,
                        reason=state_decision.reason,
                        pose_xyz=frame_pose,
                        sync_ms=sync_ms,
                        fallback_reason="slow_frame_dropped",
                    )
                    self._record_timing(
                        t_h1 - t_h0, t_d1 - t_d0, t_c1 - t_c0,
                        t_r1 - t_r0, 0.0, time.perf_counter() - loop_start
                    )
                    logger.warning(
                        "[%04d] SLOW FRAME DROPPED pre_ctrl=%.0fms budget=%.0fms "
                        "cloud_seq=%d pose_seq=%d yaw=%.1fdeg yaw_sp=%.1fdeg act=%d(%s) v_ned=%s",
                        self.step_count, pre_control_ms, self.max_frame_latency_ms,
                        cloud_seq, pose_seq, math.degrees(float(frame_pose[5])), yaw_deg,
                        action_id, action_name, _fmt_vec(v_ned),
                    )
                    if not state_decision.direct_land:
                        self._monitor_action_collapse(
                            action_id, action_name, rl_info, sparse_depth, valid_mask,
                            rendered_depth, sem_map, binary_semantic_vis, frame_pose,
                            sync_ms, v_body, v_ned, cloud_seq, pose_seq
                        )
                    self.visualizer.update(
                        sem_map=sem_map,
                        depth_map=rendered_depth,
                        safety_bev=safety_bev,
                        drone_pose=frame_pose,
                        binary_semantic_vis=binary_semantic_vis,
                    )
                    self.step_count += 1
                    await asyncio.sleep(max(0.0, self.sim_dt - (time.perf_counter() - loop_start)))
                    continue

                t_ctrl = time.perf_counter()
                yaw_deg = self._advance_yaw_setpoint(action_yaw_rate, t_ctrl)
                if state_decision.direct_land and state_decision.land_reference_xy_yaw is not None:
                    pass
                v_mavros_sp = np.zeros(3, dtype=np.float32)
                if self._fc_backend == "mavros" and self.fc is not None:
                    v_mavros_sp = await self.fc.send_velocity_body_or_ned_aligned_to_mavros(
                        v_body, v_ned, yaw_deg,
                    )
                else:
                    await self.fc.send_velocity_ned_yaw(v_ned[0], v_ned[1], v_ned[2], yaw_deg)
                t_ctrl_end = time.perf_counter()

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
                    roll_deg=math.degrees(float(_att[0])),
                    pitch_deg=math.degrees(float(_att[1])),
                    yaw_deg=math.degrees(float(frame_pose[5])),
                    yaw_setpoint_deg=yaw_deg,
                    height_m=float("nan") if state_decision.height_m is None else state_decision.height_m,
                    sync_ms=sync_ms,
                    direct_land=state_decision.direct_land,
                    fallback_reason="",
                    health_ctrl=health.control_pose_source.value,
                    health_cloud_src=health.perception_cloud_source.value,
                    health_pose_ok="ok" if health.fastlio_pose_healthy else "degraded",
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
                    t_r1 - t_r0, t_ctrl_end - t_ctrl, t_ctrl_end - loop_start
                )

                self.visualizer.update(
                    sem_map=sem_map,
                    depth_map=rendered_depth,
                    safety_bev=safety_bev,
                    drone_pose=frame_pose,
                    binary_semantic_vis=binary_semantic_vis,
                )

                self._log_frame(
                    action_id, action_name, frame_pose[5], yaw_deg, action_yaw_rate, sync_ms,
                    v_body, v_ned, rendered_depth, valid_mask, sem_map, rl_info,
                    cloud_seq, pose_seq, state_decision, health,
                )
                if not state_decision.direct_land:
                    self._monitor_action_collapse(
                        action_id, action_name, rl_info, sparse_depth, valid_mask,
                        rendered_depth, sem_map, binary_semantic_vis, frame_pose,
                        sync_ms, v_body, v_ned, cloud_seq, pose_seq
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

    def _advance_yaw_setpoint(self, yaw_rate_rad_s: float, now: float) -> float:
        if self._yaw_setpoint_rad is None:
            mav_yaw = float(self.fc.uavAngEular[2]) if self.fc is not None else 0.0
            self._yaw_setpoint_rad = mav_yaw
            self._last_yaw_update_t = now
        dt = self.sim_dt if self._last_yaw_update_t is None else max(0.0, now - self._last_yaw_update_t)
        self._last_yaw_update_t = now
        self._yaw_setpoint_rad = _wrap_pi(self._yaw_setpoint_rad + yaw_rate_rad_s * dt)
        return math.degrees(self._yaw_setpoint_rad)

    async def _send_zero_velocity(self, yaw_rate_rad_s: float, now: float) -> float:
        yaw_deg = self._advance_yaw_setpoint(yaw_rate_rad_s, now)
        if self.fc is not None:
            if self._fc_backend == "mavros":
                await self.fc.send_zero_velocity_fallback(yaw_deg)
            else:
                await self.fc.send_velocity_ned_yaw(0.0, 0.0, 0.0, yaw_deg)
        return yaw_deg

    async def _send_zero_velocity_logged(
        self, yaw_rate_rad_s: float, now: float,
        state: str = "", reason: str = "",
        pose_xyz=None, sync_ms=None,
        fallback_reason: str = ""
    ) -> float:
        yaw_deg = self._advance_yaw_setpoint(yaw_rate_rad_s, now)
        if self.fc is not None:
            if self._fc_backend == "mavros":
                await self.fc.send_zero_velocity_fallback(yaw_deg)
            else:
                await self.fc.send_velocity_ned_yaw(0.0, 0.0, 0.0, yaw_deg)
        # Log zero-velocity fallback
        _fc_vel = getattr(self.fc, "uavVelNED", np.zeros(3, dtype=np.float32)) if self.fc else np.zeros(3)
        _att = getattr(self.fc, "uavAngEular", np.zeros(3, dtype=np.float32)) if self.fc else np.zeros(3)
        _yaw = float(pose_xyz[5]) if pose_xyz is not None and len(pose_xyz) > 5 else float(_att[2])
        self._log_velocity_command(
            step=self.step_count,
            timestamp_s=now,
            state=state,
            reason=reason,
            action_id=-1,
            action_name="ZERO_FALLBACK",
            v_body=np.zeros(3, dtype=np.float32),
            v_ned=np.zeros(3, dtype=np.float32),
            v_mavros_sp=np.zeros(3, dtype=np.float32),
            fc_vel=_fc_vel,
            roll_deg=math.degrees(float(_att[0])),
            pitch_deg=math.degrees(float(_att[1])),
            yaw_deg=math.degrees(_yaw),
            yaw_setpoint_deg=yaw_deg,
            height_m=0.0,
            sync_ms=sync_ms,
            direct_land=False,
            fallback_reason=fallback_reason,
        )
        return yaw_deg

    async def _goto_safe_point(self):
        if self._indoor_local_mode:
            target = self._safe_ned_target
            logger.info(
                "[GOTO_SAFE] Indoor 3D target NED: n=%.2f e=%.2f d=%.2f "
                "tol_xy=%.2fm tol_z=%.2fm speed_xy=%.2fm/s speed_z=%.2fm/s",
                target[0], target[1], target[2],
                self._goto_tolerance_xy, self._goto_tolerance_z,
                self._goto_speed_xy, self._goto_speed_z,
            )
        else:
            if self._safe_ned is None:
                self._safe_ned = self._gps_to_ned_offset_3d(
                    self._safe_lat, self._safe_lon, self._safe_altitude_m,
                )
            target = self._safe_ned
            logger.info(
                "[GOTO_SAFE] GPS 3D target NED: n=%.1f e=%.1f d=%.1f "
                "tol_xy=%.1fm tol_z=%.1fm",
                target[0], target[1], target[2],
                self._goto_tolerance_xy, self._goto_tolerance_z,
            )

        goto_start = time.perf_counter()
        while self.mission_state == "GOTO_SAFE":
            current = self.fc.uavPosNED
            dx = target[0] - current[0]
            dy = target[1] - current[1]
            dz = target[2] - current[2]  # NED down: dz>0 means target is below
            dist_xy = math.sqrt(dx * dx + dy * dy)
            dist_z = abs(dz)

            if dist_xy <= self._goto_tolerance_xy and dist_z <= self._goto_tolerance_z:
                elapsed = time.perf_counter() - goto_start
                logger.info(
                    "[GOTO_SAFE] Arrived. XY error=%.2fm Z error=%.2fm time=%.1fs",
                    dist_xy, dist_z, elapsed,
                )
                self._apply_state_decision(self.state_manager.mark_goto_arrived(
                    reason="goto_safe_arrived_local" if self._indoor_local_mode else "goto_safe_arrived"
                ))
                break

            elapsed = time.perf_counter() - goto_start
            if elapsed > self._goto_max_time_s:
                logger.warning(
                    "[GOTO_SAFE] Timeout %.1fs > %.1fs. XY=%.2fm Z=%.2fm",
                    elapsed, self._goto_max_time_s, dist_xy, dist_z,
                )
                if self._goto_timeout_action == "abort":
                    self.mission_state = "ABORT"
                    await self._emergency_stop()
                else:
                    self._apply_state_decision(self.state_manager.mark_goto_arrived(
                        reason="goto_timeout_direct_land"
                    ))
                break

            # 3D 速度引导: XY 和 Z 独立限速
            speed_xy = min(self._goto_speed_xy, max(0.1, dist_xy))
            vx = speed_xy * dx / max(dist_xy, 1e-6)
            vy = speed_xy * dy / max(dist_xy, 1e-6)
            # Z: NED down 为正
            speed_z = min(self._goto_speed_z, max(0.05, dist_z))
            vz = speed_z * (1.0 if dz > 0 else -1.0)  # positive=down
            yaw_deg = self._advance_yaw_setpoint(self.yaw_rate_cmd, time.perf_counter())
            if self._fc_backend == "mavros" and self.fc is not None:
                await self.fc.send_velocity_body_or_ned_aligned_to_mavros(
                    np.array([vx, vy, vz], dtype=np.float32),
                    np.array([vx, vy, vz], dtype=np.float32),
                    yaw_deg,
                )
            else:
                await self.fc.send_velocity_ned_yaw(vx, vy, vz, yaw_deg)

            if self.step_count % 10 == 0:
                logger.info(
                    "[GOTO_SAFE] XY=%.2fm Z=%.2fm v_ned=(%.2f,%.2f,%.2f) yaw_sp=%.1fdeg elapsed=%.1fs",
                    dist_xy, dist_z, vx, vy, vz, yaw_deg, elapsed,
                )
            self.step_count += 1
            await asyncio.sleep(0.1)

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
        return np.array([dn, de, dd], dtype=np.float32)

    def _compute_local_ned_target(self, home_ned: np.ndarray, home_yaw_rad: float) -> np.ndarray:
        """[DEPRECATED] 2D body-frame offset → NED target. Use _compute_local_ned_target_3d instead."""
        return self._compute_local_ned_target_3d(home_ned, home_yaw_rad)

    def _gps_to_ned_offset(self, lat: float, lon: float) -> np.ndarray:
        """[DEPRECATED] Use _gps_to_ned_offset_3d instead."""
        return self._gps_to_ned_offset_3d(lat, lon, self._safe_altitude_m or 2.0)

    def _gps_to_ned_distance(self, lat: float, lon: float) -> float:
        target = self._gps_to_ned_offset_3d(lat, lon, self._safe_altitude_m or 2.0)
        return float(np.linalg.norm(target[:2]))

    # ------------------------------------------------------------------
    # Data/logging
    # ------------------------------------------------------------------

    def _grab_latest_snapshot(self):
        """Return one immutable FAST-LIO cloud/pose snapshot for one control decision."""
        pts = self.fastlio.points
        pose = self.fastlio.pose
        cloud_seq = self.fastlio.points_seq
        pose_seq = self.fastlio.pose_seq
        sync_ms = self.fastlio.sync_delta_ms
        if pts is None or pose is None:
            return None, None, -1, -1, None
        return pts.copy(), pose.copy(), cloud_seq, pose_seq, sync_ms

    def _record_timing(self, halss, depth, completion, rl, control, total):
        self._timing["halss"].append(halss * 1000.0)
        self._timing["depth"].append(depth * 1000.0)
        self._timing["completion"].append(completion * 1000.0)
        self._timing["rl"].append(rl * 1000.0)
        self._timing["control"].append(control * 1000.0)
        self._timing["total"].append(total * 1000.0)

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

    def _roi_bounds(self) -> dict:
        """Return the current (possibly dynamic) ROI bounds for HALSS / depth."""
        return {
            "x_min": -self._roi_half_x, "x_max": self._roi_half_x,
            "y_min": -self._roi_half_y, "y_max": self._roi_half_y,
        }

    def _compute_roi_half_from_height(self, pose_z_world: float,
                                       halss_points: np.ndarray = None) -> tuple:
        """Compute dynamic ROI half-extent from current height above ground.

        Simulates a fixed-FOV depth camera: half = H * tan(fov_half).
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
        probs = [0.0] * len(self.decomposer.action_names)
        if len(probs) > 9:
            probs[9] = 1.0
        return {
            "confidence": 1.0,
            "action_probs": probs,
            "depth_norm_min": float(np.min(depth)),
            "depth_norm_mean": float(np.mean(depth)),
            "depth_norm_max": float(np.max(depth)),
            "sem_norm_min": 30.0 if np.any(sem_safe) else 250.0,
            "sem_norm_mean": float(np.mean(np.where(sem_safe, 30.0, 250.0))),
            "sem_norm_max": 250.0,
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
                                 pose, sync_ms, v_body, v_ned, cloud_seq, pose_seq):
        self.action_monitor.observe(
            self.step_count,
            action_id,
            action_name,
            rl_info,
            {
                "sparse_depth": sparse_depth.astype(np.float32),
                "valid_mask": valid_mask.astype(np.uint8),
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
            yaw_deg = math.degrees(self._yaw_setpoint_rad or 0.0)
            if self._fc_backend == "mavros" and self.fc is not None:
                await self.fc.send_zero_velocity_fallback(yaw_deg)
            elif self.fc is not None:
                await self.fc.send_velocity_ned_yaw(0.0, 0.0, 0.0, yaw_deg)
        except Exception:
            pass
        self.visualizer.close()
        await self._stop_recording()
        self._close_velocity_log()
        self._close_drl_action_log()
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
            "step", "timestamp_s", "state", "reason",
            "action_id", "action_name",
            "v_body_x", "v_body_y", "v_body_z",
            "v_ned_x", "v_ned_y", "v_ned_z",
            "v_mavros_x", "v_mavros_y", "v_mavros_z",
            "fc_vel_x", "fc_vel_y", "fc_vel_z",
            "roll_deg", "pitch_deg", "yaw_deg",
            "yaw_setpoint_deg", "height_m",
            "sync_ms", "direct_land", "fallback_reason",
            "control_pose_source", "perception_cloud_source",
            "fastlio_pose_healthy", "fastlio_cloud_healthy",
        ])
        self._vel_log_file.flush()

        # DRL action CSV
        self._drl_log_path = self._run_dir / "drl_action_log.csv"
        self._drl_log_file = open(str(self._drl_log_path), "w", newline="")
        self._drl_log_writer = csv.writer(self._drl_log_file)
        self._drl_log_writer.writerow([
            "step", "cloud_seq", "pose_seq", "sync_ms",
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
            "roi_type": "level_body_fixed_10x10",
            "roi_x_range": [-5.0, 5.0],
            "roi_y_range": [-5.0, 5.0],
            "dmax": self.depth_max,
            "obs_size": [self.obs_h, self.obs_w],
            "yaw_rate_rad_s": self.yaw_rate_cmd,
            "sim_dt": self.sim_dt,
            "fc_backend": self._fc_backend,
            "fc_mavros_ns": self._mavros_ns,
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
        yaw_setpoint_deg: float, height_m: float,
        sync_ms: float, direct_land: bool, fallback_reason: str = "",
        health_ctrl: str = "", health_cloud_src: str = "",
        health_pose_ok: str = "", health_cloud_ok: str = "",
    ):
        if self._vel_log_writer is None:
            return
        try:
            v_body = np.asarray(v_body, dtype=np.float32).ravel()
            v_ned = np.asarray(v_ned, dtype=np.float32).ravel()
            v_mavros_sp = np.asarray(v_mavros_sp, dtype=np.float32).ravel()
            fc_vel = np.asarray(fc_vel, dtype=np.float32).ravel()
            self._vel_log_writer.writerow([
                step, f"{timestamp_s:.6f}", state, reason,
                action_id, action_name,
                f"{v_body[0]:.4f}", f"{v_body[1]:.4f}", f"{v_body[2]:.4f}",
                f"{v_ned[0]:.4f}", f"{v_ned[1]:.4f}", f"{v_ned[2]:.4f}",
                f"{v_mavros_sp[0]:.4f}", f"{v_mavros_sp[1]:.4f}", f"{v_mavros_sp[2]:.4f}",
                f"{fc_vel[0]:.4f}", f"{fc_vel[1]:.4f}", f"{fc_vel[2]:.4f}",
                f"{roll_deg:.2f}", f"{pitch_deg:.2f}", f"{yaw_deg:.2f}",
                f"{yaw_setpoint_deg:.2f}", f"{height_m:.3f}",
                f"{sync_ms:.1f}" if sync_ms is not None else "n/a",
                "1" if direct_land else "0",
                fallback_reason,
                health_ctrl, health_cloud_src,
                health_pose_ok, health_cloud_ok,
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
                step, cloud_seq, pose_seq,
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

    # ------------------------------------------------------------------
    # Rosbag recording
    # ------------------------------------------------------------------

    async def _start_recording(self):
        """Start rosbag recording if enabled."""
        if not self._record_bag:
            return
        try:
            self._setup_run_dir()
            bag_path = self._run_dir / "input.bag"
            topics = ["/ali_cloud", "/ali_odom"]
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
            else:
                logger.info("[Rosbag] Recording to %s", bag_path)
        except FileNotFoundError:
            logger.warning("[Rosbag] rosbag command not found; skipping recording")
            self._bag_process = None
        except Exception as e:
            logger.warning("[Rosbag] Failed to start recording: %s", e)
            self._bag_process = None

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
            logger.info("[Rosbag] Recording enabled: /ali_cloud + /ali_odom")

    async def _emergency_stop(self):
        logger.error("[Pipeline] EMERGENCY STOP!")
        try:
            yaw_deg = math.degrees(self._yaw_setpoint_rad or 0.0)
            if self._fc_backend == "mavros" and self.fc is not None:
                await self.fc.send_zero_velocity_fallback(yaw_deg)
            elif self.fc is not None:
                await self.fc.send_velocity_ned_yaw(0.0, 0.0, 0.0, yaw_deg)
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
    parser.add_argument("--mavsdk-address", type=str, default=None)
    parser.add_argument("--yaw-rate-rad-s", type=float, default=None,
                        help="Override uav.yaw_rate_rad_s for this experiment run")
    parser.add_argument("--onnx-model", type=str, default=None,
                        help="ONNX DRL policy path (default: zmq_pipeline.drl_control.onnx_model_path or weights/ppo2_policy.onnx)")
    parser.add_argument("--dmax", type=float, default=None,
                        help="Override depth_projection.max_range for BEV NN-fill depth and ONNX inference")
    parser.add_argument("--depth-output-scale", type=float, default=None,
                        help="Legacy compatibility only: updates old depth_completion.output_scale gate; NN-fill/ONNX does not use it")
    parser.add_argument("--flight-ready-check-only", action="store_true",
                        help="Evaluate strict gates with CLI overrides, then exit before model/ROS init")
    parser.add_argument("--allow-incomplete-experiment", action="store_true",
                        help="Bypass strict flight-ready gates for bench debugging only")
    parser.add_argument("--record-bag", action="store_true", default=None,
                        help="Enable rosbag recording of /ali_cloud and /ali_odom")
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
                _, override_failures = _validate_global_guidance_override(args)
                failures = flight_ready_failures(
                    cfg_preview,
                    global_guidance_ready=True if override_failures else _config_has_global_guidance(cfg_preview, args),
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
        mavsdk_address=args.mavsdk_address,
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
