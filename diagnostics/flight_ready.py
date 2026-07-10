"""Strict readiness checks for the full real-flight experiment."""


def iter_flight_ready_checks(cfg: dict, global_guidance_ready=None):
    """Yield ``(ok, message)`` for gates that must pass before flight control.

    ``global_guidance_ready`` can be set by the runtime pipeline after applying
    CLI GIS/safe-point overrides. When omitted, the check uses config only.
    """
    if global_guidance_ready is None:
        yield _check_global_prior_from_config(cfg)
    elif global_guidance_ready:
        yield True, "flight-ready gate: global safe-area guidance configured"
    else:
        yield (
            False,
            "flight-ready gate: no global safe-area guidance active; GIS nine-grid "
            "guidance would be skipped before DRL descent",
        )

    yield _check_output_scale(cfg)
    yield _check_yaw_rate(cfg)
    yield _check_runtime_gpu(cfg)
    yield _check_module_gpu_requirements(cfg)
    yield _check_halss_backend(cfg)
    yield _check_depth_projection(cfg)
    yield _check_depth_completion(cfg)
    yield _check_action_frame(cfg)
    yield _check_action_lateral_sign(cfg)
    yield _check_mission_state(cfg)
    yield _check_localization_contract(cfg)
    yield _check_flight_controller(cfg)
    yield _check_fastlio_health_config(cfg)
    yield _check_observation_encoding(cfg)
    yield _check_visualization_enabled(cfg)
    yield _check_binary_semantic_window(cfg)


def flight_ready_failures(cfg: dict, global_guidance_ready=None):
    return [
        message
        for ok, message in iter_flight_ready_checks(cfg, global_guidance_ready)
        if not ok
    ]


def _check_global_prior_from_config(cfg: dict):
    gp = cfg.get("global_prior", {})
    if not gp.get("enabled", False):
        return (
            False,
            "flight-ready gate: global_prior.enabled=false; GIS nine-grid guidance "
            "would be skipped before DRL descent",
        )

    prior_mode = str(gp.get("mode", "gps")).lower()

    # --- 室内三维安全点 ---
    if prior_mode == "local_body_offset":
        offset = gp.get("local_body_offset_m")
        if offset is None:
            return (
                False,
                "flight-ready gate: local_body_offset mode requires "
                "global_prior.local_body_offset_m to be set",
            )
        if not isinstance(offset, (list, tuple)) or len(offset) < 3:
            return (
                False,
                "flight-ready gate: local_body_offset_m must be 3D [forward, right, up]; "
                f"got {len(offset) if isinstance(offset, (list, tuple)) else type(offset).__name__} elements",
            )
        try:
            forward, right, up = float(offset[0]), float(offset[1]), float(offset[2])
        except (TypeError, ValueError):
            return False, "flight-ready gate: local_body_offset_m elements must be numeric"
        if up <= 0.1:
            return (
                False,
                "flight-ready gate: local_body_offset_m up must be > 0.1m for safe takeoff height; "
                f"got up={up:.2f}m",
            )
        return (
            True,
            f"flight-ready gate: indoor 3D safe point: forward={forward:.1f}m "
            f"right={right:.1f}m up={up:.1f}m",
        )

    # --- GPS/GIS 模式 ---
    if gp.get("target_lat") is not None and gp.get("target_lon") is not None:
        if gp.get("target_source") != "gis":
            return (
                False,
                "flight-ready gate: configured target_lat/target_lon must declare "
                "global_prior.target_source='gis' to prove it came from nine-grid GIS risk assessment",
            )
        target_alt = gp.get("target_altitude_m")
        if target_alt is None:
            return (
                False,
                "flight-ready gate: GPS mode requires global_prior.target_altitude_m "
                "for 3D safe point",
            )
        try:
            target_alt = float(target_alt)
        except (TypeError, ValueError):
            return False, f"flight-ready gate: invalid target_altitude_m={target_alt!r}"
        if target_alt <= 0.5:
            return (
                False,
                "flight-ready gate: target_altitude_m must be >= 0.5m; "
                f"got {target_alt:.2f}m",
            )
        return True, "flight-ready gate: GIS-derived GPS 3D safe point configured"

    if gp.get("image_path") and gp.get("bounds"):
        return True, "flight-ready gate: GIS image+bounds configured"

    return (
        False,
        "flight-ready gate: global_prior needs target_lat/target_lon+target_altitude_m "
        "or image_path+bounds or valid local_body_offset_m",
    )


def _check_output_scale(cfg: dict):
    output_scale = cfg["depth_completion"].get("output_scale")
    if output_scale is None:
        return (
            False,
            "flight-ready gate: depth_completion.output_scale is null; calibrate "
            "SparseNet scale before closing flight loop",
        )
    try:
        output_scale = float(output_scale)
    except (TypeError, ValueError):
        return False, f"flight-ready gate: invalid output_scale={output_scale!r}"
    if output_scale <= 0.0:
        return False, f"flight-ready gate: output_scale must be positive, got {output_scale}"
    return True, f"flight-ready gate: output_scale={output_scale:.6g}"


def _check_yaw_rate(cfg: dict):
    yaw_rate = float(cfg["uav"].get("yaw_rate_rad_s", 0.0))
    if abs(yaw_rate) < 1e-6:
        return (
            False,
            "flight-ready gate: uav.yaw_rate_rad_s is 0.0; set the planned yaw-fault rate",
        )
    return True, f"flight-ready gate: yaw_rate_rad_s={yaw_rate:.6g}"


def _check_runtime_gpu(cfg: dict):
    runtime = cfg.get("runtime", {})
    if runtime.get("use_gpu") is not True or runtime.get("device") != "cuda":
        return (
            False,
            "flight-ready gate: runtime must force GPU execution "
            "(runtime.use_gpu=true, runtime.device='cuda') for Orin realtime inference",
        )
    return True, "flight-ready gate: runtime GPU execution configured"


def _check_module_gpu_requirements(cfg: dict):
    if cfg.get("perception", {}).get("require_gpu") is not True:
        return (
            False,
            "flight-ready gate: perception.require_gpu must be true so HALSS "
            "Bayesian inference cannot fall back to CPU",
        )
    if cfg.get("decision", {}).get("require_gpu") is not True:
        return (
            False,
            "flight-ready gate: decision.require_gpu must be true so DRL policy "
            "inference cannot fall back to CPU",
        )
    return True, "flight-ready gate: HALSS and DRL GPU fallback disabled"


def _check_halss_backend(cfg: dict):
    backend = cfg.get("perception", {}).get("halss_backend")
    if backend != "bayesian_unet":
        return (
            False,
            "flight-ready gate: perception.halss_backend must be 'bayesian_unet' "
            "for the HALSS Bayesian segmentation experiment path",
        )
    return True, "flight-ready gate: HALSS Bayesian UNet backend configured"


def _check_depth_projection(cfg: dict):
    depth_cfg = cfg.get("depth_projection", {})
    if depth_cfg.get("mode") != "perspective":
        return (
            False,
            "flight-ready gate: depth_projection.mode must be 'perspective' "
            "for the down-looking camera projection formula",
        )
    backend = depth_cfg.get("backend")
    if backend != "torch_cuda":
        return (
            False,
            "flight-ready gate: depth_projection.backend must be 'torch_cuda' "
            "so Mid360 depth projection/z-buffer runs on GPU",
        )
    required = ("fx", "fy", "cx", "cy", "R_I_to_C")
    missing = [key for key in required if depth_cfg.get(key) is None]
    if missing:
        return (
            False,
            "flight-ready gate: depth_projection missing intrinsics/extrinsics: "
            + ",".join(missing),
        )
    return True, "flight-ready gate: perspective CUDA depth projection configured"


def _check_depth_completion(cfg: dict):
    dcfg = cfg.get("depth_completion", {})
    if dcfg.get("backend") != "sparsity_invariant_cnn":
        return (
            False,
            "flight-ready gate: depth_completion.backend must be 'sparsity_invariant_cnn'",
        )
    if dcfg.get("input_encoding") != "inverse_unit":
        return (
            False,
            "flight-ready gate: depth_completion.input_encoding must be 'inverse_unit' "
            "to use x=1-D/dmax",
        )
    return True, "flight-ready gate: SparseNet inverse-depth completion configured"


def _check_action_frame(cfg: dict):
    frame = cfg["uav"].get("action_frame", "body")
    if frame != "body":
        return (
            False,
            "flight-ready gate: uav.action_frame must be 'body' so DRL lateral "
            "actions are rotated by the latest FAST-LIO yaw",
        )
    return True, "flight-ready gate: action_frame=body (yaw-compensated action mapping)"


def _check_action_lateral_sign(cfg: dict):
    sign = cfg["uav"].get("action_lateral_sign", -1)
    try:
        sign = int(sign)
    except (TypeError, ValueError):
        return False, f"flight-ready gate: invalid action_lateral_sign={sign!r}"
    if sign != -1:
        return (
            False,
            "flight-ready gate: uav.action_lateral_sign must be -1 to match the "
            "original DeepRL quadrotor_env.py action mapping",
        )
    return True, "flight-ready gate: action_lateral_sign=-1 (DeepRL action mapping)"


def _check_observation_encoding(cfg: dict):
    obs = cfg["observation"]
    depth_mode = obs.get("depth_norm_mode")
    sem_mode = obs.get("semantic_norm_mode")
    if depth_mode != "meters_div255" or sem_mode != "gray_unit":
        return (
            False,
            "flight-ready gate: observation encoding must be DeepRL-compatible "
            "(depth_norm_mode=meters_div255, semantic_norm_mode=gray_unit)",
        )
    return True, "flight-ready gate: observation encoding matches original DeepRL scale"


def _check_mission_state(cfg: dict):
    mcfg = cfg.get("mission_state", {})
    if mcfg.get("height_source", "fastlio_z") != "fastlio_z":
        return (
            False,
            "flight-ready gate: mission_state.height_source must be 'fastlio_z' "
            "so direct-land switching uses gravity-calibrated localization height",
        )
    if mcfg.get("height_axis", "neg_z") not in ("neg_z", "pos_z", "abs_z"):
        return (
            False,
            "flight-ready gate: mission_state.height_axis must be one of "
            "'neg_z', 'pos_z', or 'abs_z'",
        )
    try:
        direct_land_trigger = float(mcfg.get("direct_land_trigger_height_m", 0.8))
        landed_height = float(mcfg.get("landed_height_m", 0.15))
        direct_land_vz = float(mcfg.get("direct_land_vz_mps", 0.25))
    except (TypeError, ValueError):
        return False, "flight-ready gate: mission_state direct-land thresholds must be numeric"
    if not (0.2 <= direct_land_trigger <= 2.0):
        return (
            False,
            "flight-ready gate: direct_land_trigger_height_m must be in [0.2, 2.0] m",
        )
    if not (0.02 <= landed_height < direct_land_trigger):
        return (
            False,
            "flight-ready gate: landed_height_m must be positive and below "
            "direct_land_trigger_height_m",
        )
    if not (0.05 <= direct_land_vz <= 0.8):
        return False, "flight-ready gate: direct_land_vz_mps must be in [0.05, 0.8] m/s"
    if mcfg.get("ground_crosscheck_action", "warn") not in ("warn", "block"):
        return False, "flight-ready gate: ground_crosscheck_action must be 'warn' or 'block'"
    return (
        True,
        "flight-ready gate: mission state uses Fast-LIO z with bounded direct-land thresholds",
    )


def _check_localization_contract(cfg: dict):
    lcfg = cfg.get("localization", {})
    mode = lcfg.get("mode", "fastlio_external_vision")
    if mode not in ("fastlio_external_vision", "mocap_external_vision"):
        return (
            False,
            "flight-ready gate: localization.mode must be 'fastlio_external_vision' "
            "or 'mocap_external_vision'",
        )
    required = (
        "fastlio_odom_topic", "world_cloud_topic",
        "external_pose_topic", "local_odom_topic",
    )
    missing = [key for key in required if not lcfg.get(key)]
    if missing:
        return (
            False,
            "flight-ready gate: localization missing topic config: " + ",".join(missing),
        )

    # FAST-LIO deskewed cloud 输入必须存在
    cloud_topic = lcfg.get("world_cloud_topic") or lcfg.get("body_cloud_topic")
    if not cloud_topic:
        return (
            False,
            "flight-ready gate: localization must define world_cloud_topic or "
            "body_cloud_topic for FAST-LIO deskewed cloud input",
        )

    # GPS fallback topic 检查
    gps_fallback_allowed = bool(lcfg.get("allow_gps_fallback", False))
    if gps_fallback_allowed:
        mavros_odom = lcfg.get("mavros_local_odom_topic")
        mavros_gps = lcfg.get("mavros_global_fix_topic")
        if not mavros_odom:
            return (
                False,
                "flight-ready gate: allow_gps_fallback=true requires "
                "localization.mavros_local_odom_topic",
            )
        if not mavros_gps:
            return (
                False,
                "flight-ready gate: allow_gps_fallback=true requires "
                "localization.mavros_global_fix_topic",
            )

    return True, f"flight-ready gate: localization contract configured ({mode})"


def _check_flight_controller(cfg: dict):
    """验证飞控后端配置."""
    fc_cfg = cfg.get("flight_controller", {})
    backend = str(fc_cfg.get("backend", "mavros")).lower()

    if backend not in ("mavros", "mavsdk"):
        return (
            False,
            f"flight-ready gate: flight_controller.backend must be 'mavros' or 'mavsdk'; got '{backend}'",
        )

    if backend == "mavros":
        mavros_ns = fc_cfg.get("mavros_ns", "/mavros")
        if not mavros_ns.startswith("/"):
            return (
                False,
                "flight-ready gate: flight_controller.mavros_ns must start with '/'",
            )
        rate = float(fc_cfg.get("setpoint_rate_hz", 20))
        if rate < 5 or rate > 100:
            return (
                False,
                f"flight-ready gate: setpoint_rate_hz must be in [5, 100]; got {rate}",
            )
        warmup = float(fc_cfg.get("offboard_warmup_s", 2.0))
        if warmup < 0.5:
            return (
                False,
                "flight-ready gate: offboard_warmup_s must be >= 0.5s for PX4 setpoint stream",
            )

    return True, f"flight-ready gate: flight controller backend={backend}"


def _check_fastlio_health_config(cfg: dict):
    """验证 FAST-LIO 健康门控配置."""
    health = cfg.get("fastlio_health", {})

    max_pose_age = float(health.get("max_pose_age_ms", 200))
    max_cloud_age = float(health.get("max_cloud_age_ms", 200))
    max_sync = float(health.get("max_cloud_odom_sync_ms", 100))
    min_pts = int(health.get("min_cloud_points", 50))
    pose_jump = float(health.get("pose_jump_threshold_m", 1.0))
    yaw_jump = float(health.get("yaw_jump_threshold_deg", 20.0))

    if max_pose_age <= 0 or max_cloud_age <= 0:
        return False, "flight-ready gate: fastlio_health max age must be > 0"
    if min_pts <= 0:
        return False, "flight-ready gate: fastlio_health min_cloud_points must be > 0"
    if pose_jump <= 0:
        return False, "flight-ready gate: fastlio_health pose_jump_threshold_m must be > 0"

    degraded_ctrl = str(health.get("degraded_control_action", "use_gps_fallback")).lower()
    if degraded_ctrl not in ("use_gps_fallback", "direct_land", "abort"):
        return (
            False,
            f"flight-ready gate: invalid degraded_control_action='{degraded_ctrl}'",
        )

    degraded_cloud = str(health.get("degraded_cloud_action", "direct_land")).lower()
    if degraded_cloud not in ("direct_land", "abort"):
        return (
            False,
            f"flight-ready gate: invalid degraded_cloud_action='{degraded_cloud}'",
        )

    return (
        True,
        "flight-ready gate: FAST-LIO health gates: "
        f"pose_age={max_pose_age:.0f}ms cloud_age={max_cloud_age:.0f}ms "
        f"min_pts={min_pts} jump={pose_jump:.1f}m yaw_jump={yaw_jump:.1f}deg "
        f"ctrl_action={degraded_ctrl} cloud_action={degraded_cloud}",
    )


def _check_visualization_enabled(cfg: dict):
    vis = cfg.get("visualization", {})
    if vis.get("enable") is not True:
        return (
            False,
            "flight-ready gate: visualization.enable must be true for the live "
            "experiment display",
        )
    if vis.get("show_binary_semantic") is not True:
        return (
            False,
            "flight-ready gate: visualization.show_binary_semantic must be true "
            "to display the HALSS binary safety map",
        )
    if vis.get("show_depth") is not True:
        return (
            False,
            "flight-ready gate: visualization.show_depth must be true to display "
            "the completed depth map",
        )
    return True, "flight-ready gate: binary semantic and depth visualization enabled"


def _check_binary_semantic_window(cfg: dict):
    title = cfg.get("visualization", {}).get("binary_semantic_window_title", "binary semantic")
    if title != "binary semantic":
        return (
            False,
            "flight-ready gate: visualization.binary_semantic_window_title must remain "
            "'binary semantic'",
        )
    return True, "flight-ready gate: binary semantic window title fixed"
