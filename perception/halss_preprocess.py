"""HALSS input preprocessing for ego-centered down-looking maps.

Two ROI modes are supported:
  - world_to_body_down_roi()  -- legacy dynamic-radius body-down ROI (deprecated for flight)
  - world_to_level_body_roi() -- fixed 10m×10m horizontal body ROI (current flight standard)

The flight pipeline and no-control test both use world_to_level_body_roi().
"""

from __future__ import annotations

import numpy as np


def _rot_z(yaw: float) -> np.ndarray:
    """2D rotation about z (yaw only).

    Standard right-hand rotation: R = [[c,-s],[s,c]] applied as pts_body = (pts_world - origin) @ R.
    For ROS ENU world (x=east, y=north): body_x = forward, body_y = left.
    For NED world (x=north, y=east): body_x = forward, body_y = right.
    The axis naming depends on the input world-frame convention.
    """
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([
        [c, -s, 0.0],
        [s,  c, 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float32)


def _rot_zyx(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Body-to-world rotation for roll/pitch/yaw in NED convention."""
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ], dtype=np.float32)


def _cfg_vec3(cfg: dict, key: str, default) -> np.ndarray:
    value = cfg.get(key, default)
    arr = np.asarray(value, dtype=np.float32)
    if arr.shape != (3,):
        raise ValueError(f"{key} must be a 3-element vector")
    return arr


def world_to_body_down_roi(
    points_world: np.ndarray,
    pose_xyz: np.ndarray,
    rpy: np.ndarray,
    cfg: dict,
) -> tuple[np.ndarray, dict]:
    """Convert FAST-LIO world points to the local down-looking HALSS ROI.

    The returned points are in an ego body frame with x=forward, y=right,
    z=down. HALSS then sees the ground as a local height field under the
    aircraft instead of a world-aligned scan footprint.
    """
    stats = {
        "input_points": 0,
        "output_points": 0,
        "roi_radius_m": float(cfg.get("halss_roi_radius_body", cfg.get("roi_radius_world", 25.0))),
        "min_down_m": float(cfg.get("halss_min_down_m", 0.05)),
        "max_down_m": float(cfg.get("halss_max_down_m", cfg.get("halss_roi_max_down_m", 30.0))),
        "yaw_only": bool(cfg.get("halss_yaw_only", True)),
    }

    if points_world is None:
        return np.empty((0, 3), dtype=np.float32), stats

    pts = np.asarray(points_world, dtype=np.float32)
    if pts.size == 0:
        return np.empty((0, 3), dtype=np.float32), stats
    if pts.ndim == 1:
        if pts.size < 3:
            raise ValueError("points_world must contain x,y,z coordinates")
        pts = pts[:3].reshape(1, 3)
    elif pts.ndim == 2 and pts.shape[1] >= 3:
        pts = pts[:, :3]
    else:
        raise ValueError("points_world must have shape (N,3) or (N,>=3)")

    pose = np.asarray(pose_xyz, dtype=np.float32)
    angles = np.asarray(rpy, dtype=np.float32)
    if pose.shape[0] < 3 or angles.shape[0] < 3:
        raise ValueError("pose_xyz and rpy must contain 3 values")

    stats["input_points"] = int(len(pts))
    roll, pitch, yaw = float(angles[0]), float(angles[1]), float(angles[2])
    yaw_offset = np.deg2rad(float(cfg.get("halss_lidar_yaw_offset_deg", cfg.get("lidar_yaw_offset_deg", 0.0))))
    yaw = yaw + yaw_offset
    if stats["yaw_only"]:
        R_gb = _rot_zyx(0.0, 0.0, yaw)
    else:
        R_gb = _rot_zyx(roll, pitch, yaw)

    # Position of the LiDAR/FAST-LIO odometry origin in the aircraft body frame.
    # If the Mid360 is in front of the CG, set this to [forward_m, 0, down_m].
    lidar_pos_body = _cfg_vec3(
        cfg,
        "halss_lidar_position_body_m",
        cfg.get("lidar_position_body_m", [0.0, 0.0, 0.0]),
    )

    # pose_origin_frame: "base_link" = /ali_odom is already body pose (no correction needed)
    #                   "lidar_imu"  = pose is at radar IMU center, need lidar_pos_body correction
    pose_frame = str(cfg.get("pose_origin_frame", "lidar_imu"))
    if pose_frame == "base_link":
        body_origin_world = pose[:3]
    else:
        body_origin_world = pose[:3] - lidar_pos_body @ R_gb.T
    pts_body = (pts - body_origin_world) @ R_gb

    # If world is ENU z-up but HALSS expects z-down (ground positive-down),
    # negate the z-axis so that "down" becomes positive.
    world_z_up = bool(cfg.get("world_z_up", False))
    if world_z_up:
        pts_body[:, 2] *= -1.0

    lateral = np.linalg.norm(pts_body[:, :2], axis=1)
    radius = float(stats["roi_radius_m"])
    min_down = float(stats["min_down_m"])
    max_down = float(stats["max_down_m"])
    keep = (
        np.isfinite(pts_body).all(axis=1)
        & (lateral <= radius)
        & (pts_body[:, 2] >= min_down)
        & (pts_body[:, 2] <= max_down)
    )

    roi = pts_body[keep].astype(np.float32, copy=False)
    stats.update({
        "output_points": int(len(roi)),
        "body_origin_world": body_origin_world.astype(np.float32),
        "lidar_position_body_m": lidar_pos_body.astype(np.float32),
        "world_z_up": world_z_up,
        "z_min_body": float(np.min(pts_body[:, 2])) if len(pts_body) else float("nan"),
        "z_max_body": float(np.max(pts_body[:, 2])) if len(pts_body) else float("nan"),
    })
    return roi, stats


def world_to_level_body_roi(
    points_world: np.ndarray,
    pose_xyz: np.ndarray,
    rpy: np.ndarray,
    cfg: dict,
    half_x: float = None,
    half_y: float = None,
) -> tuple[np.ndarray, dict]:
    """Convert FAST-LIO world points to a fixed horizontal level-body ROI.

    This is the current flight-standard ROI used by pipeline.py and
    test_live_nocontrol.py.  Key differences from world_to_body_down_roi():

      - Gravity-aligned horizontal frame: only yaw is used for orientation;
        roll and pitch are ignored so that "down" is always gravity-down.
      - Square ROI in body x/y.  If half_x/half_y are explicitly passed by the
        caller they take precedence; otherwise the static config keys
        halss_roi_half_x_m / halss_roi_half_y_m are used as fallback.
      - Optional ground candidate filter via halss_ground_min_down_m.

    Coordinate convention (level body frame):
      x = forward  (horizontal, aligned with yaw)
      y = body lateral (sign depends on world-frame convention, see _rot_z)
      z = gravity down  (positive = below aircraft)

    Parameters
    ----------
    points_world : (N,3) ndarray
        FAST-LIO world-frame (gravity-aligned) point cloud.
    pose_xyz : (3,) ndarray
        Aircraft position in world frame [x, y, z].
    rpy : (3,) ndarray
        Aircraft attitude [roll, pitch, yaw] in radians.  Only yaw is used.
    cfg : dict
        Perception config.
    half_x : float or None
        Override half-extent forward (m).  If None, read from cfg.
    half_y : float or None
        Override half-extent lateral (m).  If None, read from cfg.

    Returns
    -------
    roi : (M,3) ndarray (float32)
        Points inside the body-xy ROI, in level-body coordinates.
    stats : dict
        Diagnostic statistics (input_points, output_points, z stats, bounds).
    """
    # ---- ROI bounds: caller override > config static fallback ----
    HALF_X = half_x if half_x is not None else float(cfg.get("halss_roi_half_x_m", 5.0))
    HALF_Y = half_y if half_y is not None else float(cfg.get("halss_roi_half_y_m", 5.0))
    min_down_m = float(cfg.get("halss_min_down_m", 0.05))
    max_down_m = float(cfg.get("halss_max_down_m", cfg.get("halss_roi_max_down_m", 30.0)))
    ground_min_down_m = float(cfg.get("halss_ground_min_down_m", 0.0))

    stats: dict = {
        "input_points": 0,
        "output_points": 0,
        "roi_type": "level_body_fixed_10x10",
        "roi_x_range": [-HALF_X, HALF_X],
        "roi_y_range": [-HALF_Y, HALF_Y],
        "min_down_m": min_down_m,
        "max_down_m": max_down_m,
    }

    if points_world is None:
        return np.empty((0, 3), dtype=np.float32), stats

    pts = np.asarray(points_world, dtype=np.float32)
    if pts.size == 0:
        return np.empty((0, 3), dtype=np.float32), stats
    if pts.ndim == 1:
        if pts.size < 3:
            raise ValueError("points_world must contain x,y,z coordinates")
        pts = pts[:3].reshape(1, 3)
    elif pts.ndim == 2 and pts.shape[1] >= 3:
        pts = pts[:, :3]
    else:
        raise ValueError("points_world must have shape (N,3) or (N,>=3)")

    pose = np.asarray(pose_xyz, dtype=np.float32)
    angles = np.asarray(rpy, dtype=np.float32)
    if pose.shape[0] < 3 or angles.shape[0] < 3:
        raise ValueError("pose_xyz and rpy must contain 3 values")

    stats["input_points"] = int(len(pts))

    # ---- yaw-only horizontal alignment ----
    yaw = float(angles[2])
    yaw_offset = np.deg2rad(float(cfg.get("halss_lidar_yaw_offset_deg",
                                          cfg.get("lidar_yaw_offset_deg", 0.0))))
    yaw = yaw + yaw_offset
    R_gb = _rot_z(yaw)  # only yaw — gravity-level body frame

    # ---- body origin in world ----
    lidar_pos_body = np.asarray(
        cfg.get("halss_lidar_position_body_m",
                cfg.get("lidar_position_body_m", [0.0, 0.0, 0.0])),
        dtype=np.float32,
    )
    if lidar_pos_body.shape != (3,):
        lidar_pos_body = np.array([0.0, 0.0, 0.0], dtype=np.float32)

    pose_frame = str(cfg.get("pose_origin_frame", "base_link"))
    if pose_frame == "base_link":
        body_origin_world = pose[:3]
    else:
        body_origin_world = pose[:3] - lidar_pos_body @ R_gb.T

    # ---- transform to level body frame ----
    pts_body = (pts - body_origin_world) @ R_gb

    # ---- ENU z-up → z-down if needed ----
    world_z_up = bool(cfg.get("world_z_up", True))
    if world_z_up:
        pts_body[:, 2] *= -1.0  # now z>0 = down

    # ---- fixed square ROI filter ----
    x_ok = (pts_body[:, 0] >= -HALF_X) & (pts_body[:, 0] <= HALF_X)
    y_ok = (pts_body[:, 1] >= -HALF_Y) & (pts_body[:, 1] <= HALF_Y)
    z_ok = (pts_body[:, 2] >= min_down_m) & (pts_body[:, 2] <= max_down_m)

    # ---- optional ground candidate filter ----
    if ground_min_down_m > 0.0:
        ground_ok = pts_body[:, 2] >= ground_min_down_m
    else:
        ground_ok = np.ones(len(pts_body), dtype=bool)

    keep = (
        np.isfinite(pts_body).all(axis=1)
        & x_ok
        & y_ok
        & z_ok
        & ground_ok
    )

    roi = pts_body[keep].astype(np.float32, copy=False)
    stats.update({
        "output_points": int(len(roi)),
        "body_origin_world": body_origin_world.astype(np.float32),
        "lidar_position_body_m": lidar_pos_body.astype(np.float32),
        "world_z_up": world_z_up,
        "z_min_body": float(np.min(pts_body[:, 2])) if len(pts_body) else float("nan"),
        "z_max_body": float(np.max(pts_body[:, 2])) if len(pts_body) else float("nan"),
        "yaw_used_rad": float(yaw),
        "roll_pitch_ignored": True,
        "ground_min_down_m": ground_min_down_m if ground_min_down_m > 0.0 else None,
        "roi_xy_area_m2": (2.0 * HALF_X) * (2.0 * HALF_Y),
    })
    return roi, stats
