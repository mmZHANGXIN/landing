"""Projection compatible with the camera geometry used to train the PPO2 policy.

The training renderer produced a 752x480 down-looking image with fx=fy=455,
then resized it directly to 128x128.  This module preserves that field of view
and axis convention while projecting gravity-level body-frame LiDAR points.

Body frame: x forward, y left/right according to the caller's level-body
convention, z down.  The current flight pipeline uses y as the ROS body-left
axis after its world-to-level-body transform, so the training camera mapping is:

    X_camera = -y_body   (image column; body-left goes left)
    Y_camera = -x_body   (image row; body-forward goes up)
    Z_camera =  z_down
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from utils.valid_nearest import fill_valid_nearest


@dataclass(frozen=True)
class TrainingCameraModel:
    source_width: int = 752
    source_height: int = 480
    source_fx: float = 455.0
    source_fy: float = 455.0
    source_cx: float = 376.5
    source_cy: float = 240.5
    output_width: int = 128
    output_height: int = 128
    near_m: float = 0.05
    far_m: float = 30.0

    @classmethod
    def from_config(cls, cfg: dict, output_width: int = 128,
                    output_height: int = 128, far_m: float = 30.0):
        return cls(
            source_width=int(cfg.get("source_width", 752)),
            source_height=int(cfg.get("source_height", 480)),
            source_fx=float(cfg.get("source_fx", 455.0)),
            source_fy=float(cfg.get("source_fy", 455.0)),
            source_cx=float(cfg.get("source_cx", 376.5)),
            source_cy=float(cfg.get("source_cy", 240.5)),
            output_width=int(cfg.get("output_width", output_width)),
            output_height=int(cfg.get("output_height", output_height)),
            near_m=float(cfg.get("near_m", 0.05)),
            far_m=float(cfg.get("far_m", far_m)),
        )

    @property
    def fx(self) -> float:
        return self.source_fx * self.output_width / self.source_width

    @property
    def fy(self) -> float:
        return self.source_fy * self.output_height / self.source_height

    @property
    def cx(self) -> float:
        return self.source_cx * self.output_width / self.source_width

    @property
    def cy(self) -> float:
        return self.source_cy * self.output_height / self.source_height

    @property
    def horizontal_fov_deg(self) -> float:
        return float(np.degrees(2.0 * np.arctan(self.source_width / (2.0 * self.source_fx))))

    @property
    def vertical_fov_deg(self) -> float:
        return float(np.degrees(2.0 * np.arctan(self.source_height / (2.0 * self.source_fy))))

    def ground_half_extents(self, height_m: float) -> tuple[float, float]:
        """Return (forward half extent, lateral half extent) on flat ground."""
        height = max(float(height_m), self.near_m)
        half_forward = height * np.tan(np.deg2rad(self.vertical_fov_deg) * 0.5)
        half_lateral = height * np.tan(np.deg2rad(self.horizontal_fov_deg) * 0.5)
        return float(half_forward), float(half_lateral)


def sample_nearest_points_by_camera_rays(points_body: np.ndarray,
                                         camera: TrainingCameraModel,
                                         ray_width: int = 64,
                                         ray_height: int = 64):
    """Return at most one nearest observed point for each pinhole camera ray.

    The returned points remain in the gravity-level body frame used by HALSS
    (x=forward, y=lateral, z=down).  Empty rays produce no point.  A fixed
    ray grid bounds all downstream HALSS work while the vectorized projection
    remains linear in the incoming cloud size.
    """
    ray_width = int(ray_width)
    ray_height = int(ray_height)
    if ray_width <= 0 or ray_height <= 0:
        raise ValueError("ray_width and ray_height must be positive")

    stats = {
        "input_points": 0,
        "frustum_points": 0,
        "ray_points": 0,
        "ray_width": ray_width,
        "ray_height": ray_height,
    }
    if points_body is None:
        return np.empty((0, 3), dtype=np.float32), stats
    pts = np.asarray(points_body, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[1] < 3:
        raise ValueError("points_body must have shape (N,3) or (N,>=3)")
    pts = pts[:, :3]
    stats["input_points"] = int(len(pts))
    if len(pts) == 0:
        return np.empty((0, 3), dtype=np.float32), stats

    finite = np.isfinite(pts).all(axis=1)
    z = pts[:, 2]
    keep = finite & (z > camera.near_m) & (z < camera.far_m)
    pts = pts[keep]
    if len(pts) == 0:
        return np.empty((0, 3), dtype=np.float32), stats

    z = pts[:, 2]
    fx = camera.source_fx * ray_width / camera.source_width
    fy = camera.source_fy * ray_height / camera.source_height
    cx = camera.source_cx * ray_width / camera.source_width
    cy = camera.source_cy * ray_height / camera.source_height
    u = np.floor(fx * (-pts[:, 1]) / z + cx).astype(np.int32)
    v = np.floor(fy * (-pts[:, 0]) / z + cy).astype(np.int32)
    inside = (u >= 0) & (u < ray_width) & (v >= 0) & (v < ray_height)
    u, v, z = u[inside], v[inside], z[inside]
    stats["frustum_points"] = int(len(z))
    if len(z) == 0:
        return np.empty((0, 3), dtype=np.float32), stats

    # Fixed-size z-buffer: only the nearest observed depth survives per ray.
    flat = v * ray_width + u
    depth = np.full(ray_width * ray_height, np.inf, dtype=np.float32)
    np.minimum.at(depth, flat, z)
    valid_flat = np.flatnonzero(np.isfinite(depth))
    z_ray = depth[valid_flat]
    v_ray, u_ray = np.divmod(valid_flat, ray_width)

    # Back-project the observed depth at the ray centre.  This avoids sorting
    # the full cloud and guarantees no more than ray_width*ray_height points.
    x_camera = (u_ray.astype(np.float32) + 0.5 - cx) * z_ray / fx
    y_camera = (v_ray.astype(np.float32) + 0.5 - cy) * z_ray / fy
    sampled = np.column_stack((-y_camera, -x_camera, z_ray)).astype(
        np.float32, copy=False
    )
    stats["ray_points"] = int(len(sampled))
    return sampled, stats


def _sample_bev_labels(points_body: np.ndarray, semantic_bev: np.ndarray,
                       bounds: dict, danger_id: int,
                       semantic_bev_valid: np.ndarray | None = None):
    """Nearest-neighbour sample labels and geometric validity at each point."""
    labels = np.full(len(points_body), int(danger_id), dtype=np.uint8)
    label_valid = np.zeros(len(points_body), dtype=bool)
    if semantic_bev is None or np.asarray(semantic_bev).size == 0:
        return labels, label_valid

    sem = np.asarray(semantic_bev)
    h, w = sem.shape[:2]
    x_min, x_max = float(bounds["x_min"]), float(bounds["x_max"])
    y_min, y_max = float(bounds["y_min"]), float(bounds["y_max"])
    x_span = max(x_max - x_min, 1e-6)
    y_span = max(y_max - y_min, 1e-6)

    # HALSS normal images are flipud() before inference, so the returned
    # semantic rows run from body +y to -y. Columns still run -x to +x.
    col = np.rint((points_body[:, 0] - x_min) / x_span * (w - 1)).astype(np.int32)
    row_unflipped = np.rint(
        (points_body[:, 1] - y_min) / y_span * (h - 1)
    ).astype(np.int32)
    row = (h - 1) - row_unflipped
    inside = (row >= 0) & (row < h) & (col >= 0) & (col < w)
    labels[inside] = sem[row[inside], col[inside]].astype(np.uint8)
    if semantic_bev_valid is None:
        label_valid[inside] = True
    else:
        bev_valid = np.asarray(semantic_bev_valid, dtype=bool)
        if bev_valid.shape != sem.shape[:2]:
            raise ValueError("semantic_bev_valid must match semantic_bev shape")
        label_valid[inside] = bev_valid[row[inside], col[inside]]
    return labels, label_valid


def project_training_camera(points_body: np.ndarray, semantic_bev: np.ndarray,
                            bev_bounds: dict, camera: TrainingCameraModel,
                            danger_id: int = 9, fill_unobserved: bool = True,
                            semantic_bev_valid: np.ndarray | None = None,
                            semantic_fill_radius_px: float = 0.0):
    """Project depth and matching HALSS labels with one shared z-buffer.

    Returns sparse_depth, depth_valid_mask, semantic_map,
    semantic_valid_mask. Unknown semantic pixels are conservatively encoded as
    ``danger_id``; ``semantic_valid_mask`` keeps unknown distinct for diagnostics.

    ``fill_unobserved=True`` (default) keeps the historical behaviour: semantic
    labels are NN-filled inside the projected convex hull of observed pixels.
    ``fill_unobserved=False`` keeps conservative behaviour. If
    ``semantic_fill_radius_px > 0``, only small holes inside the semantic seed
    convex hull and within that distance of a geometrically supported seed are
    reconstructed; larger unobserved gaps remain unknown.
    """
    h, w = camera.output_height, camera.output_width
    depth = np.full((h, w), camera.far_m, dtype=np.float32)
    semantic = np.full((h, w), int(danger_id), dtype=np.uint8)
    valid_mask = np.zeros((h, w), dtype=bool)

    if points_body is None:
        return depth, valid_mask, semantic, valid_mask.copy()
    pts = np.asarray(points_body, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[1] < 3 or len(pts) == 0:
        return depth, valid_mask, semantic, valid_mask.copy()
    pts = pts[:, :3]
    finite = np.isfinite(pts).all(axis=1)
    z = pts[:, 2]
    keep = finite & (z > camera.near_m) & (z < camera.far_m)
    pts = pts[keep]
    if len(pts) == 0:
        return depth, valid_mask, semantic, valid_mask.copy()

    point_labels, point_label_valid = _sample_bev_labels(
        pts, semantic_bev, bev_bounds, danger_id,
        semantic_bev_valid=semantic_bev_valid)
    z = pts[:, 2]
    x_camera = -pts[:, 1]
    y_camera = -pts[:, 0]
    u = np.floor(camera.fx * x_camera / z + camera.cx).astype(np.int32)
    v = np.floor(camera.fy * y_camera / z + camera.cy).astype(np.int32)
    inside = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    u, v, z = u[inside], v[inside], z[inside]
    point_labels = point_labels[inside]
    point_label_valid = point_label_valid[inside]
    if len(z) == 0:
        return depth, valid_mask, semantic, valid_mask.copy()

    flat = v * w + u
    # Vectorized fixed-size z-buffer avoids sorting the full input cloud.  The
    # second pass selects the corresponding nearest semantic label; exact-depth
    # ties keep their first occurrence deterministically.
    depth_flat = depth.ravel()
    np.minimum.at(depth_flat, flat, z)
    nearest_candidates = np.flatnonzero(z == depth_flat[flat])
    candidate_flat = flat[nearest_candidates]
    _, first = np.unique(candidate_flat, return_index=True)
    chosen = nearest_candidates[first]
    chosen_flat = flat[chosen]
    semantic.ravel()[chosen_flat] = point_labels[chosen]
    valid_mask.ravel()[chosen_flat] = True
    # 深度有效不等于语义有效。局部平面支持不足的点仍贡献深度，
    # 但不能被默认解释成安全或危险语义。
    semantic_seed = np.zeros((h, w), dtype=bool)
    semantic_seed.ravel()[chosen_flat] = point_label_valid[chosen]

    if fill_unobserved:
        # HALSS is a surface classifier. Fill only the observed projected convex
        # hull; pixels outside remain explicitly unknown/danger instead of fake
        # coverage.
        semantic_valid = np.zeros((h, w), dtype=np.uint8)
        coords = np.column_stack(np.where(semantic_seed))[:, ::-1].astype(np.int32)
        if len(coords) >= 3:
            cv2.fillConvexPoly(semantic_valid, cv2.convexHull(coords), 1)
        else:
            semantic_valid[semantic_seed] = 1
        semantic_valid = semantic_valid.astype(bool)
    else:
        semantic_valid = semantic_seed.copy()
        radius = max(float(semantic_fill_radius_px), 0.0)
        if radius > 0.0 and semantic_seed.any():
            hull = np.zeros((h, w), dtype=np.uint8)
            coords = np.column_stack(np.where(semantic_seed))[:, ::-1].astype(np.int32)
            if len(coords) >= 3:
                cv2.fillConvexPoly(hull, cv2.convexHull(coords), 1)
            else:
                hull[semantic_seed] = 1
            distance = cv2.distanceTransform(
                (~semantic_seed).astype(np.uint8), cv2.DIST_L2, 5)
            semantic_valid = hull.astype(bool) & (distance <= radius)

    if semantic_seed.any() and np.any(semantic_valid & ~semantic_seed):
        _, nearest_labels = cv2.distanceTransformWithLabels(
            (~semantic_seed).astype(np.uint8), cv2.DIST_L2, 5,
            cv2.DIST_LABEL_PIXEL)
        filled_all = fill_valid_nearest(semantic, semantic_seed, nearest_labels)
        fill = semantic_valid & ~semantic_seed
        semantic[fill] = filled_all[fill]

    semantic[~semantic_valid] = np.uint8(danger_id)
    return depth, valid_mask, semantic, semantic_valid
