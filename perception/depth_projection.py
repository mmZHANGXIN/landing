"""
Down-looking depth projection for Mid360 / FAST-LIO point clouds.

The flight pipeline uses the HALSS-aligned body ROI path:
  world_to_level_body_roi() -> body points (x=forward, y=right, z=down)
  body ROI -> image grid with the same axis convention as HALSS safe_mesh
  z-buffer aggregation by nearest positive down distance.

The world-frame perspective projection is kept for geometry tests, ablations,
and log comparison.
"""

import numpy as np
import cv2


class DepthProjector:
    def __init__(self, img_width=128, img_height=128, max_range=30.0,
                 camera_fov=90.0, camera_width=752, camera_height=480,
                 mode="perspective", fx=None, fy=None, cx=None, cy=None,
                 R_I_to_C=None, backend="numpy"):
        self.out_w = img_width
        self.out_h = img_height
        self.max_range = max_range
        self.grid = max_range / img_width * 2  # 米/像素
        self.mode = mode
        self.backend = (backend or "numpy").lower()
        if self.backend in ("cuda", "torch"):
            self.backend = "torch_cuda"
        if self.backend not in ("numpy", "torch_cuda"):
            raise ValueError("depth projection backend must be 'numpy' or 'torch_cuda'")
        if fx is None or fy is None:
            f = 0.5 * img_width / np.tan(np.deg2rad(camera_fov) * 0.5)
            fx = f if fx is None else fx
            fy = f if fy is None else fy
        self.fx = float(fx)
        self.fy = float(fy)
        self.cx = float((img_width - 1) * 0.5 if cx is None else cx)
        self.cy = float((img_height - 1) * 0.5 if cy is None else cy)
        if R_I_to_C is None:
            # Body I: x=forward, y=right, z=down. Camera C: X=right, Y=forward, Z=down.
            R_I_to_C = [[0.0, 1.0, 0.0],
                        [1.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0]]
        self.R_I_to_C = np.asarray(R_I_to_C, dtype=np.float32)
        if self.R_I_to_C.shape != (3, 3):
            raise ValueError("R_I_to_C must be a 3x3 matrix")

    def _source_grid_shape(self, source_shape):
        if source_shape is None:
            return int(self.out_h), int(self.out_w)
        shape = getattr(source_shape, "shape", source_shape)
        if len(shape) < 2:
            raise ValueError("source_shape must have at least two dimensions")
        src_h, src_w = int(shape[0]), int(shape[1])
        if src_h <= 0 or src_w <= 0:
            raise ValueError("source_shape dimensions must be positive")
        return src_h, src_w

    def project(self, points_world, drone_pose):
        """World point cloud -> sparse depth map in meters."""
        if self.mode == "bev":
            return self._project_bev(points_world, drone_pose)
        return self._project_perspective(points_world, drone_pose)

    def project_body_roi(self, points_body, source_shape=None):
        """HALSS body ROI -> sparse depth map aligned with the semantic map.

        points_body must already be in the ego-centered HALSS frame:
        x=forward, y=right, z=down. This intentionally shares the same
        LiDAR-origin correction, yaw-only alignment, and down-range filtering
        performed by perception.halss_preprocess.world_to_level_body_roi().

        source_shape can be set to the HALSS safe_mesh shape. In that case
        depth is first z-buffered on the same source grid as the semantic map,
        then nearest-neighbor resized to the configured observation size.
        """
        src_h, src_w = self._source_grid_shape(source_shape)
        if self.backend == "torch_cuda":
            depth = self._project_body_roi_torch_cuda(points_body, src_h, src_w)
        else:
            depth = self._project_body_roi_numpy(points_body, src_h, src_w)
        if depth.shape != (self.out_h, self.out_w):
            depth = cv2.resize(
                depth, (self.out_w, self.out_h), interpolation=cv2.INTER_NEAREST
            )
        return depth.astype(np.float32, copy=False)

    def _project_perspective(self, points_world, drone_pose):
        """Perspective z-buffer projection into a down-looking camera frame."""
        if self.backend == "torch_cuda":
            return self._project_perspective_torch_cuda(points_world, drone_pose)
        return self._project_perspective_numpy(points_world, drone_pose)

    def _project_perspective_numpy(self, points_world, drone_pose):
        """NumPy perspective z-buffer projection; kept for tests/ablation."""
        if points_world is None:
            return np.full((self.out_h, self.out_w), self.max_range, np.float32)
        pts = np.asarray(points_world, dtype=np.float32)
        if pts.size == 0:
            return np.full((self.out_h, self.out_w), self.max_range, np.float32)
        if pts.ndim == 1:
            if pts.size < 3:
                raise ValueError("points_world must contain x,y,z coordinates")
            pts = pts[:3].reshape(1, 3)
        elif pts.ndim == 2 and pts.shape[1] >= 3:
            pts = pts[:, :3]
        else:
            raise ValueError("points_world must have shape (N,3) or (N,>=3)")
        pose = np.asarray(drone_pose, dtype=np.float32)
        if pose.size < 6:
            raise ValueError("drone_pose must be [x,y,z,roll,pitch,yaw]")

        p_g = pose[:3]
        roll, pitch, yaw = pose[3], pose[4], pose[5]
        R_gi = self._rot_zyx(roll, pitch, yaw)

        # I_k p_i = R_GI^T * (G p_i - G p_I)
        pts_i = (pts - p_g) @ R_gi
        pts_c = pts_i @ self.R_I_to_C.T

        z = pts_c[:, 2]
        valid_z = (z > 0.01) & (z < self.max_range)
        pts_c = pts_c[valid_z]
        z = z[valid_z]
        if len(z) == 0:
            return np.full((self.out_h, self.out_w), self.max_range, np.float32)

        u = np.floor(self.fx * pts_c[:, 0] / z + self.cx).astype(np.int32)
        v = np.floor(self.fy * pts_c[:, 1] / z + self.cy).astype(np.int32)
        inside = (u >= 0) & (u < self.out_w) & (v >= 0) & (v < self.out_h)
        u, v, z = u[inside], v[inside], z[inside]
        if len(z) == 0:
            return np.full((self.out_h, self.out_w), self.max_range, np.float32)

        depth = np.full((self.out_h, self.out_w), self.max_range, np.float32)
        np.minimum.at(depth.ravel(), v * self.out_w + u, z.astype(np.float32))
        return depth

    def _project_body_roi_numpy(self, points_body, src_h, src_w):
        """NumPy HALSS-aligned body ROI z-buffer projection."""
        if points_body is None:
            return np.full((src_h, src_w), self.max_range, np.float32)
        pts = np.asarray(points_body, dtype=np.float32)
        if pts.size == 0:
            return np.full((src_h, src_w), self.max_range, np.float32)
        if pts.ndim == 1:
            if pts.size < 3:
                raise ValueError("points_body must contain x,y,z coordinates")
            pts = pts[:3].reshape(1, 3)
        elif pts.ndim == 2 and pts.shape[1] >= 3:
            pts = pts[:, :3]
        else:
            raise ValueError("points_body must have shape (N,3) or (N,>=3)")

        valid = np.isfinite(pts).all(axis=1)
        pts = pts[valid]
        if len(pts) == 0:
            return np.full((src_h, src_w), self.max_range, np.float32)

        x_all = pts[:, 0]
        y_all = pts[:, 1]
        x_min, x_max = float(np.min(x_all)), float(np.max(x_all))
        y_min, y_max = float(np.min(y_all)), float(np.max(y_all))
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

        z = pts[:, 2]
        valid_z = (z > 0.01) & (z < self.max_range)
        pts = pts[valid_z]
        if len(pts) == 0:
            return np.full((src_h, src_w), self.max_range, np.float32)

        x = pts[:, 0]
        y = pts[:, 1]
        z = pts[:, 2]
        u = np.rint((x - x_min) / x_span * (src_w - 1)).astype(np.int32)
        v_unflipped = np.rint((y - y_min) / y_span * (src_h - 1)).astype(np.int32)
        v = (src_h - 1) - v_unflipped
        u = np.clip(u, 0, src_w - 1)
        v = np.clip(v, 0, src_h - 1)

        depth = np.full((src_h, src_w), self.max_range, np.float32)
        np.minimum.at(depth.ravel(), v * src_w + u, z.astype(np.float32))
        return depth

    def _project_perspective_torch_cuda(self, points_world, drone_pose):
        """CUDA perspective z-buffer projection with the same geometry as NumPy."""
        try:
            import torch
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "depth_projection.backend='torch_cuda' requires PyTorch"
            ) from exc
        if not torch.cuda.is_available():
            raise RuntimeError(
                "depth_projection.backend='torch_cuda' requires CUDA; "
                "flight depth projection must not fall back to CPU"
            )
        if points_world is None:
            return np.full((self.out_h, self.out_w), self.max_range, np.float32)

        device = torch.device("cuda")
        pts = torch.as_tensor(points_world, dtype=torch.float32, device=device)
        if pts.numel() == 0:
            return np.full((self.out_h, self.out_w), self.max_range, np.float32)
        if pts.dim() == 1:
            if pts.numel() < 3:
                raise ValueError("points_world must contain x,y,z coordinates")
            pts = pts[:3].reshape(1, 3)
        elif pts.dim() == 2 and pts.shape[1] >= 3:
            pts = pts[:, :3]
        else:
            raise ValueError("points_world must have shape (N,3) or (N,>=3)")

        pose = torch.as_tensor(drone_pose, dtype=torch.float32, device=device)
        if pose.numel() < 6:
            raise ValueError("drone_pose must be [x,y,z,roll,pitch,yaw]")

        p_g = pose[:3]
        roll, pitch, yaw = pose[3], pose[4], pose[5]
        R_gi = self._rot_zyx_torch(roll, pitch, yaw, torch)
        R_i_to_c = torch.as_tensor(self.R_I_to_C, dtype=torch.float32, device=device)

        pts_i = (pts - p_g) @ R_gi
        pts_c = pts_i @ R_i_to_c.T

        z = pts_c[:, 2]
        valid_z = (z > 0.01) & (z < float(self.max_range))
        pts_c = pts_c[valid_z]
        z = z[valid_z]
        if z.numel() == 0:
            return np.full((self.out_h, self.out_w), self.max_range, np.float32)

        u = torch.floor(float(self.fx) * pts_c[:, 0] / z + float(self.cx)).to(torch.long)
        v = torch.floor(float(self.fy) * pts_c[:, 1] / z + float(self.cy)).to(torch.long)
        inside = (u >= 0) & (u < self.out_w) & (v >= 0) & (v < self.out_h)
        u, v, z = u[inside], v[inside], z[inside]
        if z.numel() == 0:
            return np.full((self.out_h, self.out_w), self.max_range, np.float32)

        depth_flat = torch.full(
            (self.out_h * self.out_w,),
            float(self.max_range),
            dtype=torch.float32,
            device=device,
        )
        if not hasattr(depth_flat, "scatter_reduce_"):
            raise RuntimeError(
                "depth_projection.backend='torch_cuda' requires "
                "torch.Tensor.scatter_reduce_ for GPU z-buffer aggregation"
            )
        linear = v * self.out_w + u
        if hasattr(depth_flat, "scatter_reduce_"):
            depth_flat.scatter_reduce_(0, linear, z.float(), reduce="amin", include_self=True)
        else:
            # Older Jetson PyTorch builds may not expose scatter_reduce_.
            # Sort by pixel then depth; the first entry per pixel is the z-buffer minimum.
            sort_key = linear.float() * (float(self.max_range) + 1.0) + z.float()
            order = torch.argsort(sort_key)
            sorted_linear = linear[order]
            sorted_z = z.float()[order]
            keep = torch.ones_like(sorted_linear, dtype=torch.bool)
            keep[1:] = sorted_linear[1:] != sorted_linear[:-1]
            depth_flat[sorted_linear[keep]] = sorted_z[keep]
        return depth_flat.reshape(self.out_h, self.out_w).detach().cpu().numpy().astype(np.float32)

    def _project_body_roi_torch_cuda(self, points_body, src_h, src_w):
        """CUDA HALSS-aligned body ROI z-buffer projection."""
        try:
            import torch
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "depth_projection.backend='torch_cuda' requires PyTorch"
            ) from exc
        if not torch.cuda.is_available():
            raise RuntimeError(
                "depth_projection.backend='torch_cuda' requires CUDA; "
                "flight depth projection must not fall back to CPU"
            )
        if points_body is None:
            return np.full((src_h, src_w), self.max_range, np.float32)

        device = torch.device("cuda")
        pts = torch.as_tensor(points_body, dtype=torch.float32, device=device)
        if pts.numel() == 0:
            return np.full((src_h, src_w), self.max_range, np.float32)
        if pts.dim() == 1:
            if pts.numel() < 3:
                raise ValueError("points_body must contain x,y,z coordinates")
            pts = pts[:3].reshape(1, 3)
        elif pts.dim() == 2 and pts.shape[1] >= 3:
            pts = pts[:, :3]
        else:
            raise ValueError("points_body must have shape (N,3) or (N,>=3)")

        pts = pts[torch.isfinite(pts).all(dim=1)]
        if pts.numel() == 0:
            return np.full((src_h, src_w), self.max_range, np.float32)

        x_all = pts[:, 0]
        y_all = pts[:, 1]
        x_min, x_max = torch.min(x_all), torch.max(x_all)
        y_min, y_max = torch.min(y_all), torch.max(y_all)
        x_span = x_max - x_min
        y_span = y_max - y_min
        if bool(x_span < 1e-6):
            x_min = x_min - 0.5
            x_max = x_max + 0.5
            x_span = x_max - x_min
        if bool(y_span < 1e-6):
            y_min = y_min - 0.5
            y_max = y_max + 0.5
            y_span = y_max - y_min

        z = pts[:, 2]
        valid_z = (z > 0.01) & (z < float(self.max_range))
        pts = pts[valid_z]
        if pts.numel() == 0:
            return np.full((src_h, src_w), self.max_range, np.float32)

        x = pts[:, 0]
        y = pts[:, 1]
        z = pts[:, 2]
        u = torch.round((x - x_min) / x_span * float(src_w - 1)).to(torch.long)
        v_unflipped = torch.round((y - y_min) / y_span * float(src_h - 1)).to(torch.long)
        v = int(src_h - 1) - v_unflipped
        u = torch.clamp(u, 0, src_w - 1)
        v = torch.clamp(v, 0, src_h - 1)

        depth_flat = torch.full(
            (src_h * src_w,),
            float(self.max_range),
            dtype=torch.float32,
            device=device,
        )
        if not hasattr(depth_flat, "scatter_reduce_"):
            raise RuntimeError(
                "depth_projection.backend='torch_cuda' requires "
                "torch.Tensor.scatter_reduce_ for GPU z-buffer aggregation"
            )
        linear = v * src_w + u
        depth_flat.scatter_reduce_(0, linear, z.float(), reduce="amin", include_self=True)
        return depth_flat.reshape(src_h, src_w).detach().cpu().numpy().astype(np.float32)

    def _project_bev(self, points_world, drone_pose):
        """World point cloud -> BEV height-grid depth map in meters."""
        if points_world is None or len(points_world) < 3:
            return np.full((self.out_h, self.out_w), self.max_range, np.float32)

        dz = drone_pose[2]
        hr = self.max_range / 2
        # XY 裁切
        dx = points_world[:,0] - drone_pose[0]
        dy = points_world[:,1] - drone_pose[1]
        ok = (np.abs(dx) < hr) & (np.abs(dy) < hr)
        pts = points_world[ok]
        if len(pts) < 3:
            return np.full((self.out_h, self.out_w), self.max_range, np.float32)

        # 只取下方点
        below = pts[:,2] > dz
        pb = pts[below]
        if len(pb) < 3:
            return np.full((self.out_h, self.out_w), self.max_range, np.float32)

        # XY → 像素
        c = ((pb[:,0] - drone_pose[0] + hr) / self.grid).astype(int)
        r = ((hr - pb[:,1] + drone_pose[1]) / self.grid).astype(int)
        v = (c >= 0) & (c < self.out_w) & (r >= 0) & (r < self.out_h)
        c, r = c[v], r[v]; z = pb[v, 2]
        if len(c) == 0:
            return np.full((self.out_h, self.out_w), self.max_range, np.float32)

        # 每 cell 取最近的表面 (最小 Z = 最高点)
        s = np.full((self.out_h, self.out_w), np.inf, np.float32)
        np.minimum.at(s.ravel(), r * self.out_w + c, z)
        m = np.isfinite(s)
        d = np.full_like(s, self.max_range)
        # NED: Z+=down, 深度 = 表面Z - 无人机Z (正值=下方)
        d[m] = np.clip(s[m] - dz, 0, self.max_range)

        # 小空洞填充
        if m.sum() > 20:
            dm = cv2.dilate(m.astype(np.uint8), np.ones((3,3),np.uint8), iterations=1)
            fm = dm.astype(bool) & ~m
            if fm.sum() > 0:
                df = cv2.medianBlur(d.astype(np.float32), 3); d[fm] = df[fm]

        return d.astype(np.float32)

    @staticmethod
    def _rot_zyx(roll, pitch, yaw):
        """Body-to-world rotation for roll/pitch/yaw in NED convention."""
        cr, sr = np.cos(roll), np.sin(roll)
        cp, sp = np.cos(pitch), np.sin(pitch)
        cy, sy = np.cos(yaw), np.sin(yaw)
        return np.array([
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ], dtype=np.float32)

    @staticmethod
    def _rot_zyx_torch(roll, pitch, yaw, torch):
        """Body-to-world rotation for roll/pitch/yaw in NED convention."""
        cr, sr = torch.cos(roll), torch.sin(roll)
        cp, sp = torch.cos(pitch), torch.sin(pitch)
        cy, sy = torch.cos(yaw), torch.sin(yaw)
        return torch.stack([
            torch.stack([cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr]),
            torch.stack([sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr]),
            torch.stack([-sp, cp * sr, cp * cr]),
        ]).to(dtype=torch.float32)
