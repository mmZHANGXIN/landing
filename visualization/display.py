"""
实时可视化模块
显示语义图、深度图、BEV 安全评估图和飞行轨迹。
"""

import numpy as np
import cv2
import logging
from collections import deque

logger = logging.getLogger("Visualizer")


class RealtimeVisualizer:
    """
    实时多窗口可视化

    四个窗口:
      binary semantic    — 二值安全语义图 (彩色)
      2. Depth Map       — 深度图 (深度相机式灰度)
      3. BEV Safety      — 俯视安全评估 (红/绿)
      4. Flight Trajectory — 3D 飞行轨迹 (Matplotlib 或 2D 俯视)
    """

    def __init__(self, cfg: dict):
        self.enabled = cfg.get("enable", True)
        self.disp_w = cfg.get("display_width", 300)
        self.disp_h = cfg.get("display_height", 300)
        self.save_frames = cfg.get("save_frames", False)
        self.save_dir = cfg.get("save_dir", "./experiments/frames")
        self.depth_vmax_m = float(cfg.get("depth_vmax_m", 30.0))
        self._windows_ready = False

        # 轨迹缓存 (NED 坐标)
        self._trajectory = deque(maxlen=500)
        self._trajectory_2d = deque(maxlen=500)
        self._safe_class_id = 1
        self._danger_class_id = 9
        self._semantic_window = cfg.get("binary_semantic_window_title", "binary semantic")

        if self.save_frames:
            import os
            os.makedirs(self.save_dir, exist_ok=True)
            self._frame_idx = 0

    def _init_windows(self):
        cv2.namedWindow(self._semantic_window, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self._semantic_window, self.disp_w, self.disp_h)
        cv2.moveWindow(self._semantic_window, 20, 50)

        cv2.namedWindow("2.Depth Map", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("2.Depth Map", self.disp_w, self.disp_h)
        cv2.moveWindow("2.Depth Map", 20 + self.disp_w + 10, 50)

        cv2.namedWindow("3.BEV Safety", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("3.BEV Safety", self.disp_w, self.disp_h)
        cv2.moveWindow("3.BEV Safety", 20, 50 + self.disp_h + 40)

        cv2.namedWindow("4.Trajectory (Top-Down)", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("4.Trajectory (Top-Down)", self.disp_w, self.disp_h)
        cv2.moveWindow("4.Trajectory (Top-Down)", 20 + self.disp_w + 10, 50 + self.disp_h + 40)

        self._windows_ready = True

    def update(self, sem_map: np.ndarray, depth_map: np.ndarray,
               safety_bev: np.ndarray = None, drone_pose: np.ndarray = None,
               binary_semantic_vis: np.ndarray = None):
        """更新所有可视化窗口"""
        if not self.enabled:
            return

        if not self._windows_ready:
            self._init_windows()

        # ---- 1. 语义图 ----
        sem_vis = self._render_binary_semantic(binary_semantic_vis, sem_map)
        cv2.imshow(self._semantic_window, sem_vis)

        # ---- 2. 深度图 ----
        depth_vis = self._render_depth(depth_map)
        cv2.imshow("2.Depth Map", depth_vis)

        # ---- 3. BEV 安全评估 ----
        if safety_bev is not None:
            bev_vis = self._render_bev(safety_bev)
            cv2.imshow("3.BEV Safety", bev_vis)

        # ---- 4. 轨迹 ----
        if drone_pose is not None:
            self._trajectory_2d.append((drone_pose[0], drone_pose[1]))
            traj_vis = self._render_trajectory(drone_pose)
            cv2.imshow("4.Trajectory (Top-Down)", traj_vis)

        # ---- 保存帧 ----
        if self.save_frames:
            for name, img in [("binary_semantic", sem_vis), ("depth", depth_vis)]:
                cv2.imwrite(f"{self.save_dir}/{self._frame_idx:06d}_{name}.png", img)
            self._frame_idx += 1

        cv2.waitKey(1)

    def _render_semantic(self, sem_map: np.ndarray) -> np.ndarray:
        """语义图 → 彩色图像"""
        h, w = sem_map.shape[:2]
        vis = np.zeros((h, w, 3), dtype=np.uint8)
        vis[sem_map == self._safe_class_id] = [0, 255, 0]      # 安全=绿
        vis[sem_map == self._danger_class_id] = [0, 0, 255]    # 危险=红
        # 其他=灰
        mask = (sem_map != self._safe_class_id) & (sem_map != self._danger_class_id)
        vis[mask] = [128, 128, 128]
        return cv2.resize(vis, (self.disp_w, self.disp_h))

    def _render_binary_semantic(self, binary_vis: np.ndarray, sem_map: np.ndarray) -> np.ndarray:
        """HALSS binary safety visualization passthrough with semantic fallback."""
        if binary_vis is None:
            return self._render_semantic(sem_map)
        vis = binary_vis
        if vis.ndim == 2:
            vis = cv2.cvtColor(vis.astype(np.uint8), cv2.COLOR_GRAY2BGR)
        elif vis.shape[2] == 3:
            vis = vis.astype(np.uint8)
        else:
            return self._render_semantic(sem_map)
        return cv2.resize(vis, (self.disp_w, self.disp_h), interpolation=cv2.INTER_NEAREST)

    def _render_depth(self, depth_map: np.ndarray) -> np.ndarray:
        """Metric depth camera-style grayscale: 0m=black, vmax=white."""
        depth_m = np.nan_to_num(
            depth_map.astype(np.float32, copy=False),
            nan=self.depth_vmax_m,
            posinf=self.depth_vmax_m,
            neginf=0.0,
        )
        depth_norm = np.clip(depth_m / self.depth_vmax_m, 0.0, 1.0)
        depth_u8 = (depth_norm * 255.0).astype(np.uint8)
        return cv2.resize(depth_u8, (self.disp_w, self.disp_h), interpolation=cv2.INTER_NEAREST)

    def _render_bev(self, safety_bev: np.ndarray) -> np.ndarray:
        """BEV 安全网格 → 彩色图像"""
        vis = np.zeros((safety_bev.shape[0], safety_bev.shape[1], 3), dtype=np.uint8)
        vis[safety_bev] = [0, 255, 0]
        vis[~safety_bev] = [0, 0, 255]
        return cv2.resize(vis, (self.disp_w, self.disp_h))

    def _render_trajectory(self, drone_pose: np.ndarray) -> np.ndarray:
        """绘制 2D 俯视轨迹"""
        canvas_size = 500
        canvas = np.ones((canvas_size, canvas_size, 3), dtype=np.uint8) * 240

        if len(self._trajectory_2d) < 2:
            return canvas

        # 计算尺度 (世界坐标 → 像素)
        xs = np.array([p[0] for p in self._trajectory_2d])
        ys = np.array([p[1] for p in self._trajectory_2d])
        x_range = max(xs.max() - xs.min(), 5.0)
        y_range = max(ys.max() - ys.min(), 5.0)
        scale = min(canvas_size * 0.8 / x_range, canvas_size * 0.8 / y_range)
        cx = canvas_size // 2
        cy = canvas_size // 2

        # 绘制轨迹线
        for i in range(1, len(self._trajectory_2d)):
            px1 = int(cx + (xs[i - 1] - xs.mean()) * scale)
            py1 = int(cy - (ys[i - 1] - ys.mean()) * scale)
            px2 = int(cx + (xs[i] - xs.mean()) * scale)
            py2 = int(cy - (ys[i] - ys.mean()) * scale)
            cv2.line(canvas, (px1, py1), (px2, py2), (255, 0, 0), 2)

        # 绘制当前位置
        px = int(cx + (drone_pose[0] - xs.mean()) * scale)
        py = int(cy - (drone_pose[1] - ys.mean()) * scale)
        cv2.circle(canvas, (px, py), 6, (0, 0, 255), -1)

        # 绘制起点和终点
        px0 = int(cx + (xs[0] - xs.mean()) * scale)
        py0 = int(cy - (ys[0] - ys.mean()) * scale)
        cv2.circle(canvas, (px0, py0), 4, (0, 255, 0), -1)

        return canvas

    def close(self):
        cv2.destroyAllWindows()
