"""实时并排显示二值语义图和深度图。"""

import numpy as np
import cv2
import logging

logger = logging.getLogger("Visualizer")


class RealtimeVisualizer:
    """
    实时双窗口可视化：二值安全语义图 + 深度图。
    """

    def __init__(self, cfg: dict):
        self.enabled = cfg.get("enable", True)
        self.disp_w = cfg.get("display_width", 300)
        self.disp_h = cfg.get("display_height", 300)
        self.save_frames = cfg.get("save_frames", False)
        self.save_dir = cfg.get("save_dir", "./experiments/frames")
        self.depth_vmax_m = float(cfg.get("depth_vmax_m", 30.0))
        # A metric 0..30 m colour map is hard to read during the high-altitude
        # part of a landing.  ``disparity`` keeps the policy input untouched,
        # but makes nearby returns bright like a depth camera display.
        self.depth_display_mode = str(cfg.get("depth_display_mode", "disparity")).lower()
        self.depth_near_m = max(0.01, float(cfg.get("depth_near_m", 0.5)))
        self.depth_disparity_gamma = max(1.0, float(cfg.get("depth_disparity_gamma", 4.0)))
        self._windows_ready = False

        self._safe_class_id = 1
        self._danger_class_id = 9
        self._semantic_window = cfg.get("binary_semantic_window_title", "1.Binary Semantic Map")

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

        self._windows_ready = True

    def update(self, sem_map: np.ndarray, depth_map: np.ndarray,
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

        # ---- 保存帧 ----
        if self.save_frames:
            for name, img in [("binary_semantic", sem_vis), ("depth", depth_vis)]:
                cv2.imwrite(f"{self.save_dir}/{self._frame_idx:06d}_{name}.png", img)
            self._frame_idx += 1

        cv2.waitKey(1)

    def _render_semantic(self, sem_map: np.ndarray) -> np.ndarray:
        """语义图 → 黑白图像: 白色=安全, 黑色=危险, 灰色=未知 (对齐 test_live_nocontrol.py)."""
        h, w = sem_map.shape[:2]
        vis = np.full((h, w), 128, dtype=np.uint8)  # 灰色=未知
        vis[sem_map == self._safe_class_id] = 255     # 白色=安全
        vis[sem_map == self._danger_class_id] = 0      # 黑色=危险
        return cv2.resize(vis, (self.disp_w, self.disp_h), interpolation=cv2.INTER_NEAREST)

    def _render_binary_semantic(self, binary_vis: np.ndarray, sem_map: np.ndarray) -> np.ndarray:
        """HALSS binary safety visualization passthrough with semantic fallback."""
        if binary_vis is not None:
            vis = binary_vis
            if vis.ndim == 3:
                vis = cv2.cvtColor(vis.astype(np.uint8), cv2.COLOR_BGR2GRAY)
        else:
            vis = self._render_semantic(sem_map)
        vis_bgr = cv2.cvtColor(vis.astype(np.uint8), cv2.COLOR_GRAY2BGR)
        return cv2.resize(vis_bgr, (self.disp_w, self.disp_h), interpolation=cv2.INTER_NEAREST)

    def _render_depth(self, depth_map: np.ndarray) -> np.ndarray:
        """深度图 → depth-camera style disparity image + scale bar.

        The input remains metric depth.  Only this display path uses the
        inverse/nonlinear mapping, so changing visual contrast cannot change
        the ONNX observation.
        """
        depth_m = np.nan_to_num(
            depth_map.astype(np.float32, copy=False),
            nan=self.depth_vmax_m, posinf=self.depth_vmax_m, neginf=0.0,
        )
        depth_m = np.clip(depth_m, self.depth_near_m, self.depth_vmax_m)
        if self.depth_display_mode in {"disparity", "camera", "depth_camera"}:
            # gamma > 1 expands contrast close to the far clipping plane:
            # at 30 m, 25 m is visibly different from 28 m.
            distance_norm = np.clip(
                (depth_m - self.depth_near_m)
                / max(self.depth_vmax_m - self.depth_near_m, 1e-6),
                0.0, 1.0,
            )
            depth_norm = 1.0 - np.power(distance_norm, self.depth_disparity_gamma)
            scale_label = "near->far"
        else:
            depth_norm = np.clip(depth_m / self.depth_vmax_m, 0.0, 1.0)
            scale_label = "0\u2192far"
        depth_u8 = np.round(depth_norm * 255.0).astype(np.uint8)
        depth_resized = cv2.resize(depth_u8, (self.disp_w, self.disp_h), interpolation=cv2.INTER_NEAREST)
        colored = cv2.applyColorMap(depth_resized, cv2.COLORMAP_INFERNO)

        # 右侧色条 (40px 宽)
        bar_w = 40
        h = self.disp_h
        with_bar = np.zeros((h, self.disp_w + bar_w + 10, 3), dtype=np.uint8)
        with_bar[:, :self.disp_w] = colored
        # 画色条
        for row in range(h):
            val = 255 - int(row / max(h - 1, 1) * 255)  # 上=近, 下=远
            with_bar[row, self.disp_w + 5:self.disp_w + bar_w + 5] = cv2.applyColorMap(
                np.array([[val]], dtype=np.uint8), cv2.COLORMAP_INFERNO)[0, 0]
        # 色条标注
        bar_x = self.disp_w + 5
        near_label = f"{self.depth_near_m:g}m" if self.depth_display_mode in {"disparity", "camera", "depth_camera"} else "0m"
        cv2.putText(with_bar, near_label, (bar_x - 5, 15 if near_label != "0m" else h - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(with_bar, f"{int(self.depth_vmax_m)}m", (bar_x - 5, h - 5 if near_label != "0m" else 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.putText(with_bar, scale_label, (5, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        return with_bar

    def close(self):
        cv2.destroyAllWindows()
