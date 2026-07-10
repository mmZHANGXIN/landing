"""
原始 HALSS 安全评估模块
======================
完整移植自 HALO-master/HALSS/ 原始算法:
  点云 → 降采样(2m cell) → Delaunay三角剖分 → 插值高程图 →
  表面法线计算 → 角度阈值 → 骨架提取 → 圆圈着陆区检测 → 二值语义图

本模块仅做算法移植，不修改任何核心逻辑。
适配 Orin 真机部署: 移除 AirSim/RflySim 依赖, 输出兼容 pipeline API。
"""

import numpy as np
import cv2
import math
import time
from scipy import interpolate
from scipy.ndimage import gaussian_filter
from skimage.morphology import medial_axis


# ============================================================
# 核心算法函数 (从 HALSS/halss_utils.py + seg_utils.py 移植)
# ============================================================

def _surface_normal_from_interp(x_grid, y_grid, Z_interp, grid_res):
    """Delaunay 插值高程图 → 表面法线彩色图"""
    grad = np.gradient(Z_interp, y_grid, x_grid)
    dx, dy = grad[1], grad[0]
    normal = np.zeros((grid_res, grid_res, 3))
    normal[:, :, 0] = -dx
    normal[:, :, 1] = -dy
    normal[:, :, 2] = np.ones((grid_res, grid_res))
    n = np.linalg.norm(normal, axis=2).reshape(grid_res, grid_res, 1) + 1e-8
    normal_unit = normal / n
    normal_color = (normal_unit + 1) * 255 * 0.5
    r = np.nan_to_num(normal_color[:, :, 1], nan=0)
    g = np.nan_to_num(normal_color[:, :, 0], nan=0)
    b = np.nan_to_num(normal_color[:, :, 2], nan=0)
    return np.flipud(np.dstack((r, g, b)).astype(np.uint8))


def _max_possible_points(pcd, x_cell, y_cell):
    return int((pcd[:, 0].max() - pcd[:, 0].min()) / x_cell) * \
           int((pcd[:, 1].max() - pcd[:, 1].min()) / y_cell) * 2


def _downsample_pointcloud(pcd, x_cell=2.0, y_cell=2.0):
    """网格降采样: 每 cell 取最高和最低点"""
    culled_dict = {}
    for pt in pcd:
        x = int(0.5 + pt[0] / x_cell)
        y = int(0.5 + pt[1] / y_cell)
        if x not in culled_dict:
            culled_dict[x] = {}
        row = culled_dict[x]
        if y not in row:
            row[y] = [pt, pt]
        else:
            if pt[2] < row[y][0][2]:
                row[y][0] = pt
            elif pt[2] > row[y][1][2]:
                row[y][1] = pt

    count = sum(2 * len(row) for row in culled_dict.values())
    culled = np.zeros((count, 3))
    cur = 0
    for row in culled_dict.values():
        for pair in row.values():
            culled[cur] = pair[0]
            culled[cur + 1] = pair[1]
            cur += 2
    return culled


def _pc_to_surf_normal(pcd_culled, grid_res=480):
    """降采样点云 → Delaunay 插值 → 表面法线 BGR 图 (保持原版中间格式)"""
    x_vals = np.linspace(pcd_culled[:, 0].min(), pcd_culled[:, 0].max(), grid_res)
    y_vals = np.linspace(pcd_culled[:, 1].min(), pcd_culled[:, 1].max(), grid_res)
    X, Y = np.meshgrid(x_vals, y_vals)
    f_linear = interpolate.LinearNDInterpolator(
        list(zip(pcd_culled[:, 0], pcd_culled[:, 1])), pcd_culled[:, 2]
    )
    Z = f_linear(X, Y)
    surf_norm_bgr = _surface_normal_from_interp(x_vals, y_vals, Z, grid_res)
    # 转为 RGB 存储 (与原版 pc_to_surf_normal 一致: BGR → RGB)
    surf_norm_rgb = cv2.cvtColor(surf_norm_bgr, cv2.COLOR_BGR2RGB)
    return surf_norm_rgb, surf_norm_bgr


def _surf_norm_thresh(surf_norm_rgb, alpha=10.0):
    """法线角度阈值 → 二值安全图 (角度 < alpha → 安全)"""
    flat_vec = np.array([1, 0, 0])
    norm_png = surf_norm_rgb.astype(np.float32) / 255.0
    h, w = norm_png.shape[:2]
    cos_theta = flat_vec @ norm_png.reshape(h * w, 3).T
    cos_theta_mat = cos_theta.reshape(h, w)
    theta_mat = np.arccos(np.clip(cos_theta_mat, -1, 1)) * 180 / np.pi
    safety = np.zeros((h, w, 3), dtype=np.uint8)
    safety[theta_mat < alpha] = 255
    return safety, theta_mat


def _top_n_circles(distance_map, N=12):
    """从距离变换图取 Top-N 最大内切圆"""
    dm = distance_map.copy()
    centers, radii = [], []
    for _ in range(N):
        c = np.unravel_index(np.argmax(dm), dm.shape)
        r = int(np.max(dm))
        if r <= 0:
            break
        cv2.circle(dm, (c[1], c[0]), r, 0, -1)
        centers.append(c)
        radii.append(r)
    return centers, radii


def _plot_circles(centers, radii, image):
    """在图上画圆圈标注安全着陆区"""
    if len(image.shape) == 2:
        color_img = np.stack([image] * 3, axis=-1)
    else:
        color_img = image.copy()
    for (yc, xc), r in zip(centers, radii):
        cv2.circle(color_img, (xc, yc), int(r), (230, 230, 255), cv2.FILLED)
        cv2.circle(color_img, (xc, yc), int(r), (0, 0, 255), 2)
        cv2.circle(color_img, (xc, yc), 1, (0, 0, 255), -1)
    return color_img


def _landing_selection(safety_rgb, num_circles=12):
    """安全图 → 骨架 + 距离变换 → Top-N 圆圈着陆区"""
    data = safety_rgb[:, :, 0].astype(np.uint8)
    skel, distance = medial_axis(data, return_distance=True)
    kernel = np.ones((2, 2), np.uint8)
    skel_dilated = cv2.dilate(skel.astype(np.uint8), kernel, iterations=1)
    dist_on_skel = distance * skel_dilated
    centers, radii = _top_n_circles(distance, num_circles)
    circles = _plot_circles(centers, radii, data)
    # 骨架热力图
    skeleton = dist_on_skel[2:-2, 2:-2]
    skel_color = cv2.applyColorMap(
        (skeleton / (skeleton.max() + 1e-8) * 255).astype(np.uint8),
        cv2.COLORMAP_HOT
    )
    return circles, skel_color, centers, radii, distance


# ============================================================
# Orin 部署 HALSS 评估器 (兼容 pipeline API)
# ============================================================

class HALSSSafetyEvaluator:
    """
    原始 HALSS 安全评估器 — Delaunay 插值 + 法线 + 圆圈检测

    与 halss_gpu.py 保持相同 API, 可直接替换。
    """

    def __init__(self, cfg: dict):
        self.alpha = cfg.get("slope_threshold_deg", 10.0)
        self.grid_res = cfg.get("grid_resolution_halss", 200)  # 降低到200, 速度提升~6x
        self.x_cell = cfg.get("x_cell_size", 2.0)
        self.y_cell = cfg.get("y_cell_size", 2.0)
        self.max_sites = cfg.get("max_sites", 12)
        self.safe_id = cfg.get("safe_class_id", 1)
        self.danger_id = cfg.get("danger_class_id", 9)
        self.roi = cfg.get("roi_radius_world", 25.0)

    def evaluate(self, points_world: np.ndarray) -> dict:
        """
        输入: (N, 3) 世界坐标点云
        输出: dict (兼容 halss_gpu 格式, 扩展了原始可视化字段)
        """
        if points_world is None or len(points_world) < 10:
            return None

        pts = points_world[:, :3].astype(np.float64)

        # 1. 降采样 (2m cell)
        culled = _downsample_pointcloud(pts, self.x_cell, self.y_cell)
        if len(culled) < 10:
            return None

        # 2. Delaunay 插值 → 表面法线 (RGB + 原始BGR)
        surf_norm_rgb, surf_norm_bgr = _pc_to_surf_normal(culled, self.grid_res)

        # 3. 角度阈值 → 安全二值图
        safety_rgb, angle_map = _surf_norm_thresh(surf_norm_rgb, self.alpha)

        # 4. 着陆区圆圈检测
        circles_bgr, skeleton_bgr, centers, radii, distance = _landing_selection(
            safety_rgb, self.max_sites
        )

        # 5. 高度热力图 (Delaunay 插值)
        x_vals = np.linspace(pts[:, 0].min(), pts[:, 0].max(), self.grid_res)
        y_vals = np.linspace(pts[:, 1].min(), pts[:, 1].max(), self.grid_res)
        X, Y = np.meshgrid(x_vals, y_vals)
        f_linear = interpolate.LinearNDInterpolator(
            list(zip(culled[:, 0], culled[:, 1])), culled[:, 2]
        )
        Z = f_linear(X, Y)

        # 6. 构建兼容输出 + 可视化数据
        safe_mask_2d = safety_rgb[:, :, 0] > 0
        bev_data = {
            "safe_mesh": safe_mask_2d,
            "z_mesh": np.flipud(np.nan_to_num(Z, nan=np.nanmean(Z) if (~np.isnan(Z)).sum() > 0 else 0)),
            "slope_mesh": angle_map,
            "roughness_mesh": distance,
            "rows": self.grid_res,
            "cols": self.grid_res,
            # 原始 HALSS 可视化 (对齐原版)
            # surf_norm_raw: 存中间 RGB 格式 (pc_to_surf_normal 第一轮 BGR2RGB 后),
            #   显示时再做 cv2.cvtColor(BGR2RGB) 恢复原配色
            "surf_norm_raw": surf_norm_rgb,
            "skeleton_raw": skeleton_bgr,
            "circles_raw": circles_bgr,
            "angle_map": angle_map,
            "distance": distance,
        }

        return {
            "safety_probs": safe_mask_2d.astype(float).flatten(),
            "slope_mesh": angle_map,
            "roughness_mesh": distance,
            "bev_data": bev_data,
        }
