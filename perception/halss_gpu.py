"""
HALSS GPU-加速安全评估模块
从 FaultyYawLanding/perception/fine_perception.py 提取优化
适配 Mid360 + FAST-LIO2 去畸变后的世界坐标点云
"""

import numpy as np
import math
import torch
import torch.nn.functional as F


class HALSSSafetyEvaluator:
    """
    HALSS (Heuristic Adaptive LiDAR Safety Scoring)
    从去畸变点云生成 BEV 安全评估结果。

    管线:
      点云(N,3) → BEV栅格化 → GPU迭代空洞填充 →
      全局平滑 → Sobel梯度(坡度) → 局部方差(粗糙度) →
      安全/危险二值 map
    """

    def __init__(self, cfg: dict):
        self.slope_th = cfg.get("slope_threshold_deg", 10.0)
        self.rough_th = cfg.get("roughness_threshold", 0.1)
        self.grid_res = cfg.get("grid_resolution", 0.1)
        self.roi_radius = cfg.get("roi_radius_world", 25.0)
        self.safe_class_id = cfg.get("safe_class_id", 1)
        self.danger_class_id = cfg.get("danger_class_id", 9)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[HALSS] Device: {self.device}")

    def evaluate(self, points_world: np.ndarray) -> dict:
        """
        输入: 世界坐标去畸变点云 (N, 3)
        输出: dict {
            'safety_probs': 每点安全概率 (N,),
            'slope_mesh':  坡度网格,
            'roughness_mesh': 粗糙度网格,
            'bev_data': BEV 完整数据,
        }
        """
        if points_world is None or len(points_world) < 10:
            return None

        X, Y, Z = points_world[:, 0], points_world[:, 1], points_world[:, 2]

        # ---- 1. CPU 栅格化 (取局部最高点) ----
        x_max, x_min = X.max() + 0.5, X.min() - 0.5
        y_max, y_min = Y.max() + 0.5, Y.min() - 0.5
        rows = int(np.ceil((x_max - x_min) / self.grid_res))
        cols = int(np.ceil((y_max - y_min) / self.grid_res))
        rows, cols = max(rows, 5), max(cols, 5)

        ix = np.clip(np.floor((x_max - X) / self.grid_res).astype(int), 0, rows - 1)
        iy = np.clip(np.floor((Y - y_min) / self.grid_res).astype(int), 0, cols - 1)

        z_grid = np.full((rows, cols), -10000.0, dtype=np.float32)
        np.maximum.at(z_grid, (ix, iy), Z)
        mask_invalid = z_grid == -10000.0

        # ---- 2. 转 GPU ----
        z_tensor = (
            torch.from_numpy(z_grid).unsqueeze(0).unsqueeze(0).to(self.device)
        )
        mask_tensor = (
            torch.from_numpy(mask_invalid).unsqueeze(0).unsqueeze(0).to(self.device)
        )

        # ---- 3. 迭代扩散空洞填充 ----
        z_filled = z_tensor.clone()
        current_mask = mask_tensor.clone()
        kernel_size = 5
        padding = kernel_size // 2
        for _ in range(6):
            if not current_mask.any():
                break
            valid_float = (~current_mask).float()
            z_zeroed = torch.where(current_mask, torch.zeros_like(z_filled), z_filled)
            sum_z = F.avg_pool2d(z_zeroed, kernel_size, stride=1, padding=padding,
                                 count_include_pad=False) * (kernel_size ** 2)
            count_valid = F.avg_pool2d(valid_float, kernel_size, stride=1, padding=padding,
                                       count_include_pad=False) * (kernel_size ** 2)
            local_mean = sum_z / (count_valid + 1e-6)
            can_fill = current_mask & (count_valid > 0)
            z_filled = torch.where(can_fill, local_mean, z_filled)
            current_mask = current_mask & (~can_fill)

        # 全局均值填充残余空洞
        if current_mask.any():
            valid_z = z_filled[~current_mask]
            global_mean = valid_z.mean() if valid_z.numel() > 0 else torch.tensor(0.0, device=self.device)
            z_filled[current_mask] = global_mean

        # ---- 4. 全局宏观表面平滑 (5x5) ----
        z_smoothed = F.avg_pool2d(z_filled, kernel_size=5, stride=1, padding=2,
                                  count_include_pad=False)

        # ---- 5. Sobel 梯度 → 坡度 ----
        sobel_x = (
            torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                         dtype=torch.float32, device=self.device).view(1, 1, 3, 3) / 8.0
        )
        sobel_y = (
            torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                         dtype=torch.float32, device=self.device).view(1, 1, 3, 3) / 8.0
        )

        dz_dy = F.conv2d(z_smoothed, sobel_x, padding=1) / self.grid_res
        dz_dx = F.conv2d(z_smoothed, sobel_y, padding=1) / -self.grid_res

        norm_x, norm_y, norm_z = -dz_dx, -dz_dy, torch.ones_like(dz_dx)
        magnitude = torch.sqrt(norm_x ** 2 + norm_y ** 2 + norm_z ** 2)
        nz = norm_z / (magnitude + 1e-8)

        slope_rad = torch.acos(torch.clamp(nz, -1.0, 1.0))
        slope_mesh_tensor = slope_rad * (180.0 / math.pi)

        # ---- 6. 粗糙度 (局部方差) ----
        mean_z = F.avg_pool2d(z_smoothed, kernel_size, stride=1, padding=padding,
                              count_include_pad=True)
        sqr_mean_z = F.avg_pool2d(z_smoothed ** 2, kernel_size, stride=1, padding=padding,
                                  count_include_pad=True)
        roughness_mesh_tensor = torch.clamp(sqr_mean_z - mean_z ** 2, min=0.0)

        # ---- 7. 转 CPU 并构建输出 ----
        slope_mesh = slope_mesh_tensor.squeeze().cpu().numpy()
        roughness_mesh = roughness_mesh_tensor.squeeze().cpu().numpy()
        z_mesh_cpu = z_smoothed.squeeze().cpu().numpy()

        point_slopes = slope_mesh[ix, iy]
        point_roughness = roughness_mesh[ix, iy]
        is_safe = (point_slopes < self.slope_th) & (point_roughness < self.rough_th)
        safety_probs = is_safe.astype(float)

        safe_mesh = (slope_mesh < self.slope_th) & (roughness_mesh < self.rough_th)

        bev_data = {
            "x_mesh": np.linspace(x_max, x_max - (rows - 1) * self.grid_res, rows),
            "y_mesh": np.linspace(y_min, y_min + (cols - 1) * self.grid_res, cols),
            "z_mesh": z_mesh_cpu,
            "slope_mesh": slope_mesh,
            "roughness_mesh": roughness_mesh,
            "safe_mesh": safe_mesh,
            "rows": rows,
            "cols": cols,
        }

        return {
            "safety_probs": safety_probs,
            "slope_mesh": slope_mesh,
            "roughness_mesh": roughness_mesh,
            "bev_data": bev_data,
        }
