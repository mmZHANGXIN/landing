"""
HALSS 贝叶斯语义分割模块
=========================
移植自 HALO-master/HALSS/ halss_utils.py → run_network() + pc_to_surf_normal()

管线:
  点云(N,3) → Delaunay插值 → surface normal RGB →
  Unet_drop + MC Dropout (多次前向) → 均值图 + 方差图 →
  不确定性阈值 → safe_mesh (bool 栅格)

权重: HALO-master/HALSS/network_utils/unet_epoch6.pth
架构: Unet_drop, nbase=[3,32,64,128,256], nout=1, kernel_size=3
MC Dropout: p=0.2, training=True 始终开启
"""

import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from scipy import interpolate


# ============================================================
# Unet_drop 模型定义 (内联，无外部依赖)
# ============================================================

def _convbatchrelu(in_ch, out_ch, sz):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, sz, padding=sz // 2),
        nn.BatchNorm2d(out_ch, eps=1e-5),
        nn.ReLU(inplace=True),
    )


class _ConvDown(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size):
        super().__init__()
        self.conv = nn.Sequential()
        for t in range(2):
            ci = in_ch if t == 0 else out_ch
            self.conv.add_module(f"conv_{t}", _convbatchrelu(ci, out_ch, kernel_size))

    def forward(self, x):
        x = self.conv(x)
        return F.dropout(x, p=0.2, training=True, inplace=True)


class _DownSample(nn.Module):
    def __init__(self, nbase, kernel_size):
        super().__init__()
        self.down = nn.Sequential()
        self.maxpool = nn.MaxPool2d(2, 2)
        for n in range(len(nbase) - 1):
            self.down.add_module(
                f"conv_down_{n}", _ConvDown(nbase[n], nbase[n + 1], kernel_size)
            )

    def forward(self, x):
        xd = []
        for n in range(len(self.down)):
            y = self.maxpool(xd[n - 1]) if n > 0 else x
            xd.append(self.down[n](y))
        return xd


class _ConvUp(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size):
        super().__init__()
        self.conv = nn.Sequential()
        self.conv.add_module("conv_0", _convbatchrelu(in_ch, out_ch, kernel_size))
        self.conv.add_module("conv_1", _convbatchrelu(out_ch, out_ch, kernel_size))

    def forward(self, x, skip):
        x = self.conv.conv_0(x)
        x = self.conv.conv_1(x + skip)
        return F.dropout(x, p=0.2, training=True, inplace=True)


class _UpSample(nn.Module):
    def __init__(self, nbase, kernel_size):
        super().__init__()
        self.upsampling = nn.Upsample(scale_factor=2, mode="nearest")
        self.up = nn.Sequential()
        for n in range(len(nbase) - 1, 0, -1):
            self.up.add_module(
                f"conv_up_{n - 1}", _ConvUp(nbase[n], nbase[n - 1], kernel_size)
            )

    def forward(self, xd):
        x = xd[-1]
        for n in range(len(self.up)):
            if n > 0:
                x = self.upsampling(x)
            x = self.up[n](x, xd[len(xd) - 1 - n])
        return x


class UnetDrop(nn.Module):
    def __init__(self, nbase, nout, kernel_size):
        super().__init__()
        self.nbase = nbase
        self.nout = nout
        self.kernel_size = kernel_size
        self.downsample = _DownSample(nbase, kernel_size)
        nbaseup = nbase[1:] + [nbase[-1]]
        self.upsample = _UpSample(nbaseup, kernel_size)
        self.output = nn.Conv2d(nbase[1], nout, kernel_size, padding=kernel_size // 2)

    def forward(self, x):
        xd = self.downsample(x)
        x = self.upsample(xd)
        return torch.sigmoid(self.output(x))

    def load_weights(self, path, device="cuda"):
        state = torch.load(path, map_location=device)
        self.load_state_dict(state)
        return self


# ============================================================
# 辅助函数
# ============================================================

def _normalize(img):
    """Min-max normalize → float32"""
    x = img.astype(np.float32)
    return (x - x.min()) / (x.max() - x.min() + 1e-8)


def surface_normal_from_interp(x_grid, y_grid, Z, grid_res):
    """Interpolated height field -> original HALSS surface-normal image.

    The Bayesian UNet weights were trained on HALSS normal images whose
    channels are flipped/reordered by point_cloud_to_image.py + cvtColor in
    the original pipeline. Keep that exact convention here:
      network input channels = [nz, nx, ny], vertically flipped.
    """
    grad = np.gradient(Z, y_grid, x_grid)
    dx, dy = grad[1], grad[0]
    normal = np.zeros((grid_res, grid_res, 3), dtype=np.float32)
    normal[:, :, 0] = -dx
    normal[:, :, 1] = -dy
    normal[:, :, 2] = 1.0
    mag = np.sqrt((normal ** 2).sum(axis=2, keepdims=True)) + 1e-8
    normal_unit = normal / mag
    normal_color = (normal_unit + 1) * 127.5

    # Original surface_normal_from_interp_model() builds [ny, nx, nz],
    # then halss_utils.py applies cv2.COLOR_BGR2RGB before inference.
    ch0 = np.nan_to_num(normal_color[:, :, 2], nan=0.0, posinf=0.0, neginf=0.0)
    ch1 = np.nan_to_num(normal_color[:, :, 0], nan=0.0, posinf=0.0, neginf=0.0)
    ch2 = np.nan_to_num(normal_color[:, :, 1], nan=0.0, posinf=0.0, neginf=0.0)
    return np.flipud(np.dstack((ch0, ch1, ch2)).clip(0, 255).astype(np.uint8))


def scale_image(img, pcd_full):
    """将图像缩放到点云XY范围 (仿原版 scale_image)"""
    if pcd_full is None or len(pcd_full) < 3:
        return cv2.resize(img, (200, 200))
    x_min, x_max = pcd_full[:, 0].min(), pcd_full[:, 0].max()
    y_min, y_max = pcd_full[:, 1].min(), pcd_full[:, 1].max()
    if x_max <= x_min or y_max <= y_min:
        return cv2.resize(img, (200, 200))
    aspect = (y_max - y_min) / (x_max - x_min + 1e-8)
    h, w = img.shape[:2]
    if aspect > 1:
        new_h, new_w = h, int(h / aspect)
    else:
        new_h, new_w = int(w * aspect), w
    new_h, new_w = max(8, new_h), max(8, new_w)
    return cv2.resize(img, (new_w, new_h))


# ============================================================
# HALSSBayesianEvaluator
# ============================================================

class HALSSBayesianEvaluator:
    """
    HALSS 贝叶斯分割评估器。
    输入: 世界坐标点云 (N,3)
    输出: safe_mesh (bool 栅格) + 均值/方差图
    """

    def __init__(self, cfg: dict):
        # grid_res 必须被 16 整除 (Unet 有 4 级 maxpool: 2^4=16)
        self.grid_res = cfg.get("halss_grid_res", cfg.get("grid_res_halss", 64))
        if self.grid_res % 16 != 0:
            self.grid_res = ((self.grid_res + 15) // 16) * 16
        self.mc_samples = cfg.get("mc_samples", 10)
        self.uncertainty_th = cfg.get("uncertainty_threshold", 0.3)
        self.require_gpu = bool(cfg.get("require_gpu", True))
        self.weight_path = cfg.get(
            "halss_weight_path",
            "arch/3.UDPDirect30Hz_cyd_final/HALO-master (2)/HALSS/network_utils/unet_epoch6.pth",
        )

        cuda_ok = torch.cuda.is_available()
        if self.require_gpu and not cuda_ok:
            raise RuntimeError(
                "[HALSS-Bayesian] CUDA NOT available. "
                "Bayesian UNet CPU fallback is denied for flight."
            )
        self.device = torch.device("cuda" if cuda_ok else "cpu")
        self.nbase = [3, 32, 64, 128, 256]
        self.kernel_size = 3

        print(f"[HALSS-Bayesian] Loading Unet_drop from {self.weight_path} ...")
        self.net = UnetDrop(self.nbase, nout=1, kernel_size=self.kernel_size)
        self.net.load_weights(self.weight_path, device=str(self.device))
        self.net.to(self.device)
        self.net.train()  # Dropout always active
        print(
            f"[HALSS-Bayesian] Ready | device={self.device} "
            f"require_gpu={self.require_gpu} mc={self.mc_samples}"
        )

    # --------------------------------------------------------
    # Step 1: 点云 → 快速插值 → surface normal RGB
    #   使用 scipy.griddata (linear) + gaussian 平滑
    #   替代 Delaunay, 快 ~10x
    #   grid_res 必须能被 16 整除 (Unet 4级maxpool)
    # --------------------------------------------------------
    def _pc_to_surf_normal(self, pts, fixed_bounds: dict = None):
        from scipy.interpolate import griddata
        from scipy.ndimage import gaussian_filter

        grid_res = self.grid_res
        if grid_res % 16 != 0:
            grid_res = ((grid_res + 15) // 16) * 16

        pts = pts.astype(np.float64)

        # Use fixed bounds when provided (aligns semantic grid with depth grid)
        if fixed_bounds is not None:
            x_min = float(fixed_bounds.get("x_min", -5.0))
            x_max = float(fixed_bounds.get("x_max", 5.0))
            y_min = float(fixed_bounds.get("y_min", -5.0))
            y_max = float(fixed_bounds.get("y_max", 5.0))
        else:
            x_min, x_max = pts[:, 0].min(), pts[:, 0].max()
            y_min, y_max = pts[:, 1].min(), pts[:, 1].max()

        # Guard against degenerate bounds
        if x_max - x_min < 1e-6:
            x_min -= 0.5
            x_max += 0.5
        if y_max - y_min < 1e-6:
            y_min -= 0.5
            y_max += 0.5

        x_grid = np.linspace(x_min, x_max, grid_res)
        y_grid = np.linspace(y_min, y_max, grid_res)
        X, Y = np.meshgrid(x_grid, y_grid)

        # Use the original HALSS interpolation semantics for the UNet input.
        # Nearest-neighbor extrapolation across empty scan regions creates fake
        # surfaces, which is especially damaging for sparse FAST-LIO frames.
        Z = griddata(pts[:, :2], pts[:, 2], (X, Y), method="linear")
        if not np.isfinite(Z).any():
            Z = griddata(pts[:, :2], pts[:, 2], (X, Y), method="nearest")

        return surface_normal_from_interp(x_grid, y_grid, Z, grid_res)

    # --------------------------------------------------------
    # Step 2: surface normal → Unet_drop + MC Dropout
    # --------------------------------------------------------
    def _run_mc_inference(self, surf_norm_rgb):
        # 归一化 + 转 CHW
        img = _normalize(surf_norm_rgb.transpose(2, 0, 1))
        img_t = (
            torch.from_numpy(img).float().to(self.device).unsqueeze(0)
        )  # (1,3,H,W)

        # MC Dropout: 复制 mc_samples 份，批量前向
        batch = img_t.repeat(self.mc_samples, 1, 1, 1)  # (MC,3,H,W)
        with torch.no_grad():
            out = self.net(batch)  # (MC,1,H,W)

        mean = out.mean(dim=0)  # (1,H,W)
        var = out.var(dim=0)  # (1,H,W)

        mean_np = mean.squeeze().cpu().numpy()
        var_np = var.squeeze().cpu().numpy()
        return mean_np, var_np

    # --------------------------------------------------------
    # 主入口
    # --------------------------------------------------------
    def evaluate(self, points_world: np.ndarray, pcd_full=None,
                 fixed_bounds: dict = None, profile: dict = None) -> dict:
        """
        输入: 世界坐标点云 (N,3)
        可选: fixed_bounds = {"x_min","x_max","y_min","y_max"}
              若提供, 语义网格使用固定世界坐标范围 (与深度图对齐)
        输出: 与 halss_gpu.py 兼容的 dict
        """
        if points_world is None or len(points_world) < 10:
            return None

        total_start = time.perf_counter() if profile is not None else None
        pts = points_world.astype(np.float64)

        # 1. surface normal
        projection_start = time.perf_counter() if profile is not None else None
        try:
            surf_norm = self._pc_to_surf_normal(pts, fixed_bounds=fixed_bounds)  # (H,W,3) uint8
        finally:
            if profile is not None:
                profile["halss_surface_projection_ms"] = (
                    time.perf_counter() - projection_start
                ) * 1000.0

        # 2. MC Dropout 推理
        network_start = time.perf_counter() if profile is not None else None
        try:
            mean_map, var_map = self._run_mc_inference(surf_norm)  # (H,W) float
        finally:
            if profile is not None:
                profile["halss_network_ms"] = (
                    time.perf_counter() - network_start
                ) * 1000.0

        # 3. 归一化方差 + 缩放到点云范围
        var_norm = _normalize(var_map)
        if pcd_full is not None:
            var_scaled = scale_image(var_norm, pcd_full)
        else:
            var_scaled = cv2.resize(var_norm, (mean_map.shape[1], mean_map.shape[0]))

        # 4. 构建 safety_map (二值, 0/255)
        predicted = (mean_map * 255).astype(np.uint8)
        safety_map = np.full_like(predicted, 0, dtype=np.uint8)
        safety_map[predicted >= 128] = 255

        # 5. 方差阈值: 高不确定区域 → 危险
        if pcd_full is not None and len(pcd_full) > 3:
            var_for_thresh = scale_image(var_map, pcd_full)
        else:
            var_for_thresh = var_map
        var_norm_full = _normalize(var_for_thresh)

        high_uncertainty = var_norm_full > self.uncertainty_th
        safety_map[high_uncertainty] = 0

        # 6. binary fill holes
        safety_map = safety_map.astype(np.uint8)
        safe_mesh = safety_map > 0

        result = {
            "safety_probs": (mean_map.clip(0, 1) * (255 - var_norm_full * 255) / 255.0).ravel(),
            "safe_mesh": safe_mesh,
            "bev_data": {
                "safe_mesh": safe_mesh,
                "mean_map": mean_map,
                "variance_map": var_map,
                "safety_map_vis": safety_map,
            },
            "mean_map": mean_map,
            "variance_map": var_map,
            "surf_norm_rgb": surf_norm,
            "safety_map_vis": safety_map,
        }
        if profile is not None:
            profile["halss_total_ms"] = (
                time.perf_counter() - total_start
            ) * 1000.0
            profile["halss_mc_samples"] = int(self.mc_samples)
        return result

    # --------------------------------------------------------
    # 可视化 (兼容 pipeline)
    # --------------------------------------------------------
    def get_bev_result(self, points_world):
        """返回兼容 SemanticGenerator 的 bev_result"""
        result = self.evaluate(points_world)
        if result is None:
            return None
        return {"safe_mesh": result["safe_mesh"]}
