#!/usr/bin/env python3
"""window10 物理尺度平面拟合语义 (unknown 三态) + 三窗口坐标统一 + 近地深度显示 — 规格测试.

规格 (2026-08-21) 合成测试清单:
  1. 平坦地面在点云有洞和轻微帧间高度偏差时, 大部分有效区域仍判安全
     (洞单元保持灰色 unknown, 不形成黑色危险斑点);
  2. 稀疏且支持不足 (窗口观测单元 < --plane-min-support 6) 区域显示灰色,
     不形成黑色危险斑点; sem_map 仍保守编码 danger_id 送 PPO;
  3. 0.15 m 以上柱体形成连续危险区域 (单元内多点或相邻格连续支持),
     孤立离群点 (1 格 1 点) 不形成障碍;
  4. +x 标记同时出现在三个窗口上方, +y 标记同时出现在左侧
     (draw_frame_markers, 与 CompareVisualizer.update 逐帧相同调用链);
  5. 64×64 BEV 与 128×128 语义/深度经逆时针 90° + 最近邻缩放 + 共享裁剪后,
     同一物理特征落在完全相同的输出像素;
  6. 自适应深度图 (local) 能明显区分近地平面和柱体 (2%~98% 分位, 最小跨度
     0.5 m), 凸包外保持黑 (NN 填充不扩散成均匀灰), 且显示函数不改变 ONNX
     输入数组 dense_depth / valid_mask 的任何值; fixed 模式保持 0~30 m.

本文件只依赖 numpy + cv2 + 标准库, 不导入 torch / 不加载 bag.
运行:
  python3 test/test_window10_planes_display_light.py
"""

import sys
import unittest
from pathlib import Path

import numpy as np

_SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(_SCRIPTS))

import cv2  # noqa: E402

from replay_compare_common import (  # noqa: E402
    bev_roughness_downsample,
    build_three_window_displays,
    draw_frame_markers,
    make_binary_semantic_vis,
    render_depth_fixed_gray,
    render_depth_local_gray,
    roi_bounds,
)
from replay_window10 import (  # noqa: E402
    PlaneGeometrySemanticBranch,
    _physical_window_cells,
    bev_plane_semantic_map,
)

_SLOPE_TH = 10.0
_ROUGH_TH = 0.15
_SAFE_ID, _DANGER_ID = 1, 9
_G = 64
_BOUNDS = roi_bounds(2.5, 2.5)      # ±2.5 m ROI, cell ≈ 0.0794 m


# ──────────────────────────────────────────────
# 合成场景助手
# ──────────────────────────────────────────────
def _cell_of(x, y, bounds, grid=_G):
    """与 bev_roughness_downsample 相同的 单元索引映射 (行 0 = +y)."""
    x_span = bounds["x_max"] - bounds["x_min"]
    y_span = bounds["y_max"] - bounds["y_min"]
    col = int(np.rint((x - bounds["x_min"]) / x_span * (grid - 1)))
    row = (grid - 1) - int(np.rint((y - bounds["y_min"]) / y_span
                                   * (grid - 1)))
    return row, col


def _punch_cells(pts, bounds, cells, grid=_G):
    """移除落入给定单元 (r, c) 列表的所有点."""
    G = grid
    x_span = bounds["x_max"] - bounds["x_min"]
    y_span = bounds["y_max"] - bounds["y_min"]
    col = np.rint((pts[:, 0] - bounds["x_min"]) / x_span * (G - 1)).astype(np.int32)
    row = (G - 1) - np.rint(
        (pts[:, 1] - bounds["y_min"]) / y_span * (G - 1)).astype(np.int32)
    flat = row * G + col
    hole_flat = {r * G + c for r, c in cells}
    return pts[~np.isin(flat, list(hole_flat))]


def _dense_ground(seed=101, z_hi=None, frac_hi=0.4, noise=0.02, n=15000,
                  half=2.0):
    """稠密平面 (z=2.0, z-down) + 可选整体抬高的第二帧混合 (帧间高度偏差)."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(-half, half, n)
    y = rng.uniform(-half, half, n)
    z = np.full(n, 2.0, dtype=np.float64) + rng.normal(0.0, noise, n)
    if z_hi is not None:
        mask = rng.random(n) < frac_hi
        z[mask] += z_hi
    return np.column_stack([x, y, z]).astype(np.float32)


def _add_pole(pts, cx=0.6, cy=-0.4, half=0.14, h_bot=0.18, h_top=0.40,
              step=0.07, z_ground=2.0):
    """垂直柱体: 0.28 m 见方足迹 (≈3.5 单元), 高度 0.18~0.40 m 多点列."""
    xs = np.arange(cx - half, cx + half + 1e-9, step)
    ys = np.arange(cy - half, cy + half + 1e-9, step)
    hs = np.arange(h_bot, h_top + 1e-9, 0.1)
    px, py, ph = np.meshgrid(xs, ys, hs, indexing="ij")
    pole = np.column_stack([px.ravel(), py.ravel(),
                            np.full(px.size, z_ground) - ph.ravel()])
    return np.concatenate([pts, pole.astype(np.float32)]), pole.astype(np.float32)


def _sem_vis(sem_map, sem_valid):
    """显示层三态渲染 (与 CompareVisualizer.update 相同): 白/黑/灰."""
    vis = make_binary_semantic_vis(sem_map)
    vis = vis.copy()
    vis[~np.asarray(sem_valid, dtype=bool)] = 128
    return vis


# ──────────────────────────────────────────────
# 1. 平坦地面 + 洞 + 轻微帧间高度偏差 → 大部分有效区域仍安全
# ──────────────────────────────────────────────
class TestFlatGroundSafe(unittest.TestCase):

    def _scene(self):
        pts = _dense_ground(seed=101, z_hi=0.045, frac_hi=0.4)
        holes = []
        for r0, c0 in ((14, 40), (38, 22)):
            for dr in range(3):
                for dc in range(3):
                    holes.append((r0 + dr, c0 + dc))
        holes.append((50, 50))
        pts = _punch_cells(pts, _BOUNDS, holes)
        return pts, holes

    def test_mostly_safe_with_holes_and_height_deviation(self):
        """洞 + 帧间 0.045 m 偏差: 有效区域 ≥90% 安全, 危险 0,
        洞单元 unknown (灰), 不形成黑色危险斑点."""
        pts, holes = self._scene()
        bev = bev_roughness_downsample(pts, _BOUNDS, grid_res=_G)
        occ = bev.occupied
        self.assertGreaterEqual(int(occ.sum()), 1000)     # 覆盖充分
        sem, sem_valid, maps = bev_plane_semantic_map(
            bev, _SLOPE_TH, _ROUGH_TH)
        safe_occ = (sem == _SAFE_ID) & occ
        self.assertGreaterEqual(float(safe_occ.sum()) / float(occ.sum()), 0.9)
        danger_valid = (sem == _DANGER_ID) & sem_valid
        self.assertEqual(int(danger_valid.sum()), 0)
        # 洞单元: 未观测 → 灰 (unknown), 不判安全也不判危险
        hole_mask = np.zeros((_G, _G), dtype=bool)
        for r, c in holes:
            hole_mask[r, c] = True
        self.assertFalse(bool(np.any(hole_mask & occ)))      # 洞内无观测点
        self.assertFalse(bool(np.any(sem_valid[hole_mask])))
        vis = _sem_vis(sem, sem_valid)
        self.assertTrue(bool(np.all(vis[hole_mask] == 128)))
        # 三比值日志口径: danger_valid=0, unknown>0 (洞), safe 集中在有效区
        sm, info = PlaneGeometrySemanticBranch(_SLOPE_TH, _ROUGH_TH)(bev, _BOUNDS)
        self.assertEqual(info["danger_valid_ratio"], 0.0)
        self.assertGreater(info["unknown_ratio"], 0.0)
        self.assertGreater(info["safe_ratio"] / float(occ.mean()) * 0.5, 0.4)


# ──────────────────────────────────────────────
# 2. 稀疏且支持不足 → 灰色, 不形成黑色危险斑点
# ──────────────────────────────────────────────
class TestSparseUnsupportedGray(unittest.TestCase):

    def test_sparse_patch_all_gray_no_danger_blob(self):
        """2×2 单元小块 (4 单元 < min_support 6): 全部 unknown,
        sem_map 保守编码 danger_id, 显示全灰, 无黑色像素."""
        rng = np.random.default_rng(202)
        xs, ys = np.meshgrid([-0.04, 0.04], [-0.04, 0.04])
        pts = np.column_stack([xs.ravel(), ys.ravel(), np.full(4, 2.0)])
        pts = np.repeat(pts, 2, axis=0)                  # 每格 2 点
        pts = pts + rng.normal(0.0, 0.005, pts.shape).astype(np.float32)
        bev = bev_roughness_downsample(pts, _BOUNDS, grid_res=_G)
        self.assertGreaterEqual(int(bev.occupied.sum()), 4)
        sem, sem_valid, _ = bev_plane_semantic_map(
            bev, _SLOPE_TH, _ROUGH_TH)
        # 支持不足 → 无任何几何证据单元; 保守编码仍为 danger_id
        self.assertEqual(int(sem_valid.sum()), 0)
        self.assertTrue(bool(np.all(sem == _DANGER_ID)))
        # 显示层: 全灰 (128), 无黑色危险斑点 (0)
        vis = _sem_vis(sem, sem_valid)
        self.assertTrue(bool(np.all(vis == 128)))
        self.assertEqual(int((vis == 0).sum()), 0)


# ──────────────────────────────────────────────
# 3. 0.15 m+ 柱体 → 连续危险区域; 孤立离群点 → 不成障碍
# ──────────────────────────────────────────────
class TestPoleDangerAndIsolatedOutlier(unittest.TestCase):

    def _scene(self):
        pts = _dense_ground(seed=303, z_hi=None)
        pts, pole = _add_pole(pts)
        # 孤立离群点: 打 1 单元洞后放入单点 (1 格 1 点)
        o_cell = _cell_of(-1.4, 1.5, _BOUNDS)
        pts = _punch_cells(pts, _BOUNDS, [o_cell])
        outlier = np.array([[-1.4, 1.5, 2.0 - 0.35]], dtype=np.float32)
        pts = np.concatenate([pts, outlier])
        return pts, pole, o_cell

    def test_pole_continuous_danger_outlier_not_obstacle(self):
        pts, pole, o_cell = self._scene()
        bev = bev_roughness_downsample(pts, _BOUNDS, grid_res=_G)
        sem, sem_valid, _ = bev_plane_semantic_map(
            bev, _SLOPE_TH, _ROUGH_TH)
        danger = (sem == _DANGER_ID) & sem_valid

        # 柱体单元全部危险且有几何证据
        pole_cells = set()
        for px, py in pole[:, :2]:
            pole_cells.add(_cell_of(float(px), float(py), _BOUNDS))
        self.assertGreaterEqual(len(pole_cells), 4)
        for rc in pole_cells:
            self.assertTrue(bool(danger[rc]), f"柱体单元 {rc} 未判危险")

        # 连续危险区域: 最大连通分量 ≥ 柱体足迹 (单元级连续支持)
        n_comp, labels = cv2.connectedComponents(danger.astype(np.uint8))
        sizes = np.bincount(labels.ravel())
        self.assertGreaterEqual(int(sizes[1:].max()), len(pole_cells))
        # 危险单元有界 (只覆盖柱体区域, 地面不被波及)
        self.assertLessEqual(int(danger.sum()), 3 * len(pole_cells))

        # 孤立离群点: 1 格 1 点 → n_high=1 且无相邻突出格 → 不成障碍
        self.assertFalse(bool(danger[o_cell]))
        self.assertEqual(int(sem[o_cell]), _SAFE_ID)


# ──────────────────────────────────────────────
# 4. 三窗口方向标记: +x 顶部 / +y 左侧
# ──────────────────────────────────────────────
class TestFrameMarkersThreeWindows(unittest.TestCase):

    def test_markers_top_and_left_in_all_windows(self):
        """每个窗口经 draw_frame_markers 后, 顶部带 +x 白色标记,
        左侧带 +y 白色标记 (CompareVisualizer.update 的逐帧调用链)."""
        bev_bgr = np.zeros((_G, _G, 3), np.uint8)
        sem_vis = np.full((128, 128), 128, np.uint8)
        depth_bgr = np.zeros((128, 128, 3), np.uint8)
        bev_d, sem_d, dep_d = build_three_window_displays(
            bev_bgr, sem_vis, depth_bgr)
        self.assertEqual(bev_d.shape, sem_d.shape)
        self.assertEqual(sem_d.shape, dep_d.shape)
        for img in (bev_d, sem_d, dep_d):
            marked = draw_frame_markers(img, text="w",
                                        top_label="+x", left_label="+y")
            h, w = marked.shape[:2]
            top_band = marked[6:26, w // 2 - 14:w // 2 + 14]
            left_band = marked[h // 2 - 10:h // 2 + 20, 2:22]
            white_top = np.all(top_band == 255, axis=-1)
            white_left = np.all(left_band == 255, axis=-1)
            self.assertTrue(bool(white_top.any()), "顶部缺少 +x 标记")
            self.assertTrue(bool(white_left.any()), "左侧缺少 +y 标记")


# ──────────────────────────────────────────────
# 5. 64×64 BEV 与 128×128 语义/深度共享裁剪后像素对齐
# ──────────────────────────────────────────────
class TestSharedCropPixelAlignment(unittest.TestCase):

    def test_feature_lands_on_same_pixel_after_rotate_resize_crop(self):
        """同一物理特征: 64 BEV (逆时针 90° + NN 缩放) 与 128 语义/深度
        共享裁剪后落在完全相同的输出像素 — 修复旧版 128 裁剪坐标直切
        64 BEV 的错位."""
        rs, cs = 48, 80                       # 128 网格特征位置
        sem_vis = np.full((128, 128), 128, np.uint8)
        sem_vis[rs, cs] = 255
        # BEV 64 特征: 逆旋转 (自洽地用 cv2 顺时针求逆, 不硬编码公式)
        probe = np.zeros((_G, _G), np.uint8)
        probe[rs // 2, cs // 2] = 255
        pre_r, pre_c = np.nonzero(cv2.rotate(probe, cv2.ROTATE_90_CLOCKWISE))
        bev_bgr = np.zeros((_G, _G, 3), np.uint8)
        bev_bgr[pre_r[0], pre_c[0]] = (255, 255, 255)
        depth_bgr = np.zeros((128, 128, 3), np.uint8)
        depth_bgr[rs, cs] = (255, 255, 255)

        bev_d, sem_d, dep_d = build_three_window_displays(
            bev_bgr, sem_vis, depth_bgr)
        self.assertEqual(bev_d.shape, sem_d.shape)
        self.assertEqual(sem_d.shape, dep_d.shape)
        f_bev = np.unravel_index(np.argmax(bev_d.sum(axis=2)), bev_d.shape[:2])
        f_sem = np.unravel_index(np.argmax(sem_d.sum(axis=2)), sem_d.shape[:2])
        f_dep = np.unravel_index(np.argmax(dep_d.sum(axis=2)), dep_d.shape[:2])
        self.assertEqual(f_bev, f_sem)
        self.assertEqual(f_sem, f_dep)

    def test_rotation_sends_plus_x_up_plus_y_left(self):
        """旋转方向: BEV 行 0 = +y max (机体左) → 显示左侧;
        列 G-1 = +x max (机头前) → 显示顶部 (机头朝上、机体左在左)."""
        bev_bgr = np.zeros((_G, _G, 3), np.uint8)
        bev_bgr[0, _G // 2] = (255, 255, 255)        # 顶行中列: +y max
        bev_bgr[_G // 2, _G - 1] = (255, 255, 255)   # 右列中行: +x max
        sem_vis = np.full((128, 128), 128, np.uint8)   # 全灰 → 不裁剪
        bev_d, _, _ = build_three_window_displays(bev_bgr, sem_vis, None)
        # 逆时针 90°: old row r → new col r; old col c → new row G-1-c
        # NN 缩放 64→128: 64 网格 (r, c) → 128 网格块 (2r..2r+1, 2c..2c+1)
        # old(0, 32) → new(31, 0) → 128 块 (62..63, 0..1) = 左列
        # old(32, 63) → new(0, 32) → 128 块 (0..1, 64..65) = 顶行
        left_block = bev_d[62:64, 0:2]
        top_block = bev_d[0:2, 64:66]
        self.assertTrue(bool(np.all(left_block == 255)), "机体左 (+y) 未到左侧")
        self.assertTrue(bool(np.all(top_block == 255)), "机头 (+x) 未到顶部")


# ──────────────────────────────────────────────
# 6. 近地深度局部自适应显示 (不改 ONNX 输入)
# ──────────────────────────────────────────────
class TestLocalDepthDisplay(unittest.TestCase):

    def _scene(self):
        h = w = 128
        yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
        disk = ((xx - 64) ** 2 + (yy - 64) ** 2) <= 40 ** 2
        dense = np.full((h, w), 5.0, np.float32)
        dense[52:72, 48:68] = 4.2              # 柱体 (400 px ≈ 8% > 2% 分位)
        valid = disk.copy()
        return dense, valid, disk

    def test_local_gray_distinguishes_plane_and_pole(self):
        """local 模式: 分位量程 ~4.2~5.0 m → 柱体近黑、地面近白;
        凸包外 (NN 填充区) 保持黑; ONNX 输入数组完全未改变."""
        dense, valid, disk = self._scene()
        dense_before, valid_before = dense.copy(), valid.copy()
        out, (near_m, far_m) = render_depth_local_gray(dense, valid, dmax=30.0)
        # 输入数组不被显示层修改 (ONNX 仍接收原 dense_depth)
        np.testing.assert_array_equal(dense, dense_before)
        np.testing.assert_array_equal(valid, valid_before)
        self.assertAlmostEqual(near_m, 4.2, places=2)
        self.assertAlmostEqual(far_m, 5.0, places=2)
        pole_px = out[dense == 4.2]
        ground_px = out[(dense == 5.0) & disk]
        self.assertLess(float(pole_px.mean()), 40.0)      # 近黑
        self.assertGreater(float(ground_px.mean()), 200.0)  # 近白
        # 凸包外保持黑: NN 填充不把画面洗成均匀灰。fillConvexPoly 整数栅格化
        # 在弯曲凸包边缘有 ~1 px 别名环 (≤0.3 px 越界), 允许; 距边界 ≥2 px
        # 的区域必须严格为黑
        outside = ~disk
        rim = np.zeros(out.shape[:2], dtype=bool)
        yy, xx = np.meshgrid(np.arange(out.shape[0]), np.arange(out.shape[1]),
                             indexing="ij")
        rim[((xx - 64) ** 2 + (yy - 64) ** 2 <= 41.5 ** 2)] = True
        deep_out = outside & ~rim
        self.assertTrue(bool(np.all(out[deep_out] == 0)))
        self.assertLessEqual(int(((out[..., 0] > 0) & outside).sum()), 300)

    def test_fixed_mode_and_fallback(self):
        """fixed 模式保持 0~30 m 显示; 有效点不足时 local 回退 0~dmax."""
        dense, valid, _ = self._scene()
        fixed = render_depth_fixed_gray(dense, vmax_m=30.0)
        self.assertEqual(int(np.round(fixed[dense == 5.0][0, 0])),
                         int(np.round(255.0 * 5.0 / 30.0)))   # ≈ 42
        # 有效点 < min_valid → 回退固定 0~30
        tiny = np.zeros((64, 64), np.float32)
        tiny[32, 32] = 1.0
        v = np.zeros((64, 64), dtype=bool)
        v[32, 32] = True
        _, (n2, f2) = render_depth_local_gray(tiny, v, dmax=30.0)
        self.assertEqual((n2, f2), (0.0, 30.0))


# ──────────────────────────────────────────────
# 7. 物理尺度窗口: 半径按 cell 尺寸换算, 高度变化不改几何尺度
# ──────────────────────────────────────────────
class TestPhysicalWindowCells(unittest.TestCase):

    def test_radius_pixels_scales_with_cell_size(self):
        """米制半径恒定: cell 减半 → 像素半径加倍, 米制半径一致."""
        rx1, ry1, s1 = _physical_window_cells(0.5, 0.1, 0.1, 6, 1)
        rx2, ry2, s2 = _physical_window_cells(0.5, 0.05, 0.05, 6, 1)
        self.assertEqual((rx1, ry1), (5, 5))
        self.assertEqual((rx2, ry2), (10, 10))
        self.assertAlmostEqual(rx1 * 0.1, rx2 * 0.05, places=4)
        # 非对称 cell (x/y 分辨率不同) 各自换算 (round(2.5)=2 银行家舍入)
        rx3, ry3, _ = _physical_window_cells(0.5, 0.1, 0.2, 6, 1)
        self.assertEqual((rx3, ry3), (5, 2))

    def test_stride_cap_bounds_sampled_window(self):
        """低空 (极小 cell) 时步长抽稀: 采样窗口封顶 ≤ (2·target+1)²."""
        n_fine = len(np.arange(2 * 25 + 1)[::_physical_window_cells(
            0.5, 0.02, 0.02, 6, 1)[2]])
        self.assertLessEqual(n_fine * n_fine, (2 * 6 + 1) ** 2)
        _, _, s_coarse = _physical_window_cells(2.0, 0.02, 0.02, 10, 2)
        n_coarse = len(np.arange(2 * 100 + 1)[::s_coarse])
        self.assertGreaterEqual(s_coarse, 2)
        self.assertLessEqual(n_coarse * n_coarse, (2 * 10 + 1) ** 2)


# ──────────────────────────────────────────────
# 11. OOM 回归: 起飞低空微小 ROI (毫米级单元) 不爆内存
#     (Orin 2026-08-21 实测 Killed: 2.0 m 物理半径 → 上千格窗口数组)
# ──────────────────────────────────────────────
_TINY_BOUNDS = roi_bounds(0.1, 0.2)   # ±0.1 × ±0.2 m, cell ≈ 3.2~6.3 mm


class TestTinyRoiOomGuard(unittest.TestCase):
    """cell 毫米级 + 物理半径 2.0 m: 格半径钳制到 (G-1)//2, 中间数组有界."""

    def test_radius_cell_clamp(self):
        """未钳制时 2.0 m / 毫米单元 → 上千格 (原 OOM 参数); 钳制后 = 全网格."""
        rx_u, ry_u, _ = _physical_window_cells(2.0, 0.0016, 0.0032, 10, 2)
        self.assertGreater(rx_u, 1000)
        rx, ry, stride = _physical_window_cells(
            2.0, 0.0016, 0.0032, 10, 2, max_radius=(_G - 1) // 2)
        self.assertEqual((rx, ry), (31, 31))
        self.assertEqual(stride, 3)          # max(2, (31+5)//10)

    def test_dense_tiny_roi_no_oom(self):
        """毫米单元稠密地面 (Orin 复现场景): 不 OOM; 窗口=全网格, 支持充分,
        平坦地面 → 有效区全部安全."""
        rng = np.random.default_rng(7)
        x = rng.uniform(-0.1, 0.1, 6000)
        y = rng.uniform(-0.2, 0.2, 6000)
        z = np.full(6000, 2.0) + rng.normal(0.0, 0.005, 6000)
        pts = np.column_stack([x, y, z]).astype(np.float32)
        bev = bev_roughness_downsample(pts, _TINY_BOUNDS, grid_res=_G)
        self.assertGreaterEqual(int(bev.occupied.sum()), 2000)
        self.assertLess(bev.cell_size_m[0], 0.01)   # 复现毫米级单元
        sem, sem_valid, maps = bev_plane_semantic_map(
            bev, _SLOPE_TH, _ROUGH_TH)
        # 全部观测单元都有几何证据 (支持=钳制后的全网格窗口, 稠密充分)
        self.assertTrue(bool(np.all(sem_valid == np.asarray(bev.occupied,
                                                            dtype=bool))))
        self.assertEqual(int((sem[sem_valid] == _SAFE_ID).sum()),
                         int(sem_valid.sum()))

    def test_sparse_tiny_roi_unknown(self):
        """毫米单元 + 全局单元 < min_support → 直接全 unknown, 不构造窗口."""
        rng = np.random.default_rng(8)
        x = rng.uniform(-0.001, 0.001, 8)    # 全部落入同 1~2 个毫米单元
        y = rng.uniform(-0.001, 0.001, 8)
        z = np.full(8, 2.0)
        pts = np.column_stack([x, y, z]).astype(np.float32)
        bev = bev_roughness_downsample(pts, _TINY_BOUNDS, grid_res=_G)
        self.assertLess(int(bev.occupied.sum()), 6)
        sem, sem_valid, maps = bev_plane_semantic_map(
            bev, _SLOPE_TH, _ROUGH_TH)
        self.assertEqual(int(sem_valid.sum()), 0)
        self.assertTrue(bool(np.all(sem == _DANGER_ID)))   # 保守编码, 显示灰
        self.assertIn("semantic_valid", maps)


if __name__ == "__main__":
    unittest.main(verbosity=2)
