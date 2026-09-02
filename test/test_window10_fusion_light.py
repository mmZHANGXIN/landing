#!/usr/bin/env python3
"""十帧观测并集融合 + 尺度明确几何安全判定 — 规格测试计划 1–6.

规格: 任何一帧真实观测点都参与融合重建 (不再要求 occ >= min_occ);
BEV 显示相对局部地面高度; 每个候选格局部平面拟合 z=ax+by+c 判坡度/粗糙度;
突出物 (柱体/台阶) 需要空间连续性支持才标危险; 十帧均未观测的区域保持
unknown, 凸包最近邻填充不得将其升级为安全区.

本文件只依赖 numpy + (函数内部惰性) cv2, 不导入 perception (无 torch
环境下可独立运行); 涉及 perception 的保守投影用例以 skipUnless 保护.

覆盖 (对应规格「测试计划」):
  1. 10 帧互补观测掩码 → 并集覆盖全网格, fused >= 单帧覆盖;
  2. 0°/5°/15° 斜坡 → 前两者近全安全, 15° 近全危险且坡度图 ≈ 15°;
  3. 厘米级草叶噪声 → 不成片危险, 粗糙度远低于阈值;
  4. 0.3 m 柱体 → 柱体区危险、草坪区安全、突出高度图可区分;
  5. 十帧均未观测扇区 → 保持 unknown, 不被填充升级为安全;
  6. 填洞方向一致 (紧凑索引 vs 分量 label 等价) 且无 NaN 扩散.
"""

import math
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT), str(ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from replay_compare_common import (  # noqa: E402
    BevGrid,
    FusedGeometrySemanticBranch,
    _batch_robust_plane_fit,
    fused_geometric_semantic_map,
    geometric_semantic_map,
)
from replay_window10 import (WindowEntry, _voxel_dedup, fuse_window,
                             resolve_ground_z)  # noqa: E402
from utils.valid_nearest import fill_valid_nearest  # noqa: E402

G = 128
BOUNDS = {"x_min": -2.0, "x_max": 2.0, "y_min": -2.0, "y_max": 2.0}
X_SPAN = float(BOUNDS["x_max"] - BOUNDS["x_min"])
Y_SPAN = float(BOUNDS["y_max"] - BOUNDS["y_min"])
CELL = X_SPAN / max(G - 1, 1)          # 约 0.0315 m
ZERO_POSE = np.zeros(6, dtype=np.float64)


# ──────────────────────────────────────────────
# 合成数据辅助
# ──────────────────────────────────────────────
def _cell_centers_xy():
    c = np.arange(G, dtype=np.float64)
    xs = BOUNDS["x_min"] + c * CELL
    ys = BOUNDS["y_min"] + c * CELL
    return xs, ys


def _sample_grid(z_fn, rng=None, noise_m=0.0, exclude=None):
    """全网格每格 1 点 (格中心 + 单元内 ±0.3 格抖动), z = z_fn(x, y) [+ 噪声].

    exclude: 形如 (x_lo, x_hi, y_lo, y_hi) 的裁剪框, 用于制造永未观测扇区.
    """
    xs, ys = _cell_centers_xy()
    xg, yg = np.meshgrid(xs, ys, indexing="ij")
    if rng is not None:
        xg = xg + rng.uniform(-0.3, 0.3, size=xg.shape) * CELL
        yg = yg + rng.uniform(-0.3, 0.3, size=yg.shape) * CELL
    z = np.asarray(z_fn(xg, yg), dtype=np.float32)
    if noise_m > 0 and rng is not None:
        z = z + rng.normal(0.0, noise_m, size=z.shape)
    pts = np.column_stack([xg.ravel(), yg.ravel(), z.ravel()]).astype(np.float32)
    if exclude is not None:
        x_lo, x_hi, y_lo, y_hi = exclude
        keep = ~((pts[:, 0] >= x_lo) & (pts[:, 0] <= x_hi)
                 & (pts[:, 1] >= y_lo) & (pts[:, 1] <= y_hi))
        pts = pts[keep]
    return pts


def _plane(z0, slope_deg):
    a = math.tan(math.radians(float(slope_deg)))
    return lambda x, y: z0 + a * x          # 坡度沿 +x (z-down 深度随 x 增加)


def _entries(frames_points, dt=0.1):
    pose = ZERO_POSE.copy()
    return [WindowEntry(cloud_stamp=1000.0 + k * dt, pose=pose,
                        body_points=p, world_points=p)
            for k, p in enumerate(frames_points)]


def _fuse(frames_points, voxel_m=0.02):
    return fuse_window(_entries(frames_points), ZERO_POSE, BOUNDS, G,
                       voxel_m=voxel_m)


def _apply(frames_points, **branch_kw):
    """fuse_window + FusedGeometrySemanticBranch → (fused, sem_map, maps)."""
    fused = _fuse(frames_points)
    branch_kw.setdefault("slope_threshold_deg", 10.0)
    branch_kw.setdefault("roughness_threshold_m", 0.15)
    branch_kw.setdefault("prominence_threshold_m", 0.15)
    branch_kw.setdefault("min_support_pts", 8)
    branch = FusedGeometrySemanticBranch(**branch_kw)
    sem_map, sem_info = branch(fused)
    return fused, sem_map, sem_info["maps"]


def _safe_frac(sem_map, valid):
    return float(np.mean(sem_map[valid] == 1))


def _cell_index_xy(x, y):
    col = int(np.rint((x - BOUNDS["x_min"]) / X_SPAN * (G - 1)))
    row_un = int(np.rint((y - BOUNDS["y_min"]) / Y_SPAN * (G - 1)))
    return (G - 1) - row_un, col          # (row, col), 行 0 = +y


# ──────────────────────────────────────────────
# 测试 1: 10 帧互补观测掩码 → 并集覆盖
# ──────────────────────────────────────────────
class UnionCoverageTest(unittest.TestCase):
    def test_complementary_strips_union_covers_grid(self):
        xs, ys = _cell_centers_xy()
        frames = []
        for k in range(10):
            c0, c1 = k * G // 10, max(k * G // 10, (k + 1) * G // 10 - 1)
            xg, yg = np.meshgrid(xs[c0:c1 + 1], ys)
            z = np.full_like(xg, 3.0)
            frames.append(np.column_stack(
                [xg.ravel(), yg.ravel(), z.ravel()]).astype(np.float32))

        fused = _fuse(frames)
        st = fused.stats
        self.assertEqual(st["frames_used"], 10)
        self.assertEqual(st["observed_cells"], G * G)
        # 任何一帧真实观测都参与 → 并集覆盖 >= 当前单帧覆盖
        self.assertGreaterEqual(st["coverage_fused"], st["coverage_single"])
        self.assertLess(st["coverage_single"], 0.2 * G * G)   # 单帧只约 1/10
        self.assertAlmostEqual(st["frame_span_s"], 0.9, places=6)
        self.assertTrue(fused.valid.all())
        self.assertTrue((fused.frame_count_map == 1).all())
        self.assertTrue((fused.count == 1).all())
        self.assertEqual(st["fused_point_count"], G * G)      # 互补落点不受去重影响

    def test_current_frame_always_in_fusion(self):
        # 当前帧必在融合结果中: 满窗后融合覆盖 >= 当前单帧覆盖
        # (边界抖动使单帧覆盖略低于全网格, 并集仍只增不减)
        rng = np.random.default_rng(0)
        frames = [_sample_grid(_plane(3.0, 0.0), rng=rng) for _ in range(10)]
        fused = _fuse(frames)
        st = fused.stats
        self.assertGreaterEqual(st["coverage_fused"], st["coverage_single"])
        # 边界抖动可能使个别边界格 10 帧都未命中 (P≈0.5^10), 允许少量缺口
        self.assertGreaterEqual(st["coverage_fused"], G * G - 8)


# ──────────────────────────────────────────────
# 测试 2: 0°/5°/15° 斜坡
# ──────────────────────────────────────────────
class SlopeJudgementTest(unittest.TestCase):
    def _case(self, deg):
        rng = np.random.default_rng(10)
        pts = _sample_grid(_plane(3.0, deg), rng=rng, noise_m=0.002)
        fused, sem_map, maps = _apply([pts] * 3)
        valid = maps["valid"]
        frac = _safe_frac(sem_map, valid)
        core = valid.copy()
        core[:16, :] = core[-16:, :] = False
        core[:, :16] = core[:, -16:] = False
        slope_mean = float(np.nanmean(maps["slope_deg"][core]))
        return frac, slope_mean, maps

    def test_flat_and_5deg_safe(self):
        for deg in (0.0, 5.0):
            frac, slope_mean, maps = self._case(deg)
            self.assertGreaterEqual(frac, 0.98, f"{deg}° 应近全安全, 实际 {frac:.3f}")
            self.assertLess(slope_mean, float(deg) + 2.0)

    def test_15deg_danger_and_slope_map(self):
        frac, slope_mean, maps = self._case(15.0)
        self.assertLessEqual(frac, 0.02, f"15° 应近全危险, 实际 {frac:.3f}")
        self.assertAlmostEqual(slope_mean, 15.0, delta=2.5)


# ──────────────────────────────────────────────
# 测试 3: 厘米级草叶噪声 → 无成片碎块危险
# ──────────────────────────────────────────────
class GrassNoiseTest(unittest.TestCase):
    def test_cm_noise_no_mass_danger(self):
        rng = np.random.default_rng(42)
        pts = _sample_grid(_plane(3.0, 0.0), rng=rng, noise_m=0.02)
        fused, sem_map, maps = _apply([pts] * 3)
        valid = maps["valid"]
        self.assertGreaterEqual(_safe_frac(sem_map, valid), 0.95)
        rough = maps["roughness"]
        self.assertLess(float(np.nanmedian(rough[valid])), 0.05,
                        "厘米级噪声粗糙度应远低于 0.15 m 阈值")
        # 危险不得成片: 危险格应稀疏 (< 5%), 且无大片连通危险区
        danger = sem_map == 9
        self.assertLess(float(danger.mean()), 0.05)


# ──────────────────────────────────────────────
# 测试 4: 0.3 m 柱体 → 柱体危险、草坪安全、可区分
# ──────────────────────────────────────────────
class PillarTest(unittest.TestCase):
    def test_pillar_danger_grass_safe_distinguishable(self):
        rng = np.random.default_rng(7)
        ground = _sample_grid(_plane(3.0, 0.0), rng=rng, noise_m=0.002)
        # 柱体: 0.18×0.18 m 顶面, 高出地面 0.3 m (z-down: 顶面 2.7 m)
        px, py = np.linspace(-0.09, 0.09, 7), np.linspace(-0.09, 0.09, 7)
        xg, yg = np.meshgrid(px, py)
        pillar = np.column_stack([xg.ravel(), yg.ravel(),
                                  np.full(xg.size, 2.7)]).astype(np.float32)
        fused, sem_map, maps = _apply(
            [np.concatenate([ground, pillar])] * 3)
        valid = maps["valid"]

        # 柱体中心区 (半径 ~2 格) 必须危险
        r0, c0 = _cell_index_xy(0.0, 0.0)
        r1, c1 = _cell_index_xy(0.05, 0.05)
        r2, c2 = _cell_index_xy(-0.05, -0.05)
        pillar_cells = (slice(min(r1, r2), max(r1, r2) + 1),
                        slice(min(c1, c2), max(c1, c2) + 1))
        self.assertTrue((sem_map[pillar_cells] == 9).all(),
                        "柱体中心区应全部危险")

        # 突出高度图可区分: 柱体 ≈ 0.3 m, 远处草坪 ≈ 0
        prom = maps["prominence_coarse"]
        self.assertGreaterEqual(float(np.nanmax(prom[pillar_cells])), 0.2)
        # 草坪采样区须完全避开柱体 (含细/粗窗口半径, 柱体 cols 60-67):
        # y ∈ [0.35, 1.6] (rows 11..41), x ∈ [0.7, 2.0] (cols 86..126)
        rg, cg = _cell_index_xy(0.7, 0.7)
        grass = sem_map[rg - 30:rg, cg:cg + 40]
        gvalid = valid[rg - 30:rg, cg:cg + 40]
        self.assertGreaterEqual(_safe_frac(grass, gvalid), 0.99)
        self.assertLessEqual(float(np.nanmax(prom[rg - 5:rg + 5, cg - 5:cg + 5])),
                             0.03)

        # 危险格总量受限 (仅柱体区及其边缘), 不产生成片草叶误报
        self.assertLess(float((sem_map == 9).mean()), 0.03)


# ──────────────────────────────────────────────
# 测试 5: 十帧均未观测扇区 → 保持 unknown
# ──────────────────────────────────────────────
class UnobservedSectorTest(unittest.TestCase):
    def test_permanent_sector_stays_unknown(self):
        rng = np.random.default_rng(11)
        # 扇区: x > 1.0 且 y < -1.0 (右下角, 行/列 >= 96)
        sector = np.zeros((G, G), dtype=bool)
        sector[96:, 96:] = True
        pts = _sample_grid(_plane(3.0, 0.0), rng=rng,
                           exclude=(1.0, 2.0, -2.0, -1.0))
        frames = [pts.copy() for _ in range(10)]
        fused, sem_map, maps = _apply(frames)

        self.assertEqual(int(fused.valid[sector].sum()), 0)
        # 未观测格: 语义掩码 False 且地图值为 danger_id (不可能是 safe)
        self.assertFalse(maps["valid"][sector].any())
        self.assertTrue((sem_map[sector] == 9).all())
        # 填充/平滑不得把未观测区升级为安全 (语义掩码 == 真实观测掩码)
        np.testing.assert_array_equal(maps["valid"], fused.valid)
        self.assertFalse(maps["semantic_valid"][sector].any())
        # 其余区域照常安全
        ok = maps["valid"] & ~sector
        self.assertGreaterEqual(_safe_frac(sem_map, ok), 0.95)


# ──────────────────────────────────────────────
# 测试 6: 填洞方向一致 + 无 NaN 扩散
# ──────────────────────────────────────────────
class HoleFillTest(unittest.TestCase):
    def test_fill_valid_nearest_label_semantics_equivalent(self):
        """紧凑索引 label (cv2<5) 与分量 label (cv2>=5) 填充结果一致且 = 暴力最近."""
        rng = np.random.default_rng(5)
        valid = rng.random((12, 12)) > 0.35
        values = rng.normal(size=(12, 12)).astype(np.float32)
        vp = np.argwhere(valid)
        hp = np.argwhere(~valid)
        d2 = ((hp[:, None, :] - vp[None, :, :]) ** 2).sum(-1)
        nearest = np.argmin(d2, axis=1)
        ref = values.copy()
        ref[hp[:, 0], hp[:, 1]] = values[vp[nearest, 0], vp[nearest, 1]]

        # 紧凑索引: 每个有效单元独立 label 1..N
        lab_compact = np.zeros((12, 12), dtype=np.int32)
        lab_compact[vp[:, 0], vp[:, 1]] = np.arange(1, len(vp) + 1)
        lab_compact[hp[:, 0], hp[:, 1]] = nearest + 1
        # 分量: 相邻有效单元共享 label (2×2 块合并, 模拟 cv2>=5 CCOMP 语义)
        lab_comp = lab_compact.copy()
        for r in range(0, 12, 2):
            for cc in range(0, 12, 2):
                blk = lab_comp[r:r + 2, cc:cc + 2]
                if blk.max() > 0:
                    lab_comp[r:r + 2, cc:cc + 2] = np.where(
                        blk > 0, blk.min(), blk)
        self.assertLess(len(np.unique(lab_comp[valid])),
                        len(np.unique(lab_compact[valid])),
                        "分量合并必须实际发生, 否则用例无意义")
        # 洞处 label 仍指向其最近有效单元 (所属分量)
        lab_comp[hp[:, 0], hp[:, 1]] = lab_comp[vp[nearest, 0], vp[nearest, 1]]

        out_c = fill_valid_nearest(values, valid, lab_compact)
        out_g = fill_valid_nearest(values, valid, lab_comp)
        np.testing.assert_allclose(out_c, ref, atol=1e-6)
        np.testing.assert_allclose(out_g, ref, atol=1e-6)
        self.assertTrue(np.isfinite(out_c).all())
        self.assertTrue(np.isfinite(out_g).all())

    def test_geometric_hole_fill_no_nan_spread(self):
        """观测足迹内空洞: 填洞后平滑/梯度无 NaN 扩散, 洞格仍不升级为安全."""
        xs, ys = _cell_centers_xy()
        xg, yg = np.meshgrid(xs, ys, indexing="ij")
        z = (3.0 + 0.1 * xg).astype(np.float32)
        z[44:84, 44:84] = np.nan                 # 中心 40×40 空洞
        occupied = ~np.isnan(z)
        bev = BevGrid(
            points=np.column_stack([xg[occupied], yg[occupied],
                                    z[occupied]]).astype(np.float32),
            z_max=z.copy(), z_min=z.copy(),
            z_diff=np.zeros((G, G), dtype=np.float32),
            count=occupied.astype(np.int32), grid_res=G,
            bounds=BOUNDS, stats={})

        sem_map, sem_valid, slope, rough = geometric_semantic_map(
            bev, slope_threshold_deg=10.0, roughness_threshold_m=0.15)
        # 洞格是真实未观测 → 不可能是 safe; 掩码由凸包足迹给出, safe 要求 valid
        self.assertFalse((sem_map[44:84, 44:84] == 1).any())
        # 足迹内 (含洞边缘) 坡度/粗糙度全部有限 → 无 NaN 扩散
        self.assertTrue(np.isfinite(slope[sem_valid]).all())
        self.assertTrue(np.isfinite(rough[sem_valid]).all())
        # 洞外斜坡区照常安全
        outer = sem_valid & ~np.zeros_like(occupied)
        outer[44:84, 44:84] = False
        self.assertGreater(float(np.mean(sem_map[outer] == 1)), 0.9)


# ──────────────────────────────────────────────
# 回归: 批量鲁棒平面拟合 (本机 numpy 批量 solve 修复)
# ──────────────────────────────────────────────
class RobustPlaneFitTest(unittest.TestCase):
    def test_batch_fit_slope_and_outlier_robustness(self):
        rng = np.random.default_rng(1)
        N, K = 64, 49
        ox = (np.arange(7) - 3) * 0.05
        oy = ox.copy()
        X = np.array(np.meshgrid(ox, oy, indexing="ij")).reshape(2, -1)
        a, b, c = 0.3, -0.2, 3.0
        hw = (a * X[0] + b * X[1] + c)[None, :].repeat(N, axis=0)
        hw = hw + rng.normal(0.0, 0.005, size=(N, K))
        hw = hw.astype(np.float32)
        # 每窗 10% 离群点 +1.0 m
        for i in range(N):
            idx = rng.choice(K, int(0.1 * K), replace=False)
            hw[i, idx] += 1.0
        valid_w = np.ones((N, K), dtype=bool)

        params, scale, support = _batch_robust_plane_fit(
            hw, valid_w, X[0].astype(np.float32), X[1].astype(np.float32),
            min_pts=8)
        self.assertTrue(support.all())
        slope = np.degrees(np.arctan(np.hypot(params[:, 0], params[:, 1])))
        expected = np.degrees(np.arctan(math.hypot(a, b)))
        np.testing.assert_allclose(slope, expected, atol=1.0)
        np.testing.assert_allclose(params[:, 2], c, atol=0.05)
        self.assertTrue(np.isfinite(scale).all())

    def test_insufficient_support_not_fitted(self):
        N, K = 4, 9
        hw = np.zeros((N, K), dtype=np.float32)
        valid_w = np.zeros((N, K), dtype=bool)
        valid_w[:, :2] = True                    # 2 < min_pts
        ox = (np.arange(3) - 1) * 0.05
        oy = ox.copy()
        params, scale, support = _batch_robust_plane_fit(
            hw, valid_w, np.tile(ox, 3).astype(np.float32),
            np.repeat(oy, 3).astype(np.float32), min_pts=8)
        self.assertFalse(support.any())
        self.assertTrue((params == 0).all())
        self.assertTrue(np.isnan(scale).all())


class VoxelDedupTest(unittest.TestCase):
    def test_keeps_nearest_surface_and_vertical_span(self):
        pts = np.array([
            [0.001, 0.001, 3.0],     # 体素 A: 最近表面
            [0.002, 0.002, 3.5],     # 体素 A: 垂直跨度远点 → 保留
            [0.003, 0.003, 3.002],   # 体素 A: 重合点 → 丢弃
            [1.000, 1.000, 3.0],     # 体素 B: 唯一
        ], dtype=np.float32)
        out = _voxel_dedup(pts, 0.02)
        self.assertEqual(len(out), 3)
        zs = set(out[:, 2].tolist())
        self.assertEqual(zs, {3.0, 3.5})

    def test_small_span_keeps_only_surface(self):
        pts = np.array([
            [0.001, 0.001, 3.0],
            [0.001, 0.001, 3.002],   # 跨度 < 保留阈值 → 丢弃
        ], dtype=np.float32)
        out = _voxel_dedup(pts, 0.02)
        self.assertEqual(len(out), 1)
        self.assertAlmostEqual(float(out[0, 2]), 3.0)

    def test_complementary_points_preserved(self):
        pts = np.array([
            [0.001, 0.001, 3.0],
            [0.100, 0.100, 3.0],     # 不同体素 → 互补落点保留
        ], dtype=np.float32)
        out = _voxel_dedup(pts, 0.02)
        self.assertEqual(len(out), 2)


class GroundReferenceTest(unittest.TestCase):
    def test_stale_snapshot_ground_ref_not_trusted(self):
        # 规格: 不再信任快照未更新的 ground_z_ref_m=0.0 (陈旧), 一律请求
        # 起飞前稳定 odom 自动估计
        cfg = {"mission_state": {"ground_z_ref_m": 0.0}}
        value, source = resolve_ground_z(None, cfg)
        self.assertIsNone(value)
        self.assertEqual(source, "auto_estimate")

    def test_cli_ground_reference_has_highest_priority(self):
        cfg = {"mission_state": {"ground_z_ref_m": 0.0}}
        value, source = resolve_ground_z(1.25, cfg)
        self.assertEqual(value, 1.25)
        self.assertEqual(source, "--ground-z")

    def test_missing_reference_requests_auto_estimate(self):
        value, source = resolve_ground_z(None, {})
        self.assertIsNone(value)
        self.assertEqual(source, "auto_estimate")


class RelativeHeightRenderTest(unittest.TestCase):
    def test_custom_lut_shape_colors_and_invalid_black(self):
        try:
            from replay_compare_common import render_rel_height_bgr
        except Exception as exc:
            self.skipTest(f"OpenCV unavailable: {exc}")
        rel = np.array([[-0.3, 0.0, 0.3], [0.0, 0.0, 0.0]],
                       dtype=np.float32)
        valid = np.array([[True, True, True], [False, False, False]])
        out = render_rel_height_bgr(rel, valid, half_range_m=0.3)
        self.assertEqual(out.shape, (2, 3, 3))
        self.assertGreater(int(out[0, 0, 0]), int(out[0, 0, 2]))  # below=blue
        self.assertGreater(int(out[0, 2, 2]), int(out[0, 2, 0]))  # above=red
        self.assertTrue((out[1] == 0).all())


# ──────────────────────────────────────────────
# 保守投影模式 (需要 perception, 本机无 torch 时跳过)
# ──────────────────────────────────────────────
try:
    from perception.training_camera_projection import (  # noqa: E402
        TrainingCameraModel,
        project_training_camera,
    )
    HAVE_PERCEPTION = True
except Exception:
    HAVE_PERCEPTION = False


@unittest.skipUnless(HAVE_PERCEPTION, "perception 依赖 torch, 本机不可用")
class ConservativeProjectionTest(unittest.TestCase):
    def test_fill_unobserved_forbids_hull_fill(self):
        camera = TrainingCameraModel()
        bounds = {"x_min": -5.0, "x_max": 5.0, "y_min": -5.0, "y_max": 5.0}
        sem = np.ones((64, 64), dtype=np.uint8)
        pts = np.array([
            [0.0, 0.0, 2.0], [0.3, 0.0, 2.0], [-0.3, 0.0, 2.0],
            [0.0, 0.3, 2.0], [0.0, -0.3, 2.0],
        ], dtype=np.float32)

        depth, dvalid, labels, sv = project_training_camera(
            pts, sem, bounds, camera, fill_unobserved=False)
        # 语义掩码 == 深度有效掩码: 凸包内未观测像素不被填充升级
        np.testing.assert_array_equal(sv, dvalid)
        self.assertTrue((labels[~dvalid] == 9).all())

        # 回归: 默认填充模式仍扩展语义掩码
        _, _, _, sv_fill = project_training_camera(
            pts, sem, bounds, camera, fill_unobserved=True)
        self.assertGreaterEqual(int(sv_fill.sum()), int(dvalid.sum()))


if __name__ == "__main__":
    unittest.main()
