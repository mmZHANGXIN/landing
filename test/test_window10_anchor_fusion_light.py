#!/usr/bin/env python3
"""15 m 锚点地图 + 近十帧窗口 + 连续地形语义重建 — 规格测试计划.

规格: 地面基准 --ground-z > 起飞前稳定 odom 自动估计 (> 首帧回退), 不再
信任快照 ground_z_ref_m=0.0; 平滑高度曾高于 20 m 且下降穿越 15 m 触发一次
锚点融合 (起飞经过 15 m 不误触发); 锚点由触发后 10 帧静态建立并保留到着陆;
窗口负责时效更新, 与锚点冲突 (> 0.15 m) 不做平均、连续近表面为障碍;
受限配准 (重叠残差平面 → Δz/Δroll/Δpitch) 超限拒绝且不污染锚点;
连续地形语义三掩码: 真实观测/小孔洞 (≤ 0.25 m 多方向支持) 推断/大缺口
unknown; 表面网格仅跨越几何连续且非 unknown 的四邻域栅格化 (障碍台阶与
unknown 边缘不跨三角面).

本文件只依赖 numpy + (函数内部惰性) cv2, 不导入 torch; 涉及 training-camera
网格投影的用例使用 perception.training_camera_projection (纯 numpy).

覆盖 (对应规格「测试计划」):
  1. 触发序列: 起飞穿 15 m 不触发; 曾达 20 m 后下降穿越 15 m 触发一次;
     触发幂等; 未达 20 m 前振荡不触发;
  2. 地面自动估计: 起飞前稳定段中位 (≈3.47 m); 空中开始回退首帧;
     resolve_ground_z 优先级 CLI > 自动估计;
  3. 锚点生命周期: 互补两帧填满网格 (来源帧掩码齐全); 构建后静态保留,
     窗口移走 (5 m 低空) 时锚点仍补全覆盖; 多帧环形缓冲表面稳定;
  4. 融合冲突: 一致区域 0.6/0.4 加权; 连续障碍冲突 → 近表面 (min) 且不
     平均; 孤立冲突低置信/unknown;
  5. 连续地形语义: 平地全安全 (无棋盘格危险); 15° 坡危险; 厘米噪声不成片
     危险; 0.3 m 柱体局部危险 + 突出高度 ≈ 0.3; 小孔洞 (≤ 0.25 m) 重建,
     大缺口 unknown (不凸包填充);
  6. 受限配准: 注入已知 Δz/Δroll 残差平面 → 校正恢复; 超限拒绝且锚点
     不变; 重叠不足拒绝;
  7. 表面网格: 连续平面全覆盖; 障碍台阶边缘与 unknown 缺口不跨越三角面
     (深度只取两侧值, 无过渡插值).
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

from replay_anchor_fusion import (  # noqa: E402
    AnchorSemanticBranch,
    AnchorVoxelMap,
    fit_registration_correction,
    fuse_anchor_and_window,
    overlap_residual,
    project_surface_mesh,
)
from replay_window10 import (FusionTrigger, GroundEstimator, WindowEntry,
                             fuse_window, resolve_ground_z)  # noqa: E402

# training-camera 几何 (752×480 @ fx=fy=455 → 128×128), 与
# perception.training_camera_projection.TrainingCameraModel 默认值一致;
# 网格投影只依赖相机属性, 内联定义避免包导入触发 torch
class _Camera:
    output_width = 128
    output_height = 128
    near_m = 0.05
    far_m = 30.0
    fx = 455.0 * 128 / 752
    fy = 455.0 * 128 / 480
    cx = 376.5 * 128 / 752
    cy = 240.5 * 128 / 480


CAMERA = _Camera()

G = 128
BOUNDS = {"x_min": -2.0, "x_max": 2.0, "y_min": -2.0, "y_max": 2.0}
X_SPAN = float(BOUNDS["x_max"] - BOUNDS["x_min"])
Y_SPAN = float(BOUNDS["y_max"] - BOUNDS["y_min"])
CELL = X_SPAN / max(G - 1, 1)          # 约 0.0315 m
ZERO_POSE = np.zeros(6, dtype=np.float64)
SLOPE_TH = 10.0
ROUGH_TH = 0.15
PROM_TH = 0.15
FILL_RADIUS_M = 0.25


# ──────────────────────────────────────────────
# 合成数据辅助
# ──────────────────────────────────────────────
def _cell_centers_xy():
    c = np.arange(G, dtype=np.float64)
    xs = BOUNDS["x_min"] + c * CELL
    ys = BOUNDS["y_min"] + c * CELL
    return xs, ys


def _sample_grid(z_fn, rng=None, noise_m=0.0, exclude=None):
    """全网格每格 1 点 (格中心 + 单元内 ±0.3 格抖动), z = z_fn(x, y) [+ 噪声]."""
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


def _fuse(frames_points, voxel_m=0.02, pose_delta=None):
    return fuse_window(_entries(frames_points), ZERO_POSE, BOUNDS, G,
                       voxel_m=voxel_m, pose_delta=pose_delta)


def _build_anchor(frames_points, stamps=None, frame_ids=None,
                  voxel_m=0.02):
    """锚点体素图 (世界坐标 = ZERO_POSE 下水平机体坐标)."""
    anchor = AnchorVoxelMap(voxel_m=voxel_m)
    for k, p in enumerate(frames_points):
        anchor.add(p, float(stamps[k]) if stamps else 1000.0 + k,
                   int(frame_ids[k]) if frame_ids else k)
    return anchor


def _cell_stats(anchor, min_down=0.05, max_down=30.0):
    return anchor.world_to_cell_stats(ZERO_POSE, BOUNDS, G, min_down, max_down)


def _fuse_anchor(window_frames, anchor_frames, conflict_m=0.15,
                 min_down=0.05, max_down=30.0):
    """窗口融合 + 锚点统计 + 融合 → (fused, aw, ac)."""
    fused = _fuse(window_frames)
    anchor = _build_anchor(anchor_frames)
    ac = _cell_stats(anchor, min_down, max_down)
    aw = fuse_anchor_and_window(fused, ac, conflict_m=conflict_m)
    return fused, aw, ac


def _sem(aw, **kw):
    """AnchorSemanticBranch → (sem_map, maps)."""
    kw.setdefault("slope_threshold_deg", SLOPE_TH)
    kw.setdefault("roughness_threshold_m", ROUGH_TH)
    kw.setdefault("prominence_threshold_m", PROM_TH)
    kw.setdefault("surface_fill_radius_m", FILL_RADIUS_M)
    branch = AnchorSemanticBranch(**kw)
    sem_map, sem_info = branch(aw)
    return sem_map, sem_info["maps"]


def _safe_frac(sem_map, valid):
    return float(np.mean(sem_map[valid] == 1))


def _cell_index_xy(x, y):
    col = int(np.rint((x - BOUNDS["x_min"]) / X_SPAN * (G - 1)))
    row_un = int(np.rint((y - BOUNDS["y_min"]) / Y_SPAN * (G - 1)))
    return (G - 1) - row_un, col          # (row, col), 行 0 = +y


# ──────────────────────────────────────────────
# 1. 触发序列
# ──────────────────────────────────────────────
class FusionTriggerTest(unittest.TestCase):
    def _feed(self, trigger, heights):
        return [trigger.update(h) for h in heights]

    def test_takeoff_through_start_height_no_trigger(self):
        trigger = FusionTrigger(15.0, 20.0, alpha=1.0)
        hits = [t for t in self._feed(trigger, list(range(0, 26))) if t]
        self.assertEqual(hits, [])          # 起飞穿 15 m: 未达 20 m / 上升

    def test_descent_crossing_triggers_once(self):
        trigger = FusionTrigger(15.0, 20.0, alpha=1.0)
        up = list(range(0, 26))             # 0 → 25 m (曾高于 20 m)
        down = list(range(24, 8, -1))       # 24 → 9 m (下降穿越 15 m)
        hits = [t for t in self._feed(trigger, up + down) if t]
        self.assertEqual(hits, [True])      # 恰好触发一次, 且仅在下降穿越时

    def test_armed_but_never_crosses_no_trigger(self):
        trigger = FusionTrigger(15.0, 20.0, alpha=1.0)
        heights = list(range(0, 22)) + [21 - k % 2 for k in range(30)]
        self.assertEqual([t for t in self._feed(trigger, heights) if t], [])

    def test_below_arm_oscillation_no_trigger(self):
        trigger = FusionTrigger(15.0, 20.0, alpha=1.0)
        heights = [5 + k % 3 for k in range(100)]   # 5–7 m 振荡
        self.assertEqual([t for t in self._feed(trigger, heights) if t], [])

    def test_ema_smoothing_requires_sustained_descent(self):
        # alpha<1: 平滑滞后, 需持续下降到 15 m 之下才触发; 触发后幂等
        trigger = FusionTrigger(15.0, 20.0, alpha=0.3)
        heights = list(range(0, 26)) + [20, 15, 12, 10, 8]
        hits = [t for t in self._feed(trigger, heights) if t]
        self.assertEqual(len(hits), 1)         # 恰好一次


# ──────────────────────────────────────────────
# 2. 地面自动估计
# ──────────────────────────────────────────────
class GroundEstimatorTest(unittest.TestCase):
    def test_stable_pre_takeoff_median(self):
        est = GroundEstimator(min_samples=15, stable_spread_m=0.2)
        rng = np.random.default_rng(7)
        for z in 3.47 + rng.normal(0.0, 0.01, 20):
            est.update(float(z))
        self.assertIsNotNone(est.ground)
        self.assertAlmostEqual(est.ground, 3.47, delta=0.02)
        self.assertEqual(est.source, "odom_pre_takeoff_stable")

    def test_takeoff_drift_keeps_locked_value(self):
        est = GroundEstimator(min_samples=15, stable_spread_m=0.2)
        rng = np.random.default_rng(7)
        for z in 3.47 + rng.normal(0.0, 0.01, 15):
            est.update(float(z))
        locked = est.ground
        for z in 3.5 + np.arange(0, 25.0, 0.5):     # 起飞爬升
            est.update(float(z))
        self.assertEqual(est.ground, locked)        # 锁定后不再变化

    def test_airborne_start_falls_back_first_pose(self):
        est = GroundEstimator(min_samples=15, stable_spread_m=0.2,
                              max_lookback=50)
        rng = np.random.default_rng(3)
        z0 = 18.4
        est.update(z0)
        for _ in range(49):
            est.update(z0 + float(rng.uniform(-1.0, 1.0)))
        self.assertEqual(est.ground, z0)       # 回退首帧
        self.assertEqual(est.source, "first_pose_fallback")

    def test_ground_z_priority_cli_over_auto(self):
        value, source = resolve_ground_z(3.47, {})
        self.assertEqual(value, 3.47)
        self.assertEqual(source, "--ground-z")


# ──────────────────────────────────────────────
# 3. 锚点生命周期
# ──────────────────────────────────────────────
class AnchorVoxelMapTest(unittest.TestCase):
    def test_complementary_frames_fill_grid(self):
        plane = _plane(3.0, 0.0)
        left = _sample_grid(plane, exclude=(-0.1, 2.0, -2.0, 2.0))
        right = _sample_grid(plane, exclude=(-2.0, 0.1, -2.0, 2.0))
        anchor = _build_anchor([left, right])
        ac = _cell_stats(anchor)
        self.assertIsNotNone(ac)
        self.assertGreater(ac.mask.sum(), G * G * 0.9)   # 互补两帧填满
        # 来源帧掩码: 两帧的位都应出现
        self.assertTrue(bool((ac.frames & 1).any()))
        self.assertTrue(bool((ac.frames & 2).any()))
        # 表面 ≈ 平面
        valid = ac.mask
        self.assertLess(float(np.nanmedian(np.abs(ac.surface[valid] - 3.0))),
                        1e-2)

    def test_anchor_static_after_build_retained(self):
        # 触发后 10 帧建立 → 之后不再 add: 统计必须逐帧一致 (静态保留到着陆)
        rng = np.random.default_rng(11)
        frames = [_sample_grid(_plane(3.0, 0.0), rng=rng, noise_m=0.01)
                  for _ in range(10)]
        anchor = _build_anchor(frames)
        ac1 = _cell_stats(anchor)
        ac2 = _cell_stats(anchor)           # 窗口后续帧不再触碰锚点
        self.assertTrue(np.array_equal(ac1.mask, ac2.mask))
        self.assertTrue(np.array_equal(ac1.surface[ac1.mask],
                                       ac2.surface[ac2.mask]))

    def test_window_moves_away_anchor_still_fills(self):
        # 低空 (5 m) 窗口只观测一小片, 锚点 (15 m 下降段建立) 补全覆盖
        plane = _plane(3.0, 0.0)
        anchor_frames = [_sample_grid(plane) for _ in range(10)]
        anchor = _build_anchor(anchor_frames)
        ac = _cell_stats(anchor)
        small = _sample_grid(plane, exclude=(-2.0, -0.5, -2.0, 2.0))
        fused = _fuse([small])
        aw = fuse_anchor_and_window(fused, ac)
        # 锚点补洞: 窗口未观测区 surface 由锚点填充
        window_only = fused.valid
        anchor_only = aw.anchor_mask & ~aw.observed
        self.assertGreater(int(anchor_only.sum()), 0)
        filled = aw.surface[anchor_only]
        self.assertTrue(np.isfinite(filled).all())
        self.assertLess(float(np.nanmedian(np.abs(filled - 3.0))), 1e-2)
        # 保留到着陆: 冲突区为空, 锚点覆盖 + 窗口观测全覆盖
        self.assertFalse(aw.conflict.any())

    def test_ring_buffer_surface_stable_many_frames(self):
        rng = np.random.default_rng(5)
        pts = _sample_grid(_plane(3.0, 0.0), rng=rng, noise_m=0.02)
        anchor = _build_anchor([pts] * 100, frame_ids=range(100))  # >32 采样
        ac = _cell_stats(anchor)
        valid = ac.mask
        self.assertLess(float(np.nanmedian(np.abs(ac.surface[valid] - 3.0))),
                        0.05)               # 环形缓冲最近采样分位数仍稳健


# ──────────────────────────────────────────────
# 4. 窗口/锚点融合与冲突
# ──────────────────────────────────────────────
class AnchorWindowFusionTest(unittest.TestCase):
    def test_agree_weighted_mean_and_masks(self):
        plane = _plane(3.0, 0.0)
        fused, aw, ac = _fuse_anchor(
            [_sample_grid(plane, noise_m=0.005)],
            [_sample_grid(plane, noise_m=0.005)])
        self.assertTrue(aw.observed.any())
        self.assertTrue(aw.anchor_mask.any())
        self.assertFalse(aw.conflict.any())
        agree = aw.observed & aw.anchor_mask
        self.assertLess(float(np.nanmedian(np.abs(
            aw.surface[agree] - aw.anchor_surface[agree]))), 0.02)

    def test_conflict_continuous_obstacle_near_surface(self):
        # 锚点: 平地 3.0; 窗口: 中央 10×10 格障碍 (近表面 2.7 m) → 冲突区
        # 不平均, surface = min (近表面 = 障碍顶)
        plane = _plane(3.0, 0.0)
        base = _sample_grid(plane)
        z = np.where((np.abs(base[:, 0]) < 0.35) & (np.abs(base[:, 1]) < 0.35),
                     2.7, 3.0)
        obs = np.column_stack([base[:, 0], base[:, 1], z]).astype(np.float32)
        fused, aw, ac = _fuse_anchor([obs], [base])
        self.assertTrue(aw.conflict.any())
        conf = aw.conflict
        # 冲突区 surface = 近表面 (2.7), 不是平均值 2.85
        self.assertLess(float(np.nanmedian(np.abs(aw.surface[conf] - 2.7))),
                        1e-2)
        self.assertLess(float(np.nanmedian(np.abs(aw.near[conf] - 2.7))), 1e-2)
        # 冲突区内部为连续近表面 → 语义判障碍 (突出高度 ~0.3)
        sem_map, maps = _sem(aw)
        self.assertGreater(float(np.mean(sem_map[conf] == 9)), 0.9)
        self.assertLess(float(np.nanmedian(np.abs(
            maps["prominence"][conf] - 0.3))), 0.05)

    def test_isolated_conflict_low_confidence_unknown(self):
        # 单格孤立冲突 (无空间连续性) → 不判障碍也不判安全:
        # semantic_valid 排除 → 显示灰色, DRL 输入保持危险值 (低置信/unknown)
        plane = _plane(3.0, 0.0)
        base = _sample_grid(plane)
        z = base[:, 2].copy()
        r, c = _cell_index_xy(0.5, 0.5)
        idx = int(np.flatnonzero((np.rint((base[:, 0] - BOUNDS["x_min"])
                                           / X_SPAN * (G - 1)) == c)
                                 & ((G - 1) - np.rint(
                                     (base[:, 1] - BOUNDS["y_min"])
                                     / Y_SPAN * (G - 1)) == r))[0])
        z[idx] = 2.5
        obs = np.column_stack([base[:, 0], base[:, 1], z]).astype(np.float32)
        fused, aw, ac = _fuse_anchor([obs], [base])
        self.assertTrue(aw.conflict.any())
        sem_map, maps = _sem(aw)
        rr, cc = np.where(aw.conflict)
        self.assertFalse(maps["semantic_valid"][rr[0], cc[0]])
        self.assertEqual(int(sem_map[rr[0], cc[0]]), 9)   # 不升级为安全
        # 相邻正常地面不受影响: 全部安全
        safe_nb = maps["semantic_valid"].sum()
        self.assertGreater(safe_nb, 0.9 * int(aw.observed.sum()))


# ──────────────────────────────────────────────
# 5. 连续地形语义
# ──────────────────────────────────────────────
class AnchorSemanticTest(unittest.TestCase):
    def test_flat_ground_all_safe_no_checkerboard(self):
        fused, aw, ac = _fuse_anchor(
            [_sample_grid(_plane(3.0, 0.0), noise_m=0.005)],
            [_sample_grid(_plane(3.0, 0.0), noise_m=0.005)])
        sem_map, maps = _sem(aw)
        valid = aw.observed | aw.anchor_mask
        self.assertGreater(_safe_frac(sem_map, valid), 0.99)  # 平地无棋盘格
        self.assertGreater(float(maps["observed_mask"].mean()), 0.9)

    def test_15deg_slope_danger(self):
        fused, aw, ac = _fuse_anchor(
            [_sample_grid(_plane(3.0, 15.0))],
            [_sample_grid(_plane(3.0, 15.0))])
        sem_map, maps = _sem(aw)
        valid = aw.observed
        self.assertGreater(float(np.mean(sem_map[valid] == 9)), 0.9)
        self.assertGreater(float(np.nanmedian(maps["slope_deg"][valid])), 14.0)

    def test_cm_grass_noise_not_mass_danger(self):
        rng = np.random.default_rng(2)
        fused, aw, ac = _fuse_anchor(
            [_sample_grid(_plane(3.0, 0.0), rng=rng, noise_m=0.02)],
            [_sample_grid(_plane(3.0, 0.0), rng=rng, noise_m=0.02)])
        sem_map, maps = _sem(aw)
        valid = aw.observed
        # 厘米级草叶噪声: 不成片危险, 粗糙度远低于阈值
        danger = sem_map == 9
        self.assertLess(float(danger[valid].mean()), 0.05)
        self.assertGreater(_safe_frac(sem_map, valid), 0.9)
        self.assertLess(float(np.nanmedian(maps["roughness"][valid])), 0.05)

    def test_pillar_prominence_localized_danger(self):
        plane = _plane(3.0, 0.0)
        base = _sample_grid(plane)
        z = np.where((np.abs(base[:, 0]) < 0.3) & (np.abs(base[:, 1]) < 0.3),
                     2.7, 3.0)
        obs = np.column_stack([base[:, 0], base[:, 1], z]).astype(np.float32)
        fused, aw, ac = _fuse_anchor([obs], [base])
        sem_map, maps = _sem(aw)
        r, c = _cell_index_xy(0.0, 0.0)
        patch = np.zeros((G, G), dtype=bool)
        patch[r - 3:r + 4, c - 3:c + 4] = True
        # 柱体区危险, 周围草坪安全
        self.assertGreater(float(np.mean(sem_map[patch & aw.observed] == 9)),
                           0.9)
        around = aw.observed & ~patch
        self.assertGreater(_safe_frac(sem_map, around), 0.95)
        # 突出高度图 ≈ 0.3 m
        self.assertLess(float(np.nanmedian(
            np.abs(maps["prominence"][patch & aw.observed] - 0.3))), 0.06)

    def test_small_hole_reconstructed_large_gap_unknown(self):
        # 2×2 格孔洞 (≈0.06 m ≪ 0.25 m) → 推断重建; 12×12 缺口 (≈0.38 m)
        # → 中心保持 unknown, 表面不填充
        plane = _plane(3.0, 0.0)
        hole = _sample_grid(plane, exclude=(-0.06, 0.06, -0.06, 0.06))
        fused, aw, ac = _fuse_anchor([hole], [hole])
        sem_map, maps = _sem(aw)
        r, c = _cell_index_xy(0.0, 0.0)
        small = np.zeros((G, G), dtype=bool)
        small[r - 1:r + 2, c - 1:c + 2] = True
        self.assertTrue(bool((maps["inferred_mask"] & small).any()))
        filled = maps["surface"][maps["inferred_mask"] & small]
        self.assertTrue(np.isfinite(filled).all())
        self.assertLess(float(np.nanmedian(np.abs(filled - 3.0))), 0.05)

        big = _sample_grid(plane, exclude=(-0.4, 0.4, -0.4, 0.4))
        fused2, aw2, ac2 = _fuse_anchor([big], [big])
        sem_map2, maps2 = _sem(aw2)
        r2, c2 = _cell_index_xy(0.0, 0.0)
        center = np.zeros((G, G), dtype=bool)
        center[r2 - 2:r2 + 3, c2 - 2:c2 + 3] = True
        unk = maps2["unknown_mask"] & center
        self.assertGreater(int(unk.sum()), 4)       # 大缺口中心保持 unknown
        self.assertTrue(np.isnan(maps2["surface"][unk]).all())

    def test_window_and_anchor_mutual_gap_stays_unknown(self):
        # 窗口只观测左侧、锚点只观测右侧, 中间带两侧都未观测 → unknown
        # (不得凸包/最近邻填充升级为安全)
        plane = _plane(3.0, 0.0)
        win = _sample_grid(plane, exclude=(-0.2, 2.0, -2.0, 2.0))   # x < −0.2
        anc = _sample_grid(plane, exclude=(-2.0, 0.2, -2.0, 2.0))   # x > 0.2
        fused, aw, ac = _fuse_anchor([win], [anc])
        sem_map, maps = _sem(aw)
        # 中间带 (x ∈ [−0.2, 0.2], ≈12 格) 保持 unknown
        r_lo, c_lo = _cell_index_xy(-0.18, 0.0)
        r_hi, c_hi = _cell_index_xy(0.18, 0.0)
        band = np.zeros((G, G), dtype=bool)
        band[:, c_lo:c_hi + 1] = True
        unk = maps["unknown_mask"] & band
        self.assertGreater(int(unk.sum()), 8)
        self.assertTrue(np.isnan(maps["surface"][unk]).all())
        # 两侧各自安全
        self.assertGreater(_safe_frac(sem_map, maps["observed_mask"]), 0.9)


# ──────────────────────────────────────────────
# 6. 受限配准
# ──────────────────────────────────────────────
class RegistrationTest(unittest.TestCase):
    def _setup(self, window_z_fn, anchor_z_fn=_plane(3.0, 0.0),
               anchor_frames=5):
        anchor = _build_anchor([_sample_grid(anchor_z_fn)
                                for _ in range(anchor_frames)])
        ac = _cell_stats(anchor)
        fused = _fuse([_sample_grid(window_z_fn)])
        return fused, ac

    def test_recovers_injected_z_offset(self):
        fused, ac = self._setup(lambda x, y: 3.0 + 0.05 * np.ones_like(x))
        reg = fit_registration_correction(fused, ac)
        self.assertTrue(reg.accepted, reg.reason)
        # Δz = −c ≈ −0.05 (深度域常数校正方向)
        self.assertAlmostEqual(float(reg.delta[0]), -0.05, delta=0.01)
        # 施加校正后重新融合 → 残差收敛到 0
        fused_c = _fuse([_sample_grid(lambda x, y: 3.0 + 0.05 * np.ones_like(x))],
                        pose_delta=reg.delta)
        resid_after = overlap_residual(fused_c, ac)
        self.assertLess(resid_after, 0.01)

    def test_recovers_injected_roll(self):
        # 窗口表面 = 锚点 + 0.03·y → 残差平面 b = 0.03 → Δroll = −0.03
        fused, ac = self._setup(lambda x, y: 3.0 + 0.03 * y)
        reg = fit_registration_correction(fused, ac)
        self.assertTrue(reg.accepted, reg.reason)
        self.assertAlmostEqual(float(reg.delta[1]), -0.03, delta=0.005)
        self.assertAlmostEqual(float(reg.delta[0]), 0.0, delta=0.005)
        fused_c = _fuse([_sample_grid(lambda x, y: 3.0 + 0.03 * y)],
                        pose_delta=reg.delta)
        self.assertLess(overlap_residual(fused_c, ac), 0.01)

    def test_over_limit_correction_rejected_anchor_unpolluted(self):
        fused, ac = self._setup(
            lambda x, y: 3.0 + 0.5 * np.ones_like(x))     # 0.5 m ≫ 0.2 m
        reg = fit_registration_correction(fused, ac,
                                          z_lim_m=0.2, ang_lim_deg=2.0)
        self.assertFalse(reg.accepted)
        self.assertIn("超限", reg.reason)
        # 拒绝不污染锚点: 统计逐字节不变
        ac_again = _cell_stats(_build_anchor(
            [_sample_grid(_plane(3.0, 0.0)) for _ in range(5)]))
        self.assertTrue(np.array_equal(ac.mask, ac_again.mask))
        self.assertTrue(np.array_equal(ac.surface[ac.mask],
                                       ac_again.surface[ac_again.mask]))

    def test_insufficient_overlap_rejected(self):
        # 窗口与锚点观测互补不重叠 → 拒绝
        anchor = _build_anchor([_sample_grid(
            _plane(3.0, 0.0), exclude=(0.0, 2.0, -2.0, 2.0))])
        ac = _cell_stats(anchor)
        fused = _fuse([_sample_grid(
            _plane(3.0, 0.0), exclude=(-2.0, 0.0, -2.0, 2.0))])
        reg = fit_registration_correction(fused, ac)
        self.assertFalse(reg.accepted)
        self.assertIn("overlap", reg.reason)


# ──────────────────────────────────────────────
# 7. 表面网格投影
# ──────────────────────────────────────────────
class MeshProjectionTest(unittest.TestCase):
    def _mesh(self, maps):
        return project_surface_mesh(
            maps["surface"], maps["valid"], np.full((G, G), 1, np.uint8),
            maps["semantic_valid"], BOUNDS, CAMERA, max_step_m=0.12)

    def test_flat_plane_continuous_coverage(self):
        fused, aw, ac = _fuse_anchor(
            [_sample_grid(_plane(3.0, 0.0))],
            [_sample_grid(_plane(3.0, 0.0))])
        sem_map, maps = _sem(aw)
        m_depth, m_valid, m_sem, m_sem_valid = self._mesh(maps)
        self.assertGreater(float(m_valid.mean()), 0.5)   # 连续面全覆盖
        mv = m_valid & np.isfinite(m_depth)
        self.assertLess(float(np.nanmedian(np.abs(m_depth[mv] - 3.0))), 0.05)
        self.assertTrue((m_sem[m_valid] == 1).all())     # 标签 = 地面格安全

    def test_obstacle_edge_not_crossed(self):
        # 中央 0.5 m 高台 (边缘台阶 > max_step 0.12): 边界四边形不生成三角面,
        # 台面与地面各自连续; 边界处深度只取两侧值 (无过渡插值)
        plane = _plane(3.0, 0.0)
        base = _sample_grid(plane)
        z = np.where((np.abs(base[:, 0]) < 0.4) & (np.abs(base[:, 1]) < 0.4),
                     2.5, 3.0)
        obs = np.column_stack([base[:, 0], base[:, 1], z]).astype(np.float32)
        fused, aw, ac = _fuse_anchor([obs], [base])
        sem_map, maps = _sem(aw)
        m_depth, m_valid, m_sem, m_sem_valid = self._mesh(maps)
        # 高台内部连续覆盖
        r, c = _cell_index_xy(0.0, 0.0)
        patch = np.zeros((G, G), dtype=bool)
        patch[r - 5:r + 6, c - 5:c + 6] = True
        inside = patch & maps["valid"] & m_valid
        self.assertGreater(int(inside.sum()), 30)
        self.assertLess(float(np.nanmedian(np.abs(m_depth[inside] - 2.5))),
                        0.06)
        # 台面外平地区域也连续覆盖
        outside = maps["valid"] & ~patch & m_valid
        self.assertGreater(int(outside.sum()), 100)
        self.assertLess(float(np.nanmedian(np.abs(m_depth[outside] - 3.0))),
                        0.06)
        # 边界过渡: 不存在介于 2.5 与 3.0 之间的网格深度 (无跨越三角面)
        mid = m_valid & np.isfinite(m_depth)
        odd = np.abs(m_depth[mid] - 2.75) < 0.15
        self.assertLess(float(odd.mean()), 0.02)

    def test_unknown_gap_not_crossed(self):
        # 中央 12×12 缺口 (unknown): 缺口本身不被网格覆盖, 两侧各自连续
        plane = _plane(3.0, 0.0)
        gap = _sample_grid(plane, exclude=(-0.25, 0.25, -0.25, 0.25))
        fused, aw, ac = _fuse_anchor([gap], [gap])
        sem_map, maps = _sem(aw)
        m_depth, m_valid, m_sem, m_sem_valid = self._mesh(maps)
        r, c = _cell_index_xy(0.0, 0.0)
        center = np.zeros((G, G), dtype=bool)
        center[r - 3:r + 4, c - 3:c + 4] = True
        self.assertFalse(m_valid[center].any())         # 缺口不跨越
        side = maps["valid"] & m_valid
        self.assertGreater(int(side.sum()), 100)


if __name__ == "__main__":
    unittest.main(verbosity=2)
