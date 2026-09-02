#!/usr/bin/env python3
"""world-first 地图累积 + 两阶段融合质控 + 可配置 BEV 分辨率 + 诊断入口 — 规格测试.

规格 (2026-08-21):
  1. BEV 分辨率可配置: --bev-grid-res (64) / --bev-cell-size-m (0.0) /
     --model-grid-res (128); cell-size>0 时按 ROI 物理范围自动算 grid;
     物理网格融合 → 最近邻上采样到 model-grid, 上采样绝不伪造 observed;
  2. 世界坐标优先累积: 去畸变云 → 外参 → 帧时间戳位姿 → 世界坐标
     (完整云, 绝不预裁剪) → 世界滚动地图 → 当前位姿裁剪; 关键帧策略
     (--keyframe-min-translation-m 0.15 / --keyframe-min-yaw-deg 3.0);
  3. 两阶段融合质控: 阶段一粗对齐网格 (0.30 m) 中位修正 + MAD 离散度;
     阶段二逐单元接受 (min-points / max-height-span / 已被 current 或
     更新历史覆盖→dup 拒绝); 整帧拒绝仅限 overlap 不足 / |z_corr| 超限 /
     空云 / NaN / 位姿同步失败;
  4. 第 1060 帧诊断: --frame-index 1060 与 --cloud-seq 1557 定位同一帧;
     不加载 torch; 输出 PNG + JSON.

坐标约定 (与 replay_window10 body_to_world 一致):
  - 输入去畸变云 z-up (ENU); leveling 后 z 取负 → z-down;
  - 世界坐标 W' = 水平 x/y + z-down; world_z = -(z_up) - pz;
  - world_to_level_body 仅 yaw+平移 (世界帧已水平化).

本文件只依赖 numpy + 标准库 + (仅诊断用例) rosbag, 不导入 torch.
运行:
  python3 test/test_window10_world_first_light.py
"""

import math
import subprocess
import sys
import unittest
from pathlib import Path

import numpy as np

_SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(_SCRIPTS))

from replay_compare_common import (  # noqa: E402
    bev_grid_res_from_cell,
    bev_roughness_downsample,
    bev_upsample_to_model,
    roi_bounds,
    upsample_grid_nearest,
    world_to_level_body,
    _rot_z,
    _rot_zyx,
)
from replay_window10 import (  # noqa: E402
    body_to_world,
    fuse_bev_gap_fill,
    fuse_bev_world_first,
    window_evict_age,
    window_keyframe_append,
    WindowEntry,
)

# 实验 bag / 配置路径 (第 1060 帧验收用; 缺失则跳过对应用例)
_EXP_DIR = Path(__file__).resolve().parent.parent.parent / \
    "experiments" / "20260807_162946_orin_landing"
_BAG = _EXP_DIR / "input.bag"
_CONFIG = _EXP_DIR / "experiment_config_snapshot.yaml"

_IDENTITY_PERC = {
    "body_R_from_lidar_imu": [1, 0, 0, 0, 1, 0, 0, 0, 1],
    "body_T_from_lidar_imu": [0, 0, 0],
}

# 合成相机外参: 与实验配置同量级 (前向 +0.18 m, 下 +0.05 m, 15° pitch)
_CAMERA_PERC = {
    "body_R_from_lidar_imu": [1, 0, 0,
                              0, math.cos(0.15), math.sin(0.15),
                              0, -math.sin(0.15), math.cos(0.15)],
    "body_T_from_lidar_imu": [0.0, 0.18, 0.05],
}


def _make_plane_grid(x_span=4.0, y_span=4.0, n=12, z=2.0, seed=7,
                     noise=0.02):
    """合成平面网格点 (z-up 机体系, 输入) + 少量噪声."""
    rng = np.random.default_rng(seed)
    x, y = np.meshgrid(np.linspace(-x_span / 2, x_span / 2, n),
                       np.linspace(-y_span / 2, y_span / 2, n))
    pts = np.column_stack([x.ravel(), y.ravel(),
                           np.full(n * n, z, dtype=np.float64)])
    pts[:, 2] += rng.normal(0.0, noise, pts.shape[0])
    return pts.astype(np.float32)


def _make_dense_plane(x_span=6.0, y_span=6.0, n=2000, z=2.0, seed=7):
    """稠密平面点 (占用随网格分辨率增长)."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(-x_span / 2, x_span / 2, n)
    y = rng.uniform(-y_span / 2, y_span / 2, n)
    zz = np.full(n, z, dtype=np.float64) + rng.normal(0, 0.02, n)
    return np.column_stack([x, y, zz]).astype(np.float32)


def _bounds(pts):
    return roi_bounds(max(float(np.abs(pts[:, 0]).max()) + 0.5, 1.0),
                      max(float(np.abs(pts[:, 1]).max()) + 0.5, 1.0))


# ──────────────────────────────────────────────
# 1. 坐标变换: 外参、平移、yaw/roll/pitch、世界↔水平机体往返
# ──────────────────────────────────────────────
class TestCoordinateTransform(unittest.TestCase):

    def test_extrinsic_rigid_offset(self):
        """外参: 原点经 body_R/T 刚性变换到 base_link (T 平移直接生效)."""
        pts = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
        pose = np.zeros(6, dtype=np.float32)
        out = body_to_world(pts, pose, _CAMERA_PERC)
        # base_link = T = (0, 0.18, 0.05); 水平化 (pose roll/pitch=0) 不变;
        # z-up → z-down 取负 → 世界 z = -0.05
        np.testing.assert_allclose(out[0], [0.0, 0.18, -0.05], atol=1e-4)

    def test_translation_pose(self):
        """平移: 世界坐标 = 水平 x/y + z-down; z-up 输入 1 m 上 → -(1) - pz."""
        pts = np.array([[0.0, 0.0, 1.0]], dtype=np.float32)   # 1 m 上 (z-up)
        pose = np.array([10.0, -5.0, 3.0, 0.0, 0.0, 0.0], dtype=np.float32)
        out = body_to_world(pts, pose, _IDENTITY_PERC)
        np.testing.assert_allclose(out[0], [10.0, -5.0, -1.0 - 3.0], atol=1e-5)

    def test_yaw_rotation(self):
        """yaw: +90° 绕 z (z-down) → 机体 +y 落到世界 -x."""
        pts = np.array([[0.0, 2.0, 1.0]], dtype=np.float32)
        pose = np.array([0.0, 0.0, 0.0, 0.0, 0.0, math.pi / 2.0],
                        dtype=np.float32)
        out = body_to_world(pts, pose, _IDENTITY_PERC)
        np.testing.assert_allclose(out[0, :2], [-2.0, 0.0], atol=1e-5)

    def test_roll_pitch_leveling(self):
        """roll/pitch: 水平化先于 yaw; pitch 45° 时上方 1 m 点 → level 前向
        sin45 + z-down 深度 -cos45 (由 _rot_zyx 公式推导)."""
        pts = np.array([[0.0, 0.0, 1.0]], dtype=np.float32)   # 1 m 上
        pose = np.array([0.0, 0.0, 0.0, 0.0, math.pi / 4.0, 0.0],
                        dtype=np.float32)
        out = body_to_world(pts, pose, _IDENTITY_PERC)
        np.testing.assert_allclose(out[0, 0], math.sin(math.pi / 4), atol=1e-5)
        np.testing.assert_allclose(out[0, 1], 0.0, atol=1e-5)
        np.testing.assert_allclose(out[0, 2], -math.cos(math.pi / 4), atol=1e-5)

    def test_world_round_trip_same_pose_identity(self):
        """往返: 恒等外参 + roll=pitch=0 (世界帧已水平化) 下
        body→world→level_body = [x, y, -z] (yaw+平移精确抵消; z-up 输入 →
        z-down level body, 与 halss ROI 输出同一约定)."""
        rng = np.random.default_rng(11)
        pts = rng.uniform(-3, 3, size=(200, 3)).astype(np.float32)
        pose = np.array([1.5, -2.0, 4.0, 0.0, 0.0, 2.1], dtype=np.float32)
        world = body_to_world(pts, pose, _IDENTITY_PERC)
        back = world_to_level_body(world, pose)
        expected = pts.copy()
        expected[:, 2] *= -1.0
        np.testing.assert_allclose(back, expected, atol=1e-4)

    def test_world_round_trip_extrinsics_invariant(self):
        """往返 (含外参): world_to_level_body(body_to_world(p, pose), pose)
        == leveling∘extrinsics(p); yaw 与平移被精确抵消."""
        rng = np.random.default_rng(13)
        pts = rng.uniform(-2, 2, size=(150, 3)).astype(np.float32)
        pose = np.array([1.0, 0.5, 4.0, 0.03, -0.02, 0.8], dtype=np.float32)
        world = body_to_world(pts, pose, _CAMERA_PERC)
        back = world_to_level_body(world, pose)
        # 期望: leveling (roll/pitch) · extrinsics · p (行列向量约定 p @ R.T)
        r_ext = np.asarray(_CAMERA_PERC["body_R_from_lidar_imu"],
                           dtype=np.float32).reshape(3, 3)
        t_ext = np.asarray(_CAMERA_PERC["body_T_from_lidar_imu"],
                           dtype=np.float32)
        expected = pts @ r_ext.T + t_ext
        expected = expected @ _rot_zyx(float(pose[3]), float(pose[4]), 0.0).T
        expected[:, 2] *= -1.0
        np.testing.assert_allclose(back, expected, atol=1e-4)

    def test_world_round_trip_yaw_rotation_only(self):
        """world_to_level_body 仅 yaw+平移 (不重复水平化/z 翻转): 不同位姿
        对齐 = 机体链 + yaw 差旋转; z 保持 z-down 语义."""
        rng = np.random.default_rng(12)
        pts = rng.uniform(-3, 3, size=(150, 3)).astype(np.float32)
        p1 = np.zeros(6, dtype=np.float32)
        p2 = np.zeros(6, dtype=np.float32)
        p2[5] = math.pi / 2.0
        w = body_to_world(pts, p1, _IDENTITY_PERC)     # z-down: z = -z_up
        back = world_to_level_body(w, p2)
        np.testing.assert_allclose(
            back[:, :2],
            pts[:, :2] @ np.array([[0, 1.0], [-1.0, 0]]).T, atol=1e-4)
        np.testing.assert_allclose(back[:, 2], -pts[:, 2], atol=1e-4)


# ──────────────────────────────────────────────
# 2. ROI 顺序: world_points 完整、非预裁剪、当前 bounds 统一裁剪
# ──────────────────────────────────────────────
class TestRoiOrdering(unittest.TestCase):

    def test_world_first_accepts_full_uncropped_hist(self):
        """world-first: 历史帧世界点超出当前 ROI 不预裁剪, 融合在 BEV 网格化
        时按当前 bounds 裁剪; 越界点不污染融合结果."""
        cur_pts = _make_plane_grid(x_span=3.0, y_span=3.0, n=10, z=2.0, seed=1)
        bounds = _bounds(cur_pts)                      # ~±2.0 m ROI
        # 历史帧: 半格错位的 0.33 m 格网外扩到 ±6 m 的平面 (z=2.05),
        # 含大量 ROI 外点; 错位保证与当前格网不重合, 缺口单元可被补入;
        # 每格 3 点 (小抖动) 满足 min_cell_points=2 的单元级质控
        rng = np.random.default_rng(2)
        x = np.linspace(-6.0, 6.0, 37) + 0.17
        xx, yy = np.meshgrid(x, x)
        pts3 = []
        for _ in range(3):
            pts3.append(np.column_stack([
                (xx + rng.normal(0, 0.01, xx.shape)).ravel(),
                (yy + rng.normal(0, 0.01, xx.shape)).ravel(),
                np.full(xx.size, 2.05, dtype=np.float64) +
                rng.normal(0, 0.02, xx.size)]))
        hist_pts = np.concatenate(pts3).astype(np.float32)
        current_bev = bev_roughness_downsample(cur_pts, bounds, grid_res=64)
        fused = fuse_bev_world_first(current_bev, [hist_pts], bounds, 64,
                                     min_overlap_cells=20,
                                     max_z_correction_m=0.30)
        # 整帧未被拒绝 (对齐网格重叠充分, 修正 ~0.05 < 0.30)
        self.assertEqual(fused.rejected_frames, 0)
        self.assertGreater(fused.added_cells, 0)
        # 输出单元全部在 ROI 内: bev.points 的 x/y 不超过 bounds 范围
        pts_out = fused.bev.points
        self.assertTrue(np.all(np.abs(pts_out[:, 0]) <= bounds["x_max"] + 1e-3))
        self.assertTrue(np.all(np.abs(pts_out[:, 1]) <= bounds["y_max"] + 1e-3))

    def test_current_bounds_crop_happens_at_fusion(self):
        """融合后占用单元与"直接以当前 bounds 网格化完整历史点"一致
        (裁剪发生在网格化时刻, 而非入窗时刻)."""
        cur_pts = _make_plane_grid(x_span=3.0, y_span=3.0, n=10, z=2.0, seed=3)
        bounds = _bounds(cur_pts)
        hist = _make_plane_grid(x_span=12.0, y_span=12.0, n=37, z=2.05,
                                seed=4)
        current_bev = bev_roughness_downsample(cur_pts, bounds, grid_res=64)
        fused = fuse_bev_world_first(current_bev, [hist], bounds, 64,
                                     min_overlap_cells=20,
                                     max_z_correction_m=0.30)
        direct = bev_roughness_downsample(hist, bounds, grid_res=64)
        hist_union = fused.bev.occupied & ~fused.observed_mask
        self.assertTrue(np.all(hist_union <= direct.occupied))

    def test_keyframe_append_skip_and_window_cap(self):
        """关键帧策略: 窗口非空时, 平移/yaw 双低于阈值跳过; 超容量驱逐最旧."""
        from collections import deque
        win = deque(maxlen=3)
        entry = lambda stamp, pose6: WindowEntry(           # noqa: E731
            cloud_stamp=stamp, pose=np.zeros(6, np.float32),
            interp_pose=np.asarray(pose6, dtype=np.float32),
            world_points=np.zeros((0, 3), np.float32),
            source_point_count=0)
        # 首帧入窗 (窗口为空时关键帧判定不生效)
        base = np.zeros(6, np.float32)
        self.assertTrue(window_keyframe_append(
            win, entry(0.0, base), 3, 0.15, 3.0))
        # 0.01 m + 0.57° → 跳过
        e_small = base.copy()
        e_small[:2] = [0.01, 0.0]
        e_small[5] = math.radians(0.57)
        self.assertFalse(window_keyframe_append(
            win, entry(1.0, e_small), 3, 0.15, 3.0))
        self.assertEqual(len(win), 1)
        # 1.0 m → 接受
        e_big = base.copy()
        e_big[0] = 1.0
        self.assertTrue(window_keyframe_append(
            win, entry(2.0, e_big), 3, 0.15, 3.0))
        self.assertEqual(len(win), 2)
        # 6° yaw (平移不足但偏航超限) → 接受
        e_yaw = base.copy()
        e_yaw[5] = math.radians(6.0)
        self.assertTrue(window_keyframe_append(
            win, entry(3.0, e_yaw), 3, 0.15, 3.0))
        self.assertEqual(len(win), 3)
        # 容量 3: 填满后再追加 → 最旧被驱逐
        window_keyframe_append(win, entry(4.0, e_big), 3, 0.15, 3.0)
        self.assertEqual([e.cloud_stamp for e in win], [2.0, 3.0, 4.0])
        # 5.0 帧与 4.0 帧位姿完全相同 → 关键帧跳过, 不入窗
        self.assertFalse(window_keyframe_append(
            win, entry(5.0, e_big), 3, 0.15, 3.0))
        self.assertEqual(len(win), 3)
        self.assertEqual(win[0].cloud_stamp, 2.0)   # 0.0 已被驱逐

    def test_window_evict_age(self):
        """时效驱逐: 融合前移除超过 max_age_s (严格 >) 的旧帧."""
        from collections import deque
        win = deque()
        for i in range(4):
            win.append(WindowEntry(cloud_stamp=100.0 + i,
                                   pose=np.zeros(6, np.float32),
                                   interp_pose=np.zeros(6, np.float32),
                                   world_points=np.zeros((0, 3), np.float32),
                                   source_point_count=0))
        evicted = window_evict_age(win, 106.0, max_age_s=3.0)
        self.assertEqual(evicted, 3)                # 100/101/102 超龄
        self.assertEqual([e.cloud_stamp for e in win], [103.0])


# ──────────────────────────────────────────────
# 3. BEV 分辨率: 64/96/128、cell 日志、上采样不伪造 observed
# ──────────────────────────────────────────────
class TestBevResolution(unittest.TestCase):

    def test_bev_grid_res_from_cell(self):
        """cell-size>0 时按 ROI 物理范围自动算 grid."""
        bounds = roi_bounds(8.0, 8.0)      # 16×16 m
        g = bev_grid_res_from_cell(bounds, 0.25)
        self.assertEqual(g, math.ceil(16.0 / 0.25) + 1)   # 65
        g = bev_grid_res_from_cell(bounds, 1.0)
        self.assertEqual(g, 17)
        self.assertEqual(bev_grid_res_from_cell(bounds, 0.25, min_res=128), 128)

    def test_bev_resolutions_64_96_128(self):
        """64/96/128 网格化稠密云: 占用单元随分辨率单调增长, cell 尺寸写入
        日志 (span/(G-1))."""
        pts = _make_dense_plane(x_span=6.0, y_span=6.0, n=4000, z=2.0, seed=5)
        bounds = _bounds(pts)
        occ, cell = [], []
        for g in (64, 96, 128):
            bev = bev_roughness_downsample(pts, bounds, grid_res=g)
            occ.append(int(bev.occupied.sum()))
            cell.append(bev.stats["cell_x_m"])
        self.assertGreater(occ[1], occ[0])
        self.assertGreater(occ[2], occ[1])
        x_span = 2.0 * bounds["x_max"]
        self.assertAlmostEqual(cell[0], x_span / 63.0, places=6)
        self.assertAlmostEqual(cell[2], x_span / 127.0, places=6)
        self.assertEqual(bev.stats["grid_res"], 128)

    def test_cell_log_from_cell_size(self):
        """--bev-cell-size-m>0: grid 由 cell 反推, 实际 cell 尺寸 ≤ 目标."""
        pts = _make_plane_grid(x_span=5.0, y_span=5.0, n=12, z=2.0, seed=6)
        bounds = _bounds(pts)
        g = bev_grid_res_from_cell(bounds, 0.30)
        bev = bev_roughness_downsample(pts, bounds, grid_res=g)
        self.assertLessEqual(bev.stats["cell_x_m"], 0.30 + 1e-6)

    def test_upsample_never_fabricates_observed(self):
        """上采样: 输出占用 == 最近邻复制的源占用, 严格不新增 observed 单元;
        单元值逐格等于源单元值."""
        rng = np.random.default_rng(8)
        pts = np.column_stack([rng.uniform(-2, 2, size=(800, 2)),
                               np.full(800, 1.0, np.float32)])
        bounds = roi_bounds(2.5, 2.5)
        bev64 = bev_roughness_downsample(pts, bounds, grid_res=64)
        up = bev_upsample_to_model(bev64, 128)
        expected = upsample_grid_nearest(bev64.occupied, 64, 128)
        np.testing.assert_array_equal(up.occupied, expected)
        self.assertEqual(up.stats["grid_res"], 128)
        self.assertEqual(up.stats.get("upsampled_from_grid"), 64)
        for field in ("z_min", "z_max", "z_diff"):
            src = getattr(bev64, field)
            got = getattr(up, field)
            for r, c in zip(*np.nonzero(up.occupied)):
                sr = min(63, round((r + 0.5) * 64 / 128 - 0.5))
                sc = min(63, round((c + 0.5) * 64 / 128 - 0.5))
                self.assertTrue(bev64.occupied[sr, sc],
                                f"cell ({r},{c}) 上采样自未占用源单元")
                if np.isfinite(src[sr, sc]):
                    self.assertEqual(float(got[r, c]), float(src[sr, sc]))
        # 上采样占用不含源未占用单元
        self.assertFalse(np.any(up.occupied & ~expected))


# ──────────────────────────────────────────────
# 4. 融合质量控制
# ──────────────────────────────────────────────
class TestFusionQC(unittest.TestCase):

    def _fuse(self, cur_pts, hist_list, bounds=None, grid=64,
              max_z=0.30, max_cell_res=0.25, min_points=2,
              max_span=0.50, min_overlap=20):
        bounds = bounds or _bounds(cur_pts)
        current_bev = bev_roughness_downsample(cur_pts, bounds, grid_res=grid)
        return fuse_bev_world_first(
            current_bev, hist_list, bounds, grid,
            min_overlap_cells=min_overlap, max_z_correction_m=max_z,
            alignment_cell_m=0.30, max_cell_residual_m=max_cell_res,
            min_cell_points=min_points, max_height_span_m=max_span)

    def test_sparse_bad_cells_only_cell_rejection(self):
        """稀疏坏单元 → 仅单元拒绝: 单帧少数单元高度异常, 整帧不被拒绝;
        - 与当前重叠的坏单元: dup 拒绝 (current 永不被覆盖);
        - 缺口区坏单元 (单点/高跨度): 逐单元拒绝, 不进入融合结果."""
        cur = _make_plane_grid(x_span=4.0, y_span=4.0, n=10, z=2.0, seed=21)
        bounds = _bounds(cur)
        # 历史帧: 与当前同格网 (微偏移 0.05, 通过阶段一), 但 3 个重叠单元
        # 注入 +0.40 m 异常
        hist = cur.copy()
        hist[:3, 2] += 0.40
        fused = self._fuse(cur, [hist], bounds, max_cell_res=0.25)
        self.assertEqual(fused.rejected_frames, 0)          # 整帧不拒绝
        # 重叠坏单元: 被 current 覆盖 → dup 拒绝, 输出高度仍是当前帧值
        for st in fused.frame_stats:
            self.assertEqual(st.rejected, False)
        obs = fused.observed_mask
        cur_bev = bev_roughness_downsample(cur, bounds,
                                           grid_res=fused.bev.grid_res)
        np.testing.assert_array_equal(obs, cur_bev.occupied)
        for (r, c) in zip(*np.nonzero(obs)):
            self.assertEqual(float(fused.bev.z_min[r, c]),
                             float(cur_bev.z_min[r, c]))
        self.assertFalse(np.any(obs & fused.history_fill_mask))
        # 缺口区坏单元: 单点单元 (count=1 < min_cell_points=2) → 逐单元拒绝
        gap_bad = np.array([[-1.6, 1.7, 2.5]], dtype=np.float32)
        hist2 = np.concatenate([hist, gap_bad])
        fused2 = self._fuse(cur, [hist2], bounds, max_cell_res=0.25)
        self.assertEqual(fused2.rejected_frames, 0)
        self.assertGreaterEqual(fused2.rejected_cells, 1)
        # 该坏单元格未被填充 (z_min 为 NaN)
        gb_c = round((gap_bad[0, 0] - bounds["x_min"]) /
                     (bounds["x_max"] - bounds["x_min"]) * 63)
        gb_r = 63 - round((gap_bad[0, 1] - bounds["y_min"]) /
                          (bounds["y_max"] - bounds["y_min"]) * 63)
        self.assertFalse(fused2.bev.occupied[gb_r, gb_c])

    def test_global_z_bias_whole_frame_rejection(self):
        """全局 z 偏差 → 整帧拒绝: 历史帧整体 +0.5 m → |修正|=0.5 > 0.30."""
        cur = _make_plane_grid(x_span=4.0, y_span=4.0, n=10, z=2.0, seed=22)
        hist = cur.copy()
        hist[:, 2] += 0.50
        fused = self._fuse(cur, [hist], max_z=0.30)
        self.assertEqual(fused.rejected_frames, 1)
        self.assertTrue(fused.frame_stats[0].rejected)
        self.assertIn("corr", fused.frame_stats[0].reject_reason)
        self.assertEqual(fused.added_cells, 0)

    def test_current_never_overwritten(self):
        """current 永不被覆盖: 交替单元 +5 m 异常, 融合后 observed 单元
        z_min 与当前帧逐格一致; observed ∩ fill = ∅."""
        cur = _make_plane_grid(x_span=4.0, y_span=4.0, n=10, z=2.0, seed=23)
        hist = cur.copy()
        hist[::2, 2] += 5.0        # 交替单元 +5 m
        fused = self._fuse(cur, [hist])
        cur_bev = bev_roughness_downsample(cur, _bounds(cur),
                                           grid_res=fused.bev.grid_res)
        obs = fused.observed_mask
        np.testing.assert_array_equal(obs, cur_bev.occupied)
        for (r, c) in zip(*np.nonzero(obs)):
            self.assertEqual(float(fused.bev.z_min[r, c]),
                             float(cur_bev.z_min[r, c]))
        self.assertFalse(np.any(obs & fused.history_fill_mask))

    def test_fused_cells_ge_current_cells(self):
        """fused_cells ≥ current_cells; observed ⊆ fused 占用."""
        cur = _make_plane_grid(x_span=4.0, y_span=4.0, n=10, z=2.0, seed=24)
        hist = _make_plane_grid(x_span=4.0, y_span=4.0, n=10, z=2.03,
                                seed=25)
        bounds = _bounds(cur)
        fused = self._fuse(cur, [hist], bounds)
        current_bev = bev_roughness_downsample(cur, bounds, grid_res=64)
        self.assertGreaterEqual(int(fused.bev.occupied.sum()),
                                int(current_bev.occupied.sum()))
        self.assertTrue(np.all(fused.observed_mask <= fused.bev.occupied))

    def test_empty_hist_and_nan_rejection(self):
        """空历史帧 / 含 NaN 历史帧 → 整帧拒绝, 不崩溃."""
        cur = _make_plane_grid(x_span=4.0, y_span=4.0, n=10, z=2.0, seed=26)
        fused = self._fuse(cur, [np.zeros((0, 3), np.float32)])
        self.assertEqual(fused.rejected_frames, 1)
        self.assertEqual(fused.frame_stats[0].reject_reason, "empty")
        fused2 = self._fuse(cur, [np.array([[0, 0, np.nan]], np.float32)])
        self.assertEqual(fused2.rejected_frames, 1)
        self.assertIn("non-finite", fused2.frame_stats[0].reject_reason)

    def test_legacy_overlap_too_small_rejection(self):
        """legacy (fuse_bev_gap_fill): 重叠不足整帧拒绝, 不崩溃."""
        cur = _make_plane_grid(x_span=4.0, y_span=4.0, n=10, z=2.0, seed=27)
        bounds = _bounds(cur)
        hist = _make_plane_grid(x_span=12.0, y_span=12.0, n=24, z=2.05,
                                seed=28)
        current_bev = bev_roughness_downsample(cur, bounds, grid_res=64)
        fused = fuse_bev_gap_fill(current_bev, [hist], bounds, 64,
                                  min_overlap_cells=20,
                                  max_z_correction_m=0.15,
                                  max_residual_m=0.12)
        self.assertEqual(fused.rejected_frames, 1)
        self.assertIn("overlap", fused.frame_stats[0].reject_reason)


# ──────────────────────────────────────────────
# 5. 第 1060 帧诊断
# ──────────────────────────────────────────────
def _bag_available():
    return _BAG.exists() and _CONFIG.exists()


class TestDiagnoseFrame1060(unittest.TestCase):

    @unittest.skipUnless(_bag_available(), "本地无实验 bag, 跳过诊断验收")
    def test_locate_frame_1060_by_seq_and_index(self):
        """--cloud-seq 1557 与 --frame-index 1060 定位同一帧 (单遍扫描)."""
        from diagnose_pointcloud_frame import BagScan
        scan = BagScan(str(_BAG), "/cloud_registered_body",
                       "/mavros/local_position/odom", "/livox/lidar", 100.0)
        idx = scan.scan(target_seq=1557, target_idx=None)
        t = scan.clouds[idx]
        self.assertEqual(t["idx"], 1060)
        self.assertEqual(t["seq"], 1557)
        self.assertAlmostEqual(t["stamp"], 1786091506.690694, delta=1e-3)
        self.assertEqual(len(t["pts"]), 3325)
        # --frame-index 路径 (独立扫描) 同帧
        scan2 = BagScan(str(_BAG), "/cloud_registered_body",
                        "/mavros/local_position/odom", None, 100.0)
        idx2 = scan2.scan(target_seq=None, target_idx=1060)
        self.assertEqual(scan2.clouds[idx2]["seq"], 1557)

    def test_import_does_not_load_torch(self):
        """诊断模块导入不触发 torch (perception/__init__ 被绕过)."""
        code = (
            "import sys; sys.path.insert(0, %r); "
            "import diagnose_pointcloud_frame as d; "
            "assert 'torch' not in sys.modules, sys.modules.keys(); "
            "assert d.body_cloud_to_level_body_roi is not None; "
            "print('NO_TORCH_OK')" % str(_SCRIPTS)
        )
        out = subprocess.run([sys.executable, "-c", code],
                             capture_output=True, text=True)
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertIn("NO_TORCH_OK", out.stdout)

    @unittest.skipUnless(_bag_available(), "本地无实验 bag, 跳过诊断验收")
    def test_diagnose_full_run_outputs_png_json(self):
        """完整诊断运行: 输出 PNG+JSON, JSON 键完整."""
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            cmd = [sys.executable, str(_SCRIPTS / "diagnose_pointcloud_frame.py"),
                   "--bag", str(_BAG), "--config", str(_CONFIG),
                   "--frame-index", "1060", "--cloud-seq", "1557",
                   "--output-dir", tmp]
            proc = subprocess.run(cmd, capture_output=True, text=True,
                                  timeout=900)
            self.assertEqual(proc.returncode, 0, proc.stderr[-3000:])
            idx = 1060
            for fname in (f"frame_{idx}_pointcloud_pipeline.png",
                          f"frame_{idx}_bev_occupancy.png",
                          f"frame_{idx}_fusion_compare.png"):
                self.assertTrue((Path(tmp) / fname).exists(), fname)
            import json
            with open(Path(tmp) / f"frame_{idx}_stats.json") as f:
                stats = json.load(f)
            for key in ("bag_frame_index", "cloud_header_seq", "cloud_stamp",
                        "raw_point_count", "deskewed_point_count",
                        "roi_point_count", "pose_xyz", "pose_rpy_deg",
                        "roi_size_m", "grid_metrics", "fusion_metrics"):
                self.assertIn(key, stats)
            self.assertEqual(stats["bag_frame_index"], 1060)
            self.assertEqual(stats["cloud_header_seq"], 1557)
            self.assertIn("single_roi", stats["grid_metrics"])
            self.assertIn("world_first_30", stats["fusion_metrics"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
