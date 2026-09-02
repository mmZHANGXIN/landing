#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
15 m 锚点地图 + 近十帧窗口 + 连续地形语义重建 (replay_window10 的融合内核)
=========================================================================
本模块只依赖 numpy + (惰性) cv2, 不导入 perception; replay_window10.py
负责 CLI / 触发状态机 / DRL / 可视化编排, 融合与语义算法全部在此:

  AnchorVoxelMap          固定世界坐标稀疏体素图. 每个小体素保存中位高度、
                          近/远分位数、点数、来源帧和时间 (最近 ≤32 个采样,
                          环形缓冲), 构建完成后保持静态 (锚点永不更新);
  world_to_cell_stats     锚点体素 → 当前帧水平机体坐标 BEV 格统计 (表面高度
                          = 体素中位数集合的鲁棒中位, 近/远 = 近远分位数中位);
  fuse_anchor_and_window  近期窗口 + 锚点 → 融合表面. 冲突 (> conflict_m)
                          不做平均: 空间连续的近表面为障碍, 孤立冲突低置信/
                          unknown; 窗口为主 (时效), 锚点补姿态导致的扫描缺口;
  fit_registration_correction  锚点重叠区高度残差平面 → 受限 z/roll/pitch 微
                          校正 (Δz=−c, Δroll=−b, Δpitch=−a); 重叠不足或修正
                          超限时拒绝, 绝不反向污染锚点;
  anchor_geometric_semantic_map  observed/inferred/unknown 三掩码 + 真实观测
                          邻域鲁棒局部平面重建小孔洞 (≤ surface_fill_radius_m,
                          多方向支持) + 锚点地面平面突出检测 (柱体/台阶);
  project_surface_mesh    相邻有效 BEV 单元组成三角面, z-buffer 投影到训练
                          相机; 只有四邻域几何连续且非 unknown 的面才栅格化;
  render_anchor_surface_bgr / render_depth_adaptive_bgr  左/右窗口显示.

坐标约定与 replay_window10 完全一致: 统一世界 W' (ENU 水平 x/y, z-down),
当前帧水平机体坐标 x 前 / y 侧 / z 下 (z-down 为正).

配准符号推导 (小角度, ENU pose 误差 e=(Δz, Δroll=r, Δpitch=p)):
  水平化后机下深度 D(x,y) = H − x·p + y·r; 锚点世界点含构建时误差 e1, 窗口
  含当前误差 e2, 同一帧内 body→world→level-body 往返抵消自身误差, 故重叠
  残差 r_surface = window − anchor = −x·(p2−p1) + y·(r2−r1) + (e1−e2);
  拟合 r = a·x + b·y + c 得 a = p1−p2, b = r2−r1, c = e1−e2;
  修正窗口位姿: Δpitch = −a, Δroll = −b, Δz = −c 使窗口向锚点对齐.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import numpy as np

from replay_compare_common import (
    FusedWindowResult,
    _batch_robust_plane_fit,
    world_to_level_body,
)

logger = logging.getLogger("AnchorFusion")

_VOXEL_CAP = 32          # 每体素环形采样上限 (最近 N 个点足够稳定分位数)
_ANCHOR_MIN_CELLS = 64   # 锚点地面平面拟合所需最少格数


# ──────────────────────────────────────────────
# 锚点体素图
# ──────────────────────────────────────────────
class AnchorVoxelMap:
    """固定世界坐标 (W') 稀疏体素图, 每体素保存统计而非原始点云.

    每体素: 最近 ≤32 个 z-down 采样 (环形缓冲, 直接算中位/近远分位数),
    点数, 首末时间戳, 首末来源帧 + 16 位来源帧掩码. 构建完成后不再接受
    新点 (锚点静态保留到着陆, 时效更新交给近期窗口).
    """

    def __init__(self, voxel_m: float = 0.02):
        self.voxel_m = max(float(voxel_m), 1e-3)
        self._cap = 8192
        self._rows = 0
        self._key_to_row = {}            # 打包体素索引 (int64) → 行号
        self._z_buf = np.zeros((self._cap, _VOXEL_CAP), dtype=np.float32)
        self._count = np.zeros(self._cap, dtype=np.int32)
        self._center = np.zeros((self._cap, 3), dtype=np.float32)
        self._t_first = np.full(self._cap, np.inf, dtype=np.float64)
        self._t_last = np.full(self._cap, -np.inf, dtype=np.float64)
        self._frame_first = np.full(self._cap, -1, dtype=np.int32)
        self._frame_last = np.full(self._cap, -1, dtype=np.int32)
        self._frame_mask = np.zeros(self._cap, dtype=np.int32)

    @property
    def has_data(self) -> bool:
        return self._rows > 0

    @property
    def voxel_count(self) -> int:
        return self._rows

    def _grow(self):
        if self._rows < self._cap:
            return
        new_cap = int(self._cap * 1.5) + 1024
        for attr in ("_z_buf", "_count", "_center", "_t_first", "_t_last",
                     "_frame_first", "_frame_last", "_frame_mask"):
            arr = getattr(self, attr)
            out = np.zeros((new_cap,) + arr.shape[1:], dtype=arr.dtype)
            out[:self._rows] = arr[:self._rows]
            setattr(self, attr, out)
        self._cap = new_cap

    @staticmethod
    def _pack(idx: np.ndarray) -> np.ndarray:
        """体素整数坐标 (N,3) → 唯一 int64 键 (范围 ±163 m @ 0.02 m)."""
        i = idx + np.int64(8192)
        k = 2 * 8192 + 1
        return ((i[:, 0] * k + i[:, 1]) * k + i[:, 2]).astype(np.int64)

    @staticmethod
    def _unpack_center(key, voxel_m: float) -> np.ndarray:
        """打包键 → 体素中心世界坐标 (z-down). 支持标量与数组."""
        k = 2 * 8192 + 1
        ks = np.atleast_1d(np.asarray(key))
        i0, rem = np.divmod(ks, k * k)
        i1, i2 = np.divmod(rem, k)
        c = (np.stack([i0, i1, i2], axis=1).astype(np.float32)
             + 0.5) * voxel_m - 8192 * voxel_m
        return c[0] if np.ndim(key) == 0 else c

    def add(self, pts_world: np.ndarray, stamp: float, frame_id: int) -> None:
        """一帧世界点 (W', z-down) 写入锚点. 每体素只累积统计.

        stamp/frame_id 取该体素批次的 min/max; 来源帧掩码 16 位饱和.
        """
        pts = np.asarray(pts_world, dtype=np.float32)
        if pts.size == 0:
            return
        pts = pts[np.isfinite(pts).all(axis=1)]
        if len(pts) == 0:
            return
        idx = np.floor(pts[:, :3] / self.voxel_m).astype(np.int64)
        keys = self._pack(idx)
        ukeys, inverse = np.unique(keys, return_inverse=True)

        # 新体素分配行号 (Python 循环仅对去重后的键, 每帧 ≤ 数万次)
        rows = np.empty(len(ukeys), dtype=np.int64)
        new_rows = []
        for kk, uk in enumerate(ukeys):
            r = self._key_to_row.get(int(uk))
            if r is None:
                self._grow()
                r = self._rows
                self._rows += 1
                self._key_to_row[int(uk)] = r
                self._center[r] = self._unpack_center(uk, self.voxel_m)
                new_rows.append(r)
            rows[kk] = r

        # 组内环形缓冲散点: 第 k 个追加点落槽 (count_before + k) % CAP,
        # numpy 赋值重复索引取最后一次 → 最新点覆盖最旧点, 环形语义正确.
        order = np.argsort(inverse, kind="stable")
        inv_sorted = inverse[order]
        zs = pts[order, 2].astype(np.float32)
        starts = np.flatnonzero(
            np.concatenate([[True], inv_sorted[1:] != inv_sorted[:-1]]))
        ends = np.concatenate([starts[1:], [len(zs)]])
        n_per = ends - starts
        row_pt = rows[inv_sorted]
        base = (self._count[rows] % _VOXEL_CAP)[inv_sorted]
        offs = np.arange(len(zs), dtype=np.int64) - np.repeat(starts, n_per)
        slots = (base + offs) % _VOXEL_CAP
        self._z_buf[row_pt, slots] = zs
        self._count[rows] += n_per.astype(np.int32)
        np.minimum.at(self._t_first, rows, stamp)
        np.maximum.at(self._t_last, rows, stamp)
        np.minimum.at(self._frame_first, rows, int(frame_id))
        np.maximum.at(self._frame_last, rows, int(frame_id))
        self._frame_mask[rows] |= (1 << (int(frame_id) & 15))

    def _samples(self, rows: np.ndarray) -> np.ndarray:
        """(R, CAP) 有效采样矩阵, 未用槽为 inf (排序后落在尾部)."""
        cnt = self._count[rows]
        n = np.minimum(cnt, _VOXEL_CAP)
        start = (cnt - n) % _VOXEL_CAP
        cols = (start[:, None] + np.arange(_VOXEL_CAP)[None, :]) % _VOXEL_CAP
        vals = self._z_buf[rows[:, None], cols]
        valid = np.arange(_VOXEL_CAP)[None, :] < n[:, None]
        return np.where(valid, vals, np.inf)

    def _row_quantiles(self, rows: np.ndarray):
        """每行 (中位, 近分位 q10, 远分位 q90) — 最近 ≤32 采样排序后取值."""
        n = np.minimum(self._count[rows], _VOXEL_CAP)
        srt = np.sort(self._samples(rows), axis=1)
        med = srt[np.arange(len(rows)), np.clip((n - 1) // 2, 0, _VOXEL_CAP - 1)]
        q10 = srt[np.arange(len(rows)), np.clip(
            np.floor(0.1 * (n - 1)).astype(np.int64), 0, _VOXEL_CAP - 1)]
        q90 = srt[np.arange(len(rows)), np.clip(
            np.ceil(0.9 * (n - 1)).astype(np.int64), 0, _VOXEL_CAP - 1)]
        return med, q10, q90

    def world_to_cell_stats(self, pose: np.ndarray, bounds: dict,
                            grid_res: int, min_down: float,
                            max_down: float) -> "AnchorCellStats | None":
        """锚点体素 → 当前帧水平机体坐标 BEV 格统计.

        体素中心经 world_to_level_body 变换到当前帧水平机体坐标, 按动态 ROI
        裁剪, 再聚合到 (G,G) 格: 表面 = 体素中位数的中位, 近/远 = 近远分位
        数集合的中位 (对单个体素噪声稳健), 点数求和, 来源帧掩码按位或.
        """
        if self._rows == 0:
            return None
        G = int(grid_res)
        centers = np.ascontiguousarray(self._center[:self._rows])
        pts = world_to_level_body(centers, pose)
        keep = ((pts[:, 0] >= bounds["x_min"]) & (pts[:, 0] <= bounds["x_max"])
                & (pts[:, 1] >= bounds["y_min"]) & (pts[:, 1] <= bounds["y_max"])
                & (pts[:, 2] >= min_down) & (pts[:, 2] <= max_down))
        rows = np.flatnonzero(keep)
        if len(rows) == 0:
            return None
        pts = pts[keep]
        x_span = max(bounds["x_max"] - bounds["x_min"], 1e-6)
        y_span = max(bounds["y_max"] - bounds["y_min"], 1e-6)
        col = np.rint((pts[:, 0] - bounds["x_min"]) / x_span * (G - 1)
                      ).astype(np.int64)
        row_un = np.rint((pts[:, 1] - bounds["y_min"]) / y_span * (G - 1)
                         ).astype(np.int64)
        cell = (G - 1 - row_un) * G + col

        med, q10, q90 = self._row_quantiles(rows)
        cnt = self._count[rows].astype(np.float64)
        t_first = self._t_first[rows]
        t_last = self._t_last[rows]
        f_mask = self._frame_mask[rows]

        n_cells = G * G
        surface = _group_median(med.astype(np.float32), cell, n_cells)
        near = _group_median(q10.astype(np.float32), cell, n_cells)
        far = _group_median(q90.astype(np.float32), cell, n_cells)
        count = np.zeros(n_cells, dtype=np.int32)
        np.add.at(count, cell, cnt.astype(np.int32))
        frames = np.zeros(n_cells, dtype=np.int32)
        np.bitwise_or.at(frames, cell, f_mask)
        t_first_out = np.full(n_cells, np.inf, dtype=np.float64)
        t_last_out = np.full(n_cells, -np.inf, dtype=np.float64)
        np.minimum.at(t_first_out, cell, t_first)
        np.maximum.at(t_last_out, cell, t_last)
        mask = count > 0

        return AnchorCellStats(
            surface=surface.reshape(G, G), near=near.reshape(G, G),
            far=far.reshape(G, G), count=count.reshape(G, G),
            frames=frames.reshape(G, G), t_first=t_first_out.reshape(G, G),
            t_last=t_last_out.reshape(G, G), mask=mask.reshape(G, G),
            grid_res=G, bounds=dict(bounds))

    def sample_points(self) -> np.ndarray:
        """全部体素中心 (N,3) 世界坐标, 供 Bayesian 分支作融合点输入."""
        if self._rows == 0:
            return np.empty((0, 3), dtype=np.float32)
        return np.ascontiguousarray(self._center[:self._rows])


def _group_median(values: np.ndarray, keys: np.ndarray,
                  n_out: int) -> np.ndarray:
    """按 keys 分组取中位 (每组首个键的下中位), 输出长度为 n_out 的数组."""
    if len(values) == 0:
        return np.full(n_out, np.nan, dtype=np.float32)
    order = np.lexsort((values, keys))
    vs, ks = values[order], keys[order]
    starts = np.flatnonzero(np.concatenate([[True], ks[1:] != ks[:-1]]))
    ends = np.concatenate([starts[1:], [len(vs)]])
    n = ends - starts
    pos = (n - 1) // 2
    out = np.full(n_out, np.nan, dtype=np.float32)
    out[ks[starts]] = vs[starts + pos]
    return out


@dataclass
class AnchorCellStats:
    """锚点体素聚合到当前帧 BEV 格后的统计 (当前帧水平机体坐标)."""
    surface: np.ndarray        # (G,G) 表面高度 z-down (体素中位数的中位)
    near: np.ndarray           # (G,G) 近分位中位
    far: np.ndarray            # (G,G) 远分位中位
    count: np.ndarray          # (G,G) int32 体素点数合计
    frames: np.ndarray         # (G,G) int32 来源帧掩码 (按位或, 16 位)
    t_first: np.ndarray        # (G,G) float64 最早时间
    t_last: np.ndarray         # (G,G) float64 最晚时间
    mask: np.ndarray           # (G,G) bool 锚点覆盖
    grid_res: int
    bounds: dict


# ──────────────────────────────────────────────
# 每格窗口表面统计 (窗口融合结果 → 格级中位)
# ──────────────────────────────────────────────
def per_cell_median(fused: FusedWindowResult) -> np.ndarray:
    """窗口融合点按格取中位高度 (fused.points 已按 cell_index 排序)."""
    G = int(fused.grid_res)
    out = np.full((G, G), np.nan, dtype=np.float32)
    pts, ci = fused.points, fused.cell_index
    if len(pts) == 0:
        return out
    order = np.lexsort((pts[:, 2], ci))
    cs = ci[order]
    zs = pts[order, 2]
    starts = np.flatnonzero(np.concatenate([[True], cs[1:] != cs[:-1]]))
    ends = np.concatenate([starts[1:], [len(zs)]])
    n = ends - starts
    out.ravel()[cs[starts]] = zs[starts + (n - 1) // 2]
    return out


# ──────────────────────────────────────────────
# 锚点 + 窗口融合
# ──────────────────────────────────────────────
@dataclass
class AnchorWindowResult:
    """锚点 + 近期窗口的融合表面 (当前帧水平机体坐标, z-down).

    表面优先级: 窗口真实观测为主 (时效/分辨率/障碍更新), 锚点补洞; 两者
    同时覆盖且高度差 ≤ conflict_m 时按 0.6/0.4 加权 (窗口更新); 冲突
    (> conflict_m) 不做平均, 表面取近表面 (min), 由语义分支判连续障碍.
    """
    surface: np.ndarray        # (G,G) 融合表面高度
    near: np.ndarray           # (G,G) 最近表面 (突出/障碍检测用)
    far: np.ndarray            # (G,G) 最远表面
    window_surface: np.ndarray # (G,G) 窗口格中位
    window_near: np.ndarray    # (G,G) 窗口最近表面 z_min
    anchor_surface: np.ndarray # (G,G) 锚点表面
    anchor_near: np.ndarray    # (G,G) 锚点近分位
    anchor_far: np.ndarray     # (G,G) 锚点远分位
    observed: np.ndarray       # (G,G) bool 窗口真实观测
    anchor_mask: np.ndarray    # (G,G) bool 锚点覆盖
    conflict: np.ndarray       # (G,G) bool 窗口/锚点冲突 (> conflict_m)
    anchor_count: np.ndarray   # (G,G) int32 锚点体素点数
    anchor_frames: np.ndarray  # (G,G) int32 锚点来源帧掩码
    grid_res: int
    bounds: dict
    stats: dict


def fuse_anchor_and_window(fused: FusedWindowResult,
                           ac: AnchorCellStats,
                           conflict_m: float = 0.15) -> AnchorWindowResult:
    """近期窗口融合结果 + 锚点格统计 → 融合表面 (不做任何位姿平均).

    conflict_m: 窗口与锚点表面高度差超过该值视为冲突, 冲突格表面取近表面
    (min), 由语义分支按空间连续性判障碍或 unknown.
    """
    G = int(fused.grid_res)
    ws = per_cell_median(fused)
    wn = fused.z_min
    wf = fused.z_max
    wv = fused.valid
    asf = ac.surface
    an = ac.near
    af = ac.far
    am = ac.mask

    both = wv & am
    conflict = both & (np.abs(ws - asf) > float(conflict_m))
    agree = both & ~conflict

    surface = np.full((G, G), np.nan, dtype=np.float32)
    near = np.full((G, G), np.nan, dtype=np.float32)
    far = np.full((G, G), np.nan, dtype=np.float32)

    # 一致重叠: 窗口为主 (0.6), 锚点平滑 (0.4); 近/远取两源极值
    surface[agree] = 0.6 * ws[agree] + 0.4 * asf[agree]
    near[agree] = np.minimum(wn[agree], an[agree])
    far[agree] = np.maximum(wf[agree], af[agree])
    # 仅窗口: 原值
    w_only = wv & ~am
    surface[w_only] = ws[w_only]
    near[w_only] = wn[w_only]
    far[w_only] = wf[w_only]
    # 仅锚点: 锚点补洞 (姿态扫描缺口)
    a_only = am & ~wv
    surface[a_only] = asf[a_only]
    near[a_only] = an[a_only]
    far[a_only] = af[a_only]
    # 冲突: 不平均, 表面 = 近表面 (min), 供连续障碍判定
    n_min = np.minimum(wn[conflict], an[conflict])
    surface[conflict] = n_min
    near[conflict] = n_min
    far[conflict] = np.maximum(wf[conflict], af[conflict])

    stats = {
        "window_cells": int(wv.sum()),
        "anchor_cells": int(am.sum()),
        "overlap_cells": int(both.sum()),
        "conflict_cells": int(conflict.sum()),
        "fused_cells": int(np.isfinite(surface).sum()),
    }
    return AnchorWindowResult(
        surface=surface, near=near, far=far,
        window_surface=ws, window_near=wn,
        anchor_surface=asf, anchor_near=an, anchor_far=af,
        observed=wv, anchor_mask=am, conflict=conflict,
        anchor_count=ac.count, anchor_frames=ac.frames,
        grid_res=G, bounds=dict(fused.bounds), stats=stats)


# ──────────────────────────────────────────────
# 受限配准 (锚点重叠区残差平面 → z/roll/pitch 微校正)
# ──────────────────────────────────────────────
@dataclass
class RegistrationResult:
    accepted: bool
    delta: np.ndarray          # (3,) [Δz, Δroll, Δpitch] 深度域平面校正
    resid_before: float        # 校正前重叠残差 (median |r|, m)
    resid_after: float         # 校正后重叠残差 (由调用方重融合后回填)
    overlap_cells: int
    reason: str
    plane: tuple | None        # (a, b, c) 残差平面参数


def fit_registration_correction(fused: FusedWindowResult,
                                ac: AnchorCellStats,
                                z_lim_m: float = 0.20,
                                ang_lim_deg: float = 2.0,
                                min_overlap_frac: float = 0.10,
                                min_overlap_cells: int = 32):
    """锚点重叠区高度残差平面 → 受限 z/roll/pitch 微校正.

    残差 r(x,y) = 窗口表面 − 锚点表面 拟合 r = a·x + b·y + c (Tukey
    bisquare IRLS, 与语义分支同内核). 校正量 Δz = −c, Δroll = −b,
    Δpitch = −a, 由调用方以深度域平面 Δz + Δpitch·x + Δroll·y 修正窗口
    融合深度 (W' 已水平化, world_to_level_body 仅 yaw+平移, 位姿 roll/pitch
    不影响机下深度, 校正必须落在深度域而非位姿旋转);
    |Δz| > z_lim_m 或倾斜角 > ang_lim_deg 时拒绝; 重叠格数/比例不足时拒绝.
    拒绝只返回诊断, 不修改任何数据或锚点.
    """
    G = int(fused.grid_res)
    ws = per_cell_median(fused)
    overlap = (fused.valid & ac.mask
               & np.isfinite(ws) & np.isfinite(ac.surface))
    n_ov = int(overlap.sum())
    if n_ov < int(min_overlap_cells):
        return RegistrationResult(False, np.zeros(3, dtype=np.float64),
                                  float("nan"), float("nan"), n_ov,
                                  f"overlap_cells={n_ov}<{int(min_overlap_cells)}",
                                  None)
    frac = n_ov / max(int(fused.valid.sum()), 1)
    if frac < float(min_overlap_frac):
        return RegistrationResult(False, np.zeros(3, dtype=np.float64),
                                  float("nan"), float("nan"), n_ov,
                                  f"overlap_frac={frac:.3f}<{min_overlap_frac:.2f}",
                                  None)
    rows, cols = np.where(overlap)
    x_span = max(ac.bounds["x_max"] - ac.bounds["x_min"], 1e-6)
    y_span = max(ac.bounds["y_max"] - ac.bounds["y_min"], 1e-6)
    ox = (ac.bounds["x_min"] + cols / max(G - 1, 1) * x_span).astype(np.float32)
    oy = (ac.bounds["y_max"] - rows / max(G - 1, 1) * y_span).astype(np.float32)
    r = (ws[rows, cols] - ac.surface[rows, cols]).astype(np.float32)
    params, _, _ = _batch_robust_plane_fit(
        r[None, :], np.ones((1, n_ov), dtype=bool), ox, oy,
        min_pts=int(min_overlap_cells))
    a, b, c = (float(params[0, 0]), float(params[0, 1]), float(params[0, 2]))
    resid_before = float(np.median(np.abs(r)))

    delta = np.array([-c, -b, -a], dtype=np.float64)   # [Δz, Δroll, Δpitch]
    tilt = np.degrees(np.arctan(np.hypot(a, b)))
    if abs(c) > float(z_lim_m):
        return RegistrationResult(False, delta, resid_before, float("nan"),
                                  n_ov,
                                  f"z_corr={-c:+.3f}m 超限 (> {z_lim_m} m)",
                                  (a, b, c))
    if tilt > float(ang_lim_deg):
        return RegistrationResult(False, delta, resid_before, float("nan"),
                                  n_ov,
                                  f"tilt_corr={tilt:.2f}° 超限 (> {ang_lim_deg}°)",
                                  (a, b, c))
    return RegistrationResult(True, delta, resid_before, float("nan"), n_ov,
                              "ok", (a, b, c))


def overlap_residual(fused: FusedWindowResult, ac: AnchorCellStats,
                     surface: np.ndarray | None = None) -> float:
    """重叠区 median |窗口 − 锚点| 残差 (m); 无重叠返回 NaN."""
    ws = per_cell_median(fused) if surface is None else surface
    ov = (fused.valid & ac.mask & np.isfinite(ws) & np.isfinite(ac.surface))
    if not ov.any():
        return float("nan")
    return float(np.median(np.abs(ws[ov] - ac.surface[ov])))


# ──────────────────────────────────────────────
# 连续地形语义 (observed / inferred / unknown)
# ──────────────────────────────────────────────
def anchor_geometric_semantic_map(
        aw: AnchorWindowResult,
        slope_threshold_deg: float,
        roughness_threshold_m: float,
        prominence_threshold_m: float = 0.15,
        safe_id: int = 1, danger_id: int = 9,
        fine_radius_cells: int = 3,
        coarse_radius_cells: int = 15,
        coarse_stride: int = 2,
        min_support_pts: int = 8,
        surface_fill_radius_m: float = 0.25):
    """融合表面 → 连续地形语义.

    - observed_mask: 窗口真实观测格;
    - inferred_mask: 真实观测邻域内小孔洞 (距观测 ≤ surface_fill_radius_m,
      且 ≥2 个方向有观测支持) 由鲁棒局部平面重建; 大缺口不推断;
    - unknown_mask: 窗口与锚点均未覆盖 (绝不通过凸包最近邻填满);
    - 坡度/粗糙度由连续表面 (观测∪锚点∪推断) 局部平面拟合;
    - 突出高度相对锚点地面平面 (锚点覆盖不足时退回局部粗平面): 柱体/台阶
      需空间连续支持 (3×3 邻域 ≥2 格) 才标危险;
    - 冲突格: 空间连续的近表面为障碍, 孤立冲突标 unknown.

    Returns (sem_map, semantic_valid_mask, maps):
      maps = {slope_deg, roughness, prominence, rel_height, surface, near,
              far, observed_mask, inferred_mask, unknown_mask, anchor_mask,
              conflict_mask, anchor_plane, confidence, valid, semantic_valid}
    """
    import cv2

    G = int(aw.grid_res)
    bounds = aw.bounds
    x_span = max(bounds["x_max"] - bounds["x_min"], 1e-6)
    y_span = max(bounds["y_max"] - bounds["y_min"], 1e-6)
    cell_x = x_span / max(G - 1, 1)
    cell_y = y_span / max(G - 1, 1)

    def _empty():
        return np.full((G, G), np.nan, dtype=np.float32)

    observed = aw.observed.copy()
    conflict = aw.conflict.copy()
    anchor_mask = aw.anchor_mask.copy()
    surface = aw.surface.copy()
    valid_surf = np.isfinite(surface)
    # unknown 在孔洞推断之后重算: 被推断重建的格不再是 unknown

    # ── 细窗口 IRLS: 全有效表面 (观测∪锚点) 的局部平面 ──
    rows, cols, hw, vw, ox, oy = _window_fit(
        surface, valid_surf, int(fine_radius_cells), 1, G, bounds, cell_x, cell_y)
    params_fine, rough_fine, support_fine = _batch_robust_plane_fit(
        hw, vw, ox, oy, int(min_support_pts))
    slope_out, rough_out = _empty(), _empty()
    if len(rows):
        slope_out[rows, cols] = np.degrees(np.arctan(
            np.hypot(params_fine[:, 0], params_fine[:, 1])))
        rough_out[rows, cols] = rough_fine

    # ── 小孔洞推断 (仅观测邻域, 多方向支持, ≤ surface_fill_radius_m) ──
    inferred = np.zeros((G, G), dtype=bool)
    if observed.any():
        r_cells = max(1, int(round(float(surface_fill_radius_m) / cell_x)))
        dist, labels = cv2.distanceTransformWithLabels(
            (~observed).astype(np.uint8), cv2.DIST_L2, 5,
            cv2.DIST_LABEL_PIXEL)
        cand = ((~observed) & (~anchor_mask) & (dist > 0)
                & (dist <= r_cells))
        n_dir = _quadrant_support(observed, cand, r_cells)
        cand &= n_dir >= 2
        cand_r, cand_c = np.where(cand)
        if len(cand_r):
            # labels[observed] 给出每观测格所属连通分量 → 分量首个格
            flat_idx = np.flatnonzero(observed)
            comp_first = np.full(int(labels.max()) + 1, -1, dtype=np.int64)
            comp_first[labels.ravel()[flat_idx]] = flat_idx
            obs_flat = comp_first[labels[cand_r, cand_c]]
            obs_r, obs_c = np.divmod(obs_flat, G)
            # 平面参数按格查找 (拟合覆盖全部有效格)
            pf_full = np.zeros((G, G, 3), dtype=np.float32)
            pf_full[rows, cols] = params_fine
            a, b, c = (pf_full[obs_r, obs_c, 0], pf_full[obs_r, obs_c, 1],
                       pf_full[obs_r, obs_c, 2])
            xo = bounds["x_min"] + obs_c * cell_x
            yo = bounds["y_max"] - obs_r * cell_y
            xc = bounds["x_min"] + cand_c * cell_x
            yc = bounds["y_max"] - cand_r * cell_y
            inferred_z = (a * (xc - xo) + b * (yc - yo) + c).astype(np.float32)
            inferred[cand_r, cand_c] = True
            surface[cand_r, cand_c] = inferred_z
            valid_surf |= inferred

    # unknown = 推断之后仍无表面 (真实观测/锚点/推断都不覆盖)
    unknown = ~valid_surf

    # ── 粗窗口 IRLS (锚点地面平面不足时的相对高度退回基准) ──
    rows_c, cols_c, hw_c, vw_c, ox_c, oy_c = _window_fit(
        surface, valid_surf, int(coarse_radius_cells), int(coarse_stride),
        G, bounds, cell_x, cell_y)
    params_coarse, _, support_coarse = _batch_robust_plane_fit(
        hw_c, vw_c, ox_c, oy_c, int(min_support_pts))
    support_f = np.zeros((G, G), dtype=bool)
    support_c = np.zeros((G, G), dtype=bool)
    if len(rows):
        support_f[rows, cols] = support_fine
        support_c[rows_c, cols_c] = support_coarse

    # ── 锚点地面平面 (全局 IRLS, 对柱体/障碍稳健) ──
    anchor_plane_ok = False
    plane_a = plane_b = plane_c = 0.0
    an_cells = anchor_mask & valid_surf
    if int(an_cells.sum()) >= _ANCHOR_MIN_CELLS:
        rows_a, cols_a = np.where(an_cells)
        ox_a = (bounds["x_min"] + cols_a * cell_x).astype(np.float32)
        oy_a = (bounds["y_max"] - rows_a * cell_y).astype(np.float32)
        p, _, _ = _batch_robust_plane_fit(
            aw.anchor_surface[rows_a, cols_a].astype(np.float32)[None, :],
            np.ones((1, len(rows_a)), dtype=bool), ox_a, oy_a,
            min_pts=int(min(_ANCHOR_MIN_CELLS, len(rows_a))))
        plane_a, plane_b, plane_c = (float(p[0, 0]), float(p[0, 1]),
                                     float(p[0, 2]))
        anchor_plane_ok = True

    # ── 相对高度与突出 ──
    rel_out = _empty()
    valid_rel = valid_surf & np.isfinite(aw.near)
    if valid_rel.any():
        if anchor_plane_ok:
            ox_g = (bounds["x_min"]
                    + np.arange(G, dtype=np.float32) * cell_x)
            oy_g = (bounds["y_max"]
                    - np.arange(G, dtype=np.float32) * cell_y)
            plane_z = (plane_a * ox_g[None, :] + plane_b * oy_g[:, None]
                       + plane_c).astype(np.float32)
            rel_out[valid_rel] = (plane_z[valid_rel] - aw.near[valid_rel])
        else:
            z_plane_c = np.full((G, G), np.nan, dtype=np.float32)
            if len(rows_c):
                z_plane_c[rows_c, cols_c] = params_coarse[:, 2]
            rel_out[valid_rel] = (z_plane_c[valid_rel] - aw.near[valid_rel])
    prom_out = np.maximum(rel_out, 0.0)

    prom_above = (prom_out > float(prominence_threshold_m)) & valid_rel
    neigh_prom = cv2.filter2D(prom_above.astype(np.float32), -1,
                              np.ones((3, 3), np.float32))
    prom_danger = prom_above & (neigh_prom >= 2.0)
    neigh_conf = cv2.filter2D(conflict.astype(np.float32), -1,
                              np.ones((3, 3), np.float32))
    conflict_obstacle = conflict & (neigh_conf >= 2.0)
    isolated_conflict = conflict & ~conflict_obstacle

    # ── 分类 ──
    slope_ok = np.full((G, G), False)
    rough_ok = np.full((G, G), False)
    slope_ok[rows, cols] = (slope_out[rows, cols] < float(slope_threshold_deg))
    rough_ok[rows, cols] = (rough_out[rows, cols] < float(roughness_threshold_m))
    supported = support_f & support_c
    # 所有冲突 (连续=障碍, 孤立=低置信/unknown) 都不允许升级为安全地面:
    # 连续冲突已由 conflict_obstacle 排除, 孤立冲突在此一并排除 (spec:
    # 孤立冲突低置信/unknown, 不能按安全平面解释)
    safe = (valid_surf & supported & slope_ok & rough_ok
            & ~prom_danger & ~conflict_obstacle & ~conflict)
    sem_map = np.full((G, G), int(danger_id), dtype=np.uint8)
    sem_map[safe] = int(safe_id)
    # 有表面但局部支持不足 → unknown (不因默认 danger_id 显示为黑色散点)
    semantic_valid = valid_surf & supported & ~isolated_conflict

    confidence = np.full((G, G), np.nan, dtype=np.float32)
    confidence[observed] = 1.0
    confidence[anchor_mask & ~observed & valid_surf] = 0.85
    confidence[inferred] = 0.5

    maps = {
        "slope_deg": slope_out,
        "roughness": rough_out,
        "prominence": prom_out,
        "rel_height": rel_out,
        "surface": surface,
        "near": aw.near,
        "far": aw.far,
        "observed_mask": observed,
        "inferred_mask": inferred,
        "unknown_mask": unknown,
        "anchor_mask": anchor_mask,
        "conflict_mask": conflict,
        "anchor_plane": (plane_a, plane_b, plane_c) if anchor_plane_ok else None,
        "confidence": confidence,
        "valid": valid_surf,
        "semantic_valid": semantic_valid,
    }
    return sem_map, semantic_valid, maps


def _window_fit(surface: np.ndarray, valid: np.ndarray, radius: int,
                stride: int, G: int, bounds: dict, cell_x: float,
                cell_y: float):
    """对全部有效格批量提取窗口观测 (与 replay_compare_common 同构).

    Returns (rows, cols, hw, vw, ox, oy): hw/vw (N, K), ox/oy (K,) 为
    窗口内相对本格中心的偏移 (x 沿列 / y 沿行, meshgrid "ij" 顺序与
    _batch_robust_plane_fit 的 X=[ox, oy, 1] 对齐).
    """
    R = int(radius)
    K = ((2 * R) // stride + 1) ** 2
    padded_h = np.pad(surface, R, mode="constant", constant_values=np.nan)
    padded_v = np.pad(valid, R, mode="constant", constant_values=False)
    view_h = np.lib.stride_tricks.sliding_window_view(
        padded_h, (2 * R + 1, 2 * R + 1))
    view_v = np.lib.stride_tricks.sliding_window_view(
        padded_v.astype(np.uint8), (2 * R + 1, 2 * R + 1)).astype(bool)
    rows, cols = np.where(valid)
    hw = np.ascontiguousarray(view_h[rows, cols][:, ::stride, ::stride]
                              ).reshape(len(rows), K)
    vw = np.ascontiguousarray(view_v[rows, cols][:, ::stride, ::stride]
                              ).reshape(len(rows), K)
    rng = np.arange(2 * R + 1) - R
    ox = (rng[::stride] * cell_x).astype(np.float32)
    oy = (rng[::stride] * cell_y).astype(np.float32)
    oxy = np.array(np.meshgrid(oy, ox, indexing="ij")).reshape(2, -1)
    return rows, cols, hw, vw, oxy[1], oxy[0]


def _quadrant_support(observed: np.ndarray, cand: np.ndarray,
                      r_cells: int) -> np.ndarray:
    """候选孔洞四象限观测支持计数 (0..4, 边界自动裁剪).

    以候选格为中心的 (2R+1)² 邻域按 NE/NW/SE/SW 四象限统计观测格数,
    用于「多方向支持」判定 (≥2 方向).
    """
    G = observed.shape[0]
    pad = np.zeros((G + 1, G + 1), dtype=np.int32)
    pad[1:, 1:] = observed.astype(np.int32)
    cs = np.cumsum(np.cumsum(pad, axis=0), axis=1)

    def box(r0, r1, c0, c1):
        r0 = np.clip(r0, 0, G)
        r1 = np.clip(r1, 0, G)
        c0 = np.clip(c0, 0, G)
        c1 = np.clip(c1, 0, G)
        w = (r1 > r0) & (c1 > c0)
        out = np.zeros_like(r0, dtype=np.int32)
        out[w] = (cs[r1[w], c1[w]] - cs[r0[w], c1[w]]
                  - cs[r1[w], c0[w]] + cs[r0[w], c0[w]])
        return out

    r, c = np.where(cand)
    R = int(r_cells)
    q_ne = box(r - R, r, c + 1, c + R + 1)
    q_nw = box(r - R, r, c - R, c)
    q_se = box(r + 1, r + R + 1, c + 1, c + R + 1)
    q_sw = box(r + 1, r + R + 1, c - R, c)
    n_dir = np.zeros(observed.shape, dtype=np.int32)
    n_dir[r, c] = ((q_ne > 0).astype(np.int32) + (q_nw > 0).astype(np.int32)
                   + (q_se > 0).astype(np.int32) + (q_sw > 0).astype(np.int32))
    return n_dir


class AnchorSemanticBranch:
    """锚点+窗口融合几何语义分支 (观测/推断/unknown + 锚点地面突出检测).

    不调用 HALSS Bayesian; 输入为 fuse_anchor_and_window 的融合表面.
    """

    def __init__(self, slope_threshold_deg: float, roughness_threshold_m: float,
                 prominence_threshold_m: float = 0.15,
                 safe_id: int = 1, danger_id: int = 9,
                 fine_radius_cells: int = 3, coarse_radius_cells: int = 15,
                 min_support_pts: int = 8, surface_fill_radius_m: float = 0.25):
        self.slope_th = float(slope_threshold_deg)
        self.rough_th = float(roughness_threshold_m)
        self.prom_th = float(prominence_threshold_m)
        self.safe_id = int(safe_id)
        self.danger_id = int(danger_id)
        self.fine_radius = int(fine_radius_cells)
        self.coarse_radius = int(coarse_radius_cells)
        self.min_support = int(min_support_pts)
        self.fill_radius = float(surface_fill_radius_m)

    def __call__(self, aw: AnchorWindowResult):
        t0 = time.perf_counter()
        sem_map, semantic_valid, maps = anchor_geometric_semantic_map(
            aw, self.slope_th, self.rough_th,
            prominence_threshold_m=self.prom_th,
            safe_id=self.safe_id, danger_id=self.danger_id,
            fine_radius_cells=self.fine_radius,
            coarse_radius_cells=self.coarse_radius,
            min_support_pts=self.min_support,
            surface_fill_radius_m=self.fill_radius)

        def _ratio(key):
            return float(np.mean(maps[key]))

        return sem_map, {
            "branch": "anchor_geometry",
            "latency_ms": (time.perf_counter() - t0) * 1000.0,
            "obs_ratio": _ratio("observed_mask"),
            "inferred_ratio": _ratio("inferred_mask"),
            "unknown_ratio": _ratio("unknown_mask"),
            "anchor_ratio": _ratio("anchor_mask"),
            "conflict_cells": int(maps["conflict_mask"].sum()),
            "sem_valid_ratio": float(np.mean(semantic_valid)),
            "safe_ratio": float(np.mean(sem_map == self.safe_id)),
            "maps": maps,
        }


# ──────────────────────────────────────────────
# 融合表面网格投影 (连续地形 → training-camera z-buffer)
# ──────────────────────────────────────────────
def project_surface_mesh(surface: np.ndarray, valid: np.ndarray,
                         sem_map: np.ndarray, sem_valid: np.ndarray,
                         bounds: dict, camera, max_step_m: float = 0.12,
                         danger_id: int = 9):
    """相邻有效 BEV 单元组成三角面, z-buffer 投影到训练相机.

    仅当四邻域全部: 真实/锚点/推断覆盖 (valid)、几何连续
    (|Δz| ≤ max_step_m, 四边)、且语义非 unknown (sem_valid) 时栅格化;
    大缺口、unknown 与障碍边缘 (柱体台阶) 不跨越生成三角面.

    深度 = 像素内三角形重心插值; 深度通道与原始融合点 z-buffer 取 min
    (由调用方合并); 网格像素语义 = 反投影地面点所在 BEV 格标签.

    Returns (mesh_depth (h,w), mesh_valid, mesh_sem (h,w), mesh_sem_valid).
    mesh_depth 未命中处为 inf, 便于与点投影取 min.
    """
    h, w = camera.output_height, camera.output_width
    depth = np.full((h, w), np.inf, dtype=np.float32)
    vmask = np.zeros((h, w), dtype=bool)
    labels = np.full((h, w), int(danger_id), dtype=np.uint8)
    lvalid = np.zeros((h, w), dtype=bool)
    G = int(surface.shape[0])
    if G < 2 or not valid.any():
        return depth, vmask, labels, lvalid

    x_span = max(bounds["x_max"] - bounds["x_min"], 1e-6)
    y_span = max(bounds["y_max"] - bounds["y_min"], 1e-6)
    cell_x = x_span / max(G - 1, 1)
    cell_y = y_span / max(G - 1, 1)
    s = np.asarray(surface, dtype=np.float32)
    v = np.asarray(valid, dtype=bool)
    sv = np.asarray(sem_valid, dtype=bool)

    # 2×2 四格四边形: 全部有效 + 非 unknown + 四边几何连续
    d_r = np.abs(s[1:, :-1] - s[:-1, :-1])
    d_d = np.abs(s[:-1, 1:] - s[:-1, :-1])
    quad = (v[:-1, :-1] & v[1:, :-1] & v[:-1, 1:] & v[1:, 1:]
            & sv[:-1, :-1] & sv[1:, :-1] & sv[:-1, 1:] & sv[1:, 1:]
            & (d_r <= float(max_step_m)) & (d_d <= float(max_step_m)))
    rows, cols = np.where(quad)
    N = len(rows)
    if N == 0:
        return depth, vmask, labels, lvalid

    # 四角 (格中心): c00=(r,c), c01=(r,c+1), c11=(r+1,c+1), c10=(r+1,c)
    xs = (bounds["x_min"] + (cols[:, None]
                             + np.array([0.0, 0.0, 1.0, 1.0])[None, :])
          * cell_x).astype(np.float32)
    ys = (bounds["y_max"] - (rows[:, None]
                             + np.array([0.0, 1.0, 1.0, 0.0])[None, :])
          * cell_y).astype(np.float32)
    zs = np.stack([s[rows, cols], s[rows, cols + 1],
                   s[rows + 1, cols + 1], s[rows + 1, cols]], axis=1)
    xc = -ys
    yc = -xs
    u = camera.fx * xc / zs + camera.cx
    vv = camera.fy * yc / zs + camera.cy

    ok = (zs > camera.near_m).all(axis=1)
    u, vv, zs = u[ok], vv[ok], zs[ok]
    N = u.shape[0]
    if N == 0:
        return depth, vmask, labels, lvalid

    c0 = np.clip(np.floor(u.min(axis=1)).astype(np.int64), 0, w - 1)
    c1 = np.clip(np.floor(u.max(axis=1)).astype(np.int64), 0, w - 1)
    r0 = np.clip(np.floor(vv.min(axis=1)).astype(np.int64), 0, h - 1)
    r1 = np.clip(np.floor(vv.max(axis=1)).astype(np.int64), 0, h - 1)
    vis = (c1 >= c0) & (r1 >= r0)
    if not vis.any():
        return depth, vmask, labels, lvalid
    u, vv, zs, c0, c1, r0, r1 = (u[vis], vv[vis], zs[vis], c0[vis],
                                 c1[vis], r1[vis], r0[vis])
    N = u.shape[0]

    ncols = (c1 - c0 + 1).astype(np.int64)
    areas = ncols * (r1 - r0 + 1)
    P = int(areas.max())
    P = min(P, 4096)                      # 超大 bbox 四边形跳过 (近相机边界)
    if P <= 0:
        return depth, vmask, labels, lvalid
    keep_q = areas <= P
    u, vv, zs, c0, c1, r0, r1, ncols = (u[keep_q], vv[keep_q], zs[keep_q],
                                         c0[keep_q], c1[keep_q], r0[keep_q],
                                         r1[keep_q], ncols[keep_q])
    N = u.shape[0]
    if N == 0:
        return depth, vmask, labels, lvalid
    areas = ncols * (r1 - r0 + 1)

    # 每四边形填充像素网格 (填充到最大 bbox 面积)
    p_idx = np.arange(P, dtype=np.int64)[None, :]
    col_off = p_idx % ncols[:, None]
    row_off = p_idx // ncols[:, None]
    valid_p = p_idx < areas[:, None]
    U = np.clip(c0[:, None] + col_off, 0, w - 1)
    V = np.clip(r0[:, None] + row_off, 0, h - 1)

    # 三角形化: T1=(0,1,2), T2=(2,3,0); 像素空间绕向统一为 CCW
    corners = np.stack([u, vv], axis=2)                 # (N,4,2)
    s1 = (corners[:, 1, 0] - corners[:, 0, 0]) * (corners[:, 2, 1] - corners[:, 0, 1]) \
        - (corners[:, 1, 1] - corners[:, 0, 1]) * (corners[:, 2, 0] - corners[:, 0, 0])
    flip = s1 < 0                                       # 像素绕向相反 → 交换
    tri = np.stack([corners[:, 0], corners[:, 1], corners[:, 2],
                    corners[:, 0], corners[:, 3], corners[:, 2]],
                   axis=1)                              # (N,6,2): T1,T2 顶点
    z_tri = np.stack([zs[:, 0], zs[:, 1], zs[:, 2],
                      zs[:, 0], zs[:, 3], zs[:, 2]], axis=1)  # (N,6)
    if flip.any():
        tri[flip, 1], tri[flip, 2] = tri[flip, 2].copy(), tri[flip, 1].copy()
        z_tri[flip, 1], z_tri[flip, 2] = (z_tri[flip, 2].copy(),
                                          z_tri[flip, 1].copy())
        tri[flip, 4], tri[flip, 5] = tri[flip, 5].copy(), tri[flip, 4].copy()
        z_tri[flip, 4], z_tri[flip, 5] = (z_tri[flip, 5].copy(),
                                          z_tri[flip, 4].copy())

    def _rasterize(px_u, px_v):
        """批量像素 (N,K) → (hit (N,K) bool, z (N,K) float32).

        三角形化已统一像素绕向; 有向面积 d≈0 (退化) 的三角形直接判未命中.
        """
        px = np.stack([px_u, px_v], axis=2).astype(np.float64)   # (N,K,2)
        hit = np.zeros(px_u.shape, dtype=bool)
        zout = np.zeros(px_u.shape, dtype=np.float32)
        for t_idx in (0, 1):
            va, vb, vc = (tri[:, t_idx * 3 + 0], tri[:, t_idx * 3 + 1],
                          tri[:, t_idx * 3 + 2])
            e1 = vb - va
            e2 = vc - va
            d = e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0]      # 2×area, 有向
            q = px - va[:, None, :]
            lam1 = (q[:, :, 0] * e2[:, 1, None]
                    - q[:, :, 1] * e2[:, 0, None]) / d[:, None]
            lam2 = (e1[:, 0, None] * q[:, :, 1]
                    - e1[:, 1, None] * q[:, :, 0]) / d[:, None]
            lam0 = 1.0 - lam1 - lam2
            h = (np.isfinite(d[:, None]) & (lam0 >= -1e-3) & (lam1 >= -1e-3)
                 & (lam2 >= -1e-3))
            zi = (lam0 * z_tri[:, t_idx * 3 + 0][:, None]
                  + lam1 * z_tri[:, t_idx * 3 + 1][:, None]
                  + lam2 * z_tri[:, t_idx * 3 + 2][:, None])
            zout = np.where(h, np.where(~hit, zi, zout), zout)
            hit |= h
        return hit, zout

    inside, z_pix = _rasterize(U, V)         # bbox 网格采样 (大四边形)
    if inside.any():
        flat = (V * w + U).astype(np.int64)
        np.minimum.at(depth.ravel(), flat[inside], z_pix[inside])
    # 亚像素四边形兜底: 质心像素必在四边形内部 (低空格在 3 m 深处仅 ~0.8 px,
    # bbox 角点采样可能全部落空 → 无任何像素命中)
    cent_u = np.clip(np.floor(u.mean(axis=1)).astype(np.int64), 0, w - 1)
    cent_v = np.clip(np.floor(vv.mean(axis=1)).astype(np.int64), 0, h - 1)
    hit_c, z_c = _rasterize(cent_u[:, None], cent_v[:, None])
    if hit_c[:, 0].any():
        flat_c = (cent_v * w + cent_u).astype(np.int64)
        np.minimum.at(depth.ravel(), flat_c[hit_c[:, 0]],
                      z_c[hit_c[:, 0], 0])
    vmask = np.isfinite(depth)

    # 语义: 反投影命中像素 → BEV 格标签 (仅语义有效格, 其余 danger)
    if vmask.any():
        z_px = depth.astype(np.float64)
        # 只对有限深度像素反投影 (inf 深度参与 rint 会产生 invalid 值)
        rr_f, cc_f = np.where(vmask)
        zf = z_px[rr_f, cc_f]
        x_cam = (cc_f.astype(np.float64) - camera.cx) * zf \
            / max(float(camera.fx), 1e-9)
        y_cam = (rr_f.astype(np.float64) - camera.cy) * zf \
            / max(float(camera.fy), 1e-9)
        x_body = -y_cam
        y_body = -x_cam
        col = np.rint((x_body - bounds["x_min"]) / x_span * (G - 1)
                      ).astype(np.int64)
        row_un = np.rint((y_body - bounds["y_min"]) / y_span * (G - 1)
                         ).astype(np.int64)
        row = (G - 1) - row_un
        inb = ((row >= 0) & (row < G) & (col >= 0) & (col < G))
        labels[rr_f[inb], cc_f[inb]] = sem_map[row[inb], col[inb]]
        lvalid[rr_f[inb], cc_f[inb]] = sem_valid[row[inb], col[inb]]
    return depth, vmask, labels, lvalid


# ──────────────────────────────────────────────
# 显示
# ──────────────────────────────────────────────
def render_anchor_surface_bgr(maps: dict, half_range_m: float = 0.3,
                              safe_tol_m: float = 0.02) -> np.ndarray:
    """左窗口: 连续相对锚点地面高度着色.

    安全平面 (|rel| ≤ safe_tol) 绿、突出 (柱体/台阶) 红、低洼蓝, 亮度随
    |rel| 增强; 真实观测 (observed) 饱和色, 锚点补洞 (anchor-only) 暗绿,
    小孔洞推断 (inferred) 青色; unknown 灰色 (与黑底区分). 纯 numpy.
    """
    rel = np.asarray(maps["rel_height"], dtype=np.float32)
    valid = np.asarray(maps["valid"], dtype=bool) & np.isfinite(rel)
    half = max(float(half_range_m), 1e-3)
    tol = max(float(safe_tol_m), 1e-4)
    img = np.zeros(rel.shape + (3,), dtype=np.uint8)
    img[~valid] = (96, 96, 96)                     # unknown → 灰

    t = np.clip(rel[valid] / half, -1.0, 1.0)
    above = np.clip((t - tol / half) / (1.0 - tol / half), 0.0, 1.0)
    below = np.clip((-t - tol / half) / (1.0 - tol / half), 0.0, 1.0)
    flat = 1.0 - above - below
    red = np.array([0.0, 30.0, 230.0], dtype=np.float32)
    blue = np.array([220.0, 60.0, 20.0], dtype=np.float32)

    obs = valid & np.asarray(maps["observed_mask"], dtype=bool)
    inf = valid & np.asarray(maps["inferred_mask"], dtype=bool)
    aonly = (valid & np.asarray(maps["anchor_mask"], dtype=bool)
             & ~np.asarray(maps["observed_mask"], dtype=bool))
    for sel, base in ((obs, np.array([60.0, 205.0, 70.0], np.float32)),
                      (inf, np.array([190.0, 205.0, 60.0], np.float32)),
                      (aonly, np.array([45.0, 130.0, 60.0], np.float32))):
        m = sel[valid]
        if not m.any():
            continue
        col = (base[None, :] * flat[m][:, None]
               + red[None, :] * above[m][:, None]
               + blue[None, :] * below[m][:, None])
        img[valid[m]] = np.clip(col, 0, 255).astype(np.uint8)
    return img


def render_depth_adaptive_bgr(depth: np.ndarray, valid: np.ndarray,
                              low_pct: float = 5.0,
                              high_pct: float = 95.0) -> np.ndarray:
    """右窗口: 局部自适应灰度 (纯显示, DRL 仍用原始米制深度).

    有效像素按 [low_pct, high_pct] 分位数拉伸到 0..255, 无效像素黑.
    """
    import cv2
    m = np.asarray(valid, dtype=bool) & np.isfinite(depth)
    img = np.zeros(depth.shape[:2], dtype=np.uint8)
    if m.any():
        lo, hi = np.percentile(depth[m], [float(low_pct), float(high_pct)])
        scale = max(float(hi - lo), 1e-3)
        img[m] = np.clip((depth[m] - lo) / scale * 255.0, 0, 255).astype(np.uint8)
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
