#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Orin Landing — 对比脚本三: 世界系累积 (world-first) 融合 / 旧 10 帧窗口
====================================================================
从 rosbag 读取 /cloud_registered_body 与位姿, 以 replay_bev_geometry.py
为唯一基线. 两种地图帧模式 (--map-frame-mode, 默认 world-first):

world-first (默认, 每帧执行):
  去畸变点云 → Mid360/IMU 外参 → 该帧时间戳处 odom 插值位姿 → 统一世界
  坐标 (完整点云, 不裁剪) → 加入世界坐标滚动窗口 → 以当前帧位置为着陆
  中心 → 当前帧 bounds 裁剪 ROI → world_to_level_body 转当前水平机体
  坐标 → BEV 降采样 (--bev-grid-res 或 --bev-cell-size-m) → 两阶段单元级
  融合质控 (粗网格对齐 + 逐单元接受) → geometry 在物理融合网格上判定 /
  Bayesian 按需上采样到模型网格 (--model-grid-res, 默认 128) →
  training-camera 深度投影 → NN-fill → ONNX DRL.
  窗口由 --window-size / --window-max-age-s 约束, 关键帧策略跳过位移与
  偏航变化都过小的相邻帧; 历史帧从新到旧只补"当前帧与更新历史均未观测"
  的单元, 当前帧观测永不被覆盖.

legacy (--legacy-fusion 或 --map-frame-mode legacy):
  旧逻辑: 平滑高度曾 ≥ --fusion-arm-height-m 且下降穿越
  --fusion-start-height-m 后, "当前帧优先、历史帧只填空单元"的单元级
  滑窗 (最近 10 帧), 整帧残差拒绝 (重叠 ≥ --fusion-min-overlap-cells
  时中位残差修正统一 z 偏移, |修正| > --fusion-max-z-correction-m 或
  残差 std > --fusion-max-residual-m 或重叠不足 → 整帧拒绝参与补洞),
  不执行 XY/偏航 ICP, 不引入锚点或永久地图. 兼容命令:
    python replay_window10.py --legacy-fusion --window-size 10 \
      --bev-grid-res 128 [--fusion-max-z-correction-m 0.15]

BEV 分辨率: --bev-grid-res (默认 64) 或 --bev-cell-size-m (默认 0, 用
--bev-grid-res; >0 时按当前 ROI 物理范围自动算网格) 决定融合/占用网格;
--model-grid-res (默认 128) 决定送深度投影与网络的输出尺寸. 物理网格
geometry 语义保持物理融合网格以控制 Orin 计算量; 相机投影和 ONNX 仍使用
model-grid-res 输出. Bayesian 分支按需最近邻上采样, 不会把未知单元插值成
已观测 (observed_mask/history_fill_mask/unknown_mask 三掩码同步维护).

两阶段融合质控 (world-first):
  阶段一 (粗粒度帧间高度对齐, --alignment-grid-cell-m 0.30 物理网格):
  默认拟合 Δz=ax+by+c, 使用地面候选和迭代 MAD 剔除; --height-alignment
  scalar 可回归旧的统一 z 修正. 倾角、MAD、|c|或重叠不足时整帧保持
  unknown 并拒绝, 不强行压平错误历史帧.
  阶段二 (逐栅格接受): 局部残差 > --max-cell-residual-m、点数 <
  --min-cell-points、高度跨度 > --max-height-span-m、当前帧已观测、
  已被更新历史帧覆盖 → 只跳过该单元, 不拒绝整帧.

ground_z 规则: 显式 --ground-z 优先, 否则使用首个同步位姿高度 (与
单帧脚本一致; 不再信任快照 mission_state.ground_z_ref_m 陈旧值).

用法:
  python replay_window10.py \
    --bag /home/ifsc_orin/evelyn/landing/experiments/runs/20260807_162946_orin_landing/input.bag \
    --config /home/ifsc_orin/evelyn/landing/experiments/runs/20260807_162946_orin_landing/experiment_config_snapshot.yaml \
    --pose-topic /mavros/local_position/odom \
    --onnx-model weights/ppo2_policy.onnx \
    --semantic geometry  --window-size 30 --bev-grid-res 64 --cloud-source raw_imu
    [--semantic geometry|bayesian] [--map-frame-mode world-first|legacy]
    [--window-size 30] [--history-max-age-s 1.0]
    [--keyframe-min-translation-m 0.15] [--keyframe-min-yaw-deg 3.0]
    [--bev-grid-res 64] [--bev-cell-size-m 0.0] [--model-grid-res 128]
    [--alignment-grid-cell-m 0.30] [--max-cell-residual-m 0.25]
    [--min-cell-points 2] [--max-height-span-m 0.50]
    [--fusion-max-z-correction-m 0.30] [--fusion-max-residual-m 0.12]
    [--fusion-min-overlap-cells 20] [--show-fusion-mask]

几何语义 (geometry 主路径, 替代旧 GaussianBlur+Sobel):
  --plane-fine-radius-m 0.5  细尺度物理邻域 (坡度/粗糙度/窄柱)
  --plane-coarse-radius-m 2.0 粗尺度物理邻域 (局部地面/宽障碍显著性)
  --plane-min-support 6      最小支持点, 不足 → unknown 灰色 (非危险)
  --prominence-threshold-m 0.15 显著性阈值; 显著单元需单元内多点或
                              相邻单元连续支持, 孤立离群点不成障碍
  安全 = 支持足 + 坡度达标 + 鲁棒残差达标 + 无可靠显著障碍; 相机语义
  三态: 白=安全(几何证据) 黑=危险(真实证据) 灰=未观测/支持不足 (PPO
  仍按危险类编码). 投影时禁止默认危险值作为种子, 禁止凸包无限近邻
  默认在可靠语义种子凸包内生成连续语义图; 传入
  --semantic-fill-radius-px > 0 才切换为保守的小洞补缺模式.

深度显示:
  --depth-display-mode local|fixed  默认 local: 用真实投影射线的
  2%~98% 分位自适应灰度量程 (最小跨度 0.5m, 近地平面与柱体可区分),
  凸包外保持黑 (NN 填充不扩散成均匀灰); fixed 保持 0~30m.

三窗口坐标统一 (显示层, 不改 PPO 输入约定):
  机体 +x 前 / +y 左 / z 下; 三窗口均为 "机头朝上、机体左侧在画面左".
  BEV 显示时逆时针旋转 90° 并 NN 缩放至 128×128 后共享裁剪框, 与
  语义/深度完全对齐; 融合掩码与坡度/粗糙度诊断图同规则旋转; 各窗口
  标注 FWD ↑ / LEFT ←.
"""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass

import numpy as np

from replay_compare_common import (
    BayesianSemanticBranch,
    BevGrid,
    bev_grid_res_from_cell,
    bev_roughness_downsample,
    bev_upsample_to_model,
    cfg_value,
    dynamic_roi_half_extents,
    load_config,
    make_binary_semantic_vis,
    make_common_parser,
    perception_params,
    print_frame_block,
    project_depth,
    render_bev_bgr,
    render_depth_fixed_gray,
    render_depth_local_gray,
    render_sparse_depth,
    roi_bounds,
    setup_logging,
    upsample_grid_nearest,
    world_to_level_body,
    _batch_robust_plane_fit,
    _rot_z,
    _rot_zyx,
)

logger = logging.getLogger("ReplayWindow10")


# ──────────────────────────────────────────────
# 坐标变换 (公式与 perception/halss_preprocess.py 完全一致,
# 私有辅助 _cfg_mat3/_cfg_vec3 按原式复制, 改动需保持同步)
# ──────────────────────────────────────────────
def _cfg_vec3(cfg: dict, key: str, default) -> np.ndarray:
    arr = np.asarray(cfg.get(key, default), dtype=np.float32)
    if arr.shape != (3,):
        raise ValueError(f"{key} must be a 3-element vector")
    return arr


def _cfg_mat3(cfg: dict, key: str, default) -> np.ndarray:
    arr = np.asarray(cfg.get(key, default), dtype=np.float32)
    if arr.size != 9:
        raise ValueError(f"{key} must contain 9 values")
    return arr.reshape(3, 3)


def body_to_world(body_points: np.ndarray, pose: np.ndarray,
                  perc_cfg: dict) -> np.ndarray:
    """/cloud_registered_body (IMU 原点, z-up) → 统一世界坐标 W'.

    W' = ENU 水平 x/y + z-down (z 向下为正):
      1. IMU → base_link 刚性变换 (与 body_cloud_to_level_body_roi 相同);
      2. roll/pitch 水平化 → level-body (z 取负转 z-down);
      3. 绕 z 转 yaw + 平移 [px, py, -pz] → 世界.
    仅用于窗口历史帧: 触发前 IDLE 阶段不经过世界坐标往返.
    """
    pts = np.asarray(body_points, dtype=np.float32)[:, :3]
    if len(pts) == 0:
        return pts
    r_body_from_imu = _cfg_mat3(perc_cfg, "body_R_from_lidar_imu",
                                [1, 0, 0, 0, 1, 0, 0, 0, 1])
    t_body_from_imu = _cfg_vec3(perc_cfg, "body_T_from_lidar_imu", [0, 0, 0])
    roll, pitch, yaw = float(pose[3]), float(pose[4]), float(pose[5])
    # errstate: 某些平台 numpy float32 matmul SIMD 内核发出外观性除零警告
    # (结果正确); 输入已在 BagFrameSource 过滤有限值, 此处仅静默告警.
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        pts_base = pts @ r_body_from_imu.T + t_body_from_imu
        pts_level = pts_base @ _rot_zyx(roll, pitch, 0.0).T
        pts_level[:, 2] *= -1.0  # ENU up → z-down
        pts_w = pts_level @ _rot_z(yaw).T
    pts_w[:, 0] += float(pose[0])
    pts_w[:, 1] += float(pose[1])
    pts_w[:, 2] -= float(pose[2])  # 世界 z-down = 机下深度 - 机高(ENU)
    return pts_w.astype(np.float32, copy=False)


@dataclass
class WindowEntry:
    """窗口内一帧历史: 位姿 + 统一世界坐标完整点云 (供变换回当前水平机体系).

    pose        同步位姿 (与单帧基线一致, 仅诊断);
    interp_pose 该帧点云时间戳处的 odom 插值位姿 (线性位置 + 四元数
                SLERP), 用于 body_to_world 世界坐标变换;
    world_points 必须为该帧完整去畸变点云转换到世界坐标后的结果 (非 ROI
                裁剪), 当前帧 bounds 裁剪只在融合阶段统一执行;
    source_point_count 该帧原始去畸变点数 (世界坐标变换前的点数).
    """
    cloud_stamp: float
    pose: np.ndarray
    interp_pose: np.ndarray
    world_points: np.ndarray
    source_point_count: int = 0


@dataclass
class HistoryFrameStats:
    """一帧历史的质控与补洞统计 (帧序号按最新=0 递增)."""
    frame_idx: int
    overlap_cells: int
    z_correction: float        # 应用的中位 z 修正 (m); 拒绝时为 0
    robust_residual: float     # 重叠高度残差离散度 (m): legacy=std, world-first=1.4826·MAD
    rejected: bool
    reject_reason: str
    added_cells: int           # 该帧成功补入单元数
    rejected_cells: int        # 该帧被单元级质控拒绝的单元数 (world-first)
    dup_skipped: int           # 已被当前帧/更新历史覆盖而跳过的单元数
    plane_a: float = 0.0
    plane_b: float = 0.0
    plane_c: float = 0.0
    tilt_deg: float = 0.0
    mad_before: float = 0.0
    mad_after: float = 0.0
    ground_candidate_cells: int = 0
    obstacle_cells: int = 0


@dataclass
class FusionResult:
    """单元级补缺融合结果: 融合 BEV + 三掩码 + 统计."""
    bev: BevGrid
    observed_mask: np.ndarray     # 当前帧真实观测 (current_mask 别名)
    history_fill_mask: np.ndarray  # 历史帧补充单元
    unknown_mask: np.ndarray     # 全部历史帧均未观测
    frame_stats: list            # HistoryFrameStats, 最新→最旧
    added_cells: int
    dup_skipped: int
    rejected_frames: int
    rejected_cells: int = 0      # 单元级质控拒绝总数 (world-first)
    history_ground_mask: np.ndarray | None = None
    history_obstacle_mask: np.ndarray | None = None

    @property
    def current_mask(self) -> np.ndarray:
        """旧字段别名 (与 round4 npz 保存键兼容)."""
        return self.observed_mask

    @property
    def history_unknown_mask(self) -> np.ndarray:
        return self.unknown_mask


def _points_cell_flat(pts: np.ndarray, bounds: dict, grid_res: int) -> np.ndarray:
    """点 → BEV 单元 flat 下标 (与 bev_roughness_downsample 网格化完全一致)."""
    G = int(grid_res)
    x_min, x_max = float(bounds["x_min"]), float(bounds["x_max"])
    y_min, y_max = float(bounds["y_min"]), float(bounds["y_max"])
    x_span = max(x_max - x_min, 1e-6)
    y_span = max(y_max - y_min, 1e-6)
    col = np.rint((pts[:, 0] - x_min) / x_span * (G - 1)).astype(np.int32)
    row_unflipped = np.rint((pts[:, 1] - y_min) / y_span * (G - 1)).astype(np.int32)
    row = (G - 1) - row_unflipped
    inside = (row >= 0) & (row < G) & (col >= 0) & (col < G)
    out = -np.ones(len(pts), dtype=np.int32)
    out[inside] = row[inside] * G + col[inside]
    return out


def fuse_bev_gap_fill(current_bev: BevGrid,
                      hist_points: list,
                      bounds: dict,
                      grid_res: int,
                      min_overlap_cells: int,
                      max_z_correction_m: float,
                      max_residual_m: float) -> FusionResult:
    """"只补缺、不重复" 单元级滑窗融合 (legacy 模式, 规格 2026-08-20).

    hist_points 为已对齐到当前水平机体系的历史帧点云, 最新→最旧.
    每帧独立 bev_roughness_downsample() (禁止先拼点再统一 z_min/z_max),
    然后从最新历史帧向最旧历史帧遍历:

      - 当前帧已观测单元永远保留当前帧 (observed_mask), 不写入历史点;
      - 未被当前帧与更新历史帧覆盖的单元, 复制该历史帧完整单元
        (z_min/z_max/z_diff/count 与代表点);
      - 一个输出单元只来自一个源帧, 更旧帧不得重复写入 (dup_skipped).

    质控 (整帧残差拒绝): 每帧先与当前帧重叠单元计算 z_min 高度残差, 重叠
    ≥ min_overlap_cells 时以中位残差修正该帧统一 z 偏移; |修正| 超过
    max_z_correction_m、残差 std 超过 max_residual_m、或重叠不足时整帧
    拒绝参与补洞. 不做 XY/偏航 ICP, 不引入锚点/永久地图.
    全部历史帧均未观测单元保持 Unknown, 不插值、不生成连续曲面.
    """
    G = int(grid_res)
    occ_cur = current_bev.occupied
    fused_z_max = current_bev.z_max.copy()
    fused_z_min = current_bev.z_min.copy()
    fused_z_diff = current_bev.z_diff.copy()
    fused_count = current_bev.count.copy()
    fill_mask = np.zeros((G, G), dtype=bool)
    hist_union = np.zeros((G, G), dtype=bool)   # 通过质控帧的占用并集
    hist_points_out = []      # 已修正、入选补洞的历史代表点 (与 fill 顺序一致)
    frame_stats = []
    added_total = 0
    dup_total = 0
    rejected_total = 0

    for i, pts in enumerate(hist_points):
        if len(pts) == 0:
            rejected_total += 1
            frame_stats.append(HistoryFrameStats(
                i, 0, 0.0, 0.0, True, "empty", 0, 0, 0))
            continue
        hbev = bev_roughness_downsample(pts, bounds, grid_res=G)
        occ_h = hbev.occupied
        hist_union |= occ_h
        overlap = occ_h & occ_cur
        overlap_n = int(overlap.sum())
        z_corr = 0.0
        resid_std = 0.0
        rejected = False
        reason = ""
        if overlap_n < min_overlap_cells:
            rejected, reason = True, f"overlap {overlap_n}<{min_overlap_cells}"
        else:
            resid = hbev.z_min[overlap] - current_bev.z_min[overlap]
            z_corr = float(np.nanmedian(resid))
            resid_std = float(np.nanstd(resid))
            if abs(z_corr) > float(max_z_correction_m):
                rejected = True
                reason = f"|corr|={abs(z_corr):.3f}>{max_z_correction_m:.2f}m"
            elif resid_std > float(max_residual_m):
                rejected = True
                reason = f"std={resid_std:.3f}>{max_residual_m:.2f}m"
        if rejected:
            rejected_total += 1
            frame_stats.append(HistoryFrameStats(
                i, overlap_n, 0.0, resid_std, True, reason, 0, 0, 0))
            continue

        if z_corr != 0.0:
            # 统一 z 偏移修正: 平移该帧全部网格与代表点 (z_diff 不变)
            hbev.z_min -= z_corr
            hbev.z_max -= z_corr
            hbev.points[:, 2] -= z_corr
        covered = occ_cur | fill_mask
        new_cells = occ_h & ~covered
        dup_n = int((occ_h & covered).sum())
        dup_total += dup_n
        added = int(new_cells.sum())
        added_total += added
        if added:
            flat = _points_cell_flat(hbev.points, bounds, G)
            keep = np.isin(flat, np.flatnonzero(new_cells))
            hist_points_out.append(hbev.points[keep])
            fused_z_min[new_cells] = hbev.z_min[new_cells]
            fused_z_max[new_cells] = hbev.z_max[new_cells]
            fused_z_diff[new_cells] = hbev.z_diff[new_cells]
            fused_count[new_cells] = hbev.count[new_cells]
            fill_mask[new_cells] = True
        frame_stats.append(HistoryFrameStats(
            i, overlap_n, z_corr, resid_std, False, "", added, 0, dup_n))

    fused_occ = occ_cur | fill_mask
    unknown = ~fused_occ
    points = (np.concatenate([current_bev.points, *hist_points_out])
              if hist_points_out else current_bev.points)
    stats = dict(current_bev.stats)
    stats["occupied_cells"] = int(fused_occ.sum())
    stats["output_points"] = int(len(points))
    stats["history_union_cells"] = int(hist_union.sum())
    bev = BevGrid(points.astype(np.float32, copy=False), fused_z_max,
                  fused_z_min, fused_z_diff, fused_count, G, bounds, stats,
                  cell_x_m=current_bev.cell_x_m,
                  cell_y_m=current_bev.cell_y_m)
    return FusionResult(bev, occ_cur, fill_mask, unknown, frame_stats,
                        added_total, dup_total, rejected_total)


def _robust_scale(resid: np.ndarray) -> float:
    """残差离散度鲁棒尺度: 1.4826·MAD (与 std 同尺度, 不受扫描噪声离群影响)."""
    resid = np.asarray(resid, dtype=np.float64)
    if resid.size == 0:
        return 0.0
    med = float(np.nanmedian(resid))
    mad = float(np.nanmedian(np.abs(resid - med)))
    return 1.4826 * mad


def _neighbor_count(mask: np.ndarray) -> np.ndarray:
    """8-neighbour count without np.roll wrap-around."""
    m = np.asarray(mask, dtype=np.int16)
    p = np.pad(m, 1)
    out = np.zeros_like(m, dtype=np.int16)
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr or dc:
                out += p[1 + dr:1 + dr + m.shape[0],
                        1 + dc:1 + dc + m.shape[1]]
    return out


def _align_cell_xy(bounds: dict, grid_res: int) -> tuple[np.ndarray, np.ndarray]:
    """Return x/y coordinates at alignment-grid cell centres."""
    G = int(grid_res)
    xs = np.linspace(float(bounds["x_min"]), float(bounds["x_max"]), G)
    ys = np.linspace(float(bounds["y_min"]), float(bounds["y_max"]), G)
    return np.meshgrid(xs, ys[::-1])


def _fit_delta_plane(cur_align: BevGrid, hist_align: BevGrid,
                     min_cell_points: int, max_height_span_m: float,
                     min_neighbors: int = 3) -> dict:
    """Fit hist-current ground height residual with iterative MAD rejection."""
    overlap = hist_align.occupied & cur_align.occupied
    base = overlap.copy()
    base &= hist_align.count >= max(3, int(min_cell_points))
    base &= cur_align.count >= max(3, int(min_cell_points))
    base &= hist_align.z_diff <= min(float(max_height_span_m), 0.25)
    base &= cur_align.z_diff <= min(float(max_height_span_m), 0.25)
    base &= _neighbor_count(overlap) >= int(min_neighbors)
    rows, cols = np.where(base)
    result = {"overlap": int(overlap.sum()), "candidates": int(len(rows)),
              "a": 0.0, "b": 0.0, "c": 0.0, "tilt_deg": 0.0,
              "mad_before": 0.0, "mad_after": 0.0, "valid": False}
    if len(rows) < 3:
        return result
    xx, yy = _align_cell_xy({"x_min": cur_align.bounds["x_min"],
                             "x_max": cur_align.bounds["x_max"],
                             "y_min": cur_align.bounds["y_min"],
                             "y_max": cur_align.bounds["y_max"]},
                            cur_align.grid_res)
    # z_max is the ground-side proxy in z-down coordinates; z_min would select
    # the top of a thin pole whenever a cell contains both pole and ground.
    values = hist_align.z_max[rows, cols] - cur_align.z_max[rows, cols]
    finite = np.isfinite(values) & np.isfinite(xx[rows, cols]) & np.isfinite(yy[rows, cols])
    rows, cols, values = rows[finite], cols[finite], values[finite]
    if len(values) < 3:
        return result
    A = np.column_stack((xx[rows, cols], yy[rows, cols], np.ones(len(rows))))
    coef, *_ = np.linalg.lstsq(A, values, rcond=None)
    before_resid = values - A @ coef
    result["mad_before"] = _robust_scale(before_resid)
    keep = np.ones(len(values), dtype=bool)
    for _ in range(3):
        if int(keep.sum()) < 3:
            break
        coef, *_ = np.linalg.lstsq(A[keep], values[keep], rcond=None)
        resid = values - A @ coef
        med = float(np.median(resid[keep]))
        scale = max(_robust_scale(resid[keep]), 1e-4)
        keep = np.abs(resid - med) <= max(3.0 * scale, 0.08)
    if int(keep.sum()) < 3:
        return result
    coef, *_ = np.linalg.lstsq(A[keep], values[keep], rcond=None)
    after_resid = values[keep] - A[keep] @ coef
    a, b, c = map(float, coef)
    result.update(a=a, b=b, c=c, tilt_deg=float(np.degrees(np.arctan(np.hypot(a, b)))),
                  mad_after=_robust_scale(after_resid), valid=True)
    return result


def fuse_bev_world_first(current_bev: BevGrid,
                         hist_points: list,
                         bounds: dict,
                         grid_res: int,
                         min_overlap_cells: int,
                         max_z_correction_m: float,
                         alignment_cell_m: float = 0.30,
                         max_cell_residual_m: float = 0.25,
                         min_cell_points: int = 2,
                         max_height_span_m: float = 0.50,
                         height_alignment: str = "plane",
                         plane_tilt_threshold_deg: float = 1.5,
                         plane_mad_threshold_m: float = 0.12,
                         plane_z_offset_threshold_m: float = 0.30,
                         obstacle_prominence_m: float = 0.15,
                         obstacle_min_frames: int = 2,
                         obstacle_min_neighbors: int = 3) -> FusionResult:
    """两阶段单元级补缺融合 (world-first 模式, 规格 2026-08-21).

    hist_points 为已对齐到当前水平机体系的历史帧完整点云, 最新→最旧
    (世界坐标先累积, 当前帧 bounds 在此统一裁剪).

    阶段一 (粗粒度帧间高度对齐, --alignment-grid-cell-m 物理网格):
      在当前 ROI 的较粗对齐网格上估计帧间高度偏移, 中位残差 → 该帧统一
      z 修正; 离散度用 MAD 鲁棒尺度. 普通点云扫描噪声不得触发整帧拒绝;
      整帧拒绝仅限: 有效重叠 < min_overlap_cells、|整体修正| >
      max_z_correction_m、点云为空、含 NaN/Inf、位姿同步失败 (由调用方
      保证非空/有限, 此处一并防御).

    阶段二 (逐栅格接受历史观测):
      整体质控通过后, 对每个历史 BEV 单元检查:
        - 局部高度残差 (与当前帧重叠单元) > max_cell_residual_m → 拒单元;
        - 单元点数 < min_cell_points → 拒单元;
        - 高度跨度 z_diff > max_height_span_m → 拒单元;
        - 当前帧已观测 / 已被更新历史帧覆盖 → 跳过 (duplicate).
      只跳过单元, 不拒绝整帧.

    "只补缺、不重复": 当前帧已观测单元永远保留当前帧 (observed_mask),
    一个输出单元只来自一个源帧; fused_cells >= current_cells 恒成立.
    全部历史帧均未观测单元保持 Unknown, 不插值、不生成连续曲面.
    """
    G = int(grid_res)
    if G <= 0:
        raise ValueError("grid_res must be positive")
    if height_alignment not in {"scalar", "plane"}:
        raise ValueError("height_alignment must be scalar or plane")
    g_align = bev_grid_res_from_cell(bounds, float(alignment_cell_m))
    cur_align = bev_roughness_downsample(current_bev.points, bounds, grid_res=g_align)
    occ_cur = current_bev.occupied
    fused_z_max, fused_z_min = current_bev.z_max.copy(), current_bev.z_min.copy()
    fused_z_diff, fused_count = current_bev.z_diff.copy(), current_bev.count.copy()
    fill_mask = np.zeros((G, G), dtype=bool)
    ground_mask = np.zeros((G, G), dtype=bool)
    obstacle_mask = np.zeros((G, G), dtype=bool)
    hist_union = np.zeros((G, G), dtype=bool)
    hist_points_out, frame_stats, records = [], [], []
    rejected_total = rejected_cells_total = 0

    # First pass: reject bad frames and deskew their z in the current level-body.
    for i, raw_pts in enumerate(hist_points):
        pts = np.asarray(raw_pts, dtype=np.float32)
        if len(pts) == 0 or not np.isfinite(pts).all():
            rejected_total += 1
            frame_stats.append(HistoryFrameStats(i, 0, 0, 0, True,
                "empty" if len(pts) == 0 else "non_finite", 0, 0, 0))
            continue
        h_align = bev_roughness_downsample(pts, bounds, grid_res=g_align)
        fit = _fit_delta_plane(cur_align, h_align, min_cell_points,
                               max_height_span_m)
        overlap_n = fit["overlap"]
        if overlap_n < int(min_overlap_cells) or not fit["valid"]:
            rejected_total += 1
            reason = "insufficient_overlap" if overlap_n < int(min_overlap_cells) else "insufficient_ground_candidates"
            frame_stats.append(HistoryFrameStats(i, overlap_n, 0, fit["mad_after"], True,
                reason, 0, 0, 0, fit["a"], fit["b"], fit["c"], fit["tilt_deg"],
                fit["mad_before"], fit["mad_after"], fit["candidates"], 0))
            continue
        if height_alignment == "scalar":
            overlap = h_align.occupied & cur_align.occupied
            resid = h_align.z_min[overlap] - cur_align.z_min[overlap]
            c = float(np.nanmedian(resid))
            fit.update(a=0.0, b=0.0, c=c, tilt_deg=0.0,
                       mad_before=_robust_scale(resid), mad_after=_robust_scale(resid))
        if fit["tilt_deg"] > float(plane_tilt_threshold_deg):
            rejected_total += 1
            frame_stats.append(HistoryFrameStats(i, overlap_n, 0, fit["mad_after"], True,
                "plane_tilt_exceeded", 0, 0, 0, fit["a"], fit["b"], fit["c"], fit["tilt_deg"],
                fit["mad_before"], fit["mad_after"], fit["candidates"], 0))
            continue
        if fit["mad_after"] > float(plane_mad_threshold_m):
            rejected_total += 1
            frame_stats.append(HistoryFrameStats(i, overlap_n, 0, fit["mad_after"], True,
                "mad_exceeded", 0, 0, 0, fit["a"], fit["b"], fit["c"], fit["tilt_deg"],
                fit["mad_before"], fit["mad_after"], fit["candidates"], 0))
            continue
        if abs(fit["c"]) > float(plane_z_offset_threshold_m):
            rejected_total += 1
            frame_stats.append(HistoryFrameStats(i, overlap_n, 0, fit["mad_after"], True,
                "z_offset_exceeded", 0, 0, 0, fit["a"], fit["b"], fit["c"], fit["tilt_deg"],
                fit["mad_before"], fit["mad_after"], fit["candidates"], 0))
            continue
        xx, yy = _align_cell_xy(bounds, g_align)
        pts = pts.copy()
        # Evaluate correction in the same x/y frame used for fitting.
        pts[:, 2] -= fit["a"] * pts[:, 0] + fit["b"] * pts[:, 1] + fit["c"]
        hbev = bev_roughness_downsample(pts, bounds, grid_res=G)
        records.append((i, pts, hbev, fit))

    # Second pass: a historical pole is only trusted when temporally and
    # spatially supported. Unsupported historical obstacles remain unknown.
    support = np.zeros((G, G), dtype=np.int16)
    obstacle_candidates = []
    for _, _, hbev, fit in records:
        occ_h = hbev.occupied
        pred = np.nanmedian(cur_align.z_max[cur_align.occupied]) if cur_align.occupied.any() else np.nan
        obs = np.zeros((G, G), dtype=bool)
        if np.isfinite(pred):
            obs = occ_h & ((pred - hbev.z_min) > float(obstacle_prominence_m))
        support += obs.astype(np.int16)
        obstacle_candidates.append(obs)
    supported_obstacles = (support >= max(1, int(obstacle_min_frames)))
    supported_obstacles &= _neighbor_count(support > 0) >= int(obstacle_min_neighbors)

    added_total = dup_total = 0
    for rec_idx, (i, pts, hbev, fit) in enumerate(records):
        occ_h = hbev.occupied
        hist_union |= occ_h
        covered = occ_cur | fill_mask
        dup_n = int((occ_h & covered).sum())
        dup_total += dup_n
        cand = occ_h & ~covered
        reject_cell = np.zeros((G, G), dtype=bool)
        both = cand & occ_cur
        if both.any():
            reject_cell[both] |= np.abs(hbev.z_min[both] - current_bev.z_min[both]) > float(max_cell_residual_m)
        reject_cell[cand] |= hbev.count[cand] < int(min_cell_points)
        # A tall cell is precisely where a pole may live.  Height-span gating
        # applies to ground candidates only; supported historical obstacles
        # must not be erased by the ground quality filter.
        tall = cand & ~obstacle_candidates[rec_idx]
        reject_cell[tall] |= hbev.z_diff[tall] > float(max_height_span_m)
        unsupported_obs = obstacle_candidates[rec_idx] & ~supported_obstacles
        reject_cell |= unsupported_obs
        accepted = cand & ~reject_cell
        accepted_obs = accepted & supported_obstacles
        accepted_ground = accepted & ~supported_obstacles
        rejected_n = int(cand.sum() - accepted.sum())
        rejected_cells_total += rejected_n
        added = int(accepted.sum())
        added_total += added
        if added:
            flat = _points_cell_flat(hbev.points, bounds, G)
            keep = np.isin(flat, np.flatnonzero(accepted))
            hist_points_out.append(hbev.points[keep])
            fused_z_min[accepted], fused_z_max[accepted] = hbev.z_min[accepted], hbev.z_max[accepted]
            fused_z_diff[accepted], fused_count[accepted] = hbev.z_diff[accepted], hbev.count[accepted]
            fill_mask[accepted] = True
            ground_mask[accepted_ground] = True
            obstacle_mask[accepted_obs] = True
        old = frame_stats[i] if i < len(frame_stats) else None
        stat = HistoryFrameStats(i, fit["overlap"], fit["c"], fit["mad_after"], False, "", added,
            rejected_n, dup_n, fit["a"], fit["b"], fit["c"], fit["tilt_deg"], fit["mad_before"],
            fit["mad_after"], fit["candidates"], int(obstacle_candidates[rec_idx].sum()))
        # frame_stats contains rejected entries indexed by source frame only
        frame_stats.append(stat)

    fused_occ = occ_cur | fill_mask
    unknown = ~fused_occ
    points = (np.concatenate([current_bev.points, *hist_points_out])
              if hist_points_out else current_bev.points)
    stats = dict(current_bev.stats)
    stats["occupied_cells"] = int(fused_occ.sum())
    stats["output_points"] = int(len(points))
    stats["history_union_cells"] = int(hist_union.sum())
    bev = BevGrid(points.astype(np.float32, copy=False), fused_z_max,
                  fused_z_min, fused_z_diff, fused_count, G, bounds, stats,
                  cell_x_m=current_bev.cell_x_m,
                  cell_y_m=current_bev.cell_y_m)
    return FusionResult(bev, occ_cur, fill_mask, unknown, frame_stats,
                        added_total, dup_total, rejected_total,
                        rejected_cells_total, ground_mask, obstacle_mask)


def window_evict_age(window: deque, stamp: float, max_age_s: float) -> int:
    """世界坐标滚动窗口: 移除与当前时间差超过 max_age_s 的旧帧 (融合前执行)."""
    if max_age_s <= 0.0:
        return 0
    removed = 0
    while window and (stamp - window[0].cloud_stamp) > max_age_s:
        window.popleft()
        removed += 1
    return removed


def window_keyframe_append(window: deque, entry: WindowEntry,
                           window_size: int, min_translation_m: float,
                           min_yaw_deg: float,
                           max_rp_delta_deg: float = 1.0,
                           min_vertical_translation_m: float = 0.10) -> bool:
    """世界坐标滚动窗口: 关键帧判定 → 入窗 → 容量钳制.

    - 新帧与窗口最后一帧的位移和偏航变化都很小时跳过该帧 (不加入窗口;
      当前帧本身永远参与融合, 此处只影响历史补洞来源);
    - 超过 window_size 时移除最旧帧.
    Returns: 该帧是否实际入窗 (供日志).
    """
    if window:
        last = window[-1]
        dxyz = np.asarray(entry.interp_pose[:3], dtype=np.float64) \
            - np.asarray(last.interp_pose[:3], dtype=np.float64)
        dyaw = float(np.asarray(entry.interp_pose[5], dtype=np.float64)
                     - np.asarray(last.interp_pose[5], dtype=np.float64))
        dyaw = abs((dyaw + np.pi) % (2.0 * np.pi) - np.pi)
        trans = float(np.hypot(dxyz[0], dxyz[1]))
        drp = float(np.degrees(np.hypot(
            entry.interp_pose[3] - last.interp_pose[3],
            entry.interp_pose[4] - last.interp_pose[4])))
        if (trans < float(min_translation_m)
                and abs(float(dxyz[2])) < float(min_vertical_translation_m)
                and np.degrees(dyaw) < float(min_yaw_deg)
                and drp < float(max_rp_delta_deg)):
            return False
    window.append(entry)
    while len(window) > int(window_size):
        window.popleft()
    return True


def render_fusion_mask(current_mask: np.ndarray, fill_mask: np.ndarray,
                       unknown_mask: np.ndarray, grid_res: int,
                       obstacle_mask: np.ndarray | None = None) -> np.ndarray:
    """融合掩码: 绿当前、蓝历史地面、红历史障碍、灰 unknown."""
    img = np.zeros((int(grid_res), int(grid_res), 3), dtype=np.uint8)
    img[current_mask] = (60, 200, 60)     # BGR: 绿
    img[fill_mask] = (230, 150, 40)       # BGR: 蓝
    if obstacle_mask is not None:
        img[np.asarray(obstacle_mask, dtype=bool)] = (40, 40, 220)
    img[unknown_mask] = (70, 70, 70)      # BGR: 灰
    return img


# ──────────────────────────────────────────────
# 几何语义主路径: 物理尺度双窗口局部平面拟合 (规格 2026-08-21)
# ──────────────────────────────────────────────
def _physical_window_cells(radius_m: float, cell_x_m: float, cell_y_m: float,
                           target_cells: int, min_stride: int = 1,
                           max_radius: int = 0) -> tuple:
    """物理半径 (米) → (rx, ry, stride): 窗口行列半径 (格) 与采样步长.

    固定 5×5 像素核的实际物理尺寸随高度剧烈变化 (动态 ROI), 这里按当前
    cell_x/cell_y 分别换算行列半径, 保证坡度/粗糙度/突出判定的物理尺度恒定;
    步长把采样窗口单元数钳制到 ≤ (2·target_cells+1)², 只抽稀采样、不改变
    物理跨度.

    max_radius > 0 时把格半径钳制到该上限: 起飞低空 ROI 极小时 cell_x 可小至
    毫米级, 2 m 物理半径会换算成上千格, 批量窗口切片的中间数组瞬间爆内存
    (Orin 实测 OOM Killed); 钳制到 (grid_res-1)//2 使窗口退化为整个 BEV 网格,
    语义判定尺度在该高度下失去物理意义, 但几何证据 (支持/坡度/突出) 仍成立.
    """
    rx = max(1, int(round(float(radius_m) / max(float(cell_x_m), 1e-6))))
    ry = max(1, int(round(float(radius_m) / max(float(cell_y_m), 1e-6))))
    if max_radius > 0:
        rx = min(rx, int(max_radius))
        ry = min(ry, int(max_radius))
    target = max(int(target_cells), 1)
    m = max(rx, ry)
    stride = max(int(min_stride), (m + target // 2) // target)
    stride = min(stride, m)   # 保证采样窗口包含中心行/列
    return rx, ry, stride


def _plane_window_view(valid: np.ndarray, z_min: np.ndarray,
                       rx: int, ry: int, stride: int,
                       cell_x_m: float, cell_y_m: float):
    """观测单元批量窗口切片: 返回 (rows, cols, hw, vw, ox, oy).

    hw/vw: (N, K) 窗口内单元高度 (z_min) 与观测掩码, 未观测单元 NaN/False;
    ox/oy: (K,) 窗口单元相对偏移 (米), 与 _batch_robust_plane_fit 的
    X=[ox, oy, 1] 对齐: axis0=行 (y 偏移), axis1=列 (x 偏移), 展平顺序
    x 变化最快 (与窗口视图展平一致, 否则 a/b 错位产生虚假突出).
    """
    R = max(rx, ry)
    padded_h = np.pad(z_min, R, mode="constant", constant_values=np.nan)
    padded_v = np.pad(valid, R, mode="constant", constant_values=False)
    view_h = np.lib.stride_tricks.sliding_window_view(
        padded_h, (2 * R + 1, 2 * R + 1))
    view_v = np.lib.stride_tricks.sliding_window_view(
        padded_v.astype(np.uint8), (2 * R + 1, 2 * R + 1)).astype(bool)
    rows, cols = np.where(valid)
    if len(rows) == 0:
        return rows, cols, np.empty((0, 0), np.float32), \
            np.empty((0, 0), bool), np.empty((0,), np.float32), \
            np.empty((0,), np.float32)
    hw = view_h[rows, cols]
    vw = view_v[rows, cols]
    # 非对称半径: 截取中心 (2·ry+1)×(2·rx+1) 子窗口后按 stride 抽稀
    # (数组索引 rows/cols 得到 (N, 2R+1, 2R+1) 3 维, 展平为 (N, K);
    # row-major 展平 = ox 变化最快, 与 oxy 展平顺序一致)
    hw = np.ascontiguousarray(
        hw[:, R - ry:R + ry + 1:stride, R - rx:R + rx + 1:stride]).reshape(
        len(rows), -1)
    vw = np.ascontiguousarray(
        vw[:, R - ry:R + ry + 1:stride, R - rx:R + rx + 1:stride]).reshape(
        len(rows), -1)
    sx = np.arange(2 * rx + 1)[::stride] - rx
    sy = np.arange(2 * ry + 1)[::stride] - ry
    ox = (sx * cell_x_m).astype(np.float32)
    oy = (sy * cell_y_m).astype(np.float32)
    oxy = np.array(np.meshgrid(oy, ox, indexing="ij")).reshape(2, -1)
    return rows, cols, hw, vw, oxy[1], oxy[0]


def bev_plane_semantic_map(bev: BevGrid,
                           slope_threshold_deg: float,
                           roughness_threshold_m: float,
                           prominence_threshold_m: float = 0.15,
                           safe_id: int = 1, danger_id: int = 9,
                           fine_radius_m: float = 0.5,
                           coarse_radius_m: float = 2.0,
                           min_support: int = 6):
    """融合 BEV → 物理尺度双窗口鲁棒平面拟合语义 (window10 geometry 主路径).

    每个候选格 (真实观测) 用周围真实观测单元高度 (z_min) 鲁棒拟合
    z = a·x + b·y + c (Tukey bisquare IRLS, 批量):
      - 细尺度 (默认 0.5 m 半径) 判坡度 / 粗糙度 / 窄柱体突出;
      - 粗尺度 (默认 2.0 m 半径) 估计局部地面与宽障碍突出高度.
    窗口行列半径按 cell_x/cell_y 物理尺寸换算, 高度变化不改变判定尺度.

    危险突出需邻域空间支持: 单元内多点高出阈值, 或相邻格同样突出; 孤立
    离群点 (1 格 1 点) 不直接形成障碍. 支持不足 (窗口观测单元 < min_support)
    的观测格保持 unknown, 不默认判危险.

    低空微小 ROI 的毫米级单元会把物理半径换算成上千格, 超大滑动窗口中间
    数组会耗尽内存 (Orin 实测 OOM Killed): 窗口格半径钳制到 (G-1)//2,
    退化为整个 BEV 网格; 无观测或全局观测单元 < min_support 直接返回全
    unknown, 不构造窗口.

    Returns (sem_map uint8, semantic_valid_mask bool, maps dict):
      sem_map: 安全= safe_id, 其余 (含 unknown) 保守编码为 danger_id;
      semantic_valid_mask: 有几何证据的单元 (支持充分才参与安全/危险判定),
        False 的观测单元显示为灰色、送策略仍为 danger;
      maps: {slope_deg, roughness, prominence_fine, prominence_coarse,
             rel_height, obs_count, semantic_valid, valid} (未观测 NaN).
    """
    import cv2

    G = int(bev.grid_res)
    valid = np.asarray(bev.occupied, dtype=bool)
    z_min = np.asarray(bev.z_min, dtype=np.float32)
    cell_x_m, cell_y_m = bev.cell_size_m
    prom_th = float(prominence_threshold_m)
    min_sup = int(min_support)

    def _empty_map():
        return np.full((G, G), np.nan, dtype=np.float32)

    # 前置保护: 无观测 / 网格过小 / 全局观测单元不足 min_support → 全部
    # unknown (sem_map 保守编码 danger, 显示灰), 不进入平面拟合. 全局单元
    # < min_support 时任何窗口支持必然不足, 直接短路 (含低空毫米级单元,
    # 避免无谓的窗口构造).
    if (G < 8 or not valid.any()
            or int(np.count_nonzero(valid)) < min_sup):
        sem_map = np.full((G, G), int(danger_id), dtype=np.uint8)
        semantic_valid_mask = np.zeros((G, G), dtype=bool)
        empty = _empty_map()
        maps = {
            "slope_deg": empty.copy(),
            "roughness": empty.copy(),
            "prominence_fine": empty.copy(),
            "prominence_coarse": empty.copy(),
            "rel_height": empty.copy(),
            "obs_count": np.asarray(bev.count, dtype=np.float32),
            "semantic_valid": semantic_valid_mask,
            "valid": valid,
        }
        return sem_map, semantic_valid_mask, maps

    slope_out, rough_out = _empty_map(), _empty_map()
    prom_fine_out, prom_coarse_out = _empty_map(), _empty_map()
    rel_out = _empty_map()

    # 钳制格半径到网格能容纳的最大窗口: 毫米级单元时物理半径换算出的
    # 窗口切片中间数组 (N, 2R+1, 2R+1) 可达数十 GB (Orin 实测 OOM), 钳制
    # 后窗口退化为整个 BEV 网格, 内存有界.
    max_radius = max((G - 1) // 2, 1)

    def _scale_fit(radius_m, target_cells, min_stride):
        rx, ry, stride = _physical_window_cells(
            radius_m, cell_x_m, cell_y_m, target_cells, min_stride,
            max_radius=max_radius)
        rows, cols, hw, vw, ox, oy = _plane_window_view(
            valid, z_min, rx, ry, stride, cell_x_m, cell_y_m)
        params, resid, support = _batch_robust_plane_fit(
            hw, vw, ox, oy, min_sup)
        return rows, cols, params, resid, support

    # ── 细尺度: 坡度 / 粗糙度 / 窄突出 ──
    rows, cols, params_f, rough_f, support_f = _scale_fit(
        fine_radius_m, target_cells=6, min_stride=1)
    if len(rows):
        slope_out[rows, cols] = np.degrees(
            np.arctan(np.hypot(params_f[:, 0], params_f[:, 1])))
        rough_out[rows, cols] = rough_f

    # ── 粗尺度: 局部地面估计 / 宽突出 (窗口覆盖全部观测单元, 与细尺度同格) ──
    _, _, params_c, _, support_c = _scale_fit(
        coarse_radius_m, target_cells=10, min_stride=2)

    # ── 突出高度与空间连续支持 ──
    # z-down: 单元 z 越小物理位置越高; 突出 = 局部平面高度 - 单元最近表面.
    # 每格只保留 zmax/zmin 代表点: 同格多点 (两点均高出) 或相邻格连续突出
    # 才确认障碍, 抑制孤立噪点.
    n_high_f = np.zeros(G * G, dtype=np.int32)
    n_high_c = np.zeros(G * G, dtype=np.int32)
    z_th_f = np.full(G * G, np.inf, dtype=np.float32)
    z_th_c = np.full(G * G, np.inf, dtype=np.float32)
    if len(rows) and len(bev.points):
        pf = np.asarray(bev.points, dtype=np.float32)
        cell_of = _points_cell_flat(pf, bev.bounds, G).astype(np.int64, copy=False)

        # 融合后的代表点可能在历史帧坐标变换/高度校正后落到当前 ROI 外。
        # _points_cell_flat() 用 -1 表示这类点；这些点不能参与阈值索引或
        # np.bincount，否则会触发负索引异常，或者错误索引最后一个 cell。
        point_valid = (
            (cell_of >= 0)
            & (cell_of < G * G)
            & np.isfinite(pf[:, :3]).all(axis=1)
        )
        pf = pf[point_valid]
        cell_of = cell_of[point_valid]
        if len(pf) == 0:
            cell_of = np.empty(0, dtype=np.int64)
        flat_idx = rows * G + cols
        z_th_f[flat_idx] = params_f[:, 2] - prom_th
        z_th_c[flat_idx] = params_c[:, 2] - prom_th
        if len(pf):
            above_f = pf[:, 2] < z_th_f[cell_of]
            above_c = pf[:, 2] < z_th_c[cell_of]
            n_high_f = np.bincount(
                cell_of[above_f], minlength=G * G
            ).reshape(G, G)
            n_high_c = np.bincount(
                cell_of[above_c], minlength=G * G
            ).reshape(G, G)

    safe = np.zeros((G, G), dtype=bool)
    classification_valid = np.zeros((G, G), dtype=bool)
    if len(rows):
        zcell = z_min[rows, cols]
        rel_f = params_f[:, 2] - zcell
        rel_c = params_c[:, 2] - zcell
        prom_fine_out[rows, cols] = np.maximum(rel_f, 0.0)
        prom_coarse_out[rows, cols] = np.maximum(rel_c, 0.0)
        rel_out[rows, cols] = rel_c

        above_self_f = (rel_f > prom_th) & (n_high_f[rows, cols] >= 1)
        above_self_c = (rel_c > prom_th) & (n_high_c[rows, cols] >= 1)
        prom_above_f = np.zeros((G, G), dtype=bool)
        prom_above_c = np.zeros((G, G), dtype=bool)
        prom_above_f[rows[above_self_f], cols[above_self_f]] = True
        prom_above_c[rows[above_self_c], cols[above_self_c]] = True
        # 空间连续支持: 3×3 邻域内突出格计数 (含自身) >= 2
        neigh_f = cv2.filter2D(prom_above_f.astype(np.float32), -1,
                               np.ones((3, 3), np.float32))
        neigh_c = cv2.filter2D(prom_above_c.astype(np.float32), -1,
                               np.ones((3, 3), np.float32))
        prom_danger_f = prom_above_f & ((n_high_f >= 2) | (neigh_f >= 2))
        prom_danger_c = prom_above_c & ((n_high_c >= 2) | (neigh_c >= 2))

        supported = support_f & support_c
        slope_ok = slope_out[rows, cols] < float(slope_threshold_deg)
        rough_ok = rough_out[rows, cols] < float(roughness_threshold_m)
        safe_mask = (slope_ok & rough_ok & supported
                     & ~prom_danger_f[rows, cols]
                     & ~prom_danger_c[rows, cols])
        safe[rows[safe_mask], cols[safe_mask]] = True
        classification_valid[rows[supported], cols[supported]] = True

    # 三态: 白=安全 (有几何证据), 黑=危险 (有几何证据), 灰=未观测/支持不足
    # (sem_map 保持 danger_id 保守编码, 显示层由 semantic_valid_mask=False 渲染灰)
    sem_map = np.full((G, G), int(danger_id), dtype=np.uint8)
    sem_map[safe] = int(safe_id)
    semantic_valid_mask = valid & classification_valid

    maps = {
        "slope_deg": slope_out,
        "roughness": rough_out,
        "prominence_fine": prom_fine_out,
        "prominence_coarse": prom_coarse_out,
        "rel_height": rel_out,
        "obs_count": np.asarray(bev.count, dtype=np.float32),
        "semantic_valid": semantic_valid_mask,
        "valid": valid,
    }
    return sem_map, semantic_valid_mask, maps


class PlaneGeometrySemanticBranch:
    """geometry 主路径分支 (window10): 物理尺度双窗口鲁棒平面拟合.

    细尺度 (默认 0.5 m) 判坡度/粗糙度/窄柱体, 粗尺度 (默认 2.0 m) 判宽障碍
    突出; 支持不足的观测单元保持 unknown (semantic_valid_mask=False, 灰色),
    只有几何证据充分的单元才参与安全/危险判定. 纯 numpy/cv2, 不调用 HALSS.
    """

    def __init__(self, slope_threshold_deg: float, roughness_threshold_m: float,
                 prominence_threshold_m: float = 0.15,
                 safe_id: int = 1, danger_id: int = 9,
                 fine_radius_m: float = 0.5, coarse_radius_m: float = 2.0,
                 min_support: int = 6):
        self.slope_th = float(slope_threshold_deg)
        self.rough_th = float(roughness_threshold_m)
        self.prom_th = float(prominence_threshold_m)
        self.safe_id = int(safe_id)
        self.danger_id = int(danger_id)
        self.fine_radius_m = float(fine_radius_m)
        self.coarse_radius_m = float(coarse_radius_m)
        self.min_support = int(min_support)

    def __call__(self, bev: BevGrid, bounds: dict):
        t0 = time.perf_counter()
        sem_map, semantic_valid, maps = bev_plane_semantic_map(
            bev, self.slope_th, self.rough_th,
            prominence_threshold_m=self.prom_th,
            safe_id=self.safe_id, danger_id=self.danger_id,
            fine_radius_m=self.fine_radius_m,
            coarse_radius_m=self.coarse_radius_m,
            min_support=self.min_support)

        def _finite_stat(values, reducer):
            arr = np.asarray(values, dtype=np.float32)
            finite = arr[np.isfinite(arr)]
            return float(reducer(finite)) if finite.size else float("nan")

        return sem_map, {
            "branch": "geometry_planes",
            "latency_ms": (time.perf_counter() - t0) * 1000.0,
            "slope_deg_mean": _finite_stat(maps["slope_deg"], np.mean),
            "slope_deg_max": _finite_stat(maps["slope_deg"], np.max),
            "roughness_mean_m": _finite_stat(maps["roughness"], np.mean),
            "safe_ratio": float(np.mean(sem_map == self.safe_id)),
            "sem_valid_ratio": float(np.mean(semantic_valid)),
            "danger_valid_ratio": float(np.mean(
                (sem_map == self.danger_id) & semantic_valid)),
            "unknown_ratio": float(np.mean(~semantic_valid)),
            "maps": maps,
        }


class FusionTrigger:
    """滑动窗口启动触发 (EMA 平滑相对高度).

    仅当 曾高于 arm_height、平滑高度正在下降、且从上向下穿越
    start_height 时触发一次; 起飞经过 start_height 时不触发
    (未达 arm_height / 上升中). 触发后幂等: 主循环接管窗口状态.
    """

    def __init__(self, start_height_m: float, arm_height_m: float,
                 alpha: float = 0.3):
        self.start_h = float(start_height_m)
        self.arm_h = float(arm_height_m)
        self.alpha = float(alpha)
        self.smooth = None
        self.prev = None
        self.max_h = float("-inf")
        self._fired = False

    @property
    def fired(self) -> bool:
        return self._fired

    def update(self, rel_height: float) -> bool:
        """喂入一帧相对高度; 返回该帧是否为首次触发 (仅一次 True)."""
        h = float(rel_height)
        self.smooth = (h if self.smooth is None
                       else self.alpha * h + (1.0 - self.alpha) * self.smooth)
        self.max_h = max(self.max_h, self.smooth)
        descending = self.prev is not None and self.smooth <= self.prev
        self.prev = self.smooth
        if (not self._fired and self.max_h >= self.arm_h
                and self.smooth <= self.start_h and descending):
            self._fired = True
            return True
        return False


def main():
    parser = make_common_parser(
        "离线回放: 世界系累积 (world-first) 两阶段单元级融合 / legacy 旧 10 帧"
        "滑窗 + training-camera 深度投影 + ONNX DRL"
    )
    parser.add_argument("--semantic", type=str, default="geometry",
                        choices=["geometry", "bayesian"],
                        help="语义分支: geometry=几何分支 (默认, 无需 GPU); "
                             "bayesian=HALSS Bayesian (需 torch+CUDA)")
    parser.add_argument("--map-frame-mode", type=str, default="world-first",
                        choices=["world-first", "legacy"],
                        help="地图帧模式: world-first=世界坐标先累积、当前 ROI "
                             "后裁剪 + 两阶段融合质控 (默认); legacy=旧 10 帧"
                             "'只补缺、不重复'整帧残差拒绝逻辑")
    parser.add_argument("--legacy-fusion", action="store_true",
                        help="等价 --map-frame-mode legacy (保留旧逻辑便于 A/B 对比)")
    parser.add_argument("--window-size", type=int, default=30,
                        help="滑动窗口最大帧数 (默认 30); world-first 模式窗口"
                             "按关键帧策略增长, legacy 模式触发帧后每帧递增")
    parser.add_argument("--history-max-age-s", "--window-max-age-s", dest="window_max_age_s",
                        type=float, default=1.0,
                        help="历史帧最大时间 (秒, world-first; <=0 禁用年龄剔除)")
    parser.add_argument("--keyframe-min-translation-m", type=float, default=0.15,
                        help="关键帧最小位移 (米): 新帧与最近关键帧位移和偏航"
                             "变化都小于阈值时跳过该帧 (不入历史窗口)")
    parser.add_argument("--keyframe-min-yaw-deg", type=float, default=3.0,
                        help="关键帧最小偏航变化 (度), 与位移阈值同时满足才跳过")
    parser.add_argument("--keyframe-max-rp-delta-deg", type=float, default=1.0,
                        help="相邻关键帧 roll/pitch 变化超过该值时强制入窗")
    parser.add_argument("--bev-grid-res", type=int, default=64,
                        help="BEV 融合/占用网格分辨率 (默认 64, 建议 64/96/128);"
                             "与 --bev-cell-size-m 二选一")
    parser.add_argument("--bev-cell-size-m", type=float, default=0.0,
                        help="可选物理栅格尺寸 (米): >0 时按当前 ROI 物理范围"
                             "自动计算 BEV 网格大小, 忽略 --bev-grid-res")
    parser.add_argument("--model-grid-res", type=int, default=128,
                        help="送入深度投影与网络的模型网格分辨率 (默认 128);"
                             "物理网格融合后按最近邻上采样, 不生成新有效单元")
    parser.add_argument("--alignment-grid-cell-m", type=float, default=0.30,
                        help="阶段一粗粒度帧间高度对齐网格物理尺寸 (米, 建议 "
                             "0.25–0.40; 默认 0.30)")
    parser.add_argument("--max-cell-residual-m", type=float, default=0.25,
                        help="阶段二逐栅格接受: 单元局部高度残差上限 (米, 默认"
                             " 0.25), 超过只拒该单元")
    parser.add_argument("--min-cell-points", type=int, default=2,
                        help="阶段二逐栅格接受: 单元最少点数 (默认 2), 不足拒该单元")
    parser.add_argument("--max-height-span-m", type=float, default=0.50,
                        help="阶段二逐栅格接受: 单元高度跨度上限 (米, 默认 0.50),"
                             " z_diff 超过拒该单元")
    parser.add_argument("--slope-threshold-deg", type=float, default=None,
                        help="坡度安全阈值 (度); 与单帧脚本一致, 默认取配置 "
                             "slope_threshold_deg (缺省 10.0)")
    parser.add_argument("--roughness-threshold-m", type=float, default=None,
                        help="粗糙度安全阈值 (米); 与单帧脚本一致, 默认取配置 "
                             "geometric_roughness_threshold_m (缺省 0.15)")
    parser.add_argument("--plane-fine-radius-m", type=float, default=0.5,
                        help="几何语义细尺度物理邻域半径 (米, 默认 0.5): 判坡度/"
                             "粗糙度/窄柱体突出; 窗口半径按 cell_x/cell_y 物理"
                             "尺寸换算, 高度变化不改变判定尺度")
    parser.add_argument("--plane-coarse-radius-m", type=float, default=2.0,
                        help="几何语义粗尺度物理邻域半径 (米, 默认 2.0): 估计"
                             "局部地面与宽障碍突出高度")
    parser.add_argument("--plane-min-support", type=int, default=6,
                        help="几何语义最小邻域支持单元数 (默认 6): 支持不足的"
                             "观测单元标记为 unknown (灰色), 不默认判危险")
    parser.add_argument("--prominence-threshold-m", type=float, default=0.15,
                        help="突出高度阈值 (米, 默认 0.15): 高出局部平面超过"
                             "阈值的单元需同格多点或相邻格连续支持才判危险")
    parser.add_argument("--semantic-fill-radius-px", type=float, default=0.0,
                        help="相机语义投影小洞补全半径 (像素, 默认 0): 0 表示"
                             "在可靠语义种子凸包内连续生成语义图; >0 表示仅补"
                             "该半径内的小洞")
    parser.add_argument("--depth-display-mode", type=str, default="local",
                        choices=["local", "fixed"],
                        help="右窗口深度显示模式: local=真实射线 2%%~98%% 分位"
                             "局部自适应灰度 (最小跨度 0.5 m, 凸包外黑色, 默认);"
                             "fixed=固定 0~dmax 灰度用于绝对距离对比")
    parser.add_argument("--fusion-start-height-m", type=float, default=15.0,
                        help="legacy 窗口融合启动高度 (米, 平滑高度下降穿越该值触发)")
    parser.add_argument("--fusion-arm-height-m", type=float, default=20.0,
                        help="legacy 触发武装高度 (米): 曾高于该值才开始监测下降"
                             "穿越, 避免起飞经过启动高度误触发")
    parser.add_argument("--fusion-max-z-correction-m", type=float, default=0.30,
                        help="历史帧重叠高度残差的中位修正上限 (米, 默认 0.30): "
                             "|修正| 超过则整帧拒绝参与补洞; legacy 复现旧行为"
                             "可显式传 0.15")
    parser.add_argument("--fusion-max-residual-m", type=float, default=0.12,
                        help="legacy 历史帧重叠高度残差离散度上限 (米, 默认"
                             " 0.12): std 超过则整帧拒绝参与补洞")
    parser.add_argument("--fusion-min-overlap-cells", type=int, default=20,
                        help="历史帧与当前帧最小重叠单元数 (默认 20): 重叠不足"
                             "不做修正并整帧拒绝补洞")
    parser.add_argument("--height-alignment", choices=["scalar", "plane"], default="plane",
                        help="历史高度对齐方式 (默认 plane; scalar 用于回归对比)")
    parser.add_argument("--plane-tilt-threshold-deg", type=float, default=1.5)
    parser.add_argument("--plane-mad-threshold-m", type=float, default=0.12)
    parser.add_argument("--plane-z-offset-threshold-m", type=float, default=0.30)
    parser.add_argument("--history-obstacle-min-frames", type=int, default=2)
    parser.add_argument("--history-obstacle-min-neighbors", type=int, default=3)
    parser.add_argument("--show-fusion-mask", action="store_true",
                        help="显示融合三掩码窗口: 绿=当前观测, 蓝=历史补洞, "
                             "灰=全部历史帧均未观测 (Unknown)")
    args = parser.parse_args()
    setup_logging()

    cfg = load_config(args.config)
    params = perception_params(cfg)
    if args.dmax is not None:
        params["dmax"] = float(args.dmax)
    dmax = params["dmax"]
    perc = params["perc_cfg"]
    mode = "legacy" if args.legacy_fusion else str(args.map_frame_mode)
    window_size = int(max(1, args.window_size))
    model_res = int(max(1, args.model_grid_res))
    start_h = float(args.fusion_start_height_m)
    arm_h = float(args.fusion_arm_height_m)
    max_z_corr = float(args.fusion_max_z_correction_m)
    max_resid = float(args.fusion_max_residual_m)
    min_overlap = int(max(1, args.fusion_min_overlap_cells))
    align_cell = float(args.alignment_grid_cell_m)
    max_cell_resid = float(args.max_cell_residual_m)
    min_cell_pts = int(max(1, args.min_cell_points))
    max_height_span = float(args.max_height_span_m)
    max_age_s = float(args.window_max_age_s)
    kf_trans = float(args.keyframe_min_translation_m)
    kf_yaw_deg = float(args.keyframe_min_yaw_deg)
    kf_rp_deg = float(args.keyframe_max_rp_delta_deg)
    slope_th = cfg_value(args.slope_threshold_deg,
                         perc.get("slope_threshold_deg"), 10.0)
    rough_th = cfg_value(args.roughness_threshold_m,
                         perc.get("geometric_roughness_threshold_m"), 0.15)
    plane_fine_m = float(args.plane_fine_radius_m)
    plane_coarse_m = float(args.plane_coarse_radius_m)
    plane_min_sup = int(max(1, args.plane_min_support))
    prom_th = float(args.prominence_threshold_m)
    sem_fill_px = float(args.semantic_fill_radius_px)
    depth_mode = str(args.depth_display_mode)
    logger.info("[Window10] mode=%s window=%d max_age=%.1fs kf=(%.2fm,%.1fdeg,rp=%.1fdeg) "
                "bev=%d cell=%.2fm model=%d semantic=%s align=%.2fm "
                "cell_resid=%.2fm min_pts=%d span=%.2fm start=%.1f m "
                "arm=%.1f m slope_th=%.2f deg rough_th=%.3f m "
                "plane=(fine=%.2fm,coarse=%.2fm,support=%d,prom=%.2fm) "
                "sem_fill=%.1fpx depth=%s "
                "z_corr_max=%.2f m resid_max=%.2f m overlap_min=%d",
                mode, window_size, max_age_s, kf_trans, kf_yaw_deg, kf_rp_deg,
                int(args.bev_grid_res), float(args.bev_cell_size_m), model_res,
                args.semantic, align_cell, max_cell_resid, min_cell_pts,
                max_height_span, start_h, arm_h, slope_th, rough_th,
                plane_fine_m, plane_coarse_m, plane_min_sup, prom_th,
                sem_fill_px, depth_mode,
                max_z_corr, max_resid, min_overlap)

    # ── 模块初始化 (惰性, 与 run_standard_replay 对齐) ──
    from replay_compare_common import (BagFrameSource, CompareVisualizer,
                                       FrameSaver, ONNXDRL)
    from control.action_decomposer import ActionDecomposer
    from perception.training_camera_projection import TrainingCameraModel
    from perception.halss_preprocess import body_cloud_to_level_body_roi

    camera = TrainingCameraModel.from_config(
        cfg.get("depth_projection", {}).get("training_camera", {}),
        output_width=params["obs_w"], output_height=params["obs_h"], far_m=dmax)
    source = BagFrameSource(
        args.bag, cfg, cloud_topic=args.cloud_topic, pose_topic=args.pose_topic,
        raw_topic=args.raw_topic, imu_topic=args.imu_topic,
        cloud_source=args.cloud_source, max_sync_ms=params["max_sync_ms"])
    vis = None
    action_names = []
    action_counts = []
    total_processed = 0
    start_wall = time.perf_counter()
    try:
        drl = ONNXDRL(
            args.onnx_model, obs_h=params["obs_h"], obs_w=params["obs_w"],
            dmax=dmax,
            depth_norm_mode=str(cfg.get("observation", {}).get(
                "depth_norm_mode", "raw_meters_graph_scaled")),
            semantic_norm_mode=str(cfg.get("observation", {}).get(
                "semantic_norm_mode", "raw_gray_graph_scaled")))
        decomposer = ActionDecomposer(cfg.get("uav", {}))
        action_names = decomposer.action_names

        if args.semantic == "bayesian":
            branch = BayesianSemanticBranch(
                cfg, obs_w=params["obs_w"], obs_h=params["obs_h"],
                danger_id=params["danger_id"])
        else:
            # geometry 主路径: 物理尺度双窗口局部平面拟合 (替换旧
            # GaussianBlur+Sobel 判定), 支持不足 → unknown 三态
            branch = PlaneGeometrySemanticBranch(
                slope_th, rough_th,
                prominence_threshold_m=prom_th,
                safe_id=params["safe_id"], danger_id=params["danger_id"],
                fine_radius_m=plane_fine_m,
                coarse_radius_m=plane_coarse_m,
                min_support=plane_min_sup)

        saver = FrameSaver(args.save_dir, "window10") if args.save_dir else None
        vis = None if args.no_display else CompareVisualizer(
            dmax=dmax, show_pointcloud=not args.no_pointcloud,
            show_raw_compare=args.show_raw_compare)

        # ground_z 规则与单帧脚本一致: --ground-z 优先, 否则首个位姿高度
        ground_z = args.ground_z
        trigger = FusionTrigger(start_h, arm_h) if mode == "legacy" else None
        window = deque(maxlen=window_size)
        frame_count = 0
        action_counts = [0] * len(action_names)
        bev_log_key = None
        skipped_frames = 0

        skipped = 0
        for frame in source:
            # ground_z 必须用 bag 的第一帧位姿初始化，不能因为
            # --skip-frames 而改用第 N 帧。否则跳过起始帧后动态 ROI 会把
            # 第一个处理帧误认为地面高度，ROI 退化到最小尺寸。
            if ground_z is None:
                ground_z = float(frame.pose[2])
                logger.info("[Ground] First bag pose_z=%.2f set as ground_z",
                            ground_z)
            if args.skip_frames > 0 and skipped < args.skip_frames:
                skipped += 1
                continue
            rel_h = float(frame.pose[2]) - float(ground_z)

            half_x, half_y = dynamic_roi_half_extents(
                params, float(frame.pose[2]), float(ground_z), camera)
            bounds = roi_bounds(half_x, half_y)

            # ── BEV 融合网格: --bev-cell-size-m > 0 按物理尺寸自动, 否则 --bev-grid-res ──
            grid_res = int(args.bev_grid_res)
            if args.bev_cell_size_m > 0:
                grid_res = bev_grid_res_from_cell(bounds, args.bev_cell_size_m)
            cell_x_m = (float(bounds["x_max"]) - float(bounds["x_min"])) \
                / max(grid_res - 1, 1)
            cell_y_m = (float(bounds["y_max"]) - float(bounds["y_min"])) \
                / max(grid_res - 1, 1)
            new_bev_key = (grid_res, round(cell_x_m, 4), round(cell_y_m, 4))
            if new_bev_key != bev_log_key:
                bev_log_key = new_bev_key
                logger.info("[BEV] grid=%dx%d cell_x=%.3fm cell_y=%.3fm "
                            "roi=%.1fx%.1fm",
                            grid_res, grid_res, cell_x_m, cell_y_m,
                            float(bounds["x_max"]) - float(bounds["x_min"]),
                            float(bounds["y_max"]) - float(bounds["y_min"]))

            # ── 点云时间戳处 odom 插值位姿 (位置线性 + 四元数 SLERP;
            #    无夹住样本时回退同步位姿, 不外推) ──
            interp_pose = source.pose_at_interp(frame.cloud_stamp)
            if interp_pose is None:
                interp_pose = frame.pose

            # ── 当前帧 → 统一世界坐标 (完整去畸变点云, 不裁剪 ROI) ──
            world_points = body_to_world(frame.cloud_pts, interp_pose, perc)

            # ── 当前帧 → 当前水平机体坐标 (roll/pitch 已在世界帧中消去,
            #    仅 yaw+平移; 与 body_to_world 同一位姿 → 恒等往返) ──
            level_points = world_to_level_body(world_points, interp_pose)
            current_bev = bev_roughness_downsample(
                level_points, bounds, grid_res=grid_res)
            # input_points counts finite points before the current ROI bounds
            # are applied.  At the first frame ground_z is initialized from
            # the vehicle height, so a dynamic ROI can legitimately collapse
            # to its minimum size and contain no points.  Skip such frames
            # before triggering fusion/semantic processing.
            if (not current_bev.occupied.any()
                    or current_bev.stats["output_points"] < 10):
                logger.debug("[Frame] Sparse ROI: %d points, skip",
                             current_bev.stats["output_points"])
                continue

            # ── 窗口状态 ──
            win_frames = 0
            window_active = False
            if mode == "legacy":
                # 旧触发: 曾 ≥ arm_h 且平滑高度下降穿越 start_h ──
                if trigger.update(rel_h):
                    window.clear()
                    logger.info("[State] window 启动: 曾达 %.1f m 下降穿越 %.1f m "
                                "(H=%.2f m)", trigger.max_h, start_h, rel_h)
                if trigger.fired:
                    window_active = True
                    win_frames = len(window) + 1
                    window.append(WindowEntry(
                        cloud_stamp=frame.cloud_stamp, pose=frame.pose.copy(),
                        interp_pose=interp_pose, world_points=world_points,
                        source_point_count=int(len(frame.cloud_pts))))
            else:
                # world-first: 先按年龄剔除旧帧, 融合后再关键帧入窗
                window_evict_age(window, frame.cloud_stamp, max_age_s)
                if len(window) > 0:
                    window_active = True
                    win_frames = len(window) + 1

            # ── 单元级"只补缺、不重复"融合 ──
            # 历史帧保存的是完整世界点云, 统一用当前帧 (插值) 位姿对齐回
            # 当前水平机体系, 最后统一按当前帧 bounds 裁剪 (BEV 网格化丢弃
            # 界外点); 从最新→最旧只填当前帧与更新历史均未观测的单元
            t0 = time.perf_counter()
            fill_stats = None
            if mode == "world-first" and window_active:
                hist_pts = [world_to_level_body(e.world_points, interp_pose)
                            for e in reversed(list(window))]
                fill_stats = fuse_bev_world_first(
                    current_bev, hist_pts, bounds, grid_res,
                    min_overlap, max_z_corr, align_cell,
                    max_cell_resid, min_cell_pts, max_height_span,
                    height_alignment=args.height_alignment,
                    plane_tilt_threshold_deg=args.plane_tilt_threshold_deg,
                    plane_mad_threshold_m=args.plane_mad_threshold_m,
                    plane_z_offset_threshold_m=args.plane_z_offset_threshold_m,
                    obstacle_prominence_m=prom_th,
                    obstacle_min_frames=args.history_obstacle_min_frames,
                    obstacle_min_neighbors=args.history_obstacle_min_neighbors)
            elif mode == "legacy" and trigger.fired and len(window) > 1:
                hist_pts = [world_to_level_body(e.world_points, interp_pose)
                            for e in reversed(list(window)[:-1])]
                fill_stats = fuse_bev_gap_fill(
                    current_bev, hist_pts, bounds, grid_res,
                    min_overlap, max_z_corr, max_resid)

            if fill_stats is not None:
                bev = fill_stats.bev
                observed_mask = fill_stats.observed_mask
                fill_mask = fill_stats.history_fill_mask
                unknown_mask = fill_stats.unknown_mask
                history_ground_mask = (fill_stats.history_ground_mask
                                       if fill_stats.history_ground_mask is not None
                                       else np.zeros_like(fill_mask))
                history_obstacle_mask = (fill_stats.history_obstacle_mask
                                         if fill_stats.history_obstacle_mask is not None
                                         else np.zeros_like(fill_mask))
            else:
                bev = current_bev
                observed_mask = current_bev.occupied
                fill_mask = np.zeros((grid_res, grid_res), dtype=bool)
                unknown_mask = ~observed_mask
                history_ground_mask = np.zeros_like(fill_mask)
                history_obstacle_mask = np.zeros_like(fill_mask)
            if len(bev.points) == 0:
                logger.debug("[Frame] Empty BEV, skip")
                continue

            # ── 覆盖诊断: 融合 BEV 有效栅格数 ≥ 当前单帧 (仅填空) ──
            fused_cells = int(bev.occupied.sum())
            single_cells = int(current_bev.occupied.sum())
            if fill_stats is not None and fused_cells < single_cells:
                logger.error(
                    "[Fuse] 融合 BEV 覆盖 %d < 当前单帧 %d — 融合丢失单元!",
                    fused_cells, single_cells)

            # ── 融合日志: [Fuse] 汇总 + 每历史帧 [FuseFrame] ──
            added_cells = 0
            dup_skipped = 0
            if fill_stats is not None:
                added_cells = fill_stats.added_cells
                dup_skipped = fill_stats.dup_skipped
                history_cells = int(bev.stats.get("history_union_cells", 0))
                accepted_frames = sum(
                    1 for st in fill_stats.frame_stats if not st.rejected)
                logger.info(
                    "[Fuse] current_cells=%d history_cells=%d "
                    "accepted_history_frames=%d rejected_history_frames=%d "
                    "rejected_cells=%d added_cells=%d duplicate_cells=%d "
                    "unknown_cells=%d coverage=%.3f win=%d",
                    single_cells, history_cells, accepted_frames,
                    fill_stats.rejected_frames, fill_stats.rejected_cells,
                    added_cells, dup_skipped, int(unknown_mask.sum()),
                    fused_cells / float(grid_res * grid_res), win_frames)
                for st in fill_stats.frame_stats:
                    logger.info(
                        "[FuseFrame] frame_index=%d overlap_cells=%d "
                        "ground_candidates=%d plane=(%+.5f,%+.5f,%+.3f) "
                        "tilt=%.3fdeg mad_before=%.3f mad_after=%.3f "
                        "accepted_cells=%d rejected_cells=%d reject_reason=%s",
                        st.frame_idx, st.overlap_cells, st.ground_candidate_cells,
                        st.plane_a, st.plane_b, st.plane_c, st.tilt_deg,
                        st.mad_before, st.mad_after, st.added_cells, st.rejected_cells,
                        st.reject_reason or "none")

            # ── 物理网格 / 模型网格分离 ──
            # geometry 的局部平面拟合必须留在物理融合网格 (通常 64×64)。
            # 若先复制到 128×128，占用格和窗口矩阵约放大 4 倍，Orin 上会
            # 显著拖慢每帧处理；training-camera 投影和 ONNX 本身仍输出
            # model_res×model_res，因此 geometry 不需要 BEV 上采样。
            # Bayesian 分支保留原有模型网格输入约定。
            if model_res != grid_res and args.semantic == "bayesian":
                bev = bev_upsample_to_model(bev, model_res)
            observed_mask_m = upsample_grid_nearest(
                observed_mask, grid_res, model_res)
            fill_mask_m = upsample_grid_nearest(fill_mask, grid_res, model_res)
            unknown_mask_m = ~(observed_mask_m | fill_mask_m)
            history_ground_mask_m = upsample_grid_nearest(
                history_ground_mask, grid_res, model_res)
            history_obstacle_mask_m = upsample_grid_nearest(
                history_obstacle_mask, grid_res, model_res)

            # ── world-first: 当前帧入窗 (关键帧策略) ──
            if mode == "world-first":
                if window_keyframe_append(
                        window, WindowEntry(
                            cloud_stamp=frame.cloud_stamp,
                            pose=frame.pose.copy(), interp_pose=interp_pose,
                            world_points=world_points,
                            source_point_count=int(len(frame.cloud_pts))),
                        window_size, kf_trans, kf_yaw_deg,
                        max_rp_delta_deg=kf_rp_deg):
                    skipped_frames = 0
                else:
                    skipped_frames += 1
                    logger.debug(
                        "[Window] frame t=%.3f 位移/偏航变化过小, 关键帧跳过 "
                        "(累计 %d)", frame.cloud_stamp, skipped_frames)

            # ── 语义分支 (与单帧脚本参数一致, 模型网格) ──
            sem_map, sem_info = branch(bev, bounds)

            # ── training-camera 深度投影 ──
            # geometry 主路径: 语义种子只允许几何支持点 (semantic_valid)。
            # 默认在这些可靠种子的投影凸包内生成连续语义图，否则语义图
            # 会退化为稀疏激光点的形状；显式指定 >0 时才使用保守小洞模式。
            # bayesian 保持历史凸包填充行为 (fill_unobserved=True)。
            if args.semantic == "bayesian":
                fill_unobserved = True
                sem_bev_valid = None
                sem_fill_px_eff = 0.0
            else:
                sem_bev_valid = sem_info.get("maps", {}).get("semantic_valid")
                if sem_fill_px > 0.0:
                    fill_unobserved = False
                    sem_fill_px_eff = float(sem_fill_px)
                else:
                    fill_unobserved = True
                    sem_fill_px_eff = 0.0
            sparse_depth, valid_mask, sem_map, semantic_valid_mask = project_depth(
                bev.points, sem_map, bounds, camera, danger_id=params["danger_id"],
                fill_unobserved=fill_unobserved,
                semantic_bev_valid=sem_bev_valid,
                semantic_fill_radius_px=sem_fill_px_eff)
            # Do not mask semantic pixels by the sparse BEV occupancy mask:
            # that would make the semantic image look identical to the raw
            # point cloud.  project_depth already computes the continuous
            # evidence/convex-hull validity after projection.  Only pixels
            # rejected by that projection remain neutral unknown.
            sem_map = sem_map.copy()
            sem_map[~semantic_valid_mask] = np.uint8(128)
            dense_depth = render_sparse_depth(sparse_depth, valid_mask, dmax)

            # ── ONNX DRL ──
            t_onnx = time.perf_counter()
            action_id, rl_info = drl.predict(dense_depth, sem_map)
            onnx_ms = (time.perf_counter() - t_onnx) * 1000.0
            action_name = decomposer.action_id_to_name(action_id)
            action_counts[action_id] += 1
            total_ms = (time.perf_counter() - t0) * 1000.0
            frame_count += 1
            total_processed += 1

            sem_safe_ratio = float(np.mean(sem_map == params["safe_id"]))
            # 三态比值: 安全(有几何证据) / 危险(有真实几何证据) / unknown
            # (未观测或支持不足, 显示为灰, 仍按 danger 编码进 PPO 输入)
            sem_danger_valid_ratio = float(
                np.mean((sem_map == params["danger_id"]) & semantic_valid_mask))
            sem_unknown_ratio = float(np.mean(~semantic_valid_mask))
            sem_latency_ms = float(sem_info.get("latency_ms", 0.0))
            depth_valid_ratio = float(np.mean(valid_mask))

            print_frame_block(
                "window10", frame_count,
                n_raw=len(frame.raw_pts) if frame.raw_pts is not None else None,
                n_deskewed=int(len(frame.cloud_pts)),
                n_processed=int(len(bev.points)),
                window_frames=(win_frames if window_active else None),
                sync_ms=frame.sync_ms,
                depth_valid_ratio=depth_valid_ratio,
                sem_safe_ratio=sem_safe_ratio,
                action_id=action_id, action_name=action_name,
                probs=rl_info["action_probs"], action_names=action_names,
                onnx_ms=onnx_ms, total_ms=total_ms,
                cloud_source=frame.cloud_source)

            if frame_count % 5 == 0:
                fused_str = f"{fused_cells}/{single_cells}"
                if window_active:
                    fused_str += f"+{added_cells}-{dup_skipped}"
                logger.info(
                    "[%04d] %s source=%s mode=%s act=%d(%s) pts=%d cells=%s "
                    "win=%d H=%.2f sem_safe=%.2f sem_danger=%.2f "
                    "sem_unknown=%.2f depth_valid=%.2f conf=%.2f "
                    "sem=%.0fms %.0fms",
                    frame_count, "window10", frame.cloud_source, mode, action_id, action_name,
                    int(len(bev.points)), fused_str,
                    win_frames, rel_h, sem_safe_ratio, sem_danger_valid_ratio,
                    sem_unknown_ratio, depth_valid_ratio,
                    rl_info.get("confidence", 0.0), sem_latency_ms, total_ms)

            # ── 可视化 (左图单帧 Processed BEV 配色, 右图深度按模式) ──
            if vis is not None:
                text = (f"Frame: {frame_count}  Height: {rel_h:.1f} m  "
                        f"Action: {action_id} ({action_name})")
                # 深度显示模式: local 用真实投影射线的 2%~98% 分位做自适应
                # 灰度量程 (min_span=0.5m, 近地平面与柱体可区分), 凸包外
                # 保持黑 (NN 填充不扩散成均匀灰); fixed 保持 0~dmax
                if depth_mode == "local":
                    depth_bgr_disp, (dnear, dfar) = render_depth_local_gray(
                        dense_depth, valid_mask, dmax)
                    depth_text = f"{dnear:.1f}~{dfar:.1f}m"
                else:
                    depth_bgr_disp = render_depth_fixed_gray(dense_depth,
                                                             vmax_m=dmax)
                    depth_text = f"0~{dmax:.0f}m"
                vis.update(render_bev_bgr(bev), sem_map,
                           semantic_valid_mask, dense_depth, text=text,
                           depth_bgr=depth_bgr_disp, depth_text=depth_text)
                if args.show_fusion_mask:
                    vis.update_diag(
                        {"4.Fusion Mask (G=cur B=hist gray=unknown)":
                         render_fusion_mask(observed_mask_m, fill_mask_m,
                                            unknown_mask_m, model_res,
                                            history_obstacle_mask_m)},
                        text=text)
                if vis.show_raw_compare:
                    vis.update_3d(raw_pts=frame.raw_pts, raw_stamp=frame.raw_stamp,
                                  deskewed_pts=frame.cloud_pts,
                                  deskewed_stamp=frame.cloud_stamp)

            # ── 保存 ──
            if saver is not None:
                binary_semantic = make_binary_semantic_vis(
                    sem_map, safe_id=params["safe_id"], danger_id=params["danger_id"])
                binary_semantic[~semantic_valid_mask] = 128
                saver.save(frame_count, {
                    "sparse_depth": sparse_depth,
                    "dense_depth": dense_depth,
                    "sem_map": sem_map,
                    "binary_semantic": binary_semantic,
                    "valid_mask": valid_mask,
                    "semantic_valid_mask": semantic_valid_mask,
                    "pose": frame.pose,
                    "cloud_stamp": frame.cloud_stamp,
                    "pose_stamp": frame.pose_stamp,
                    "raw_stamp": frame.raw_stamp,
                    "sync_ms": frame.sync_ms,
                    "cloud_seq": frame.cloud_seq,
                    "action_id": action_id,
                    "action_probs": np.asarray(rl_info["action_probs"], dtype=np.float32),
                    "raw_points": frame.raw_pts,
                    "deskewed_points": frame.cloud_pts,
                    "processed_points": bev.points,
                    "fused_cells": fused_cells,
                    "single_cells": single_cells,
                    "window_frames": win_frames,
                    "window_active": float(window_active),
                    "observed_mask": observed_mask_m,
                    "current_mask": observed_mask_m,
                    "history_fill_mask": fill_mask_m,
                    "unknown_mask": unknown_mask_m,
                    "history_ground_mask": history_ground_mask_m,
                    "history_obstacle_mask": history_obstacle_mask_m,
                    "added_cells": added_cells,
                    "dup_skipped_cells": dup_skipped,
                    "fused_coverage": fused_cells / float(grid_res * grid_res),
                }, save_raw_arrays=args.save_raw_arrays, summary={
                    "cloud_stamp": frame.cloud_stamp, "pose_stamp": frame.pose_stamp,
                    "raw_stamp": frame.raw_stamp or "", "sync_ms": frame.sync_ms,
                    "cloud_seq": frame.cloud_seq, "action_id": action_id,
                    "n_raw": len(frame.raw_pts) if frame.raw_pts is not None else "",
                    "n_deskewed": len(frame.cloud_pts),
                    "n_processed": len(bev.points),
                    "fused_cells": fused_cells,
                    "single_cells": single_cells,
                    "window_frames": win_frames,
                    "depth_valid_ratio": f"{depth_valid_ratio:.3f}",
                    "sem_safe_ratio": f"{sem_safe_ratio:.3f}",
                    "onnx_ms": f"{onnx_ms:.1f}", "total_ms": f"{total_ms:.1f}"})

            # ── 速率控制 / 帧数上限 ──
            if args.rate > 0:
                elapsed = time.perf_counter() - t0
                sleep = max(0.0, (1.0 / args.rate) - elapsed)
                if sleep > 0:
                    time.sleep(sleep)
            if args.max_frames > 0 and frame_count >= args.max_frames:
                logger.info("[Replay] Max frames reached: %d", frame_count)
                break
    finally:
        source.close()
        from replay_compare_common import log_summary
        elapsed = time.perf_counter() - start_wall
        log_summary("window10", total_processed, elapsed, action_names,
                    action_counts)
        if vis is not None:
            vis.close()


if __name__ == "__main__":
    main()
