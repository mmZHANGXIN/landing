#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Orin Landing — 第 1060 帧原始点云 / 去畸变点云 / ROI 点云对比诊断
================================================================
独立诊断入口, 只依赖 rosbag / numpy / scipy / matplotlib / pyyaml,
不初始化 torch / HALSS / ONNX Runtime / DRL (帧定位与统计全部实时
从 bag 计算, 不硬编码第 1060 帧数值).

目标帧定位 (--cloud-seq 优先, 未指定时用 --frame-index):
  - --frame-index N : bag 中第 N 个 /cloud_registered_body 消息 (0 基);
  - --cloud-seq  S  : PointCloud2 header.seq == S 的消息.
两者必须明确区分: bag 消息序号 ≠ header.seq.

输出 (保存到 --output-dir):
  frame_<index>_pointcloud_pipeline.png  A/B/C: 原始 Livox / 去畸变 /
                                          level-body ROI 三视图;
  frame_<index>_bev_occupancy.png        E: 64/96/128 分辨率占用对比
                                          (单帧 ROI + 30 帧 world-first);
  frame_<index>_stats.json               全部统计 (见 _build_stats).

坐标链严格沿用当前感知处理链 (与 replay_window10 完全一致):
  body_R_from_lidar_imu / body_T_from_lidar_imu 外参 → PX4 roll/pitch
  水平化 → ENU z-up 转 z-down → 当前动态 ROI 裁剪; 融合对比使用同一
  body_to_world → world_to_level_body 世界系往返, 不在本脚本重新定义
  坐标系.

D 节五面板 (统一坐标范围与统一颜色范围):
  单帧去畸变 ROI / 10 帧 crop-first (每帧先按自身 ROI 裁剪再拼世界,
  仅用于对比禁止流程) / 10 帧 world-first / 30 帧 world-first /
  当前 replay_window10 实际融合结果 (legacy 10 帧 fuse_bev_gap_fill,
  旧质控参数 min_overlap=20, z_corr=0.15, resid=0.12).

用法:
  python diagnose_pointcloud_frame.py \
    --bag experiments/20260807_162946_orin_landing/input.bag \
    --config experiments/20260807_162946_orin_landing/experiment_config_snapshot.yaml \
    --frame-index 1060 [--cloud-seq 1557] [--output-dir diagnose_frame]
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np

# 项目包根目录 (orinlanding/) 与 scripts/ 加入 sys.path
_SCRIPT_DIR = Path(__file__).resolve().parent
_PACKAGE_ROOT = _SCRIPT_DIR.parent
for _p in (str(_PACKAGE_ROOT), str(_SCRIPT_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import rosbag  # noqa: E402  (允许依赖之一)

from replay_compare_common import (  # noqa: E402
    bev_roughness_downsample,
    custom_msg_to_numpy,
    dynamic_roi_half_extents,
    load_config,
    pc2_to_numpy,
    perception_params,
    quat_slerp,
    quat_to_euler,
    roi_bounds,
    stamp_to_sec,
    world_to_level_body,
    _rot_z,
)
from replay_window10 import (  # noqa: E402
    body_to_world,
    fuse_bev_gap_fill,
    fuse_bev_world_first,
)

logger = logging.getLogger("DiagnoseFrame")


# ──────────────────────────────────────────────
# 绕过 perception/__init__ (导入 torch) 直接加载纯 numpy 感知模块
# ──────────────────────────────────────────────
def _load_numpy_module(relpath: str, module_name: str):
    """importlib 按文件路径加载纯 numpy 模块, 不触发 perception 包 __init__."""
    path = _PACKAGE_ROOT / relpath
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load module file: {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


# halss_preprocess 仅依赖 numpy (body_cloud_to_level_body_roi 原样可用)
_halss = _load_numpy_module("perception/halss_preprocess.py",
                            "diagnose_halss_preprocess")
body_cloud_to_level_body_roi = _halss.body_cloud_to_level_body_roi


def _level_body_roi_silent(points, roll_rad, pitch_rad, cfg, half_x, half_y):
    """包装 body_cloud_to_level_body_roi: float32 SIMD matmul 的除零/溢出
    告警为外观性 (与 body_to_world 同因, 实测结果正确), 静默输出."""
    with np.errstate(all="ignore"):
        return body_cloud_to_level_body_roi(
            points, roll_rad, pitch_rad, cfg, half_x=half_x, half_y=half_y)


class _DiagnosticCamera:
    """与 perception/training_camera_projection.TrainingCameraModel 公式一致
    (source 尺寸/fx/fy → FOV → ground_half_extents), 纯 numpy 复刻,
    避免 cv2 与感知包依赖; 改动需与源文件保持同步."""

    def __init__(self, cfg: dict):
        t = cfg.get("depth_projection", {}).get("training_camera", {}) or {}
        self.source_w = float(t.get("source_width", 752))
        self.source_h = float(t.get("source_height", 480))
        self.source_fx = float(t.get("source_fx", 455.0))
        self.source_fy = float(t.get("source_fy", 455.0))
        self.near_m = float(t.get("near_m", 0.05))

    @property
    def horizontal_fov_deg(self) -> float:
        return float(np.degrees(2.0 * np.arctan(self.source_w / (2.0 * self.source_fx))))

    @property
    def vertical_fov_deg(self) -> float:
        return float(np.degrees(2.0 * np.arctan(self.source_h / (2.0 * self.source_fy))))

    def ground_half_extents(self, height_m: float) -> tuple[float, float]:
        height = max(float(height_m), self.near_m)
        half_forward = height * np.tan(np.deg2rad(self.vertical_fov_deg) * 0.5)
        half_lateral = height * np.tan(np.deg2rad(self.horizontal_fov_deg) * 0.5)
        return float(half_forward), float(half_lateral)


# ──────────────────────────────────────────────
# 单遍 bag 扫描: 全部去畸变帧 + 位姿 + 目标附近原始扫描
# ──────────────────────────────────────────────
class BagScan:
    """一次遍历收集: 全部 cloud 帧 (内存 ~50 MB)、全部位姿、滚动原始扫描.

    目标帧找到后继续收集原始扫描直到 target+1.0 s 即提前终止, 减少解码量.
    """

    def __init__(self, bag_path: str, cloud_topic: str, pose_topic: str,
                 raw_topic: str, max_sync_ms: float):
        self.bag_path = str(bag_path)
        self.cloud_topic = cloud_topic
        self.pose_topic = pose_topic
        self.raw_topic = raw_topic
        self.max_sync_ms = float(max_sync_ms)
        self.clouds = []              # dict: idx/seq/stamp/pts/frame_id/pose6
        self.poses = []               # (stamp, pose6, quat)
        self.raw_scans = []           # 目标帧 ±1s 内 (stamp, pts)
        self._raw_roll = deque(maxlen=12)
        self._target = None           # 找到目标帧的 bag cloud 序号
        self._scan_done = False

    # ── 访问器 ──
    def pose_at(self, t: float) -> np.ndarray | None:
        """点云时间戳处位姿: 位置线性 + 四元数 SLERP; 无夹住样本回退最近."""
        poses = self.poses
        if not poses:
            return None
        times = np.array([p[0] for p in poses])
        if t <= times[0] or t >= times[-1]:
            i = int(np.argmin(np.abs(times - t)))
            return poses[i][1].copy()
        i = int(np.searchsorted(times, t, side="right")) - 1
        t0, p0, q0 = poses[i]
        t1, p1, q1 = poses[i + 1]
        frac = (t - t0) / max(t1 - t0, 1e-9)
        xyz = p0[:3] + frac * (p1[:3] - p0[:3])
        q = quat_slerp(q0, q1, frac)
        roll, pitch, yaw = quat_to_euler(q[0], q[1], q[2], q[3])
        return np.array([xyz[0], xyz[1], xyz[2], roll, pitch, yaw],
                        dtype=np.float32)

    def scan(self, target_seq: int | None, target_idx: int | None) -> int:
        """执行遍历, 返回目标帧的 bag cloud 序号 (找不到抛 RuntimeError)."""
        bag = rosbag.Bag(self.bag_path, "r")
        try:
            topics = [self.cloud_topic, self.pose_topic]
            if self.raw_topic:
                topics.append(self.raw_topic)
            for topic_name, msg, ros_stamp in bag.read_messages(topics=topics):
                stamp = (stamp_to_sec(msg.header.stamp)
                         if hasattr(msg, "header") and msg.header is not None
                         else stamp_to_sec(ros_stamp))
                if topic_name == self.pose_topic:
                    pose6, quat = self._odom_to_pose6(msg)
                    if pose6 is not None:
                        self.poses.append((stamp, pose6, quat))
                    continue
                if self.raw_topic and topic_name == self.raw_topic:
                    try:
                        raw_pts = custom_msg_to_numpy(msg)
                    except Exception:
                        continue
                    entry = (stamp, raw_pts)
                    self._raw_roll.append(entry)
                    if self._target is not None:
                        self.raw_scans.append(entry)
                    continue
                # ── cloud 帧 ──
                cloud_pts = pc2_to_numpy(msg)
                if len(cloud_pts) == 0:
                    continue
                seq = int(getattr(msg.header, "seq", 0))
                idx = len(self.clouds)
                self.clouds.append({
                    "idx": idx, "seq": seq, "stamp": float(stamp),
                    "pts": cloud_pts,
                    "frame_id": str(getattr(msg.header, "frame_id", "")),
                    "pose6": self.pose_at(stamp),
                })
                if self._target is None:
                    if target_seq is not None and seq == target_seq:
                        self._target = idx
                    elif target_seq is None and target_idx is not None \
                            and idx == target_idx:
                        self._target = idx
                if self._target is not None:
                    # 继续收集原始扫描到 target+1.0 s 后提前终止
                    if stamp > self.clouds[self._target]["stamp"] + 1.0:
                        self._scan_done = True
                        break
        finally:
            bag.close()
        if self._target is None:
            raise RuntimeError(
                f"未找到目标帧: frame-index={target_idx} cloud-seq={target_seq}; "
                f"bag 内共 {len(self.clouds)} 个 cloud 消息, seq 范围 "
                f"{self.clouds[0]['seq'] if self.clouds else '-'}.."
                f"{self.clouds[-1]['seq'] if self.clouds else '-'}")
        # 目标帧 ±1s 之外的滚动原始扫描并入 (目标时刻之前的最近扫描)
        for entry in self._raw_roll:
            if abs(entry[0] - self.clouds[self._target]["stamp"]) <= 1.0:
                self.raw_scans.append(entry)
        return self._target

    @staticmethod
    def _odom_to_pose6(msg):
        try:
            p = msg.pose.pose.position
            q = msg.pose.pose.orientation
        except AttributeError:
            return None, None
        roll, pitch, yaw = quat_to_euler(q.x, q.y, q.z, q.w)
        pose6 = np.array([p.x, p.y, p.z, roll, pitch, yaw], dtype=np.float32)
        quat = np.array([q.x, q.y, q.z, q.w], dtype=np.float64)
        return pose6, quat

    def nearest_raw(self, stamp: float) -> tuple[float | None, np.ndarray | None,
                                                 float | None]:
        if not self.raw_scans:
            return None, None, None
        best = min(self.raw_scans, key=lambda e: abs(e[0] - stamp))
        return best[0], best[1], abs(best[0] - stamp) * 1000.0


# ──────────────────────────────────────────────
# 占用率指标 (numpy + scipy)
# ──────────────────────────────────────────────
def grid_occupancy_metrics(occupied: np.ndarray, cell_x_m: float,
                           cell_y_m: float) -> dict:
    """BEV 占用率指标: 占用格数/覆盖率/凸包内空洞/最大空洞半径/NN 距离.

    - convex_hull_inside_holes: 严格位于占用凸包内部 (不含边界) 的空洞格;
    - max_hole_radius_m: 空洞格到最近占用格的距离变换最大值;
    - median/p90_nearest_neighbor_m: 占用格两两最近距离 (排除自身).
    """
    from scipy.ndimage import distance_transform_edt
    from scipy.spatial import ConvexHull, cKDTree

    occ = np.asarray(occupied, dtype=bool)
    G = int(occ.shape[0])
    holes = ~occ
    occupied_cells = int(occ.sum())
    result = {
        "occupied_cells": occupied_cells,
        "coverage_ratio": float(occ.mean()),
        "convex_hull_inside_holes": {"count": 0, "ratio": 0.0},
        "max_hole_radius_m": 0.0,
        "median_nearest_neighbor_m": float("nan"),
        "p90_nearest_neighbor_m": float("nan"),
    }
    if occupied_cells < 3:
        return result
    rows, cols = np.nonzero(occ)
    try:
        hull = ConvexHull(np.column_stack([cols, rows]))
    except Exception:
        return result
    try:
        from matplotlib.path import Path as MplPath
        hull_path = MplPath(
            np.column_stack([cols[hull.vertices], rows[hull.vertices]]))
        cc, rr = np.meshgrid(np.arange(G), np.arange(G))
        in_hull = hull_path.contains_points(
            np.column_stack([cc.ravel(), rr.ravel()])).reshape(G, G)
    except Exception:
        in_hull = None
    if in_hull is not None:
        hole_in_hull = holes & in_hull
        result["convex_hull_inside_holes"] = {
            "count": int(hole_in_hull.sum()),
            "ratio": float(hole_in_hull.mean()),
        }
        if hole_in_hull.any():
            dt = distance_transform_edt(holes, sampling=(cell_y_m, cell_x_m))
            result["max_hole_radius_m"] = float(dt[hole_in_hull].max())
    # 占用格最近邻 (物理米; 排除自身 → k=2 取第二个)
    centers = np.column_stack([cols * cell_x_m, rows * cell_y_m])
    tree = cKDTree(centers)
    dist, _ = tree.query(centers, k=min(2, occupied_cells))
    nn = dist[:, 1] if occupied_cells > 1 else dist[:, 0]
    result["median_nearest_neighbor_m"] = float(np.median(nn))
    result["p90_nearest_neighbor_m"] = float(np.percentile(nn, 90))
    return result


# ──────────────────────────────────────────────
# 融合面板构造
# ──────────────────────────────────────────────
def bev_union_fill(bevs):
    """crop-first 对比用朴素"只补缺"并集: newest→oldest, 无质控.

    Returns: (z_min (G,G), occupied (G,G), point_count).
    """
    G = bevs[0].grid_res
    z_min = bevs[0].z_min.copy()
    occupied = bevs[0].occupied.copy()
    point_count = int(bevs[0].stats["output_points"])
    for b in bevs[1:]:
        point_count += int(b.stats["output_points"])
        new = b.occupied & ~occupied
        if new.any():
            z_min[new] = b.z_min[new]
            occupied |= new
    return z_min, occupied, point_count


def crop_first_world(level_roi: np.ndarray, pose: np.ndarray) -> np.ndarray:
    """已 level 的 ROI 点 → 世界坐标: 仅 yaw + 平移 (与 body_to_world 世界系一致,
    不重复水平化 / 不重复 z 翻转). 仅供"crop-first 禁止流程"对比."""
    pts = np.asarray(level_roi, dtype=np.float32)
    if len(pts) == 0:
        return pts
    out = pts @ _rot_z(float(pose[5])).T
    out[:, 0] += float(pose[0])
    out[:, 1] += float(pose[1])
    out[:, 2] -= float(pose[2])
    return out.astype(np.float32, copy=False)


def build_fusion_panels(scan: BagScan, target: dict, params: dict,
                        perc_cfg: dict, ground_z: float, grid_res: int,
                        window_size: int, camera: _DiagnosticCamera,
                        legacy_z_corr: float = 0.15,
                        legacy_resid: float = 0.12) -> dict:
    """D 节五面板: 单帧 / crop-first 10 / world-first 10 / world-first 30 /
    legacy 10 (当前 replay_window10 旧实现). Returns {name: (bev or dict)}."""
    idx = target["idx"]
    lo = max(0, idx - window_size + 1)
    frames = scan.clouds[lo:idx + 1]           # 含当前帧, 从旧到新
    cur = frames[-1]
    cur_pose = scan.pose_at(cur["stamp"])
    if cur_pose is None:
        cur_pose = cur["pose6"]
    half_x, half_y = dynamic_roi_half_extents(
        params, float(cur["pose6"][2]), ground_z, camera)
    bounds = roi_bounds(half_x, half_y)

    # 每帧: 插值位姿 + 完整世界点云
    win = []                                   # (stamp, pose, world_points)
    for f in frames:
        pose = scan.pose_at(f["stamp"])
        if pose is None:
            pose = f["pose6"]
        win.append((f["stamp"], pose, body_to_world(f["pts"], pose, perc_cfg)))

    # 当前帧 → 当前水平机体 → BEV (单帧面板 + 各融合面板公共 current_bev)
    cur_world = win[-1][2]
    cur_level = world_to_level_body(cur_world, cur_pose)
    current_bev = bev_roughness_downsample(
        cur_level, bounds, grid_res=grid_res)

    # 历史帧对齐到当前水平机体 (最新→最旧)
    hist_full = [world_to_level_body(w, cur_pose) for _, _, w in
                 reversed(win[:-1])]

    panels = {}
    panels["single_frame"] = {"bev": current_bev, "kind": "bev"}

    # crop-first 10: 每帧先按自身 ROI 裁剪再拼世界 (仅最近 10 帧)
    crop_bevs = []
    for f, (stamp, pose, _w) in zip(frames[-10:], win[-10:]):
        hx, hy = dynamic_roi_half_extents(
            params, float(f["pose6"][2]), ground_z, camera)
        roi, _ = _level_body_roi_silent(
            f["pts"], float(pose[3]), float(pose[4]), perc_cfg,
            half_x=hx, half_y=hy)
        pw = crop_first_world(roi, pose)
        pl = world_to_level_body(pw, cur_pose)
        crop_bevs.append(bev_roughness_downsample(pl, bounds, grid_res=grid_res))
    z_min_cf, occ_cf, pts_cf = bev_union_fill(crop_bevs)
    panels["crop_first_10"] = {
        "kind": "map", "z_min": z_min_cf, "occupied": occ_cf,
        "point_count": pts_cf,
        "n_history": len(crop_bevs) - 1}

    # world-first 10 / 30
    for n in (10, 30):
        hist = hist_full[:n - 1]
        fused = fuse_bev_world_first(
            current_bev, hist, bounds, grid_res,
            min_overlap_cells=20, max_z_correction_m=0.30,
            alignment_cell_m=0.30, max_cell_residual_m=0.25,
            min_cell_points=2, max_height_span_m=0.50)
        panels[f"world_first_{n}"] = {
            "kind": "fusion", "fusion": fused, "n_history": len(hist)}

    # 当前 replay_window10 实际融合结果 = legacy 10 帧 fuse_bev_gap_fill
    # (旧质控参数: 与第 1060 帧实验数据实测时一致的 0.15/0.12/20)
    hist_legacy = hist_full[:9]
    fused_legacy = fuse_bev_gap_fill(
        current_bev, hist_legacy, bounds, grid_res,
        min_overlap_cells=20, max_z_correction_m=legacy_z_corr,
        max_residual_m=legacy_resid)
    panels["legacy_10"] = {
        "kind": "fusion", "fusion": fused_legacy, "n_history": len(hist_legacy)}

    panels["_meta"] = {
        "bounds": bounds, "cur_pose": cur_pose, "cur_stamp": cur["stamp"],
        "current_bev": current_bev, "window": win,
    }
    return panels


def panel_zmin(panel) -> tuple[np.ndarray, np.ndarray]:
    """面板 → (z_min, occupied) 统一接口."""
    kind = panel["kind"]
    if kind == "bev":
        return panel["bev"].z_min, panel["bev"].occupied
    if kind == "map":
        return panel["z_min"], panel["occupied"]
    return panel["fusion"].bev.z_min, panel["fusion"].bev.occupied


def panel_stats(panel) -> dict:
    """面板 → 融合/占用统计 (写 stats.json)."""
    z_min, occupied = panel_zmin(panel)
    kind = panel["kind"]
    out = {"occupied_cells": int(occupied.sum()),
           "coverage_ratio": float(occupied.mean()),
           "n_history": int(panel.get("n_history", 0))}
    if kind == "bev":
        out["point_count"] = int(panel["bev"].stats["output_points"])
    elif kind == "map":
        out["point_count"] = int(panel.get("point_count", 0))
    else:
        fused = panel["fusion"]
        out.update({
            "point_count": int(fused.bev.stats["output_points"]),
            "added_cells": fused.added_cells,
            "duplicate_cells": fused.dup_skipped,
            "rejected_frames": fused.rejected_frames,
            "rejected_cells": fused.rejected_cells,
            "accepted_history_frames": sum(
                1 for st in fused.frame_stats if not st.rejected),
            "frame_stats": [{
                "frame_index": st.frame_idx,
                "overlap_cells": st.overlap_cells,
                "z_correction": round(float(st.z_correction), 6),
                "robust_residual": round(float(st.robust_residual), 6),
                "accepted_cells": st.added_cells,
                "rejected_cells": st.rejected_cells,
                "reject_reason": st.reject_reason,
            } for st in fused.frame_stats],
        })
    return out


# ──────────────────────────────────────────────
# 统计汇总
# ──────────────────────────────────────────────
def _build_stats(scan: BagScan, target: dict, raw_stamp, raw_pts, raw_delta,
                 level_roi: np.ndarray, panels: dict, grid_metrics: dict) -> dict:
    cur = target
    stats = {
        "bag_frame_index": cur["idx"],
        "cloud_header_seq": cur["seq"],
        "cloud_stamp": round(float(cur["stamp"]), 6),
        "raw_stamp": (round(float(raw_stamp), 6) if raw_stamp is not None else None),
        "raw_cloud_delta_ms": (round(float(raw_delta), 3) if raw_delta is not None else None),
        "raw_point_count": int(len(raw_pts)) if raw_pts is not None else 0,
        "deskewed_point_count": int(len(cur["pts"])),
        "roi_point_count": int(len(level_roi)),
        "pose_xyz": [round(float(v), 6) for v in cur["pose6"][:3]],
        "pose_rpy_deg": [round(float(np.degrees(v)), 6) for v in cur["pose6"][3:]],
        "frame_id": cur["frame_id"],
        "roi_size_m": [round(float(panels["_meta"]["bounds"]["x_max"])
                             - float(panels["_meta"]["bounds"]["x_min"]), 4),
                       round(float(panels["_meta"]["bounds"]["y_max"])
                             - float(panels["_meta"]["bounds"]["y_min"]), 4)],
        "grid_metrics": grid_metrics,
        "fusion_metrics": {name: panel_stats(p)
                           for name, p in panels.items()
                           if not name.startswith("_")},
    }
    return stats


# ──────────────────────────────────────────────
# 绘图 (matplotlib 惰性导入)
# ──────────────────────────────────────────────
def _fmt(n):
    return f"{n:,}"


def save_pipeline_figure(scan, target, raw_stamp, raw_pts, level_roi,
                         output_path: Path, show: bool):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(21, 6))
    raw_pts_arr = np.asarray(raw_pts) if raw_pts is not None else None
    sections = [
        ("A. Raw Livox (/livox/lidar)", raw_pts_arr, raw_stamp),
        ("B. Deskewed (/cloud_registered_body)", target["pts"], target["stamp"]),
        ("C. Level-body ROI (extrinsic+leveled)", level_roi, None),
    ]
    for ax, (title, pts, stamp) in zip(axes, sections):
        ax.set_title(title)
        if pts is None or len(pts) == 0:
            ax.text(0.5, 0.5, "n/a", ha="center", va="center", transform=ax.transAxes)
            ax.set_axis_off()
            continue
        p = np.asarray(pts)[:, :3]
        ax.scatter(p[:, 0], p[:, 1], c=p[:, 2], cmap="inferno", s=1.0)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        t = f"{_fmt(len(p))} pts"
        if stamp is not None:
            t += f"  t={float(stamp):.3f}s"
        ax.set_title(f"{title}\n{t}")
    fig.tight_layout()
    fig.savefig(output_path, dpi=110)
    logger.info("[PNG] %s", output_path)
    if show:
        plt.show()
    plt.close(fig)


def save_occupancy_figure(single_metrics: dict, world_metrics: dict,
                          z_maps: dict, output_path: Path, show: bool,
                          bounds: dict, model_grids: tuple = (64, 96, 128)):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows, cols = 2, len(model_grids)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 6, 12),
                             constrained_layout=True)
    all_vmin = all_vmax = None
    for z_min in z_maps["single"].values():
        v = z_min[np.isfinite(z_min)]
        if v.size:
            all_vmin = min(all_vmin, float(v.min())) if all_vmin is not None else float(v.min())
            all_vmax = max(all_vmax, float(v.max())) if all_vmax is not None else float(v.max())
    for z_min in z_maps["world"].values():
        v = z_min[np.isfinite(z_min)]
        if v.size:
            all_vmin = min(all_vmin, float(v.min())) if all_vmin is not None else float(v.min())
            all_vmax = max(all_vmax, float(v.max())) if all_vmax is not None else float(v.max())
    if all_vmin is None:
        all_vmin, all_vmax = 0.0, 30.0

    def _draw(ax, z_min, metrics, title):
        masked = np.ma.masked_invalid(z_min)
        im = ax.imshow(masked, origin="lower", cmap="inferno",
                       vmin=all_vmin, vmax=all_vmax,
                       extent=[float(bounds["x_min"]), float(bounds["x_max"]),
                               float(bounds["y_min"]), float(bounds["y_max"])])
        m = metrics
        ax.set_title(f"{title}\nocc={m['occupied_cells']} cov={m['coverage_ratio']:.3f}\n"
                     f"hull_holes={m['convex_hull_inside_holes']['count']} "
                     f"max_hole={m['max_hole_radius_m']:.2f}m\n"
                     f"nn_med={m['median_nearest_neighbor_m']:.3f}m "
                     f"nn_p90={m['p90_nearest_neighbor_m']:.3f}m")
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        return im

    im = None
    for j, g in enumerate(model_grids):
        im = _draw(axes[0, j], z_maps["single"][g], single_metrics[g], f"Single ROI {g}²")
        im = _draw(axes[1, j], z_maps["world"][g], world_metrics[g], f"World-first 30 {g}²")
    fig.colorbar(im, ax=axes, shrink=0.8, label="z_min (m)")
    fig.suptitle("BEV occupancy at 64/96/128 (unified z scale)", fontsize=14)
    fig.savefig(output_path, dpi=110)
    logger.info("[PNG] %s", output_path)
    if show:
        plt.show()
    plt.close(fig)


def save_fusion_figure(panels: dict, output_path: Path, show: bool,
                       grid_res: int):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = ["single_frame", "crop_first_10", "world_first_10",
             "world_first_30", "legacy_10"]
    labels = {
        "single_frame": "1. Single deskewed ROI",
        "crop_first_10": "2. 10-frame crop-first",
        "world_first_10": "3. 10-frame world-first",
        "world_first_30": "4. 30-frame world-first",
        "legacy_10": "5. replay_window10 legacy 10",
    }
    fig, axes = plt.subplots(1, len(names), figsize=(5 * len(names), 5.5))
    all_vmin = all_vmax = None
    zs = {}
    for name in names:
        z, _ = panel_zmin(panels[name])
        v = z[np.isfinite(z)]
        if v.size:
            all_vmin = min(all_vmin, float(v.min())) if all_vmin is not None else float(v.min())
            all_vmax = max(all_vmax, float(v.max())) if all_vmax is not None else float(v.max())
        zs[name] = z
    if all_vmin is None:
        all_vmin, all_vmax = 0.0, 30.0
    bounds = panels["_meta"]["bounds"]
    for name, ax in zip(names, axes):
        z = zs[name]
        st = panel_stats(panels[name])
        ax.imshow(np.ma.masked_invalid(z), origin="lower", cmap="inferno",
                  vmin=all_vmin, vmax=all_vmax,
                  extent=[float(bounds["x_min"]), float(bounds["x_max"]),
                          float(bounds["y_min"]), float(bounds["y_max"])])
        ax.set_title(f"{labels[name]}\nocc={st['occupied_cells']} "
                     f"cov={st['coverage_ratio']:.3f} "
                     f"hist={st['n_history']}")
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
    fig.suptitle(f"Multi-frame fusion comparison (grid={grid_res}, "
                 f"unified z scale)", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(output_path, dpi=110)
    logger.info("[PNG] %s", output_path)
    if show:
        plt.show()
    plt.close(fig)


# ──────────────────────────────────────────────
# 主流程
# ──────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="第 1060 帧原始/去畸变/ROI 点云对比诊断 (无 torch 依赖)")
    parser.add_argument("--bag", type=str, required=True, help="rosbag 路径")
    parser.add_argument("--config", type=str, required=True,
                        help="experiment_config_snapshot.yaml 路径")
    parser.add_argument("--frame-index", type=int, default=1060,
                        help="bag 中第 N 个 cloud 消息 (0 基, 默认 1060)")
    parser.add_argument("--cloud-seq", type=int, default=None,
                        help="PointCloud2 header.seq 定位 (优先于 --frame-index)")
    parser.add_argument("--pose-topic", type=str,
                        default="/mavros/local_position/odom", help="位姿话题")
    parser.add_argument("--raw-topic", type=str, default="/livox/lidar",
                        help="原始 Livox 话题 (仅诊断参考)")
    parser.add_argument("--output-dir", type=str, default="diagnose_frame",
                        help="输出目录 (PNG + JSON)")
    parser.add_argument("--window-size", type=int, default=30,
                        help="world-first 融合窗口帧数 (默认 30)")
    parser.add_argument("--bev-grid-res", type=int, default=64,
                        help="融合/BEV 诊断网格分辨率 (默认 64)")
    parser.add_argument("--show", action="store_true",
                        help="显示 matplotlib 窗口 (默认只保存)")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    t_start = time.perf_counter()
    cfg = load_config(args.config)
    params = perception_params(cfg)
    perc_cfg = params["perc_cfg"]
    camera = _DiagnosticCamera(cfg)

    # ── 帧定位 (cloud-seq 优先; bag 序号与 header.seq 明确区分) ──
    scan = BagScan(args.bag, "/cloud_registered_body", args.pose_topic,
                   args.raw_topic, params["max_sync_ms"])
    if args.cloud_seq is not None:
        logger.info("[Locate] by --cloud-seq=%d (优先)", args.cloud_seq)
    else:
        logger.info("[Locate] by --frame-index=%d (bag 第 N 个 cloud 消息)",
                    args.frame_index)
    target_idx = scan.scan(target_seq=args.cloud_seq,
                           target_idx=args.frame_index)
    target = scan.clouds[target_idx]
    if target["pose6"] is None:
        raise RuntimeError(f"目标帧 {target_idx} 无可用位姿 (bag 无 {args.pose_topic})")
    logger.info("[Locate] bag_frame_index=%d cloud_header_seq=%d stamp=%.6f "
                "n=%d frame_id=%s",
                target["idx"], target["seq"], target["stamp"],
                len(target["pts"]), target["frame_id"])

    # ground_z: 与 replay 一致, 首帧位姿高度
    ground_z = float(scan.clouds[0]["pose6"][2])

    # ── A/B: 原始 + 去畸变 ──
    raw_stamp, raw_pts, raw_delta = scan.nearest_raw(target["stamp"])

    # ── C: 去畸变 → 外参 + 水平化 + z-down → 动态 ROI ──
    half_x, half_y = dynamic_roi_half_extents(
        params, float(target["pose6"][2]), ground_z, camera)
    level_roi, roi_stats = _level_body_roi_silent(
        target["pts"], float(target["pose6"][3]), float(target["pose6"][4]),
        perc_cfg, half_x=half_x, half_y=half_y)

    # ── D: 五面板融合对比 (统一 bounds / 统一色标) ──
    panels = build_fusion_panels(
        scan, target, params, perc_cfg, ground_z, int(args.bev_grid_res),
        int(args.window_size), camera)
    bounds = panels["_meta"]["bounds"]

    # ── E: 64/96/128 占用率 (单帧 ROI + 30 帧 world-first) ──
    grid_metrics = {"single_roi": {}, "world_first_30": {}}
    z_maps = {"single": {}, "world": {}}
    for g in (64, 96, 128):
        gx = (float(bounds["x_max"]) - float(bounds["x_min"])) / max(g - 1, 1)
        gy = (float(bounds["y_max"]) - float(bounds["y_min"])) / max(g - 1, 1)
        bev_s = bev_roughness_downsample(level_roi, bounds, grid_res=g)
        grid_metrics["single_roi"][g] = grid_occupancy_metrics(
            bev_s.occupied, gx, gy)
        z_maps["single"][g] = bev_s.z_min
        w30 = panels["world_first_30"]["fusion"].bev
        wz = bev_roughness_downsample(
            w30.points, bounds, grid_res=g)
        grid_metrics["world_first_30"][g] = grid_occupancy_metrics(
            wz.occupied, gx, gy)
        z_maps["world"][g] = wz.z_min

    # ── 统计 JSON ──
    stats = _build_stats(scan, target, raw_stamp, raw_pts, raw_delta,
                         level_roi, panels, grid_metrics)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    idx = target["idx"]
    json_path = out_dir / f"frame_{idx}_stats.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    logger.info("[JSON] %s", json_path)

    # ── 控制台汇总 ──
    print("\n=== 第 %d 帧诊断 (seq=%d) ===" % (idx, target["seq"]))
    print(f"raw=/livox/lidar: {stats['raw_point_count']} pts, "
          f"t={stats['raw_stamp']}, Δ={stats['raw_cloud_delta_ms']} ms")
    print(f"deskewed=/cloud_registered_body: {stats['deskewed_point_count']} pts, "
          f"t={stats['cloud_stamp']}, frame_id={target['frame_id']}")
    print(f"level-body ROI: {stats['roi_point_count']} pts, "
          f"roi_size={stats['roi_size_m']} m")
    print(f"pose xyz={stats['pose_xyz']} rpy_deg={stats['pose_rpy_deg']}")
    for name, m in stats["fusion_metrics"].items():
        rej = m.get("rejected_frames", "-")
        print(f"  {name:16s} occ={m['occupied_cells']:6d} "
              f"cov={m['coverage_ratio']:.3f} hist={m['n_history']:2d} "
              f"rej_frames={rej} rej_cells={m.get('rejected_cells', '-')} "
              f"added={m.get('added_cells', '-')}")

    # ── PNG ──
    try:
        save_pipeline_figure(scan, target, raw_stamp, raw_pts, level_roi,
                             out_dir / f"frame_{idx}_pointcloud_pipeline.png",
                             args.show)
        save_fusion_figure(panels,
                           out_dir / f"frame_{idx}_fusion_compare.png",
                           args.show, int(args.bev_grid_res))
        save_occupancy_figure(grid_metrics["single_roi"],
                              grid_metrics["world_first_30"], z_maps,
                              out_dir / f"frame_{idx}_bev_occupancy.png",
                              args.show, bounds)
    except Exception as exc:
        logger.warning("[PNG] 绘图失败 (仅 JSON 已保存): %s", exc)
    logger.info("[Done] %.1fs elapsed", time.perf_counter() - t_start)
    return stats


if __name__ == "__main__":
    main()
