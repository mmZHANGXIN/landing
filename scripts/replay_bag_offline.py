#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Orin Landing — 离线 rosbag 回放感知 + DRL 推理管线
====================================================
与 pipeline.py 高度一致的感知决策管线，从 rosbag 读取 FAST-LIO 去畸变点云和位姿，
跑完整的 HALSS 贝叶斯语义 + training-camera 深度投影 + NN-fill + ONNX PPO 推理，
实时 OpenCV 可视化 + Matplotlib 3D 去畸变点云窗口，并打印每帧动作。

支持:
  - 室内 world-cloud 模式: /ali_cloud + /ali_odom
  - 室外 body-cloud 模式: /cloud_registered_body + /mavros/local_position/odom

用法:
  source /opt/ros/noetic/setup.bash
  python scripts/replay_bag_offline.py --bag experiments/runs/20260807_162946_orin_landing/input.bag --config experiments/runs/20260807_162946_orin_landing/experiment_config_snapshot.yaml --onnx-model weights/ppo2_policy.onnx
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
import cv2

# 将项目根目录添加到 sys.path (与 test_live_nocontrol.py 一致)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("ReplayBag")

# ──────────────────────────────────────────────
# 解析 PointCloud2 → numpy (与 pipeline.py 一致)
# ──────────────────────────────────────────────
def _pc2_to_numpy(msg) -> np.ndarray:
    """sensor_msgs/PointCloud2 → (N, 3) float32, 滤除 NaN."""
    field_offsets = {f.name: f.offset for f in msg.fields}
    if not all(k in field_offsets for k in ("x", "y", "z")):
        return np.empty((0, 3), dtype=np.float32)
    n = msg.width * msg.height
    endian = ">f4" if msg.is_bigendian else "<f4"
    dtype = np.dtype({
        "names": ["x", "y", "z"],
        "formats": [endian, endian, endian],
        "offsets": [field_offsets["x"], field_offsets["y"], field_offsets["z"]],
        "itemsize": msg.point_step,
    })
    arr = np.frombuffer(msg.data, dtype=dtype, count=n)
    pts = np.column_stack((arr["x"], arr["y"], arr["z"])).astype(np.float32, copy=False)
    return pts[np.isfinite(pts).all(axis=1)]


def _quat_to_euler(x, y, z, w):
    """四元数 → 欧拉角 (roll, pitch, yaw) rad."""
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)
    sinp = 2.0 * (w * y - z * x)
    pitch = math.asin(max(-1.0, min(1.0, sinp))) if abs(sinp) < 1 else math.copysign(math.pi / 2, sinp)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return roll, pitch, yaw


def _stamp_to_sec(stamp) -> float:
    return float(stamp.secs) + float(stamp.nsecs) * 1e-9


# ──────────────────────────────────────────────
# Config 加载 (与 pipeline.py 完全一致)
# ──────────────────────────────────────────────
def _load_config(path: str) -> dict:
    config_path = Path(path).resolve()
    if not config_path.is_file() and config_path.parent.name != "runs":
        # Orin 上实验数据在 experiments/runs/<name>/, 开发机在 experiments/<name>/
        # 传入路径不存在时自动尝试 runs/ 变体, 两种机器布局均直接可运行
        alt = (config_path.parents[1] / "runs" / config_path.parent.name / config_path.name)
        if alt.is_file():
            logger.warning("[Config] %s 不存在, 改用 %s", config_path, alt)
            config_path = alt
    import yaml
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    parent = cfg.pop("extends", None)
    if parent:
        parent_path = Path(parent)
        if not parent_path.is_absolute():
            parent_path = config_path.parent / parent_path
        base = _load_config(str(parent_path))
        return _merge_config_overrides(base, cfg)
    return cfg


def _merge_config_overrides(cfg: dict, overrides: dict) -> dict:
    for section, values in overrides.items():
        if isinstance(values, dict) and isinstance(cfg.get(section), dict):
            cfg.setdefault(section, {})
            _merge_config_overrides(cfg[section], values)
        else:
            cfg[section] = values
    return cfg


# ──────────────────────────────────────────────
# 与 pipeline.py 共享的辅助函数
# ──────────────────────────────────────────────
CLASS_TO_GRAY = {
    -1: 0, 0: 10, 1: 30, 2: 60, 3: 70, 4: 20,
     5: 40, 6: 80, 7: 90, 8: 50, 9: 250,
}


def make_binary_semantic_vis(sem_map, safe_id=1, danger_id=9):
    sem_vis = np.full(sem_map.shape, 128, dtype=np.uint8)
    sem_vis[sem_map == safe_id] = 255
    sem_vis[sem_map == danger_id] = 0
    return sem_vis


def _limited_xy_velocity(dx, dy, max_speed_mps, kp_s=1.0):
    distance = math.hypot(float(dx), float(dy))
    if distance <= 1e-9:
        return 0.0, 0.0
    speed = min(float(max_speed_mps), float(kp_s) * distance)
    scale = speed / distance
    return float(dx) * scale, float(dy) * scale


def _limited_axis_velocity(error, max_speed_mps, kp_s=1.0):
    return max(-float(max_speed_mps), min(float(max_speed_mps), float(kp_s) * float(error)))


def _top_probs(probs, action_names=None, k=3):
    if probs is None:
        return "p=n/a"
    pairs = sorted(enumerate(probs), key=lambda item: item[1], reverse=True)[:k]
    if action_names is None:
        return "p=" + ",".join(f"{idx}:{prob:.2f}" for idx, prob in pairs)
    return "p=" + ",".join(f"{idx}:{action_names[idx]}:{prob:.2f}" for idx, prob in pairs)


# ──────────────────────────────────────────────
# BEV 深度投影 (与 pipeline.py 共享)
# ──────────────────────────────────────────────
def project_bev_depth(pts_body, grid_res=64, out_size=128, max_range=30.0,
                      half_x=5.0, half_y=5.0):
    empty = np.full((out_size, out_size), max_range, dtype=np.float32)
    bounds = {"x_min": -half_x, "x_max": half_x,
              "y_min": -half_y, "y_max": half_y}
    if pts_body is None or len(pts_body) == 0:
        return empty, bounds
    pts = np.asarray(pts_body, dtype=np.float32)
    pts = pts[np.isfinite(pts).all(axis=1)]
    if len(pts) == 0:
        return empty, bounds
    z_all = pts[:, 2]
    pts = pts[(z_all > 0.01) & (z_all < max_range)]
    if len(pts) == 0:
        return empty, bounds
    x_span = 2.0 * half_x
    y_span = 2.0 * half_y
    x_min, y_min = -half_x, -half_y
    col_idx = np.rint((pts[:, 0] - x_min) / x_span * (grid_res - 1)).astype(np.int32)
    row_unflipped = np.rint((pts[:, 1] - y_min) / y_span * (grid_res - 1)).astype(np.int32)
    row_idx = (grid_res - 1) - row_unflipped
    valid = (row_idx >= 0) & (row_idx < grid_res) & (col_idx >= 0) & (col_idx < grid_res)
    row_idx, col_idx = row_idx[valid], col_idx[valid]
    z_vals = pts[valid, 2]
    if len(z_vals) == 0:
        return empty, bounds
    accum = np.zeros((grid_res, grid_res), dtype=np.float32)
    count = np.zeros((grid_res, grid_res), dtype=np.int32)
    np.add.at(accum, (row_idx, col_idx), z_vals)
    np.add.at(count, (row_idx, col_idx), 1)
    mask = count > 0
    grid = np.full((grid_res, grid_res), np.nan, dtype=np.float32)
    grid[mask] = accum[mask] / count[mask]
    if grid_res != out_size:
        grid = cv2.resize(grid, (out_size, out_size), interpolation=cv2.INTER_NEAREST)
        grid[np.isnan(grid) | (grid <= 0)] = max_range
    else:
        grid = np.where(np.isnan(grid), max_range, grid)
    return grid.astype(np.float32), bounds


def render_sparse_depth(sparse_depth, valid_mask, dmax, min_valid=5, median_ksize=5):
    if valid_mask.sum() < min_valid:
        return np.where(valid_mask, sparse_depth, dmax).astype(np.float32)
    invalid = ~valid_mask
    _, labels = cv2.distanceTransformWithLabels(
        invalid.astype(np.uint8),
        distanceType=cv2.DIST_L2, maskSize=5,
        labelType=cv2.DIST_LABEL_PIXEL,
    )
    valid_coords = np.column_stack(np.where(valid_mask))
    label_vals = labels[invalid]
    nearest_idx = np.clip(label_vals - 1, 0, len(valid_coords) - 1)
    filled = sparse_depth.copy()
    filled[invalid] = sparse_depth[
        valid_coords[nearest_idx, 0],
        valid_coords[nearest_idx, 1],
    ]
    if median_ksize >= 3:
        smoothed = cv2.medianBlur(filled.astype(np.float32), median_ksize)
    else:
        smoothed = filled.astype(np.float32)
    rendered = np.where(valid_mask, sparse_depth, smoothed)
    return np.clip(rendered, 0.0, dmax).astype(np.float32)


def render_depth_fixed_gray(depth_m, vmax_m=30.0):
    """主深度图固定量程显示: 0 m→黑, vmax m→白 (纯显示, 不改输入).

    固定映射 norm = clip(depth / vmax, 0, 1): 相同距离跨帧始终得到相同灰度,
    30 m 填充值恒为纯白, 量程不随帧变化, 承担绝对距离判断。
    """
    depth = np.nan_to_num(
        np.asarray(depth_m, dtype=np.float32),
        nan=vmax_m, posinf=vmax_m, neginf=0.0,
    )
    norm = np.clip(depth / max(float(vmax_m), 1e-6), 0.0, 1.0)
    return cv2.cvtColor(np.round(norm * 255.0).astype(np.uint8), cv2.COLOR_GRAY2BGR)


def render_depth_local_contrast(
    depth_m, semantic_valid_mask,
    vmax_m=30.0,
    pct_low=2.0, pct_high=98.0, pad_frac=0.05,
    min_span_m=0.5, min_valid=32,
):
    """局部深度对比度窗口: 逐帧自动量程, 近黑远白 (纯显示, 不改输入).

    有效统计区域 = semantic_valid_mask & (0 < depth < vmax), 排除外围 unknown
    和 NN-fill 填充的 dmax 格。量程取有效深度 pct_low/pct_high 百分位并外扩
    pad_frac (默认 5%), 跨度不足 min_span_m 时围绕中值扩展, 有效像素不足
    min_valid 时回退固定 [0, vmax_m] 量程。映射: near→0(黑), far→255(白),
    不做边缘检测, 不叠加彩色轮廓。输出 BGR uint8。

    Returns: (bgr_image, info) — info: auto_range/near_m/far_m/n_valid
    """
    depth = np.nan_to_num(
        np.asarray(depth_m, dtype=np.float32),
        nan=vmax_m, posinf=vmax_m, neginf=0.0,
    )
    mask = np.asarray(semantic_valid_mask, dtype=bool)
    if mask.shape != depth.shape:
        mask = cv2.resize(mask.astype(np.uint8),
                          (depth.shape[1], depth.shape[0]),
                          interpolation=cv2.INTER_NEAREST).astype(bool)
    stat_mask = mask & np.isfinite(depth) & (depth > 0.0) & (depth < vmax_m)
    n_valid = int(np.count_nonzero(stat_mask))
    if n_valid >= min_valid:
        low, high = np.percentile(depth[stat_mask], [pct_low, pct_high])
        pad = (float(high) - float(low)) * pad_frac
        near_used = float(low) - pad
        far_used = float(high) + pad
        if far_used - near_used < min_span_m:
            mid = 0.5 * (float(low) + float(high))
            near_used = mid - 0.5 * min_span_m
            far_used = mid + 0.5 * min_span_m
        auto_range = True
    else:
        near_used, far_used = 0.0, float(vmax_m)
        auto_range = False
    norm = np.clip((depth - near_used) / max(far_used - near_used, 1e-6), 0.0, 1.0)
    bgr = cv2.cvtColor(np.round(norm * 255.0).astype(np.uint8), cv2.COLOR_GRAY2BGR)

    info = {
        "auto_range": auto_range,
        "near_m": float(near_used),
        "far_m": float(far_used),
        "n_valid": n_valid,
    }
    return bgr, info


# ──────────────────────────────────────────────
# ONNX DRL (与 pipeline.py 完全一致)
# ──────────────────────────────────────────────
class ONNXDRL:
    def __init__(self, onnx_path: str, obs_h=128, obs_w=128, dmax=30.0,
                 depth_norm_mode="raw_meters_graph_scaled",
                 semantic_norm_mode="raw_gray_graph_scaled"):
        import onnxruntime as ort
        if not Path(onnx_path).is_file():
            raise FileNotFoundError(f"ONNX model not found: {onnx_path}")
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        opts.inter_op_num_threads = 1
        self.session = ort.InferenceSession(onnx_path, opts)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        in_shape = self.session.get_inputs()[0].shape
        self.layout = "chw" if len(in_shape) == 4 and in_shape[1] in (2, 3) else "hwc"
        self.obs_h, self.obs_w = obs_h, obs_w
        self.dmax = dmax
        if depth_norm_mode != "raw_meters_graph_scaled":
            raise ValueError("ONNX graph already contains input/truediv")
        if semantic_norm_mode != "raw_gray_graph_scaled":
            raise ValueError("ONNX graph already contains input/truediv")
        dummy = np.zeros((1, obs_h, obs_w, 2), dtype=np.float32)
        self._forward(dummy)
        logger.info("[ONNX] model=%s layout=%s warmup=OK", onnx_path, self.layout)

    def _forward(self, obs_raw):
        if self.layout == "chw":
            inp = np.transpose(obs_raw, (0, 3, 1, 2)).astype(np.float32)
        else:
            inp = obs_raw.astype(np.float32)
        return self.session.run([self.output_name], {self.input_name: inp})[0]

    def predict(self, depth_map, sem_map):
        depth_clipped = np.clip(
            np.nan_to_num(depth_map, nan=self.dmax, posinf=self.dmax, neginf=0.0),
            0.0, self.dmax,
        )
        depth_ch = depth_clipped.astype(np.float32)
        sem_int = np.clip(sem_map, -1, 9).astype(np.int16)
        sem_ch = np.zeros_like(sem_int, dtype=np.float32)
        for class_id, gray_val in CLASS_TO_GRAY.items():
            sem_ch[sem_int == class_id] = float(gray_val)
        obs = np.expand_dims(np.stack([depth_ch, sem_ch], axis=-1), axis=0)
        logits = self._forward(obs)[0]
        logits = np.asarray(logits, dtype=np.float32)
        exps = np.exp(logits - float(np.max(logits)))
        probs = exps / max(float(np.sum(exps)), 1e-12)
        action = int(np.argmax(logits))
        depth_after = depth_ch / 255.0
        sem_after = sem_ch / 255.0
        info = {
            "depth_raw_median": float(np.median(depth_map)),
            "depth_input_mean": float(depth_ch.mean()),
            "depth_input_min": float(depth_ch.min()),
            "depth_input_max": float(depth_ch.max()),
            "sem_input_mean": float(sem_ch.mean()),
            "sem_input_min": float(sem_ch.min()),
            "sem_input_max": float(sem_ch.max()),
            "sem_input_unique": [round(float(x), 6) for x in sorted(np.unique(sem_ch))],
            "obs_raw_min": float(obs.min()),
            "obs_raw_max": float(obs.max()),
            "softmax_probs": probs.astype(float).tolist(),
            "action_probs": probs.astype(float).tolist(),
            "confidence": float(np.max(probs)),
            "depth_norm_min": float(depth_after.min()),
            "depth_norm_mean": float(depth_after.mean()),
            "depth_norm_max": float(depth_after.max()),
            "sem_norm_min": float(sem_after.min()),
            "sem_norm_mean": float(sem_after.mean()),
            "sem_norm_max": float(sem_after.max()),
            "logits": logits.astype(float).tolist(),
        }
        return action, info


# ──────────────────────────────────────────────
# OpenCV 可视化 (与 pipeline.py RealtimeVisualizer 完全对齐)
# ──────────────────────────────────────────────
class ReplayVisualizer:
    """四窗口: 二值安全语义图 + 主深度图 (固定 0~30m 近黑远白) +
    局部深度对比度窗口 (逐帧自动量程) + 3D 去畸变点云.

    主深度图固定量程承担绝对距离判断 (跨帧灰度恒定), 局部对比度窗口仅用于
    观察当前帧细微深度变化, 均为纯灰度无彩色叠加. 3D 点云窗口使用 Matplotlib
    (惰性导入, 优先 TkAgg), 显示每个成功处理帧实际送入 HALSS 的 halss_pts ——
    已完成外参变换、roll/pitch leveling、ROI 过滤和高度方向转换,
    level-body 坐标 x=forward / y=lateral / z=down. 点按 z 值着色 (inferno),
    与语义图、深度图在同一次 update() 中同步刷新, 通过 draw_idle()/
    flush_events() 保持窗口可旋转缩放且不阻塞回放.
    """

    def __init__(self, dmax=30.0, depth_vmax_m=30.0, display_width=300,
                 depth_display_mode="fixed_gray", depth_near_m=0.5,
                 depth_display_pct_low=2.0, depth_display_pct_high=98.0,
                 depth_display_min_span_m=0.5, depth_display_min_valid=32,
                 show_pointcloud=True, show_local_depth=True):
        self.dmax = dmax
        self.depth_vmax_m = depth_vmax_m
        mode = str(depth_display_mode).lower()
        self.depth_display_mode = (
            "fixed_gray" if mode == "fixed_gray" else "legacy_inferno")
        self.depth_near_m = max(0.01, float(depth_near_m))
        self.depth_display_pct_low = float(depth_display_pct_low)
        self.depth_display_pct_high = float(depth_display_pct_high)
        self.depth_display_min_span_m = float(depth_display_min_span_m)
        self.depth_display_min_valid = int(depth_display_min_valid)
        self.display_width = int(display_width)
        self._windows_ready = False
        # 局部深度对比度窗口 (逐帧自动量程) 开关, --no-local-depth 可关闭
        self.show_local_depth = bool(show_local_depth)
        # 深度图显示高基于源图高宽比，不固定为正方形
        self._disp_h = 300
        # Matplotlib 3D 点云 (惰性导入; --no-display 时不构造本类, 故不会导入)
        self.show_pointcloud = bool(show_pointcloud)
        self._mpl = None          # matplotlib.pyplot 模块
        self._pc_fig = None       # 3D 点云 figure / axes / scatter
        self._pc_ax = None
        self._pc_scatter = None
        if self.show_pointcloud:
            try:
                self._mpl = self._import_matplotlib()
            except Exception as exc:
                logger.warning(
                    "[Vis] Matplotlib unavailable (%s); 3D point cloud window "
                    "disabled, OpenCV windows continue.", exc)
                self._mpl = None

    def _import_matplotlib(self):
        """惰性导入 matplotlib; 优先 TkAgg (与 test_live_nocontrol.py 一致),
        失败时回退到默认后端 (MacOSX/Qt/Agg 等), 均失败则抛异常由调用方降级."""
        try:
            import matplotlib
            matplotlib.use("TkAgg")
            import matplotlib.pyplot as plt
            import mpl_toolkits.mplot3d  # 注册 3d projection (版本不匹配时易缺)
            return plt
        except Exception as exc:
            # 常见于 conda python 缺 tkinter: 回退默认后端后窗口不会弹出,
            # 必须明确警告, 否则无窗口且无任何报错, 极难排查
            logger.warning(
                "[Vis] TkAgg 不可用 (%s), 回退默认后端 — 3D 窗口可能无法显示. "
                "建议用系统 python3 (python3-tk / X11 转发).", exc)
            try:
                import matplotlib.pyplot as plt  # 默认后端
                import mpl_toolkits.mplot3d
                return plt
            except Exception:
                raise

    def _init_pointcloud_window(self):
        """创建 3D 点云窗口 (标题 4.Deskewed Point Cloud, level-body 坐标轴).

        所有资源先在局部变量中创建成功, 再一次性挂载到实例属性, 避免
        半初始化状态 (fig 已建但 ax 缺失时 update 会崩溃).
        """
        self._mpl.ion()
        fig = self._mpl.figure(num="4.Deskewed Point Cloud", figsize=(6, 6))
        ax = fig.add_subplot(111, projection="3d")
        ax.set_xlabel("x forward (m)")
        ax.set_ylabel("y lateral (m)")
        ax.set_zlabel("z down (m)")
        ax.set_title("0 pts")
        fig.canvas.draw_idle()
        fig.canvas.flush_events()
        # 全部成功后才挂载
        self._pc_fig, self._pc_ax = fig, ax

    def _update_pointcloud(self, point_cloud, cloud_stamp):
        """刷新 3D 点云: 全量显示 halss_pts (不下采样), 按 z 值着色.

        空点云/None 时清空散点集合并保留窗口, 不影响语义/深度窗口.
        只读输入数组, 不改写坐标.
        """
        if self._mpl is None:
            return
        if self._pc_fig is None:
            try:
                self._init_pointcloud_window()
            except Exception as exc:
                # 窗口初始化失败 (如 matplotlib/numpy 版本不匹配导致
                # 3d projection 缺失): 禁用窗口降级, 不向上抛异常
                logger.warning("[Vis] 3D window init failed (%s); "
                               "point cloud window disabled.", exc)
                self._mpl = None
                return
        if self._pc_ax is None:
            return
        # 移除上一帧散点 (空点云时仅清空内容)
        if self._pc_scatter is not None:
            self._pc_scatter.remove()
            self._pc_scatter = None
        n = 0
        if point_cloud is not None:
            pts = np.asarray(point_cloud, dtype=np.float32)
            if pts.ndim == 2 and pts.shape[1] >= 3 and len(pts) > 0:
                pts = pts[:, :3]
                n = len(pts)
                self._pc_scatter = self._pc_ax.scatter(
                    pts[:, 0], pts[:, 1], pts[:, 2],
                    c=pts[:, 2], cmap="inferno", s=1.0, depthshade=False,
                )
        title = f"{n} pts"
        if cloud_stamp is not None:
            title += f"  ·  t={float(cloud_stamp):.3f}s"
        self._pc_ax.set_title(title)
        try:
            self._pc_fig.canvas.draw_idle()
            self._pc_fig.canvas.flush_events()
        except Exception as exc:
            # 窗口被关闭 (X 按钮) 或 X11 连接断开 → Tk 应用已销毁:
            # 禁用窗口降级, 避免每个回调都抛 TclError 刷屏
            logger.warning("[Vis] Window closed or unavailable (%s); "
                           "point cloud window disabled.", exc)
            self._mpl = None

    def _init_windows(self):
        cv2.namedWindow("1.Binary Semantic Map", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("1.Binary Semantic Map", self.display_width, self._disp_h)
        cv2.moveWindow("1.Binary Semantic Map", 20, 50)

        cv2.namedWindow("2.Depth Map (0-30m)", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("2.Depth Map (0-30m)", self.display_width, self._disp_h)
        cv2.moveWindow("2.Depth Map (0-30m)", 20 + self.display_width + 10, 50)

        if self.show_local_depth:
            cv2.namedWindow("3.Local Depth Contrast", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("3.Local Depth Contrast", self.display_width, self._disp_h)
            cv2.moveWindow("3.Local Depth Contrast",
                           20 + 2 * (self.display_width + 10), 50)

        self._windows_ready = True

    def update(self, depth_map, sem_map, binary_semantic_vis,
               point_cloud=None, cloud_stamp=None, semantic_valid_mask=None):
        """更新四窗口。语义图自适应裁剪填满窗口; 主深度图固定 0~30m 近黑远白
        (跨帧灰度恒定), 局部深度对比度窗口逐帧自动量程, 均为纯灰度.

        DRL 输入始终是 128x128 完整图像（含外围 unknown 区域）。
        可视化只显示有效语义区域的裁剪放大视图。
        point_cloud/cloud_stamp: 本帧实际送入 HALSS 的 halss_pts 及其 ROS 时间戳,
        与语义图、深度图来自同一帧; 空点云或 None 时清空 3D 窗口内容.
        semantic_valid_mask: 本帧有效感知区域掩码 (False=外围 unknown/填充),
        供局部对比度窗口自动量程使用; None 时视为全图有效.
        """
        h, w = depth_map.shape[:2]
        disp_w = self.display_width
        disp_h = self._disp_h

        if not self._windows_ready:
            self._init_windows()

        # ---- 1. 二值语义图: 自适应裁剪非灰色区域后 resize 填满窗口 ----
        if binary_semantic_vis is not None:
            sem_disp = binary_semantic_vis
            if sem_disp.ndim == 3:
                sem_disp = cv2.cvtColor(sem_disp.astype(np.uint8), cv2.COLOR_BGR2GRAY)
        else:
            sem_disp = make_binary_semantic_vis(sem_map)

        # 自适应裁剪: 找到非灰色 (≠128) 像素的包围盒，裁掉纯灰色边框
        valid_mask = sem_disp != 128
        if valid_mask.any():
            rows = np.any(valid_mask, axis=1)
            cols = np.any(valid_mask, axis=0)
            r_min, r_max = np.where(rows)[0][[0, -1]]
            c_min, c_max = np.where(cols)[0][[0, -1]]
            # 加一点 padding 避免边缘贴死
            pad = 2
            r_min = max(0, r_min - pad)
            r_max = min(h - 1, r_max + pad)
            c_min = max(0, c_min - pad)
            c_max = min(w - 1, c_max + pad)
            sem_cropped = sem_disp[r_min:r_max + 1, c_min:c_max + 1]
        else:
            sem_cropped = sem_disp  # 全是灰色时保留原图

        sem_bgr = cv2.cvtColor(sem_cropped.astype(np.uint8), cv2.COLOR_GRAY2BGR)
        sem_bgr = cv2.resize(sem_bgr, (disp_w, disp_h), interpolation=cv2.INTER_NEAREST)
        cv2.imshow("1.Binary Semantic Map", sem_bgr)

        # ---- 2. 主深度图: fixed_gray (固定 0~vmax 近黑远白) 或 legacy_inferno ----
        if self.depth_display_mode == "legacy_inferno":
            depth_m = np.nan_to_num(
                depth_map.astype(np.float32, copy=False),
                nan=self.depth_vmax_m, posinf=self.depth_vmax_m, neginf=0.0,
            )
            depth_m = np.clip(depth_m, self.depth_near_m, self.depth_vmax_m)
            depth_u8 = np.round(
                np.clip(depth_m / max(self.depth_vmax_m, 1e-6), 0.0, 1.0) * 255.0
            ).astype(np.uint8)
            depth_img = cv2.applyColorMap(depth_u8, cv2.COLORMAP_INFERNO)
        else:  # fixed_gray: 0m→黑, vmax m→白, 跨帧灰度恒定
            depth_img = render_depth_fixed_gray(depth_map, vmax_m=self.depth_vmax_m)

        # 深度图也用同样的 bounding box 裁剪（与语义图空间对齐）
        if valid_mask.any():
            depth_img = depth_img[r_min:r_max + 1, c_min:c_max + 1]
        depth_resized = cv2.resize(depth_img, (disp_w, disp_h),
                                   interpolation=cv2.INTER_NEAREST)

        # 主窗口右侧色条
        bar_w = 40
        bar_x = disp_w + 5
        with_bar = np.zeros((disp_h, disp_w + bar_w + 10, 3), dtype=np.uint8)
        with_bar[:, :disp_w] = depth_resized
        if self.depth_display_mode == "legacy_inferno":
            # legacy: inferno 色条 (底部黑 0m → 顶部亮 vmax m)
            for row in range(disp_h):
                val = 255 - int(row / max(disp_h - 1, 1) * 255)
                with_bar[row, bar_x:bar_x + bar_w] = cv2.applyColorMap(
                    np.array([[val]], dtype=np.uint8), cv2.COLORMAP_INFERNO)[0, 0]
            cv2.putText(with_bar, "0m", (bar_x - 5, disp_h - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            cv2.putText(with_bar, f"{int(self.depth_vmax_m)}m", (bar_x - 5, 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            cv2.putText(with_bar, "0->far", (5, 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        else:
            # fixed_gray: 固定灰阶色条 (底部黑 0m → 顶部白 vmax m), 不随帧变化
            for row in range(disp_h):
                val = 255 - int(row / max(disp_h - 1, 1) * 255)
                with_bar[row, bar_x:bar_x + bar_w] = (val, val, val)
            cv2.putText(with_bar, "0m", (bar_x - 5, disp_h - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            cv2.putText(with_bar, f"{int(self.depth_vmax_m)}m", (bar_x - 5, 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            cv2.putText(with_bar, f"fixed 0-{int(self.depth_vmax_m)}m", (5, 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        cv2.imshow("2.Depth Map (0-30m)", with_bar)

        # ---- 3. 局部深度对比度窗口: 逐帧自动量程, 纯灰度, 无彩色叠加 ----
        if self.show_local_depth:
            if semantic_valid_mask is None:
                smask = np.ones(depth_map.shape[:2], dtype=bool)
            else:
                smask = np.asarray(semantic_valid_mask, dtype=bool)
                if smask.shape != depth_map.shape[:2]:
                    smask = cv2.resize(
                        smask.astype(np.uint8),
                        (depth_map.shape[1], depth_map.shape[0]),
                        interpolation=cv2.INTER_NEAREST,
                    ).astype(bool)
            local_img, local_info = render_depth_local_contrast(
                depth_map, smask,
                vmax_m=self.depth_vmax_m,
                pct_low=self.depth_display_pct_low,
                pct_high=self.depth_display_pct_high,
                min_span_m=self.depth_display_min_span_m,
                min_valid=self.depth_display_min_valid,
            )
            # 与主深度图同样的 bounding box 裁剪 (空间对齐)
            if valid_mask.any():
                local_img = local_img[r_min:r_max + 1, c_min:c_max + 1]
            local_resized = cv2.resize(local_img, (disp_w, disp_h),
                                       interpolation=cv2.INTER_NEAREST)
            loc_with_bar = np.zeros((disp_h, disp_w + bar_w + 10, 3), dtype=np.uint8)
            loc_with_bar[:, :disp_w] = local_resized
            # 灰阶色条: 顶部白 (far) → 底部黑 (near), 标注当前帧实际量程
            for row in range(disp_h):
                val = 255 - int(row / max(disp_h - 1, 1) * 255)
                loc_with_bar[row, bar_x:bar_x + bar_w] = (val, val, val)
            cv2.putText(loc_with_bar, f"{local_info['far_m']:.1f}m", (bar_x - 5, 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            cv2.putText(loc_with_bar, f"{local_info['near_m']:.2f}m",
                        (bar_x - 5, disp_h - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            cv2.putText(loc_with_bar,
                        "auto range" if local_info["auto_range"] else "fixed range",
                        (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            cv2.putText(loc_with_bar,
                        f"{local_info['near_m']:.2f}~{local_info['far_m']:.1f} m",
                        (5, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            cv2.imshow("3.Local Depth Contrast", loc_with_bar)

        cv2.waitKey(1)

        # ---- 4. 3D 去畸变点云 (halss_pts, z 值着色, 与上述窗口同帧同步) ----
        if self.show_pointcloud:
            self._update_pointcloud(point_cloud, cloud_stamp)

    def close(self):
        if self._pc_fig is not None:
            try:
                self._mpl.close(self._pc_fig)
            except Exception:
                pass
            self._pc_fig = None
            self._pc_ax = None
            self._pc_scatter = None
        cv2.destroyAllWindows()


# ──────────────────────────────────────────────
# 主回放逻辑
# ──────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="离线 rosbag 回放 — 感知 + ONNX DRL 推理 (与 pipeline.py 一致)"
    )
    parser.add_argument("--bag", type=str, required=True,
                        help="输入的 rosbag 路径 (experiments/runs/*/input.bag)")
    parser.add_argument("--config", type=str, required=True,
                        help="实验配置路径 (experiment_outdoor_gps.yaml)")
    parser.add_argument("--onnx-model", type=str, default="weights/ppo2_policy.onnx",
                        help="ONNX DRL 策略模型路径")
    parser.add_argument("--dmax", type=float, default=30.0,
                        help="深度最大距离 (米)")
    parser.add_argument("--ground-z", type=float, default=None,
                        help="手动指定地面 Z (米); 默认取第一帧 pose.z")
    parser.add_argument("--max-frames", type=int, default=0,
                        help="最多处理 N 帧 (0=处理全部)")
    parser.add_argument("--no-display", action="store_true",
                        help="关闭全部可视化窗口 (且不导入 Matplotlib)")
    parser.add_argument("--no-pointcloud", action="store_true",
                        help="仅关闭 Matplotlib 3D 点云窗口 (保留语义/深度窗口)")
    parser.add_argument("--depth-display-mode", default=None,
                        choices=["fixed_gray", "legacy_inferno"],
                        help="主深度图显示模式: fixed_gray=固定 0~30m 近黑远白 "
                             "(默认); legacy_inferno=旧版 inferno 色表 (对比用)")
    parser.add_argument("--no-local-depth", action="store_true",
                        help="关闭局部深度对比度窗口 (3.Local Depth Contrast), "
                             "保留主深度图")
    parser.add_argument("--rate", type=float, default=0.0,
                        help="回放速率倍率 (0=尽可能快, 1=实时)")
    parser.add_argument("--world-cloud", action="store_true",
                        help="强制使用 world-cloud 模式 (/ali_cloud + /ali_odom)")
    parser.add_argument("--body-cloud", action="store_true",
                        help="强制使用 body-cloud 模式（位姿源由 localization.mode 决定）")
    args = parser.parse_args()

    # ── 加载配置 ──
    cfg = _load_config(args.config)
    logger.info("Config loaded: %s", args.config)

    # ── 确定话题和模式 ──
    loc_cfg = cfg.get("localization", {})
    localization_mode = str(loc_cfg.get("mode", "")).lower()
    px4_pose_authoritative = localization_mode == "gps_px4_fastlio_perception"
    use_body_cloud = args.body_cloud or (
        not args.world_cloud
        and bool(loc_cfg.get("use_body_cloud",
                             loc_cfg.get("mode", "") == "gps_px4_fastlio_perception"))
    )

    if use_body_cloud:
        cloud_topic = str(loc_cfg.get("body_cloud_topic", "/cloud_registered_body"))
        px4_odom_topic = "/mavros/local_position/odom"
        fastlio_odom_topic = str(loc_cfg.get("fastlio_odom_topic", "/ali_odom"))
        body_pose_topic = px4_odom_topic if px4_pose_authoritative else fastlio_odom_topic
        logger.info(
            "Mode: body-cloud cloud=%s pose=%s authority=%s",
            cloud_topic,
            body_pose_topic,
            "px4_ekf" if px4_pose_authoritative else "fastlio",
        )
    else:
        cloud_topic = str(loc_cfg.get("world_cloud_topic", "/ali_cloud"))
        fastlio_odom_topic = str(loc_cfg.get("fastlio_odom_topic", "/ali_odom"))
        px4_odom_topic = None
        logger.info("Mode: world-cloud  cloud=%s  odom=%s",
                     cloud_topic, fastlio_odom_topic)

    # ── 检查 rosbag ──
    bag_path = Path(args.bag)
    if not bag_path.is_file():
        logger.error("Bag not found: %s", bag_path)
        sys.exit(1)

    # ── 初始化感知模块 ──
    perc_cfg = cfg["perception"]
    depth_cfg = cfg["depth_projection"]
    obs_cfg = cfg["observation"]
    uav_cfg = cfg.get("uav", {})
    runtime_cfg = cfg.get("runtime", {})

    obs_h = int(obs_cfg.get("img_height", 128))
    obs_w = int(obs_cfg.get("img_width", 128))
    depth_max = args.dmax
    safe_id = int(perc_cfg.get("safe_class_id", 1))
    danger_id = int(perc_cfg.get("danger_class_id", 9))
    max_sync_ms = float(runtime_cfg.get("max_cloud_odom_sync_ms", 100.0))
    projection_mode = str(depth_cfg.get("mode", "training_camera")).lower()
    sim_dt = float(uav_cfg.get("sim_dt", 0.25))

    # ROI 参数
    roi_half_x = float(perc_cfg.get("halss_roi_half_x_m", 5.0))
    roi_half_y = float(perc_cfg.get("halss_roi_half_y_m", 5.0))
    roi_dynamic = bool(perc_cfg.get("halss_roi_dynamic_enabled", True))
    roi_fov_half_rad = math.radians(float(perc_cfg.get("halss_roi_fov_half_deg", 45.0)))
    roi_min_half = float(perc_cfg.get("halss_roi_min_half_m", 0.5))
    roi_max_half = float(perc_cfg.get("halss_roi_max_half_m", 15.0))
    roi_height_src = str(perc_cfg.get("halss_roi_height_source", "pose_z"))
    halss_ray_sampling = bool(perc_cfg.get("halss_pinhole_ray_sampling_enabled", False))
    halss_ray_grid = int(perc_cfg.get("halss_pinhole_ray_grid_res", 64))

    # 延时导入感知模块 (与 pipeline.py _import_runtime_deps 一致)
    from perception.halss_bayesian import HALSSBayesianEvaluator
    from perception.halss_preprocess import (
        world_to_level_body_roi,
        body_cloud_to_level_body_roi,
    )
    from perception.semantic_generator import SemanticGenerator
    from perception.training_camera_projection import (
        TrainingCameraModel,
        project_training_camera,
        sample_nearest_points_by_camera_rays,
    )
    from control.action_decomposer import ActionDecomposer

    # ── 初始化各模块 ──
    logger.info("[Init] HALSS Bayesian evaluator...")
    halss = HALSSBayesianEvaluator(perc_cfg)

    logger.info("[Init] Semantic generator...")
    sem_gen = SemanticGenerator({**perc_cfg, "img_width": obs_w, "img_height": obs_h})

    logger.info("[Init] Training camera model...")
    training_camera = TrainingCameraModel.from_config(
        depth_cfg.get("training_camera", {}),
        output_width=obs_w, output_height=obs_h, far_m=depth_max,
    )
    logger.info("  FOV=%.1fx%.1fdeg scaled_fx=%.3f fy=%.3f",
                training_camera.horizontal_fov_deg, training_camera.vertical_fov_deg,
                training_camera.fx, training_camera.fy)

    logger.info("[Init] ONNX DRL...")
    drl = ONNXDRL(
        args.onnx_model,
        obs_h=obs_h, obs_w=obs_w, dmax=depth_max,
        depth_norm_mode=str(obs_cfg.get("depth_norm_mode", "raw_meters_graph_scaled")),
        semantic_norm_mode=str(obs_cfg.get("semantic_norm_mode", "raw_gray_graph_scaled")),
    )

    logger.info("[Init] Action decomposer...")
    decomposer = ActionDecomposer(uav_cfg)
    action_names = decomposer.action_names
    logger.info("  %d actions, frame=%s sign=%d",
                len(action_names), decomposer.action_frame, decomposer.action_lateral_sign)

    # ── 可视化 ──
    vis_cfg = cfg.get("visualization", {})
    show_pointcloud = (not args.no_pointcloud) and bool(
        vis_cfg.get("show_pointcloud", True))
    # 主深度图模式: CLI 显式参数 > 配置 (仅两种有效值) > 默认 fixed_gray
    cfg_depth_mode = str(vis_cfg.get("depth_display_mode", "")).lower()
    depth_display_mode = args.depth_display_mode
    if depth_display_mode is None:
        depth_display_mode = (
            cfg_depth_mode if cfg_depth_mode in {"fixed_gray", "legacy_inferno"}
            else "fixed_gray")
    show_local_depth = (not args.no_local_depth) and bool(
        vis_cfg.get("show_local_depth", True))
    display = None if args.no_display else ReplayVisualizer(
        dmax=depth_max,
        depth_vmax_m=float(vis_cfg.get("depth_vmax_m", depth_max)),
        display_width=int(vis_cfg.get("display_width", 300)),
        depth_display_mode=depth_display_mode,
        depth_near_m=float(vis_cfg.get("depth_near_m", 0.5)),
        depth_display_pct_low=float(vis_cfg.get("depth_display_pct_low", 2.0)),
        depth_display_pct_high=float(vis_cfg.get("depth_display_pct_high", 98.0)),
        depth_display_min_span_m=float(vis_cfg.get("depth_display_min_span_m", 0.5)),
        depth_display_min_valid=int(vis_cfg.get("depth_display_min_valid", 32)),
        show_pointcloud=show_pointcloud,
        show_local_depth=show_local_depth,
    )

    # ──────────────────────────────────────────
    # 开始读取 rosbag
    # ──────────────────────────────────────────
    import rosbag
    from sensor_msgs.msg import PointCloud2
    from nav_msgs.msg import Odometry

    logger.info("[Bag] Opening %s ...", bag_path)
    bag = rosbag.Bag(str(bag_path), "r")

    # 统计 bag 中的话题
    topics_info = bag.get_type_and_topic_info()
    logger.info("[Bag] Topics in bag: %s",
                ", ".join(sorted(topics_info.topics.keys())))

    # ── 扫描 bag 获取 cloud + odom 消息 ──
    # 策略: 遍历 bag, 将 cloud 消息放入队列; 同时维护 odom 滑动窗口,
    # 对每个 cloud 找到最近的 odom 并处理.
    # 这样可以避免将整个 bag 读入内存, 同时保证帧顺序.
    import rosbag  # noqa
    from sensor_msgs.msg import PointCloud2  # noqa
    from nav_msgs.msg import Odometry  # noqa

    # Body cloud needs timestamp-matched roll/pitch. Outdoor uses PX4 EKF;
    # indoor full-SLAM uses FAST-LIO /ali_odom.
    if use_body_cloud:
        primary_cloud_topic = cloud_topic
        primary_odom_topic = body_pose_topic
        read_topics = [primary_cloud_topic, primary_odom_topic]
        if px4_pose_authoritative and fastlio_odom_topic != primary_odom_topic:
            read_topics.append(fastlio_odom_topic)
    else:
        primary_cloud_topic = cloud_topic
        primary_odom_topic = fastlio_odom_topic
        read_topics = [primary_cloud_topic, primary_odom_topic]

    # 验证 bag 中是否包含必要话题
    available = set(topics_info.topics.keys())
    for t in read_topics:
        if t not in available:
            logger.warning("[Bag] Topic %s not found in bag", t)

    logger.info("[Bag] Reading topics: %s", read_topics)

    # ── 状态变量 ──
    frame_count = 0
    action_counts = [0] * 10
    ground_z_world = args.ground_z
    last_cloud_stamp = None
    last_cloud_pts = None
    last_odom_stamp = None
    last_odom_msg = None
    last_fastlio_stamp = None
    last_fastlio_pose = None

    # PX4 odom 滑动窗口 (body-cloud 模式): 存储 (stamp, msg) 元组
    px4_odom_window = deque(maxlen=50)

    # 用于统计
    start_wall = time.perf_counter()
    total_processed = 0

    try:
        for topic_name, msg, ros_stamp in bag.read_messages(topics=read_topics):
            stamp_sec = _stamp_to_sec(msg.header.stamp) if hasattr(msg, 'header') else _stamp_to_sec(ros_stamp)

            # ── body-cloud 匹配位姿，按场景来自 PX4 或 FAST-LIO ──
            if use_body_cloud and topic_name == primary_odom_topic:
                px4_odom_window.append((stamp_sec, msg))
                continue

            # ── 如果是 FAST-LIO odom (world-cloud 模式或诊断用) ──
            if topic_name == fastlio_odom_topic:
                if use_body_cloud:
                    # 仅诊断记录, 不用于感知
                    p = msg.pose.pose.position
                    q = msg.pose.pose.orientation
                    r, pch, y = _quat_to_euler(q.x, q.y, q.z, q.w)
                    last_fastlio_pose = np.array([p.x, p.y, p.z, r, pch, y], dtype=np.float32)
                    last_fastlio_stamp = stamp_sec
                else:
                    # world-cloud 模式: 直接作为位姿源
                    p = msg.pose.pose.position
                    q = msg.pose.pose.orientation
                    r, pch, y = _quat_to_euler(q.x, q.y, q.z, q.w)
                    last_odom_msg = np.array([p.x, p.y, p.z, r, pch, y], dtype=np.float32)
                    last_odom_stamp = stamp_sec
                    last_odom_raw = msg
                continue

            # ── 如果是点云 ──
            if topic_name == primary_cloud_topic:
                cloud_pts = _pc2_to_numpy(msg)
                cloud_stamp = stamp_sec

                if len(cloud_pts) == 0:
                    logger.warning("[Frame] Empty cloud at t=%.3f, skipping", cloud_stamp)
                    continue

                # 获取匹配的位姿
                if use_body_cloud:
                    # 从场景指定的 odom 窗口找到最近位姿
                    if len(px4_odom_window) == 0:
                        logger.warning("[Frame] No matched odom available at t=%.3f", cloud_stamp)
                        continue
                    # 找时间最近的位姿
                    odom_times = np.array([t for t, _ in px4_odom_window])
                    nearest_idx = np.argmin(np.abs(odom_times - cloud_stamp))
                    nearest_stamp = float(odom_times[nearest_idx])
                    sync_ms = abs(nearest_stamp - cloud_stamp) * 1000.0
                    if sync_ms > max_sync_ms:
                        logger.warning("[Frame] Sync %.0fms > %.0fms at t=%.3f, skip",
                                       sync_ms, max_sync_ms, cloud_stamp)
                        continue
                    _, pose_msg = px4_odom_window[nearest_idx]
                    q = pose_msg.pose.pose.orientation
                    roll, pitch, _ = _quat_to_euler(q.x, q.y, q.z, q.w)
                    # 获取定位源位置和 yaw
                    p = pose_msg.pose.pose.position
                    px4_pos = np.array([p.x, p.y, p.z], dtype=np.float32)
                    _, _, px4_yaw = _quat_to_euler(q.x, q.y, q.z, q.w)

                    # 组合 frame_pose: [x, y, z, roll, pitch, yaw] (ENU yaw)
                    frame_pose = np.array([
                        px4_pos[0], px4_pos[1], px4_pos[2],
                        roll, pitch, px4_yaw,
                    ], dtype=np.float32)

                    # body_cloud_to_level_body_roi: 只需要 roll, pitch
                    try:
                        projection_pts, halss_stats = body_cloud_to_level_body_roi(
                            cloud_pts, roll, pitch, perc_cfg,
                            half_x=roi_half_x, half_y=roi_half_y,
                        )
                    except Exception as e:
                        logger.warning("[Frame] body_cloud_to_level_body_roi error: %s", e)
                        continue

                    # Pinhole ray sampling (如果启用)
                    if halss_ray_sampling:
                        halss_pts, ray_stats = sample_nearest_points_by_camera_rays(
                            projection_pts, training_camera,
                            ray_width=halss_ray_grid, ray_height=halss_ray_grid,
                        )
                        halss_stats["pre_ray_points"] = int(len(projection_pts))
                        halss_stats["frustum_points"] = int(ray_stats["frustum_points"])
                        halss_stats["ray_grid"] = [halss_ray_grid, halss_ray_grid]
                        halss_stats["output_points"] = int(len(halss_pts))
                        # Shared sampling: depth projection also uses the same ray-sampled points.
                        projection_pts = halss_pts
                    else:
                        halss_pts = projection_pts
                        halss_stats["output_points"] = int(len(halss_pts))

                else:
                    # world-cloud 模式
                    if last_odom_msg is None:
                        logger.warning("[Frame] No odom available at t=%.3f", cloud_stamp)
                        continue
                    sync_ms = abs(last_odom_stamp - cloud_stamp) * 1000.0
                    if sync_ms > max_sync_ms:
                        logger.warning("[Frame] Sync %.0fms > %.0fms at t=%.3f, skip",
                                       sync_ms, max_sync_ms, cloud_stamp)
                        continue
                    frame_pose = last_odom_msg.copy()

                    try:
                        halss_pts, halss_stats = world_to_level_body_roi(
                            cloud_pts, frame_pose[:3], frame_pose[3:],
                            perc_cfg, half_x=roi_half_x, half_y=roi_half_y,
                        )
                    except Exception as e:
                        logger.warning("[Frame] world_to_level_body_roi error: %s", e)
                        continue
                    projection_pts = halss_pts

                # ── 检查 ROI 点数 ──
                if halss_stats.get("output_points", 0) < 10:
                    logger.debug("[Frame] Sparse ROI: %d points, skip",
                                 halss_stats.get("output_points", 0))
                    continue

                # ── 记录地面 Z (第一帧) ──
                if ground_z_world is None:
                    ground_z_world = float(frame_pose[2])
                    logger.info("[Ground] First frame pose_z=%.2f set as ground_z", ground_z_world)

                # ── 动态 FOV ROI ──
                if roi_dynamic:
                    H = abs(float(frame_pose[2]) - ground_z_world)
                    H = max(H, 0.1)
                    if projection_mode == "training_camera":
                        cur_half_x, cur_half_y = training_camera.ground_half_extents(H)
                        cur_half_x = max(roi_min_half, min(roi_max_half, cur_half_x))
                        cur_half_y = max(roi_min_half, min(roi_max_half, cur_half_y))
                    else:
                        half = H * math.tan(roi_fov_half_rad)
                        cur_half_x = cur_half_y = max(roi_min_half, min(roi_max_half, half))
                else:
                    cur_half_x, cur_half_y = roi_half_x, roi_half_y

                # ── ROI bounds ──
                roi_bounds = {
                    "x_min": -cur_half_x, "x_max": cur_half_x,
                    "y_min": -cur_half_y, "y_max": cur_half_y,
                }

                # ── 1. HALSS 语义 ──
                t0 = time.perf_counter()
                try:
                    halss_result = halss.evaluate(halss_pts, fixed_bounds=roi_bounds)
                except Exception as e:
                    logger.warning("[Frame] HALSS error: %s", e)
                    halss_result = None

                if halss_result is not None:
                    bev_data = halss_result.get("bev_data", halss_result)
                    sem_map = sem_gen.generate(bev_data)
                else:
                    sem_map = np.full((obs_h, obs_w), danger_id, dtype=np.uint8)
                t_halss = time.perf_counter() - t0

                # ── 2. 深度投影 ──
                t1 = time.perf_counter()
                semantic_valid_mask = np.ones_like(sem_map, dtype=bool)
                if projection_mode == "training_camera":
                    sparse_depth, valid_mask, sem_map, semantic_valid_mask = \
                        project_training_camera(
                            projection_pts, sem_map, roi_bounds, training_camera,
                            danger_id=danger_id,
                        )
                    binary_semantic_vis = make_binary_semantic_vis(
                        sem_map, safe_id=safe_id, danger_id=danger_id,
                    )
                    binary_semantic_vis[~semantic_valid_mask] = 128
                else:
                    sparse_depth, _ = project_bev_depth(
                        halss_pts,
                        grid_res=int(perc_cfg.get("halss_grid_res", 64)),
                        out_size=obs_w, max_range=depth_max,
                        half_x=cur_half_x, half_y=cur_half_y,
                    )
                    valid_mask = (sparse_depth < depth_max) & (sparse_depth > 0.01)
                    binary_semantic_vis = make_binary_semantic_vis(
                        sem_map, safe_id=safe_id, danger_id=danger_id,
                    )
                t_depth = time.perf_counter() - t1

                # ── 3. NN-fill 深度渲染 ──
                t2 = time.perf_counter()
                rendered_depth = render_sparse_depth(sparse_depth, valid_mask, depth_max)
                t_completion = time.perf_counter() - t2

                # ── 4. ONNX DRL 推理 ──
                t3 = time.perf_counter()
                action_id, rl_info = drl.predict(rendered_depth, sem_map)
                rl_info["semantic_valid_ratio"] = float(np.mean(semantic_valid_mask))
                t_drl = time.perf_counter() - t3

                # ── 5. 动作解算 ──
                action_name = decomposer.action_id_to_name(action_id)
                execution_yaw = float(frame_pose[5])  # ENU yaw for logging
                v_body, v_ned, action_yaw_rate = decomposer.decompose(
                    action_id, execution_yaw,
                )
                action_counts[action_id] += 1

                total_ms = (time.perf_counter() - t0) * 1000
                frame_count += 1
                total_processed += 1

                # ── 打印每帧结果 ──
                safe_ratio = float(np.mean(sem_map == safe_id))
                danger_ratio = float(np.mean(sem_map == danger_id))
                conf = rl_info.get("confidence", 0.0)
                top3_str = _top_probs(rl_info.get("action_probs"), action_names, k=3)

                print(
                    f"\r[{frame_count:04d}] ACT={action_id}({action_name})  "
                    f"pts={halss_stats.get('output_points', 0):4d}  "
                    f"sem_safe={safe_ratio:.2f} sem_danger={danger_ratio:.2f}  "
                    f"depth={float(np.mean(rendered_depth)):.1f}m  "
                    f"conf={conf:.2f}  {top3_str}  "
                    f"lat={total_ms:.0f}ms  "
                    f"(H={t_halss*1000:.0f} D={t_depth*1000:.0f} C={t_completion*1000:.0f} RL={t_drl*1000:.0f})  "
                    f"v_ned=({v_ned[0]:+.2f},{v_ned[1]:+.2f},{v_ned[2]:+.2f})  "
                    f"yaw_rate={action_yaw_rate:+.2f}",
                    end="", flush=True,
                )

                # ── 日志 (每 5 帧) ──
                if frame_count % 5 == 0:
                    logger.info(
                        "[%04d] act=%d(%s) pts=%d sem_safe=%.2f sem_danger=%.2f "
                        "depth_mean=%.1fm conf=%.2f %s lat=%.0fms "
                        "v_ned=(%.2f,%.2f,%.2f) yr=%.2f",
                        frame_count, action_id, action_name,
                        halss_stats.get('output_points', 0),
                        safe_ratio, danger_ratio,
                        float(np.mean(rendered_depth)), conf, top3_str,
                        total_ms,
                        v_ned[0], v_ned[1], v_ned[2], action_yaw_rate,
                    )
                    logger.info(
                        "  depth_in=[%.1f,%.1f,%.1f] sem_in=[%.0f,%.0f,%.0f] "
                        "obs_raw=[%.1f,%.1f]",
                        rl_info["depth_input_min"], rl_info["depth_input_mean"],
                        rl_info["depth_input_max"],
                        rl_info["sem_input_min"], rl_info["sem_input_mean"],
                        rl_info["sem_input_max"],
                        rl_info["obs_raw_min"], rl_info["obs_raw_max"],
                    )

                # ── 可视化 (三窗口同一帧: 语义 / 深度 / 3D 去畸变点云) ──
                if display is not None:
                    display.update(
                        rendered_depth, sem_map, binary_semantic_vis,
                        point_cloud=halss_pts, cloud_stamp=cloud_stamp,
                        semantic_valid_mask=semantic_valid_mask,
                    )

                # ── 速率控制 ──
                if args.rate > 0:
                    elapsed = time.perf_counter() - t0
                    sleep = max(0.0, (1.0 / args.rate) - elapsed)
                    if sleep > 0:
                        time.sleep(sleep)

                # ── 最大帧数限制 ──
                if args.max_frames > 0 and frame_count >= args.max_frames:
                    logger.info("[Replay] Max frames reached: %d", args.max_frames)
                    break

    except KeyboardInterrupt:
        logger.info("[Replay] Interrupted by user.")
    finally:
        bag.close()
        elapsed = time.perf_counter() - start_wall
        logger.info("=" * 60)
        logger.info("Replay summary:")
        logger.info("  Total frames: %d", total_processed)
        logger.info("  Elapsed: %.1fs", elapsed)
        if total_processed > 0:
            logger.info("  Avg fps: %.1f", total_processed / max(elapsed, 0.01))
        total_acts = sum(action_counts)
        if total_acts > 0:
            logger.info("  Action distribution:")
            for i, cnt in enumerate(action_counts):
                if cnt > 0:
                    pct = 100.0 * cnt / total_acts
                    logger.info("    %s: %d (%.1f%%)", action_names[i], cnt, pct)
        logger.info("=" * 60)

        if display is not None:
            display.close()


if __name__ == "__main__":
    main()
