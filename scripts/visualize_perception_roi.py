#!/usr/bin/env python3
"""
FAST-LIO 世界云 → 感知下视 ROI 可视化 (对齐 pipeline.py)
========================================================
提取 pipeline.py 当前感知链路实际使用的下视点云, 用于调试安装参数 / ROI 范围。

数据流 (与 pipeline.py 完全一致):
  /ali_cloud (世界系去畸变点云) + /ali_odom (重力校准位姿)
    → world_to_level_body_roi() (yaw 对齐 → 水平机体, 固定 10m×10m ROI)
    → 可视化输出

用法:
  source /opt/ros/noetic/setup.bash
  conda activate fylanding
  python scripts/visualize_perception_roi.py
  python scripts/visualize_perception_roi.py --no-display --save-dir /tmp/roi_debug
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("VisPerceptionROI")

# ============================================================
# 加载配置 (与 pipeline.py 同一份)
# ============================================================
import yaml

CFG_PATH = _PROJECT_ROOT / "config" / "experiment_config.yaml"
with open(CFG_PATH, "r") as f:
    CFG = yaml.safe_load(f)

PERC_CFG = CFG.get("perception", {})
LOC_CFG = CFG.get("localization", {})
HEALTH_CFG = CFG.get("fastlio_health", {})

ODOM_TOPIC = LOC_CFG.get("fastlio_odom_topic", "/ali_odom")
CLOUD_TOPIC = LOC_CFG.get("world_cloud_topic", "/ali_cloud")

ROI_HALF_X = float(PERC_CFG.get("halss_roi_half_x_m", 5.0))
ROI_HALF_Y = float(PERC_CFG.get("halss_roi_half_y_m", 5.0))
ROI_DYNAMIC = bool(PERC_CFG.get("halss_roi_dynamic_enabled", True))
ROI_FOV_HALF_RAD = np.radians(float(PERC_CFG.get("halss_roi_fov_half_deg", 45.0)))
ROI_MIN_HALF = float(PERC_CFG.get("halss_roi_min_half_m", 0.5))
ROI_MAX_HALF = float(PERC_CFG.get("halss_roi_max_half_m", 15.0))
ROI_HEIGHT_SRC = str(PERC_CFG.get("halss_roi_height_source", "pose_z"))
MIN_DOWN = float(PERC_CFG.get("halss_min_down_m", 0.05))
MAX_DOWN = float(PERC_CFG.get("halss_max_down_m", 30.0))
GROUND_MIN_DOWN = float(PERC_CFG.get("halss_ground_min_down_m", 0.0))
YAW_ONLY = bool(PERC_CFG.get("halss_yaw_only", True))
WORLD_Z_UP = bool(PERC_CFG.get("world_z_up", False))
ROI_AREA_M2 = (2.0 * ROI_HALF_X) * (2.0 * ROI_HALF_Y)

LIDAR_POS_BODY = np.array(
    PERC_CFG.get("halss_lidar_position_body_m", [0.13, 0.0, 0.08]),
    dtype=np.float32,
)
LIDAR_PITCH_DEG = float(PERC_CFG.get("halss_lidar_pitch_down_deg", 26.0))
LIDAR_YAW_OFFSET_DEG = float(PERC_CFG.get("halss_lidar_yaw_offset_deg", 0.0))

SAFE_ID = int(PERC_CFG.get("safe_class_id", 1))
DANGER_ID = int(PERC_CFG.get("danger_class_id", 9))

# ============================================================
# 复用 pipeline.py 的感知预处理
# ============================================================
import rospy
from perception.halss_preprocess import world_to_level_body_roi

# FAST-LIO 接口 (轻量, 只需 pose + cloud, 不初始化 HALSS/DRL)
from odometry import FastLIOInterface


# ============================================================
# ROS 桥接
# ============================================================
def init_ros_bridge() -> FastLIOInterface:
    from nav_msgs.msg import Odometry
    from sensor_msgs.msg import PointCloud2

    if not rospy.core.is_initialized():
        rospy.init_node("visualize_perception_roi", anonymous=False)

    fastlio = FastLIOInterface(use_ros=True)
    fastlio._odom_sub = rospy.Subscriber(
        ODOM_TOPIC, Odometry, fastlio.odometry_callback, queue_size=10,
    )
    fastlio._cloud_sub = rospy.Subscriber(
        CLOUD_TOPIC, PointCloud2, fastlio.pointcloud_callback, queue_size=10,
    )
    logger.info("[Bridge] Subscribed: %s + %s", ODOM_TOPIC, CLOUD_TOPIC)
    return fastlio


def grab_snapshot(fastlio: FastLIOInterface):
    pts = fastlio.points
    pose = fastlio.pose
    if pts is None or pose is None:
        return None, None
    return pts.copy(), pose.copy()


# ============================================================
# 可视化
# ============================================================

def _fmt_v(v) -> str:
    return f"[{v[0]:.2f},{v[1]:.2f},{v[2]:.2f}]"


def _auto_zoom(ax, pts, cols=(0, 1), min_span=2.0):
    if pts is None or len(pts) == 0:
        return
    x = pts[:, cols[0]]
    y = pts[:, cols[1]]
    xm = (x.min() + x.max()) / 2
    ym = (y.min() + y.max()) / 2
    hs_x = max((x.max() - x.min()) / 2, min_span / 2) * 1.15
    hs_y = max((y.max() - y.min()) / 2, min_span / 2) * 1.15
    ax.set_xlim(xm - hs_x, xm + hs_x)
    ax.set_ylim(ym - hs_y, ym + hs_y)


def live_display(fastlio: FastLIOInterface, save_dir: str = None, max_frames: int = 0):
    import matplotlib
    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    plt.ion()
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    fig.canvas.manager.set_window_title("Perception ROI — FAST-LIO → HALSS Down-Looking Points")
    plt.show(block=False)
    fig.suptitle("", fontsize=11)

    imgs = {"td_world": None, "td_body": None, "side": None, "text": None}
    ready = False
    frame_idx = 0
    last_seq = -1
    ground_z_world = None
    cur_half_x = ROI_HALF_X
    cur_half_y = ROI_HALF_Y
    cur_height = float("nan")

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    while not rospy.is_shutdown():
        pts_world, pose = grab_snapshot(fastlio)
        if pts_world is None or pose is None:
            plt.pause(0.05)
            continue

        seq = fastlio.points_seq
        if seq <= last_seq:
            plt.pause(0.005)
            continue
        last_seq = seq

        # ---- 动态 FOV ROI 计算 ----
        if ground_z_world is None:
            ground_z_world = float(pose[2])
        if ROI_DYNAMIC:
            if ROI_HEIGHT_SRC == "pointcloud_median":
                _rough, _st = world_to_level_body_roi(
                    pts_world, pose[:3], pose[3:], PERC_CFG,
                    half_x=ROI_HALF_X, half_y=ROI_HALF_Y,
                )
                if len(_rough) > 10:
                    H = float(np.median(_rough[:, 2]))
                else:
                    H = abs(float(pose[2]) - ground_z_world)
            else:
                H = abs(float(pose[2]) - ground_z_world)
            H = max(H, 0.1)
            cur_half_x = H * np.tan(ROI_FOV_HALF_RAD)
            cur_half_x = max(ROI_MIN_HALF, min(ROI_MAX_HALF, cur_half_x))
            cur_half_y = cur_half_x
            cur_height = H
        else:
            cur_half_x = ROI_HALF_X
            cur_half_y = ROI_HALF_Y
            cur_height = abs(float(pose[2]) - ground_z_world)

        # ---- 与 pipeline.py 完全相同的变换 ----
        t0 = time.perf_counter()
        roi_pts, stats = world_to_level_body_roi(
            pts_world,
            pose[:3],
            pose[3:],
            PERC_CFG,
            half_x=cur_half_x,
            half_y=cur_half_y,
        )
        dt_ms = (time.perf_counter() - t0) * 1000
        frame_idx += 1

        n_world = stats["input_points"]
        n_roi = stats["output_points"]
        ratio = n_roi / max(n_world, 1) * 100
        sync_ms = fastlio.sync_delta_ms

        # ---- 世界云下采样 (用于俯视图) ----
        n_ds = min(n_world, 30000)
        if n_ds < n_world:
            idx_w = np.random.choice(n_world, n_ds, replace=False)
            pts_world_ds = pts_world[idx_w]
        else:
            pts_world_ds = pts_world

        # ---- 更新图表 ----
        pose_xyz = pose[:3]
        yaw_deg = math.degrees(float(pose[5]))

        fig.suptitle(
            f"Frame #{frame_idx} | World pts: {n_world:,} → ROI: {n_roi:,} ({ratio:.0f}%) | "
            f"pose={_fmt_v(pose_xyz)} yaw={yaw_deg:.0f}° | sync={sync_ms:.0f}ms | "
            f"lat={dt_ms:.0f}ms | "
            f"roi=±{cur_half_x:.1f}m" + (f" H={cur_height:.1f}m" if ROI_DYNAMIC else " (static)") + " "
            f"z=[{stats.get('min_down_m',MIN_DOWN):.2f},{stats.get('max_down_m',MAX_DOWN):.1f}]m"
            + (f" ground_min={GROUND_MIN_DOWN:.2f}m" if GROUND_MIN_DOWN > 0 else ""),
            fontsize=10,
        )

        # --- [0,0] 世界云俯视图 (N-E) ---
        ax = axes[0, 0]
        if not ready:
            imgs["td_world"] = ax.scatter(
                pts_world_ds[:, 0], pts_world_ds[:, 1],
                c="lightblue", s=0.3, alpha=0.6,
            )
            ax.plot(pose_xyz[0], pose_xyz[1], "r*", markersize=12, label="drone")
            ax.legend(fontsize=8)
            ax.set_xlabel("World N (m)")
            ax.set_ylabel("World E (m)")
            ax.set_title("FAST-LIO World Cloud (top-down)")
            ax.set_aspect("equal")
            ax.grid(True, alpha=0.3)
        else:
            imgs["td_world"].set_offsets(pts_world_ds[:, :2])
            for line in ax.lines:
                if line.get_marker() == "*":
                    line.set_data([pose_xyz[0]], [pose_xyz[1]])
        _auto_zoom(ax, pts_world_ds[:, :2], min_span=5.0)

        # --- [0,1] 机体系 ROI 俯视图 (X-Y) ---
        ax = axes[0, 1]
        if not ready:
            if len(roi_pts) > 0:
                imgs["td_body"] = ax.scatter(
                    roi_pts[:, 0], roi_pts[:, 1],
                    c=roi_pts[:, 2], s=1.5, cmap="turbo", alpha=0.8,
                    vmin=0, vmax=min(MAX_DOWN, 15),
                )
                cbar = plt.colorbar(imgs["td_body"], ax=ax, label="z_down (m)", shrink=0.8)
            else:
                imgs["td_body"] = ax.scatter([], [], s=1)
            ax.plot(
                LIDAR_POS_BODY[0], LIDAR_POS_BODY[1],
                "r*", markersize=12, label="LiDAR",
            )
            # LiDAR x-axis (forward) direction arrow
            arrow_len = cur_half_x * 0.2
            ax.arrow(
                LIDAR_POS_BODY[0], LIDAR_POS_BODY[1],
                arrow_len, 0,
                head_width=arrow_len * 0.3, head_length=arrow_len * 0.3,
                fc="red", ec="red", alpha=0.8, label="LiDAR +x (fwd)",
            )
            rect = mpatches.Rectangle(
                (-cur_half_x, -cur_half_y), 2.0 * cur_half_x, 2.0 * cur_half_y,
                fill=False, color="red", linestyle="--", alpha=0.5, linewidth=1.5,
            )
            ax.add_patch(rect)
            ax.legend(fontsize=8)
            ax.set_xlabel("Body X: forward (m)")
            ax.set_ylabel("Body Y: lateral (m)")
            ax.set_title("Body-Frame ROI (top-down, z depth colormap)")
            ax.set_aspect("equal")
            ax.grid(True, alpha=0.3)
        else:
            if len(roi_pts) > 0:
                imgs["td_body"].set_offsets(roi_pts[:, :2])
                imgs["td_body"].set_array(roi_pts[:, 2])
            else:
                imgs["td_body"].set_offsets(np.empty((0, 2)))
        _auto_zoom(ax, roi_pts if len(roi_pts) > 0 else np.array([[0, 0]]), min_span=1.5)

        # --- [1,0] 机体系 ROI 侧视图 (X-Z) ---
        ax = axes[1, 0]
        if not ready:
            if len(roi_pts) > 0:
                imgs["side"] = ax.scatter(
                    roi_pts[:, 0], roi_pts[:, 2],
                    c=roi_pts[:, 1], s=1.5, cmap="coolwarm", alpha=0.8,
                )
            else:
                imgs["side"] = ax.scatter([], [], s=1)
            ax.axhline(y=MIN_DOWN, color="gray", linestyle=":", alpha=0.5, label=f"min_down={MIN_DOWN}m")
            ax.axhline(y=MAX_DOWN, color="gray", linestyle=":", alpha=0.5, label=f"max_down={MAX_DOWN}m")
            ax.plot(LIDAR_POS_BODY[0], LIDAR_POS_BODY[2], "r*", markersize=12, label="LiDAR")
            ax.legend(fontsize=7)
            ax.set_xlabel("Body X: forward (m)")
            ax.set_ylabel("Body Z: down (m)")
            ax.set_title("Body-Frame ROI (side view, X-Z)")
            ax.invert_yaxis()
            ax.grid(True, alpha=0.3)
        else:
            if len(roi_pts) > 0:
                imgs["side"].set_offsets(np.column_stack([roi_pts[:, 0], roi_pts[:, 2]]))
            else:
                imgs["side"].set_offsets(np.empty((0, 2)))
        _auto_zoom(ax, roi_pts if len(roi_pts) > 0 else np.array([[0, 0]]),
                   cols=(0, 2), min_span=1.0)

        # --- [1,1] 统计文本 ---
        ax = axes[1, 1]
        ax.clear()
        ax.axis("off")
        lines = [
            "=== Config (from experiment_config.yaml) ===",
            f"odom_topic:      {ODOM_TOPIC}",
            f"cloud_topic:     {CLOUD_TOPIC}",
            f"roi_mode:        {'dynamic FOV=%.0f°' % np.degrees(ROI_FOV_HALF_RAD*2) if ROI_DYNAMIC else 'static'}",
            f"roi_half:        ±{cur_half_x:.1f} m" + (f" (H={cur_height:.1f}m)" if ROI_DYNAMIC else ""),
            f"roi_min/max:     [{ROI_MIN_HALF:.1f}, {ROI_MAX_HALF:.1f}] m" if ROI_DYNAMIC else f"roi_area:        {(2*cur_half_x)*(2*cur_half_y):.0f} m²",
            f"z_range:         [{MIN_DOWN}, {MAX_DOWN}] m",
            f"ground_min_down: {GROUND_MIN_DOWN} m" + (" (off)" if GROUND_MIN_DOWN <= 0 else " (active)"),
            f"yaw_only:        {YAW_ONLY}",
            f"world_z_up:      {WORLD_Z_UP}",
            f"lidar_pos_body:  {_fmt_v(LIDAR_POS_BODY)} m",
            f"lidar_pitch:     {LIDAR_PITCH_DEG}°",
            f"lidar_yaw_off:   {LIDAR_YAW_OFFSET_DEG}°",
            "",
            "=== Current Frame ===",
            f"world pts:       {n_world:,}",
            f"ROI pts:         {n_roi:,} ({ratio:.0f}%)",
            f"cloud_seq:       {seq}",
            f"pose_seq:        {fastlio.pose_seq}",
            f"sync_ms:         {sync_ms:.0f}",
            f"pose:            {_fmt_v(pose_xyz)}",
            f"yaw:             {yaw_deg:.1f}°",
            f"transform_lat:   {dt_ms:.0f}ms",
            "",
            "=== ROI Stats ===",
        ]
        if n_roi > 0:
            lines += [
                f"z_min:           {roi_pts[:, 2].min():.2f} m",
                f"z_max:           {roi_pts[:, 2].max():.2f} m",
                f"z_mean:          {roi_pts[:, 2].mean():.2f} m",
                f"z_median:        {np.median(roi_pts[:, 2]):.2f} m",
                f"x_range:         [{roi_pts[:, 0].min():.1f}, {roi_pts[:, 0].max():.1f}] m",
                f"y_range:         [{roi_pts[:, 1].min():.1f}, {roi_pts[:, 1].max():.1f}] m",
                f"density:         {n_roi / max((2*cur_half_x)*(2*cur_half_y), 1e-6):.0f} pts/m²",
            ]
        else:
            lines.append("  (empty ROI)")

        ax.text(0.05, 0.95, "\n".join(lines), transform=ax.transAxes,
                fontfamily="monospace", fontsize=8.5, verticalalignment="top")

        ready = True
        fig.canvas.draw_idle()
        fig.canvas.flush_events()

        # 保存帧
        if save_dir and frame_idx % 10 == 0:
            fig.savefig(
                os.path.join(save_dir, f"frame_{frame_idx:06d}.png"),
                dpi=100, bbox_inches="tight",
            )
            np.savez_compressed(
                os.path.join(save_dir, f"frame_{frame_idx:06d}.npz"),
                roi_pts=roi_pts.astype(np.float32),
                pose=pose.astype(np.float32),
                stats=np.array([n_world, n_roi, sync_ms or 0, dt_ms], dtype=np.float32),
            )

        if max_frames > 0 and frame_idx >= max_frames:
            logger.info("Reached max_frames=%d, stopping.", max_frames)
            break

        plt.pause(0.02)

    plt.close(fig)
    logger.info("Display closed. Total frames: %d", frame_idx)


# ============================================================
# main
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description="FAST-LIO → 感知 ROI 点云可视化 (对齐 pipeline.py)",
    )
    parser.add_argument("--no-display", action="store_true", help="不弹出窗口")
    parser.add_argument("--save-dir", default=None, help="保存帧到目录")
    parser.add_argument("--max-frames", type=int, default=0, help="最大帧数, 0=无限")
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info(" Perception ROI Visualizer (pipeline-aligned)")
    logger.info("=" * 60)
    logger.info("  Config:       %s", CFG_PATH)
    logger.info("  odom_topic:   %s", ODOM_TOPIC)
    logger.info("  cloud_topic:  %s", CLOUD_TOPIC)
    logger.info("  roi_mode:     %s", "dynamic FOV=%.0f°" % np.degrees(ROI_FOV_HALF_RAD*2) if ROI_DYNAMIC else "static ±%.0f×±%.0fm" % (ROI_HALF_X, ROI_HALF_Y))
    logger.info("  z_range:      [%.2f, %.1f] m", MIN_DOWN, MAX_DOWN)
    logger.info("  ground_min:   %.2f m", GROUND_MIN_DOWN)

    # 初始化 ROS 桥接
    fastlio = init_ros_bridge()

    logger.info("Waiting for FAST-LIO data...")
    while not fastlio.initialized:
        time.sleep(0.05)
    logger.info("FAST-LIO ready. Starting visualization...")

    if args.no_display:
        logger.info("Display disabled. Press Ctrl+C to exit.")
        try:
            while not rospy.is_shutdown():
                time.sleep(0.5)
        except KeyboardInterrupt:
            pass
    else:
        live_display(fastlio, save_dir=args.save_dir, max_frames=args.max_frames)

    rospy.signal_shutdown("done")
    logger.info("Done.")


if __name__ == "__main__":
    main()
