#!/usr/bin/env python3
"""
visualize_fastlio_bag.py — 可视化 FAST-LIO 单帧 rosbag 点云 (ROS1 Noetic 版)
============================================================================

读取 capture_fastlio_frame.py 采集的单帧 rosbag (.bag),
显示去畸变点云 3D 散点图 (按高程着色) 和机体位姿。

用法:
  source /opt/ros/noetic/setup.bash
  python3 scripts/visualize_fastlio_bag.py bags/capture_frames/frame_0000_20260609_165119/frame.bag

  # 交互模式 (3D 可旋转缩放):
  python3 scripts/visualize_fastlio_bag.py <bag_path> --interactive

  # 保存图片不显示窗口:
  python3 scripts/visualize_fastlio_bag.py <bag_path> --save vis.png

  # 对比多个帧:
  python3 scripts/visualize_fastlio_bag.py frame_0000/frame.bag frame_0001/frame.bag

输出:
  ┌─────────────────────────┬─────────────────────────┐
  │  3D 点云 (高程着色)       │  俯视图 (BEV)            │
  │  ● = 点云                │  白色 = 点密度            │
  │  ▲ = 机体位置/朝向        │  红色箭头 = 机体朝向      │
  └─────────────────────────┴─────────────────────────┘
"""

from __future__ import annotations

import argparse
import sys
import math
import logging
from pathlib import Path

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s",
)
logger = logging.getLogger("VizBag")


# ============================================================
# 解析 PointCloud2 → numpy (与 pipeline.py / FastLIOInterface 一致)
# ============================================================
def _pc2_to_numpy(msg) -> np.ndarray:
    """sensor_msgs/PointCloud2 → (N, 3) float32 数组, 滤除 NaN."""
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


def _quat_to_yaw(x, y, z, w) -> float:
    """四元数 → yaw 角 (rad)."""
    siny = 2.0 * (w * z + x * y)
    cosy = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny, cosy)


# ============================================================
# Rosbag 读取 (ROS1)
# ============================================================
def read_bag(bag_path: str) -> tuple[np.ndarray | None, np.ndarray | None]:
    """
    读取 ROS1 rosbag, 返回 (cloud_pts, odom_pose).

    cloud_pts: (N, 3) float32 或 None
    odom_pose: [x, y, z, roll, pitch, yaw] 或 None
    """
    import rosbag
    from sensor_msgs.msg import PointCloud2
    from nav_msgs.msg import Odometry

    bag = rosbag.Bag(str(bag_path), 'r')

    cloud_pts = None
    odom_pose = None

    for topic_name, msg, t in bag.read_messages(
        topics=["/cloud_registered", "/Odometry"]
    ):
        if topic_name == "/cloud_registered":
            cloud_pts = _pc2_to_numpy(msg)

        elif topic_name == "/Odometry":
            p = msg.pose.pose.position
            q = msg.pose.pose.orientation
            yaw = _quat_to_yaw(q.x, q.y, q.z, q.w)
            # roll/pitch 从四元数近似
            sinr = 2.0 * (q.w * q.x + q.y * q.z)
            cosr = 1.0 - 2.0 * (q.x * q.x + q.y * q.y)
            roll = math.atan2(sinr, cosr)
            sinp = 2.0 * (q.w * q.y - q.z * q.x)
            pitch = math.asin(max(-1.0, min(1.0, sinp)))
            odom_pose = np.array([p.x, p.y, p.z, roll, pitch, yaw], dtype=np.float32)

    bag.close()
    return cloud_pts, odom_pose


# ============================================================
# 可视化
# ============================================================
def _plot_3d(ax, cloud: np.ndarray, pose: np.ndarray | None, title: str):
    """3D 散点图: 点云按 Z 高程着色, 机体位姿标记."""
    if cloud is not None and len(cloud) > 0:
        z = cloud[:, 2]
        z_min, z_max = z.min(), z.max()
        z_norm = np.clip((z - z_min) / (z_max - z_min + 1e-8), 0, 1)
        import matplotlib.pyplot as _plt
        colors = _plt.cm.viridis(z_norm)
        ax.scatter(
            cloud[:, 0], cloud[:, 1], cloud[:, 2],
            c=colors, s=0.3, alpha=0.6, marker=".",
        )
    else:
        logger.warning("Empty point cloud for %s", title)

    if pose is not None:
        px, py, pz = pose[0], pose[1], pose[2]
        yaw = pose[5]
        arrow_len = 2.0
        dx = arrow_len * math.cos(yaw)
        dy = arrow_len * math.sin(yaw)
        ax.scatter([px], [py], [pz], c="red", s=80, marker="^", edgecolors="k")
        ax.quiver(px, py, pz, dx, dy, 0,
                  color="red", linewidth=2, arrow_length_ratio=0.2)

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title(title)


def _plot_bev(ax, cloud: np.ndarray, pose: np.ndarray | None, title: str,
              grid_res: float = 0.1):
    """俯视密度图: 2D 栅格点计数 + 机体位姿."""
    if cloud is not None and len(cloud) > 0:
        x, y = cloud[:, 0], cloud[:, 1]
        x_min, x_max = x.min(), x.max()
        y_min, y_max = y.min(), y.max()
        x_bins = max(10, int((x_max - x_min) / grid_res))
        y_bins = max(10, int((y_max - y_min) / grid_res))
        hist, x_edges, y_edges = np.histogram2d(x, y, bins=[x_bins, y_bins])
        ax.imshow(
            hist.T, origin="lower",
            extent=[x_min, x_max, y_min, y_max],
            cmap="hot", aspect="equal", interpolation="nearest",
        )
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
    else:
        ax.set_xlim(-30, 30)
        ax.set_ylim(-30, 30)

    if pose is not None:
        px, py, yaw = pose[0], pose[1], pose[5]
        arrow_len = 2.0
        dx = arrow_len * math.cos(yaw)
        dy = arrow_len * math.sin(yaw)
        ax.scatter(px, py, c="cyan", s=100, marker="^", edgecolors="k", zorder=5)
        ax.arrow(px, py, dx, dy, color="cyan", width=0.5,
                 head_width=1.5, head_length=1.0, zorder=5)

    ax.set_title(title)


def visualize_single(bag_dir: str, save_path: str | None = None,
                     interactive: bool = False):
    """可视化单个 rosbag 帧 (3D + BEV 并排)."""
    import matplotlib
    if not interactive and save_path is not None:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import mpl_toolkits.mplot3d  # noqa: F401  注册 3D projection

    cloud, pose = read_bag(bag_dir)
    bag_name = Path(bag_dir).name

    n_pts = len(cloud) if cloud is not None else 0
    pose_str = ""
    if pose is not None:
        pose_str = (f"pos=({pose[0]:.1f},{pose[1]:.1f},{pose[2]:.1f}) "
                    f"yaw={math.degrees(pose[5]):.0f}°")
    logger.info("%s: %d pts  %s", bag_name, n_pts, pose_str)

    fig = plt.figure(figsize=(14, 6))

    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    _plot_3d(ax1, cloud, pose, f"{bag_name}\n3D View ({n_pts} pts)")

    ax2 = fig.add_subplot(1, 2, 2)
    _plot_bev(ax2, cloud, pose, f"{bag_name}\nBEV ({n_pts} pts)")

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Saved → %s", save_path)

    if interactive or save_path is None:
        plt.show()
    else:
        plt.close(fig)


def visualize_multi(bag_dirs: list[str], save_path: str | None = None):
    """多帧 3D 并排对比."""
    import matplotlib
    if save_path is not None:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import mpl_toolkits.mplot3d  # noqa: F401  注册 3D projection

    n = len(bag_dirs)
    cols = min(3, n)
    rows = (n + cols - 1) // cols

    fig = plt.figure(figsize=(6 * cols, 5 * rows))

    for i, bag_dir in enumerate(bag_dirs):
        cloud, pose = read_bag(bag_dir)
        bag_name = Path(bag_dir).name
        n_pts = len(cloud) if cloud is not None else 0
        pose_str = ""
        if pose is not None:
            pose_str = (f"pos=({pose[0]:.1f},{pose[1]:.1f},{pose[2]:.1f}) "
                        f"yaw={math.degrees(pose[5]):.0f}°")
        logger.info("%s: %d pts  %s", bag_name, n_pts, pose_str)

        ax = fig.add_subplot(rows, cols, i + 1, projection="3d")
        _plot_3d(ax, cloud, pose, f"{bag_name} ({n_pts} pts)")

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Saved → %s", save_path)

    plt.show()


# ============================================================
# CLI
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description="可视化 FAST-LIO 单帧 rosbag 点云"
    )
    parser.add_argument(
        "bags", nargs="+",
        help="rosbag 目录路径 (一个或多个)",
    )
    parser.add_argument(
        "--save", default=None,
        help="保存图片到指定路径 (不显示窗口)",
    )
    parser.add_argument(
        "--interactive", action="store_true",
        help="3D 交互模式 (可旋转缩放, 阻塞直到关闭窗口)",
    )
    args = parser.parse_args()

    # 验证所有路径
    for bag_dir in args.bags:
        p = Path(bag_dir)
        if not p.is_dir():
            logger.error("Not a directory: %s", bag_dir)
            sys.exit(1)
        if not (p / "metadata.yaml").exists():
            logger.error("No metadata.yaml in %s — not a rosbag2 directory?", bag_dir)
            sys.exit(1)

    if len(args.bags) == 1:
        visualize_single(
            args.bags[0],
            save_path=args.save,
            interactive=args.interactive,
        )
    else:
        visualize_multi(args.bags, save_path=args.save)


if __name__ == "__main__":
    main()
