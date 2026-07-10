#!/usr/bin/env python3
"""
capture_fastlio_frame.py — 交互式保存 FAST-LIO 单帧点云为 rosbag (ROS1 Noetic 版)
==================================================================================

逐帧捕捉 FAST-LIO 发布的 /cloud_registered (去畸变点云) 和 /Odometry (位姿),
每次按 Enter 将当前最新一帧保存为一个独立的 rosbag (.bag 格式)。

用法:
  source /opt/ros/noetic/setup.bash
  source ~/fast_lio_ws/devel/setup.bash
  python3 scripts/capture_fastlio_frame.py

  # 可选参数:
  python3 scripts/capture_fastlio_frame.py --output-dir bags/my_frames

输出结构:
  bags/capture_frames/
    frame_0000_20260609_143025/
      metadata.yaml
      frame.bag
    frame_0001_20260609_143032/
      metadata.yaml
      frame.bag
    ...

回放某一帧:
  rosbag play bags/capture_frames/frame_0000_20260609_143025/frame.bag
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
import threading
from pathlib import Path

import rospy
from nav_msgs.msg import Odometry
from sensor_msgs.msg import PointCloud2

import rosbag
import yaml


class FrameCapture:
    """订阅 FAST-LIO 输出, 缓存最新帧, 按需写 rosbag."""

    def __init__(self):
        rospy.init_node("fastlio_frame_capture")

        self._cloud_lock = threading.Lock()
        self._odom_lock = threading.Lock()

        self.latest_cloud = None  # PointCloud2 | None
        self.latest_odom = None   # Odometry | None

        rospy.Subscriber("/cloud_registered", PointCloud2, self._cloud_cb)
        rospy.Subscriber("/Odometry", Odometry, self._odom_cb)
        logger = logging.getLogger("FrameCapture")
        logger.info("Subscribed: /cloud_registered + /Odometry")
        self._logger = logger

    def _cloud_cb(self, msg: PointCloud2):
        with self._cloud_lock:
            self.latest_cloud = msg

    def _odom_cb(self, msg: Odometry):
        with self._odom_lock:
            self.latest_odom = msg

    def grab(self):
        """线程安全地获取最新 cloud + odometry 快照."""
        with self._cloud_lock:
            cloud = self.latest_cloud
        with self._odom_lock:
            odom = self.latest_odom
        return cloud, odom


def _write_bag(output_dir: str, cloud_msg, odom_msg, logger) -> bool:
    """
    将一帧 cloud + odometry 消息写入 rosbag (.bag).

    返回 True 表示写入成功, False 表示全部为空.
    """
    if cloud_msg is None and odom_msg is None:
        return False

    os.makedirs(output_dir, exist_ok=True)
    bag_path = os.path.join(output_dir, "frame.bag")

    bag = rosbag.Bag(bag_path, 'w')

    if cloud_msg is not None:
        bag.write("/cloud_registered", cloud_msg)
    if odom_msg is not None:
        bag.write("/Odometry", odom_msg)

    bag.close()

    # 写 metadata.yaml
    meta = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "topics": [],
    }
    if cloud_msg is not None:
        meta["topics"].append("/cloud_registered")
    if odom_msg is not None:
        meta["topics"].append("/Odometry")
    with open(os.path.join(output_dir, "metadata.yaml"), "w") as f:
        yaml.dump(meta, f)

    return True


def _input_reader(stop_event: threading.Event, capture_event: threading.Event):
    """后台线程: 读取 stdin, 按 Enter 触发 capture, 按 q 触发 stop."""
    print("  [Enter] 保存当前帧   q+[Enter] 退出")
    print()
    try:
        while not stop_event.is_set():
            line = sys.stdin.readline()
            if not line:
                break
            line = line.strip().lower()
            if line == "q":
                stop_event.set()
                break
            elif line == "":
                capture_event.set()
    except EOFError:
        pass


def main():
    parser = argparse.ArgumentParser(
        description="FAST-LIO 单帧 rosbag 交互式采集"
    )
    parser.add_argument(
        "--output-dir", default="bags/capture_frames",
        help="输出根目录 (默认: bags/capture_frames)",
    )
    parser.add_argument(
        "--also-raw", action="store_true",
        help="同时订阅 /livox/lidar 原始点云 (如果可用)",
    )
    args = parser.parse_args()

    node = FrameCapture()  # rospy.init_node() called inside
    # ROS1 subscribers need rospy.spin() in background
    spin_thread = threading.Thread(target=rospy.spin, daemon=True)
    spin_thread.start()

    log = logging.getLogger("FrameCapture")

    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    # ---- banner ----
    print()
    print("=" * 55)
    print("  FAST-LIO 单帧 Rosbag 采集 (ROS1)")
    print("=" * 55)
    print(f"  输出目录: {output_root.resolve()}")
    print()
    print("  话题:")
    print("    /cloud_registered  — 去畸变点云 (世界坐标)")
    print("    /Odometry          — 6-DoF 位姿")
    print("=" * 55)

    capture_event = threading.Event()
    stop_event = threading.Event()

    input_thread = threading.Thread(
        target=_input_reader, args=(stop_event, capture_event), daemon=True,
    )
    input_thread.start()

    frame_count = 0

    try:
        while not rospy.is_shutdown() and not stop_event.is_set():
            cloud, odom = node.grab()
            # ---- 状态行 ----
            parts = [f"[{frame_count} saved]"]
            if cloud is not None:
                n = cloud.width * cloud.height
                parts.append(f"cloud: {n} pts")
            else:
                parts.append("cloud: (waiting)")
            if odom is not None:
                p = odom.pose.pose.position
                parts.append(f"pose: ({p.x:.1f}, {p.y:.1f}, {p.z:.1f})")
            else:
                parts.append("pose: (waiting)")
            print("\r" + "  ".join(parts), end="", flush=True)

            if not capture_event.is_set():
                time.sleep(0.1)
                continue

            # ---- 保存当前帧 ----
            capture_event.clear()
            cloud, odom = node.grab()

            if cloud is None and odom is None:
                print("\n[WARN] 尚无数据, 跳过...")
                continue

            ts = time.strftime("%Y%m%d_%H%M%S")
            frame_dir = output_root / f"frame_{frame_count:04d}_{ts}"

            print(f"\n[Saving] → {frame_dir.name}")
            try:
                ok = _write_bag(str(frame_dir), cloud, odom, log)
            except RuntimeError as exc:
                print(f"[FAIL] {exc}")
                continue
            if ok:
                frame_count += 1
                n_pts = cloud.width * cloud.height if cloud is not None else 0
                print(f"[ OK ]  frame_{frame_count - 1:04d}  "
                      f"({n_pts} pts)  total={frame_count}")
                log.info(f"Saved frame {frame_count - 1:04d} → {frame_dir}")
            print()

    except KeyboardInterrupt:
        print("\n")
    finally:
        stop_event.set()
        rospy.signal_shutdown("interrupted")
        print(f"Done. 共采集 {frame_count} 帧.")
        print(f"输出: {output_root.resolve()}")


if __name__ == "__main__":
    main()
