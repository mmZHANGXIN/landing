#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Orin Landing — 原始 Livox 点云实时回放与 3D 可视化 (ROS1 Noetic)
================================================================
实时订阅 /livox/lidar (livox_ros_driver2/CustomMsg), 将 CustomMsg.points
的 x/y/z 转换为 (N, 3) float32 点云, 用与 replay_bag_offline.py 一致的
Matplotlib 交互式 3D 窗口逐帧刷新显示。

当前脚本不运行 HALSS 与深度投影: 只打开 3D 点云窗口, 不创建语义图
和深度图窗口。

两种消息模式 (由 --msg-type 选择):
  custom       原始点云 (默认): 订阅 /livox/lidar (CustomMsg), 不做外参
               变换 / 姿态 leveling / ROI 裁剪 / 下采样, 窗口标题
               "4.Raw Livox Point Cloud", 坐标轴标注原始 Livox 坐标系
               (x/y/z (m)), 与 replay_bag_offline.py 中 level-body
               (x forward / y lateral / z down) 的坐标语义区分。
  pointcloud2  去畸变点云: 订阅 /cloud_registered_body (sensor_msgs/
               PointCloud2, FAST-LIO 去畸变输出), 窗口标题
               "4.Deskewed Point Cloud", 坐标轴 level-body 语义, 与
               replay_bag_offline.py 的 3D 窗口一致。

用法 (通常由 replay_orin_landing_bag.sh 启动):
  source /opt/ros/noetic/setup.bash          # 需要已运行 roscore 和 rosbag play
  source ~/livox_noetic_ws/devel/setup.bash  # livox_ros_driver2 的 ROS1 消息包
  python3 scripts/replay_raw_livox_visualization.py [--msg-type custom|pointcloud2]
                                                     [--topic <话题>] [--no-display]

依赖: rospy + livox_ros_driver2 (或 livox_ros_driver) 的 CustomMsg 消息类。
若 rospy 可用但报 "CustomMsg 消息包未安装", 说明 Orin 上只有 ROS2 的
livox_ws, 缺 ROS1 消息定义 — 编译最小消息包后重跑:
  source /opt/ros/noetic/setup.bash
  mkdir -p ~/livox_noetic_ws/src/livox_ros_driver2/msg
  cd ~/livox_noetic_ws/src/livox_ros_driver2
  cp ~/livox_ws/src/livox_ros_driver2/msg/*.msg msg/
  # CMakeLists.txt:
  #   cmake_minimum_required(VERSION 3.0.2)
  #   project(livox_ros_driver2)
  #   find_package(catkin REQUIRED COMPONENTS message_generation std_msgs rospy)
  #   add_message_files(FILES CustomMsg.msg CustomPoint.msg)
  #   generate_messages(DEPENDENCIES std_msgs)
  #   catkin_package(CATKIN_DEPENDS message_runtime std_msgs)
  # package.xml: build_depend message_generation, exec_depend message_runtime
  cd ~/livox_noetic_ws && catkin_make
  source devel/setup.bash
"""

from __future__ import annotations

import argparse
import logging
import sys

import numpy as np

# rospy / livox 消息仅在 ROS1 Python 环境可用; import 失败时降级为 None,
# 由 main() 给出明确报错 (解析函数不依赖 rospy, 可独立测试)。
try:
    import rospy
except ImportError:
    rospy = None

try:
    from livox_ros_driver2.msg import CustomMsg
    _MSG_PACKAGE = "livox_ros_driver2"
except ImportError:
    try:
        from livox_ros_driver.msg import CustomMsg
        _MSG_PACKAGE = "livox_ros_driver"
    except ImportError:
        CustomMsg = None
        _MSG_PACKAGE = None

# sensor_msgs/PointCloud2 (去畸变点云 /cloud_registered_body 用, 标准消息)
try:
    from sensor_msgs.msg import PointCloud2
except ImportError:
    PointCloud2 = None

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("RawLivoxReplay")


# ──────────────────────────────────────────────
# CustomMsg → numpy (原始 Livox 点云, 不做任何变换)
# ──────────────────────────────────────────────
def _custom_msg_to_numpy(msg) -> np.ndarray:
    """livox_ros_driver2/CustomMsg → (N, 3) float32, 滤除 NaN.

    仅取 points 的 x/y/z (原始 Livox 坐标系), 不做外参变换、leveling、
    ROI 裁剪或下采样。空消息 / 无 points 返回 (0, 3) 空数组。
    """
    points = getattr(msg, "points", None)
    if points is None or len(points) == 0:
        return np.empty((0, 3), dtype=np.float32)
    pts = np.array([[p.x, p.y, p.z] for p in points], dtype=np.float32)
    if len(pts) == 0:
        return pts
    return pts[np.isfinite(pts).all(axis=1)]


def _pointcloud2_to_numpy(msg) -> np.ndarray:
    """sensor_msgs/PointCloud2 → (N, 3) float32, 滤除 NaN.

    按 point_step 逐点取 x/y/z 字段 (处理 row_step 含 padding 的情况),
    不依赖 rospy 的 PointCloud2 工具函数, 可独立测试。
    """
    if msg is None or msg.width == 0 or msg.height == 0:
        return np.empty((0, 3), dtype=np.float32)
    point_step, row_step = int(msg.point_step), int(msg.row_step)
    fields = {f.name: f.offset for f in msg.fields}
    if not {"x", "y", "z"}.issubset(fields):
        return np.empty((0, 3), dtype=np.float32)
    n = int(msg.width) * int(msg.height)
    raw = np.frombuffer(msg.data, dtype=np.uint8)
    need = n * point_step
    if len(raw) < need:
        return np.empty((0, 3), dtype=np.float32)
    if point_step < 4 or row_step < point_step:
        return np.empty((0, 3), dtype=np.float32)
    data = raw[:need].reshape(n, point_step)
    pts = np.empty((n, 3), dtype=np.float32)
    for i, name in enumerate(("x", "y", "z")):
        off = fields[name]
        # (n, 4) uint8 → float32 列 (需连续内存才能 view)
        col = np.ascontiguousarray(data[:, off:off + 4]).view(np.float32).ravel()
        pts[:, i] = col
    ok = np.isfinite(pts).all(axis=1)
    return pts[ok]


def _stamp_to_sec(stamp) -> float:
    return float(stamp.secs) + float(stamp.nsecs) * 1e-9


# ──────────────────────────────────────────────
# 3D 点云窗口 (与 replay_bag_offline.py 的 3D 点云窗口完全一致)
# ──────────────────────────────────────────────
class RawLivoxVisualizer:
    """Livox 点云 3D 窗口 (原始或去畸变, 由构造参数决定).

    交互方式 / 刷新方式 / 散点样式 / inferno 着色 / 标题信息 / 生命周期管理
    与 replay_bag_offline.py 的 3D 点云窗口一致; 区别仅在于窗口标题与
    坐标轴语义由消息类型决定:
      - custom (原始点云):  "4.Raw Livox Point Cloud", 坐标轴 raw Livox
        坐标系 (x/y/z (m)), 不含任何变换;
      - pointcloud2 (去畸变): "4.Deskewed Point Cloud", 坐标轴 level-body
        语义 (x forward / y lateral / z down), 与 replay_bag_offline.py 一致.
    """

    def __init__(self, show_display: bool = True,
                 window_title: str = "4.Raw Livox Point Cloud",
                 axes_labels: tuple = ("x (m) — raw Livox frame",
                                       "y (m) — raw Livox frame",
                                       "z (m) — raw Livox frame")):
        self._mpl = None          # matplotlib.pyplot 模块 (惰性导入)
        self._pc_fig = None       # 3D 点云 figure / axes / scatter
        self._pc_ax = None
        self._pc_scatter = None
        self._window_title = window_title
        self._axes_labels = axes_labels
        if not show_display:
            return
        try:
            self._mpl = self._import_matplotlib()
        except Exception as exc:
            logger.warning(
                "[Vis] Matplotlib unavailable (%s); 3D point cloud window "
                "disabled.", exc)
            self._mpl = None

    def _import_matplotlib(self):
        """惰性导入 matplotlib; 优先 TkAgg (与 replay_bag_offline.py 一致)."""
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
        """创建 3D 点云窗口 (窗口标题与坐标轴语义由构造参数决定).

        所有资源先在局部变量中创建成功, 再一次性挂载到实例属性, 避免
        半初始化状态 (fig 已建但 ax 缺失时 update() 会崩溃).
        """
        self._mpl.ion()
        fig = self._mpl.figure(num=self._window_title, figsize=(6, 6))
        ax = fig.add_subplot(111, projection="3d")
        ax.set_xlabel(self._axes_labels[0])
        ax.set_ylabel(self._axes_labels[1])
        ax.set_zlabel(self._axes_labels[2])
        ax.set_title("0 pts")
        fig.canvas.draw_idle()
        fig.canvas.flush_events()
        # 全部成功后才挂载
        self._pc_fig, self._pc_ax = fig, ax

    def update(self, point_cloud, cloud_stamp):
        """刷新 3D 点云: 全量显示原始点云 (不下采样), 按 z 值着色.

        空点云/None 时清空散点集合并保留窗口; 只读输入数组, 不改写坐标.
        通过 draw_idle()/flush_events() 保持窗口可旋转缩放且不阻塞回放.
        """
        if self._mpl is None:
            return
        if self._pc_fig is None:
            try:
                self._init_pointcloud_window()
            except Exception as exc:
                # 窗口初始化失败 (如 matplotlib/numpy 版本不匹配导致
                # 3d projection 缺失): 禁用窗口降级为纯日志模式, 不向
                # rospy 回调抛异常刷屏
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
            # 禁用窗口降级为纯日志模式, 避免每个回调都抛 TclError 刷屏
            logger.warning("[Vis] Window closed or unavailable (%s); "
                           "point cloud window disabled.", exc)
            self._mpl = None

    def close(self):
        if self._pc_fig is not None:
            try:
                self._mpl.close(self._pc_fig)
            except Exception:
                pass
            self._pc_fig = None
            self._pc_ax = None
            self._pc_scatter = None


# ──────────────────────────────────────────────
# 主循环: 订阅 /livox/lidar, 逐帧刷新窗口
# ──────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Livox 点云实时回放 + 3D 可视化 "
                    "(订阅点云话题, 由 rosbag play 提供数据)"
    )
    parser.add_argument("--msg-type", type=str,
                        choices=["custom", "pointcloud2"], default="custom",
                        help="消息类型: custom=原始点云 CustomMsg "
                             "(默认, 4.Raw Livox Point Cloud), "
                             "pointcloud2=去畸变点云 PointCloud2 "
                             "(4.Deskewed Point Cloud)")
    parser.add_argument("--topic", type=str, default="/livox/lidar",
                        help="订阅的点云话题 (默认 /livox/lidar; "
                             "pointcloud2 模式常用 /cloud_registered_body)")
    parser.add_argument("--no-display", action="store_true",
                        help="仅订阅并打印日志, 不打开 Matplotlib 3D 窗口")
    parser.add_argument("--log-every", type=int, default=10,
                        help="每 N 帧打印一次日志 (默认 10)")
    args = parser.parse_args()

    if rospy is None:
        logger.error("rospy unavailable — ROS1 Python 环境未加载. "
                     "先执行: source /opt/ros/noetic/setup.bash")
        sys.exit(1)

    # ---- 按消息类型选择解析函数 / 消息类 / 窗口语义 ----
    if args.msg_type == "custom":
        if CustomMsg is None:
            logger.error("livox_ros_driver2 (或 livox_ros_driver) 的 "
                         "CustomMsg 消息包未安装, 无法订阅 /livox/lidar.")
            sys.exit(1)
        msg_cls = CustomMsg
        to_numpy = _custom_msg_to_numpy
        window_title = "4.Raw Livox Point Cloud"
        axes_labels = ("x (m) — raw Livox frame",
                       "y (m) — raw Livox frame",
                       "z (m) — raw Livox frame")
        logger.info("Mode: raw point cloud (%s)", _MSG_PACKAGE)
    else:
        if PointCloud2 is None:
            logger.error("sensor_msgs.msg.PointCloud2 不可用 — ROS1 环境未加载.")
            sys.exit(1)
        msg_cls = PointCloud2
        to_numpy = _pointcloud2_to_numpy
        window_title = "4.Deskewed Point Cloud"
        axes_labels = ("x forward (m)", "y lateral (m)", "z down (m)")
        logger.info("Mode: deskewed point cloud (PointCloud2)")

    rospy.init_node("replay_raw_livox_visualization")
    vis = RawLivoxVisualizer(show_display=not args.no_display,
                             window_title=window_title,
                             axes_labels=axes_labels)
    logger.info("Subscribed to %s", args.topic)

    frame_count = 0

    def on_pointcloud(msg):
        nonlocal frame_count
        pts = to_numpy(msg)
        stamp = (_stamp_to_sec(msg.header.stamp)
                 if hasattr(msg, "header") else None)
        frame_count += 1
        vis.update(pts, stamp)
        if frame_count % args.log_every == 0:
            logger.info("[%06d] pts=%d t=%.3fs",
                        frame_count, len(pts),
                        float(stamp) if stamp is not None else -1.0)

    rospy.Subscriber(args.topic, msg_cls, on_pointcloud, queue_size=10)

    try:
        rospy.spin()
    except KeyboardInterrupt:
        logger.info("[Replay] Interrupted by user.")
    finally:
        vis.close()
        logger.info("[Replay] Visualizer closed.")


if __name__ == "__main__":
    main()
