"""
FAST-LIO2 ROS1 接口
订阅 FAST-LIO2 发布的去畸变点云和里程计位姿。
"""

import numpy as np
import logging

logger = logging.getLogger("FastLIOInterface")


class FastLIOInterface:
    """
    FAST-LIO2 数据接收器

    在 ROS1 环境中通过回调接收数据，在非 ROS 环境中通过轮询获取最新值。

    订阅话题:
      /Odometry          (nav_msgs/Odometry)       — 6-DoF 位姿
      /cloud_registered  (sensor_msgs/PointCloud2) — 去畸变点云(世界坐标)
    """

    def __init__(self, use_ros: bool = True):
        self.use_ros = use_ros
        self._latest_pose = None       # [x, y, z, roll, pitch, yaw]
        self._latest_points = None     # (N, 3) np.ndarray
        self._latest_pose_stamp = None
        self._latest_points_stamp = None
        self._pose_seq = 0
        self._points_seq = 0
        self._initialized = False

        if use_ros:
            self._init_ros()

    @property
    def seq(self):
        """当前最新数据序列号 (取 pose 和 points 中较新的)"""
        return max(self._pose_seq, self._points_seq)

    @property
    def pose_seq(self):
        """最新位姿序列号。"""
        return self._pose_seq

    @property
    def points_seq(self):
        """最新点云序列号。"""
        return self._points_seq

    def _init_ros(self):
        """初始化 ROS1 订阅"""
        try:
            import rospy
            from nav_msgs.msg import Odometry
            from sensor_msgs.msg import PointCloud2

            self._rospy = rospy
            self._Odometry = Odometry
            self._PointCloud2 = PointCloud2
            logger.info("[FastLIO] ROS1 interface ready.")
        except ImportError as e:
            logger.warning(f"[FastLIO] ROS1 import failed: {e}. Using non-ROS mode.")
            self.use_ros = False

    def odometry_callback(self, msg):
        """ROS1 里程计回调"""
        # 位置
        pos = msg.pose.pose.position
        quat = msg.pose.pose.orientation
        # 四元数 → 欧拉角
        r, p, y = self._quat_to_euler(quat.x, quat.y, quat.z, quat.w)
        self._latest_pose = np.array([pos.x, pos.y, pos.z, r, p, y], dtype=np.float32)
        self._latest_pose_stamp = self._stamp_to_sec(msg.header.stamp)
        self._pose_seq += 1

    def pointcloud_callback(self, msg):
        """ROS1 点云回调 — 用 numpy 解析 PointCloud2"""
        try:
            # 查找 x, y, z 字段偏移
            field_offsets = {}
            for field in msg.fields:
                field_offsets[field.name] = field.offset

            if not all(k in field_offsets for k in ('x', 'y', 'z')):
                logger.error("[FastLIO] PointCloud2 missing x/y/z fields")
                return

            n_points = msg.width * msg.height
            if n_points == 0:
                return

            # 每点占 msg.point_step 字节, x/y/z 为 float32。用 structured dtype
            # 避免 byte-slice + view 在存在 padding 时解析错位。
            dtype = np.dtype({
                "names": ["x", "y", "z"],
                "formats": ["<f4", "<f4", "<f4"],
                "offsets": [field_offsets["x"], field_offsets["y"], field_offsets["z"]],
                "itemsize": msg.point_step,
            })
            arr = np.frombuffer(msg.data, dtype=dtype, count=n_points)
            pts = np.column_stack((arr["x"], arr["y"], arr["z"])).astype(np.float32, copy=False)

            # 过滤 NaN
            valid = np.isfinite(pts).all(axis=1)
            pts = pts[valid]

            if len(pts) > 0:
                self._latest_points = pts
                self._latest_points_stamp = self._stamp_to_sec(msg.header.stamp)
                self._points_seq += 1
                self._initialized = True
        except Exception as e:
            logger.error(f"[FastLIO] PointCloud callback error: {e}")

    def _quat_to_euler(self, x, y, z, w):
        """四元数 → 欧拉角 (roll, pitch, yaw)"""
        import math
        # roll
        sinr_cosp = 2.0 * (w * x + y * z)
        cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
        roll = math.atan2(sinr_cosp, cosr_cosp)
        # pitch
        sinp = 2.0 * (w * y - z * x)
        if abs(sinp) >= 1:
            pitch = math.copysign(math.pi / 2.0, sinp)
        else:
            pitch = math.asin(sinp)
        # yaw
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        return roll, pitch, yaw

    @property
    def pose(self) -> np.ndarray:
        """获取最新位姿 [x, y, z, roll, pitch, yaw]"""
        return self._latest_pose

    @property
    def points(self) -> np.ndarray:
        """获取最新去畸变点云 (N, 3)"""
        return self._latest_points

    @property
    def pose_stamp(self):
        """最新位姿 ROS header 时间戳 (秒)。"""
        return self._latest_pose_stamp

    @property
    def points_stamp(self):
        """最新点云 ROS header 时间戳 (秒)。"""
        return self._latest_points_stamp

    @property
    def sync_delta_ms(self):
        """最新点云与位姿 header 时间差绝对值 (毫秒)。"""
        if self._latest_pose_stamp is None or self._latest_points_stamp is None:
            return None
        return abs(self._latest_points_stamp - self._latest_pose_stamp) * 1000.0

    @property
    def initialized(self) -> bool:
        return self._initialized

    def set_pose(self, pose: np.ndarray, stamp=None):
        """非 ROS 模式下手动设置位姿"""
        self._latest_pose = pose
        self._latest_pose_stamp = stamp
        self._pose_seq += 1

    def set_points(self, points: np.ndarray, stamp=None):
        """非 ROS 模式下手动设置点云"""
        self._latest_points = points
        self._latest_points_stamp = stamp
        self._points_seq += 1
        self._initialized = True

    @staticmethod
    def _stamp_to_sec(stamp) -> float:
        return float(stamp.secs) + float(stamp.nsecs) * 1e-9
