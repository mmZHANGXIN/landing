"""
MAVROS 飞控通信封装
==================
通过 ROS 话题/服务控制 PX4 Offboard 模式。

坐标系约定 (对齐 test_mavros_velocity.py):
  - MAVROS /local_position/odom 原始为 ENU (x=east, y=north, z=up)
  - /mavros/setpoint_raw/local 以 FRAME_LOCAL_NED 发送, 实际对应 ENU 约定
  - 内部存储: uavPosENU / uavVelENU / uavYawENU (原始 ENU)
  - 向后兼容: uavPosNED / uavVelNED / uavAngEular (自动从 ENU 转换)

控制流程 (新):
  connect() → start_hold_stream() → wait_for_manual_offboard_and_arm()
  → 自动进入 pipeline 任务阶段 (GOTO_SAFE / DRL descent)

OFFBOARD 丢失 / disarmed 安全:
  - 状态回调检测到 OFFBOARD 退出或 disarm → 设置 _safety_fallback 标志
  - pipeline 应在每帧检查此标志并进入 HOLD/ABORT
"""

from __future__ import annotations

import asyncio
from collections import deque
import logging
import math
import threading
import time
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger("MAVROSController")


def _wrap_pi(angle_rad: float) -> float:
    return (angle_rad + math.pi) % (2.0 * math.pi) - math.pi


# ---------------------------------------------------------------------------
# 坐标系转换工具 (ENU ↔ NED)
# ---------------------------------------------------------------------------

def enu_to_ned_position(enu_x: float, enu_y: float, enu_z: float) -> Tuple[float, float, float]:
    """ENU (x=east, y=north, z=up) → NED (north, east, down)."""
    return (float(enu_y), float(enu_x), float(-enu_z))


def ned_to_enu_position(ned_n: float, ned_e: float, ned_d: float) -> Tuple[float, float, float]:
    """NED (north, east, down) → ENU (x=east, y=north, z=up)."""
    return (float(ned_e), float(ned_n), float(-ned_d))


def enu_to_ned_velocity(enu_vx: float, enu_vy: float, enu_vz: float) -> Tuple[float, float, float]:
    """ENU velocity → NED velocity."""
    return (float(enu_vy), float(enu_vx), float(-enu_vz))


def ned_to_enu_velocity(ned_vn: float, ned_ve: float, ned_vd: float) -> Tuple[float, float, float]:
    """NED velocity → ENU velocity."""
    return (float(ned_ve), float(ned_vn), float(-ned_vd))


def enu_yaw_to_ned_yaw(yaw_enu_rad: float) -> float:
    """ENU yaw (0=East, CCW+) → NED yaw (0=North, CCW+)."""
    return _wrap_pi(math.pi / 2.0 - yaw_enu_rad)


def ned_yaw_to_enu_yaw(yaw_ned_rad: float) -> float:
    """NED yaw (0=North, CCW+) → ENU yaw (0=East, CCW+)."""
    return _wrap_pi(math.pi / 2.0 - yaw_ned_rad)


# ---------------------------------------------------------------------------
# MAVROSController
# ---------------------------------------------------------------------------

class MAVROSController:
    """
    PX4 飞控 MAVROS 封装 — ENU 原生, 兼容 NED 旧接口。

    提供:
      - 状态读取 (armed, offboard, mode, odom ENU/NED, GPS)
      - setpoint_raw/local 持续流 (Offboard 心跳)
      - 位置/速度 setpoint (ENU 原生 + NED 兼容)
      - 手动 OFFBOARD + 解锁等待
      - 安全 fallback 检测
    """

    def __init__(
        self,
        mavros_ns: str = "/mavros",
        setpoint_rate_hz: float = 20.0,
        offboard_warmup_s: float = 2.0,
    ):
        self._mavros_ns = mavros_ns.rstrip("/")
        self._setpoint_rate_hz = float(setpoint_rate_hz)
        self._offboard_warmup_s = float(offboard_warmup_s)

        # ROS 句柄
        self._rospy = None
        self._node_initialized = False
        # Callbacks and the setpoint heartbeat share one atomic state snapshot.
        # RLock also permits _publish_current_position_hold() to account for a
        # publication while it already owns the state lock.
        self._lock = threading.RLock()

        # ---- 原始 ENU 状态 (对齐 test_mavros_velocity.py) ----
        self.uavPosENU = np.zeros(3, dtype=np.float32)       # [x_east, y_north, z_up]
        self.uavVelENU = np.zeros(3, dtype=np.float32)       # [vx_east, vy_north, vz_up]
        self.uavYawENU = 0.0                                  # ENU yaw (0=East, CCW+)
        self.uavYawRateENU = 0.0                              # measured yaw rate (rad/s)
        self.uavRollENU = 0.0
        self.uavPitchENU = 0.0
        self._local_odom_stamp = None
        self._odom_history = deque(maxlen=400)

        # ---- 向后兼容 NED 字段 (从 ENU 自动转换) ----
        self.uavPosNED = np.zeros(3, dtype=np.float32)       # [north, east, down]
        self.uavVelNED = np.zeros(3, dtype=np.float32)       # [vn, ve, vd]
        self.uavAngEular = np.zeros(3, dtype=np.float32)      # [roll, pitch, yaw] NED yaw
        self.uavAngRate = np.zeros(3, dtype=np.float32)       # [p, q, r]
        self.uavThrust = 0.0
        self.uavLatLon = (0.0, 0.0)                           # (lat, lon) 度
        self.gpsStatus = None
        self.gpsHorizontalAccuracyM = None
        self._gps_stamp = None

        self.isVehicleCrash = False
        self.landed_state_on_ground = False
        self.isArmed = None
        self.isOffboard = None
        self.flightMode = None
        self._mavros_connected = False

        # 安全 fallback: OFFBOARD 丢失或 disarm 时置位
        self._safety_fallback = False
        self._safety_fallback_reason = ""

        # 遥测就绪标志
        self._enu_ready = False
        self._gps_ready = False
        self._attitude_ready = False
        self._state_ready = False

        # 兼容旧就绪标志
        self._ned_ready = False

        # setpoint 发布
        self._setpoint_pub = None
        self._last_setpoint_time = None
        self._setpoint_thread = None
        self._setpoint_running = False
        self._current_setpoint = None  # (type_mask, vx, vy, vz, yaw, yaw_rate)
        self._setpoint_publish_count = 0
        self._setpoint_stream_started_at = None
        self._hold_stream_active = False  # True = 持续发送当前位置 hold
        # Optional mixed setpoint with independent position/velocity fields.
        # Existing APIs keep using _current_setpoint; this is only populated
        # for modes such as GOTO (velocity XY + position Z).
        self._current_mixed_setpoint = None

        # 最后发送的 setpoint (用于日志)
        self._last_sent_mavros_sp = np.zeros(3, dtype=np.float32)

        # 遥测 subscribers
        self._subs = []

        logger.info(
            "[MAVROS] Initialized: ns=%s rate=%.1fHz warmup=%.1fs (ENU-native)",
            self._mavros_ns, self._setpoint_rate_hz, self._offboard_warmup_s,
        )

    # ------------------------------------------------------------------
    # 连接 / 初始化
    # ------------------------------------------------------------------

    async def connect(self):
        """初始化 ROS 节点并订阅 MAVROS 话题."""
        try:
            import rospy
            from mavros_msgs.msg import State, ExtendedState, PositionTarget
            from mavros_msgs.srv import CommandBool, CommandBoolRequest, SetMode, SetModeRequest
            from geometry_msgs.msg import PoseStamped, TwistStamped
            from sensor_msgs.msg import NavSatFix
            from nav_msgs.msg import Odometry

            self._rospy = rospy
            self._State = State
            self._ExtendedState = ExtendedState
            self._PositionTarget = PositionTarget
            self._CommandBool = CommandBool
            self._CommandBoolRequest = CommandBoolRequest
            self._SetMode = SetMode
            self._SetModeRequest = SetModeRequest
            self._PoseStamped = PoseStamped
            self._TwistStamped = TwistStamped
            self._NavSatFix = NavSatFix
            self._Odometry = Odometry

            if not rospy.core.is_initialized():
                rospy.init_node("orin_mavros_controller", anonymous=False)
            self._node_initialized = True

        except ImportError as e:
            logger.error("[MAVROS] ROS/MAVROS import failed: %s", e)
            raise RuntimeError(f"MAVROS dependencies missing: {e}")

        # --- 订阅 ---
        ns = self._mavros_ns
        rospy = self._rospy

        self._subs.append(
            rospy.Subscriber(f"{ns}/state", self._State, self._state_cb, queue_size=10)
        )
        self._subs.append(
            rospy.Subscriber(f"{ns}/extended_state", self._ExtendedState, self._extended_state_cb, queue_size=10)
        )
        self._subs.append(
            rospy.Subscriber(
                f"{ns}/local_position/odom", self._Odometry, self._local_odom_cb, queue_size=10
            )
        )
        self._subs.append(
            rospy.Subscriber(
                f"{ns}/global_position/global", self._NavSatFix, self._gps_cb, queue_size=10
            )
        )

        # --- 发布 setpoint_raw/local ---
        self._setpoint_pub = rospy.Publisher(
            f"{ns}/setpoint_raw/local", self._PositionTarget, queue_size=10
        )

        # 等待 MAVROS 连接
        logger.info("[MAVROS] Waiting for MAVROS connection...")
        timeout = 15.0
        start = time.perf_counter()
        rate = rospy.Rate(10)
        while not rospy.is_shutdown() and (time.perf_counter() - start) < timeout:
            if self._mavros_connected:
                logger.info("[MAVROS] Connected to FCU!")
                break
            rate.sleep()
        else:
            if not self._mavros_connected:
                raise TimeoutError("[MAVROS] FCU connection timeout")

        # 等待状态和遥测就绪
        await self._wait_telemetry_ready(timeout_s=10.0)

        # 启动 setpoint 持续发布线程
        self._start_setpoint_stream()

        logger.info("[MAVROS] Init complete: armed=%s offboard=%s mode=%s (ENU-native)",
                     self.isArmed, self.isOffboard, self.flightMode)

    async def _wait_telemetry_ready(self, timeout_s: float = 10.0):
        """等待遥测通道就绪."""
        start = time.perf_counter()
        rate = self._rospy.Rate(20)
        while time.perf_counter() - start < timeout_s:
            if self._state_ready and self._enu_ready:
                logger.info("[MAVROS] Telemetry channels ready (gps=%s).", self._gps_ready)
                return
            rate.sleep()
        missing = []
        if not self._state_ready:
            missing.append("state")
        if not self._enu_ready:
            missing.append("enu_odom")
        logger.warning("[MAVROS] Telemetry partial after %.1fs; missing: %s "
                       "(gps=%s, indoor OK)", timeout_s, ",".join(missing), self._gps_ready)

    # ------------------------------------------------------------------
    # 话题回调 (ROS spinner 线程)
    # ------------------------------------------------------------------

    def _state_cb(self, msg):
        with self._lock:
            was_offboard = self.isOffboard
            was_armed = self.isArmed

            self._mavros_connected = getattr(msg, "connected", False)
            self.isArmed = bool(getattr(msg, "armed", False))
            self.flightMode = str(getattr(msg, "mode", ""))
            self.isOffboard = "OFFBOARD" in self.flightMode.upper()
            self._state_ready = True

            # ---- 安全 fallback 检测 ----
            if was_offboard and not self.isOffboard:
                self._safety_fallback = True
                self._safety_fallback_reason = "OFFBOARD_lost"
                logger.error("[MAVROS] SAFETY: OFFBOARD lost! mode=%s", self.flightMode)
            if was_armed and not self.isArmed:
                self._safety_fallback = True
                self._safety_fallback_reason = "disarmed"
                logger.error("[MAVROS] SAFETY: disarmed!")

    def _extended_state_cb(self, msg):
        with self._lock:
            landed_state = int(getattr(msg, "landed_state", 0))
            self.landed_state_on_ground = (landed_state == 1)
            self.isVehicleCrash = self.landed_state_on_ground

    def _local_odom_cb(self, msg: "Odometry"):
        """MAVROS /local_position/odom — 原始 ENU 存储, 同时维护 NED 兼容字段."""
        with self._lock:
            pose = msg.pose.pose
            twist = msg.twist.twist

            # ---- 原始 ENU ----
            enu_x = float(pose.position.x)  # east
            enu_y = float(pose.position.y)  # north
            enu_z = float(pose.position.z)  # up
            self.uavPosENU = np.array([enu_x, enu_y, enu_z], dtype=np.float32)

            enu_vx = float(twist.linear.x)
            enu_vy = float(twist.linear.y)
            enu_vz = float(twist.linear.z)
            self.uavVelENU = np.array([enu_vx, enu_vy, enu_vz], dtype=np.float32)
            self.uavYawRateENU = float(twist.angular.z)

            # ENU yaw
            q = pose.orientation
            _, _, yaw_enu = self._quat_to_euler(q.x, q.y, q.z, q.w)
            self.uavYawENU = float(yaw_enu)

            # ---- 兼容 NED 字段 ----
            ned_n, ned_e, ned_d = enu_to_ned_position(enu_x, enu_y, enu_z)
            self.uavPosNED = np.array([ned_n, ned_e, ned_d], dtype=np.float32)

            vn, ve, vd = enu_to_ned_velocity(enu_vx, enu_vy, enu_vz)
            self.uavVelNED = np.array([vn, ve, vd], dtype=np.float32)

            roll_enu, pitch_enu, _ = self._quat_to_euler(q.x, q.y, q.z, q.w)
            self.uavRollENU = float(roll_enu)
            self.uavPitchENU = float(pitch_enu)
            yaw_ned = enu_yaw_to_ned_yaw(yaw_enu)
            self.uavAngEular = np.array([pitch_enu, roll_enu, yaw_ned], dtype=np.float32)

            stamp = float(msg.header.stamp.to_sec()) if hasattr(msg.header.stamp, "to_sec") else (
                float(msg.header.stamp.secs) + float(msg.header.stamp.nsecs) * 1e-9
            )
            received_perf = time.perf_counter()
            self._local_odom_stamp = stamp
            self._odom_history.append({
                "stamp": stamp,
                "position_enu": self.uavPosENU.copy(),
                "velocity_enu": self.uavVelENU.copy(),
                "roll": self.uavRollENU,
                "pitch": self.uavPitchENU,
                "yaw_enu": self.uavYawENU,
                "yaw_rate_enu": self.uavYawRateENU,
                "received_perf": received_perf,
            })

            self._enu_ready = True
            self._ned_ready = True
            self._attitude_ready = True

    def get_odom_nearest(self, stamp_s: float, max_delta_ms: float = 100.0):
        """Return the PX4 odometry sample nearest a ROS timestamp."""
        with self._lock:
            if not self._odom_history:
                return None
            sample = min(self._odom_history, key=lambda item: abs(item["stamp"] - float(stamp_s)))
            delta_ms = abs(float(sample["stamp"]) - float(stamp_s)) * 1000.0
            if delta_ms > float(max_delta_ms):
                return None
            result = dict(sample)
            result["position_enu"] = sample["position_enu"].copy()
            result["velocity_enu"] = sample["velocity_enu"].copy()
            result["sync_ms"] = delta_ms
            return result

    def _gps_cb(self, msg: "NavSatFix"):
        with self._lock:
            lat = float(msg.latitude)
            lon = float(msg.longitude)
            self.gpsStatus = int(msg.status.status)
            stamp = msg.header.stamp
            self._gps_stamp = (
                float(stamp.to_sec()) if hasattr(stamp, "to_sec")
                else float(stamp.secs) + float(stamp.nsecs) * 1e-9
            )
            covariance_type = int(getattr(msg, "position_covariance_type", 0))
            covariance = list(getattr(msg, "position_covariance", []))
            if covariance_type != 0 and len(covariance) >= 5:
                horizontal_variance = max(float(covariance[0]), float(covariance[4]))
                self.gpsHorizontalAccuracyM = (
                    math.sqrt(horizontal_variance)
                    if math.isfinite(horizontal_variance) and horizontal_variance >= 0.0
                    else None
                )
            else:
                self.gpsHorizontalAccuracyM = None
            if self._valid_lat_lon(lat, lon) and self.gpsStatus >= 0:
                self.uavLatLon = (lat, lon)
                self._gps_ready = True
            else:
                self._gps_ready = False

    def gps_health(self, max_age_s: float = 2.0, max_horizontal_accuracy_m: float = 5.0):
        """Return ``(healthy, reason)`` for the latest MAVROS NavSatFix."""
        with self._lock:
            if not self._gps_ready or self.gpsStatus is None or self.gpsStatus < 0:
                return False, "gps_no_fix"
            if self._gps_stamp is None:
                return False, "gps_timestamp_missing"
            now_s = (
                float(self._rospy.Time.now().to_sec())
                if self._rospy is not None else time.time()
            )
            age_s = max(0.0, now_s - float(self._gps_stamp))
            if age_s > float(max_age_s):
                return False, f"gps_stale:{age_s:.2f}s"
            hacc = self.gpsHorizontalAccuracyM
            if hacc is not None and hacc > float(max_horizontal_accuracy_m):
                return False, f"gps_horizontal_accuracy:{hacc:.2f}m"
            return True, "ok" if hacc is not None else "ok_covariance_unknown"

    # ------------------------------------------------------------------
    # Setpoint 持续流 (Offboard 心跳)
    # ------------------------------------------------------------------

    def _start_setpoint_stream(self):
        """启动后台线程持续发布 setpoint_raw (Offboard 模式要求)."""
        if self._setpoint_running:
            return
        self._setpoint_running = True
        self._setpoint_stream_started_at = time.perf_counter()
        self._setpoint_thread = threading.Thread(
            target=self._setpoint_stream_loop, daemon=True
        )
        self._setpoint_thread.start()
        logger.info("[MAVROS] Setpoint stream started at %.1fHz", self._setpoint_rate_hz)

    def _stop_setpoint_stream(self):
        self._setpoint_running = False
        if self._setpoint_thread and self._setpoint_thread.is_alive():
            self._setpoint_thread.join(timeout=2.0)

    # PositionTarget type_mask 常量
    _MASK_IGNORE_PX = 1
    _MASK_IGNORE_PY = 2
    _MASK_IGNORE_PZ = 4
    _MASK_IGNORE_VX = 8
    _MASK_IGNORE_VY = 16
    _MASK_IGNORE_VZ = 32
    _MASK_IGNORE_AFX = 64
    _MASK_IGNORE_AFY = 128
    _MASK_IGNORE_AFZ = 256
    _MASK_IGNORE_YAW = 1024
    _MASK_IGNORE_YAW_RATE = 2048

    _MASK_VELOCITY_YAW = (
        _MASK_IGNORE_PX | _MASK_IGNORE_PY | _MASK_IGNORE_PZ
        | _MASK_IGNORE_AFX | _MASK_IGNORE_AFY | _MASK_IGNORE_AFZ
        | _MASK_IGNORE_YAW_RATE
    )  # = 2503

    _MASK_POSITION_YAW = (
        _MASK_IGNORE_VX | _MASK_IGNORE_VY | _MASK_IGNORE_VZ
        | _MASK_IGNORE_AFX | _MASK_IGNORE_AFY | _MASK_IGNORE_AFZ
        | _MASK_IGNORE_YAW_RATE
    )  # = 2552

    _MASK_VELOCITY_YAWRATE = (
        _MASK_IGNORE_PX | _MASK_IGNORE_PY | _MASK_IGNORE_PZ
        | _MASK_IGNORE_AFX | _MASK_IGNORE_AFY | _MASK_IGNORE_AFZ
        | _MASK_IGNORE_YAW
    )  # = 1479: 速度控制 + yaw_rate (角速度)

    _MASK_POSITION_YAWRATE = (
        _MASK_IGNORE_VX | _MASK_IGNORE_VY | _MASK_IGNORE_VZ
        | _MASK_IGNORE_AFX | _MASK_IGNORE_AFY | _MASK_IGNORE_AFZ
        | _MASK_IGNORE_YAW
    )  # = 1528: 位置控制 + yaw_rate (角速度)

    _MASK_LANDING_YAWRATE = (
        _MASK_IGNORE_PZ | _MASK_IGNORE_VX | _MASK_IGNORE_VY
        | _MASK_IGNORE_AFX | _MASK_IGNORE_AFY | _MASK_IGNORE_AFZ
        | _MASK_IGNORE_YAW
    )  # = 1516: 位置 XY + 速度 Z + yaw_rate (降落)

    _MASK_POSITION_XY_VELOCITY_Z_YAW = (
        _MASK_IGNORE_PZ | _MASK_IGNORE_VX | _MASK_IGNORE_VY
        | _MASK_IGNORE_AFX | _MASK_IGNORE_AFY | _MASK_IGNORE_AFZ
        | _MASK_IGNORE_YAW_RATE
    )  # = 2524: 位置 XY + 速度 Z + yaw

    _MASK_VELOCITY_XY_POSITION_Z_YAWRATE = (
        _MASK_IGNORE_PX | _MASK_IGNORE_PY | _MASK_IGNORE_VZ
        | _MASK_IGNORE_AFX | _MASK_IGNORE_AFY | _MASK_IGNORE_AFZ
        | _MASK_IGNORE_YAW
    )  # = 1507: 速度 XY + 位置 Z + yaw_rate (限速 GOTO)

    def _setpoint_stream_loop(self):
        """后台线程: 以固定频率发布 setpoint_raw."""
        rate = self._rospy.Rate(self._setpoint_rate_hz)
        while self._setpoint_running and not self._rospy.is_shutdown():
            try:
                if self._hold_stream_active:
                    self._publish_current_position_hold()
                else:
                    with self._lock:
                        sp = self._current_setpoint
                        mixed_sp = self._current_mixed_setpoint
                    if sp is None:
                        self._publish_setpoint(
                            type_mask=self._MASK_VELOCITY_YAW,
                            vx=0.0, vy=0.0, vz=0.0,
                            yaw=0.0, yaw_rate=0.0,
                        )
                    elif mixed_sp is not None:
                        self._publish_mixed_setpoint(*mixed_sp)
                    else:
                        self._publish_setpoint(*sp)
            except Exception as e:
                logger.warning("[MAVROS] Setpoint publish error: %s", e)
            rate.sleep()

    def _publish_current_position_hold(self):
        """发布当前位置 hold setpoint (ENU)."""
        with self._lock:
            if not self._enu_ready:
                return
            x, y, z = float(self.uavPosENU[0]), float(self.uavPosENU[1]), float(self.uavPosENU[2])
            yaw = float(self.uavYawENU)
        self._publish_setpoint(
            type_mask=self._MASK_POSITION_YAW,
            vx=x, vy=y, vz=z,
            yaw=yaw, yaw_rate=0.0,
        )
        self._last_sent_mavros_sp = np.array([x, y, z], dtype=np.float32)

    def _publish_setpoint(self, type_mask: int, vx: float, vy: float, vz: float,
                          yaw: float, yaw_rate: float = 0.0):
        """发布单帧 PositionTarget 到 /mavros/setpoint_raw/local (ENU 约定).

        同时填充 position 和 velocity 字段, PX4 根据 type_mask 决定使用哪些.
        支持混合模式 (如位置 XY + 速度 Z).
        """
        self._publish_mixed_setpoint(
            type_mask,
            float(vx), float(vy), float(vz),
            float(vx), float(vy), float(vz),
            float(yaw), float(yaw_rate),
        )

    def _publish_mixed_setpoint(
        self, type_mask: int,
        px: float, py: float, pz: float,
        vx: float, vy: float, vz: float,
        yaw: float, yaw_rate: float = 0.0,
    ):
        """Publish independent ENU position/velocity fields selected by type_mask."""
        msg = self._PositionTarget()
        msg.header.stamp = self._rospy.Time.now()
        msg.coordinate_frame = self._PositionTarget.FRAME_LOCAL_NED
        msg.type_mask = int(type_mask)

        msg.position.x = float(px)
        msg.position.y = float(py)
        msg.position.z = float(pz)
        msg.velocity.x = float(vx)
        msg.velocity.y = float(vy)
        msg.velocity.z = float(vz)

        # 同时填 yaw 和 yaw_rate — PX4 按 mask 位选择性读取
        msg.yaw = float(yaw)
        msg.yaw_rate = float(yaw_rate)

        self._last_sent_mavros_sp = np.array([vx, vy, vz], dtype=np.float32)
        try:
            self._setpoint_pub.publish(msg)
            with self._lock:
                self._setpoint_publish_count += 1
        except Exception:
            pass

    def _update_setpoint(self, type_mask: int, vx: float, vy: float, vz: float,
                         yaw: float, yaw_rate: float = 0.0):
        """更新后台流式发布的 setpoint 值 (ENU 约定)."""
        with self._lock:
            self._current_setpoint = (type_mask, vx, vy, vz, yaw, yaw_rate)
            self._current_mixed_setpoint = None
            self._last_setpoint_time = time.perf_counter()

    def _update_mixed_setpoint(
        self, type_mask: int,
        px: float, py: float, pz: float,
        vx: float, vy: float, vz: float,
        yaw: float, yaw_rate: float = 0.0,
    ):
        """Atomically update a setpoint with independent position/velocity values."""
        with self._lock:
            # Keep the compact tuple for status reporting and compatibility.
            self._current_setpoint = (type_mask, vx, vy, vz, yaw, yaw_rate)
            self._current_mixed_setpoint = (
                type_mask, px, py, pz, vx, vy, vz, yaw, yaw_rate
            )
            self._last_setpoint_time = time.perf_counter()

    def get_setpoint_status(self) -> dict:
        """Return an atomic snapshot used by experiment logging."""
        with self._lock:
            sp = self._current_setpoint
            updated = self._last_setpoint_time
            count = self._setpoint_publish_count
            started = self._setpoint_stream_started_at
        elapsed = 0.0 if started is None else max(0.0, time.perf_counter() - started)
        return {
            "type_mask": None if sp is None else int(sp[0]),
            "yaw_rate_rad_s": 0.0 if sp is None else float(sp[5]),
            "updated_monotonic_s": updated,
            "publish_rate_hz": 0.0 if elapsed <= 0.0 else float(count) / elapsed,
        }

    # ------------------------------------------------------------------
    # 手动 OFFBOARD + 解锁等待 (新流程)
    # ------------------------------------------------------------------

    def start_hold_stream(self):
        """启动当前位置 hold 流: 持续发布当前 ENU 位置作为 setpoint.

        在等待遥控器切 OFFBOARD+解锁期间调用, 确保 setpoint 持续发布.
        """
        if not self._setpoint_running:
            self._start_setpoint_stream()
        self._hold_stream_active = True
        # 重置安全标志
        self._safety_fallback = False
        self._safety_fallback_reason = ""
        logger.info("[MAVROS] Hold stream active (current position hold)")

    def stop_hold_stream(self):
        """停止当前位置 hold 流, 恢复手动 setpoint 控制."""
        self._hold_stream_active = False
        logger.info("[MAVROS] Hold stream stopped")

    async def wait_for_manual_offboard_and_arm(
        self,
        timeout_s: float = 120.0,
        post_arm_yaw_rate_rad_s=None,
    ) -> bool:
        """等待遥控器切 OFFBOARD 并解锁.

        阻塞直到:
          - isArmed=True AND isOffboard=True → 返回 True
          - 超时 → 返回 False

        期间持续发送当前位置 hold setpoint (由 start_hold_stream 启动).
        """
        logger.info(
            "[MAVROS] Waiting for manual OFFBOARD + arm (timeout=%.0fs)...", timeout_s
        )
        logger.info("[MAVROS] RC: arm first, then switch to OFFBOARD")

        start = time.perf_counter()
        rate = self._rospy.Rate(10)

        while time.perf_counter() - start < timeout_s:
            if self._rospy.is_shutdown():
                return False

            if self.isArmed and self.isOffboard:
                elapsed = time.perf_counter() - start
                logger.info(
                    "[MAVROS] >>> Manual OFFBOARD + arm detected! "
                    "armed=%s offboard=%s mode=%s elapsed=%.1fs",
                    self.isArmed, self.isOffboard, self.flightMode, elapsed,
                )
                # 预热: 等待 OFFBOARD 稳定
                await asyncio.sleep(self._offboard_warmup_s)
                # Atomically hand off from the ground-safe hold before
                # disabling hold_stream. Callers choose fixed yaw or yaw-rate.
                with self._lock:
                    x, y, z = (float(v) for v in self.uavPosENU)
                if post_arm_yaw_rate_rad_s is None:
                    with self._lock:
                        yaw = float(self.uavYawENU)
                    self._update_setpoint(
                        type_mask=self._MASK_POSITION_YAW,
                        vx=x, vy=y, vz=z,
                        yaw=yaw, yaw_rate=0.0,
                    )
                    handoff_text = f"yaw_hold={math.degrees(yaw):.1f}deg"
                else:
                    self._update_setpoint(
                        type_mask=self._MASK_POSITION_YAWRATE,
                        vx=x, vy=y, vz=z,
                        yaw=0.0,
                        yaw_rate=float(post_arm_yaw_rate_rad_s),
                    )
                    handoff_text = f"yaw_rate={float(post_arm_yaw_rate_rad_s):.3f}rad/s"
                self.stop_hold_stream()
                logger.info(
                    "[MAVROS] Post-arm position heartbeat: "
                    "ENU=(%.2f,%.2f,%.2f) %s @ %.1fHz",
                    x, y, z, handoff_text, self._setpoint_rate_hz,
                )
                return True

            if self.isOffboard and not self.isArmed:
                if int((time.perf_counter() - start) * 2) % 6 == 0:
                    logger.info("[MAVROS] OFFBOARD active, waiting for arm...")
            elif self.isArmed and not self.isOffboard:
                if int((time.perf_counter() - start) * 2) % 6 == 0:
                    logger.info("[MAVROS] Armed, waiting for OFFBOARD switch...")

            rate.sleep()

        logger.error("[MAVROS] Timeout waiting for OFFBOARD+arm after %.0fs", timeout_s)
        return False

    @property
    def safety_fallback(self) -> bool:
        """OFFBOARD 丢失或 disarm 时为 True. pipeline 应检查此标志."""
        return self._safety_fallback

    @property
    def safety_fallback_reason(self) -> str:
        return self._safety_fallback_reason

    def clear_safety_fallback(self):
        """清除安全 fallback 标志 (例如切换到 HOLD 后)."""
        self._safety_fallback = False
        self._safety_fallback_reason = ""

    # ------------------------------------------------------------------
    # 服务调用
    # ------------------------------------------------------------------

    def _call_service(self, service_name: str, service_type, request, timeout_s: float = 5.0):
        """同步调用 MAVROS service."""
        try:
            rospy = self._rospy
            rospy.wait_for_service(service_name, timeout=timeout_s)
            proxy = rospy.ServiceProxy(service_name, service_type)
            resp = proxy(request)
            return resp
        except self._rospy.ROSException as e:
            logger.error("[MAVROS] Service %s call failed: %s", service_name, e)
            raise

    async def arm(self):
        """通过 /mavros/cmd/arming 解锁 (不推荐 — 新流程由遥控器完成)."""
        ns = self._mavros_ns
        logger.info("[MAVROS] Arming via %s/cmd/arming ...", ns)
        req = self._CommandBoolRequest(value=True)

        max_retries = 10
        for attempt in range(1, max_retries + 1):
            resp = self._call_service(f"{ns}/cmd/arming", self._CommandBool, req)
            if resp.success:
                logger.info("[MAVROS] Armed successfully (attempt %d).", attempt)
                return
            if resp.result == 1:
                if attempt < max_retries:
                    logger.warning(
                        "[MAVROS] Arm temporarily rejected (attempt %d/%d), retrying...",
                        attempt, max_retries,
                    )
                    await asyncio.sleep(0.5)
                    continue
            raise RuntimeError(
                f"[MAVROS] Arm failed (attempt {attempt}): "
                f"success={resp.success} result={resp.result}"
            )

    async def disarm(self):
        """通过 /mavros/cmd/arming 上锁."""
        ns = self._mavros_ns
        logger.info("[MAVROS] Disarming via %s/cmd/arming ...", ns)
        req = self._CommandBoolRequest(value=False)
        resp = self._call_service(f"{ns}/cmd/arming", self._CommandBool, req)
        if not resp.success:
            logger.warning("[MAVROS] Disarm failed: %s", resp.result)
        else:
            logger.info("[MAVROS] Disarmed.")

    async def init_offboard(self):
        """切换到 Offboard 模式 (不推荐 — 新流程由遥控器完成)."""
        ns = self._mavros_ns
        logger.info(
            "[MAVROS] Initializing Offboard mode (warmup=%.1fs)...",
            self._offboard_warmup_s,
        )

        if not self._setpoint_running:
            self._start_setpoint_stream()

        self._update_setpoint(
            type_mask=self._MASK_VELOCITY_YAW,
            vx=0.0, vy=0.0, vz=0.0,
            yaw=0.0,
        )

        await asyncio.sleep(self._offboard_warmup_s)

        logger.info("[MAVROS] Requesting OFFBOARD mode...")
        req = self._SetModeRequest()
        req.custom_mode = "OFFBOARD"
        resp = self._call_service(f"{ns}/set_mode", self._SetMode, req)

        if not resp.mode_sent:
            raise RuntimeError("[MAVROS] Failed to set OFFBOARD mode")

        await asyncio.sleep(0.5)
        logger.info("[MAVROS] Offboard mode active: %s", self.isOffboard)

    # ------------------------------------------------------------------
    # 等待遥测
    # ------------------------------------------------------------------

    async def wait_for_home(self, timeout_s: float = 10.0) -> Tuple[np.ndarray, tuple, np.ndarray]:
        """等待遥测就绪，返回 (home_ned, (lat,lon), attitude_ned)."""
        start = time.perf_counter()
        rate = self._rospy.Rate(20)
        while time.perf_counter() - start < timeout_s:
            with self._lock:
                if self._ned_ready and self._gps_ready and self._attitude_ready:
                    return (
                        self.uavPosNED.copy(),
                        self.uavLatLon,
                        self.uavAngEular.copy(),
                    )
            rate.sleep()
        raise TimeoutError(
            f"[MAVROS] Home telemetry not ready: "
            f"ned={self._ned_ready} gps={self._gps_ready} attitude={self._attitude_ready}"
        )

    async def wait_for_local_pose(self, timeout_s: float = 10.0) -> Tuple[np.ndarray, np.ndarray]:
        """等待本地 NED pose 和 attitude 遥测就绪 (兼容旧接口)."""
        start = time.perf_counter()
        rate = self._rospy.Rate(20)
        while time.perf_counter() - start < timeout_s:
            with self._lock:
                if self._ned_ready and self._attitude_ready:
                    return self.uavPosNED.copy(), self.uavAngEular.copy()
            rate.sleep()
        raise TimeoutError(
            f"[MAVROS] Local telemetry not ready: "
            f"ned={self._ned_ready} attitude={self._attitude_ready}"
        )

    async def wait_for_local_pose_enu(self, timeout_s: float = 10.0) -> Tuple[np.ndarray, float]:
        """等待本地 ENU pose 和 yaw 遥测就绪."""
        start = time.perf_counter()
        rate = self._rospy.Rate(20)
        while time.perf_counter() - start < timeout_s:
            with self._lock:
                if self._enu_ready:
                    return self.uavPosENU.copy(), float(self.uavYawENU)
            rate.sleep()
        raise TimeoutError(f"[MAVROS] ENU telemetry not ready: enu={self._enu_ready}")

    # ------------------------------------------------------------------
    # 飞控指令 — ENU 原生
    # ------------------------------------------------------------------

    async def send_velocity_enu_yaw(self, vx: float, vy: float, vz: float,
                                     yaw_deg: float = 0.0):
        """发送 ENU 速度 + yaw setpoint (对齐 test_mavros_velocity.py).

        :param vx: 东向速度 (m/s, ENU)
        :param vy: 北向速度 (m/s, ENU)
        :param vz: 上向速度 (m/s, ENU, 正值=上升)
        :param yaw_deg: yaw 角 (degree, ENU: 0=East)
        """
        yaw_rad = math.radians(float(yaw_deg))
        self._update_setpoint(
            type_mask=self._MASK_VELOCITY_YAW,
            vx=float(vx),
            vy=float(vy),
            vz=float(vz),
            yaw=yaw_rad,
            yaw_rate=0.0,
        )

    async def send_velocity_enu_yaw_rate(self, vx: float, vy: float, vz: float,
                                          yaw_rate_rad_s: float = 0.0):
        """发送 ENU 速度 + yaw_rate setpoint (角速度控制).

        :param vx: 东向速度 (m/s, ENU)
        :param vy: 北向速度 (m/s, ENU)
        :param vz: 上向速度 (m/s, ENU, 正值=上升)
        :param yaw_rate_rad_s: 偏航角速度 (rad/s, CCW+)
        """
        self._update_setpoint(
            type_mask=self._MASK_VELOCITY_YAWRATE,
            vx=float(vx),
            vy=float(vy),
            vz=float(vz),
            yaw=0.0,
            yaw_rate=float(yaw_rate_rad_s),
        )

    async def send_position_enu_yaw(self, x: float, y: float, z: float,
                                     yaw_deg: float = 0.0):
        """发送 ENU 位置 + yaw setpoint.

        :param x: 东向 (m, ENU)
        :param y: 北向 (m, ENU)
        :param z: 上向 (m, ENU, 正值=上升)
        :param yaw_deg: yaw 角 (degree, ENU: 0=East)
        """
        yaw_rad = math.radians(float(yaw_deg))
        self._update_setpoint(
            type_mask=self._MASK_POSITION_YAW,
            vx=float(x),
            vy=float(y),
            vz=float(z),
            yaw=yaw_rad,
            yaw_rate=0.0,
        )

    async def send_position_enu_yaw_rate(self, x: float, y: float, z: float,
                                          yaw_rate_rad_s: float = 0.0):
        """发送 ENU 位置 + yaw_rate setpoint (位置控制 + 角速度偏航).

        :param x: 东向 (m, ENU)
        :param y: 北向 (m, ENU)
        :param z: 上向 (m, ENU, 正值=上升)
        :param yaw_rate_rad_s: 偏航角速度 (rad/s, CCW+)
        """
        self._update_setpoint(
            type_mask=self._MASK_POSITION_YAWRATE,
            vx=float(x),
            vy=float(y),
            vz=float(z),
            yaw=0.0,
            yaw_rate=float(yaw_rate_rad_s),
        )

    async def send_velocity_xy_position_z_enu_yaw_rate(
        self, vx: float, vy: float, z: float,
        yaw_rate_rad_s: float = 0.0,
    ):
        """发送 ENU 混合 setpoint：限速速度 XY + 高度位置 Z + yaw_rate。"""
        self._update_mixed_setpoint(
            type_mask=self._MASK_VELOCITY_XY_POSITION_Z_YAWRATE,
            px=0.0,
            py=0.0,
            pz=float(z),
            vx=float(vx),
            vy=float(vy),
            vz=0.0,
            yaw=0.0,
            yaw_rate=float(yaw_rate_rad_s),
        )

    async def send_position_xy_velocity_z_enu_yaw_rate(
            self, x: float, y: float, vz: float,
            yaw_rate_rad_s: float = 0.0):
        """发送 ENU 混合 setpoint（位置 XY + 速度 Z + yaw_rate）。

        PX4 锁定 XY 位置，以速度 vz 控制升降，同时以 yaw_rate 旋转。

        :param x: 东向位置 (m, ENU)
        :param y: 北向位置 (m, ENU)
        :param vz: 上向速度 (m/s, ENU, 负值=下降)
        :param yaw_rate_rad_s: 偏航角速度 (rad/s, CCW+)
        """
        self._update_setpoint(
            type_mask=self._MASK_LANDING_YAWRATE,
            vx=float(x),
            vy=float(y),
            vz=float(vz),
            yaw=0.0,
            yaw_rate=float(yaw_rate_rad_s),
        )

    async def send_position_xy_velocity_z_enu_yaw(
            self, x: float, y: float, vz: float,
            yaw_deg: float = 0.0):
        """发送 ENU 混合 setpoint（位置 XY + 速度 Z + 固定 yaw）。"""
        self._update_setpoint(
            type_mask=self._MASK_POSITION_XY_VELOCITY_Z_YAW,
            vx=float(x),
            vy=float(y),
            vz=float(vz),
            yaw=math.radians(float(yaw_deg)),
            yaw_rate=0.0,
        )

    async def send_landing_enu_yaw_rate(self, x: float, y: float, vz: float,
                                         yaw_rate_rad_s: float = 0.0):
        """兼容接口：固定 XY 并以速度 Z 降落。"""
        await self.send_position_xy_velocity_z_enu_yaw_rate(
            x, y, vz, yaw_rate_rad_s
        )

    # ------------------------------------------------------------------
    # 飞控指令 — NED 兼容 (自动转换 → ENU)
    # ------------------------------------------------------------------

    async def send_velocity_ned_yaw(self, vn: float, ve: float, vd: float,
                                     yaw_ned_deg: float = 0.0):
        """发送 NED 速度 setpoint (自动转换为 ENU 后发布).

        :param vn: 北向速度 (m/s, NED)
        :param ve: 东向速度 (m/s, NED)
        :param vd: 下向速度 (m/s, NED, 正值=下降)
        :param yaw_ned_deg: yaw 角 (degree, NED: 0=North)
        """
        enu_vx, enu_vy, enu_vz = ned_to_enu_velocity(float(vn), float(ve), float(vd))
        enu_yaw_deg = math.degrees(ned_yaw_to_enu_yaw(math.radians(float(yaw_ned_deg))))
        await self.send_velocity_enu_yaw(enu_vx, enu_vy, enu_vz, enu_yaw_deg)

    async def send_velocity_ned(self, vx: float, vy: float, vz: float,
                                yaw_deg: float = 0.0):
        """兼容旧调用 (NED → ENU 自动转换)."""
        await self.send_velocity_ned_yaw(vx, vy, vz, yaw_deg)

    async def send_position_ned(self, n: float, e: float, d: float,
                                 yaw_ned_deg: float = 0.0):
        """发送 NED 位置 setpoint (自动转换为 ENU 后发布).

        :param n: 北向 (m, NED)
        :param e: 东向 (m, NED)
        :param d: 下向 (m, NED, 正值)
        :param yaw_ned_deg: yaw 角 (degree, NED: 0=North)
        """
        enu_x, enu_y, enu_z = ned_to_enu_position(float(n), float(e), float(d))
        enu_yaw_deg = math.degrees(ned_yaw_to_enu_yaw(math.radians(float(yaw_ned_deg))))
        await self.send_position_enu_yaw(enu_x, enu_y, enu_z, enu_yaw_deg)

    # ------------------------------------------------------------------
    # 新接口: body/NED → MAVROS setpoint (带日志)
    # ------------------------------------------------------------------

    async def send_velocity_body_or_ned_aligned_to_mavros(
        self,
        v_body: np.ndarray,
        v_ned: np.ndarray,
        yaw_ned_deg: float,
    ) -> np.ndarray:
        """将 pipeline 速度控制量转换为 MAVROS setpoint 并发布.

        接受:
          - v_body: [vx_fwd, vy_right, vz_down] 机体系速度 (仅用于日志)
          - v_ned:  [vn, ve, vd] NED 速度 (实际控制量)
          - yaw_ned_deg: NED yaw setpoint (degree)

        内部转换: NED → ENU → 发布到 /mavros/setpoint_raw/local

        Returns:
          v_mavros_sp: (3,) ndarray — 实际发布到 MAVROS 的 ENU 速度 [vx_east, vy_north, vz_up]
        """
        # NED → ENU
        enu_vx, enu_vy, enu_vz = ned_to_enu_velocity(
            float(v_ned[0]), float(v_ned[1]), float(v_ned[2])
        )
        enu_yaw_deg = math.degrees(ned_yaw_to_enu_yaw(math.radians(float(yaw_ned_deg))))

        await self.send_velocity_enu_yaw(enu_vx, enu_vy, enu_vz, enu_yaw_deg)

        v_mavros_sp = np.array([enu_vx, enu_vy, enu_vz], dtype=np.float32)
        return v_mavros_sp

    # ------------------------------------------------------------------
    # 安全 fallback: 零速度 / 位置 hold
    # ------------------------------------------------------------------

    async def send_zero_velocity_fallback(self, yaw_ned_deg: float = 0.0):
        """安全 fallback: 发送零速度 (ENU)."""
        enu_yaw_deg = math.degrees(ned_yaw_to_enu_yaw(math.radians(float(yaw_ned_deg))))
        await self.send_velocity_enu_yaw(0.0, 0.0, 0.0, enu_yaw_deg)

    async def send_current_position_hold(self):
        """安全 fallback: 发送当前位置 hold (ENU)."""
        with self._lock:
            if not self._enu_ready:
                return
            x = float(self.uavPosENU[0])
            y = float(self.uavPosENU[1])
            z = float(self.uavPosENU[2])
            yaw = float(self.uavYawENU)
        await self.send_position_enu_yaw(x, y, z, math.degrees(yaw))

    # ------------------------------------------------------------------
    # 关闭
    # ------------------------------------------------------------------

    async def close(self):
        """关闭控制器."""
        logger.info("[MAVROS] Closing controller...")
        self.stop_hold_stream()
        self._stop_setpoint_stream()
        for sub in self._subs:
            try:
                sub.unregister()
            except Exception:
                pass
        self._subs.clear()
        logger.info("[MAVROS] Controller closed.")

    # ------------------------------------------------------------------
    # 工具方法
    # ------------------------------------------------------------------

    @staticmethod
    def _quat_to_euler(x: float, y: float, z: float, w: float) -> Tuple[float, float, float]:
        """四元数 → 欧拉角 (roll, pitch, yaw) [rad]."""
        sinr_cosp = 2.0 * (w * x + y * z)
        cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
        roll = math.atan2(sinr_cosp, cosr_cosp)

        sinp = 2.0 * (w * y - z * x)
        if abs(sinp) >= 1.0:
            pitch = math.copysign(math.pi / 2.0, sinp)
        else:
            pitch = math.asin(sinp)

        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        yaw = math.atan2(siny_cosp, cosy_cosp)

        return roll, pitch, yaw

    @staticmethod
    def _valid_lat_lon(lat: float, lon: float) -> bool:
        if not (math.isfinite(lat) and math.isfinite(lon)):
            return False
        if abs(lat) < 1e-9 and abs(lon) < 1e-9:
            return False
        return -90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0

    @property
    def connected(self) -> bool:
        return self._mavros_connected

    @property
    def last_sent_mavros_sp(self) -> np.ndarray:
        """最后发送到 MAVROS 的 setpoint 向量 (3,), ENU 约定."""
        return self._last_sent_mavros_sp.copy()
