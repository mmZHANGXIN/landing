"""
MAVSDK 飞控通信封装
替代 RflySim 的 PX4MavCtrlV4，提供 NED 速度/位置控制接口。
"""

import asyncio
import logging
import numpy as np
import math

logger = logging.getLogger("MAVSDKController")


class MAVSDKController:
    """
    PX4 飞控 MAVSDK 封装

    提供:
      - 解锁/上锁
      - Offboard 模式
      - SendVelNED (速度控制, NED坐标系)
      - SendPosNED (位置控制)
      - 读取无人机状态 (位置, 姿态, 速度, 角速度)
    """

    def __init__(self, system_address: str = "udp://:14540"):
        self._address = system_address
        self._drone = None
        self._async_loop = None

        # 状态缓存
        self.uavPosNED = np.zeros(3, dtype=np.float32)      # [x, y, z]
        self.uavAngEular = np.zeros(3, dtype=np.float32)     # [roll, pitch, yaw]
        self.uavVelNED = np.zeros(3, dtype=np.float32)       # [vx, vy, vz]
        self.uavAngRate = np.zeros(3, dtype=np.float32)      # [p, q, r]
        self.uavThrust = 0.0
        self.uavLatLon = (0.0, 0.0)                          # (lat, lon) 度
        self.isVehicleCrash = False
        self.landed_state_on_ground = False
        self.isArmed = None
        self.isOffboard = None
        self.flightMode = None
        self._ned_ready = False
        self._gps_ready = False
        self._attitude_ready = False

        self._telemetry_task = None

    async def connect(self):
        """连接到 PX4"""
        from mavsdk import System
        self._drone = System()
        await self._drone.connect(system_address=self._address)
        logger.info(f"[MAVSDK] Waiting for drone to connect at {self._address}...")
        async for state in self._drone.core.connection_state():
            if state.is_connected:
                logger.info("[MAVSDK] Drone connected!")
                break
        # 启动遥测轮询
        self._telemetry_task = asyncio.ensure_future(self._poll_telemetry())

    async def _poll_telemetry(self):
        """持续轮询无人机状态"""
        import mavsdk
        while True:
            try:
                async for pos in self._drone.telemetry.position_velocity_ned():
                    self.uavPosNED = np.array([pos.position.north_m,
                                                pos.position.east_m,
                                                pos.position.down_m], dtype=np.float32)
                    self.uavVelNED = np.array([pos.velocity.north_m_s,
                                                pos.velocity.east_m_s,
                                                pos.velocity.down_m_s], dtype=np.float32)
                    self._ned_ready = True
                    break  # 取一次

                async for att in self._drone.telemetry.attitude_euler():
                    self.uavAngEular = np.array([att.roll_deg * math.pi / 180.0,
                                                  att.pitch_deg * math.pi / 180.0,
                                                  att.yaw_deg * math.pi / 180.0], dtype=np.float32)
                    self._attitude_ready = True
                    break

                # GPS 位置
                try:
                    async for gps in self._drone.telemetry.position():
                        lat_lon = (gps.latitude_deg, gps.longitude_deg)
                        if self._valid_lat_lon(lat_lon):
                            self.uavLatLon = lat_lon
                            self._gps_ready = True
                        break
                except Exception:
                    pass

                # 角速度 (可通过 IMU 或 attitude 获取)
                # 简化: 用前后姿态差分估算; 实际应用建议订阅 IMU 话题

                async for in_air in self._drone.telemetry.landed_state():
                    self.landed_state_on_ground = (in_air == mavsdk.telemetry.LandedState.ON_GROUND)
                    self.isVehicleCrash = self.landed_state_on_ground
                    break

                try:
                    async for armed in self._drone.telemetry.armed():
                        self.isArmed = bool(armed)
                        break
                except Exception:
                    pass

                try:
                    async for flight_mode in self._drone.telemetry.flight_mode():
                        self.flightMode = flight_mode
                        mode_name = str(flight_mode).upper()
                        self.isOffboard = "OFFBOARD" in mode_name
                        break
                except Exception:
                    pass

            except Exception as e:
                logger.warning(f"[MAVSDK] Telemetry error: {e}")

            await asyncio.sleep(0.05)  # 20Hz 轮询

    @staticmethod
    def _valid_lat_lon(lat_lon) -> bool:
        try:
            lat = float(lat_lon[0])
            lon = float(lat_lon[1])
        except (TypeError, ValueError, IndexError):
            return False
        if not (math.isfinite(lat) and math.isfinite(lon)):
            return False
        if abs(lat) < 1e-9 and abs(lon) < 1e-9:
            return False
        return -90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0

    async def wait_for_home(self, timeout_s: float = 10.0):
        """Wait until NED, attitude, and GPS telemetry are ready for home capture."""
        start = asyncio.get_event_loop().time()
        while asyncio.get_event_loop().time() - start < timeout_s:
            if self._ned_ready and self._gps_ready and self._attitude_ready:
                return self.uavPosNED.copy(), self.uavLatLon, self.uavAngEular.copy()
            await asyncio.sleep(0.05)
        raise TimeoutError(
            "[MAVSDK] Home telemetry not ready: "
            f"ned={self._ned_ready} gps={self._gps_ready} attitude={self._attitude_ready}"
        )

    async def wait_for_local_pose(self, timeout_s: float = 10.0):
        """Wait until local NED and attitude telemetry are ready."""
        start = asyncio.get_event_loop().time()
        while asyncio.get_event_loop().time() - start < timeout_s:
            if self._ned_ready and self._attitude_ready:
                return self.uavPosNED.copy(), self.uavAngEular.copy()
            await asyncio.sleep(0.05)
        raise TimeoutError(
            "[MAVSDK] Local telemetry not ready: "
            f"ned={self._ned_ready} attitude={self._attitude_ready}"
        )

    async def arm(self):
        """解锁"""
        logger.info("[MAVSDK] Arming...")
        await self._drone.action.arm()

    async def disarm(self):
        """上锁"""
        logger.info("[MAVSDK] Disarming...")
        await self._drone.action.disarm()

    async def init_offboard(self):
        """初始化 Offboard 模式"""
        import mavsdk
        logger.info("[MAVSDK] Setting offboard mode...")
        await self._drone.offboard.set_velocity_ned(
            mavsdk.offboard.VelocityNedYaw(0.0, 0.0, 0.0, 0.0))
        await self._drone.offboard.start()

    async def send_velocity_ned_yaw(self, vx: float, vy: float, vz: float,
                                    yaw_deg: float = 0.0):
        """
        发送 NED 速度 + yaw 角 setpoint。

        MAVSDK VelocityNedYaw 的第四个参数是 yaw_deg, 不是 yaw rate。
        若需要设定偏航角速度, 在上层按 dt 将 yaw_rate 积分成 yaw_deg 后调用本方法。

        :param vx: 北向速度 (m/s)
        :param vy: 东向速度 (m/s)
        :param vz: 下向速度 (m/s, 正值=下降)
        :param yaw_deg: yaw 角 (degree)
        """
        import mavsdk
        await self._drone.offboard.set_velocity_ned(
            mavsdk.offboard.VelocityNedYaw(vx, vy, vz, yaw_deg))

    async def send_velocity_ned(self, vx: float, vy: float, vz: float,
                                yaw_deg: float = 0.0):
        """兼容旧调用: 发送 NED 速度 + yaw 角 setpoint (degree)。"""
        await self.send_velocity_ned_yaw(vx, vy, vz, yaw_deg)

    async def send_position_ned(self, x: float, y: float, z: float, yaw_deg: float = 0.0):
        """
        发送 NED 位置指令
        :param x: 北向 (m)
        :param y: 东向 (m)
        :param z: 下方 (m, 正值)
        :param yaw_deg: 偏航角 (degree)
        """
        import mavsdk
        await self._drone.offboard.set_position_ned(
            mavsdk.offboard.PositionNedYaw(x, y, z, yaw_deg))

    async def close(self):
        if self._telemetry_task:
            self._telemetry_task.cancel()
