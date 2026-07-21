"""
定位源管理器 — FAST-LIO 主定位 + GPS/EKF fallback

职责:
  1. 每帧评估 FAST-LIO pose/cloud 健康状态
  2. 决策 control_pose 来源 (fastlio vs mavros_gps_ekf)
  3. 决策 perception_cloud 来源 (始终为 fastlio_deskewed_cloud)
  4. 日志输出当前来源，避免现场误判

健康判定:
  - fastlio_pose_healthy: 位姿新鲜度、跳变检测
  - fastlio_cloud_healthy: 点云新鲜度、点数、同步偏移

退化动作:
  - pose 退化 + cloud 正常 + GPS fallback 允许 → control=gps_fallback, cloud=fastlio
  - pose 退化 + cloud 也退化 → 按 degraded_cloud_action (direct_land | abort)
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger("PoseSourceManager")


class ControlPoseSource(str, Enum):
    FASTLIO = "fastlio"
    GPS_FALLBACK = "gps_fallback"


class CloudSource(str, Enum):
    FASTLIO_DESKEWED = "fastlio_deskewed"
    DEGRADED = "degraded"


class DegradedAction(str, Enum):
    USE_GPS_FALLBACK = "use_gps_fallback"
    DIRECT_LAND = "direct_land"
    ABORT = "abort"


@dataclass
class PoseHealthReport:
    """每帧定位健康报告."""
    fastlio_pose_healthy: bool
    fastlio_cloud_healthy: bool
    control_pose_source: ControlPoseSource
    perception_cloud_source: CloudSource
    # 诊断细节
    pose_age_ms: Optional[float] = None
    cloud_age_ms: Optional[float] = None
    cloud_odom_sync_ms: Optional[float] = None
    cloud_points: Optional[int] = None
    pose_jump_m: Optional[float] = None
    yaw_jump_deg: Optional[float] = None
    degraded_reason: Optional[str] = None
    # 是否需要执行退化动作
    degraded_action: Optional[DegradedAction] = None


class PoseSourceManager:
    """
    FAST-LIO 主定位 + GPS fallback 管理。

    cfg 示例 (fastlio_health):
      max_pose_age_ms: 200
      max_cloud_age_ms: 200
      max_cloud_odom_sync_ms: 100
      min_cloud_points: 50
      pose_jump_threshold_m: 1.0
      yaw_jump_threshold_deg: 20.0
      degraded_control_action: "use_gps_fallback"
      degraded_cloud_action: "direct_land"

    cfg 示例 (localization):
      control_pose_primary: "fastlio_gravity_aligned"
      control_pose_fallback: "mavros_gps_ekf"
      allow_gps_fallback: true
      gps_fallback_only_when_fastlio_degraded: true
      perception_cloud_source: "fastlio_deskewed_gravity_aligned"
    """

    def __init__(self, cfg: dict):
        # --- FAST-LIO 健康门控 ---
        health_cfg = cfg.get("fastlio_health", {})
        self.max_pose_age_ms = float(health_cfg.get("max_pose_age_ms", 200))
        self.max_cloud_age_ms = float(health_cfg.get("max_cloud_age_ms", 200))
        self.max_cloud_odom_sync_ms = float(health_cfg.get("max_cloud_odom_sync_ms", 100))
        self.min_cloud_points = int(health_cfg.get("min_cloud_points", 50))
        self.pose_jump_threshold_m = float(health_cfg.get("pose_jump_threshold_m", 1.0))
        self.yaw_jump_threshold_deg = float(health_cfg.get("yaw_jump_threshold_deg", 20.0))

        degraded_ctrl = str(health_cfg.get("degraded_control_action", "use_gps_fallback")).lower()
        try:
            self.degraded_control_action = DegradedAction(degraded_ctrl)
        except ValueError:
            self.degraded_control_action = DegradedAction.USE_GPS_FALLBACK

        degraded_cloud = str(health_cfg.get("degraded_cloud_action", "direct_land")).lower()
        try:
            self.degraded_cloud_action = DegradedAction(degraded_cloud)
        except ValueError:
            self.degraded_cloud_action = DegradedAction.DIRECT_LAND

        # --- 定位源配置 ---
        loc_cfg = cfg.get("localization", {})
        self.require_fastlio_pose = bool(loc_cfg.get("fastlio_pose_required", True))
        self.control_pose_primary = str(loc_cfg.get("control_pose_primary", "fastlio_gravity_aligned"))
        self.control_pose_fallback = str(loc_cfg.get("control_pose_fallback", "mavros_gps_ekf"))
        self.allow_gps_fallback = bool(loc_cfg.get("allow_gps_fallback", True))
        self.gps_fallback_only_when_degraded = bool(
            loc_cfg.get("gps_fallback_only_when_fastlio_degraded", True)
        )
        self.perception_cloud_source_name = str(
            loc_cfg.get("perception_cloud_source", "fastlio_deskewed_gravity_aligned")
        )

        # 状态追踪
        self._last_pose: Optional[np.ndarray] = None  # [x, y, z]
        self._last_yaw: Optional[float] = None
        self._last_pose_time: Optional[float] = None
        self._pose_jump_triggered: bool = False
        self._yaw_jump_triggered: bool = False

        # 当前报告
        self._current_report = PoseHealthReport(
            fastlio_pose_healthy=True,
            fastlio_cloud_healthy=True,
            control_pose_source=ControlPoseSource.FASTLIO,
            perception_cloud_source=CloudSource.FASTLIO_DESKEWED,
        )

        logger.info(
            "[PoseSrc] Init: pose_max_age=%.0fms cloud_max_age=%.0fms sync_max=%.0fms "
            "min_pts=%d pose_jump=%.1fm yaw_jump=%.1fdeg",
            self.max_pose_age_ms, self.max_cloud_age_ms, self.max_cloud_odom_sync_ms,
            self.min_cloud_points, self.pose_jump_threshold_m, self.yaw_jump_threshold_deg,
        )
        logger.info(
            "[PoseSrc] GPS fallback: allow=%s only_when_degraded=%s",
            self.allow_gps_fallback, self.gps_fallback_only_when_degraded,
        )

    # ------------------------------------------------------------------
    # 主入口
    # ------------------------------------------------------------------

    def evaluate(
        self,
        fastlio_pose: Optional[np.ndarray],     # [x, y, z, roll, pitch, yaw] or None
        fastlio_pose_stamp: Optional[float],     # ROS time seconds
        fastlio_points: Optional[np.ndarray],    # (N, 3) or None
        fastlio_points_stamp: Optional[float],   # ROS time seconds
        now: Optional[float] = None,
    ) -> PoseHealthReport:
        """
        评估当前帧 FAST-LIO 健康状态并返回决策。

        Args:
            fastlio_pose: FAST-LIO 位姿 [x,y,z,roll,pitch,yaw] (NED)
            fastlio_pose_stamp: 位姿 ROS header 时间戳 (秒)
            fastlio_points: FAST-LIO 去畸变点云 (N,3)
            fastlio_points_stamp: 点云 ROS header 时间戳 (秒)
            now: 当前时间戳 (秒), 默认 time.perf_counter()

        Returns:
            PoseHealthReport: 健康评估与源决策报告
        """
        if now is None:
            now = time.perf_counter()

        # --- 评估 pose 健康 ---
        pose_age_ms = None
        pose_jump_m = None
        yaw_jump_deg = None
        pose_healthy = True

        if not self.require_fastlio_pose:
            # Perception-frontend mode deliberately has no FAST-LIO odometry.
            # PX4 supplies the timestamp-matched control pose and attitude.
            pose_healthy = True
        elif fastlio_pose is None:
            pose_healthy = False
        elif fastlio_pose_stamp is not None:
            pose_age_ms = (now - fastlio_pose_stamp) * 1000.0
            if pose_age_ms > self.max_pose_age_ms:
                pose_healthy = False

        # 跳变检测
        if (
            self.require_fastlio_pose
            and pose_healthy
            and fastlio_pose is not None
            and len(fastlio_pose) >= 6
        ):
            pose_xyz = np.asarray(fastlio_pose[:3], dtype=np.float32)
            yaw = float(fastlio_pose[5])

            if self._last_pose is not None:
                pose_jump_m = float(np.linalg.norm(pose_xyz - self._last_pose))
                if pose_jump_m > self.pose_jump_threshold_m:
                    pose_healthy = False
                    self._pose_jump_triggered = True
                else:
                    self._pose_jump_triggered = False

            if self._last_yaw is not None:
                yaw_diff = abs(self._wrap_pi(yaw - self._last_yaw))
                yaw_jump_deg = math.degrees(yaw_diff)
                if yaw_jump_deg > self.yaw_jump_threshold_deg:
                    pose_healthy = False
                    self._yaw_jump_triggered = True
                else:
                    self._yaw_jump_triggered = False

            self._last_pose = pose_xyz.copy()
            self._last_yaw = yaw
            self._last_pose_time = now

        # --- 评估 cloud 健康 ---
        cloud_age_ms = None
        cloud_points = None
        cloud_odom_sync_ms = None
        cloud_healthy = True

        if fastlio_points is None:
            cloud_healthy = False
        else:
            cloud_points = len(fastlio_points)
            if cloud_points < self.min_cloud_points:
                cloud_healthy = False

        if fastlio_points_stamp is not None and cloud_healthy:
            cloud_age_ms = (now - fastlio_points_stamp) * 1000.0
            if cloud_age_ms > self.max_cloud_age_ms:
                cloud_healthy = False

        if fastlio_pose_stamp is not None and fastlio_points_stamp is not None:
            cloud_odom_sync_ms = abs(fastlio_points_stamp - fastlio_pose_stamp) * 1000.0
            if cloud_odom_sync_ms > self.max_cloud_odom_sync_ms:
                # 同步偏差过大只影响 pose 健康判断, 不直接影响 cloud 健康
                pass

        # --- 决策 ---
        degraded_reason = None
        degraded_action = None
        control_source = ControlPoseSource.FASTLIO
        cloud_source = CloudSource.FASTLIO_DESKEWED

        if not pose_healthy:
            degraded_reason = self._build_pose_degraded_reason(
                pose_age_ms, pose_jump_m, yaw_jump_deg
            )
            if not cloud_healthy:
                # 双退化 → 按 cloud 退化动作
                degraded_action = self.degraded_cloud_action
                cloud_source = CloudSource.DEGRADED
                control_source = ControlPoseSource.GPS_FALLBACK
            elif self.allow_gps_fallback:
                # pose 退化但 cloud 正常 → GPS fallback
                degraded_action = DegradedAction.USE_GPS_FALLBACK
                control_source = ControlPoseSource.GPS_FALLBACK
                # cloud 仍然使用 FAST-LIO deskewed
            else:
                # GPS fallback 不允许 → 按 pose 退化动作
                degraded_action = self.degraded_control_action
                if degraded_action == DegradedAction.USE_GPS_FALLBACK:
                    degraded_action = DegradedAction.DIRECT_LAND  # 不允许 fallback 则降级

        elif not cloud_healthy:
            degraded_reason = self._build_cloud_degraded_reason(
                cloud_age_ms, cloud_points
            )
            degraded_action = self.degraded_cloud_action
            cloud_source = CloudSource.DEGRADED
            # pose 健康, cloud 退化: 控制仍用 FAST-LIO, 但感知输入已不可用

        report = PoseHealthReport(
            fastlio_pose_healthy=pose_healthy,
            fastlio_cloud_healthy=cloud_healthy,
            control_pose_source=control_source,
            perception_cloud_source=cloud_source,
            pose_age_ms=pose_age_ms,
            cloud_age_ms=cloud_age_ms,
            cloud_odom_sync_ms=cloud_odom_sync_ms,
            cloud_points=cloud_points,
            pose_jump_m=pose_jump_m,
            yaw_jump_deg=yaw_jump_deg,
            degraded_reason=degraded_reason,
            degraded_action=degraded_action,
        )

        # 变更时日志
        prev = self._current_report
        if (control_source != prev.control_pose_source
                or cloud_source != prev.perception_cloud_source
                or pose_healthy != prev.fastlio_pose_healthy
                or cloud_healthy != prev.fastlio_cloud_healthy):
            logger.warning(
                "[PoseSrc] Health change: "
                "pose_healthy=%s→%s cloud_healthy=%s→%s "
                "control=%s→%s cloud_src=%s→%s "
                "degraded_action=%s reason=%s",
                prev.fastlio_pose_healthy, pose_healthy,
                prev.fastlio_cloud_healthy, cloud_healthy,
                prev.control_pose_source.value, control_source.value,
                prev.perception_cloud_source.value, cloud_source.value,
                degraded_action.value if degraded_action else "none",
                degraded_reason or "n/a",
            )

        self._current_report = report
        return report

    def get_current_report(self) -> PoseHealthReport:
        return self._current_report

    # ------------------------------------------------------------------
    # 辅助
    # ------------------------------------------------------------------

    def _build_pose_degraded_reason(
        self,
        pose_age_ms: Optional[float],
        pose_jump_m: Optional[float],
        yaw_jump_deg: Optional[float],
    ) -> str:
        parts = []
        if pose_age_ms is not None and pose_age_ms > self.max_pose_age_ms:
            parts.append(f"pose_age={pose_age_ms:.0f}ms > {self.max_pose_age_ms:.0f}ms")
        if pose_jump_m is not None and pose_jump_m > self.pose_jump_threshold_m:
            parts.append(f"pose_jump={pose_jump_m:.2f}m > {self.pose_jump_threshold_m:.2f}m")
        if yaw_jump_deg is not None and yaw_jump_deg > self.yaw_jump_threshold_deg:
            parts.append(f"yaw_jump={yaw_jump_deg:.1f}deg > {self.yaw_jump_threshold_deg:.1f}deg")
        if not parts:
            parts.append("pose_unavailable")
        return "; ".join(parts)

    def _build_cloud_degraded_reason(
        self,
        cloud_age_ms: Optional[float],
        cloud_points: Optional[int],
    ) -> str:
        parts = []
        if cloud_age_ms is not None and cloud_age_ms > self.max_cloud_age_ms:
            parts.append(f"cloud_age={cloud_age_ms:.0f}ms > {self.max_cloud_age_ms:.0f}ms")
        if cloud_points is not None and cloud_points < self.min_cloud_points:
            parts.append(f"cloud_points={cloud_points} < {self.min_cloud_points}")
        if not parts:
            parts.append("cloud_unavailable")
        return "; ".join(parts)

    @staticmethod
    def _wrap_pi(angle_rad: float) -> float:
        return (angle_rad + math.pi) % (2.0 * math.pi) - math.pi
