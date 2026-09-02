"""Mission-state manager for the landing experiment.

The manager is intentionally side-effect free: it does not publish setpoints,
call MAVSDK/MAVROS services, or read ROS topics.  The flight pipeline feeds it
fresh telemetry/perception facts and executes the returned decision.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from typing import Optional, Tuple


class MissionState(str, Enum):
    IDLE = "IDLE"
    WAIT_LOCALIZATION = "WAIT_LOCALIZATION"
    READY = "READY"
    GOTO_SAFE = "GOTO_SAFE"
    HOLD_FOR_MANUAL = "HOLD_FOR_MANUAL"
    DRL_DESCENT = "DRL_DESCENT"
    DIRECT_LAND = "DIRECT_LAND"
    LANDED = "LANDED"
    ABORT = "ABORT"
    EMERGENCY_STOP = "EMERGENCY_STOP"


class BoundaryAction(str, Enum):
    DIRECT_LAND = "DIRECT_LAND"
    HOLD = "HOLD"
    ABORT = "ABORT"


class GlobalPriorMode(str, Enum):
    GPS = "gps"
    LOCAL_BODY_OFFSET = "local_body_offset"


class ControlMode(str, Enum):
    HOLD = "HOLD"
    GOTO_SAFE = "GOTO_SAFE"
    DRL_VELOCITY = "DRL_VELOCITY"
    DIRECT_LAND_VELOCITY = "DIRECT_LAND_VELOCITY"
    LANDED = "LANDED"
    ABORT = "ABORT"


@dataclass(frozen=True)
class StateInputs:
    now: float
    pose_xyz: Optional[Tuple[float, float, float]] = None
    yaw_rad: Optional[float] = None
    velocity_xyz: Optional[Tuple[float, float, float]] = None
    pose_age_ms: Optional[float] = None
    cloud_odom_sync_ms: Optional[float] = None
    perception_age_s: Optional[float] = None
    perception_ok: bool = True
    landed_state_on_ground: bool = False
    offboard_active: Optional[bool] = None
    armed: Optional[bool] = None
    step_count: int = 0
    max_steps: Optional[int] = None
    ground_clearance_p05_m: Optional[float] = None
    ground_clearance_min_m: Optional[float] = None


@dataclass(frozen=True)
class StateDecision:
    state: MissionState
    previous_state: MissionState
    control_mode: ControlMode
    reason: str
    height_m: Optional[float]
    ground_clearance_p05_m: Optional[float]
    ground_clearance_min_m: Optional[float]
    allow_drl: bool
    direct_land: bool
    landed: bool
    abort: bool
    transition: bool
    land_reference_xy_yaw: Optional[Tuple[float, float, float]]
    direct_land_vz_mps: float
    continue_yaw_rate: bool


class MissionStateManager:
    """State transitions for safe landing.

    Height is normalized to positive-up clearance above the configured ground
    reference.  For this project the default is Fast-LIO/PX4 local z in a NED
    style convention, so height = ground_z_ref - z.
    """

    def __init__(self, cfg: dict = None):
        cfg = cfg or {}
        self.height_source = str(cfg.get("height_source", "fastlio_z"))
        self.height_axis = str(cfg.get("height_axis", "neg_z"))
        self.ground_z_ref_m = float(cfg.get("ground_z_ref_m", 0.0))
        self.auto_ground_z_ref = bool(cfg.get("auto_ground_z_ref", False))
        self.direct_land_enabled = bool(cfg.get("direct_land_enabled", True))
        self.direct_land_trigger_height_m = float(
            cfg.get("direct_land_trigger_height_m", cfg.get("direct_land_trigger_m", 0.8))
        )
        self.landed_height_m = float(cfg.get("landed_height_m", 0.15))
        self.direct_land_vz_mps = float(cfg.get("direct_land_vz_mps", 0.25))
        self.direct_land_lock_xy_yaw = bool(cfg.get("direct_land_lock_xy_yaw", True))
        self.direct_land_continue_yaw_rate = bool(cfg.get("direct_land_continue_yaw_rate", True))
        self.perception_timeout_s = float(cfg.get("perception_timeout_s", 1.5))
        self.pose_timeout_s = float(cfg.get("pose_timeout_s", 0.2))
        self.max_cloud_odom_sync_ms = float(cfg.get("max_cloud_odom_sync_ms", 100.0))
        self.ground_crosscheck_enabled = bool(cfg.get("ground_crosscheck_enabled", True))
        self.ground_crosscheck_action = str(cfg.get("ground_crosscheck_action", "warn")).lower()
        self.ground_crosscheck_max_error_m = float(cfg.get("ground_crosscheck_max_error_m", 0.5))

        # --- XY boundary protection (indoor safety) ---
        self.boundary_enable = bool(cfg.get("boundary_enable", False))
        self.boundary_x_min = float(cfg.get("boundary_x_min", -9999.0))
        self.boundary_x_max = float(cfg.get("boundary_x_max", 9999.0))
        self.boundary_y_min = float(cfg.get("boundary_y_min", -9999.0))
        self.boundary_y_max = float(cfg.get("boundary_y_max", 9999.0))
        boundary_action_raw = str(cfg.get("boundary_trigger_action", "direct_land")).upper()
        try:
            self.boundary_trigger_action = BoundaryAction(boundary_action_raw)
        except ValueError:
            self.boundary_trigger_action = BoundaryAction.DIRECT_LAND

        # --- Global prior mode ---
        prior_mode_raw = str(cfg.get("global_prior_mode", "gps")).lower()
        try:
            self.global_prior_mode = GlobalPriorMode(prior_mode_raw)
        except ValueError:
            self.global_prior_mode = GlobalPriorMode.GPS

        self.state = MissionState.IDLE
        self._land_reference_xy_yaw: Optional[Tuple[float, float, float]] = None
        self._last_reason = "initialized"

    @property
    def land_reference_xy_yaw(self) -> Optional[Tuple[float, float, float]]:
        return self._land_reference_xy_yaw

    def reset(self, state: MissionState = MissionState.IDLE, reason: str = "reset") -> StateDecision:
        previous = self.state
        self.state = MissionState(state)
        self._land_reference_xy_yaw = None
        self._last_reason = reason
        return self._decision(
            previous=previous,
            reason=reason,
            height_m=None,
            ground_p05=None,
            ground_min=None,
            transition=previous != self.state,
        )

    def start_after_takeoff(self, use_global_guidance: bool) -> StateDecision:
        target = MissionState.GOTO_SAFE if use_global_guidance else MissionState.DRL_DESCENT
        previous = self.state
        self.state = target
        reason = "global_guidance_enabled" if use_global_guidance else "no_global_guidance"
        self._last_reason = reason
        return self._decision(previous, reason, None, None, None, previous != self.state)

    def mark_goto_arrived(self, reason: str = "goto_safe_arrived") -> StateDecision:
        previous = self.state
        self.state = MissionState.DRL_DESCENT
        self._last_reason = reason
        return self._decision(previous, reason, None, None, None, previous != self.state)

    def update(self, inputs: StateInputs) -> StateDecision:
        previous = self.state
        height_m = self.height_from_pose(inputs.pose_xyz)
        ground_p05 = self._finite_or_none(inputs.ground_clearance_p05_m)
        ground_min = self._finite_or_none(inputs.ground_clearance_min_m)
        reason = self._last_reason

        if self.auto_ground_z_ref and inputs.pose_xyz is not None and self.state == MissionState.IDLE:
            self.ground_z_ref_m = float(inputs.pose_xyz[2])

        if self.state in (MissionState.LANDED, MissionState.ABORT, MissionState.EMERGENCY_STOP):
            return self._decision(previous, reason, height_m, ground_p05, ground_min, False)

        if self.state == MissionState.HOLD_FOR_MANUAL:
            if inputs.offboard_active is False:
                self.state = MissionState.IDLE
                reason = "manual_takeover"
            elif inputs.armed is False:
                self.state = MissionState.LANDED
                reason = "disarmed_during_manual_hold"
            self._last_reason = reason
            return self._decision(previous, reason, height_m, ground_p05, ground_min,
                                  previous != self.state)

        # --- XY boundary check (indoor safety, checked before other transitions) ---
        if self.boundary_enable and self.state in (
            MissionState.GOTO_SAFE, MissionState.DRL_DESCENT, MissionState.DIRECT_LAND,
        ):
            boundary_reason = self._boundary_breach_reason(inputs)
            if boundary_reason is not None:
                if self.boundary_trigger_action == BoundaryAction.ABORT:
                    self.state = MissionState.ABORT
                    reason = boundary_reason
                elif self.boundary_trigger_action == BoundaryAction.HOLD:
                    self.state = MissionState.DRL_DESCENT
                    reason = boundary_reason
                else:
                    self.state = MissionState.DIRECT_LAND
                    reason = boundary_reason
                    self._capture_land_reference(inputs)
                self._last_reason = reason
                return self._decision(previous, reason, height_m, ground_p05, ground_min, True)

        if inputs.offboard_active is False and self.state not in (MissionState.IDLE, MissionState.WAIT_LOCALIZATION):
            self.state = MissionState.IDLE
            self._land_reference_xy_yaw = None
            reason = "offboard_disengaged"
        elif inputs.armed is False and self.state not in (MissionState.IDLE, MissionState.LANDED):
            self.state = MissionState.IDLE
            self._land_reference_xy_yaw = None
            reason = "disarmed"
        elif self._pose_stale(inputs):
            if self.state in (MissionState.IDLE, MissionState.WAIT_LOCALIZATION, MissionState.READY):
                self.state = MissionState.WAIT_LOCALIZATION
                reason = "waiting_for_fresh_pose"
            else:
                self.state = MissionState.ABORT
                reason = "pose_timeout"
        elif self._sync_stale(inputs):
            if self.state in (MissionState.DRL_DESCENT, MissionState.DIRECT_LAND):
                self.state = MissionState.ABORT
                reason = "cloud_odom_sync_timeout"
        elif inputs.max_steps is not None and inputs.step_count >= inputs.max_steps:
            self.state = MissionState.ABORT
            reason = "max_steps_reached"
        elif inputs.landed_state_on_ground:
            self.state = MissionState.LANDED
            reason = "px4_landed_state"
        elif height_m is not None and height_m <= self.landed_height_m and self.state in (
            MissionState.DRL_DESCENT,
            MissionState.DIRECT_LAND,
        ):
            self.state = MissionState.LANDED
            reason = "landed_height_reached"
        elif self.state == MissionState.READY:
            self.state = MissionState.DRL_DESCENT
            reason = "ready_to_drl_descent"
        elif self.state == MissionState.DRL_DESCENT:
            direct_reason = self._direct_land_reason(inputs, height_m, ground_p05)
            if direct_reason is not None:
                self.state = MissionState.DIRECT_LAND
                reason = direct_reason
                self._capture_land_reference(inputs)
            elif (
                inputs.perception_age_s is not None
                and inputs.perception_age_s >= self.perception_timeout_s
            ):
                self.state = MissionState.HOLD_FOR_MANUAL
                reason = "perception_timeout_manual_takeover"
        elif self.state == MissionState.DIRECT_LAND:
            if self._land_reference_xy_yaw is None:
                self._capture_land_reference(inputs)
            reason = "direct_land_active"

        self._last_reason = reason
        return self._decision(
            previous=previous,
            reason=reason,
            height_m=height_m,
            ground_p05=ground_p05,
            ground_min=ground_min,
            transition=previous != self.state,
        )

    def height_from_pose(self, pose_xyz: Optional[Tuple[float, float, float]]) -> Optional[float]:
        if pose_xyz is None:
            return None
        try:
            z = float(pose_xyz[2])
        except (TypeError, ValueError, IndexError):
            return None
        if not math.isfinite(z):
            return None
        if self.height_axis == "pos_z":
            return z - self.ground_z_ref_m
        if self.height_axis == "abs_z":
            return abs(z - self.ground_z_ref_m)
        return self.ground_z_ref_m - z

    def _direct_land_reason(
        self,
        inputs: StateInputs,
        height_m: Optional[float],
        ground_p05: Optional[float],
    ) -> Optional[str]:
        if not self.direct_land_enabled:
            return None
        if height_m is not None and height_m <= self.direct_land_trigger_height_m:
            if self._ground_crosscheck_blocks(height_m, ground_p05):
                return None
            if self._ground_crosscheck_warns(height_m, ground_p05):
                return "height_below_direct_land_trigger_crosscheck_warn"
            return "height_below_direct_land_trigger"
        return None

    def _ground_crosscheck_blocks(self, height_m: float, ground_p05: Optional[float]) -> bool:
        if not self.ground_crosscheck_enabled or ground_p05 is None:
            return False
        if self.ground_crosscheck_action != "block":
            return False
        return abs(float(ground_p05) - float(height_m)) > self.ground_crosscheck_max_error_m

    def _ground_crosscheck_warns(self, height_m: float, ground_p05: Optional[float]) -> bool:
        if not self.ground_crosscheck_enabled or ground_p05 is None:
            return False
        if self.ground_crosscheck_action == "block":
            return False
        return abs(float(ground_p05) - float(height_m)) > self.ground_crosscheck_max_error_m

    def _pose_stale(self, inputs: StateInputs) -> bool:
        if inputs.pose_xyz is None:
            return True
        if inputs.pose_age_ms is None:
            return False
        return float(inputs.pose_age_ms) > self.pose_timeout_s * 1000.0

    def _sync_stale(self, inputs: StateInputs) -> bool:
        if inputs.cloud_odom_sync_ms is None:
            return False
        return float(inputs.cloud_odom_sync_ms) > self.max_cloud_odom_sync_ms

    def _boundary_breach_reason(self, inputs: StateInputs) -> Optional[str]:
        """Check if current XY position exceeds configured boundary."""
        if inputs.pose_xyz is None:
            return None
        x = float(inputs.pose_xyz[0])
        y = float(inputs.pose_xyz[1])
        if x < self.boundary_x_min:
            return f"boundary_breach_x_min: x={x:.2f} < {self.boundary_x_min:.2f}"
        if x > self.boundary_x_max:
            return f"boundary_breach_x_max: x={x:.2f} > {self.boundary_x_max:.2f}"
        if y < self.boundary_y_min:
            return f"boundary_breach_y_min: y={y:.2f} < {self.boundary_y_min:.2f}"
        if y > self.boundary_y_max:
            return f"boundary_breach_y_max: y={y:.2f} > {self.boundary_y_max:.2f}"
        return None

    def _capture_land_reference(self, inputs: StateInputs) -> None:
        if not self.direct_land_lock_xy_yaw or inputs.pose_xyz is None:
            return
        yaw = 0.0 if inputs.yaw_rad is None else float(inputs.yaw_rad)
        self._land_reference_xy_yaw = (
            float(inputs.pose_xyz[0]),
            float(inputs.pose_xyz[1]),
            yaw,
        )

    def _decision(
        self,
        previous: MissionState,
        reason: str,
        height_m: Optional[float],
        ground_p05: Optional[float],
        ground_min: Optional[float],
        transition: bool,
    ) -> StateDecision:
        mode = self._control_mode_for_state(self.state)
        return StateDecision(
            state=self.state,
            previous_state=previous,
            control_mode=mode,
            reason=reason,
            height_m=height_m,
            ground_clearance_p05_m=ground_p05,
            ground_clearance_min_m=ground_min,
            allow_drl=mode == ControlMode.DRL_VELOCITY,
            direct_land=mode == ControlMode.DIRECT_LAND_VELOCITY,
            landed=self.state == MissionState.LANDED,
            abort=self.state in (MissionState.ABORT, MissionState.EMERGENCY_STOP),
            transition=transition,
            land_reference_xy_yaw=self._land_reference_xy_yaw,
            direct_land_vz_mps=self.direct_land_vz_mps,
            continue_yaw_rate=self.direct_land_continue_yaw_rate,
        )

    @staticmethod
    def _control_mode_for_state(state: MissionState) -> ControlMode:
        if state == MissionState.GOTO_SAFE:
            return ControlMode.GOTO_SAFE
        if state == MissionState.DRL_DESCENT:
            return ControlMode.DRL_VELOCITY
        if state == MissionState.DIRECT_LAND:
            return ControlMode.DIRECT_LAND_VELOCITY
        if state == MissionState.LANDED:
            return ControlMode.LANDED
        if state in (MissionState.ABORT, MissionState.EMERGENCY_STOP):
            return ControlMode.ABORT
        return ControlMode.HOLD

    @staticmethod
    def _finite_or_none(value: Optional[float]) -> Optional[float]:
        if value is None:
            return None
        try:
            v = float(value)
        except (TypeError, ValueError):
            return None
        return v if math.isfinite(v) else None


__all__ = [
    "ControlMode",
    "MissionState",
    "MissionStateManager",
    "StateDecision",
    "StateInputs",
]
