#!/usr/bin/env python3
"""
MAVROS position control test -- takeoff -> hover -> land (PX4 internal closed-loop)
====================================================================================

Procedure:
  1. Script starts -> streams current position setpoint @20Hz
  2. RC: arm -> OFFBOARD
  3. Script detects -> takeoff to target height -> hover -> land -> DISARM

Pose source: /mavros/local_position/odom (ENU convention, same as setpoint convention)

Usage:
  python test_mavros_velocity.py --target-z 1.0
"""

import argparse
import asyncio
import math
import sys
import threading
import time
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

SETPOINT_RATE_HZ = 20.0
OFFBOARD_WARMUP_S = 2.0
MAVROS_NS = "/mavros"


def _quat_to_yaw(x, y, z, w):
    """Quaternion (xyzw) → yaw [rad], ENU convention (0=east, CCW+)."""
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _fmt_vec(v) -> str:
    return f"[{v[0]:.2f},{v[1]:.2f},{v[2]:.2f}]"


async def test_mavros_position(target_z_m, tol_z_m, hover_s, land_z_m):
    """MAVROS position control: takeoff -> hover -> land."""

    # --- 1. subscribe /mavros/local_position/odom (ENU, same as setpoint convention) ---
    import rospy
    from nav_msgs.msg import Odometry

    if not rospy.core.is_initialized():
        rospy.init_node("test_mavros_velocity", anonymous=False)

    _odom_lock = threading.Lock()
    _odom_enu = None  # (x, y, z, yaw) raw ENU

    def _odom_cb(msg: Odometry):
        nonlocal _odom_enu
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        yaw = _quat_to_yaw(q.x, q.y, q.z, q.w)
        with _odom_lock:
            _odom_enu = (float(p.x), float(p.y), float(p.z), yaw)

    odom_sub = rospy.Subscriber(f"{MAVROS_NS}/local_position/odom", Odometry, _odom_cb, queue_size=10)

    # wait for first odom message
    print("waiting for odometry...")
    rate = rospy.Rate(20)
    while _odom_enu is None:
        rate.sleep()

    # --- 2. MAVROS (FC interface only, we don't use its NED-converted fields) ---
    from control.mavros_controller import MAVROSController
    fc = MAVROSController(
        mavros_ns=MAVROS_NS,
        setpoint_rate_hz=SETPOINT_RATE_HZ,
        offboard_warmup_s=OFFBOARD_WARMUP_S,
    )
    await fc.connect()

    def _get_odom():
        """Thread-safe odom read. Returns (x, y, z, yaw_rad) in ENU."""
        with _odom_lock:
            return _odom_enu

    x0, y0, z0, yaw0 = _get_odom()
    home_yaw_deg = math.degrees(yaw0)
    print(f"MAVROS connected. armed={fc.isArmed} mode={fc.flightMode} offboard={fc.isOffboard}")
    print(f"odom (ENU): pos=[{x0:.2f},{y0:.2f},{z0:.2f}] yaw={home_yaw_deg:.0f}deg")

    # --- 3. stream current position setpoint, wait for RC arm+OFFBOARD ---
    print("")
    print("=" * 50)
    print(f"Streaming current position setpoint @ {SETPOINT_RATE_HZ:.0f}Hz ...")
    print("RC: arm first, then switch to OFFBOARD")
    print("=" * 50)

    hold_pos = True

    async def _hold_position():
        while hold_pos:
            x, y, z, _ = _get_odom()
            await fc.send_position_ned(x, y, z, home_yaw_deg)
            await asyncio.sleep(0.05)

    hold_task = asyncio.create_task(_hold_position())

    while True:
        if fc.isArmed and fc.isOffboard:
            print(f">>> armed={fc.isArmed} offboard={fc.isOffboard} -- PX4 position hold active!")
            break
        if fc.isOffboard and not fc.isArmed:
            print("   OFFBOARD active, arm RC to proceed...")
        await asyncio.sleep(0.3)

    await asyncio.sleep(OFFBOARD_WARMUP_S)
    hold_pos = False
    hold_task.cancel()
    try:
        await hold_task
    except asyncio.CancelledError:
        pass

    # --- 4. record home point ---
    x0, y0, z0, yaw0 = _get_odom()
    print(f"  home z={z0:.2f}m (ENU, up+)")

    # --- 5. takeoff (velocity climb @ 2 m/s, ENU: vz positive = up) ---
    CLIMB_SPEED = 2.0
    print("")
    print("=" * 50)
    print(f"Takeoff to {target_z_m:.1f}m height @ {CLIMB_SPEED:.1f} m/s")
    print("=" * 50)

    climb_start = time.perf_counter()
    last_log = 0.0
    while True:
        _, _, cur_z, _ = _get_odom()
        height_m = cur_z - z0

        if height_m >= target_z_m:
            print(f">>> reached target height! {height_m:.2f}m elapsed={time.perf_counter() - climb_start:.1f}s")
            break
        if time.perf_counter() - climb_start > 15.0:
            print(f"*** takeoff timeout 15s, height={height_m:.2f}m target={target_z_m:.2f}m")
            break

        await fc.send_velocity_ned_yaw(0.0, 0.0, CLIMB_SPEED, home_yaw_deg)

        now = time.perf_counter()
        if now - last_log >= 0.5:
            print(f"   height={height_m:.2f}m target={target_z_m:.2f}m elapsed={now - climb_start:.1f}s")
            last_log = now
        await asyncio.sleep(0.05)

    # stabilise with position hold
    print("stabilising...")
    target_z = z0 + target_z_m
    for _ in range(int(1.0 / 0.1)):
        await fc.send_position_ned(x0, y0, target_z, home_yaw_deg)
        await asyncio.sleep(0.1)

    # --- 6. hover (position hold) ---
    print("")
    print(f"Hover {hover_s:.1f} sec ...")
    hover_start = time.perf_counter()
    while time.perf_counter() - hover_start < hover_s:
        await fc.send_position_ned(x0, y0, target_z, home_yaw_deg)
        await asyncio.sleep(0.5)
    print("hover complete")

    # --- 7. land (velocity descent, ENU: vz negative = down) ---
    DESCENT_SPEED = 0.5
    land_z = z0 + land_z_m
    print("")
    print("=" * 50)
    print(f"Land to {land_z_m:.1f}m height @ {DESCENT_SPEED:.1f} m/s")
    print("=" * 50)

    land_start = time.perf_counter()
    last_land_log = 0.0
    while True:
        _, _, cur_z, _ = _get_odom()
        height_m = cur_z - z0
        now = time.perf_counter()
        if now - last_land_log >= 0.5:
            print(f"   height={height_m:.2f}m target={land_z_m:.2f}m elapsed={now - land_start:.1f}s")
            last_land_log = now
        if height_m <= land_z_m + tol_z_m:
            print(f">>> descent complete at {height_m:.2f}m")
            break
        if time.perf_counter() - land_start > 15.0:
            print(f"*** land timeout 15s, height={height_m:.2f}m")
            break
        await fc.send_velocity_ned_yaw(0.0, 0.0, -DESCENT_SPEED, home_yaw_deg)
        await asyncio.sleep(0.05)

    # stabilise near ground
    print("stabilising near ground...")
    for _ in range(int(1.0 / 0.1)):
        await fc.send_position_ned(x0, y0, land_z, home_yaw_deg)
        await asyncio.sleep(0.1)

    # --- 8. DISARM ---
    print("touchdown hover 1 sec...")
    for _ in range(10):
        await fc.send_position_ned(x0, y0, land_z, home_yaw_deg)
        await asyncio.sleep(0.1)

    print("disarming (DISARM)...")
    await fc.disarm()
    await asyncio.sleep(0.5)
    print(">>> disarmed!")

    # --- 9. cleanup ---
    print("")
    print("=" * 50)
    print("test complete! switch RC back to MANUAL")
    print("=" * 50)
    odom_sub.unregister()
    await fc.close()
    print("MAVROS connection closed.")


def main():
    parser = argparse.ArgumentParser(description="MAVROS position control takeoff->hover->land test")
    parser.add_argument("--target-z", type=float, default=1.0, help="takeoff height (m), default 1.0")
    parser.add_argument("--tol-z", type=float, default=0.15, help="Z tolerance (m), default 0.15")
    parser.add_argument("--hover", type=float, default=2.0, help="hover time (s), default 2.0")
    parser.add_argument("--land-z", type=float, default=0.2, help="land height (m), default 0.2")
    parser.add_argument("--mavros-ns", type=str, default="/mavros")
    args = parser.parse_args()

    global MAVROS_NS
    MAVROS_NS = args.mavros_ns

    print("=" * 60)
    print(" MAVROS Position Control Test")
    print("=" * 60)
    print(f"  target height: {args.target_z:.1f} m")
    print(f"  tol Z:         {args.tol_z:.2f} m")
    print(f"  hover:         {args.hover:.1f} s")
    print(f"  MAVROS NS:     {MAVROS_NS}")
    print("=" * 60)
    print("")
    print("  procedure:")
    print("  1. FC powered, MAVROS running, FAST-LIO -> vision_pose OK")
    print("  2. Run this script (streams current position hold setpoint)")
    print("  3. RC: arm first, then OFFBOARD")
    print("  4. Script takes over -> takeoff -> hover -> land -> DISARM")
    print("  5. RC back to MANUAL")
    print("")

    try:
        asyncio.run(test_mavros_position(
            target_z_m=args.target_z,
            tol_z_m=args.tol_z,
            hover_s=args.hover,
            land_z_m=args.land_z,
        ))
    except KeyboardInterrupt:
        print("user interrupted!")
    except Exception as e:
        print(f"test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
