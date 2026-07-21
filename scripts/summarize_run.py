#!/usr/bin/env python3
"""Summarize mission-stage durations and per-cloud inference timing for one run."""

import argparse
import csv
import math
from pathlib import Path


STAGES = (
    ("arming_to_takeoff", "ARMED", "TAKEOFF_STARTED"),
    ("takeoff_to_high_altitude", "TAKEOFF_STARTED", "HIGH_ALTITUDE_REACHED"),
    ("high_altitude_to_goto", "HIGH_ALTITUDE_REACHED", "GOTO_STARTED"),
    ("goto", "GOTO_STARTED", "GOTO_ARRIVED"),
    ("goto_arrival_to_drl", "GOTO_ARRIVED", "DRL_DESCENT_STARTED"),
    ("drl_to_direct_land", "DRL_DESCENT_STARTED", "DIRECT_LAND_STARTED"),
    ("direct_land_to_ground", "DIRECT_LAND_STARTED", "PX4_ON_GROUND"),
    ("high_altitude_to_ground", "HIGH_ALTITUDE_REACHED", "PX4_ON_GROUND"),
    ("drl_descent_to_ground", "DRL_DESCENT_STARTED", "PX4_ON_GROUND"),
    ("ground_to_disarmed", "PX4_ON_GROUND", "DISARMED"),
)


def _read_csv(path: Path):
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _finite_float(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _percentile(values, fraction):
    ordered = sorted(values)
    if not ordered:
        return None
    position = (len(ordered) - 1) * fraction
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _describe(rows, field):
    values = [_finite_float(row.get(field)) for row in rows]
    values = [value for value in values if value is not None]
    if not values:
        return "n/a"
    return (
        f"mean={sum(values) / len(values):.2f} ms  "
        f"p50={_percentile(values, 0.50):.2f} ms  "
        f"p95={_percentile(values, 0.95):.2f} ms  max={max(values):.2f} ms"
    )


def summarize(run_dir: Path) -> int:
    event_rows = _read_csv(run_dir / "mission_events.csv")
    timing_rows = _read_csv(run_dir / "frame_timing.csv")
    event_times = {
        row["event"]: _finite_float(row.get("timestamp_ros_s")) for row in event_rows
    }

    print(f"Run: {run_dir}")
    print("\nMission stages (ROS time):")
    for label, start, end in STAGES:
        t0, t1 = event_times.get(start), event_times.get(end)
        value = "n/a" if t0 is None or t1 is None else f"{max(0.0, t1 - t0):.3f} s"
        print(f"  {label:30s} {value}")

    accepted_rows = [row for row in timing_rows if row.get("accepted") == "1"]
    stale_rows = [row for row in timing_rows if row.get("accepted") != "1"]
    print("\nPer-cloud timing (accepted inference frames):")
    print(f"  frames={len(accepted_rows)} stale_or_rejected={len(stale_rows)}")
    for field in (
        "pointcloud_preprocess_ms", "halss_ms", "depth_projection_ms",
        "depth_completion_ms", "onnx_ms", "perception_inference_ms",
        "pipeline_total_ms", "result_age_ms",
    ):
        print(f"  {field:26s} {_describe(accepted_rows, field)}")
    return 0 if event_rows or timing_rows else 2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path, help="experiments/runs/<timestamp>_orin_landing")
    args = parser.parse_args()
    raise SystemExit(summarize(args.run_dir))


if __name__ == "__main__":
    main()
