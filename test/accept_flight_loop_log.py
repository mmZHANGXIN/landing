#!/usr/bin/env python3
"""Acceptance checks for closed-loop pipeline logs.

This is a lightweight post-run gate for ``pipeline.py --mode ros`` logs. It
checks that strict gates were not bypassed, GIS global guidance happened before
DRL descent, and DRL frames contain yaw-aware velocity commands and observation
diagnostics.
"""

from __future__ import annotations

import argparse
import math
import re
from collections import Counter
from pathlib import Path


ACTION_NAMES_BY_SIGN = {
    -1: ["HOVER", "N", "NW", "W", "SW", "S", "SE", "E", "NE", "DESCEND"],
    1: ["HOVER", "N", "NE", "E", "SE", "S", "SW", "W", "NW", "DESCEND"],
}


FRAME_RE = re.compile(
    r"\[(?P<seq>\d+)\]\s+act=(?P<action>\d+)\((?P<name>[^)]+)\)\s+"
    r"H=(?P<H>[0-9.]+)ms\s+D=(?P<D>[0-9.]+)ms\s+"
    r"C=(?P<C>[0-9.]+)ms\s+RL=(?P<RL>[0-9.]+)ms\s+"
    r"total=(?P<total>[0-9.]+)ms\s+"
    r"\|\s+yaw=(?P<yaw>-?[0-9.]+)deg\s+yaw_sp=(?P<yaw_sp>-?[0-9.]+)deg\s+"
    r"yr=(?P<yr>-?[0-9.]+)\s+sync=(?P<sync>[0-9.]+)ms\s+v_body=\[(?P<v_body>[^\]]+)\]\s+"
    r"v_ned=\[(?P<v_ned>[^\]]+)\]\s+"
    r"\|\s+depth=(?P<depth_min>-?[0-9.]+)/(?P<depth_mean>-?[0-9.]+)/(?P<depth_max>-?[0-9.]+)m\s+"
    r"obsD=(?P<obsd_min>-?[0-9.]+)/(?P<obsd_mean>-?[0-9.]+)/(?P<obsd_max>-?[0-9.]+)\s+"
    r"obsS=(?P<obss_min>-?[0-9.]+)/(?P<obss_mean>-?[0-9.]+)/(?P<obss_max>-?[0-9.]+)\s+"
    r"valid=(?P<valid>[0-9.]+)\s+sem_safe=(?P<sem_safe>[0-9.]+)\s+"
    r"sem_danger=(?P<sem_danger>[0-9.]+)"
)

MAPPING_RE = re.compile(
    r"Action mapping\s+frame=(?P<frame>\w+)\s+"
    r"lateral_sign=(?P<sign>-?\d+)\s+\(act3=(?P<act3>\w+)\)"
)

SAFE_POINT_RE = re.compile(
    r"\[GlobalPrior\]\s+Safe point from (?P<source>[^:]+):\s+"
    r"lat=(?P<lat>-?[0-9.]+)\s+lon=(?P<lon>-?[0-9.]+)"
)

GLOBAL_TARGET_RE = re.compile(
    r"\[Pipeline\]\s+Global guidance target:\s+"
    r"(?P<lat>-?[0-9.]+),\s*(?P<lon>-?[0-9.]+)"
)

GOTO_TARGET_RE = re.compile(
    r"\[GOTO_SAFE\]\s+Target NED:\s+"
    r"(?P<north>-?[0-9.]+),\s*(?P<east>-?[0-9.]+)\s+"
    r"tolerance=(?P<tolerance>[0-9.]+)m"
)

GOTO_ARRIVED_RE = re.compile(
    r"\[GOTO_SAFE\]\s+Arrived\.\s+XY error=(?P<xy_error>[0-9.]+)m"
)


def _percentile(values, q):
    if not values:
        return math.nan
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * q / 100.0
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    frac = pos - lo
    return ordered[lo] * (1.0 - frac) + ordered[hi] * frac


def _parse_vec(text):
    values = []
    for part in text.split(","):
        part = part.strip()
        if part:
            values.append(float(part))
    return values


def _yaw_transform_error(frame):
    v_body = frame.get("v_body") or []
    v_ned = frame.get("v_ned") or []
    if len(v_body) < 3 or len(v_ned) < 3:
        return math.nan
    yaw = math.radians(frame["yaw"])
    c = math.cos(yaw)
    s = math.sin(yaw)
    expected = [
        c * v_body[0] - s * v_body[1],
        s * v_body[0] + c * v_body[1],
        v_body[2],
    ]
    return math.sqrt(sum((expected[idx] - v_ned[idx]) ** 2 for idx in range(3)))


def _longest_run(values):
    if not values:
        return 0
    longest = 1
    current = 1
    for idx in range(1, len(values)):
        if values[idx] == values[idx - 1]:
            current += 1
        else:
            longest = max(longest, current)
            current = 1
    return max(longest, current)


def _line_index(lines, pattern):
    for idx, line in enumerate(lines):
        if pattern in line:
            return idx
    return -1


def parse_log(path: Path):
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    frames = []
    action_mapping = None
    safe_points = []
    global_targets = []
    goto_targets = []
    goto_arrivals = []
    for idx, line in enumerate(lines):
        mapping_match = MAPPING_RE.search(line)
        if mapping_match:
            action_mapping = {
                "frame": mapping_match.group("frame"),
                "sign": int(mapping_match.group("sign")),
                "act3": mapping_match.group("act3"),
            }
        safe_match = SAFE_POINT_RE.search(line)
        if safe_match:
            safe_points.append({
                "line": idx,
                "source": safe_match.group("source").strip(),
                "lat": float(safe_match.group("lat")),
                "lon": float(safe_match.group("lon")),
            })
        target_match = GLOBAL_TARGET_RE.search(line)
        if target_match:
            global_targets.append({
                "line": idx,
                "lat": float(target_match.group("lat")),
                "lon": float(target_match.group("lon")),
            })
        goto_target_match = GOTO_TARGET_RE.search(line)
        if goto_target_match:
            goto_targets.append({
                "line": idx,
                "north": float(goto_target_match.group("north")),
                "east": float(goto_target_match.group("east")),
                "tolerance": float(goto_target_match.group("tolerance")),
            })
        goto_arrived_match = GOTO_ARRIVED_RE.search(line)
        if goto_arrived_match:
            goto_arrivals.append({
                "line": idx,
                "xy_error": float(goto_arrived_match.group("xy_error")),
            })
        match = FRAME_RE.search(line)
        if not match:
            continue
        data = match.groupdict()
        frame = {
            "line": idx,
            "seq": int(data["seq"]),
            "action": int(data["action"]),
            "action_name": data["name"],
            "has_probs": "p=" in line,
            "v_body": _parse_vec(data["v_body"]),
            "v_ned": _parse_vec(data["v_ned"]),
        }
        for key in (
            "H", "D", "C", "RL", "total", "yaw", "yaw_sp", "yr",
            "sync",
            "depth_min", "depth_mean", "depth_max",
            "obsd_min", "obsd_mean", "obsd_max",
            "obss_min", "obss_mean", "obss_max",
            "valid", "sem_safe", "sem_danger",
        ):
            frame[key] = float(data[key])
        frames.append(frame)

    events = {
        "flight_ready_pass": _line_index(lines, "[FlightReady] Strict experiment gates passed"),
        "flight_ready_bypass": _line_index(lines, "Strict gates bypassed"),
        "global_safe_point": _line_index(lines, "[GlobalPrior] Safe point"),
        "global_target": _line_index(lines, "[Pipeline] Global guidance target"),
        "goto_target": _line_index(lines, "[GOTO_SAFE] Target NED"),
        "goto_arrived": _line_index(lines, "[GOTO_SAFE] Arrived"),
        "drl_start": _line_index(lines, "[Pipeline] Starting DRL descent control loop"),
        "fastlio_ready": _line_index(lines, "[Pipeline] FAST-LIO ready"),
        "landed": _line_index(lines, "[Pipeline] Target altitude reached"),
        "aborted": _line_index(lines, "ABORT"),
        "fatal": _line_index(lines, "Fatal error"),
        "emergency": _line_index(lines, "Emergency"),
    }
    slow_frames = sum(1 for line in lines if "SLOW FRAME DROPPED" in line)
    action_collapse = sum(1 for line in lines if "DRL action collapse" in line)
    pointcloud_errors = sum(
        1 for line in lines
        if "PointCloud" in line and ("error" in line.lower() or "exception" in line.lower())
    )
    return frames, events, {
        "slow_frames": slow_frames,
        "action_collapse": action_collapse,
        "pointcloud_errors": pointcloud_errors,
        "line_count": len(lines),
        "safe_points": safe_points,
        "global_targets": global_targets,
        "goto_targets": goto_targets,
        "goto_arrivals": goto_arrivals,
    }, action_mapping


def summarize(frames):
    actions = [frame["action"] for frame in frames]
    totals = [frame["total"] for frame in frames]
    module_p95 = {
        key: _percentile([frame[key] for frame in frames], 95)
        for key in ("H", "D", "C", "RL")
    }
    yaw_rates = [abs(frame["yr"]) for frame in frames]
    yaw_spans = []
    if frames:
        yaw_spans.append(max(frame["yaw_sp"] for frame in frames) - min(frame["yaw_sp"] for frame in frames))
    nonzero_velocity = sum(
        1 for frame in frames
        if len(frame["v_ned"]) >= 3 and any(abs(value) > 1e-6 for value in frame["v_ned"])
    )
    yaw_errors = [
        value for value in (_yaw_transform_error(frame) for frame in frames)
        if math.isfinite(value)
    ]
    return {
        "count": len(frames),
        "actions": Counter(actions),
        "unique_actions": sorted(set(actions)),
        "longest_action_run": _longest_run(actions),
        "total_p50": _percentile(totals, 50),
        "total_p95": _percentile(totals, 95),
        "module_p95": module_p95,
        "sync_p95": _percentile([frame["sync"] for frame in frames], 95),
        "yaw_transform_error_p95": _percentile(yaw_errors, 95),
        "yr_mean_abs": sum(yaw_rates) / len(yaw_rates) if yaw_rates else 0.0,
        "yaw_sp_span": max(yaw_spans) if yaw_spans else 0.0,
        "valid_mean": sum(frame["valid"] for frame in frames) / len(frames) if frames else 0.0,
        "sem_safe_mean": sum(frame["sem_safe"] for frame in frames) / len(frames) if frames else 0.0,
        "sem_danger_mean": sum(frame["sem_danger"] for frame in frames) / len(frames) if frames else 0.0,
        "with_probs": sum(1 for frame in frames if frame["has_probs"]),
        "nonzero_velocity": nonzero_velocity,
    }


def _event_ok(events, name):
    return events.get(name, -1) >= 0


def _ordered(events, before, after):
    return _event_ok(events, before) and _event_ok(events, after) and events[before] < events[after]


def action_name_mismatches(frames, action_lateral_sign):
    expected_names = ACTION_NAMES_BY_SIGN[action_lateral_sign]
    mismatches = []
    for frame in frames:
        action = frame["action"]
        expected = expected_names[action] if 0 <= action < len(expected_names) else None
        if expected != frame["action_name"]:
            mismatches.append(
                (frame["seq"], action, frame["action_name"], expected)
            )
    return mismatches


def validate_action_mapping(action_mapping, action_lateral_sign):
    expected_names = ACTION_NAMES_BY_SIGN[action_lateral_sign]
    expected_act3 = expected_names[3]
    if action_mapping is None:
        return "runtime action mapping log is missing"
    failures = []
    if action_mapping.get("frame") != "body":
        failures.append(f"frame={action_mapping.get('frame')} expected=body")
    if action_mapping.get("sign") != action_lateral_sign:
        failures.append(
            f"lateral_sign={action_mapping.get('sign')} expected={action_lateral_sign}"
        )
    if action_mapping.get("act3") != expected_act3:
        failures.append(
            f"act3={action_mapping.get('act3')} expected={expected_act3}"
        )
    return "; ".join(failures)


def _parse_lat_lon(value: str):
    parts = [float(part.strip()) for part in value.split(",")]
    if len(parts) != 2:
        raise ValueError("expected lat,lon")
    return parts[0], parts[1]


def _gps_error_m(a_lat, a_lon, b_lat, b_lon):
    meters_per_deg_lat = 111320.0
    mean_lat = math.radians((a_lat + b_lat) * 0.5)
    meters_per_deg_lon = 111320.0 * math.cos(mean_lat)
    dn = (a_lat - b_lat) * meters_per_deg_lat
    de = (a_lon - b_lon) * meters_per_deg_lon
    return math.sqrt(dn * dn + de * de)


def validate_expected_safe_point(counters, expected_lat, expected_lon, tolerance_m):
    failures = []
    safe_points = counters.get("safe_points") or []
    global_targets = counters.get("global_targets") or []
    if not safe_points:
        failures.append("no [GlobalPrior] Safe point lat/lon log found")
    else:
        best = min(
            _gps_error_m(item["lat"], item["lon"], expected_lat, expected_lon)
            for item in safe_points
        )
        if best > tolerance_m:
            failures.append(
                f"safe-point GPS mismatch {best:.2f}m > {tolerance_m:.2f}m"
            )
    if not global_targets:
        failures.append("no [Pipeline] Global guidance target lat/lon log found")
    else:
        best = min(
            _gps_error_m(item["lat"], item["lon"], expected_lat, expected_lon)
            for item in global_targets
        )
        if best > tolerance_m:
            failures.append(
                f"global guidance target GPS mismatch {best:.2f}m > {tolerance_m:.2f}m"
            )
    return "; ".join(failures)


def validate_goto_arrival(counters):
    goto_targets = counters.get("goto_targets") or []
    goto_arrivals = counters.get("goto_arrivals") or []
    failures = []
    if not goto_targets:
        failures.append("no parseable [GOTO_SAFE] Target NED numeric log found")
    if not goto_arrivals:
        failures.append("no parseable [GOTO_SAFE] Arrived XY error log found")
    if not goto_targets or not goto_arrivals:
        return "; ".join(failures)

    target = goto_targets[-1]
    arrival = goto_arrivals[-1]
    values = [target["north"], target["east"], target["tolerance"], arrival["xy_error"]]
    if not all(math.isfinite(value) for value in values):
        failures.append("GOTO_SAFE target/arrival values must be finite")
    if target["tolerance"] <= 0.0:
        failures.append(f"GOTO_SAFE tolerance must be positive, got {target['tolerance']:.2f}m")
    if arrival["xy_error"] > target["tolerance"]:
        failures.append(
            f"GOTO_SAFE arrived XY error {arrival['xy_error']:.2f}m > "
            f"tolerance {target['tolerance']:.2f}m"
        )
    if math.hypot(target["north"], target["east"]) <= 1e-3:
        failures.append(
            "GOTO_SAFE target NED is near zero; safe point may be unset or equal to home"
        )
    return "; ".join(failures)


def _check_module_p95(summary, limits):
    names = {
        "H": "HALSS",
        "D": "depth projection",
        "C": "depth completion",
        "RL": "DRL",
    }
    failures = []
    if summary["count"] <= 0:
        return failures
    for key, limit in limits.items():
        if limit is None:
            continue
        value = summary["module_p95"][key]
        if value > limit:
            failures.append(f"{names[key]} P95 {value:.1f}ms > {limit:.1f}ms")
    return failures


def print_summary(summary, events, counters):
    print("Flight-loop log acceptance summary")
    print(f"  frames: {summary['count']}")
    print(f"  actions: {dict(summary['actions'])}")
    print(f"  unique_actions: {summary['unique_actions']}")
    print(f"  longest_action_run: {summary['longest_action_run']}")
    print(f"  total_ms: p50={summary['total_p50']:.1f} p95={summary['total_p95']:.1f}")
    print(f"  cloud_odom_sync_ms: p95={summary['sync_p95']:.1f}")
    print(f"  yaw_transform_error: p95={summary['yaw_transform_error_p95']:.3f}")
    print(
        "  module_p95_ms: "
        f"H={summary['module_p95']['H']:.1f} "
        f"D={summary['module_p95']['D']:.1f} "
        f"C={summary['module_p95']['C']:.1f} "
        f"RL={summary['module_p95']['RL']:.1f}"
    )
    print(f"  yaw_rate_abs_mean: {summary['yr_mean_abs']:.3f}")
    print(f"  yaw_sp_span_deg: {summary['yaw_sp_span']:.2f}")
    print(f"  nonzero_velocity_frames: {summary['nonzero_velocity']}")
    print(
        "  semantics_mean: "
        f"safe={summary['sem_safe_mean']:.3f} danger={summary['sem_danger_mean']:.3f}"
    )
    print(f"  valid_mean: {summary['valid_mean']:.3f}")
    print(f"  frames_with_probs: {summary['with_probs']}")
    print(f"  slow_frames: {counters['slow_frames']}")
    print(f"  action_collapse_warnings: {counters['action_collapse']}")
    print(f"  pointcloud_errors: {counters['pointcloud_errors']}")
    print(f"  goto_targets: {counters.get('goto_targets')}")
    print(f"  goto_arrivals: {counters.get('goto_arrivals')}")
    print(
        "  events: "
        + ", ".join(f"{key}={value}" for key, value in sorted(events.items()))
    )


def main():
    parser = argparse.ArgumentParser(description="Accept closed-loop pipeline log")
    parser.add_argument("log", help="Log captured from pipeline.py --mode ros")
    parser.add_argument("--min-drl-frames", type=int, default=20)
    parser.add_argument("--max-total-p95-ms", type=float, default=100.0)
    parser.add_argument("--max-halss-p95-ms", type=float, default=70.0,
                        help="Fail if HALSS H= P95 exceeds this budget")
    parser.add_argument("--max-depth-p95-ms", type=float, default=15.0,
                        help="Fail if depth projection D= P95 exceeds this budget")
    parser.add_argument("--max-completion-p95-ms", type=float, default=45.0,
                        help="Fail if depth completion C= P95 exceeds this budget")
    parser.add_argument("--max-rl-p95-ms", type=float, default=30.0,
                        help="Fail if DRL RL= P95 exceeds this budget")
    parser.add_argument("--max-sync-p95-ms", type=float, default=100.0,
                        help="Fail if cloud/odom header sync P95 exceeds this budget")
    parser.add_argument("--max-yaw-transform-error", type=float, default=0.15,
                        help="Fail if body->NED velocity rotation error P95 exceeds this m/s")
    parser.add_argument("--expected-yaw-rate", type=float, default=None)
    parser.add_argument("--yaw-rate-tol", type=float, default=0.05)
    parser.add_argument("--require-global-guidance", action="store_true",
                        help="Require GOTO_SAFE target and arrival before DRL descent")
    parser.add_argument("--require-landing", action="store_true",
                        help="Require target-altitude reached log")
    parser.add_argument("--require-action-probs", action="store_true")
    parser.add_argument("--expected-safe-point", default=None,
                        help="Expected GIS-derived safe point as lat,lon")
    parser.add_argument("--safe-point-tolerance-m", type=float, default=2.0)
    parser.add_argument("--max-action-run", type=int, default=60,
                        help="Fail if the same DRL action repeats longer than this")
    parser.add_argument("--allow-single-action", action="store_true",
                        help="Do not fail on one unique DRL action; still report it")
    parser.add_argument("--allow-slow-frames", action="store_true")
    parser.add_argument("--allow-action-collapse", action="store_true")
    parser.add_argument("--action-lateral-sign", type=int, choices=(-1, 1), default=-1,
                        help="Expected runtime action mapping: -1 original DeepRL, +1 mirror")
    args = parser.parse_args()

    frames, events, counters, action_mapping = parse_log(Path(args.log))
    summary = summarize(frames)
    print_summary(summary, events, counters)
    print(f"  action_mapping: {action_mapping}")
    print(f"  safe_points: {counters.get('safe_points')}")
    print(f"  global_targets: {counters.get('global_targets')}")

    failures = []
    if not _event_ok(events, "flight_ready_pass"):
        failures.append("strict FlightReady pass log is missing")
    if _event_ok(events, "flight_ready_bypass"):
        failures.append("strict FlightReady gates were bypassed")
    if not _event_ok(events, "fastlio_ready"):
        failures.append("FAST-LIO ready log is missing")

    mapping_failure = validate_action_mapping(action_mapping, args.action_lateral_sign)
    if mapping_failure:
        failures.append("action mapping evidence invalid: " + mapping_failure)
    mismatches = action_name_mismatches(frames, args.action_lateral_sign)
    if mismatches:
        shown = ", ".join(
            f"seq{seq}: act={action} logged={logged} expected={expected}"
            for seq, action, logged, expected in mismatches[:5]
        )
        failures.append(
            "action id/name mismatch for action_lateral_sign="
            f"{args.action_lateral_sign}: {shown}"
        )

    if args.expected_safe_point:
        try:
            expected_lat, expected_lon = _parse_lat_lon(args.expected_safe_point)
            safe_failure = validate_expected_safe_point(
                counters,
                expected_lat,
                expected_lon,
                args.safe_point_tolerance_m,
            )
            if safe_failure:
                failures.append(safe_failure)
        except ValueError as exc:
            failures.append(f"invalid --expected-safe-point: {exc}")

    if args.require_global_guidance:
        for event in ("global_safe_point", "global_target", "goto_target", "goto_arrived", "drl_start"):
            if not _event_ok(events, event):
                failures.append(f"required global-guidance event missing: {event}")
        goto_failure = validate_goto_arrival(counters)
        if goto_failure:
            failures.append(goto_failure)
        if _event_ok(events, "goto_arrived") and frames and events["goto_arrived"] > frames[0]["line"]:
            failures.append("first DRL action frame appeared before GOTO_SAFE arrival")
        if not _ordered(events, "goto_target", "goto_arrived"):
            failures.append("GOTO_SAFE arrival did not occur after target log")
        if not _ordered(events, "goto_arrived", "drl_start"):
            failures.append("DRL descent did not start after GOTO_SAFE arrival")

    if summary["count"] < args.min_drl_frames:
        failures.append(f"only {summary['count']} DRL frames parsed; need {args.min_drl_frames}")
    if summary["count"] and summary["total_p95"] > args.max_total_p95_ms:
        failures.append(
            f"total P95 {summary['total_p95']:.1f}ms > {args.max_total_p95_ms:.1f}ms"
        )
    if summary["count"] and summary["sync_p95"] > args.max_sync_p95_ms:
        failures.append(
            f"cloud/odom sync P95 {summary['sync_p95']:.1f}ms > {args.max_sync_p95_ms:.1f}ms"
        )
    if summary["count"] and summary["yaw_transform_error_p95"] > args.max_yaw_transform_error:
        failures.append(
            f"yaw transform error P95 {summary['yaw_transform_error_p95']:.3f} > "
            f"{args.max_yaw_transform_error:.3f}"
        )
    failures.extend(_check_module_p95(summary, {
        "H": args.max_halss_p95_ms,
        "D": args.max_depth_p95_ms,
        "C": args.max_completion_p95_ms,
        "RL": args.max_rl_p95_ms,
    }))

    if args.expected_yaw_rate is not None:
        expected = abs(args.expected_yaw_rate)
        low = max(0.0, expected - args.yaw_rate_tol)
        high = expected + args.yaw_rate_tol
        if summary["yr_mean_abs"] < low or summary["yr_mean_abs"] > high:
            failures.append(
                f"mean |yr| {summary['yr_mean_abs']:.3f} outside "
                f"{expected:.3f}±{args.yaw_rate_tol:.3f}"
            )
    elif summary["yr_mean_abs"] <= 1e-6:
        failures.append("yaw_rate is zero; yaw-fault command was not logged")

    if summary["yaw_sp_span"] <= 0.0 and summary["yr_mean_abs"] > 1e-6 and summary["count"] > 1:
        failures.append("yaw_sp did not change despite nonzero yaw_rate")
    if summary["nonzero_velocity"] <= 0:
        failures.append("no nonzero NED velocity commands logged")
    if not args.allow_single_action and len(summary["unique_actions"]) <= 1:
        failures.append("all parsed DRL frames selected one action")
    if summary["longest_action_run"] > args.max_action_run:
        failures.append(
            f"longest repeated action run {summary['longest_action_run']} > {args.max_action_run}"
        )
    if args.require_action_probs and summary["with_probs"] < summary["count"]:
        failures.append(
            f"only {summary['with_probs']}/{summary['count']} frames include action probabilities"
        )
    if counters["slow_frames"] and not args.allow_slow_frames:
        failures.append(f"{counters['slow_frames']} slow frames were dropped")
    if counters["action_collapse"] and not args.allow_action_collapse:
        failures.append(f"{counters['action_collapse']} action collapse warnings present")
    if counters["pointcloud_errors"]:
        failures.append(f"{counters['pointcloud_errors']} PointCloud callback errors present")
    if _event_ok(events, "fatal"):
        failures.append("fatal pipeline error was logged")
    if _event_ok(events, "emergency"):
        failures.append("emergency stop was logged")
    if _event_ok(events, "aborted"):
        failures.append("ABORT was logged")
    if args.require_landing and not _event_ok(events, "landed"):
        failures.append("landing target-altitude log is missing")

    if failures:
        for failure in failures:
            print(f"[FAIL] {failure}")
        return 1
    print("[OK] flight-loop log acceptance passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
