#!/usr/bin/env python3
"""Acceptance checks for Mid360/FAST-LIO no-control live logs.

This is the lightweight post-run gate for ``test_live_nocontrol.py``. It checks
that timing, yaw-fault command, DRL observation stats, semantic ratios, and
action distribution were actually logged and are not obviously degenerate.
"""

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
    r"total=(?P<total>[0-9.]+)ms.*?"
    r"yaw=(?P<yaw>-?[0-9.]+)(?:°|deg)\s+"
    r"(?:yaw_sp=-?[0-9.]+deg\s+)?yr=(?P<yr>-?[0-9.]+).*?"
    r"sync=(?P<sync>[0-9.]+)ms.*?"
    r"v_body=\[(?P<v_body>[^\]]+)\]\s+v_ned=\[(?P<v_ned>[^\]]+)\].*?"
    r"depth=(?P<depth_min>-?[0-9.]+)/(?P<depth_mean>-?[0-9.]+)/(?P<depth_max>-?[0-9.]+)m\s+"
    r"obsD=(?P<obsd_min>-?[0-9.]+)/(?P<obsd_mean>-?[0-9.]+)/(?P<obsd_max>-?[0-9.]+)\s+"
    r"obsS=(?P<obss_min>-?[0-9.]+)/(?P<obss_mean>-?[0-9.]+)/(?P<obss_max>-?[0-9.]+)\s+"
    r"valid=(?P<valid>[0-9.]+)\s+sem_safe=(?P<sem_safe>[0-9.]+)\s+"
    r"sem_danger=(?P<sem_danger>[0-9.]+)"
)

LEGACY_FRAME_RE = re.compile(
    r"\[(?P<seq>\d+)\]\s+act=(?P<action>\d+)\((?P<name>[^)]+)\)\s+"
    r"H=(?P<H>[0-9.]+)ms\s+D=(?P<D>[0-9.]+)ms\s+"
    r"C=(?P<C>[0-9.]+)ms\s+RL=(?P<RL>[0-9.]+)ms\s+"
    r"total=(?P<total>[0-9.]+)ms.*?"
    r"yaw=(?P<yaw>-?[0-9.]+)(?:°|deg)\s+yr=(?P<yr>-?[0-9.]+).*?"
    r"v_body=(?P<v_body>\[[^\]]+\])\s+v_ned=(?P<v_ned>\[[^\]]+\])"
)

MAPPING_RE = re.compile(
    r"Action mapping:\s+frame=(?P<frame>\w+)\s+"
    r"lateral_sign=(?P<sign>-?\d+)\s+act3=(?P<act3>\w+)"
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


def parse_log(path: Path):
    frames = []
    legacy_frames = []
    collapse_lines = 0
    missing_callback_errors = 0
    action_mapping = None
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        mapping_match = MAPPING_RE.search(line)
        if mapping_match:
            action_mapping = {
                "frame": mapping_match.group("frame"),
                "sign": int(mapping_match.group("sign")),
                "act3": mapping_match.group("act3"),
            }
        if "DRL action collapse" in line:
            collapse_lines += 1
        if "PointCloud" in line and ("error" in line.lower() or "exception" in line.lower()):
            missing_callback_errors += 1
        match = FRAME_RE.search(line)
        if not match:
            legacy_match = LEGACY_FRAME_RE.search(line)
            if legacy_match:
                legacy_frames.append({
                    "seq": int(legacy_match.group("seq")),
                    "action": int(legacy_match.group("action")),
                    "action_name": legacy_match.group("name"),
                    "total": float(legacy_match.group("total")),
                    "yaw": float(legacy_match.group("yaw")),
                    "yr": float(legacy_match.group("yr")),
                })
            continue
        data = match.groupdict()
        frame = {
            "seq": int(data["seq"]),
            "action": int(data["action"]),
            "action_name": data["name"],
            "has_probs": "p=" in line,
            "v_body": _parse_vec(data["v_body"]),
            "v_ned": _parse_vec(data["v_ned"]),
        }
        for key in (
            "H", "D", "C", "RL", "total", "yaw", "yr",
            "sync",
            "depth_min", "depth_mean", "depth_max",
            "obsd_min", "obsd_mean", "obsd_max",
            "obss_min", "obss_mean", "obss_max",
            "valid", "sem_safe", "sem_danger",
        ):
            frame[key] = float(data[key])
        frames.append(frame)
    return frames, legacy_frames, collapse_lines, missing_callback_errors, action_mapping


def summarize(frames):
    actions = [frame["action"] for frame in frames]
    totals = [frame["total"] for frame in frames]
    module_p95 = {
        key: _percentile([frame[key] for frame in frames], 95)
        for key in ("H", "D", "C", "RL")
    }
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
        "total_max": max(totals) if totals else math.nan,
        "module_p95": module_p95,
        "sync_p95": _percentile([frame["sync"] for frame in frames], 95),
        "yaw_transform_error_p95": _percentile(yaw_errors, 95),
        "valid_mean": sum(frame["valid"] for frame in frames) / len(frames) if frames else 0.0,
        "sem_safe_mean": sum(frame["sem_safe"] for frame in frames) / len(frames) if frames else 0.0,
        "sem_danger_mean": sum(frame["sem_danger"] for frame in frames) / len(frames) if frames else 0.0,
        "depth_mean": sum(frame["depth_mean"] for frame in frames) / len(frames) if frames else math.nan,
        "yr_mean_abs": (
            sum(abs(frame["yr"]) for frame in frames) / len(frames) if frames else 0.0
        ),
        "with_probs": sum(1 for frame in frames if frame["has_probs"]),
    }


def print_summary(summary, legacy_frames, collapse_lines, callback_errors, action_mapping):
    print("No-control log acceptance summary")
    print(f"  frames: {summary['count']}")
    print(f"  actions: {dict(summary['actions'])}")
    print(f"  unique_actions: {summary['unique_actions']}")
    print(f"  longest_action_run: {summary['longest_action_run']}")
    print(
        "  total_ms: "
        f"p50={summary['total_p50']:.1f} p95={summary['total_p95']:.1f} "
        f"max={summary['total_max']:.1f}"
    )
    print(f"  cloud_odom_sync_ms: p95={summary['sync_p95']:.1f}")
    print(f"  yaw_transform_error: p95={summary['yaw_transform_error_p95']:.3f}")
    print(
        "  module_p95_ms: "
        f"H={summary['module_p95']['H']:.1f} "
        f"D={summary['module_p95']['D']:.1f} "
        f"C={summary['module_p95']['C']:.1f} "
        f"RL={summary['module_p95']['RL']:.1f}"
    )
    print(f"  valid_mean: {summary['valid_mean']:.3f}")
    print(
        "  semantics_mean: "
        f"safe={summary['sem_safe_mean']:.3f} danger={summary['sem_danger_mean']:.3f}"
    )
    print(f"  depth_mean_m: {summary['depth_mean']:.3f}")
    print(f"  yaw_rate_abs_mean: {summary['yr_mean_abs']:.3f}")
    print(f"  frames_with_probs: {summary['with_probs']}")
    print(f"  legacy_frames_without_obs: {len(legacy_frames)}")
    print(f"  action_mapping: {action_mapping}")
    print(f"  action_collapse_warnings: {collapse_lines}")
    print(f"  pointcloud_callback_errors: {callback_errors}")


def _outside(value, low, high):
    return value < low or value > high


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


def legacy_summary(legacy_frames):
    if not legacy_frames:
        return {}
    return {
        "count": len(legacy_frames),
        "actions": Counter(frame["action"] for frame in legacy_frames),
        "unique_actions": sorted(set(frame["action"] for frame in legacy_frames)),
        "longest_action_run": _longest_run([frame["action"] for frame in legacy_frames]),
        "total_p95": _percentile([frame["total"] for frame in legacy_frames], 95),
        "yr_mean_abs": sum(abs(frame["yr"]) for frame in legacy_frames) / len(legacy_frames),
    }


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


def main():
    parser = argparse.ArgumentParser(description="Accept no-control live pipeline log")
    parser.add_argument("log", help="Log captured from test_live_nocontrol.py")
    parser.add_argument("--min-samples", type=int, default=120)
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
    parser.add_argument("--expected-yaw-rate", type=float, default=None,
                        help="Expected nonzero yaw_rate_rad_s from this run")
    parser.add_argument("--yaw-rate-tol", type=float, default=0.05)
    parser.add_argument("--min-valid-ratio", type=float, default=0.005)
    parser.add_argument("--max-sem-one-class-ratio", type=float, default=0.98)
    parser.add_argument("--max-action-run", type=int, default=60,
                        help="Fail if the same action repeats longer than this")
    parser.add_argument("--allow-single-action", action="store_true",
                        help="Do not fail on one unique action; still report it")
    parser.add_argument("--require-action-probs", action="store_true",
                        help="Require p=<id>:<name>:<prob> fields in every frame")
    parser.add_argument("--allow-collapse-warning", action="store_true",
                        help="Do not fail if DRL action collapse warnings appear")
    parser.add_argument("--action-lateral-sign", type=int, choices=(-1, 1), default=-1,
                        help="Expected runtime action mapping: -1 original DeepRL, +1 mirror")
    args = parser.parse_args()

    frames, legacy_frames, collapse_lines, callback_errors, action_mapping = parse_log(Path(args.log))
    summary = summarize(frames)
    print_summary(summary, legacy_frames, collapse_lines, callback_errors, action_mapping)
    if legacy_frames:
        legacy = legacy_summary(legacy_frames)
        print(
            "Legacy no-control frames without observation evidence: "
            f"count={legacy['count']} actions={dict(legacy['actions'])} "
            f"unique={legacy['unique_actions']} p95={legacy['total_p95']:.1f}ms "
            f"mean_abs_yr={legacy['yr_mean_abs']:.3f}"
        )

    failures = []
    if legacy_frames:
        failures.append(
            "legacy no-control log format detected; rerun test_live_nocontrol.py so each "
            "frame includes depth/obsD/obsS/valid/sem_safe/sem_danger/action probabilities"
        )
    if summary["count"] < args.min_samples:
        failures.append(f"only {summary['count']} frames parsed; need {args.min_samples}")
    mapping_failure = validate_action_mapping(action_mapping, args.action_lateral_sign)
    if mapping_failure:
        failures.append("action mapping evidence invalid: " + mapping_failure)
    mismatches = action_name_mismatches(frames, args.action_lateral_sign)
    mismatches.extend(action_name_mismatches(legacy_frames, args.action_lateral_sign))
    if mismatches:
        shown = ", ".join(
            f"seq{seq}: act={action} logged={logged} expected={expected}"
            for seq, action, logged, expected in mismatches[:5]
        )
        failures.append(
            "action id/name mismatch for action_lateral_sign="
            f"{args.action_lateral_sign}: {shown}"
        )
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
        if _outside(summary["yr_mean_abs"], expected - args.yaw_rate_tol, expected + args.yaw_rate_tol):
            failures.append(
                f"mean |yr| {summary['yr_mean_abs']:.3f} outside "
                f"{expected:.3f}±{args.yaw_rate_tol:.3f}"
            )
    elif summary["yr_mean_abs"] <= 1e-6:
        failures.append("yaw_rate is zero; no yaw-fault command was logged")
    if summary["valid_mean"] < args.min_valid_ratio:
        failures.append(
            f"valid depth ratio mean {summary['valid_mean']:.3f} < {args.min_valid_ratio:.3f}"
        )
    if summary["sem_safe_mean"] > args.max_sem_one_class_ratio:
        failures.append(
            f"semantic safe mean {summary['sem_safe_mean']:.3f} > {args.max_sem_one_class_ratio:.3f}"
        )
    if summary["sem_danger_mean"] > args.max_sem_one_class_ratio:
        failures.append(
            f"semantic danger mean {summary['sem_danger_mean']:.3f} > {args.max_sem_one_class_ratio:.3f}"
        )
    if not args.allow_single_action and len(summary["unique_actions"]) <= 1:
        failures.append("all parsed frames selected one action")
    if summary["longest_action_run"] > args.max_action_run:
        failures.append(
            f"longest repeated action run {summary['longest_action_run']} > {args.max_action_run}"
        )
    if args.require_action_probs and summary["with_probs"] < summary["count"]:
        failures.append(
            f"only {summary['with_probs']}/{summary['count']} frames include action probabilities"
        )
    if collapse_lines and not args.allow_collapse_warning:
        failures.append(f"{collapse_lines} DRL action collapse warnings present")
    if callback_errors:
        failures.append(f"{callback_errors} PointCloud callback errors present")

    if failures:
        for failure in failures:
            print(f"[FAIL] {failure}")
        return 1
    print("[OK] no-control log acceptance passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
