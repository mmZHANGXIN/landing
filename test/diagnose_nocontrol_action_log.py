#!/usr/bin/env python3
"""Explain repeated DRL actions in no-control Mid360/FAST-LIO logs.

This is a field-debug helper, not a replacement for accept_nocontrol_log.py.
It turns logs such as repeated ``act=3(E)`` with ``yr=0.00`` into concrete
diagnostic findings and next commands.
"""

from __future__ import annotations

import argparse
import math
from collections import Counter
from pathlib import Path

from accept_nocontrol_log import (
    ACTION_NAMES_BY_SIGN,
    action_name_mismatches,
    legacy_summary,
    parse_log,
    summarize,
    validate_action_mapping,
)


def _fmt_counter(counter):
    if not counter:
        return "{}"
    return "{" + ", ".join(f"{key}: {value}" for key, value in sorted(counter.items())) + "}"


def _issue(code, severity, message, evidence, next_step):
    return {
        "code": code,
        "severity": severity,
        "message": message,
        "evidence": evidence,
        "next": next_step,
    }


def _obs_span_issue(summary):
    if summary["count"] <= 0:
        return None
    if summary["valid_mean"] < 0.005:
        return _issue(
            "LOW_VALID_DEPTH",
            "FAIL",
            "Sparse depth has too few valid pixels for a reliable DRL observation.",
            f"valid_mean={summary['valid_mean']:.3f}",
            "Inspect Mid360 mounting, FAST-LIO cloud frame, depth_projection intrinsics/extrinsics, and saved sparse_depth/valid_mask arrays.",
        )
    if summary["sem_safe_mean"] > 0.98 or summary["sem_danger_mean"] > 0.98:
        return _issue(
            "SEMANTIC_ONE_CLASS",
            "FAIL",
            "HALSS semantic map is nearly one class, so the policy may see a constant scene.",
            f"sem_safe_mean={summary['sem_safe_mean']:.3f} sem_danger_mean={summary['sem_danger_mean']:.3f}",
            "Open saved binary semantic frames and compare them with HALSS safety_map_vis; check Bayesian UNet weights/backend and point cloud ROI.",
        )
    return None


def analyze_log(path: Path, expected_sign=-1, expected_yaw_rate=None, max_action_run=60):
    frames, legacy_frames, collapse_lines, callback_errors, action_mapping = parse_log(path)
    summary = summarize(frames)
    legacy = legacy_summary(legacy_frames)
    issues = []

    mapping_failure = validate_action_mapping(action_mapping, expected_sign)
    if mapping_failure:
        issues.append(_issue(
            "ACTION_MAPPING_MISMATCH",
            "FAIL",
            "Runtime action names do not match the expected DeepRL action mapping.",
            mapping_failure,
            "Use uav.action_lateral_sign=-1 for the default experiment; accept +1 only for an explicit mirrored-action ablation.",
        ))

    mismatches = action_name_mismatches(frames, expected_sign)
    mismatches.extend(action_name_mismatches(legacy_frames, expected_sign))
    if mismatches:
        shown = ", ".join(
            f"seq{seq}: act={action} logged={logged} expected={expected}"
            for seq, action, logged, expected in mismatches[:5]
        )
        issues.append(_issue(
            "ACTION_NAME_MISMATCH",
            "FAIL",
            "Logged action id/name pairs contradict the expected runtime mapping.",
            shown,
            "Rerun with startup log 'Action mapping: frame=body lateral_sign=-1 act3=W' before interpreting act=3 as a physical direction.",
        ))

    if legacy_frames:
        issues.append(_issue(
            "LEGACY_LOG_FORMAT",
            "FAIL",
            "The log lacks observation statistics, sync evidence, and action probabilities.",
            f"legacy_frames={len(legacy_frames)}",
            "Rerun test_live_nocontrol.py so every frame includes sync, depth, obsD/obsS, valid, sem_safe/sem_danger, and p=...",
        ))

    yaw_mean = summary["yr_mean_abs"] if summary["count"] else legacy.get("yr_mean_abs", 0.0)
    if expected_yaw_rate is not None:
        expected = abs(float(expected_yaw_rate))
        if abs(yaw_mean - expected) > 0.05:
            issues.append(_issue(
                "YAW_RATE_MISMATCH",
                "FAIL",
                "The run did not log the requested yaw-fault rate.",
                f"mean_abs_yr={yaw_mean:.3f} expected={expected:.3f}+/-0.050",
                "Rerun with --yaw-rate-rad-s <rate> and --require-yaw-rate; zero-yaw logs are not valid yaw-fault evidence.",
            ))
    elif yaw_mean <= 1e-6:
        issues.append(_issue(
            "ZERO_YAW_RATE",
            "FAIL",
            "The run used yr=0.00, so it did not exercise the yaw-fault control path.",
            f"mean_abs_yr={yaw_mean:.3f}",
            "Set uav.yaw_rate_rad_s or pass --yaw-rate-rad-s <rate> and --require-yaw-rate.",
        ))

    action_counter = summary["actions"] if summary["count"] else legacy.get("actions", Counter())
    unique_actions = summary["unique_actions"] if summary["count"] else legacy.get("unique_actions", [])
    longest = summary["longest_action_run"] if summary["count"] else legacy.get("longest_action_run", 0)
    if len(unique_actions) <= 1 and action_counter:
        action_id = unique_actions[0]
        expected_names = ACTION_NAMES_BY_SIGN.get(expected_sign, ACTION_NAMES_BY_SIGN[-1])
        expected_name = expected_names[action_id] if 0 <= action_id < len(expected_names) else "?"
        issues.append(_issue(
            "SINGLE_ACTION_COLLAPSE",
            "FAIL",
            "All parsed frames selected one action.",
            f"actions={_fmt_counter(action_counter)} expected_name_for_action={expected_name}",
            "Run diagnose_drl_policy.py --scan-modes on saved *_calib_frame.npz files to separate observation collapse from policy/encoding mismatch.",
        ))
    elif longest > max_action_run:
        issues.append(_issue(
            "LONG_ACTION_RUN",
            "FAIL",
            "The same action repeats longer than the allowed run length.",
            f"longest_action_run={longest} max_action_run={max_action_run}",
            "Inspect action probabilities and saved action-collapse NPZ snapshots around the repeated segment.",
        ))

    if collapse_lines:
        issues.append(_issue(
            "COLLAPSE_WARNING_PRESENT",
            "FAIL",
            "Runtime action-collapse monitor already detected repeated deterministic actions.",
            f"collapse_warnings={collapse_lines}",
            "Open the saved *_action_collapse_a<id>_*.npz and run diagnose_drl_policy.py --frame on that snapshot.",
        ))

    if summary["count"] and summary["with_probs"] < summary["count"]:
        issues.append(_issue(
            "MISSING_ACTION_PROBS",
            "WARN",
            "Not every modern frame includes PPO action probabilities.",
            f"frames_with_probs={summary['with_probs']}/{summary['count']}",
            "Ensure stable-baselines3 policy distribution is available; formal acceptance should use --require-action-probs.",
        ))

    obs_issue = _obs_span_issue(summary)
    if obs_issue is not None:
        issues.append(obs_issue)

    if callback_errors:
        issues.append(_issue(
            "POINTCLOUD_CALLBACK_ERRORS",
            "FAIL",
            "PointCloud callback errors occurred while parsing live Mid360/FAST-LIO data.",
            f"errors={callback_errors}",
            "Fix PointCloud2 field offsets and ROS topic types before interpreting DRL behavior.",
        ))

    return {
        "frames": frames,
        "legacy_frames": legacy_frames,
        "summary": summary,
        "legacy_summary": legacy,
        "action_mapping": action_mapping,
        "issues": issues,
    }


def print_report(result):
    summary = result["summary"]
    legacy = result["legacy_summary"]
    print("No-control action diagnosis")
    print(f"  modern_frames: {summary['count']}")
    print(f"  legacy_frames: {len(result['legacy_frames'])}")
    print(f"  action_mapping: {result['action_mapping']}")
    print(f"  modern_actions: {_fmt_counter(summary['actions'])}")
    if legacy:
        print(f"  legacy_actions: {_fmt_counter(legacy['actions'])}")
    yaw_mean = summary["yr_mean_abs"] if summary["count"] else legacy.get("yr_mean_abs", math.nan)
    print(f"  yaw_rate_abs_mean: {yaw_mean:.3f}")
    if summary["count"]:
        print(f"  valid_mean: {summary['valid_mean']:.3f}")
        print(f"  semantic_mean: safe={summary['sem_safe_mean']:.3f} danger={summary['sem_danger_mean']:.3f}")
        print(f"  frames_with_probs: {summary['with_probs']}/{summary['count']}")
    print("")
    if not result["issues"]:
        print("[OK] No action-collapse diagnosis issues detected in this log.")
        return
    print("Findings")
    for issue in result["issues"]:
        print(f"[{issue['severity']}] {issue['code']}: {issue['message']}")
        print(f"  evidence: {issue['evidence']}")
        print(f"  next: {issue['next']}")


def main():
    parser = argparse.ArgumentParser(description="Diagnose repeated no-control DRL actions")
    parser.add_argument("log", help="Log captured from test_live_nocontrol.py")
    parser.add_argument("--action-lateral-sign", type=int, choices=(-1, 1), default=-1)
    parser.add_argument("--expected-yaw-rate", type=float, default=None)
    parser.add_argument("--max-action-run", type=int, default=60)
    parser.add_argument("--fail-on-issues", action="store_true")
    args = parser.parse_args()

    result = analyze_log(
        Path(args.log),
        expected_sign=args.action_lateral_sign,
        expected_yaw_rate=args.expected_yaw_rate,
        max_action_run=args.max_action_run,
    )
    print_report(result)
    if args.fail_on_issues and result["issues"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
