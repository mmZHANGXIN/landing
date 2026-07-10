#!/usr/bin/env python3
"""Accept a JSON report written by diagnose_drl_policy.py."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


ACTION_NAMES_BY_SIGN = {
    -1: ["HOVER", "N", "NW", "W", "SW", "S", "SE", "E", "NE", "DESCEND"],
    1: ["HOVER", "N", "NE", "E", "SE", "S", "SW", "W", "NW", "DESCEND"],
}


def _load(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _encoding_key(encoding):
    return (
        str(encoding.get("depth_mode")),
        str(encoding.get("semantic_mode")),
    )


def _finite(value):
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def _row_failures(rows, action_names, require_probs):
    failures = []
    for idx, row in enumerate(rows):
        action = row.get("action")
        if not isinstance(action, int) or not (0 <= action < len(action_names)):
            failures.append(f"row{idx}: invalid action={action!r}")
            continue
        expected = action_names[action]
        if row.get("action_name") != expected:
            failures.append(
                f"row{idx}: action {action} name={row.get('action_name')!r} expected={expected!r}"
            )
        for key in (
            "depth_norm_min", "depth_norm_mean", "depth_norm_max",
            "sem_norm_min", "sem_norm_mean", "sem_norm_max",
        ):
            if not _finite(row.get(key)):
                failures.append(f"row{idx}: {key} is not finite")
        if require_probs and (row.get("confidence") is None or row.get("top3") == "p=n/a"):
            failures.append(f"row{idx}: action probabilities unavailable")
    return failures


def accept_report(report, args):
    failures = []
    expected_names = ACTION_NAMES_BY_SIGN[args.action_lateral_sign]

    if report.get("action_lateral_sign") != args.action_lateral_sign:
        failures.append(
            "action_lateral_sign="
            f"{report.get('action_lateral_sign')!r} expected={args.action_lateral_sign}"
        )
    if report.get("action_frame") != "body":
        failures.append(f"action_frame={report.get('action_frame')!r} expected='body'")
    if report.get("action_names") != expected_names:
        failures.append("action_names do not match expected runtime mapping")

    items = report.get("items") or []
    if len(items) < args.min_items:
        failures.append(f"only {len(items)} input items; need {args.min_items}")
    if args.require_live_frame and report.get("input_set") != "live frames":
        failures.append(f"input_set={report.get('input_set')!r} expected='live frames'")

    encodings = report.get("encodings") or []
    if len(encodings) < args.min_encodings:
        failures.append(f"only {len(encodings)} encodings; need {args.min_encodings}")

    available = {_encoding_key(encoding) for encoding in encodings}
    for item in args.required_encoding:
        depth_mode, sem_mode = item.split(",", 1)
        if (depth_mode, sem_mode) not in available:
            failures.append(f"required encoding missing: {depth_mode},{sem_mode}")

    collapsed = report.get("collapsed_encodings") or []
    if collapsed and not args.allow_collapse:
        labels = ", ".join(
            f"{item.get('depth_mode')}/{item.get('semantic_mode')}->{item.get('action_name')}"
            for item in collapsed
        )
        failures.append("collapsed encodings present: " + labels)

    for encoding in encodings:
        depth_mode, sem_mode = _encoding_key(encoding)
        unique = encoding.get("unique_actions") or []
        if encoding.get("collapsed") and not args.allow_collapse:
            failures.append(f"encoding {depth_mode},{sem_mode} collapsed")
        if len(unique) < args.min_unique_actions and not args.allow_collapse:
            failures.append(
                f"encoding {depth_mode},{sem_mode} produced {len(unique)} unique actions; "
                f"need {args.min_unique_actions}"
            )
        expected_unique_names = [
            expected_names[action] for action in unique
            if isinstance(action, int) and 0 <= action < len(expected_names)
        ]
        if encoding.get("unique_action_names") != expected_unique_names:
            failures.append(f"encoding {depth_mode},{sem_mode} unique action names mismatch")
        failures.extend(
            f"encoding {depth_mode},{sem_mode}: {failure}"
            for failure in _row_failures(encoding.get("rows") or [], expected_names, args.require_probs)
        )

    return failures


def main():
    parser = argparse.ArgumentParser(description="Accept DRL diagnosis JSON evidence")
    parser.add_argument("report", help="JSON written by diagnose_drl_policy.py --out-json")
    parser.add_argument("--action-lateral-sign", type=int, choices=(-1, 1), default=-1)
    parser.add_argument("--required-encoding", action="append",
                        default=["meters_div255,gray_unit"],
                        help="Required depth_mode,semantic_mode pair")
    parser.add_argument("--min-items", type=int, default=1)
    parser.add_argument("--min-encodings", type=int, default=1)
    parser.add_argument("--min-unique-actions", type=int, default=2)
    parser.add_argument("--require-live-frame", action="store_true")
    parser.add_argument("--require-probs", action="store_true")
    parser.add_argument("--allow-collapse", action="store_true")
    args = parser.parse_args()

    report = _load(Path(args.report))
    encodings = report.get("encodings") or []
    print("DRL diagnosis acceptance summary")
    print(f"  input_set: {report.get('input_set')}")
    print(f"  items: {len(report.get('items') or [])}")
    print(f"  action_frame/sign: {report.get('action_frame')}/{report.get('action_lateral_sign')}")
    print(f"  encodings: {[','.join(_encoding_key(item)) for item in encodings]}")
    print(f"  collapsed_encodings: {len(report.get('collapsed_encodings') or [])}")

    failures = accept_report(report, args)
    if failures:
        for failure in failures:
            print(f"[FAIL] {failure}")
        return 1
    print("[OK] DRL diagnosis evidence accepted")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
