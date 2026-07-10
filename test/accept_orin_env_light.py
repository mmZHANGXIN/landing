#!/usr/bin/env python3
"""Validate check_orin_env.py Markdown evidence."""

from __future__ import annotations

import argparse
import re
from pathlib import Path


REQUIRED_PASS_KEYS = (
    "jetpack",
    "nvcc",
    "ros2_cli",
    "ros_setup",
    "rclpy",
    "torch_cuda",
    "opencv",
    "numpy",
    "yaml",
    "stable_baselines3",
    "mavsdk",
)


def parse_status_rows(text: str):
    rows = {}
    for line in text.splitlines():
        if not line.startswith("| "):
            continue
        parts = [part.strip() for part in line.strip().strip("|").split("|")]
        if len(parts) < 5 or parts[0] in ("Key", "---"):
            continue
        key, required, status, detail, fix = parts[:5]
        rows[key] = {
            "required": required == "True",
            "status": status,
            "detail": detail,
            "fix": fix,
        }
    return rows


def validate(path: Path):
    text = path.read_text(encoding="utf-8", errors="replace")
    failures = []
    if "# Orin Environment Check" not in text:
        failures.append("not an Orin environment Markdown report")
    match = re.search(r"required_failures:\s*`(?P<count>\d+)`", text)
    if not match:
        failures.append("missing required_failures field")
    elif int(match.group("count")) != 0:
        failures.append(f"required_failures={match.group('count')} expected=0")

    rows = parse_status_rows(text)
    for key in REQUIRED_PASS_KEYS:
        row = rows.get(key)
        if row is None:
            failures.append(f"missing row: {key}")
            continue
        if not row["required"]:
            failures.append(f"{key} was not required; rerun check_orin_env.py with --strict --require-jetson")
        if row["status"] != "PASS":
            failures.append(f"{key} status={row['status']} expected=PASS")
    return failures, rows


def main():
    parser = argparse.ArgumentParser(description="Accept strict Orin environment Markdown evidence")
    parser.add_argument("report", help="experiments/logs/orin_env.md")
    args = parser.parse_args()

    path = Path(args.report)
    if not path.is_file():
        print(f"[FAIL] missing report: {path}")
        return 1

    failures, rows = validate(path)
    print("Orin environment acceptance summary")
    print(f"  report: {path}")
    print(f"  rows: {sorted(rows)}")
    if failures:
        for failure in failures:
            print(f"[FAIL] {failure}")
        return 1
    print("[OK] strict Orin environment evidence accepted")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
