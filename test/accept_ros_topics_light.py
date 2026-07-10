#!/usr/bin/env python3
"""Lightweight acceptance for Mid360/FAST-LIO ROS topic evidence logs.

The script parses text captured from ``ros2 topic hz`` and checks that the
required Mid360 and FAST-LIO topics publish at plausible rates. It deliberately
uses only the Python standard library so logs can be evaluated offboard too.

Recommended capture on Orin:

  ros2 topic hz /livox/lidar 2>&1 | tee experiments/logs/hz_livox_lidar.log
  ros2 topic hz /livox/imu 2>&1 | tee experiments/logs/hz_livox_imu.log
  ros2 topic hz /cloud_registered 2>&1 | tee experiments/logs/hz_cloud_registered.log
  ros2 topic hz /Odometry 2>&1 | tee experiments/logs/hz_odometry.log
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path


TOPIC_DEFAULTS = {
    "/livox/lidar": {"min_hz": 5.0, "max_hz": 80.0, "min_samples": 5},
    "/livox/imu": {"min_hz": 100.0, "max_hz": 500.0, "min_samples": 20},
    "/cloud_registered": {"min_hz": 5.0, "max_hz": 40.0, "min_samples": 5},
    "/Odometry": {"min_hz": 5.0, "max_hz": 80.0, "min_samples": 5},
}

RATE_RE = re.compile(r"average rate:\s*(?P<hz>[0-9.]+)")
MIN_RE = re.compile(r"min:\s*(?P<min>[0-9.]+)s")
MAX_RE = re.compile(r"max:\s*(?P<max>[0-9.]+)s")
STDDEV_RE = re.compile(r"std dev:\s*(?P<std>[0-9.]+)s")
WINDOW_RE = re.compile(r"window:\s*(?P<window>\d+)")


def parse_topic_hz_text(text: str):
    rates = []
    windows = []
    min_periods = []
    max_periods = []
    stddevs = []
    for line in text.splitlines():
        rate_match = RATE_RE.search(line)
        if rate_match:
            rates.append(float(rate_match.group("hz")))
        window_match = WINDOW_RE.search(line)
        if window_match:
            windows.append(int(window_match.group("window")))
        min_match = MIN_RE.search(line)
        if min_match:
            min_periods.append(float(min_match.group("min")))
        max_match = MAX_RE.search(line)
        if max_match:
            max_periods.append(float(max_match.group("max")))
        std_match = STDDEV_RE.search(line)
        if std_match:
            stddevs.append(float(std_match.group("std")))
    errors = [
        line.strip()
        for line in text.splitlines()
        if "error" in line.lower() or "exception" in line.lower() or "not published" in line.lower()
    ]
    return {
        "rates": rates,
        "last_rate": rates[-1] if rates else None,
        "windows": windows,
        "last_window": windows[-1] if windows else 0,
        "min_period": min_periods[-1] if min_periods else None,
        "max_period": max_periods[-1] if max_periods else None,
        "stddev": stddevs[-1] if stddevs else None,
        "errors": errors,
    }


def _topic_from_path(path: Path):
    name = path.name.lower()
    if "livox" in name and "imu" in name:
        return "/livox/imu"
    if "livox" in name and "lidar" in name:
        return "/livox/lidar"
    if "cloud_registered" in name or "cloud" in name:
        return "/cloud_registered"
    if "odometry" in name or "odom" in name:
        return "/Odometry"
    return None


def _parse_topic_arg(value: str):
    if "=" in value:
        topic, path = value.split("=", 1)
        return topic.strip(), Path(path)
    path = Path(value)
    topic = _topic_from_path(path)
    if topic is None:
        raise ValueError(
            f"Cannot infer topic from {value!r}; pass TOPIC=PATH, e.g. /Odometry=hz_odom.log"
        )
    return topic, path


def evaluate_topic(topic: str, path: Path, thresholds: dict):
    text = path.read_text(encoding="utf-8", errors="replace")
    parsed = parse_topic_hz_text(text)
    failures = []
    rate = parsed["last_rate"]
    window = parsed["last_window"]

    if not parsed["rates"]:
        failures.append(f"{topic}: no 'average rate' samples parsed")
    elif rate < thresholds["min_hz"]:
        failures.append(f"{topic}: average rate {rate:.2f}Hz < {thresholds['min_hz']:.2f}Hz")
    elif rate > thresholds["max_hz"]:
        failures.append(f"{topic}: average rate {rate:.2f}Hz > {thresholds['max_hz']:.2f}Hz")

    if window < thresholds["min_samples"]:
        failures.append(f"{topic}: window {window} < {thresholds['min_samples']} samples")

    if parsed["errors"]:
        failures.append(f"{topic}: log contains errors: {parsed['errors'][:3]}")

    return {
        "topic": topic,
        "path": str(path),
        "thresholds": thresholds,
        "parsed": parsed,
        "passed": not failures,
        "failures": failures,
    }


def main():
    parser = argparse.ArgumentParser(description="Accept saved ros2 topic hz logs")
    parser.add_argument(
        "logs",
        nargs="+",
        help="Topic logs as TOPIC=PATH or inferable paths such as hz_cloud_registered.log",
    )
    parser.add_argument("--require-all", action="store_true",
                        help="Require all default Mid360/FAST-LIO topics to be present")
    parser.add_argument("--min-lidar-hz", type=float, default=TOPIC_DEFAULTS["/livox/lidar"]["min_hz"])
    parser.add_argument("--min-imu-hz", type=float, default=TOPIC_DEFAULTS["/livox/imu"]["min_hz"])
    parser.add_argument("--min-cloud-hz", type=float, default=TOPIC_DEFAULTS["/cloud_registered"]["min_hz"])
    parser.add_argument("--min-odom-hz", type=float, default=TOPIC_DEFAULTS["/Odometry"]["min_hz"])
    parser.add_argument("--max-jitter-s", type=float, default=None,
                        help="Optional maximum accepted std dev from ros2 topic hz")
    args = parser.parse_args()

    thresholds_by_topic = {topic: dict(values) for topic, values in TOPIC_DEFAULTS.items()}
    thresholds_by_topic["/livox/lidar"]["min_hz"] = args.min_lidar_hz
    thresholds_by_topic["/livox/imu"]["min_hz"] = args.min_imu_hz
    thresholds_by_topic["/cloud_registered"]["min_hz"] = args.min_cloud_hz
    thresholds_by_topic["/Odometry"]["min_hz"] = args.min_odom_hz

    results = []
    failures = []
    seen_topics = set()
    for value in args.logs:
        try:
            topic, path = _parse_topic_arg(value)
        except ValueError as exc:
            failures.append(str(exc))
            continue
        if topic not in thresholds_by_topic:
            failures.append(f"Unsupported topic {topic!r}; expected one of {sorted(thresholds_by_topic)}")
            continue
        result = evaluate_topic(topic, path, thresholds_by_topic[topic])
        stddev = result["parsed"]["stddev"]
        if args.max_jitter_s is not None and stddev is not None and stddev > args.max_jitter_s:
            result["failures"].append(
                f"{topic}: std dev {stddev:.4f}s > {args.max_jitter_s:.4f}s"
            )
            result["passed"] = False
        results.append(result)
        seen_topics.add(topic)
        failures.extend(result["failures"])

    if args.require_all:
        missing = [topic for topic in TOPIC_DEFAULTS if topic not in seen_topics]
        if missing:
            failures.append("missing required topic logs: " + ", ".join(missing))

    print("ROS topic acceptance summary")
    for result in results:
        parsed = result["parsed"]
        status = "OK" if result["passed"] else "FAIL"
        rate = parsed["last_rate"]
        rate_text = "n/a" if rate is None else f"{rate:.2f}Hz"
        stddev = parsed["stddev"]
        std_text = "n/a" if stddev is None else f"{stddev:.4f}s"
        print(
            f"  [{status}] {result['topic']}: rate={rate_text} "
            f"window={parsed['last_window']} std={std_text} file={result['path']}"
        )

    if failures:
        for failure in failures:
            print(f"[FAIL] {failure}")
        return 1
    print("[OK] ROS topic evidence accepted")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
