#!/usr/bin/env python3
"""Analyze Orin landing timing logs.

Parses lines emitted by ``test_live_nocontrol.py`` and ``pipeline.py`` such as:

  [0146] act=3(W) H=67ms D=6ms C=32ms RL=24ms total=129ms

The script prints P50/P95/max latency and fails if latency exceeds configured
acceptance thresholds.
"""

import argparse
import re
import sys
from pathlib import Path


TIMING_RE = re.compile(
    r"\[(?P<seq>\d+)\]\s+act=(?P<action>\d+)\([^)]+\)\s+"
    r"H=(?P<H>[0-9.]+)ms\s+D=(?P<D>[0-9.]+)ms\s+"
    r"C=(?P<C>[0-9.]+)ms\s+RL=(?P<RL>[0-9.]+)ms\s+"
    r"total=(?P<total>[0-9.]+)ms"
)


def _percentile(values, q):
    if not values:
        return float("nan")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * q / 100.0
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    frac = pos - lo
    return ordered[lo] * (1.0 - frac) + ordered[hi] * frac


def parse_log(path: Path):
    samples = []
    slow_dropped = 0
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if "SLOW FRAME DROPPED" in line:
            slow_dropped += 1
        match = TIMING_RE.search(line)
        if not match:
            continue
        item = {"seq": int(match.group("seq")), "action": int(match.group("action"))}
        for key in ("H", "D", "C", "RL", "total"):
            item[key] = float(match.group(key))
        samples.append(item)
    return samples, slow_dropped


def summarize(samples, budget_ms):
    totals = [item["total"] for item in samples]
    over_budget = [value for value in totals if value > budget_ms]
    summary = {
        "count": len(samples),
        "budget_ms": budget_ms,
        "over_budget": len(over_budget),
        "over_budget_ratio": (len(over_budget) / len(samples)) if samples else 0.0,
        "hz_mean": (1000.0 / (sum(totals) / len(totals))) if totals else 0.0,
        "modules": {},
    }
    for key in ("H", "D", "C", "RL", "total"):
        values = [item[key] for item in samples]
        summary["modules"][key] = {
            "p50": _percentile(values, 50),
            "p95": _percentile(values, 95),
            "max": max(values) if values else float("nan"),
        }
    return summary


def print_summary(summary, slow_dropped):
    print("Timing summary")
    print(f"  samples: {summary['count']}")
    print(f"  budget_ms: {summary['budget_ms']:.1f}")
    print(f"  over_budget: {summary['over_budget']} ({summary['over_budget_ratio'] * 100:.1f}%)")
    print(f"  slow_frame_dropped_lines: {slow_dropped}")
    print(f"  mean_rate: {summary['hz_mean']:.2f} Hz")
    print("  module       p50_ms   p95_ms   max_ms")
    for key in ("H", "D", "C", "RL", "total"):
        stats = summary["modules"][key]
        print(f"  {key:6s} {stats['p50']:8.1f} {stats['p95']:8.1f} {stats['max']:8.1f}")


def main():
    parser = argparse.ArgumentParser(description="Analyze Orin landing timing log")
    parser.add_argument("log", help="Log file captured from no-control or flight pipeline")
    parser.add_argument("--budget-ms", type=float, default=100.0,
                        help="Frame total latency budget")
    parser.add_argument("--max-p95-ms", type=float, default=None,
                        help="Fail if total P95 exceeds this value; defaults to budget")
    parser.add_argument("--max-over-budget-ratio", type=float, default=0.05,
                        help="Fail if more than this fraction of frames exceeds budget")
    parser.add_argument("--min-samples", type=int, default=30,
                        help="Fail if fewer timing samples are parsed")
    args = parser.parse_args()

    samples, slow_dropped = parse_log(Path(args.log))
    summary = summarize(samples, args.budget_ms)
    print_summary(summary, slow_dropped)

    failures = []
    if summary["count"] < args.min_samples:
        failures.append(f"only {summary['count']} samples parsed; need {args.min_samples}")
    max_p95 = args.max_p95_ms if args.max_p95_ms is not None else args.budget_ms
    total_p95 = summary["modules"]["total"]["p95"]
    if summary["count"] and total_p95 > max_p95:
        failures.append(f"total P95 {total_p95:.1f}ms > {max_p95:.1f}ms")
    if summary["over_budget_ratio"] > args.max_over_budget_ratio:
        failures.append(
            f"over-budget ratio {summary['over_budget_ratio']:.3f} > {args.max_over_budget_ratio:.3f}"
        )

    if failures:
        for failure in failures:
            print(f"[FAIL] {failure}")
        return 1
    print("[OK] timing gates passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
