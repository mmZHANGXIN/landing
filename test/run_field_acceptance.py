#!/usr/bin/env python3
"""Run the full field evidence acceptance command with standard paths."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PYTHON = sys.executable


def _cmd(*parts):
    return [str(part) for part in parts]


def build_status_cmd(args):
    cmd = _cmd(
        PYTHON,
        ROOT / "field_evidence_status.py",
        "--strict",
        "--validate-artifacts",
        "--expected-yaw-rate", args.expected_yaw_rate,
        "--max-flight-action-run", args.max_flight_action_run,
        "--max-halss-p95-ms", args.max_halss_p95_ms,
        "--max-depth-p95-ms", args.max_depth_p95_ms,
        "--max-completion-p95-ms", args.max_completion_p95_ms,
        "--max-rl-p95-ms", args.max_rl_p95_ms,
        "--max-sync-p95-ms", args.max_sync_p95_ms,
        "--max-yaw-transform-error", args.max_yaw_transform_error,
    )
    if args.gis_prior:
        cmd.extend(["--gis-prior", args.gis_prior])
    if args.gis_prior_file:
        cmd.append("--gis-prior-file")
    if args.expected_safe_point:
        cmd.extend(["--expected-safe-point", args.expected_safe_point])
    cmd.extend(["--safe-point-tolerance-m", args.safe_point_tolerance_m])
    if args.evidence_status_md:
        cmd.extend(["--out-md", args.evidence_status_md])
    if args.gis_bounds:
        cmd.extend(["--gis-bounds", args.gis_bounds])
    if args.skip_env_check:
        cmd.append("--skip-orin-env")
    return cmd


def build_env_cmd(args):
    return _cmd(
        PYTHON,
        ROOT / "check_orin_env.py",
        "--strict",
        "--require-jetson",
        "--out-md",
        args.orin_env_md,
    )


def build_acceptance_cmd(args):
    cmd = _cmd(
        PYTHON,
        ROOT / "run_acceptance_light.py",
        "--gis-prior", args.gis_prior,
        "--nocontrol-log", args.nocontrol_log,
        "--frame-dir", args.frame_dir,
        "--require-frames",
        "--drl-diagnosis-json", args.drl_diagnosis_json,
        "--drl-diagnosis-live",
        "--sparsenet-calibration-json", args.sparsenet_calibration_json,
        "--depth-projection-cuda-log", args.depth_projection_cuda_log,
        "--flight-log", args.flight_log,
        "--max-flight-action-run", args.max_flight_action_run,
        "--max-halss-p95-ms", args.max_halss_p95_ms,
        "--max-depth-p95-ms", args.max_depth_p95_ms,
        "--max-completion-p95-ms", args.max_completion_p95_ms,
        "--max-rl-p95-ms", args.max_rl_p95_ms,
        "--max-sync-p95-ms", args.max_sync_p95_ms,
        "--max-yaw-transform-error", args.max_yaw_transform_error,
        "--require-global-guidance",
        "--expected-yaw-rate", args.expected_yaw_rate,
        "--require-action-probs",
    )
    if not args.skip_env_check:
        cmd.extend(["--orin-env-md", args.orin_env_md])
    if args.gis_bounds:
        cmd.extend(["--gis-bounds", args.gis_bounds])
    if args.expected_safe_point:
        cmd.extend(["--expected-safe-point", args.expected_safe_point])
    cmd.extend(["--safe-point-tolerance-m", args.safe_point_tolerance_m])
    if args.require_landing:
        cmd.append("--require-landing")
    if args.strict_flight_ready:
        cmd.append("--strict-flight-ready")
    for topic, path in (
        ("/livox/lidar", "experiments/logs/hz_livox_lidar.log"),
        ("/livox/imu", "experiments/logs/hz_livox_imu.log"),
        ("/cloud_registered", "experiments/logs/hz_cloud_registered.log"),
        ("/Odometry", "experiments/logs/hz_odometry.log"),
    ):
        cmd.extend(["--ros-topic-log", f"{topic}={path}"])
    cmd.append("--require-all-ros-topics")
    return cmd


def _run(name, cmd, dry_run):
    print(f"[FIELD] {name}")
    print("$ " + " ".join(cmd))
    if dry_run:
        return 0
    proc = subprocess.run(cmd, cwd=str(ROOT))
    return proc.returncode


def main():
    parser = argparse.ArgumentParser(description="Run full OrinLanding field acceptance with standard evidence paths")
    parser.add_argument("--expected-yaw-rate", required=True,
                        help="Yaw-fault rate used in no-control and closed-loop logs")
    parser.add_argument("--gis-prior", default="experiments/logs",
                        help="GIS global_prior JSON file or directory")
    parser.add_argument("--gis-prior-file", action="store_true",
                        help="Treat --gis-prior as a single JSON file")
    parser.add_argument("--gis-bounds", default=None,
                        help="lon_left,lat_bottom,lon_right,lat_top for GIS JSON acceptance")
    parser.add_argument("--expected-safe-point", default=None,
                        help="Optional expected closed-loop safe point as lat,lon; defaults to GIS prior best_center_gps")
    parser.add_argument("--safe-point-tolerance-m", default="2.0")
    parser.add_argument("--nocontrol-log", default="experiments/logs/nocontrol.log")
    parser.add_argument("--frame-dir", default="experiments/frames")
    parser.add_argument("--drl-diagnosis-json", default="experiments/logs/drl_live_frame.json")
    parser.add_argument("--sparsenet-calibration-json", default="experiments/logs/sparsenet_scale.json")
    parser.add_argument("--depth-projection-cuda-log", default="experiments/logs/depth_projection_cuda.log")
    parser.add_argument("--flight-log", default="experiments/logs/pipeline.log")
    parser.add_argument("--max-flight-action-run", default="60",
                        help="Fail field acceptance if a closed-loop DRL action repeats longer than this")
    parser.add_argument("--max-halss-p95-ms", default="70")
    parser.add_argument("--max-depth-p95-ms", default="15")
    parser.add_argument("--max-completion-p95-ms", default="45")
    parser.add_argument("--max-rl-p95-ms", default="30")
    parser.add_argument("--max-sync-p95-ms", default="100")
    parser.add_argument("--max-yaw-transform-error", default="0.15")
    parser.add_argument("--evidence-status-md", default="experiments/logs/field_evidence_status.md",
                        help="Markdown status report written by field_evidence_status.py")
    parser.add_argument("--orin-env-md", default="experiments/logs/orin_env.md",
                        help="Markdown environment report written by check_orin_env.py")
    parser.add_argument("--skip-env-check", action="store_true",
                        help="Skip strict Orin environment check (bench/debug only)")
    parser.add_argument("--strict-flight-ready", action="store_true")
    parser.add_argument("--require-landing", action="store_true")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print commands without executing them")
    args = parser.parse_args()

    env_cmd = build_env_cmd(args)
    status_cmd = build_status_cmd(args)
    acceptance_cmd = build_acceptance_cmd(args)

    if not args.skip_env_check:
        rc = _run("Orin environment", env_cmd, args.dry_run)
        if rc != 0:
            print(f"[FIELD] Orin environment failed with exit code {rc}")
            return rc
    rc = _run("Evidence status", status_cmd, args.dry_run)
    if rc != 0:
        print(f"[FIELD] Evidence status failed with exit code {rc}")
        return rc
    rc = _run("Unified field acceptance", acceptance_cmd, args.dry_run)
    if rc != 0:
        print(f"[FIELD] Unified field acceptance failed with exit code {rc}")
        return rc
    print("[FIELD] Acceptance complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
