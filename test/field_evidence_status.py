#!/usr/bin/env python3
"""Summarize field evidence files expected before real-flight acceptance."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional


ROOT = Path(__file__).resolve().parent


@dataclass
class Evidence:
    key: str
    description: str
    path: Path
    pattern: Optional[str]
    required: bool
    accept_command: str
    produce_hint: str
    validator: Optional[str] = None

    def matches(self):
        base = self.path if self.path.is_absolute() else ROOT / self.path
        if self.pattern is None:
            return [base] if base.exists() else []
        if not base.exists():
            return []
        return sorted(base.glob(self.pattern))

    def valid_matches(self, args=None):
        matches = self.matches()
        if self.validator is None:
            return matches, None
        base = self.path if self.path.is_absolute() else ROOT / self.path
        if self.validator.startswith("dir:"):
            if not matches:
                return matches, None
            ok, reason = _validate_match(base, self.validator, args)
            return (matches if ok else []), (None if ok else reason)
        valid = []
        errors = []
        for path in matches:
            ok, reason = _validate_match(path, self.validator, args)
            if ok:
                valid.append(path)
            else:
                errors.append(f"{_fmt_path(path)}: {reason}")
        return valid, "; ".join(errors) if errors else None


def _evidence_items(args):
    logs = Path(args.log_dir)
    frames = Path(args.frame_dir)
    items = []
    validate = bool(getattr(args, "validate_artifacts", False))
    module_budget_args = (
        "--max-halss-p95-ms <70> --max-depth-p95-ms <15> "
        "--max-completion-p95-ms <45> --max-rl-p95-ms <30> "
        "--max-sync-p95-ms <100> --max-yaw-transform-error <0.15>"
    )
    if not args.skip_orin_env:
        items.append(Evidence(
            "orin_env",
            "Strict Orin runtime environment Markdown report",
            logs / "orin_env.md",
            None,
            True,
            "python accept_orin_env_light.py experiments/logs/orin_env.md",
            "Run check_orin_env.py on the Jetson Orin after sourcing ROS/Python environments.",
            validator="strict_orin_env",
        ))
    items.extend([
        Evidence(
            "gis_prior",
            "GIS nine-grid global safe-area JSON",
            Path(args.gis_prior),
            None if args.gis_prior_file else "global_prior_*.json",
            True,
            f"python accept_gis_prior_light.py {args.gis_prior} --bounds <lon_left,lat_bottom,lon_right,lat_top>",
            "Run the GIS prior step with config.global_prior.save_dir enabled; "
            "the JSON must include source_image_path, image_size_px, bounds, "
            "and source_sem_mask_path or segmentation_source.",
            validator="gis_prior" if validate else None,
        ),
        Evidence(
            "ros_livox_lidar",
            "Livox lidar topic-rate log",
            logs / "hz_livox_lidar.log",
            None,
            True,
            "python accept_ros_topics_light.py /livox/lidar=experiments/logs/hz_livox_lidar.log ... --require-all",
            "timeout 8 ros2 topic hz /livox/lidar 2>&1 | tee experiments/logs/hz_livox_lidar.log",
            validator="ros_topic:/livox/lidar" if validate else None,
        ),
        Evidence(
            "ros_livox_imu",
            "Livox IMU topic-rate log",
            logs / "hz_livox_imu.log",
            None,
            True,
            "python accept_ros_topics_light.py /livox/imu=experiments/logs/hz_livox_imu.log ... --require-all",
            "timeout 8 ros2 topic hz /livox/imu 2>&1 | tee experiments/logs/hz_livox_imu.log",
            validator="ros_topic:/livox/imu" if validate else None,
        ),
        Evidence(
            "ros_cloud_registered",
            "FAST-LIO deskewed cloud topic-rate log",
            logs / "hz_cloud_registered.log",
            None,
            True,
            "python accept_ros_topics_light.py /cloud_registered=experiments/logs/hz_cloud_registered.log ... --require-all",
            "timeout 8 ros2 topic hz /cloud_registered 2>&1 | tee experiments/logs/hz_cloud_registered.log",
            validator="ros_topic:/cloud_registered" if validate else None,
        ),
        Evidence(
            "ros_odometry",
            "FAST-LIO odometry topic-rate log",
            logs / "hz_odometry.log",
            None,
            True,
            "python accept_ros_topics_light.py /Odometry=experiments/logs/hz_odometry.log ... --require-all",
            "timeout 8 ros2 topic hz /Odometry 2>&1 | tee experiments/logs/hz_odometry.log",
            validator="ros_topic:/Odometry" if validate else None,
        ),
        Evidence(
            "depth_projection_cuda",
            "Orin CUDA depth projection parity log",
            logs / "depth_projection_cuda.log",
            None,
            True,
            "python accept_depth_projection_cuda_light.py experiments/logs/depth_projection_cuda.log",
            "python test_depth_projection_cuda.py 2>&1 | tee experiments/logs/depth_projection_cuda.log",
            validator="depth_projection_cuda" if validate else None,
        ),
        Evidence(
            "nocontrol_log",
            "120s Mid360/FAST-LIO no-control pipeline log",
            logs / "nocontrol.log",
            None,
            True,
            "python accept_nocontrol_log.py experiments/logs/nocontrol.log "
            f"--expected-yaw-rate <rate> --max-total-p95-ms 100 {module_budget_args} "
            "--require-action-probs && python diagnose_nocontrol_action_log.py "
            "experiments/logs/nocontrol.log --expected-yaw-rate <rate> --fail-on-issues",
            "python test_live_nocontrol.py --depth-output-scale <scale> "
            "--yaw-rate-rad-s <rate> --save-raw-arrays --save-frames "
            "--require-depth-completion --require-rl-model --require-yaw-rate "
            "--duration-sec 120 "
            "2>&1 | tee experiments/logs/nocontrol.log",
            validator="nocontrol_log" if validate else None,
        ),
        Evidence(
            "raw_arrays",
            "Saved no-control raw calibration arrays",
            frames,
            "*_calib_frame.npz",
            True,
            "python inspect_nocontrol_artifacts_light.py experiments/frames --require-frames "
            "--max-sync-ms <100> --max-yaw-transform-error <0.15>",
            "Run test_live_nocontrol.py with --save-raw-arrays --save-frames --save-dir experiments/frames.",
            validator="dir:nocontrol_artifacts" if validate else None,
        ),
        Evidence(
            "binary_semantic_frames",
            "Saved HALSS binary semantic PNG frames",
            frames,
            "*_binary_semantic.png",
            True,
            "python compare_saved_binary_semantic_light.py --frame-dir experiments/frames --grayscale",
            "Run test_live_nocontrol.py with --save-frames.",
            validator="dir:binary_semantic" if validate else None,
        ),
        Evidence(
            "depth_frames",
            "Saved depth PNG frames",
            frames,
            "*_depth.png",
            True,
            "python inspect_nocontrol_artifacts_light.py experiments/frames --require-frames "
            "--max-sync-ms <100> --max-yaw-transform-error <0.15>",
            "Run test_live_nocontrol.py with --save-frames.",
            validator="dir:nocontrol_frames" if validate else None,
        ),
        Evidence(
            "drl_diagnosis",
            "Live-frame DRL policy diagnosis JSON",
            logs / "drl_live_frame.json",
            None,
            True,
            "python accept_drl_diagnosis_light.py experiments/logs/drl_live_frame.json --require-live-frame --require-probs --min-items 8",
            "python diagnose_drl_policy.py --scan-modes --frame-glob 'experiments/frames/*_calib_frame.npz' --out-json experiments/logs/drl_live_frame.json --fail-on-collapse",
            validator="drl_diagnosis" if validate else None,
        ),
        Evidence(
            "sparsenet_scale",
            "SparseNet output_scale calibration JSON",
            logs / "sparsenet_scale.json",
            None,
            True,
            "python accept_sparsenet_calibration_light.py experiments/logs/sparsenet_scale.json --require-improvement",
            "python calibrate_sparsenet_scale.py --input experiments/frames/000000_calib_frame.npz --target-depth-m <measured_height> --out-json experiments/logs/sparsenet_scale.json",
            validator="sparsenet_scale" if validate else None,
        ),
        Evidence(
            "pipeline_log",
            "Closed-loop pipeline flight log",
            logs / "pipeline.log",
            None,
            True,
            "python accept_flight_loop_log.py experiments/logs/pipeline.log "
            "--require-global-guidance --expected-yaw-rate <rate> --max-total-p95-ms 100 "
            f"{module_budget_args} --max-action-run 60 --require-action-probs",
            "python pipeline.py ... --safe-point-source gis --depth-output-scale <scale> --yaw-rate-rad-s <rate> 2>&1 | tee experiments/logs/pipeline.log",
            validator="pipeline_log" if validate else None,
        ),
        Evidence(
            "acceptance_report",
            "Unified acceptance report",
            logs,
            "acceptance_light_*.json",
            False,
            "Inspect the latest experiments/logs/acceptance_light_*.md/json",
            "python run_acceptance_light.py --gis-prior <dir_or_json> "
            "--nocontrol-log experiments/logs/nocontrol.log --frame-dir experiments/frames "
            "--require-frames --drl-diagnosis-json experiments/logs/drl_live_frame.json "
            "--drl-diagnosis-live --sparsenet-calibration-json experiments/logs/sparsenet_scale.json "
            "--depth-projection-cuda-log experiments/logs/depth_projection_cuda.log "
            "--flight-log experiments/logs/pipeline.log --require-global-guidance "
            f"--expected-yaw-rate <rate> {module_budget_args} --max-flight-action-run 60 "
            "--require-action-probs",
        ),
        Evidence(
            "requirement_audit",
            "Original requirement audit report",
            logs / "requirement_audit.md",
            None,
            False,
            "python audit_experiment_requirements_light.py --strict-field --expected-yaw-rate <rate> --gis-bounds <bounds>",
            "python audit_experiment_requirements_light.py --strict-local --out-md experiments/logs/requirement_audit.md",
        ),
    ])
    return items


def _fmt_path(path):
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _short_failure_output(text: str, limit: int = 700) -> str:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    fail_lines = [line for line in lines if line.startswith("[FAIL]")]
    chosen = fail_lines[:5] if fail_lines else lines[-8:]
    summary = "; ".join(chosen)
    return summary[:limit] + ("..." if len(summary) > limit else "")


def _load_gis_best_center(path_value):
    if not path_value:
        return None
    path = Path(path_value)
    if path.is_dir():
        files = sorted(path.glob("global_prior_*.json"))
        if not files:
            return None
        path = files[-1]
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    gps = data.get("best_center_gps")
    if not (isinstance(gps, list) and len(gps) == 2):
        return None
    return f"{float(gps[0]):.8f},{float(gps[1]):.8f}"


def _run_validator(command):
    proc = subprocess.run(
        [str(part) for part in command],
        cwd=str(ROOT),
        text=True,
        capture_output=True,
    )
    if proc.returncode == 0:
        return True, "ok"
    text = (proc.stdout or "") + (proc.stderr or "")
    return False, "$ " + " ".join(str(part) for part in command) + " :: " + _short_failure_output(text)


def _run_validators(commands):
    failures = []
    for command in commands:
        ok, reason = _run_validator(command)
        if not ok:
            failures.append(reason)
    if failures:
        return False, " || ".join(failures)
    return True, "ok"


def _validator_command(path: Path, validator: str, args):
    py = sys.executable
    if validator == "gis_prior":
        cmd = [py, ROOT / "accept_gis_prior_light.py", path]
        if getattr(args, "gis_bounds", None):
            cmd.extend(["--bounds", args.gis_bounds])
        return cmd
    if validator.startswith("ros_topic:"):
        topic = validator.split(":", 1)[1]
        return [py, ROOT / "accept_ros_topics_light.py", f"{topic}={path}"]
    if validator == "nocontrol_log":
        cmd = [
            py, ROOT / "accept_nocontrol_log.py", path,
            "--min-samples", args.min_nocontrol_samples,
            "--max-total-p95-ms", args.max_nocontrol_p95_ms,
            "--max-halss-p95-ms", args.max_halss_p95_ms,
            "--max-depth-p95-ms", args.max_depth_p95_ms,
            "--max-completion-p95-ms", args.max_completion_p95_ms,
            "--max-rl-p95-ms", args.max_rl_p95_ms,
            "--max-sync-p95-ms", args.max_sync_p95_ms,
            "--max-yaw-transform-error", args.max_yaw_transform_error,
            "--require-action-probs",
        ]
        if getattr(args, "expected_yaw_rate", None):
            cmd.extend(["--expected-yaw-rate", args.expected_yaw_rate])
        diag_cmd = [
            py, ROOT / "diagnose_nocontrol_action_log.py", path,
            "--max-action-run", args.max_flight_action_run,
            "--fail-on-issues",
        ]
        if getattr(args, "expected_yaw_rate", None):
            diag_cmd.extend(["--expected-yaw-rate", args.expected_yaw_rate])
        return [cmd, diag_cmd]
    if validator == "dir:nocontrol_artifacts":
        return [
            py, ROOT / "inspect_nocontrol_artifacts_light.py", path,
            "--min-raw", "1",
            "--max-sync-ms", args.max_sync_p95_ms,
            "--max-yaw-transform-error", args.max_yaw_transform_error,
        ]
    if validator == "dir:nocontrol_frames":
        return [
            py, ROOT / "inspect_nocontrol_artifacts_light.py", path,
            "--min-raw", "1", "--require-frames",
            "--max-sync-ms", args.max_sync_p95_ms,
            "--max-yaw-transform-error", args.max_yaw_transform_error,
        ]
    if validator == "dir:binary_semantic":
        return [py, ROOT / "compare_saved_binary_semantic_light.py", "--frame-dir", path, "--grayscale"]
    if validator == "drl_diagnosis":
        return [
            py, ROOT / "accept_drl_diagnosis_light.py", path,
            "--require-live-frame", "--require-probs", "--min-items", "8",
        ]
    if validator == "sparsenet_scale":
        return [py, ROOT / "accept_sparsenet_calibration_light.py", path, "--require-improvement"]
    if validator == "depth_projection_cuda":
        return [py, ROOT / "accept_depth_projection_cuda_light.py", path]
    if validator == "pipeline_log":
        cmd = [
            py, ROOT / "accept_flight_loop_log.py", path,
            "--min-drl-frames", args.min_flight_frames,
            "--max-total-p95-ms", args.max_flight_p95_ms,
            "--max-halss-p95-ms", args.max_halss_p95_ms,
            "--max-depth-p95-ms", args.max_depth_p95_ms,
            "--max-completion-p95-ms", args.max_completion_p95_ms,
            "--max-rl-p95-ms", args.max_rl_p95_ms,
            "--max-sync-p95-ms", args.max_sync_p95_ms,
            "--max-yaw-transform-error", args.max_yaw_transform_error,
            "--max-action-run", args.max_flight_action_run,
            "--require-global-guidance",
            "--require-action-probs",
        ]
        if getattr(args, "expected_yaw_rate", None):
            cmd.extend(["--expected-yaw-rate", args.expected_yaw_rate])
        return cmd
    return None


def _validate_match(path: Path, validator: str, args=None):
    if validator == "strict_orin_env":
        try:
            from accept_orin_env_light import validate
            failures, _ = validate(path)
        except Exception as exc:
            return False, f"Orin env validation failed: {exc}"
        if failures:
            return False, "; ".join(failures)
        return True, "ok"
    command = _validator_command(path, validator, args)
    if command is not None:
        if command and isinstance(command[0], list):
            return _run_validators(command)
        return _run_validator(command)
    return False, f"unknown validator={validator}"


def _print_item(item, matches, show_hints):
    status = "PASS" if matches else ("MISS" if item.required else "INFO")
    count = len(matches)
    suffix = f" ({count})" if item.pattern else ""
    print(f"[{status}] {item.key}{suffix}: {item.description}")
    if matches:
        latest = matches[-1]
        print(f"       latest: {_fmt_path(latest)}")
    elif show_hints:
        print(f"       produce: {item.produce_hint}")
        print(f"       accept:  {item.accept_command}")


def _collect_status(items, args=None):
    rows = []
    missing = []
    for item in items:
        matches, error = item.valid_matches(args)
        if item.required and not matches:
            missing.append(item.key)
        rows.append((item, matches, error))
    return rows, missing


def _write_markdown(path: Path, rows, missing):
    path = path if path.is_absolute() else ROOT / path
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Field Evidence Status",
        "",
        f"- generated_at: `{datetime.now().isoformat(timespec='seconds')}`",
        f"- required_missing: `{len(missing)}`",
        f"- missing_keys: `{', '.join(missing) if missing else 'none'}`",
        "",
        "| Key | Status | Count | Latest | Produce | Accept |",
        "| --- | --- | ---: | --- | --- | --- |",
    ]
    for item, matches, error in rows:
        status = "PASS" if matches else ("MISS" if item.required else "INFO")
        latest = _fmt_path(matches[-1]) if matches else ""
        lines.append(
            "| {key} | {status} | {count} | {latest} | `{produce}` | `{accept}` |".format(
                key=item.key,
                status=status,
                count=len(matches),
                latest=latest if latest else (error or ""),
                produce=item.produce_hint.replace("`", "'"),
                accept=item.accept_command.replace("`", "'"),
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main():
    parser = argparse.ArgumentParser(description="Summarize expected OrinLanding field evidence files")
    parser.add_argument("--log-dir", default="experiments/logs")
    parser.add_argument("--frame-dir", default="experiments/frames")
    parser.add_argument("--gis-prior", default="experiments/logs",
                        help="GIS prior JSON file or directory containing global_prior_*.json")
    parser.add_argument("--gis-prior-file", action="store_true",
                        help="Treat --gis-prior as a single JSON file instead of a directory")
    parser.add_argument("--strict", action="store_true",
                        help="Exit nonzero when any required evidence is missing")
    parser.add_argument("--no-hints", action="store_true")
    parser.add_argument("--out-md", default=None,
                        help="Optional Markdown report path for the evidence status")
    parser.add_argument("--skip-orin-env", action="store_true",
                        help="Do not require experiments/logs/orin_env.md (bench/debug only)")
    parser.add_argument("--validate-artifacts", action="store_true",
                        help="Run acceptance validators for present evidence instead of checking file presence only")
    parser.add_argument("--expected-yaw-rate", default=None,
                        help="Expected yaw-fault rate passed to no-control and closed-loop log validators")
    parser.add_argument("--gis-bounds", default=None,
                        help="lon_left,lat_bottom,lon_right,lat_top passed to GIS prior validator")
    parser.add_argument("--min-nocontrol-samples", default="120")
    parser.add_argument("--max-nocontrol-p95-ms", default="100")
    parser.add_argument("--min-flight-frames", default="20")
    parser.add_argument("--max-flight-p95-ms", default="100")
    parser.add_argument("--max-flight-action-run", default="60")
    parser.add_argument("--expected-safe-point", default=None,
                        help="Expected closed-loop safe point as lat,lon; defaults to GIS prior best_center_gps")
    parser.add_argument("--safe-point-tolerance-m", default="2.0")
    parser.add_argument("--max-halss-p95-ms", default="70")
    parser.add_argument("--max-depth-p95-ms", default="15")
    parser.add_argument("--max-completion-p95-ms", default="45")
    parser.add_argument("--max-rl-p95-ms", default="30")
    parser.add_argument("--max-sync-p95-ms", default="100")
    parser.add_argument("--max-yaw-transform-error", default="0.15")
    args = parser.parse_args()

    items = _evidence_items(args)
    rows, missing = _collect_status(items, args)
    print("Field evidence status")
    for item, matches, error in rows:
        _print_item(item, matches, show_hints=not args.no_hints)
        if error:
            print(f"       invalid: {error}")
    print("")
    print(f"required_missing: {len(missing)}")
    if missing:
        print("missing_keys: " + ", ".join(missing))
    if args.out_md:
        out_path = _write_markdown(Path(args.out_md), rows, missing)
        print(f"markdown_report: {_fmt_path(out_path)}")
    if args.strict and missing:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
