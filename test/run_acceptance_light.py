#!/usr/bin/env python3
"""Run lightweight acceptance checks and write an evidence report.

The default mode is intended for laptops or pre-Orin staging machines. It runs
checks that do not need ROS, CUDA, OpenCV, NumPy, Torch, Mid360, or a flight
controller. The strict flight-ready preflight is included as a gate preview:
by default only the known field-configuration gaps are allowed to fail.
Pass ``--strict-flight-ready`` when the config is expected to be ready for the
real flight loop.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PYTHON = sys.executable


EXPECTED_DEFAULT_FLIGHT_READY_FAILURES = (
    "global_prior.enabled=false",
    "depth_completion.output_scale is null",
    "uav.yaw_rate_rad_s is 0.0",
)

REQUIRED_VISUALIZATION_OKS = (
    "binary semantic and depth visualization enabled",
    "binary semantic window title fixed",
)


def _cmd(*parts):
    return [str(part) for part in parts]


def _escape_table(text):
    return str(text).replace("|", "\\|").replace("\n", "<br>")


def _short_output(stdout, stderr, limit=1200):
    text = (stdout or "") + (stderr or "")
    text = text.strip()
    if len(text) <= limit:
        return text
    return text[:limit] + "...<truncated>"


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


def _run_command(name, command, evaluator):
    start = time.perf_counter()
    proc = subprocess.run(
        command,
        cwd=str(ROOT),
        text=True,
        capture_output=True,
    )
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    passed, reason = evaluator(proc)
    return {
        "name": name,
        "command": command,
        "returncode": proc.returncode,
        "elapsed_ms": round(elapsed_ms, 1),
        "passed": bool(passed),
        "reason": reason,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
    }


def _eval_zero(proc):
    if proc.returncode == 0:
        return True, "command exited 0"
    return False, f"command exited {proc.returncode}"


def _flight_ready_fail_lines(text):
    return [
        line.strip()
        for line in text.splitlines()
        if "[FAIL] flight-ready gate:" in line
    ]


def _make_eval_flight_ready(strict):
    def _eval(proc):
        text = (proc.stdout or "") + (proc.stderr or "")
        if proc.returncode == 0:
            return True, "strict flight-ready preflight passed"
        if strict:
            return False, f"strict flight-ready required, command exited {proc.returncode}"

        failures = _flight_ready_fail_lines(text)
        if not failures:
            return False, "preflight failed but no flight-ready failure lines were parsed"

        unexpected = [
            line for line in failures
            if not any(expected in line for expected in EXPECTED_DEFAULT_FLIGHT_READY_FAILURES)
        ]
        if unexpected:
            return False, "unexpected flight-ready failures: " + "; ".join(unexpected)

        missing_vis = [
            item for item in REQUIRED_VISUALIZATION_OKS
            if item not in text
        ]
        if missing_vis:
            return False, "visualization OK evidence missing: " + ", ".join(missing_vis)

        return (
            True,
            "flight-ready preview has only expected field-configuration gaps",
        )

    return _eval


def _build_checks(args):
    cfg = Path(args.config)
    policy = Path(args.policy)
    checks = [
        (
            "preflight config/assets without model load",
            _cmd(PYTHON, ROOT / "preflight_check.py", "--config", cfg, "--skip-model-load"),
            _eval_zero,
        ),
        (
            "DRL SB3 metadata",
            _cmd(PYTHON, ROOT / "inspect_drl_model.py", policy),
            _eval_zero,
        ),
        (
            "GIS nine-grid lightweight contract",
            _cmd(PYTHON, ROOT / "test_gis_nine_grid_light.py"),
            _eval_zero,
        ),
        (
            "GIS global-prior JSON acceptance self-test",
            _cmd(PYTHON, ROOT / "test_accept_gis_prior_light.py"),
            _eval_zero,
        ),
        (
            "HALSS visualization lightweight contract",
            _cmd(PYTHON, ROOT / "test_halss_visualization_light.py"),
            _eval_zero,
        ),
        (
            "visualization source contract",
            _cmd(PYTHON, ROOT / "test_visualization_contract_light.py"),
            _eval_zero,
        ),
        (
            "depth projection lightweight contract",
            _cmd(PYTHON, ROOT / "test_depth_projection_light.py"),
            _eval_zero,
        ),
        (
            "CUDA depth projection log acceptance self-test",
            _cmd(PYTHON, ROOT / "test_accept_depth_projection_cuda_light.py"),
            _eval_zero,
        ),
        (
            "GPU fail-closed runtime contract",
            _cmd(PYTHON, ROOT / "test_gpu_fail_closed_contract_light.py"),
            _eval_zero,
        ),
        (
            "SparseNet sparse convolution lightweight contract",
            _cmd(PYTHON, ROOT / "test_sparse_conv_light.py"),
            _eval_zero,
        ),
        (
            "yaw-aware action decomposition",
            _cmd(PYTHON, ROOT / "control" / "action_decomposer.py"),
            _eval_zero,
        ),
        (
            "runtime action-name labels",
            _cmd(PYTHON, ROOT / "test_action_names_light.py"),
            _eval_zero,
        ),
        (
            "field runbook source contract",
            _cmd(PYTHON, ROOT / "test_runbook_contract_light.py"),
            _eval_zero,
        ),
        (
            "field evidence status self-test",
            _cmd(PYTHON, ROOT / "test_field_evidence_status_light.py"),
            _eval_zero,
        ),
        (
            "Orin environment check self-test",
            _cmd(PYTHON, ROOT / "test_check_orin_env_light.py"),
            _eval_zero,
        ),
        (
            "Orin environment report acceptance self-test",
            _cmd(PYTHON, ROOT / "test_accept_orin_env_light.py"),
            _eval_zero,
        ),
        (
            "requirement traceability source contract",
            _cmd(PYTHON, ROOT / "test_requirement_traceability_light.py"),
            _eval_zero,
        ),
        (
            "experiment requirement audit self-test",
            _cmd(PYTHON, ROOT / "test_audit_experiment_requirements_light.py"),
            _eval_zero,
        ),
        (
            "field acceptance wrapper self-test",
            _cmd(PYTHON, ROOT / "test_run_field_acceptance_light.py"),
            _eval_zero,
        ),
        (
            "closed-loop diagnostic snapshot contract",
            _cmd(PYTHON, ROOT / "test_pipeline_diagnostic_snapshot_light.py"),
            _eval_zero,
        ),
        (
            "MAVSDK home telemetry contract",
            _cmd(PYTHON, ROOT / "test_mavsdk_home_telemetry_light.py"),
            _eval_zero,
        ),
        (
            "timing log acceptance self-test",
            _cmd(PYTHON, ROOT / "test_analyze_timing_log_light.py"),
            _eval_zero,
        ),
        (
            "no-control log acceptance self-test",
            _cmd(PYTHON, ROOT / "test_accept_nocontrol_log_light.py"),
            _eval_zero,
        ),
        (
            "no-control action diagnosis self-test",
            _cmd(PYTHON, ROOT / "test_diagnose_nocontrol_action_log_light.py"),
            _eval_zero,
        ),
        (
            "no-control runtime contract self-test",
            _cmd(PYTHON, ROOT / "test_nocontrol_runtime_contract_light.py"),
            _eval_zero,
        ),
        (
            "no-control artifact acceptance self-test",
            _cmd(PYTHON, ROOT / "test_inspect_nocontrol_artifacts_light.py"),
            _eval_zero,
        ),
        (
            "DRL diagnosis JSON acceptance self-test",
            _cmd(PYTHON, ROOT / "test_accept_drl_diagnosis_light.py"),
            _eval_zero,
        ),
        (
            "SparseNet calibration JSON acceptance self-test",
            _cmd(PYTHON, ROOT / "test_accept_sparsenet_calibration_light.py"),
            _eval_zero,
        ),
        (
            "saved binary semantic NPZ/PNG comparison self-test",
            _cmd(PYTHON, ROOT / "test_compare_saved_binary_semantic_light.py"),
            _eval_zero,
        ),
        (
            "ROS topic evidence acceptance self-test",
            _cmd(PYTHON, ROOT / "test_accept_ros_topics_light.py"),
            _eval_zero,
        ),
        (
            "flight-loop log acceptance self-test",
            _cmd(PYTHON, ROOT / "test_accept_flight_loop_log_light.py"),
            _eval_zero,
        ),
        (
            "strict flight-ready gate regression",
            _cmd(PYTHON, ROOT / "test_flight_ready_gates.py"),
            _eval_zero,
        ),
        (
            "flight-ready preflight preview",
            _cmd(PYTHON, ROOT / "preflight_check.py", "--config", cfg, "--flight-ready", "--skip-model-load"),
            _make_eval_flight_ready(args.strict_flight_ready),
        ),
        (
            "pipeline flight-ready check-only with GIS override",
            _cmd(
                PYTHON,
                ROOT / "pipeline.py",
                "--config", cfg,
                "--mode", "ros",
                "--safe-point", args.safe_point,
                "--safe-point-source", "gis",
                "--depth-output-scale", args.depth_output_scale,
                "--yaw-rate-rad-s", args.yaw_rate_rad_s,
                "--flight-ready-check-only",
            ),
            _eval_zero,
        ),
    ]

    if args.frame_dir:
        artifact_cmd = _cmd(
            PYTHON,
            ROOT / "inspect_nocontrol_artifacts_light.py",
            Path(args.frame_dir),
            "--min-raw",
            args.min_raw_artifacts,
            "--max-sync-ms",
            args.max_sync_p95_ms,
            "--max-yaw-transform-error",
            args.max_yaw_transform_error,
        )
        if args.require_frames:
            artifact_cmd.append("--require-frames")
        checks.append((
            "saved no-control artifacts",
            artifact_cmd,
            _eval_zero,
        ))
        if args.require_frames:
            checks.append((
                "saved binary semantic matches HALSS NPZ output",
                _cmd(
                    PYTHON,
                    ROOT / "compare_saved_binary_semantic_light.py",
                    "--frame-dir",
                    Path(args.frame_dir),
                    "--grayscale",
                ),
                _eval_zero,
            ))

    if args.drl_diagnosis_json:
        drl_cmd = _cmd(
            PYTHON,
            ROOT / "accept_drl_diagnosis_light.py",
            Path(args.drl_diagnosis_json),
            "--min-items",
            args.min_drl_diagnosis_items,
            "--min-encodings",
            args.min_drl_diagnosis_encodings,
            "--min-unique-actions",
            args.min_drl_unique_actions,
        )
        if args.drl_diagnosis_live:
            drl_cmd.append("--require-live-frame")
        if args.require_action_probs:
            drl_cmd.append("--require-probs")
        checks.append((
            "saved DRL diagnosis JSON evidence",
            drl_cmd,
            _eval_zero,
        ))

    if args.sparsenet_calibration_json:
        scale_cmd = _cmd(
            PYTHON,
            ROOT / "accept_sparsenet_calibration_light.py",
            Path(args.sparsenet_calibration_json),
            "--max-scaled-median-abs-m",
            args.max_depth_calib_median_abs_m,
            "--max-scaled-p90-abs-m",
            args.max_depth_calib_p90_abs_m,
        )
        if args.require_depth_calib_improvement:
            scale_cmd.append("--require-improvement")
        checks.append((
            "saved SparseNet calibration JSON evidence",
            scale_cmd,
            _eval_zero,
        ))

    if args.depth_projection_cuda_log:
        checks.append((
            "saved CUDA depth projection parity evidence",
            _cmd(PYTHON, ROOT / "accept_depth_projection_cuda_light.py", Path(args.depth_projection_cuda_log)),
            _eval_zero,
        ))

    if args.orin_env_md:
        checks.append((
            "saved strict Orin environment evidence",
            _cmd(PYTHON, ROOT / "accept_orin_env_light.py", Path(args.orin_env_md)),
            _eval_zero,
        ))

    if args.gis_prior:
        gis_cmd = _cmd(PYTHON, ROOT / "accept_gis_prior_light.py", Path(args.gis_prior))
        if args.gis_bounds:
            gis_cmd.extend(["--bounds", args.gis_bounds])
        checks.append((
            "saved GIS global-prior evidence",
            gis_cmd,
            _eval_zero,
        ))

    if args.nocontrol_log:
        timing_cmd = _cmd(
            PYTHON,
            ROOT / "analyze_timing_log.py",
            Path(args.nocontrol_log),
            "--budget-ms",
            args.max_nocontrol_p95_ms,
            "--max-p95-ms",
            args.max_nocontrol_p95_ms,
            "--min-samples",
            args.min_nocontrol_samples,
        )
        checks.append((
            "no-control timing evidence",
            timing_cmd,
            _eval_zero,
        ))

        nocontrol_cmd = _cmd(
            PYTHON,
            ROOT / "accept_nocontrol_log.py",
            Path(args.nocontrol_log),
            "--min-samples",
            args.min_nocontrol_samples,
            "--max-total-p95-ms",
            args.max_nocontrol_p95_ms,
            "--max-halss-p95-ms",
            args.max_halss_p95_ms,
            "--max-depth-p95-ms",
            args.max_depth_p95_ms,
            "--max-completion-p95-ms",
            args.max_completion_p95_ms,
            "--max-rl-p95-ms",
            args.max_rl_p95_ms,
            "--max-sync-p95-ms",
            args.max_sync_p95_ms,
            "--max-yaw-transform-error",
            args.max_yaw_transform_error,
        )
        if args.expected_yaw_rate:
            nocontrol_cmd.extend(["--expected-yaw-rate", args.expected_yaw_rate])
        if args.require_action_probs:
            nocontrol_cmd.append("--require-action-probs")
        checks.append((
            "no-control integrated log evidence",
            nocontrol_cmd,
            _eval_zero,
        ))

        nocontrol_diag_cmd = _cmd(
            PYTHON,
            ROOT / "diagnose_nocontrol_action_log.py",
            Path(args.nocontrol_log),
            "--max-action-run",
            args.max_flight_action_run,
            "--fail-on-issues",
        )
        if args.expected_yaw_rate:
            nocontrol_diag_cmd.extend(["--expected-yaw-rate", args.expected_yaw_rate])
        checks.append((
            "no-control action diagnosis evidence",
            nocontrol_diag_cmd,
            _eval_zero,
        ))

    if args.ros_topic_log:
        ros_cmd = _cmd(PYTHON, ROOT / "accept_ros_topics_light.py")
        ros_cmd.extend(args.ros_topic_log)
        if args.require_all_ros_topics:
            ros_cmd.append("--require-all")
        checks.append((
            "saved ROS topic hz evidence",
            ros_cmd,
            _eval_zero,
        ))

    if args.flight_log:
        flight_cmd = _cmd(
            PYTHON,
            ROOT / "accept_flight_loop_log.py",
            Path(args.flight_log),
            "--min-drl-frames",
            args.min_flight_frames,
            "--max-total-p95-ms",
            args.max_flight_p95_ms,
            "--max-halss-p95-ms",
            args.max_halss_p95_ms,
            "--max-depth-p95-ms",
            args.max_depth_p95_ms,
            "--max-completion-p95-ms",
            args.max_completion_p95_ms,
            "--max-rl-p95-ms",
            args.max_rl_p95_ms,
            "--max-sync-p95-ms",
            args.max_sync_p95_ms,
            "--max-yaw-transform-error",
            args.max_yaw_transform_error,
            "--max-action-run",
            args.max_flight_action_run,
        )
        if args.expected_yaw_rate:
            flight_cmd.extend(["--expected-yaw-rate", args.expected_yaw_rate])
        expected_safe_point = args.expected_safe_point or _load_gis_best_center(args.gis_prior)
        if expected_safe_point:
            flight_cmd.extend([
                "--expected-safe-point", expected_safe_point,
                "--safe-point-tolerance-m", args.safe_point_tolerance_m,
            ])
        if args.require_global_guidance:
            flight_cmd.append("--require-global-guidance")
        if args.require_landing:
            flight_cmd.append("--require-landing")
        if args.require_action_probs:
            flight_cmd.append("--require-action-probs")
        checks.append((
            "closed-loop pipeline log evidence",
            flight_cmd,
            _eval_zero,
        ))

    return checks


def _write_json(path, report):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")


def _write_markdown(path, report):
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Lightweight Acceptance Report",
        "",
        f"- generated_at: `{report['generated_at']}`",
        f"- root: `{report['root']}`",
        f"- overall: `{'PASS' if report['overall_passed'] else 'FAIL'}`",
        f"- strict_flight_ready: `{report['strict_flight_ready']}`",
        "",
        "| Check | Status | Return | ms | Reason |",
        "| --- | --- | ---: | ---: | --- |",
    ]
    for check in report["checks"]:
        lines.append(
            "| {name} | {status} | {returncode} | {elapsed_ms:.1f} | {reason} |".format(
                name=_escape_table(check["name"]),
                status="PASS" if check["passed"] else "FAIL",
                returncode=check["returncode"],
                elapsed_ms=check["elapsed_ms"],
                reason=_escape_table(check["reason"]),
            )
        )
    lines.extend(["", "## Command Output", ""])
    for check in report["checks"]:
        lines.extend([
            f"### {check['name']}",
            "",
            "```text",
            "$ " + " ".join(check["command"]),
            _short_output(check["stdout"], check["stderr"], limit=2000),
            "```",
            "",
        ])
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Run lightweight OrinLanding acceptance checks")
    parser.add_argument("--config", default=str(ROOT / "config" / "experiment_config.yaml"))
    parser.add_argument("--policy", default=str(ROOT / "weights" / "last_step_model_sb3.zip"))
    parser.add_argument("--report-dir", default=str(ROOT / "experiments" / "logs"))
    parser.add_argument("--strict-flight-ready", action="store_true",
                        help="Require preflight_check.py --flight-ready to exit 0")
    parser.add_argument("--safe-point", default="31.0,121.0",
                        help="GIS-derived safe-point used only for check-only preview")
    parser.add_argument("--depth-output-scale", default="40",
                        help="Calibrated SparseNet scale used only for check-only preview")
    parser.add_argument("--yaw-rate-rad-s", default="0.35",
                        help="Yaw-fault rate used only for check-only preview")
    parser.add_argument("--frame-dir", default=None,
                        help="Optional experiments/frames directory from a no-control run")
    parser.add_argument("--require-frames", action="store_true",
                        help="When --frame-dir is set, require saved PNG frame pairs")
    parser.add_argument("--min-raw-artifacts", default="1",
                        help="Minimum *_calib_frame.npz files when --frame-dir is set")
    parser.add_argument("--drl-diagnosis-json", default=None,
                        help="Optional diagnose_drl_policy.py --out-json report to inspect")
    parser.add_argument("--drl-diagnosis-live", action="store_true",
                        help="Require --drl-diagnosis-json to come from saved live frames")
    parser.add_argument("--min-drl-diagnosis-items", default="8")
    parser.add_argument("--min-drl-diagnosis-encodings", default="1")
    parser.add_argument("--min-drl-unique-actions", default="2")
    parser.add_argument("--sparsenet-calibration-json", default=None,
                        help="Optional calibrate_sparsenet_scale.py --out-json report to inspect")
    parser.add_argument("--depth-projection-cuda-log", default=None,
                        help="Optional test_depth_projection_cuda.py tee log to inspect")
    parser.add_argument("--orin-env-md", default=None,
                        help="Optional check_orin_env.py --out-md report to inspect")
    parser.add_argument("--max-depth-calib-median-abs-m", default="0.5")
    parser.add_argument("--max-depth-calib-p90-abs-m", default="1.5")
    parser.add_argument("--require-depth-calib-improvement", action="store_true")
    parser.add_argument("--gis-prior", default=None,
                        help="Optional global_prior_*.json file or output directory to inspect")
    parser.add_argument("--gis-bounds", default=None,
                        help="Optional lon_left,lat_bottom,lon_right,lat_top bounds for --gis-prior")
    parser.add_argument("--nocontrol-log", default=None,
                        help="Optional test_live_nocontrol.py log to inspect")
    parser.add_argument("--min-nocontrol-samples", default="120")
    parser.add_argument("--max-nocontrol-p95-ms", default="100")
    parser.add_argument("--max-halss-p95-ms", default="70")
    parser.add_argument("--max-depth-p95-ms", default="15")
    parser.add_argument("--max-completion-p95-ms", default="45")
    parser.add_argument("--max-rl-p95-ms", default="30")
    parser.add_argument("--max-sync-p95-ms", default="100")
    parser.add_argument("--max-yaw-transform-error", default="0.15")
    parser.add_argument("--ros-topic-log", action="append", default=[],
                        help="Optional TOPIC=PATH ros2 topic hz logs to inspect")
    parser.add_argument("--require-all-ros-topics", action="store_true",
                        help="When --ros-topic-log is set, require Livox lidar/imu, cloud, and odometry logs")
    parser.add_argument("--flight-log", default=None,
                        help="Optional pipeline.py --mode ros log to inspect")
    parser.add_argument("--min-flight-frames", default="20")
    parser.add_argument("--max-flight-p95-ms", default="100")
    parser.add_argument("--max-flight-action-run", default="60")
    parser.add_argument("--expected-safe-point", default=None,
                        help="Optional expected closed-loop safe point as lat,lon")
    parser.add_argument("--safe-point-tolerance-m", default="2.0")
    parser.add_argument("--expected-yaw-rate", default=None)
    parser.add_argument("--require-global-guidance", action="store_true")
    parser.add_argument("--require-landing", action="store_true")
    parser.add_argument("--require-action-probs", action="store_true")
    args = parser.parse_args()

    checks = []
    for name, command, evaluator in _build_checks(args):
        print(f"[RUN] {name}")
        result = _run_command(name, command, evaluator)
        checks.append(result)
        status = "PASS" if result["passed"] else "FAIL"
        print(f"[{status}] {name}: {result['reason']} ({result['elapsed_ms']:.1f} ms)")

    overall = all(check["passed"] for check in checks)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "root": str(ROOT),
        "strict_flight_ready": bool(args.strict_flight_ready),
        "overall_passed": bool(overall),
        "checks": checks,
    }
    report_dir = Path(args.report_dir)
    json_path = report_dir / f"acceptance_light_{stamp}.json"
    md_path = report_dir / f"acceptance_light_{stamp}.md"
    _write_json(json_path, report)
    _write_markdown(md_path, report)

    print("")
    print(f"JSON report: {json_path}")
    print(f"Markdown report: {md_path}")
    print(f"Overall: {'PASS' if overall else 'FAIL'}")
    return 0 if overall else 1


if __name__ == "__main__":
    raise SystemExit(main())
