#!/usr/bin/env python3
"""Audit original experiment requirements against current evidence.

This stdlib-only audit is intentionally conservative: it treats local source
contracts as proof only for implementation/runbook coverage, and marks hardware
or flight evidence as FIELD_REQUIRED until accepted artifacts exist.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PYTHON = sys.executable


@dataclass
class Requirement:
    key: str
    description: str
    local_needles: tuple[tuple[str, str], ...]
    field_keys: tuple[str, ...] = ()


REQUIREMENTS = (
    Requirement(
        "offline_gis_nine_grid",
        "Offline GIS segmentation and nine-grid global safe-area target",
        (
            ("EXPERIMENT_ACCEPTANCE.md", "GIS result JSON is machine-checkable"),
            ("FIELD_ATOMIC_CHECKLIST.md", "global_prior_*.json"),
            ("preprocessing/global_safety_prior.py", "best_center_gps"),
        ),
        ("gis_prior",),
    ),
    Requirement(
        "global_guidance_before_drl",
        "Position guidance reaches GIS safe area before DRL descent",
        (
            ("EXPERIMENT_ACCEPTANCE.md", "Pipeline enters GOTO_SAFE before DRL descent"),
            ("accept_flight_loop_log.py", "required global-guidance event missing"),
            ("pipeline.py", "_goto_safe_point"),
        ),
        ("pipeline_log",),
    ),
    Requirement(
        "mid360_fastlio_deskewed_cloud_pose",
        "Mid360 and FAST-LIO provide deskewed cloud plus pose/yaw",
        (
            ("EXPERIMENT_ACCEPTANCE.md", "FAST-LIO And Mid360 Data"),
            ("test_live_nocontrol.py", "/cloud_registered"),
            ("test_live_nocontrol.py", "/Odometry"),
        ),
        ("ros_livox_lidar", "ros_livox_imu", "ros_cloud_registered", "ros_odometry"),
    ),
    Requirement(
        "halss_bayesian_binary_semantic",
        "Deskewed point cloud generates HALSS Bayesian binary semantic map",
        (
            ("EXPERIMENT_ACCEPTANCE.md", "HALSS Binary Semantic"),
            ("config/experiment_config.yaml", 'halss_backend: "bayesian_unet"'),
            ("perception/halss_bayesian.py", "safety_map_vis"),
        ),
        ("nocontrol_log", "binary_semantic_frames"),
    ),
    Requirement(
        "halss_visual_identity",
        "Displayed binary semantic view is identical to HALSS visualization",
        (
            ("EXPERIMENT_ACCEPTANCE.md", "Saved binary semantic equals HALSS NPZ output"),
            ("visualization/display.py", "_render_binary_semantic"),
            ("compare_saved_binary_semantic_light.py", "binary_semantic_vis"),
        ),
        ("raw_arrays", "binary_semantic_frames"),
    ),
    Requirement(
        "perspective_depth_projection",
        "Point cloud is projected with down-looking perspective z-buffer geometry",
        (
            ("EXPERIMENT_ACCEPTANCE.md", "Perspective projection"),
            ("config/experiment_config.yaml", 'backend: "torch_cuda"'),
            ("perception/depth_projection.py", "_project_perspective_torch_cuda"),
        ),
        ("depth_projection_cuda", "nocontrol_log", "raw_arrays"),
    ),
    Requirement(
        "sparsity_invariant_depth_completion",
        "SparseNet depth completion uses inverse-depth sparse convolution and calibrated scale",
        (
            ("EXPERIMENT_ACCEPTANCE.md", "SparseNet calibration JSON is accepted"),
            ("config/experiment_config.yaml", 'input_encoding: "inverse_unit"'),
            ("perception/sparse_depth_completion.py", "output_scale"),
        ),
        ("sparsenet_scale", "raw_arrays", "depth_frames"),
    ),
    Requirement(
        "drl_policy_inference",
        "Depth and semantic maps feed DeepRL policy and terminal action output",
        (
            ("EXPERIMENT_ACCEPTANCE.md", "Live action probabilities are logged"),
            ("rl/rl_agent.py", "predict_with_info"),
            ("test_live_nocontrol.py", "act={action_id}"),
        ),
        ("drl_diagnosis", "nocontrol_log", "pipeline_log"),
    ),
    Requirement(
        "yaw_fault_action_decomposition",
        "Discrete actions are converted with current FAST-LIO yaw for yaw-fault flight",
        (
            ("EXPERIMENT_ACCEPTANCE.md", "Yaw-Fault Action Decomposition"),
            ("config/experiment_config.yaml", 'action_frame: "body"'),
            ("control/action_decomposer.py", "_body_to_ned"),
        ),
        ("nocontrol_log", "pipeline_log"),
    ),
    Requirement(
        "realtime_gpu_budget",
        "GPU runtime and module/total latency budgets are enforced",
        (
            ("EXPERIMENT_ACCEPTANCE.md", "Realtime claims are measured, not assumed"),
            ("EXPERIMENT_ACCEPTANCE.md", "CUDA projection matches the reference geometry"),
            ("accept_nocontrol_log.py", "module_p95_ms"),
            ("accept_flight_loop_log.py", "module_p95_ms"),
        ),
        ("orin_env", "depth_projection_cuda", "nocontrol_log", "pipeline_log"),
    ),
    Requirement(
        "live_visual_outputs",
        "No-control run shows binary semantic and depth windows and saves evidence frames",
        (
            ("EXPERIMENT_ACCEPTANCE.md", "`binary semantic` window opens and updates"),
            ("config/experiment_config.yaml", 'binary_semantic_window_title: "binary semantic"'),
            ("test_live_nocontrol.py", "Depth (m)"),
        ),
        ("binary_semantic_frames", "depth_frames", "raw_arrays"),
    ),
    Requirement(
        "closed_loop_field_readiness",
        "Final field acceptance proves real-flight readiness",
        (
            ("EXPERIMENT_ACCEPTANCE.md", "Controlled Flight Readiness"),
            ("field_evidence_status.py", "--validate-artifacts"),
            ("run_field_acceptance.py", "--strict-flight-ready"),
        ),
        (
            "orin_env", "gis_prior", "ros_livox_lidar", "ros_livox_imu",
            "ros_cloud_registered", "ros_odometry", "depth_projection_cuda", "nocontrol_log",
            "raw_arrays", "binary_semantic_frames", "depth_frames",
            "drl_diagnosis", "sparsenet_scale", "pipeline_log",
        ),
    ),
)


def _read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8", errors="replace")


def _local_missing(req: Requirement):
    missing = []
    for relpath, needle in req.local_needles:
        path = ROOT / relpath
        if not path.exists():
            missing.append(f"{relpath}:missing file")
            continue
        if needle not in _read(relpath):
            missing.append(f"{relpath}:missing {needle!r}")
    return missing


def _field_missing(args):
    cmd = [
        PYTHON,
        str(ROOT / "field_evidence_status.py"),
        "--no-hints",
        "--validate-artifacts",
        "--expected-yaw-rate",
        args.expected_yaw_rate,
        "--gis-bounds",
        args.gis_bounds,
        "--max-halss-p95-ms",
        args.max_halss_p95_ms,
        "--max-depth-p95-ms",
        args.max_depth_p95_ms,
        "--max-completion-p95-ms",
        args.max_completion_p95_ms,
        "--max-rl-p95-ms",
        args.max_rl_p95_ms,
    ]
    proc = subprocess.run(cmd, cwd=str(ROOT), text=True, capture_output=True)
    missing = set()
    invalid = {}
    for line in proc.stdout.splitlines():
        if line.startswith("missing_keys:"):
            keys = line.split(":", 1)[1]
            missing.update(key.strip() for key in keys.split(",") if key.strip())
        elif line.startswith("[MISS] "):
            key = line.split(":", 1)[0].split()[1]
            missing.add(key)
        elif "       invalid:" in line and missing:
            invalid[sorted(missing)[-1]] = line.strip()
    return missing, invalid, proc.stdout


def audit(args):
    field_missing, field_invalid, status_text = _field_missing(args)
    rows = []
    for req in REQUIREMENTS:
        local_missing = _local_missing(req)
        missing_fields = [key for key in req.field_keys if key in field_missing]
        invalid_fields = [key for key in req.field_keys if key in field_invalid]
        if local_missing:
            status = "LOCAL_FAIL"
        elif missing_fields or invalid_fields:
            status = "FIELD_REQUIRED"
        else:
            status = "PASS"
        rows.append((req, status, local_missing, missing_fields, invalid_fields))
    return rows, status_text


def _write_markdown(path: Path, rows):
    lines = [
        "# Experiment Requirement Audit",
        "",
        "| Requirement | Status | Missing local evidence | Missing field evidence | Invalid field evidence |",
        "| --- | --- | --- | --- | --- |",
    ]
    for req, status, local_missing, missing_fields, invalid_fields in rows:
        lines.append(
            "| {key} | {status} | {local} | {field} | {invalid} |".format(
                key=req.key,
                status=status,
                local="<br>".join(local_missing) if local_missing else "-",
                field=", ".join(missing_fields) if missing_fields else "-",
                invalid=", ".join(invalid_fields) if invalid_fields else "-",
            )
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main():
    parser = argparse.ArgumentParser(description="Audit experiment requirements against current evidence")
    parser.add_argument("--expected-yaw-rate", default="0.35")
    parser.add_argument("--gis-bounds", default="120.0,30.0,121.0,31.0")
    parser.add_argument("--max-halss-p95-ms", default="70")
    parser.add_argument("--max-depth-p95-ms", default="15")
    parser.add_argument("--max-completion-p95-ms", default="45")
    parser.add_argument("--max-rl-p95-ms", default="30")
    parser.add_argument("--out-md", default=None)
    parser.add_argument("--strict-local", action="store_true",
                        help="Fail if any requirement lacks local source/runbook evidence")
    parser.add_argument("--strict-field", action="store_true",
                        help="Fail unless all field evidence is present and accepted")
    args = parser.parse_args()

    rows, _ = audit(args)
    print("Experiment requirement audit")
    counts = {"PASS": 0, "FIELD_REQUIRED": 0, "LOCAL_FAIL": 0}
    for req, status, local_missing, missing_fields, invalid_fields in rows:
        counts[status] += 1
        print(f"[{status}] {req.key}: {req.description}")
        if local_missing:
            print("       local_missing: " + "; ".join(local_missing))
        if missing_fields:
            print("       field_missing: " + ", ".join(missing_fields))
        if invalid_fields:
            print("       field_invalid: " + ", ".join(invalid_fields))
    print(
        "summary: "
        f"PASS={counts['PASS']} FIELD_REQUIRED={counts['FIELD_REQUIRED']} "
        f"LOCAL_FAIL={counts['LOCAL_FAIL']}"
    )
    if args.out_md:
        out = _write_markdown(Path(args.out_md), rows)
        print(f"markdown_report: {out}")
    if args.strict_local and counts["LOCAL_FAIL"]:
        return 1
    if args.strict_field and (counts["LOCAL_FAIL"] or counts["FIELD_REQUIRED"]):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
