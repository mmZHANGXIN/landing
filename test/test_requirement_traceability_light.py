#!/usr/bin/env python3
"""Trace the original experiment requirements to acceptance-document evidence."""

from pathlib import Path


ROOT = Path(__file__).resolve().parent


REQUIREMENTS = {
    "offline_gis_segmentation_and_nine_grid": [
        "Offline GIS Global Safe Area",
        "Nine-grid risk",
        "GIS result JSON is machine-checkable",
        "best_cell",
        "best_center_gps",
    ],
    "global_guidance_before_drl": [
        "Pipeline enters GOTO_SAFE before DRL descent",
        "[GOTO_SAFE] Target NED",
        "[GOTO_SAFE] Arrived",
        "require-global-guidance",
    ],
    "mid360_fastlio_deskewed_cloud_and_pose": [
        "FAST-LIO And Mid360 Data",
        "/cloud_registered",
        "/Odometry",
        "deskewed cloud",
        "roll/pitch/yaw",
    ],
    "halss_bayesian_binary_semantic": [
        "HALSS Binary Semantic",
        "Bayesian HALSS backend",
        "safe_mesh",
        "safety_map_vis",
        "binary semantic",
    ],
    "halss_visualization_pixel_identity": [
        "Display uses HALSS visualization passthrough",
        "Saved binary semantic equals HALSS NPZ output",
        "HALSS visualization pixel match",
        "compare_halss_visualization_light.py",
    ],
    "perspective_depth_projection": [
        "Down-Looking Depth Projection And SparseNet",
        "Perspective projection",
        "depth_projection.mode: \"perspective\"",
        "depth_projection.backend: \"torch_cuda\"",
        "fx/fy/cx/cy/R_I_to_C",
        "CUDA projection matches the reference geometry",
        "accept_depth_projection_cuda_light.py",
        "z-buffer",
    ],
    "sparsity_invariant_depth_completion": [
        "Sparse convolution formula",
        "SparseNet uses `x=1-D/dmax`",
        "input_encoding: \"inverse_unit\"",
        "mask max-pool",
        "SparseNet calibration JSON is accepted",
    ],
    "drl_policy_inference_and_action_output": [
        "DRL Policy And Observation Distribution",
        "Live DRL diagnosis JSON is accepted",
        "Live action probabilities are logged",
        "Terminal prints `act=<id>(<name>)`",
        "Dummy policy is forbidden",
    ],
    "yaw_fault_body_to_ned_action_decomposition": [
        "Yaw-Fault Action Decomposition",
        "action_frame=body",
        "current FAST-LIO yaw",
        "yaw_rate_rad_s",
        "v_body",
        "v_ned",
    ],
    "gpu_realtime_single_frame_budget": [
        "Runtime is configured for GPU inference",
        "HALSS/DRL CPU fallback is disabled",
        "No CUDA cold-start in flight loop",
        "Timing log passes budget gates",
        "max-total-p95-ms 100",
        "SLOW FRAME DROPPED",
    ],
    "live_no_control_visual_output": [
        "No-Control Online Test",
        "`binary semantic` window opens and updates",
        "Depth window opens and updates",
        "save-frames",
        "field_evidence_status.py --strict",
    ],
    "closed_loop_flight_readiness": [
        "Controlled Flight Readiness",
        "preflight_check.py --flight-ready",
        "accept_flight_loop_log.py",
        "run_acceptance_light.py --gis-prior",
        "required_missing: 0",
    ],
}


def _read_acceptance():
    return (ROOT / "EXPERIMENT_ACCEPTANCE.md").read_text(encoding="utf-8")


def _missing_for(text, needles):
    return [needle for needle in needles if needle not in text]


def test_requirement_traceability():
    text = _read_acceptance()
    failures = {}
    for requirement, needles in REQUIREMENTS.items():
        missing = _missing_for(text, needles)
        if missing:
            failures[requirement] = missing
    assert not failures, failures


def main():
    test_requirement_traceability()
    print("=== Lightweight requirement traceability acceptance ===")
    for requirement in sorted(REQUIREMENTS):
        print(f"  OK {requirement}")
    print("=== ALL PASSED ===")


if __name__ == "__main__":
    main()
