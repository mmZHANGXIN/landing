#!/usr/bin/env python3
"""Lightweight checks that the field runbook still matches acceptance gates."""

from pathlib import Path


ROOT = Path(__file__).resolve().parent


def _read(name):
    return (ROOT / name).read_text(encoding="utf-8")


def _require_all(text, needles, label):
    missing = [needle for needle in needles if needle not in text]
    assert not missing, f"{label} missing: {missing}"


def test_readme_no_control_command_collects_required_evidence():
    text = _read("README.md")
    _require_all(
        text,
        [
            "python test_live_nocontrol.py \\",
            "--depth-output-scale <calibrated_scale>",
            "--yaw-rate-rad-s <yaw_rate>",
            "--save-raw-arrays",
            "--save-frames",
            "--require-depth-completion",
            "--require-rl-model",
            "--duration-sec 120",
            "python test_requirement_traceability_light.py",
            "python audit_experiment_requirements_light.py --strict-local",
            "python check_orin_env.py --strict --require-jetson --out-md experiments/logs/orin_env.md",
            "python accept_orin_env_light.py experiments/logs/orin_env.md",
            "python test_mavsdk_home_telemetry_light.py",
            "tee experiments/logs/nocontrol.log",
            "python analyze_timing_log.py experiments/logs/nocontrol.log --budget-ms 100 --max-p95-ms 100",
            "python accept_nocontrol_log.py experiments/logs/nocontrol.log \\",
            "python diagnose_nocontrol_action_log.py experiments/logs/nocontrol.log \\",
            "--expected-yaw-rate <yaw_rate>",
            "--max-halss-p95-ms 70",
            "--max-depth-p95-ms 15",
            "--max-completion-p95-ms 45",
            "--max-rl-p95-ms 30",
            "--max-sync-p95-ms 100",
            "--max-yaw-transform-error 0.15",
            "--expected-safe-point <best_center_lat,best_center_lon>",
            "--require-action-probs",
            "Action mapping: frame=body lateral_sign=-1 act3=W",
            "默认实验映射下 act=3(E) 会失败",
            "python accept_drl_diagnosis_light.py experiments/logs/drl_live_frame.json \\",
            "--frame-glob 'experiments/frames/*_calib_frame.npz'",
            "--min-items 8",
            "python accept_sparsenet_calibration_light.py experiments/logs/sparsenet_scale.json \\",
            "python inspect_nocontrol_artifacts_light.py experiments/frames --require-frames",
            "--max-sync-ms 100",
            "cloud_odom_sync_ms",
            "python compare_saved_binary_semantic_light.py --frame-dir experiments/frames --grayscale",
            "python run_field_acceptance.py \\",
            "--gis-prior <global_prior_save_dir_or_json>",
            "source_image_path",
            "segmentation_source",
            "--strict-flight-ready",
            "python field_evidence_status.py --strict",
            "python test_depth_projection_cuda.py 2>&1 | tee experiments/logs/depth_projection_cuda.log",
            "python accept_depth_projection_cuda_light.py experiments/logs/depth_projection_cuda.log",
            "python audit_experiment_requirements_light.py --strict-field \\",
            "--validate-artifacts",
            "field_evidence_status.md",
            "orin_env.md",
            "--skip-env-check",
        ],
        "README no-control evidence command",
    )


def test_readme_unified_field_report_includes_all_real_experiment_evidence():
    text = _read("README.md")
    _require_all(
        text,
        [
            "python run_acceptance_light.py \\",
            "--gis-prior <global_prior_save_dir_or_json>",
            "--gis-bounds lon_left,lat_bottom,lon_right,lat_top",
            "--nocontrol-log experiments/logs/nocontrol.log",
            "no-control action diagnosis evidence",
            "--frame-dir experiments/frames",
            "--require-frames",
            "--drl-diagnosis-json experiments/logs/drl_live_frame.json",
            "--drl-diagnosis-live",
            "--min-drl-diagnosis-items 8",
            "--sparsenet-calibration-json experiments/logs/sparsenet_scale.json",
            "--depth-projection-cuda-log experiments/logs/depth_projection_cuda.log",
            "--orin-env-md experiments/logs/orin_env.md",
            "--ros-topic-log /livox/lidar=experiments/logs/hz_livox_lidar.log",
            "--ros-topic-log /livox/imu=experiments/logs/hz_livox_imu.log",
            "--ros-topic-log /cloud_registered=experiments/logs/hz_cloud_registered.log",
            "--ros-topic-log /Odometry=experiments/logs/hz_odometry.log",
            "--require-all-ros-topics",
            "--expected-yaw-rate <yaw_rate>",
            "--max-halss-p95-ms 70",
            "--max-depth-p95-ms 15",
            "--max-completion-p95-ms 45",
            "--max-rl-p95-ms 30",
            "--max-sync-p95-ms 100",
            "--max-yaw-transform-error 0.15",
            "--max-flight-action-run 60",
            "--require-action-probs",
        ],
        "README unified field report command",
    )


def test_readme_gis_and_flight_loop_are_not_optional_in_flight_path():
    text = _read("README.md")
    _require_all(
        text,
        [
            "--flight-ready-check-only 只验证参数, 不会实际运行GIS分割/九宫格",
            "不会生成 global_prior_*.json",
            "python accept_gis_prior_light.py <global_prior_save_dir_or_json> \\",
            "python pipeline.py --config ./config/experiment_config.yaml --mode ros \\",
            "--safe-point-source gis",
            "tee experiments/logs/pipeline.log",
            "python accept_flight_loop_log.py experiments/logs/pipeline.log \\",
            "--require-global-guidance",
            "闭环验收同样会拒绝默认实验映射下的 act=3(E) 日志",
            "--max-halss-p95-ms 70",
            "--max-depth-p95-ms 15",
            "--max-completion-p95-ms 45",
            "--max-rl-p95-ms 30",
            "--max-sync-p95-ms 100",
            "--max-yaw-transform-error 0.15",
            "python run_acceptance_light.py \\",
            "--flight-log experiments/logs/pipeline.log",
        ],
        "README GIS/closed-loop flight evidence",
    )


def test_readme_realtime_claims_are_measured_not_assumed():
    text = _read("README.md")
    _require_all(
        text,
        [
            "在线管线（飞行中实时循环，验收以实测 P95 为准）",
            "HALSS Bayesian",
            "CPU surface-normal 预处理 + CUDA UNet/MC Dropout 推理",
            "PyTorch CUDA 透视投影 + GPU z-buffer",
            "不使用静态估计值作为通过证据",
            "`total` P95 `<=100ms`",
            "不允许按原频率飞行",
        ],
        "README measured realtime claims",
    )
    forbidden = ["目标 ~26Hz", "**~35ms**", "约 28Hz"]
    present = [needle for needle in forbidden if needle in text]
    assert not present, f"README still has assumed realtime claims: {present}"


def test_acceptance_document_tracks_action_mapping_and_known_gaps():
    text = _read("EXPERIMENT_ACCEPTANCE.md")
    _require_all(
        text,
        [
            "uav.action_lateral_sign=-1",
            "action 3 is `W`",
            "`act=3` is W in original DeepRL, E only in mirrored mode",
            "Action mapping: frame=body lateral_sign=-1 act3=W",
            "[Init] Action mapping frame=body lateral_sign=-1 (act3=W)",
            "logs with default `action_lateral_sign=-1` reject `act=3(E)`",
            "No-control and closed-loop logs pass action mapping checks",
            "depth_completion.output_scale",
            "Global guidance waits for valid home telemetry",
            "[Pipeline] Home telemetry",
            "accept_sparsenet_calibration_light.py",
            "accept_drl_diagnosis_light.py",
            "diagnose_nocontrol_action_log.py",
            "ACTION_MAPPING_MISMATCH",
            "ZERO_YAW_RATE",
            "LEGACY_LOG_FORMAT",
            "SINGLE_ACTION_COLLAPSE",
            "SEMANTIC_ONE_CLASS",
            "test_pipeline_diagnostic_snapshot_light.py",
            "cloud_odom_sync_ms",
            "--frame-glob 'experiments/frames/*_calib_frame.npz'",
            "--require-live-frame --require-probs --min-items 8",
            "python field_evidence_status.py --strict",
            "--validate-artifacts",
            "required_missing: 0",
            "python run_field_acceptance.py --expected-yaw-rate <rate>",
            "field_evidence_status.md",
            "orin_env.md",
            "Original requirements stay traceable",
            "test_requirement_traceability_light.py",
            "Requirement evidence audit is explicit",
            "audit_experiment_requirements_light.py --strict-local",
            "FIELD_REQUIRED",
            "audit_experiment_requirements_light.py --strict-field",
            "global_prior.enabled=true",
            "source_image_path",
            "image_size_px",
            "source_sem_mask_path",
            "segmentation_source",
            "uav.yaw_rate_rad_s",
            "max_cloud_odom_sync_ms",
            "python run_acceptance_light.py --gis-prior <dir_or_json>",
            "--nocontrol-log experiments/logs/nocontrol.log",
            "--drl-diagnosis-json experiments/logs/drl_live_frame.json",
            "--sparsenet-calibration-json experiments/logs/sparsenet_scale.json",
            "--depth-projection-cuda-log experiments/logs/depth_projection_cuda.log",
            "--orin-env-md experiments/logs/orin_env.md",
            "--flight-log experiments/logs/pipeline.log",
            "--require-global-guidance",
            "--expected-safe-point <best_center_lat,best_center_lon>",
            "--expected-yaw-rate <rate>",
            "--require-action-probs",
            "--max-flight-action-run 60",
            "--max-halss-p95-ms 70",
            "--max-depth-p95-ms 15",
            "--max-completion-p95-ms 45",
            "--max-rl-p95-ms 30",
            "--max-sync-p95-ms 100",
            "--max-yaw-transform-error 0.15",
            "yaw transform",
            "sync",
            "best_center_gps",
            "accept_depth_projection_cuda_light.py",
            "depth_projection_cuda.log",
            "HALSS CPU preprocessing risk is visible",
            "Depth projection GPU path is budgeted",
            "Realtime claims are measured, not assumed",
            "Some preprocessing is still CPU-bound",
        ],
        "EXPERIMENT_ACCEPTANCE action mapping and field gaps",
    )


def test_field_atomic_checklist_tracks_operator_sequence():
    text = _read("FIELD_ATOMIC_CHECKLIST.md")
    _require_all(
        text,
        [
            "Orin 真机实验现场原子清单",
            "binary_semantic_window_title",
            "Action mapping: frame=body lateral_sign=-1 act3=W",
            "[Pipeline] Home telemetry",
            "默认实验下出现 `act=3(E)` 视为不合格",
            "python accept_gis_prior_light.py <global_prior_save_dir_or_json>",
            "source_image_path",
            "image_size_px",
            "segmentation_source",
            "python accept_ros_topics_light.py",
            "python test_depth_projection_cuda.py 2>&1 | tee experiments/logs/depth_projection_cuda.log",
            "python accept_depth_projection_cuda_light.py experiments/logs/depth_projection_cuda.log",
            "python calibrate_sparsenet_scale.py",
            "python test_live_nocontrol.py \\",
            "--duration-sec 120",
            "python compare_saved_binary_semantic_light.py --frame-dir experiments/frames --grayscale",
            "python diagnose_drl_policy.py --scan-modes",
            "python diagnose_nocontrol_action_log.py experiments/logs/nocontrol.log",
            "--frame-glob 'experiments/frames/*_calib_frame.npz'",
            "--min-items 8",
            "python preflight_check.py --config ./config/experiment_config.yaml --flight-ready",
            "python accept_flight_loop_log.py experiments/logs/pipeline.log",
            "python field_evidence_status.py --strict",
            "python field_evidence_status.py --strict --validate-artifacts",
            "python audit_experiment_requirements_light.py --strict-field",
            "--out-md experiments/logs/field_evidence_status.md",
            "python check_orin_env.py --strict --require-jetson --out-md experiments/logs/orin_env.md",
            "python accept_orin_env_light.py experiments/logs/orin_env.md",
            "python run_field_acceptance.py \\",
            "orin_env.md",
            "required_missing: 0",
            "depth_projection_cuda",
            "不再出现 `FIELD_REQUIRED` 或 `LOCAL_FAIL`",
            "experiments/logs/field_evidence_status.md",
            "--skip-env-check",
            "--max-halss-p95-ms 70",
            "--max-depth-p95-ms 15",
            "--max-completion-p95-ms 45",
            "--max-rl-p95-ms 30",
            "--max-sync-p95-ms 100",
            "--max-yaw-transform-error 0.15",
            "`sync`",
            "best_center_gps",
            "max_cloud_odom_sync_ms=100",
            "诊断 NPZ",
            "HALSS 显示一致性",
            "HALSS CPU surface-normal",
            "torch_cuda",
            "现场 P50/P95/max",
        ],
        "FIELD_ATOMIC_CHECKLIST operator sequence",
    )


def main():
    test_readme_no_control_command_collects_required_evidence()
    test_readme_unified_field_report_includes_all_real_experiment_evidence()
    test_readme_gis_and_flight_loop_are_not_optional_in_flight_path()
    test_readme_realtime_claims_are_measured_not_assumed()
    test_acceptance_document_tracks_action_mapping_and_known_gaps()
    test_field_atomic_checklist_tracks_operator_sequence()
    print("=== Lightweight runbook contract acceptance ===")
    print("  OK no-control command saves raw arrays, frames, timing, probabilities, and yaw-rate evidence")
    print("  OK unified field report command includes GIS, Mid360/FAST-LIO topics, no-control frames, and yaw evidence")
    print("  OK GIS check-only limitation and closed-loop global-guidance gates are documented")
    print("  OK realtime claims are tied to measured P95 evidence")
    print("  OK action mapping and remaining field gaps are explicit")
    print("  OK field atomic checklist tracks the operator sequence")
    print("=== ALL PASSED ===")


if __name__ == "__main__":
    main()
