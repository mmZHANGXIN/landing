# Orin Landing Experiment Acceptance

This checklist tracks the real-flight experiment scope:
GIS global safe-area guidance, Mid360/FAST-LIO deskewed point clouds, HALSS
binary safety semantics, down-looking depth completion, DeepRL inference, and
yaw-aware velocity control.

## Gate 0: Environment And Assets

| Atomic item | Evidence command | Pass criteria |
| --- | --- | --- |
| Lightweight evidence bundle is generated | `python run_acceptance_light.py` | all non-hardware gates pass and `experiments/logs/acceptance_light_*.md/json` are written |
| Field evidence status is visible | `python field_evidence_status.py --strict --validate-artifacts --expected-yaw-rate <rate> --gis-bounds lon_left,lat_bottom,lon_right,lat_top --max-halss-p95-ms 70 --max-depth-p95-ms 15 --max-completion-p95-ms 45 --max-rl-p95-ms 30 --max-sync-p95-ms 100 --max-yaw-transform-error 0.15 --out-md experiments/logs/field_evidence_status.md` after field runs | required Orin environment, GIS, ROS topic, no-control, artifact, DRL diagnosis, SparseNet calibration, and pipeline evidence files are present and accepted; missing/invalid items print produce/accept hints and the Markdown status report is written |
| Original requirements stay traceable | `python test_requirement_traceability_light.py` | GIS, FAST-LIO, HALSS, depth completion, DRL, yaw-aware control, realtime, visualization, and closed-loop readiness are all covered by acceptance text |
| Requirement evidence audit is explicit | `python audit_experiment_requirements_light.py --strict-local` | local source/runbook evidence for every original requirement is present; requirements that still need Orin/Mid360/flight artifacts are reported as `FIELD_REQUIRED`, not silently counted as complete |
| Orin environment compatibility is reported | `python check_orin_env.py --strict --require-jetson --out-md experiments/logs/orin_env.md` then `python accept_orin_env_light.py experiments/logs/orin_env.md` on Orin | JetPack, NVCC, ROS2 CLI/setup, rclpy, PyTorch CUDA, OpenCV with `DISPLAY` available, NumPy, PyYAML, SB3, and MAVSDK rows are all `Required=True` and `PASS`; Markdown report is written and accepted |
| Jetson CUDA is available | `python -c "import torch; print(torch.cuda.is_available())"` | prints `True` |
| Runtime is configured for GPU inference | `python test_flight_ready_gates.py` | includes `runtime GPU` failure case |
| HALSS/DRL CPU fallback is disabled | `python test_gpu_fail_closed_contract_light.py` and `python test_flight_ready_gates.py` | `perception.require_gpu=true` and `decision.require_gpu=true`; strict gates reject disabled module GPU requirements and runtime modules fail closed instead of silently using CPU |
| Config parses | `python preflight_check.py --skip-model-load` | no `[FAIL]` except CUDA/model checks skipped by design |
| HALSS weights exist | `python preflight_check.py --skip-model-load` | `[OK] HALSS weight exists` |
| SparseNet weights exist | `python preflight_check.py --skip-model-load` | `[OK] SparseNet weight exists` |
| DRL SB3 policy exists | `python preflight_check.py --skip-model-load` | `[OK] DRL policy exists` |
| DRL metadata matches online encoding | `python inspect_drl_model.py weights/last_step_model_sb3.zip` | shape `(128,128,2)`, `normalize_images=false`, custom `SB2CNN`, no `[FAIL]` |
| SparseNet loads on GPU | `python preflight_check.py` | `[OK] SparseNet DepthCompletion loads, validates, and warms up` |
| DRL policy loads | `python preflight_check.py` | `[OK] DRL SB3 policy loads`; no dummy policy fallback |
| SparseNet scale risk is visible | `python preflight_check.py --skip-model-load` | warns when `depth_completion.output_scale` is still `null` |
| Strict flight gates reject incomplete config | `python preflight_check.py --flight-ready --skip-model-load` before calibration/GIS/yaw setup | fails on missing GIS guidance, null `output_scale`, or zero yaw rate |
| Strict flight-ready evidence bundle is clean after field config | `python run_acceptance_light.py --strict-flight-ready` after GIS target, `output_scale`, and yaw rate are configured | overall `PASS`; no remaining flight-ready failures |
| Flight pipeline enforces strict gates | `python pipeline.py --config ./config/experiment_config.yaml --mode ros` with incomplete config | exits before model/ROS/MAVSDK startup unless `--allow-incomplete-experiment` is explicitly set |
| Global guidance waits for valid home telemetry | `python test_mavsdk_home_telemetry_light.py` and runtime log | MAVSDK rejects zero GPS, waits for NED/GPS/attitude readiness, and pipeline logs `[Pipeline] Home telemetry` before converting GIS safe point to NED |
| CLI experiment overrides are gate-checked | `python pipeline.py --mode ros --safe-point <lat>,<lon> --safe-point-source gis --depth-output-scale <scale> --yaw-rate-rad-s <rate> --flight-ready-check-only` | strict gates pass without loading models/ROS |
| Manual safe-point cannot bypass GIS stage | `python test_flight_ready_gates.py` | includes `manual safe-point` failure case unless `--safe-point-source gis` is declared |
| CLI global guidance overrides are validated | `python pipeline.py --mode ros --gis-image <sat.png> --gis-bounds lon_left,lat_bottom,lon_right,lat_top --depth-output-scale <scale> --yaw-rate-rad-s <rate> --flight-ready-check-only` | missing image/mask, missing bounds, or malformed safe-point fails before model/ROS init |
| Live visualization cannot be disabled for flight | `python test_flight_ready_gates.py` | includes failure cases for `visualization.enable`, `show_binary_semantic`, and `show_depth` |
| Flight gate behavior has regression coverage | `python test_flight_ready_gates.py` | all positive/negative check-only cases pass without numpy/torch/ROS |

## Gate 1: Offline GIS Global Safe Area

| Atomic item | Evidence command | Pass criteria |
| --- | --- | --- |
| Nine-grid risk works without heavy deps | `python test_gis_nine_grid_light.py` | best cell is center, risk grid is 3x3, GPS center is inside bounds |
| Nine-grid risk works on mock mask with full stack | `python test_gis_nine_grid.py --mock` on Orin | `ALL GIS 验收 PASSED` |
| Bounds missing refuses GPS target | `python test_gis_nine_grid.py --mock` | `best_center_gps=None` case passes |
| Bounds produce GPS target | `python test_gis_nine_grid.py --image <sat.png> --mask <sem.png> --bounds lon_left,lat_bottom,lon_right,lat_top` | GPS lies inside bounds |
| GIS result JSON is machine-checkable | after a real GIS run writes `global_prior.save_dir` (not check-only), run `python accept_gis_prior_light.py <save_dir_or_global_prior_json> --bounds lon_left,lat_bottom,lon_right,lat_top` | `source_image_path`, `image_size_px`, `bounds`, and `source_sem_mask_path` or `segmentation_source` are recorded; `risk_grid` is 3x3, `best_cell`/`min_risk` match the lowest cell, and `best_center_gps` lies inside bounds |
| Flight config enables global guidance when needed | inspect `config/experiment_config.yaml` | `global_prior.enabled=true` and either `target_lat/lon` with `target_source: "gis"` or `image_path+bounds` |
| Pipeline enters GOTO_SAFE before DRL descent | runtime log | logs parseable `[GOTO_SAFE] Target NED: <north>, <east> tolerance=<m>m` then `[GOTO_SAFE] Arrived. XY error=<m>m`; arrival error is no larger than tolerance and target NED is not near zero |

## Gate 2: FAST-LIO And Mid360 Data

| Atomic item | Evidence command | Pass criteria |
| --- | --- | --- |
| Livox driver publishes points | `ros2 topic hz /livox/lidar` | stable sensor rate |
| Topic rates are saved and accepted | capture four `ros2 topic hz` logs, then run `python accept_ros_topics_light.py /livox/lidar=<log> /livox/imu=<log> /cloud_registered=<log> /Odometry=<log> --require-all` | Livox lidar/imu and FAST-LIO cloud/odometry logs meet minimum rate/window thresholds |
| FAST-LIO publishes deskewed cloud | `ros2 topic hz /cloud_registered` | stable rate, no long dropouts |
| FAST-LIO publishes pose | `ros2 topic hz /Odometry` | stable rate |
| PointCloud2 parsing supports field offsets | `python test_live_nocontrol.py` | no PointCloud callback errors |
| Odom pose includes roll/pitch/yaw | `python test_live_nocontrol.py` | depth projection receives 6-DoF pose |

## Gate 3: HALSS Binary Semantic

| Atomic item | Evidence command | Pass criteria |
| --- | --- | --- |
| Bayesian HALSS weight loads | `python preflight_check.py` | no HALSS weight missing failure |
| Bayesian HALSS backend is mandatory | `python test_flight_ready_gates.py` | includes `HALSS backend` failure case |
| HALSS produces `safe_mesh` and `safety_map_vis` | `python test_live_nocontrol.py` | semantic window updates from HALSS output |
| HALSS CPU preprocessing risk is visible | inspect `perception/halss_bayesian.py` and no-control timing log | current path records `H=` timing for CPU surface-normal preprocessing plus CUDA Bayesian UNet inference; if H P95 pushes total P95 over budget, reduce MC/grid settings or port preprocessing to GPU and rerun acceptance |
| Required live windows are enabled | inspect config and run `python preflight_check.py --flight-ready --skip-model-load` | `visualization.enable=true`, `show_binary_semantic=true`, and `show_depth=true` pass the strict gate |
| Window title is fixed | visual check | window title is exactly `binary semantic` |
| Window title is flight-gated | `python test_flight_ready_gates.py` | includes `binary semantic title` failure case |
| Display uses HALSS visualization passthrough | code path in `visualization/display.py` | `binary_semantic_vis` is passed to `_render_binary_semantic` |
| Saved binary semantic/depth frames are present | run `python test_live_nocontrol.py --save-frames --save-dir experiments/frames` then `python inspect_nocontrol_artifacts_light.py experiments/frames --require-frames --max-sync-ms 100 --max-yaw-transform-error 0.15` | matching `*_binary_semantic.png` and `*_depth.png` pairs are valid PNGs |
| Saved binary semantic equals HALSS NPZ output | `python compare_saved_binary_semantic_light.py --frame-dir experiments/frames --grayscale` | each compared `*_binary_semantic.png` exactly matches the same-frame `binary_semantic_vis` from `*_calib_frame.npz` |
| HALSS visualization pixel match | `python compare_halss_visualization.py --reference <halss_ref.png> --candidate <saved_binary_semantic.png> --grayscale` | exact match or configured diff threshold passes |
| HALSS visualization pixel match without OpenCV | `python compare_halss_visualization_light.py --reference <halss_ref.png> --candidate <saved_binary_semantic.png> --grayscale` | exact match with `mean_abs_diff=0` and `max_abs_diff=0` |
| HALSS visualization comparison self-test | `python test_halss_visualization_light.py` | exact-match, one-pixel mismatch, PNG grayscale, and nearest resize checks pass |
| Visualization contract has source-level guard | `python test_visualization_contract_light.py` | no-control and CV2 display paths keep `binary semantic`, HALSS passthrough, and nearest interpolation |
| Safety ratios are logged | `python test_live_nocontrol.py` | log contains `sem_safe=` and `sem_danger=` |

## Gate 4: Down-Looking Depth Projection And SparseNet

| Atomic item | Evidence command | Pass criteria |
| --- | --- | --- |
| Perspective projection formula works without heavy deps | `python test_depth_projection_light.py` | center depth, single-point projection, `R_GI^T(Gp-GpI)` translation/yaw transform, `R_I_to_C` body-to-camera axes, z-buffer, and invalid-point `dmax` checks pass |
| Perspective projection geometry works with DepthProjector | `python test_depth_projection.py` on Orin | `DepthProjector tests passed`; the NumPy reference path keeps one valid point, applies pose inverse transform and down-looking camera extrinsics, and leaves behind/out-of-frame/max-range points at `dmax` |
| CUDA projection matches the reference geometry | `python test_depth_projection_cuda.py 2>&1 \| tee experiments/logs/depth_projection_cuda.log` on Orin, then `python accept_depth_projection_cuda_light.py experiments/logs/depth_projection_cuda.log` | PyTorch CUDA is available and `torch_cuda` projection matches the NumPy reference for body-axis, z-buffer, invalid-point, translated/yawed-pose, and single-point cases; the tee log is accepted as field evidence |
| Projection mode is experiment path | inspect config | `depth_projection.mode: "perspective"` |
| Depth projection uses GPU for flight | inspect config and run `python test_flight_ready_gates.py` | `depth_projection.backend: "torch_cuda"` and the strict flight gate rejects `backend: "numpy"` |
| Flight gate rejects BEV depth fallback | `python test_flight_ready_gates.py` | includes `depth projection mode` failure case |
| Intrinsics are configured | inspect config | `fx/fy/cx/cy/R_I_to_C` present |
| Depth projection GPU path is budgeted | inspect `perception/depth_projection.py` and no-control timing log | `torch_cuda` perspective projection records `D=` timing; D P95 must fit the total latency gate or be optimized/revalidated before flight |
| Sparse convolution formula works without heavy deps | `python test_sparse_conv_light.py` | inverse-depth encoding, valid-count normalization, zero-mask, mask max-pool, and output-scale checks pass |
| SparseNet uses `x=1-D/dmax` | inspect config/log | `input_encoding: "inverse_unit"` and DepthCompletion startup log shows it |
| Flight gate rejects non-inverse SparseNet encoding | `python test_flight_ready_gates.py` | includes `SparseNet input encoding` failure case |
| Sparse depth validity is logged | `python test_live_nocontrol.py` | log contains `valid=` |
| Dense depth stats are logged | `python test_live_nocontrol.py` | log contains `depth=min/mean/max` |
| Raw calibration arrays can be saved | run `python test_live_nocontrol.py --save-raw-arrays --save-dir experiments/frames` then `python inspect_nocontrol_artifacts_light.py experiments/frames --max-sync-ms 100 --max-yaw-transform-error 0.15` | `*_calib_frame.npz` contains `sparse_depth`, `valid_mask`, `dense_depth`, `sem_map`, `binary_semantic_vis`, pose, yaw, `cloud_odom_sync_ms`, action, and velocity fields; saved `v_ned` matches yaw-rotated `v_body` |
| SparseNet output scale is estimated | `python calibrate_sparsenet_scale.py --input experiments/frames/000000_calib_frame.npz --target-depth-m <measured_height> --out-json experiments/logs/sparsenet_scale.json` | prints `Suggested config: depth_completion.output_scale: ...` and writes calibration JSON |
| SparseNet calibration JSON is accepted | `python accept_sparsenet_calibration_light.py experiments/logs/sparsenet_scale.json --require-improvement` | positive finite `suggested_output_scale`, enough fit pixels, acceptable scaled median/P90 error, and scaled median error no worse than raw |
| SparseNet output is calibrated | rerun with calibrated config | `output_scale` set from ground measurement; dense depth median error acceptable |

## Gate 5: DRL Policy And Observation Distribution

| Atomic item | Evidence command | Pass criteria |
| --- | --- | --- |
| Synthetic observations do not collapse silently | `python diagnose_drl_policy.py --scan-modes --out-json experiments/logs/drl_synthetic.json --fail-on-collapse` | JSON report is written and command fails if any encoding maps all cases to one action |
| Original DeepRL encoding is default | inspect config | `depth_norm_mode: "meters_div255"` and `semantic_norm_mode: "gray_unit"` |
| Flight-ready gate rejects non-DeepRL observation encoding | `python test_flight_ready_gates.py` | includes `bad observation encoding` failure case |
| SB3 metadata is checked without CUDA | `python inspect_drl_model.py` | no unexpected observation shape/normalization warnings before inference tests |
| Alternative encodings are checked | `python diagnose_drl_policy.py --scan-modes` | action distribution recorded for `unit/unit`, `meters_div255/gray_unit`, `meters/raw`, `inverse_unit/unit` |
| Live saved frames are checked | `python diagnose_drl_policy.py --scan-modes --frame-glob 'experiments/frames/*_calib_frame.npz' --out-json experiments/logs/drl_live_frame.json --fail-on-collapse` | action distribution/probabilities are recorded across at least 8 live frames; command fails on single-action collapse |
| Live DRL diagnosis JSON is accepted | `python accept_drl_diagnosis_light.py experiments/logs/drl_live_frame.json --require-live-frame --require-probs --min-items 8` | action mapping is `body/-1`, required `meters_div255,gray_unit` encoding exists, action names match runtime mapping, and checked encodings are not collapsed |
| Live observation stats are logged | `python test_live_nocontrol.py` | log contains `obsD=` and `obsS=` |
| Live action probabilities are logged | `python test_live_nocontrol.py` | log contains `p=<id>:<prob>` |
| Repeated-action collapse is captured | `python test_live_nocontrol.py` with a repeated action stream | log contains `DRL action collapse` and writes `*_action_collapse_a<id>_*.npz` |
| Closed-loop collapse snapshots keep evidence | `python test_pipeline_diagnostic_snapshot_light.py` | pipeline action-collapse NPZ payload includes `binary_semantic_vis`, `cloud_odom_sync_ms`, `yaw_rad`, `action_id`, `v_body`, and `v_ned` for post-run HALSS/sync/yaw diagnosis |
| Dummy policy is forbidden in flight | `python preflight_check.py` | no `[FAIL] DRL model load failed` |
| Action-3-only collapse is explained before flight | `python diagnose_nocontrol_action_log.py experiments/logs/nocontrol.log --expected-yaw-rate <rate> --fail-on-issues`, then `python diagnose_drl_policy.py --scan-modes --frame-glob 'experiments/frames/*_calib_frame.npz' --out-json experiments/logs/drl_live_frame.json --fail-on-collapse` | log-level diagnosis rejects `ACTION_MAPPING_MISMATCH`, `ACTION_NAME_MISMATCH`, `ZERO_YAW_RATE`, `LEGACY_LOG_FORMAT`, `SINGLE_ACTION_COLLAPSE`, `MISSING_ACTION_PROBS`, `LOW_VALID_DEPTH`, or `SEMANTIC_ONE_CLASS`; frame-level report either shows action diversity or identifies observation degeneration/model conversion issue |

## Gate 6: Yaw-Fault Action Decomposition

| Atomic item | Evidence command | Pass criteria |
| --- | --- | --- |
| Body-frame yaw compensation passes tests | `python control/action_decomposer.py` | `ALL PASSED` |
| `action_frame` config is honored | `python control/action_decomposer.py` | includes `action_frame=ned` test |
| Flight-ready gate requires body-frame yaw compensation | `python test_flight_ready_gates.py` | includes `NED action frame` failure case |
| Original DeepRL action sign is default | inspect config and `python control/action_decomposer.py` | `uav.action_lateral_sign=-1`; action 3 is `W` with `v_body=[0,-1,0]`, matching `quadrotor_env.py` |
| DRL diagnostics use configured action names | `python test_action_names_light.py` | default labels `act3=W`, mirror labels `act3=E`, top-prob labels match runtime mapping |
| Flight-ready gate rejects mirrored actions | `python test_flight_ready_gates.py` | includes `mirrored action sign` failure case |
| Mirrored E-action transform is reproducible if enabled | `python control/action_decomposer.py` | includes `action_lateral_sign=+1` mirror test mapping `act=3(E), yaw=10°` to `v_ned≈[-0.17,0.98,0]` |
| Runtime action mapping is logged | `python test_live_nocontrol.py` or `python pipeline.py --mode ros` | startup log contains `Action mapping: frame=body lateral_sign=-1 act3=W` or `[Init] Action mapping frame=body lateral_sign=-1 (act3=W)` for the default experiment |
| Action labels match runtime mapping | `python accept_nocontrol_log.py experiments/logs/nocontrol.log ...` and `python accept_flight_loop_log.py experiments/logs/pipeline.log ...` | logs with default `action_lateral_sign=-1` reject `act=3(E)` and require `act=3(W)` |
| Yaw rate is configured | inspect config | `uav.yaw_rate_rad_s` set to experiment value, not `0.0`, for yaw-fault test |
| Yaw rate can be swept without editing YAML | `python pipeline.py --flight-ready-check-only --safe-point <lat>,<lon> --depth-output-scale <scale> --yaw-rate-rad-s <rate>` | check passes and logs preview gates passed |
| Live logs include yaw, sync, and velocity | `python test_live_nocontrol.py` | log contains `yaw=`, `yr=`, `sync=`, `v_body=`, `v_ned=` |
| No-control test uses the same yaw/scale overrides | `python test_live_nocontrol.py --depth-output-scale <scale> --yaw-rate-rad-s <rate> --save-raw-arrays` | startup log reports matching `yaw_rate_rad_s`, `output_scale`, `max_cloud_odom_sync_ms`, and raw arrays are saved |
| No-control acceptance forbids fallback models | `python test_live_nocontrol.py --require-depth-completion --require-rl-model ...` | exits if SparseNet is unavailable or RLAgent falls back to dummy policy |
| No-control run can be bounded automatically | `python test_live_nocontrol.py --duration-sec 120 ...` or `--max-frames 30` | process exits cleanly after the requested duration/frame count |
| No-control log passes integrated acceptance | `python accept_nocontrol_log.py experiments/logs/nocontrol.log --expected-yaw-rate <rate> --max-total-p95-ms 100 --max-halss-p95-ms 70 --max-depth-p95-ms 15 --max-completion-p95-ms 45 --max-rl-p95-ms 30 --max-sync-p95-ms 100 --max-yaw-transform-error 0.15 --require-action-probs` | timing, module P95 budgets, cloud/odom sync P95, yaw-aware body-to-NED velocity transform error, yaw rate, valid depth, semantic ratios, action diversity, and probabilities pass |
| Closed-loop log proves GIS-to-DRL order and yaw-aware commands | `python accept_flight_loop_log.py experiments/logs/pipeline.log --require-global-guidance --expected-safe-point <best_center_lat,best_center_lon> --expected-yaw-rate <rate> --max-total-p95-ms 100 --max-halss-p95-ms 70 --max-depth-p95-ms 15 --max-completion-p95-ms 45 --max-rl-p95-ms 30 --max-sync-p95-ms 100 --max-yaw-transform-error 0.15 --max-action-run 60 --require-action-probs` | strict gates pass, logged safe point and global guidance target match GIS `best_center_gps`, GOTO_SAFE target/arrival happens before DRL, target NED is finite/nonzero and arrival XY error is within tolerance, DRL frames include yaw/yaw_sp/yr/sync, `v_ned` matches yaw-rotated `v_body`, observations, action probabilities, per-module timing budgets, and no single-action collapse |

## Gate 7: Realtime And Flight Safety

| Atomic item | Evidence command | Pass criteria |
| --- | --- | --- |
| Pre-control latency budget enforced | runtime log | slow frames log `SLOW FRAME DROPPED` and send zero velocity |
| Single-frame timing visible | `python test_live_nocontrol.py` | log contains `H= D= C= RL= total=` |
| Timing log passes budget gates | `python analyze_timing_log.py experiments/logs/nocontrol.log --budget-ms 100 --max-p95-ms 100` plus integrated log gates with `--max-halss-p95-ms 70 --max-depth-p95-ms 15 --max-completion-p95-ms 45 --max-rl-p95-ms 30 --max-sync-p95-ms 100 --max-yaw-transform-error 0.15` | prints P50/P95/max and `[OK] timing gates passed`; integrated gates fail if one module, cloud/odom sync, or yaw transform consistency exceeds its budget even when total P95 passes |
| Realtime claims are measured, not assumed | README and timing report | documentation does not claim fixed 28Hz/35ms performance; field acceptance requires no-control and closed-loop `total` P95 within the chosen control budget |
| Integrated no-control log gate has self-test | `python test_accept_nocontrol_log_light.py` | good synthetic log passes and zero-yaw single-action log fails |
| No-control action diagnosis has self-test | `python test_diagnose_nocontrol_action_log_light.py` | legacy `act=3(E), yr=0.00` logs produce actionable findings and modern multi-action yaw-fault logs pass |
| No CUDA cold-start in flight loop | `python preflight_check.py` | SparseNet warmup completed before flight |
| MAVSDK offboard initializes before control | runtime log | offboard starts before descent loop |
| Slow-frame behavior preserves yaw fault | runtime log | yaw setpoint continues to advance on dropped frames |
| Emergency stop path exists | induce exception in simulation/no-prop test | zero velocity and disarm are attempted |

## Gate 8: No-Control Online Test

Run with Mid360 and FAST-LIO connected, no flight controller command:

```bash
source /opt/ros/galactic/setup.bash
source ~/livox_ws/install/setup.bash
python test_live_nocontrol.py
```

For automated acceptance on Orin, prefer:

```bash
python test_live_nocontrol.py \
  --depth-output-scale <scale> \
  --yaw-rate-rad-s <rate> \
  --save-raw-arrays \
  --save-frames \
  --require-depth-completion \
  --require-rl-model \
  --duration-sec 120
```

Pass criteria:

- `binary semantic` window opens and updates.
- Depth window opens and updates.
- If `visualization.save_frames=true`, `experiments/frames/*_binary_semantic.png` and `*_depth.png` are written.
- Terminal prints `act=<id>(<name>)`.
- Terminal prints startup action mapping; default experiment shows `frame=body lateral_sign=-1 act3=W`.
- Terminal prints `obsD`, `obsS`, `p=...`, `yaw`, `yr`, `sync`, `v_body`, `v_ned`.
- Total frame latency is recorded; if it exceeds the budget, optimization or slower control rate is required before flight.

## Gate 9: Controlled Flight Readiness

Do not fly until all are true:

- `python preflight_check.py` exits with `failed=0`.
- `python preflight_check.py --flight-ready` exits with `failed=0`.
- `python pipeline.py --config ./config/experiment_config.yaml --mode ros` does not report `[FlightReady]` failures.
- `python accept_flight_loop_log.py experiments/logs/pipeline.log --require-global-guidance --expected-safe-point <best_center_lat,best_center_lon> --expected-yaw-rate <rate> --max-total-p95-ms 100 --max-halss-p95-ms 70 --max-depth-p95-ms 15 --max-completion-p95-ms 45 --max-rl-p95-ms 30 --max-sync-p95-ms 100 --max-yaw-transform-error 0.15 --max-action-run 60 --require-action-probs` passes after the closed-loop run.
- `python run_field_acceptance.py --expected-yaw-rate <rate> --gis-prior <dir_or_json> --gis-bounds lon_left,lat_bottom,lon_right,lat_top --max-halss-p95-ms 70 --max-depth-p95-ms 15 --max-completion-p95-ms 45 --max-rl-p95-ms 30 --max-sync-p95-ms 100 --max-yaw-transform-error 0.15 --max-flight-action-run 60 --strict-flight-ready` passes, writes `experiments/logs/orin_env.md` and `experiments/logs/field_evidence_status.md`, and prints the unified field acceptance report.
- `python run_acceptance_light.py --gis-prior <dir_or_json> --nocontrol-log experiments/logs/nocontrol.log --frame-dir experiments/frames --require-frames --drl-diagnosis-json experiments/logs/drl_live_frame.json --drl-diagnosis-live --sparsenet-calibration-json experiments/logs/sparsenet_scale.json --depth-projection-cuda-log experiments/logs/depth_projection_cuda.log --orin-env-md experiments/logs/orin_env.md --flight-log experiments/logs/pipeline.log --require-global-guidance --expected-yaw-rate <rate> --max-halss-p95-ms 70 --max-depth-p95-ms 15 --max-completion-p95-ms 45 --max-rl-p95-ms 30 --max-sync-p95-ms 100 --max-yaw-transform-error 0.15 --max-flight-action-run 60 --require-action-probs` produces an overall `PASS` field evidence report.
- `python field_evidence_status.py --strict --validate-artifacts --expected-yaw-rate <rate> --gis-bounds lon_left,lat_bottom,lon_right,lat_top --max-halss-p95-ms 70 --max-depth-p95-ms 15 --max-completion-p95-ms 45 --max-rl-p95-ms 30 --max-sync-p95-ms 100 --max-yaw-transform-error 0.15` reports `required_missing: 0` before the final field evidence report.
- `python audit_experiment_requirements_light.py --strict-field --expected-yaw-rate <rate> --gis-bounds lon_left,lat_bottom,lon_right,lat_top` passes only when every original requirement has accepted field evidence.
- `python test_depth_projection_light.py` passes locally and `python test_depth_projection.py` passes on Orin.
- `python diagnose_drl_policy.py --scan-modes` and a multi-frame `--frame-glob '*_calib_frame.npz'` live run have been saved with results for the chosen observation encoding.
- `python test_live_nocontrol.py --duration-sec 120 ...` runs for at least 2 minutes and exits cleanly.
- Live logs show non-degenerate depth and semantic observations.
- No-control and closed-loop logs pass action mapping checks; default logs do not contain `act=3(E)`.
- No-control and closed-loop logs are not single-action collapsed; long repeated-action runs fail acceptance.
- `uav.yaw_rate_rad_s` is set to the planned yaw-fault rate.
- `uav.action_frame=body` so lateral actions are rotated by current FAST-LIO yaw.
- `uav.action_lateral_sign=-1` unless a deliberate mirrored-action ablation is being tested.
- `observation.depth_norm_mode=meters_div255` and `observation.semantic_norm_mode=gray_unit`.
- The aircraft is restrained or propellers are removed for the first MAVSDK offboard dry-run.

## Current Known Risks To Close

| Risk | Why it matters | Closure evidence |
| --- | --- | --- |
| SparseNet bias missing from converted checkpoint | Dense depth scale can be attenuated | Calibrated `output_scale` and depth error report |
| DeepRL SB2 to SB3 equivalence not formally proven | Policy may prefer one action due to conversion/normalization | `diagnose_drl_policy.py` plus comparison against original SB2 if available |
| Action direction names can be misread if lateral sign is changed | `act=3` is W in original DeepRL, E only in mirrored mode | `uav.action_lateral_sign` logged/checked and `control/action_decomposer.py` tests pass |
| HALSS visualization equality is passthrough until reference image is compared | Requirement says identical HALSS visualization | Save HALSS notebook output and run `compare_halss_visualization.py` or `compare_halss_visualization_light.py` against current `*_binary_semantic.png` |
| GIS guidance disabled by default | Full experiment includes global safe-area guidance | `global_prior.enabled=true` with GPS target or GIS image+bounds |
| Full 30Hz may not be met | Logs showed 60-140ms total previously | Timing log P50/P95 within chosen control budget |
| Some preprocessing is still CPU-bound | HALSS surface-normal generation and depth projection can dominate latency on Orin | No-control `H/D/total` P95 evidence, plus optimization or reduced control frequency if budget fails |
