# Lightweight Acceptance Report

- generated_at: `2026-06-07T15:59:37`
- root: `/Users/evelyn/Desktop/orinlanding/orinlanding`
- overall: `PASS`
- strict_flight_ready: `False`

| Check | Status | Return | ms | Reason |
| --- | --- | ---: | ---: | --- |
| preflight config/assets without model load | PASS | 0 | 30.9 | command exited 0 |
| DRL SB3 metadata | PASS | 0 | 23.7 | command exited 0 |
| GIS nine-grid lightweight contract | PASS | 0 | 17.5 | command exited 0 |
| GIS global-prior JSON acceptance self-test | PASS | 0 | 64.4 | command exited 0 |
| HALSS visualization lightweight contract | PASS | 0 | 25.3 | command exited 0 |
| visualization source contract | PASS | 0 | 15.3 | command exited 0 |
| depth projection lightweight contract | PASS | 0 | 12.4 | command exited 0 |
| SparseNet sparse convolution lightweight contract | PASS | 0 | 11.4 | command exited 0 |
| yaw-aware action decomposition | PASS | 0 | 15.8 | command exited 0 |
| runtime action-name labels | PASS | 0 | 22.3 | command exited 0 |
| field runbook source contract | PASS | 0 | 16.0 | command exited 0 |
| field evidence status self-test | PASS | 0 | 447.4 | command exited 0 |
| Orin environment check self-test | PASS | 0 | 87.1 | command exited 0 |
| Orin environment report acceptance self-test | PASS | 0 | 95.9 | command exited 0 |
| requirement traceability source contract | PASS | 0 | 15.2 | command exited 0 |
| field acceptance wrapper self-test | PASS | 0 | 57.0 | command exited 0 |
| timing log acceptance self-test | PASS | 0 | 62.4 | command exited 0 |
| no-control log acceptance self-test | PASS | 0 | 170.6 | command exited 0 |
| no-control artifact acceptance self-test | PASS | 0 | 72.3 | command exited 0 |
| DRL diagnosis JSON acceptance self-test | PASS | 0 | 83.0 | command exited 0 |
| SparseNet calibration JSON acceptance self-test | PASS | 0 | 103.0 | command exited 0 |
| saved binary semantic NPZ/PNG comparison self-test | PASS | 0 | 75.7 | command exited 0 |
| ROS topic evidence acceptance self-test | PASS | 0 | 60.1 | command exited 0 |
| flight-loop log acceptance self-test | PASS | 0 | 148.3 | command exited 0 |
| strict flight-ready gate regression | PASS | 0 | 714.3 | command exited 0 |
| flight-ready preflight preview | PASS | 1 | 25.9 | flight-ready preview has only expected field-configuration gaps |
| pipeline flight-ready check-only with GIS override | PASS | 0 | 36.8 | command exited 0 |

## Command Output

### preflight config/assets without model load

```text
$ /Library/Developer/CommandLineTools/usr/bin/python3 /Users/evelyn/Desktop/orinlanding/orinlanding/preflight_check.py --config /Users/evelyn/Desktop/orinlanding/orinlanding/config/experiment_config.yaml --skip-model-load
[OK] Config loaded: /Users/evelyn/Desktop/orinlanding/orinlanding/config/experiment_config.yaml
[OK] Config key present: runtime
[OK] Config key present: observation.img_width
[OK] Config key present: observation.img_height
[OK] Config key present: perception.halss_weight_path
[OK] Config key present: perception.halss_backend
[OK] Config key present: depth_projection.mode
[OK] Config key present: depth_projection.fx
[OK] Config key present: depth_projection.fy
[OK] Config key present: depth_projection.cx
[OK] Config key present: depth_projection.cy
[OK] Config key present: depth_projection.R_I_to_C
[OK] Config key present: depth_completion.backend
[OK] Config key present: depth_completion.weight_path
[OK] Config key present: depth_completion.input_encoding
[OK] Config key present: decision.policy_weights_path
[OK] Config key present: uav.action_frame
[OK] Config key present: uav.action_lateral_sign
[OK] Config key present: uav.yaw_rate_rad_s
[OK] Config key present: visualization.binary_semantic_window_title
[WARN] depth_completion.output_scale is null; converted SparseNet checkpoint is known to be attenuated until calibrated
[OK] HALSS weight exists: /Users/evelyn/Desktop/orinlanding/orinlanding/arch/3.UDPDirect30Hz_cyd_final/HALO-master (2)/HALSS/network_utils/unet_epoch6.pth
[OK] SparseNet weight exists: /Users/evelyn/Desktop/orinlanding/orinlanding/weights/sparsenet.pth
[OK] DRL policy exists: /Users/evelyn/Desktop/orinlanding/orinlanding/weights/last_step_model_sb3.zip
[WARN] global_prior.enabled=false; pipeline will skip GIS safe-area guidance
[OK] DRL metadata readable: shape=(128, 128, 2) normalize_images=False
[WARN] Skipping depth projection geometry: missing dependency (No module named 'numpy')
[OK] Action decomposer frame=body lateral_sign=-1 yaw compensation passed
[OK] Configured yaw_rate_rad_s=0.000
[WARN] Skipping CUDA check: torch import failed (No module named 'torch')

Preflight summary: failed=0, warnings=4
```

### DRL SB3 metadata

```text
$ /Library/Developer/CommandLineTools/usr/bin/python3 /Users/evelyn/Desktop/orinlanding/orinlanding/inspect_drl_model.py /Users/evelyn/Desktop/orinlanding/orinlanding/weights/last_step_model_sb3.zip
path: /Users/evelyn/Desktop/orinlanding/orinlanding/weights/last_step_model_sb3.zip
exists: True  zip: True
shape: (128, 128, 2)
low/high: 0.0 / 1.0
normalize_images: False
features_extractor: <class '__main__.SB2CNN'>
net_arch: {'pi': [64, 64], 'vf': [64, 32]}
```

### GIS nine-grid lightweight contract

```text
$ /Library/Developer/CommandLineTools/usr/bin/python3 /Users/evelyn/Desktop/orinlanding/orinlanding/test_gis_nine_grid_light.py
=== Lightweight GIS nine-grid acceptance ===
  OK risk_grid shape=3x3
  OK best_cell=(1,1) risk=0.0 center_px=(150,150)
  OK best_center_gps=(22.730000, 113.910000)
=== ALL PASSED ===
```

### GIS global-prior JSON acceptance self-test

```text
$ /Library/Developer/CommandLineTools/usr/bin/python3 /Users/evelyn/Desktop/orinlanding/orinlanding/test_accept_gis_prior_light.py
=== Lightweight GIS global-prior acceptance ===
  OK valid GIS JSON passes
  OK invalid risk/GPS JSON fails
=== ALL PASSED ===
```

### HALSS visualization lightweight contract

```text
$ /Library/Developer/CommandLineTools/usr/bin/python3 /Users/evelyn/Desktop/orinlanding/orinlanding/test_halss_visualization_light.py
=== Lightweight HALSS visualization acceptance ===
  OK exact PGM match passes
  OK one-pixel mismatch fails strict gate
  OK RGB PNG grayscale conversion
  OK nearest-neighbor resize
=== ALL PASSED ===
```

### visualization source contract

```text
$ /Library/Developer/CommandLineTools/usr/bin/python3 /Users/evelyn/Desktop/orinlanding/orinlanding/test_visualization_contract_light.py
=== Lightweight visualization contract acceptance ===
  OK no-control window title is binary semantic
  OK config declares binary semantic window title
  OK no-control semantic/depth imshow use nearest interpolation
  OK CV2 visualizer uses HALSS passthrough with nearest resize
=== ALL PASSED ===
```

### depth projection lightweight contract

```text
$ /Library/Developer/CommandLineTools/usr/bin/python3 /Users/evelyn/Desktop/orinlanding/orinlanding/test_depth_projection_light.py
=== Lightweight depth projection acceptance ===
  OK center pixel depth=5m
  OK z-buffer keeps nearest point
  OK yaw=90deg maps world east to body forward camera row
=== ALL PASSED ===
```

### SparseNet sparse convolution lightweight contract

```text
$ /Library/Developer/CommandLineTools/usr/bin/python3 /Users/evelyn/Desktop/orinlanding/orinlanding/test_sparse_conv_light.py
=== Lightweight sparse convolution acceptance ===
  OK inverse depth encoding/decoding
  OK sparse conv divides by valid count
  OK all-zero mask remains finite
  OK mask max-pool expands valid region
  OK output_scale preserves inverse-depth direction
=== ALL PASSED ===
```

### yaw-aware action decomposition

```text
$ /Library/Developer/CommandLineTools/usr/bin/python3 /Users/evelyn/Desktop/orinlanding/orinlanding/control/action_decomposer.py
=== ActionDecomposer Unit Tests ===
  ✅ yaw=0: forward=N → v_ned=[1,0,0]
  ✅ yaw=0: DeepRL action 7(E) → v_ned=[0,1,0]
  ✅ yaw=90°: forward=E → v_ned=[0,1,0]
  ✅ yaw=90°: DeepRL action 3(W) → v_ned=[1,0,0]
  ✅ yaw=180°: forward=S → v_ned=[-1,0,0]
  ✅ descend: z preserved, yaw_rate=0.5
  ✅ hover: all zeros
  ✅ hover+yaw: velocity zero, yaw_rate=0.3
  ✅ action_frame=ned: N action remains world-N at yaw=90°
  ✅ DeepRL default: act=3 is W, v_body=[0,-1,0]
  ✅ mirror mode: act=3(E), yaw=10° -> v_ned≈[-0.17,0.98,0]
=== ALL PASSED ===
```

### runtime action-name labels

```text
$ /Library/Developer/CommandLineTools/usr/bin/python3 /Users/evelyn/Desktop/orinlanding/orinlanding/test_action_names_light.py
=== Lightweight action-name acceptance ===
  OK default action_lateral_sign=-1: act3=W, act7=E
  OK mirror action_lateral_sign=+1: act3=E, act7=W
  OK DRL diagnostic top-prob labels use runtime names
=== ALL PASSED ===
```

### field runbook source contract

```text
$ /Library/Developer/CommandLineTools/usr/bin/python3 /Users/evelyn/Desktop/orinlanding/orinlanding/test_runbook_contract_light.py
=== Lightweight runbook contract acceptance ===
  OK no-control command saves raw arrays, frames, timing, probabilities, and yaw-rate evidence
  OK unified field report command includes GIS, Mid360/FAST-LIO topics, no-control frames, and yaw evidence
  OK GIS check-only limitation and closed-loop global-guidance gates are documented
  OK realtime claims are tied to measured P95 evidence
  OK action mapping and remaining field gaps are explicit
  OK field atomic checklist tracks the operator sequence
=== ALL PASSED ===
```

### field evidence status self-test

```text
$ /Library/Developer/CommandLineTools/usr/bin/python3 /Users/evelyn/Desktop/orinlanding/orinlanding/test_field_evidence_status_light.py
=== Lightweight field evidence status acceptance ===
  OK complete evidence passes strict status
  OK missing evidence fails strict status with hints
  OK markdown evidence report is written
  OK non-strict Orin environment report fails strict status
  OK skip-orin-env allows bench evidence status
  OK validate-artifacts rejects placeholder evidence files
=== ALL PASSED ===
```

### Orin environment check self-test

```text
$ /Library/Developer/CommandLineTools/usr/bin/python3 /Users/evelyn/Desktop/orinlanding/orinlanding/test_check_orin_env_light.py
=== Lightweight Orin environment check ===
  OK non-strict environment scan writes Markdown
  OK require-jetson fails on non-Jetson hosts
  OK strict DISPLAY requirement is documented
=== ALL PASSED ===
```

### Orin environment report acceptance self-test

```text
$ /Library/Developer/CommandLineTools/usr/bin/python3 /Users/evelyn/Desktop/orinlanding/orinlanding/test_accept_orin_env_light.py
=== Lightweight Orin environment acceptance ===
  OK strict report passes
  OK non-strict report fails
  OK CUDA failure report fails
  OK missing MAVSDK report row fails
=== ALL PASSED ===
```

### requirement traceability source contract

```text
$ /Library/Developer/CommandLineTools/usr/bin/python3 /Users/evelyn/Desktop/orinlanding/orinlanding/test_requirement_traceability_light.py
=== Lightweight requirement traceability acceptance ===
  OK closed_loop_flight_readiness
  OK drl_policy_inference_and_action_output
  OK global_guidance_before_drl
  OK gpu_realtime_single_frame_budget
  OK halss_bayesian_binary_semantic
  OK halss_visualization_pixel_identity
  OK live_no_control_visual_output
  OK mid360_fastlio_deskewed_cloud_and_pose
  OK offline_gis_segmentation_and_nine_grid
  OK perspective_depth_projection
  OK sparsity_invariant_depth_completion
  OK yaw_fault_body_to_ned_action_decomposition
=== ALL PASSED ===
```

### field acceptance wrapper self-test

```text
$ /Library/Developer/CommandLineTools/usr/bin/python3 /Users/evelyn/Desktop/orinlanding/orinlanding/test_run_field_acceptance_light.py
=== Lightweight field acceptance wrapper ===
  OK dry-run prints full evidence status and unified acceptance commands
  OK dry-run can skip Orin environment check for bench debugging
=== ALL PASSED ===
```

### timing log acceptance self-test

```text
$ /Library/Developer/CommandLineTools/usr/bin/python3 /Users/evelyn/Desktop/orinlanding/orinlanding/test_analyze_timing_log_light.py
=== Lightweight timing log acceptance ===
  OK good timing log passes
  OK short/slow timing log fails
=== ALL PASSED ===
```

### no-control log acceptance self-test

```text
$ /Library/Developer/CommandLineTools/usr/bin/python3 /Users/evelyn/Desktop/orinlanding/orinlanding/test_accept_nocontrol_log_light.py
=== Lightweight no-control log acceptance ===
  OK good log passes
  OK zero-yaw single-action log fails
  OK mirrored action labels fail unless explicitly requested
  OK missing startup action mapping log fails
  OK module-level timing budget fails independently of total P95
  OK legacy act=3(E) logs get specific diagnosis
=== ALL PASSED ===
```

### no-control artifact acceptance self-test

```text
$ /Library/Developer/CommandLineTools/usr/bin/python3 /Users/evelyn/Desktop/orinlanding/orinlanding/test_inspect_nocontrol_artifacts_light.py
=== Lightweight no-control artifact acceptance ===
  OK good artifacts pass
  OK missing key fails
=== ALL PASSED ===
```

### DRL diagnosis JSON acceptance self-test

```text
$ /Library/Developer/CommandLineTools/usr/bin/python3 /Users/evelyn/Desktop/orinlanding/orinlanding/test_accept_drl_diagnosis_light.py
=== Lightweight DRL diagnosis JSON acceptance ===
  OK good report passes
  OK collapsed report fails
  OK action-name mismatch fails
=== ALL PASSED ===
```

### SparseNet calibration JSON acceptance self-test

```text
$ /Library/Developer/CommandLineTools/usr/bin/python3 /Users/evelyn/Desktop/orinlanding/orinlanding/test_accept_sparsenet_calibration_light.py
=== Lightweight SparseNet calibration JSON acceptance ===
  OK good report passes
  OK too-large scale fails
  OK too-few pixels fail
  OK excessive scaled error fails
=== ALL PASSED ===
```

### saved binary semantic NPZ/PNG comparison self-test

```text
$ /Library/Developer/CommandLineTools/usr/bin/python3 /Users/evelyn/Desktop/orinlanding/orinlanding/test_compare_saved_binary_semantic_light.py
=== Lightweight saved binary semantic comparison ===
  OK matching NPZ/PNG passes
  OK mismatched NPZ/PNG fails
=== ALL PASSED ===
```

### ROS topic evidence acceptance self-test

```text
$ /Library/Developer/CommandLineTools/usr/bin/python3 /Users/evelyn/Desktop/orinlanding/orinlanding/test_accept_ros_topics_light.py
=== Lightweight ROS topic acceptance ===
  OK good topic logs pass
  OK low-rate/missing-topic logs fail
=== ALL PASSED ===
```

### flight-loop log acceptance self-test

```text
$ /Library/Developer/CommandLineTools/usr/bin/python3 /Users/evelyn/Desktop/orinlanding/orinlanding/test_accept_flight_loop_log_light.py
=== Lightweight flight-loop log acceptance ===
  OK good GIS-to-DRL log passes
  OK bypass/zero-yaw log fails
  OK mirrored action labels fail by default
  OK missing startup action mapping log fails
  OK nonzero-yaw single-action log fails
  OK module-level timing budget fails independently of total P95
=== ALL PASSED ===
```

### strict flight-ready gate regression

```text
$ /Library/Developer/CommandLineTools/usr/bin/python3 /Users/evelyn/Desktop/orinlanding/orinlanding/test_flight_ready_gates.py
=== FlightReady gate regression tests ===
  OK fail: default config
  OK fail: bad safe-point
  OK pass: GIS safe-point override
  OK fail: manual safe-point
  OK fail: mirrored action sign
  OK fail: NED action frame
  OK fail: bad observation encoding
  OK fail: runtime GPU
  OK fail: HALSS backend
  OK fail: depth projection mode
  OK fail: SparseNet input encoding
  OK fail: visualization enable
  OK fail: binary semantic display
  OK fail: depth display
  OK fail: binary semantic title
  OK fail: missing GIS image
  OK pass: GIS override
  OK fail: check-only bypass
=== ALL FlightReady gate tests PASSED ===
```

### flight-ready preflight preview

```text
$ /Library/Developer/CommandLineTools/usr/bin/python3 /Users/evelyn/Desktop/orinlanding/orinlanding/preflight_check.py --config /Users/evelyn/Desktop/orinlanding/orinlanding/config/experiment_config.yaml --flight-ready --skip-model-load
[OK] Config loaded: /Users/evelyn/Desktop/orinlanding/orinlanding/config/experiment_config.yaml
[OK] Config key present: runtime
[OK] Config key present: observation.img_width
[OK] Config key present: observation.img_height
[OK] Config key present: perception.halss_weight_path
[OK] Config key present: perception.halss_backend
[OK] Config key present: depth_projection.mode
[OK] Config key present: depth_projection.fx
[OK] Config key present: depth_projection.fy
[OK] Config key present: depth_projection.cx
[OK] Config key present: depth_projection.cy
[OK] Config key present: depth_projection.R_I_to_C
[OK] Config key present: depth_completion.backend
[OK] Config key present: depth_completion.weight_path
[OK] Config key present: depth_completion.input_encoding
[OK] Config key present: decision.policy_weights_path
[OK] Config key present: uav.action_frame
[OK] Config key present: uav.action_lateral_sign
[OK] Config key present: uav.yaw_rate_rad_s
[OK] Config key present: visualization.binary_semantic_window_title
[WARN] depth_completion.output_scale is null; converted SparseNet checkpoint is known to be attenuated until calibrated
[OK] HALSS weight exists: /Users/evelyn/Desktop/orinlanding/orinlanding/arch/3.UDPDirect30Hz_cyd_final/HALO-master (2)/HALSS/network_utils/unet_epoch6.pth
[OK] SparseNet weight exists: /Users/evelyn/Desktop/orinlanding/orinlanding/weights/sparsenet.pth
[OK] DRL policy exists: /Users/evelyn/Desktop/orinlanding/orinlanding/weights/last_step_model_sb3.zip
[WARN] global_prior.enabled=false; pipeline will skip GIS safe-area guidance
[FAIL] flight-ready gate: global_prior.enabled=false; GIS nine-grid guidance would be skipped before DRL descent
[FAIL] flight-ready gate: depth_completion.output_scale is null; calibrate SparseNet scale before closing flight loop
[FAIL] flight-ready gate: uav.yaw_rate_rad_s is 0.0; set the planned yaw-fault rate
[OK] flight-ready gate: runtime GPU execution configured
[OK] flight-ready gate: HALSS Bayesian UNet backend ...<truncated>
```

### pipeline flight-ready check-only with GIS override

```text
$ /Library/Developer/CommandLineTools/usr/bin/python3 /Users/evelyn/Desktop/orinlanding/orinlanding/pipeline.py --config /Users/evelyn/Desktop/orinlanding/orinlanding/config/experiment_config.yaml --mode ros --safe-point 31.0,121.0 --safe-point-source gis --depth-output-scale 40 --yaw-rate-rad-s 0.35 --flight-ready-check-only
2026-06-07 15:59:37,683 [INFO] OrinLanding: [FlightReady] Preview gates passed before model initialization
2026-06-07 15:59:37,683 [INFO] OrinLanding: [FlightReady] Check-only mode complete.
```
