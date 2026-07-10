# Lightweight Acceptance Report

- generated_at: `2026-06-07T13:29:59`
- root: `/Users/evelyn/Desktop/orinlanding/orinlanding`
- overall: `PASS`
- strict_flight_ready: `False`

| Check | Status | Return | ms | Reason |
| --- | --- | ---: | ---: | --- |
| preflight config/assets without model load | PASS | 0 | 32.3 | command exited 0 |
| DRL SB3 metadata | PASS | 0 | 22.5 | command exited 0 |
| GIS nine-grid lightweight contract | PASS | 0 | 17.3 | command exited 0 |
| HALSS visualization lightweight contract | PASS | 0 | 29.9 | command exited 0 |
| visualization source contract | PASS | 0 | 15.6 | command exited 0 |
| depth projection lightweight contract | PASS | 0 | 12.5 | command exited 0 |
| SparseNet sparse convolution lightweight contract | PASS | 0 | 10.8 | command exited 0 |
| yaw-aware action decomposition | PASS | 0 | 14.9 | command exited 0 |
| runtime action-name labels | PASS | 0 | 23.9 | command exited 0 |
| no-control log acceptance self-test | PASS | 0 | 61.7 | command exited 0 |
| no-control artifact acceptance self-test | PASS | 0 | 73.1 | command exited 0 |
| strict flight-ready gate regression | PASS | 0 | 749.4 | command exited 0 |
| flight-ready preflight preview | PASS | 1 | 27.7 | flight-ready preview has only expected field-configuration gaps |
| pipeline flight-ready check-only with GIS override | PASS | 0 | 39.1 | command exited 0 |

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

### no-control log acceptance self-test

```text
$ /Library/Developer/CommandLineTools/usr/bin/python3 /Users/evelyn/Desktop/orinlanding/orinlanding/test_accept_nocontrol_log_light.py
=== Lightweight no-control log acceptance ===
  OK good log passes
  OK zero-yaw single-action log fails
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
[WARN] depth_completion.output_scale is null; converted SparseNet checkpoint is known to be attenuated until calibrated
[OK] HALSS weight exists: /Users/evelyn/Desktop/orinlanding/orinlanding/arch/3.UDPDirect30Hz_cyd_final/HALO-master (2)/HALSS/network_utils/unet_epoch6.pth
[OK] SparseNet weight exists: /Users/evelyn/Desktop/orinlanding/orinlanding/weights/sparsenet.pth
[OK] DRL policy exists: /Users/evelyn/Desktop/orinlanding/orinlanding/weights/last_step_model_sb3.zip
[WARN] global_prior.enabled=false; pipeline will skip GIS safe-area guidance
[FAIL] flight-ready gate: global_prior.enabled=false; GIS nine-grid guidance would be skipped before DRL descent
[FAIL] flight-ready gate: depth_completion.output_scale is null; calibrate SparseNet scale before closing flight loop
[FAIL] flight-ready gate: uav.yaw_rate_rad_s is 0.0; set the planned yaw-fault rate
[OK] flight-ready gate: runtime GPU execution configured
[OK] flight-ready gate: HALSS Bayesian UNet backend configured
[OK] flight-ready gate: perspective depth projection conf...<truncated>
```

### pipeline flight-ready check-only with GIS override

```text
$ /Library/Developer/CommandLineTools/usr/bin/python3 /Users/evelyn/Desktop/orinlanding/orinlanding/pipeline.py --config /Users/evelyn/Desktop/orinlanding/orinlanding/config/experiment_config.yaml --mode ros --safe-point 31.0,121.0 --safe-point-source gis --depth-output-scale 40 --yaw-rate-rad-s 0.35 --flight-ready-check-only
2026-06-07 13:29:59,015 [INFO] OrinLanding: [FlightReady] Preview gates passed before model initialization
2026-06-07 13:29:59,015 [INFO] OrinLanding: [FlightReady] Check-only mode complete.
```
