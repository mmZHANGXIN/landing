# Orin 真机实验现场原子清单

本清单用于 Mid360 + FAST-LIO + HALSS Bayesian + SparseNet + DeepRL + MAVSDK 的偏航失控安全降落实验。执行原则是每一步都留下文件证据，最后用统一验收脚本收口。

## 0. 固定实验参数

原子项：

- 设定本轮偏航角速度 `<yaw_rate>`，单位 rad/s，不能为 `0.0`。
- 使用标定后的 SparseNet `depth_completion.output_scale=<calibrated_scale>`。
- 默认动作映射保持 `uav.action_frame=body` 和 `uav.action_lateral_sign=-1`。
- 二值语义窗口标题键 `binary_semantic_window_title` 固定为 `binary semantic`。
- 观测编码保持 `depth_norm_mode=meters_div255`、`semantic_norm_mode=gray_unit`。
- HALSS/DRL 在线推理必须禁止 CPU fallback：`perception.require_gpu=true`、`decision.require_gpu=true`。
- MAVSDK 必须在全局引导前拿到有效 NED、姿态和非零 GPS home；pipeline 日志必须先记录 `[Pipeline] Home telemetry`，再把 GIS safe point 转为 NED。

验收：

```bash
python check_orin_env.py --strict --require-jetson --out-md experiments/logs/orin_env.md
python accept_orin_env_light.py experiments/logs/orin_env.md
python preflight_check.py --config ./config/experiment_config.yaml --skip-model-load
python test_gpu_fail_closed_contract_light.py
python test_flight_ready_gates.py
python test_mavsdk_home_telemetry_light.py
```

通过标准：

- `experiments/logs/orin_env.md` 写入成功，且 `required_failures: 0`。
- `accept_orin_env_light.py` 通过，确认该报告来自 strict Orin 检查。
- 严格环境检查中 OpenCV 必须能 import，且 `DISPLAY` 已设置，保证 `binary semantic` 和深度图窗口可显示。
- 配置检查无 `[FAIL]`。
- `test_gpu_fail_closed_contract_light.py` 通过，证明 HALSS Bayesian、DRL policy、pipeline、no-control、preflight 和诊断入口不会在真机路径静默 CPU fallback。
- `test_flight_ready_gates.py` 全部通过。
- 默认动作映射启动日志必须是 `Action mapping: frame=body lateral_sign=-1 act3=W` 或 `[Init] Action mapping frame=body lateral_sign=-1 (act3=W)`。
- 默认实验下出现 `act=3(E)` 视为不合格，除非明确做 `action_lateral_sign=+1` 镜像消融。

## 1. 离线 GIS 九宫格全局安全区

原子项：

- 准备 GIS 卫星图和地理边界 `lon_left,lat_bottom,lon_right,lat_top`。
- 使用配置中的分割网络生成语义 mask，或提供已有 `sem_mask_path`。
- 运行九宫格风险评估，保存 `global_prior_*.json`。
- 从最低综合风险格得到 `best_center_gps`，后续作为全局引导目标。

验收：

```bash
python accept_gis_prior_light.py <global_prior_save_dir_or_json> \
  --bounds lon_left,lat_bottom,lon_right,lat_top
```

通过标准：

- JSON 中存在 `risk_grid` 且为 `3x3`。
- `best_cell` 与 `min_risk` 对应风险最低格。
- JSON 必须记录 `source_image_path`、`image_size_px`、`bounds`，并记录 `source_sem_mask_path` 或 `segmentation_source` 以证明语义来源。
- `best_center_gps` 落在 GIS bounds 内。
- 闭环日志中必须先出现可解析的 `[GOTO_SAFE] Target NED: <north>, <east> tolerance=<m>m` 和 `[GOTO_SAFE] Arrived. XY error=<m>m`，到达误差不超过 tolerance，再进入 DRL 下降。

## 2. Mid360 与 FAST-LIO 数据源

原子项：

- 启动 Livox driver。
- 启动 FAST-LIO。
- 保存四个 topic hz 证据文件。

验收：

```bash
timeout 8 ros2 topic hz /livox/lidar 2>&1 | tee experiments/logs/hz_livox_lidar.log
timeout 8 ros2 topic hz /livox/imu 2>&1 | tee experiments/logs/hz_livox_imu.log
timeout 8 ros2 topic hz /cloud_registered 2>&1 | tee experiments/logs/hz_cloud_registered.log
timeout 8 ros2 topic hz /Odometry 2>&1 | tee experiments/logs/hz_odometry.log

python accept_ros_topics_light.py \
  /livox/lidar=experiments/logs/hz_livox_lidar.log \
  /livox/imu=experiments/logs/hz_livox_imu.log \
  /cloud_registered=experiments/logs/hz_cloud_registered.log \
  /Odometry=experiments/logs/hz_odometry.log \
  --require-all
```

通过标准：

- `/livox/lidar` 和 `/livox/imu` 稳定发布。
- `/cloud_registered` 为 FAST-LIO 去畸变点云，稳定发布。
- `/Odometry` 稳定发布，能提供实时位姿和 yaw。

## 3. 下视深度投影几何

原子项：

- 确认 `depth_projection.mode="perspective"`，不能使用 BEV fallback。
- 确认 `depth_projection.backend="torch_cuda"`，真机深度投影和 z-buffer 聚合走 GPU。
- 确认 `fx/fy/cx/cy/R_I_to_C` 与 Mid360/下视相机系约定一致。
- 验证投影公式为 `I_k p_i=R_GI^T(Gp_i-Gp_I)`、`C p_i=R_I_to_C I_k p_i`、透视内参投影、z-buffer 取最近 `Zc`。

验收：

```bash
python test_depth_projection_light.py
python test_depth_projection.py
python test_depth_projection_cuda.py 2>&1 | tee experiments/logs/depth_projection_cuda.log
python accept_depth_projection_cuda_light.py experiments/logs/depth_projection_cuda.log
```

通过标准：

- 轻量测试通过中心点、单点、位姿平移、偏航、`R_I_to_C` 轴映射、z-buffer、背后/越界/最大距离点保持 `dmax`。
- Orin 上真实 `DepthProjector` NumPy 路径通过同样几何契约。
- Orin 上 `torch_cuda` 路径与 NumPy 参考输出一致，且 CUDA 不可用时不得通过；日志保存为 `experiments/logs/depth_projection_cuda.log`。
- 现场 raw arrays 中 `valid_mask` 不是全零，`sparse_depth` 有物理有效值。

## 4. SparseNet 深度补全标定

原子项：

- 在静止平地或已知高度场景采集 raw arrays。
- 用实测雷达到地面距离 `<measured_height>` 标定 `output_scale`。
- 将建议值填入配置或运行命令参数。

验收：

```bash
python test_live_nocontrol.py \
  --save-raw-arrays \
  --save-dir experiments/frames \
  --max-frames 30

python calibrate_sparsenet_scale.py \
  --input experiments/frames/000000_calib_frame.npz \
  --target-depth-m <measured_height> \
  --out-json experiments/logs/sparsenet_scale.json

python accept_sparsenet_calibration_light.py experiments/logs/sparsenet_scale.json \
  --require-improvement
```

通过标准：

- `experiments/logs/sparsenet_scale.json` 存在。
- `suggested_output_scale` 为正数且有限。
- 标定后 median/P90 深度误差满足门槛。
- `depth_completion.output_scale` 不再为 `null`。

## 5. 无飞控全链路实时试验

原子项：

- 连接 Mid360，启动 FAST-LIO，但不连接或不控制飞控。
- 运行完整感知和 DRL 管线 120 秒。
- 保存 raw arrays、二值语义图、深度图和日志。

验收：

```bash
python test_live_nocontrol.py \
  --depth-output-scale <calibrated_scale> \
  --yaw-rate-rad-s <yaw_rate> \
  --save-raw-arrays \
  --save-frames \
  --require-depth-completion \
  --require-rl-model \
  --require-yaw-rate \
  --duration-sec 120 2>&1 | tee experiments/logs/nocontrol.log

python analyze_timing_log.py experiments/logs/nocontrol.log --budget-ms 100 --max-p95-ms 100
python accept_nocontrol_log.py experiments/logs/nocontrol.log \
  --expected-yaw-rate <yaw_rate> \
  --max-total-p95-ms 100 \
  --max-halss-p95-ms 70 \
  --max-depth-p95-ms 15 \
  --max-completion-p95-ms 45 \
  --max-rl-p95-ms 30 \
  --max-sync-p95-ms 100 \
  --max-yaw-transform-error 0.15 \
  --require-action-probs
python diagnose_nocontrol_action_log.py experiments/logs/nocontrol.log \
  --expected-yaw-rate <yaw_rate> \
  --fail-on-issues
python inspect_nocontrol_artifacts_light.py experiments/frames --require-frames \
  --max-sync-ms 100 \
  --max-yaw-transform-error 0.15
```

通过标准：

- 可视化窗口标题为 `binary semantic`，窗口实时更新。
- 保存 `*_binary_semantic.png`、`*_depth.png`、`*_calib_frame.npz`。
- `*_calib_frame.npz` 必须包含 `sparse_depth`、`valid_mask`、`dense_depth`、`sem_map`、`binary_semantic_vis`、`yaw_rad`、`cloud_odom_sync_ms`、`action_id`、`v_body`、`v_ned`。
- 终端逐帧输出 `act=<id>(<name>)`、`H/D/C/RL/total`、`yaw`、`yr`、`sync`、`v_body`、`v_ned`、`obsD`、`obsS`、`valid`、`sem_safe`、`sem_danger`、`p=...`。
- 启动日志输出 `max_cloud_odom_sync_ms=100`；若未设置非零 `<yaw_rate>`，`--require-yaw-rate` 会直接退出。
- `sync` 表示 `/cloud_registered` 与 `/Odometry` header 时间差, P95 不超过 100 ms。
- `v_ned` 必须与 `v_body` 按同帧 FAST-LIO `yaw` 旋转后的结果一致, yaw transform error P95 不超过 0.15 m/s。
- `yr` 均值与 `<yaw_rate>` 一致。
- 默认动作映射下日志不包含 `act=3(E)`。
- 单帧总耗时 P95 不超过 100 ms，或明确降低控制频率并重新验收。
- 如果 `H` 项 P95 接近或超过预算，需要记录 HALSS CPU surface-normal 预处理和 Bayesian UNet 推理的优化计划；如果 `D` 项 P95 过高，需要记录 NumPy 深度投影优化或降频方案。
- 不允许只凭模型在 GPU 上推理就认定全链路 GPU 加速达标；最终以 `analyze_timing_log.py` 的现场 P50/P95/max 为准。

## 6. HALSS 二值安全语义图一致性

原子项：

- 使用 HALSS Bayesian 输出的 `safety_map_vis` 作为显示源。
- 保存当前窗口对应的 `*_binary_semantic.png` 和同帧 `*_calib_frame.npz`。
- 如有原始 notebook 参考图，做像素级比较。

验收：

```bash
python compare_saved_binary_semantic_light.py --frame-dir experiments/frames --grayscale
python compare_halss_visualization_light.py \
  --reference <halss_ref.png> \
  --candidate experiments/frames/000000_binary_semantic.png \
  --grayscale
```

通过标准：

- 保存 PNG 与 NPZ 中 `binary_semantic_vis` 完全一致。
- 与 HALSS 参考可视化 `mean_abs_diff=0` 且 `max_abs_diff=0`，除非预先写明允许阈值。

## 7. DRL 动作塌缩诊断

原子项：

- 对合成观测和实测帧分别运行 DRL 诊断。
- 至少保存 8 个 `*_calib_frame.npz` 实测帧, 再生成 `drl_live_frame.json`。
- 如无飞控日志出现长期单一动作，优先诊断观测分布，再看模型转换和动作映射。

验收：

```bash
python diagnose_drl_policy.py --config ./config/experiment_config.yaml --scan-modes \
  --out-json experiments/logs/drl_synthetic.json \
  --fail-on-collapse

python diagnose_nocontrol_action_log.py experiments/logs/nocontrol.log \
  --expected-yaw-rate <yaw_rate> \
  --fail-on-issues

python diagnose_drl_policy.py --scan-modes \
  --frame-glob 'experiments/frames/*_calib_frame.npz' \
  --out-json experiments/logs/drl_live_frame.json \
  --fail-on-collapse

python accept_drl_diagnosis_light.py experiments/logs/drl_live_frame.json \
  --require-live-frame \
  --require-probs \
  --min-items 8
```

通过标准：

- JSON 中动作映射为 `body/-1`。
- 必须包含 `meters_div255,gray_unit` 编码结果。
- 动作概率存在。
- `diagnose_nocontrol_action_log.py` 不报告 `ACTION_MAPPING_MISMATCH`、`ZERO_YAW_RATE`、`LEGACY_LOG_FORMAT`、`SINGLE_ACTION_COLLAPSE`、`SEMANTIC_ONE_CLASS`。
- 不允许所有测试观测都塌缩为同一动作；如确实塌缩，报告必须能定位是深度、语义、归一化、模型权重还是动作映射问题。

## 8. 闭环前严格门禁

原子项：

- 配置或命令行提供 GIS 来源安全点。
- 提供标定后的 `output_scale`。
- 设置非零 `<yaw_rate>`。
- 不使用 `--allow-incomplete-experiment`。

验收：

```bash
python preflight_check.py --config ./config/experiment_config.yaml --flight-ready

python pipeline.py --config ./config/experiment_config.yaml --mode ros \
  --safe-point <lat>,<lon> \
  --safe-point-source gis \
  --depth-output-scale <calibrated_scale> \
  --yaw-rate-rad-s <yaw_rate> \
  --flight-ready-check-only
```

通过标准：

- `preflight_check.py --flight-ready` 无 `[FAIL]`。
- check-only 日志显示 `[FlightReady] Preview gates passed before model initialization`。
- 任意手工 safe point 未声明 `--safe-point-source gis` 必须失败。

## 9. 闭环飞行日志验收

原子项：

- 首次 Offboard 闭环必须固定无人机或拆桨。
- 运行 pipeline，并保存 `experiments/logs/pipeline.log`。
- 日志必须证明先全局引导到 GIS 最低风险区，再进入 DRL 下降。

验收：

```bash
python pipeline.py --config ./config/experiment_config.yaml --mode ros \
  --safe-point <lat>,<lon> \
  --safe-point-source gis \
  --depth-output-scale <calibrated_scale> \
  --yaw-rate-rad-s <yaw_rate> 2>&1 | tee experiments/logs/pipeline.log

python accept_flight_loop_log.py experiments/logs/pipeline.log \
  --require-global-guidance \
  --expected-safe-point <best_center_lat,best_center_lon> \
  --expected-yaw-rate <yaw_rate> \
  --max-total-p95-ms 100 \
  --max-halss-p95-ms 70 \
  --max-depth-p95-ms 15 \
  --max-completion-p95-ms 45 \
  --max-rl-p95-ms 30 \
  --max-sync-p95-ms 100 \
  --max-yaw-transform-error 0.15 \
  --max-action-run 60 \
  --require-action-probs
```

通过标准：

- 日志含 `[FlightReady] Strict experiment gates passed.`。
- 日志含 `[Pipeline] FAST-LIO ready.`。
- `[GOTO_SAFE] Target NED` 和 `[GOTO_SAFE] Arrived` 出现在 DRL 帧之前。
- `GOTO_SAFE` 目标 NED 有限且非零，到达 `XY error` 不超过日志中的 tolerance。
- pipeline 日志中的 `[GlobalPrior] Safe point` 与 `[Pipeline] Global guidance target` 必须匹配 GIS JSON 的 `best_center_gps`。
- DRL 帧包含 `yaw`、`yaw_sp`、`yr`、`sync`、`v_body`、`v_ned`、`obsD`、`obsS`、`p=...`。
- `sync` P95 不超过 100 ms, 否则不能证明点云和 FAST-LIO 位姿同刻解算。
- `v_ned` 必须与 `v_body` 按同帧 FAST-LIO `yaw` 旋转后的结果一致, yaw transform error P95 不超过 0.15 m/s。
- yaw setpoint 随 `<yaw_rate>` 推进。
- 默认实验映射下闭环日志同样拒绝 `act=3(E)`。
- 闭环 DRL 帧不能全部为同一动作，且最长连续重复动作不超过 `--max-action-run`。
- 若闭环出现 `DRL action collapse`，保存的诊断 NPZ 必须包含 `binary_semantic_vis`、`cloud_odom_sync_ms`、`yaw_rad`、`action_id`、`v_body`、`v_ned`，用于复查 HALSS 显示一致性、FAST-LIO 同步和 yaw 速度解算。

## 10. 最终统一验收

原子项：

- 检查所有现场证据是否齐全。
- 跑统一验收。
- 保存最终 `acceptance_light_*.md/json` 报告。

验收：

```bash
python field_evidence_status.py --strict --validate-artifacts \
  --expected-yaw-rate <yaw_rate> \
  --gis-bounds lon_left,lat_bottom,lon_right,lat_top \
  --max-halss-p95-ms 70 \
  --max-depth-p95-ms 15 \
  --max-completion-p95-ms 45 \
  --max-rl-p95-ms 30 \
  --max-sync-p95-ms 100 \
  --max-yaw-transform-error 0.15 \
  --max-flight-action-run 60
python field_evidence_status.py --strict --validate-artifacts \
  --expected-yaw-rate <yaw_rate> \
  --gis-bounds lon_left,lat_bottom,lon_right,lat_top \
  --max-halss-p95-ms 70 \
  --max-depth-p95-ms 15 \
  --max-completion-p95-ms 45 \
  --max-rl-p95-ms 30 \
  --max-sync-p95-ms 100 \
  --max-yaw-transform-error 0.15 \
  --max-flight-action-run 60 \
  --out-md experiments/logs/field_evidence_status.md

python run_field_acceptance.py \
  --expected-yaw-rate <yaw_rate> \
  --gis-prior <global_prior_save_dir_or_json> \
  --gis-bounds lon_left,lat_bottom,lon_right,lat_top \
  --max-halss-p95-ms 70 \
  --max-depth-p95-ms 15 \
  --max-completion-p95-ms 45 \
  --max-rl-p95-ms 30 \
  --max-sync-p95-ms 100 \
  --max-yaw-transform-error 0.15 \
  --max-flight-action-run 60 \
  --strict-flight-ready

python audit_experiment_requirements_light.py --strict-field \
  --expected-yaw-rate <yaw_rate> \
  --gis-bounds lon_left,lat_bottom,lon_right,lat_top
```

通过标准：

- `field_evidence_status.py --strict --validate-artifacts` 输出 `required_missing: 0`，且所有已存在证据均通过对应 accept 脚本。
- `depth_projection_cuda` 证据项为 `PASS`，`experiments/logs/depth_projection_cuda.log` 通过 `accept_depth_projection_cuda_light.py`。
- `audit_experiment_requirements_light.py --strict-field` 输出所有原始需求为 `PASS`，不再出现 `FIELD_REQUIRED` 或 `LOCAL_FAIL`。
- 现场证据清单包含 `orin_env.md`，且环境报告必须由 Orin 上的 strict 检查生成。
- `run_field_acceptance.py` 会默认生成 `experiments/logs/field_evidence_status.md`，记录每个证据项的 PASS/MISS、生产命令和验收命令。
- 最终 `acceptance_light_*.md` 中包含 `saved strict Orin environment evidence` 行。
- `--skip-env-check` / `--skip-orin-env` 只允许本地或台架调试使用，真机最终验收不得使用。
- `run_field_acceptance.py` 最终输出 `[FIELD] Acceptance complete`。
- 最新 `experiments/logs/acceptance_light_*.md/json` 总体为 `PASS`。
