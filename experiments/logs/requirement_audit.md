# Experiment Requirement Audit

| Requirement | Status | Missing local evidence | Missing field evidence | Invalid field evidence |
| --- | --- | --- | --- | --- |
| offline_gis_nine_grid | FIELD_REQUIRED | - | gis_prior | - |
| global_guidance_before_drl | FIELD_REQUIRED | - | pipeline_log | - |
| mid360_fastlio_deskewed_cloud_pose | FIELD_REQUIRED | - | ros_livox_lidar, ros_livox_imu, ros_cloud_registered, ros_odometry | - |
| halss_bayesian_binary_semantic | FIELD_REQUIRED | - | nocontrol_log, binary_semantic_frames | - |
| halss_visual_identity | FIELD_REQUIRED | - | raw_arrays, binary_semantic_frames | - |
| perspective_depth_projection | FIELD_REQUIRED | - | nocontrol_log, raw_arrays | - |
| sparsity_invariant_depth_completion | FIELD_REQUIRED | - | sparsenet_scale, raw_arrays, depth_frames | - |
| drl_policy_inference | FIELD_REQUIRED | - | drl_diagnosis, nocontrol_log, pipeline_log | - |
| yaw_fault_action_decomposition | FIELD_REQUIRED | - | nocontrol_log, pipeline_log | - |
| realtime_gpu_budget | FIELD_REQUIRED | - | orin_env, nocontrol_log, pipeline_log | orin_env |
| live_visual_outputs | FIELD_REQUIRED | - | binary_semantic_frames, depth_frames, raw_arrays | - |
| closed_loop_field_readiness | FIELD_REQUIRED | - | orin_env, gis_prior, ros_livox_lidar, ros_livox_imu, ros_cloud_registered, ros_odometry, nocontrol_log, raw_arrays, binary_semantic_frames, depth_frames, drl_diagnosis, sparsenet_scale, pipeline_log | orin_env |
