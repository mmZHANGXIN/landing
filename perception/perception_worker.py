"""Spawn-safe isolated perception worker for the live landing pipeline.

The module deliberately imports CUDA/model dependencies only inside the child
entry point.  The parent process therefore never creates a CUDA context and a
hung SciPy/PyTorch/ONNX native call can be recovered by terminating the child.
"""

from __future__ import annotations

import os
import queue
import time
import traceback


def _stage_write(fd: int, generation: int, cloud_seq: int, stage: str,
                 event: str, elapsed_ms=None) -> None:
    elapsed = "" if elapsed_ms is None else f" elapsed_ms={float(elapsed_ms):.3f}"
    line = (
        f"monotonic_s={time.perf_counter():.6f} generation={generation} "
        f"cloud_seq={cloud_seq} stage={stage} event={event}{elapsed}\n"
    )
    os.write(fd, line.encode("utf-8", errors="replace"))


def _put_latest(target_queue, value) -> None:
    """Best-effort latest-only put for result/status queues."""
    try:
        target_queue.put_nowait(value)
        return
    except queue.Full:
        pass
    try:
        target_queue.get_nowait()
    except queue.Empty:
        pass
    try:
        target_queue.put_nowait(value)
    except queue.Full:
        pass


def perception_worker_main(cfg, onnx_model_path: str, generation: int,
                           request_queue, result_queue, status_queue,
                           stage_log_path: str) -> None:
    """Load all models in the child and process immutable perception jobs."""
    stage_fd = os.open(
        stage_log_path,
        os.O_WRONLY | os.O_CREAT | os.O_APPEND,
        0o644,
    )
    try:
        # Lazy imports are essential: multiprocessing spawn imports this module
        # in a pristine interpreter before this function is called.
        import cv2
        import numpy as np
        import onnxruntime as ort
        import pipeline as pipeline_module
        # pipeline keeps heavy imports lazy for preflight.  Populate only the
        # three globals used by its pure perception helpers; do not initialize
        # ROS/MAVROS in the child.
        pipeline_module.np = np
        pipeline_module.cv2 = cv2
        pipeline_module.ort = ort
        ONNXDRL = pipeline_module.ONNXDRL
        make_binary_semantic_vis = pipeline_module.make_binary_semantic_vis
        project_bev_depth = pipeline_module.project_bev_depth
        render_sparse_depth = pipeline_module.render_sparse_depth
        from perception.halss_bayesian import HALSSBayesianEvaluator
        from perception.semantic_generator import SemanticGenerator
        from perception.training_camera_projection import (
            TrainingCameraModel,
            project_training_camera,
        )

        obs_cfg = cfg["observation"]
        perc_cfg = cfg["perception"]
        depth_cfg = cfg["depth_projection"]
        obs_h = int(obs_cfg.get("img_height", 128))
        obs_w = int(obs_cfg.get("img_width", 128))
        depth_max = float(depth_cfg.get("max_range", 30.0))
        safe_id = int(perc_cfg.get("safe_class_id", 1))
        danger_id = int(perc_cfg.get("danger_class_id", 9))
        projection_mode = str(depth_cfg.get("mode", "training_camera")).lower()
        training_camera = TrainingCameraModel.from_config(
            depth_cfg.get("training_camera", {}),
            output_width=obs_w,
            output_height=obs_h,
            far_m=depth_max,
        )

        _stage_write(stage_fd, generation, -1, "WORKER_INIT", "START")
        init_start = time.perf_counter()
        halss_init_start = time.perf_counter()
        _stage_write(stage_fd, generation, -1, "HALSS_INIT", "START")
        halss = HALSSBayesianEvaluator(perc_cfg)
        _stage_write(
            stage_fd, generation, -1, "HALSS_INIT", "END",
            (time.perf_counter() - halss_init_start) * 1000.0,
        )
        semantic_init_start = time.perf_counter()
        _stage_write(stage_fd, generation, -1, "SEMANTIC_INIT", "START")
        sem_gen = SemanticGenerator({**perc_cfg, "img_width": obs_w, "img_height": obs_h})
        _stage_write(
            stage_fd, generation, -1, "SEMANTIC_INIT", "END",
            (time.perf_counter() - semantic_init_start) * 1000.0,
        )
        onnx_init_start = time.perf_counter()
        _stage_write(stage_fd, generation, -1, "ONNX_INIT", "START")
        drl = ONNXDRL(
            onnx_model_path,
            obs_h=obs_h,
            obs_w=obs_w,
            dmax=depth_max,
            depth_norm_mode=str(obs_cfg.get("depth_norm_mode", "raw_meters_graph_scaled")),
            semantic_norm_mode=str(obs_cfg.get("semantic_norm_mode", "raw_gray_graph_scaled")),
            require_gpu=bool(cfg.get("decision", {}).get("require_gpu", True)),
            allow_cpu_fallback_if_no_gpu_ep=bool(
                cfg.get("decision", {}).get(
                    "allow_cpu_fallback_if_no_gpu_ep", False,
                )
            ),
        )
        _stage_write(
            stage_fd, generation, -1, "ONNX_INIT", "END",
            (time.perf_counter() - onnx_init_start) * 1000.0,
        )
        _stage_write(
            stage_fd, generation, -1, "WORKER_INIT", "END",
            (time.perf_counter() - init_start) * 1000.0,
        )
        _put_latest(status_queue, {
            "kind": "ready", "generation": generation,
            "monotonic_s": time.perf_counter(),
            "onnx_providers": list(drl.active_providers),
        })

        while True:
            job = request_queue.get()
            if job is None:
                break
            cloud_seq = int(job["cloud_seq"])
            frame_start = time.perf_counter()
            _put_latest(status_queue, {
                "kind": "started", "generation": generation,
                "cloud_seq": cloud_seq, "monotonic_s": frame_start,
            })
            profile = dict(job.get("profile") or {})
            profile["frame_start_perf"] = frame_start
            profile["cloud_seq"] = cloud_seq
            try:
                halss_start = time.perf_counter()
                _stage_write(stage_fd, generation, cloud_seq, "HALSS", "START")
                halss_result = halss.evaluate(
                    job["halss_points"],
                    fixed_bounds=job["roi_bounds"],
                    profile=profile,
                )
                halss_ms = (time.perf_counter() - halss_start) * 1000.0
                _stage_write(stage_fd, generation, cloud_seq, "HALSS", "END", halss_ms)

                semantic_start = time.perf_counter()
                if halss_result is not None:
                    sem_map = sem_gen.generate(halss_result.get("bev_data", halss_result))
                else:
                    sem_map = np.full((obs_h, obs_w), danger_id, dtype=np.uint8)
                binary_semantic_vis = make_binary_semantic_vis(
                    sem_map, safe_id=safe_id, danger_id=danger_id,
                )
                semantic_ms = (time.perf_counter() - semantic_start) * 1000.0

                depth_start = time.perf_counter()
                semantic_valid_mask = np.ones_like(sem_map, dtype=bool)
                if projection_mode == "training_camera":
                    sparse_depth, valid_mask, sem_map, semantic_valid_mask = (
                        project_training_camera(
                            job["projection_points"], sem_map,
                            job["roi_bounds"], training_camera,
                            danger_id=danger_id,
                        )
                    )
                    binary_semantic_vis = make_binary_semantic_vis(
                        sem_map, safe_id=safe_id, danger_id=danger_id,
                    )
                    binary_semantic_vis[~semantic_valid_mask] = 128
                else:
                    sparse_depth, _ = project_bev_depth(
                        job["halss_points"],
                        grid_res=int(perc_cfg.get("halss_grid_res", 64)),
                        out_size=obs_w, max_range=depth_max,
                        half_x=float(job["half_x"]), half_y=float(job["half_y"]),
                    )
                    valid_mask = (sparse_depth < depth_max) & (sparse_depth > 0.01)
                depth_ms = (time.perf_counter() - depth_start) * 1000.0

                completion_start = time.perf_counter()
                rendered_depth = render_sparse_depth(
                    sparse_depth, valid_mask, depth_max,
                )
                completion_ms = (time.perf_counter() - completion_start) * 1000.0

                onnx_start = time.perf_counter()
                _stage_write(stage_fd, generation, cloud_seq, "ONNX", "START")
                action_id, rl_info = drl.predict(rendered_depth, sem_map, profile=profile)
                onnx_ms = (time.perf_counter() - onnx_start) * 1000.0
                _stage_write(stage_fd, generation, cloud_seq, "ONNX", "END", onnx_ms)
                rl_info["semantic_valid_ratio"] = float(np.mean(semantic_valid_mask))
                profile.update({
                    "semantic_generation_ms": semantic_ms,
                    "depth_projection_ms": depth_ms,
                    "depth_completion_ms": completion_ms,
                    "depth_valid_ratio": float(np.mean(valid_mask)),
                    "semantic_valid_ratio": float(np.mean(semantic_valid_mask)),
                    "action_id": int(action_id),
                })
                _put_latest(result_queue, {
                    "ok": True,
                    "generation": generation,
                    "cloud_seq": cloud_seq,
                    "cloud_stamp_ros_s": job.get("cloud_stamp_ros_s"),
                    "completed_monotonic_s": time.perf_counter(),
                    "action_id": int(action_id),
                    "rl_info": rl_info,
                    "sparse_depth": sparse_depth,
                    "valid_mask": valid_mask,
                    "semantic_valid_mask": semantic_valid_mask,
                    "rendered_depth": rendered_depth,
                    "sem_map": sem_map,
                    "binary_semantic_vis": binary_semantic_vis,
                    "timing": {
                        "halss_ms": halss_ms,
                        "semantic_ms": semantic_ms,
                        "depth_ms": depth_ms,
                        "completion_ms": completion_ms,
                        "onnx_ms": onnx_ms,
                        "total_ms": (time.perf_counter() - frame_start) * 1000.0,
                    },
                    "profile": profile,
                })
            except BaseException as exc:
                _stage_write(stage_fd, generation, cloud_seq, "FRAME", "ERROR")
                _put_latest(result_queue, {
                    "ok": False,
                    "generation": generation,
                    "cloud_seq": cloud_seq,
                    "cloud_stamp_ros_s": job.get("cloud_stamp_ros_s"),
                    "completed_monotonic_s": time.perf_counter(),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                    "profile": profile,
                })
    except BaseException as exc:
        _stage_write(stage_fd, generation, -1, "WORKER", "FATAL")
        print(
            f"[PerceptionWorker] FATAL generation={generation}: "
            f"{type(exc).__name__}: {exc}",
            flush=True,
        )
        traceback.print_exc()
        _put_latest(status_queue, {
            "kind": "fatal", "generation": generation,
            "monotonic_s": time.perf_counter(),
            "error_type": type(exc).__name__, "error": str(exc),
            "traceback": traceback.format_exc(),
        })
    finally:
        try:
            os.close(stage_fd)
        except OSError:
            pass
