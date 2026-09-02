#!/usr/bin/env python3
"""Unit tests for mission-event and per-cloud timing CSV records."""

import csv
import asyncio
import io
import queue
from types import SimpleNamespace

import numpy as np

import pipeline as pipeline_module
from pipeline import (
    FRAME_TIMING_BASE_FIELDS,
    FRAME_TIMING_DETAILED_FIELDS,
    OrinLandingPipeline,
)


# Runtime dependencies are normally installed by OrinLandingPipeline.__init__.
# These focused logger tests intentionally bypass heavyweight model/ROS setup.
pipeline_module.np = np


class _FlushableStringIO(io.StringIO):
    def flush(self):
        return None


def _csv_rows(buffer):
    return list(csv.reader(io.StringIO(buffer.getvalue())))


def test_mission_event_is_deduplicated_and_captures_pose():
    pipeline = OrinLandingPipeline.__new__(OrinLandingPipeline)
    output = _FlushableStringIO()
    pipeline._event_log_file = output
    pipeline._event_log_writer = csv.writer(output)
    pipeline._recorded_mission_events = set()
    pipeline._rospy = None
    pipeline.mission_state = "GOTO_SAFE"
    pipeline.fc = SimpleNamespace(
        uavPosENU=np.array([1.0, 2.0, 30.0]),
        isArmed=True,
        isOffboard=True,
        landed_state_on_ground=False,
    )
    pipeline.fastlio = SimpleNamespace(pose=np.array([4.0, 5.0, 29.5, 0.0, 0.0, 0.0]))
    pipeline.state_manager = SimpleNamespace(height_from_pose=lambda pose: pose[2] - 0.5)

    assert pipeline._log_mission_event("GOTO_STARTED", "test") is True
    assert pipeline._log_mission_event("GOTO_STARTED", "duplicate") is False

    rows = _csv_rows(output)
    assert len(rows) == 1
    assert rows[0][0] == "GOTO_STARTED"
    assert rows[0][3:5] == ["GOTO_SAFE", "test"]
    assert rows[0][5:8] == ["1.0000", "2.0000", "30.0000"]
    assert rows[0][8:12] == ["4.0000", "5.0000", "29.5000", "29.000"]
    assert rows[0][12:] == ["1", "1", "0"]


def test_record_timing_writes_only_actual_perception_frames():
    pipeline = OrinLandingPipeline.__new__(OrinLandingPipeline)
    pipeline._timing = {
        "halss": [], "depth": [], "completion": [], "rl": [], "control": [], "total": []
    }
    pipeline._rospy = None
    output = _FlushableStringIO()
    pipeline._frame_timing_file = output
    pipeline._frame_timing_writer = csv.writer(output)

    pipeline._record_timing(0.010, 0.020, 0.030, 0.040, 0.005, 0.110)
    assert _csv_rows(output) == []

    pipeline._record_timing(
        0.010, 0.020, 0.030, 0.040, 0.005, 0.110,
        cloud_stamp_ros_s=100.0,
        cloud_seq=20,
        pose_seq=21,
        state="DRL_DESCENT",
        sync_ms=4.0,
        source_age_ms=12.0,
        result_age_ms=112.0,
        newer_frames=2,
        accepted=False,
        fallback_reason="inference_result_too_old",
        perception_executed=True,
        pointcloud_preprocess_s=0.005,
    )

    row = _csv_rows(output)[0]
    assert row[1:5] == ["100.000000", "20", "21", "DRL_DESCENT"]
    assert row[6:12] == [
        "5.000", "10.000", "20.000", "30.000", "40.000", "105.000"
    ]
    assert row[13:19] == [
        "110.000", "12.000", "112.000", "2", "0", "inference_result_too_old"
    ]


def test_detailed_frame_timing_appends_profile_and_failure_fields():
    pipeline = OrinLandingPipeline.__new__(OrinLandingPipeline)
    pipeline.enable_detailed_profiling = True
    pipeline._timing = {
        "halss": [], "depth": [], "completion": [], "rl": [], "control": [], "total": []
    }
    pipeline._rospy = None
    pipeline.mission_state = "DRL_DESCENT"
    output = _FlushableStringIO()
    pipeline._frame_timing_file = output
    pipeline._frame_timing_writer = csv.writer(output)

    profile = {
        "frame_id": 7,
        "timestamp": 101.25,
        "cloud_points": 2400,
        "valid_projection_points": 900,
        "halss_mc_samples": 5,
        "action_id": 9,
        "action_name": "DESCEND",
        "stage_error": "onnx_inference",
        "error_message": "RuntimeError: test failure",
    }
    pipeline._log_frame_timing(
        cloud_stamp_ros_s=100.0, cloud_seq=20, pose_seq=0,
        state="DRL_DESCENT", sync_ms=4.0,
        pointcloud_preprocess_ms=1.0, halss_ms=2.0, depth_ms=3.0,
        completion_ms=4.0, onnx_ms=0.0, control_ms=0.0,
        pipeline_total_ms=12.0, source_age_ms=8.0, result_age_ms=None,
        newer_frames=0, accepted=False, fallback_reason="stage_error",
        detailed_profile=profile,
    )

    row = _csv_rows(output)[0]
    assert len(row) == len(FRAME_TIMING_BASE_FIELDS) + len(FRAME_TIMING_DETAILED_FIELDS)
    detail = dict(zip(FRAME_TIMING_DETAILED_FIELDS, row[len(FRAME_TIMING_BASE_FIELDS):]))
    assert detail["frame_id"] == "7"
    assert detail["timestamp"] == "101.250000"
    assert detail["cloud_points"] == "2400"
    assert detail["action_name"] == "DESCEND"
    assert detail["stage_error"] == "onnx_inference"
    assert detail["error_message"] == "RuntimeError: test failure"


def test_latest_only_submission_replaces_unstarted_frame():
    pipeline = OrinLandingPipeline.__new__(OrinLandingPipeline)
    pipeline._perception_request_queue = queue.Queue(maxsize=1)
    pipeline._submit_perception_job({"cloud_seq": 10})
    pipeline._submit_perception_job({"cloud_seq": 11})
    assert pipeline._perception_request_queue.get_nowait()["cloud_seq"] == 11


def test_perception_failure_captures_full_xyz_and_keeps_yaw_rate():
    calls = []

    class _FC:
        uavPosENU = np.array([1.0, 2.0, 30.0], dtype=np.float32)

        async def send_position_enu_yaw_rate(self, x, y, z, yaw_rate):
            calls.append((x, y, z, yaw_rate))

    pipeline = OrinLandingPipeline.__new__(OrinLandingPipeline)
    pipeline.fc = _FC()
    pipeline.yaw_rate_cmd = 5.0
    pipeline._perception_failure_since = None
    pipeline._perception_hold_enu_xyz = None
    pipeline.state_manager = SimpleNamespace(perception_timeout_s=20.0)
    asyncio.run(pipeline._hold_for_perception_failure("test", None))
    pipeline.fc.uavPosENU[:] = [9.0, 9.0, 9.0]
    asyncio.run(pipeline._hold_for_perception_failure("test", None))
    assert calls == [(1.0, 2.0, 30.0, 5.0), (1.0, 2.0, 30.0, 5.0)]


def test_async_frame_writer_flushes_before_shutdown():
    pipeline = OrinLandingPipeline.__new__(OrinLandingPipeline)
    output = _FlushableStringIO()
    pipeline._frame_timing_file = output
    pipeline._frame_timing_writer = csv.writer(output)
    pipeline._perception_gate_file = _FlushableStringIO()
    pipeline._perception_gate_writer = csv.writer(pipeline._perception_gate_file)
    pipeline._diagnostic_log_running = False
    pipeline._diagnostic_log_queue = None
    pipeline._diagnostic_log_thread = None
    pipeline._diagnostic_log_dropped = 0
    pipeline._start_diagnostic_log_writer()
    pipeline._enqueue_diagnostic_row("frame_timing", ["frame", "1"])
    pipeline._stop_diagnostic_log_writer()
    assert _csv_rows(output) == [["frame", "1"]]


def test_mavros_safety_and_shutdown_reason_are_persisted():
    async def _async_noop(*args, **kwargs):
        return None

    pipeline = OrinLandingPipeline.__new__(OrinLandingPipeline)
    output = _FlushableStringIO()
    pipeline._event_log_file = output
    pipeline._event_log_writer = csv.writer(output)
    pipeline._recorded_mission_events = set()
    pipeline._rospy = None
    pipeline._px4_pose_authoritative = True
    pipeline._shutdown_reason = ""
    pipeline._shutdown_event_logged = False
    pipeline.mission_state = "GOTO_SAFE"
    pipeline.fc = SimpleNamespace(
        uavPosENU=np.array([1.0, 2.0, 30.0]),
        isArmed=True,
        isOffboard=False,
        flightMode="POSCTL",
        safety_fallback=True,
        safety_fallback_reason="OFFBOARD_lost",
        landed_state_on_ground=False,
        send_velocity_enu_yaw_rate=_async_noop,
    )
    pipeline.fastlio = SimpleNamespace(pose=None)
    pipeline.state_manager = SimpleNamespace(height_from_pose=lambda pose: pose[2])

    assert asyncio.run(pipeline._check_active_flight_safety("GOTO_SAFE")) is False
    assert pipeline.mission_state == "IDLE"
    rows = _csv_rows(output)
    assert rows[0][0] == "MAVROS_SAFETY_FALLBACK"
    assert "phase=GOTO_SAFE" in rows[0][4]
    assert "armed=True;offboard=False;mode=POSCTL" in rows[0][4]
    assert rows[1][0] == "MANUAL_TAKEOVER"

    pipeline._stop_perception_worker = lambda graceful=True: None
    pipeline.visualizer = SimpleNamespace(close=lambda: None)
    pipeline._stop_recording = _async_noop
    pipeline._close_velocity_log = lambda: None
    pipeline._close_drl_action_log = lambda: None
    pipeline._close_experiment_logs = lambda: None
    asyncio.run(pipeline._shutdown())
    rows = _csv_rows(output)
    assert rows[2][0] == "SHUTDOWN_REASON"
    assert "mavros_safety_fallback" in rows[2][4]
    assert "armed=True;offboard=False;mode=POSCTL" in rows[2][4]


def test_fatal_error_event_contains_flattened_traceback():
    pipeline = OrinLandingPipeline.__new__(OrinLandingPipeline)
    output = _FlushableStringIO()
    pipeline._event_log_file = output
    pipeline._event_log_writer = csv.writer(output)
    pipeline._recorded_mission_events = set()
    pipeline._rospy = None
    pipeline._px4_pose_authoritative = True
    pipeline.mission_state = "DRL_DESCENT"
    pipeline.fc = SimpleNamespace(
        uavPosENU=np.zeros(3), isArmed=True, isOffboard=True,
        landed_state_on_ground=False,
    )
    pipeline.fastlio = SimpleNamespace(pose=None)
    pipeline.state_manager = SimpleNamespace(height_from_pose=lambda pose: pose[2])

    pipeline._record_fatal_error(RuntimeError("boom"), "line one\nline two\n")
    row = _csv_rows(output)[0]
    assert row[0] == "FATAL_ERROR"
    assert "type=RuntimeError;message=boom" in row[4]
    assert "traceback=line one\\nline two\\n" in row[4]
