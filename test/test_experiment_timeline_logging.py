#!/usr/bin/env python3
"""Unit tests for mission-event and per-cloud timing CSV records."""

import csv
import io
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
