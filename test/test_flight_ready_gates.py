#!/usr/bin/env python3
"""Lightweight regression tests for strict flight-ready gates.

No numpy/torch/ROS dependencies are required. These tests exercise the real
pipeline CLI in check-only mode so the safety gates stay wired to the entry
point used before flight.
"""

import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable


def _run(args):
    cmd = [PYTHON, str(ROOT / "pipeline.py"), "--config", str(ROOT / "config" / "experiment_config.yaml")]
    cmd.extend(args)
    return subprocess.run(cmd, cwd=str(ROOT), text=True, capture_output=True)


def _combined(result):
    return result.stdout + result.stderr


def _expect_fail(name, args, snippets):
    result = _run(args)
    text = _combined(result)
    assert result.returncode != 0, f"{name}: expected failure, got {result.returncode}\n{text}"
    for snippet in snippets:
        assert snippet in text, f"{name}: missing {snippet!r}\n{text}"
    print(f"  OK fail: {name}")


def _expect_ok(name, args, snippets):
    result = _run(args)
    text = _combined(result)
    assert result.returncode == 0, f"{name}: expected success, got {result.returncode}\n{text}"
    for snippet in snippets:
        assert snippet in text, f"{name}: missing {snippet!r}\n{text}"
    print(f"  OK pass: {name}")


def test_default_config_rejected():
    _expect_fail(
        "default config",
        ["--mode", "ros", "--flight-ready-check-only"],
        [
            "no global safe-area guidance active",
        ],
    )


def test_safe_point_format_checked():
    _expect_fail(
        "bad safe-point",
        [
            "--mode", "ros",
            "--safe-point", "bad",
            "--depth-output-scale", "40",
            "--yaw-rate-rad-s", "0.35",
            "--flight-ready-check-only",
        ],
        ["invalid --safe-point"],
    )


def test_safe_point_override_passes():
    _expect_ok(
        "GIS safe-point override",
        [
            "--mode", "ros",
            "--safe-point", "31.0,121.0",
            "--safe-point-source", "gis",
            "--depth-output-scale", "40",
            "--yaw-rate-rad-s", "0.35",
            "--flight-ready-check-only",
        ],
        ["Preview gates passed", "Check-only mode complete"],
    )


def test_manual_safe_point_rejected():
    _expect_fail(
        "manual safe-point",
        [
            "--mode", "ros",
            "--safe-point", "31.0,121.0",
            "--depth-output-scale", "40",
            "--yaw-rate-rad-s", "0.35",
            "--flight-ready-check-only",
        ],
        ["--safe-point requires --safe-point-source gis"],
    )


def test_mirrored_action_sign_rejected():
    with tempfile.TemporaryDirectory() as tmp:
        cfg = Path(tmp) / "mirrored_action.yaml"
        text = (ROOT / "config" / "experiment_config.yaml").read_text(encoding="utf-8")
        text = text.replace("action_lateral_sign: -1", "action_lateral_sign: 1")
        cfg.write_text(text, encoding="utf-8")
        result = subprocess.run(
            [
                PYTHON, str(ROOT / "pipeline.py"),
                "--config", str(cfg),
                "--mode", "ros",
                "--safe-point", "31.0,121.0",
                "--safe-point-source", "gis",
                "--depth-output-scale", "40",
                "--yaw-rate-rad-s", "0.35",
                "--flight-ready-check-only",
            ],
            cwd=str(ROOT),
            text=True,
            capture_output=True,
        )
        text = _combined(result)
        assert result.returncode != 0, f"mirrored action sign: expected failure\n{text}"
        assert "action_lateral_sign must be -1" in text, text
        print("  OK fail: mirrored action sign")


def test_ned_action_frame_rejected():
    with tempfile.TemporaryDirectory() as tmp:
        cfg = Path(tmp) / "ned_action_frame.yaml"
        text = (ROOT / "config" / "experiment_config.yaml").read_text(encoding="utf-8")
        text = text.replace('action_frame: "body"', 'action_frame: "ned"')
        cfg.write_text(text, encoding="utf-8")
        result = subprocess.run(
            [
                PYTHON, str(ROOT / "pipeline.py"),
                "--config", str(cfg),
                "--mode", "ros",
                "--safe-point", "31.0,121.0",
                "--safe-point-source", "gis",
                "--depth-output-scale", "40",
                "--yaw-rate-rad-s", "0.35",
                "--flight-ready-check-only",
            ],
            cwd=str(ROOT),
            text=True,
            capture_output=True,
        )
        text = _combined(result)
        assert result.returncode != 0, f"NED action frame: expected failure\n{text}"
        assert "action_frame must be 'body'" in text, text
        print("  OK fail: NED action frame")


def test_non_deeprl_observation_encoding_rejected():
    with tempfile.TemporaryDirectory() as tmp:
        cfg = Path(tmp) / "bad_observation_encoding.yaml"
        text = (ROOT / "config" / "experiment_config.yaml").read_text(encoding="utf-8")
        text = text.replace('depth_norm_mode: "meters_div255"', 'depth_norm_mode: "unit"')
        cfg.write_text(text, encoding="utf-8")
        result = subprocess.run(
            [
                PYTHON, str(ROOT / "pipeline.py"),
                "--config", str(cfg),
                "--mode", "ros",
                "--safe-point", "31.0,121.0",
                "--safe-point-source", "gis",
                "--depth-output-scale", "40",
                "--yaw-rate-rad-s", "0.35",
                "--flight-ready-check-only",
            ],
            cwd=str(ROOT),
            text=True,
            capture_output=True,
        )
        text = _combined(result)
        assert result.returncode != 0, f"bad observation encoding: expected failure\n{text}"
        assert "observation encoding must be DeepRL-compatible" in text, text
        print("  OK fail: bad observation encoding")


def _expect_rejected_config(name, replacements, snippet):
    with tempfile.TemporaryDirectory() as tmp:
        cfg = Path(tmp) / f"{name.replace(' ', '_')}.yaml"
        text = (ROOT / "config" / "experiment_config.yaml").read_text(encoding="utf-8")
        for old, new in replacements:
            text = text.replace(old, new)
        cfg.write_text(text, encoding="utf-8")
        result = subprocess.run(
            [
                PYTHON, str(ROOT / "pipeline.py"),
                "--config", str(cfg),
                "--mode", "ros",
                "--safe-point", "31.0,121.0",
                "--safe-point-source", "gis",
                "--depth-output-scale", "40",
                "--yaw-rate-rad-s", "0.35",
                "--flight-ready-check-only",
            ],
            cwd=str(ROOT),
            text=True,
            capture_output=True,
        )
        text = _combined(result)
        assert result.returncode != 0, f"{name}: expected failure\n{text}"
        assert snippet in text, f"{name}: missing {snippet!r}\n{text}"
        print(f"  OK fail: {name}")


def test_runtime_gpu_required():
    _expect_rejected_config(
        "runtime GPU",
        [('use_gpu: true', 'use_gpu: false')],
        "runtime must force GPU execution",
    )


def test_module_gpu_fallback_disabled_required():
    _expect_rejected_config(
        "HALSS GPU fallback",
        [('require_gpu: true                 # 真机禁止 HALSS Bayesian 推理静默回退 CPU',
          'require_gpu: false                # 真机禁止 HALSS Bayesian 推理静默回退 CPU')],
        "perception.require_gpu must be true",
    )
    _expect_rejected_config(
        "DRL GPU fallback",
        [('require_gpu: true                 # 真机禁止 SB3/PyTorch 策略静默回退 CPU',
          'require_gpu: false                # 真机禁止 SB3/PyTorch 策略静默回退 CPU')],
        "decision.require_gpu must be true",
    )


def test_halss_bayesian_backend_required():
    _expect_rejected_config(
        "HALSS backend",
        [('halss_backend: "bayesian_unet"', 'halss_backend: "gpu_sobel"')],
        "halss_backend must be 'bayesian_unet'",
    )


def test_perspective_depth_projection_required():
    _expect_rejected_config(
        "depth projection mode",
        [('mode: "perspective"', 'mode: "bev"')],
        "depth_projection.mode must be 'perspective'",
    )


def test_cuda_depth_projection_backend_required():
    _expect_rejected_config(
        "depth projection backend",
        [('backend: "torch_cuda"', 'backend: "numpy"')],
        "depth_projection.backend must be 'torch_cuda'",
    )


def test_inverse_depth_completion_encoding_required():
    _expect_rejected_config(
        "SparseNet input encoding",
        [('input_encoding: "inverse_unit"', 'input_encoding: "unit"')],
        "depth_completion.input_encoding must be 'inverse_unit'",
    )


def test_fastlio_height_source_required():
    _expect_rejected_config(
        "mission height source",
        [('height_source: "fastlio_z"', 'height_source: "pointcloud_min"')],
        "mission_state.height_source must be 'fastlio_z'",
    )


def test_localization_mode_required():
    _expect_rejected_config(
        "localization mode",
        [('mode: "fastlio_external_vision"', 'mode: "raw_lidar_only"')],
        "localization.mode must be 'fastlio_external_vision' or 'mocap_external_vision'",
    )


def test_visualization_enable_required():
    _expect_rejected_config(
        "visualization enable",
        [('enable: true', 'enable: false')],
        "visualization.enable must be true",
    )


def test_binary_semantic_display_required():
    _expect_rejected_config(
        "binary semantic display",
        [('show_binary_semantic: true', 'show_binary_semantic: false')],
        "visualization.show_binary_semantic must be true",
    )


def test_depth_display_required():
    _expect_rejected_config(
        "depth display",
        [('show_depth: true', 'show_depth: false')],
        "visualization.show_depth must be true",
    )


def test_binary_semantic_window_title_required():
    text = (ROOT / "config" / "experiment_config.yaml").read_text(encoding="utf-8")
    text = text.replace(
        'binary_semantic_window_title: "binary semantic"',
        'binary_semantic_window_title: "semantic"',
    )
    with tempfile.TemporaryDirectory() as tmp:
        cfg = Path(tmp) / "bad_window_title.yaml"
        cfg.write_text(text, encoding="utf-8")
        result = subprocess.run(
            [
                PYTHON, str(ROOT / "pipeline.py"),
                "--config", str(cfg),
                "--mode", "ros",
                "--safe-point", "31.0,121.0",
                "--safe-point-source", "gis",
                "--depth-output-scale", "40",
                "--yaw-rate-rad-s", "0.35",
                "--flight-ready-check-only",
            ],
            cwd=str(ROOT),
            text=True,
            capture_output=True,
        )
        text = _combined(result)
        assert result.returncode != 0, f"binary semantic title: expected failure\n{text}"
        assert "binary_semantic_window_title must remain" in text, text
        print("  OK fail: binary semantic title")


def test_gis_override_requires_existing_image_and_bounds():
    _expect_fail(
        "missing GIS image",
        [
            "--mode", "ros",
            "--gis-image", "missing.png",
            "--depth-output-scale", "40",
            "--yaw-rate-rad-s", "0.35",
            "--flight-ready-check-only",
        ],
        ["--gis-image missing", "--gis-image requires --gis-bounds"],
    )


def test_gis_override_passes_with_existing_files():
    with tempfile.TemporaryDirectory() as tmp:
        image = Path(tmp) / "sat.png"
        mask = Path(tmp) / "mask.png"
        image.write_bytes(b"placeholder")
        mask.write_bytes(b"placeholder")
        _expect_ok(
            "GIS override",
            [
                "--mode", "ros",
                "--gis-image", str(image),
                "--gis-mask", str(mask),
                "--gis-bounds", "113.90,22.72,113.92,22.74",
                "--depth-output-scale", "40",
                "--yaw-rate-rad-s", "0.35",
                "--flight-ready-check-only",
            ],
            ["Preview gates passed", "Check-only mode complete"],
        )


def test_check_only_disallows_bypass():
    _expect_fail(
        "check-only bypass",
        [
            "--mode", "ros",
            "--flight-ready-check-only",
            "--allow-incomplete-experiment",
        ],
        ["cannot be combined"],
    )


if __name__ == "__main__":
    print("=== FlightReady gate regression tests ===")
    test_default_config_rejected()
    test_safe_point_format_checked()
    test_safe_point_override_passes()
    test_manual_safe_point_rejected()
    test_mirrored_action_sign_rejected()
    test_ned_action_frame_rejected()
    test_non_deeprl_observation_encoding_rejected()
    test_runtime_gpu_required()
    test_module_gpu_fallback_disabled_required()
    test_halss_bayesian_backend_required()
    test_perspective_depth_projection_required()
    test_cuda_depth_projection_backend_required()
    test_inverse_depth_completion_encoding_required()
    test_fastlio_height_source_required()
    test_localization_mode_required()
    test_visualization_enable_required()
    test_binary_semantic_display_required()
    test_depth_display_required()
    test_binary_semantic_window_title_required()
    test_gis_override_requires_existing_image_and_bounds()
    test_gis_override_passes_with_existing_files()
    test_check_only_disallows_bypass()
    print("=== ALL FlightReady gate tests PASSED ===")
