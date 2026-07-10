#!/usr/bin/env python3
"""Lightweight tests for GIS global-prior JSON acceptance."""

import json
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PYTHON = sys.executable


def _run(path: Path, *args):
    return subprocess.run(
        [PYTHON, str(ROOT / "accept_gis_prior_light.py"), str(path), *args],
        cwd=str(ROOT),
        text=True,
        capture_output=True,
    )


def test_good_gis_json_passes():
    data = {
        "timestamp_ms": 1,
        "source_image_path": "sat.png",
        "source_sem_mask_path": "mask.png",
        "segmentation_source": "precomputed_mask",
        "bounds": [113.90, 22.72, 113.92, 22.74],
        "image_size_px": [300, 300],
        "best_cell": [1, 1],
        "best_cell_bounds_px": [100, 100, 200, 200],
        "best_center_px": [150, 150],
        "best_center_gps": [22.73, 113.91],
        "min_risk": 0.0,
        "risk_grid": [
            [4.0, 4.0, 4.0],
            [4.0, 0.0, 4.0],
            [4.0, 4.0, 4.0],
        ],
        "has_gps_target": True,
    }
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "global_prior_1.json"
        path.write_text(json.dumps(data), encoding="utf-8")
        result = _run(path, "--bounds", "113.90,22.72,113.92,22.74")
        text = result.stdout + result.stderr
        assert result.returncode == 0, text
        assert "[OK] GIS global prior accepted" in text


def test_bad_gis_json_fails():
    data = {
        "timestamp_ms": 1,
        "bounds": [113.91, 22.72, 113.92, 22.74],
        "best_cell": [0, 0],
        "best_cell_bounds_px": [0, 0, 100, 100],
        "best_center_px": [150, 150],
        "best_center_gps": [23.73, 115.91],
        "min_risk": 2.0,
        "risk_grid": [
            [4.0, 4.0, 4.0],
            [4.0, 0.0, 4.0],
            [4.0, 4.0, 4.0],
        ],
        "has_gps_target": True,
    }
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "global_prior_1.json"
        path.write_text(json.dumps(data), encoding="utf-8")
        result = _run(path, "--bounds", "113.90,22.72,113.92,22.74")
        text = result.stdout + result.stderr
        assert result.returncode != 0, text
        assert "does not match risk_grid best" in text
        assert "best_cell" in text
        assert "recorded bounds index" in text
        assert "latitude" in text
        assert "longitude" in text


def test_missing_traceability_fails_by_default():
    data = {
        "bounds": [113.90, 22.72, 113.92, 22.74],
        "best_cell": [1, 1],
        "best_cell_bounds_px": [100, 100, 200, 200],
        "best_center_px": [150, 150],
        "best_center_gps": [22.73, 113.91],
        "min_risk": 0.0,
        "risk_grid": [
            [4.0, 4.0, 4.0],
            [4.0, 0.0, 4.0],
            [4.0, 4.0, 4.0],
        ],
        "has_gps_target": True,
    }
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "global_prior_1.json"
        path.write_text(json.dumps(data), encoding="utf-8")
        result = _run(path, "--bounds", "113.90,22.72,113.92,22.74")
        text = result.stdout + result.stderr
        assert result.returncode != 0, text
        assert "source_image_path is required" in text
        assert "image_size_px must be" in text
        assert "source_sem_mask_path or segmentation_source" in text


def test_missing_traceability_can_be_explicitly_allowed():
    data = {
        "bounds": [113.90, 22.72, 113.92, 22.74],
        "best_cell": [1, 1],
        "best_cell_bounds_px": [100, 100, 200, 200],
        "best_center_px": [150, 150],
        "best_center_gps": [22.73, 113.91],
        "min_risk": 0.0,
        "risk_grid": [
            [4.0, 4.0, 4.0],
            [4.0, 0.0, 4.0],
            [4.0, 4.0, 4.0],
        ],
        "has_gps_target": True,
    }
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "global_prior_1.json"
        path.write_text(json.dumps(data), encoding="utf-8")
        result = _run(
            path,
            "--bounds", "113.90,22.72,113.92,22.74",
            "--allow-missing-traceability",
        )
        text = result.stdout + result.stderr
        assert result.returncode == 0, text
        assert "[OK] GIS global prior accepted" in text


def main():
    test_good_gis_json_passes()
    test_bad_gis_json_fails()
    test_missing_traceability_fails_by_default()
    test_missing_traceability_can_be_explicitly_allowed()
    print("=== Lightweight GIS global-prior acceptance ===")
    print("  OK valid GIS JSON passes")
    print("  OK invalid risk/GPS JSON fails")
    print("  OK missing GIS traceability fails by default")
    print("  OK missing GIS traceability can be explicitly allowed")
    print("=== ALL PASSED ===")


if __name__ == "__main__":
    main()
