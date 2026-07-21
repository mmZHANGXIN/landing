#!/usr/bin/env python3
"""Dependency-light checks for the outdoor GOTO XY speed limiter."""

import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline import _limited_xy_velocity
from diagnostics.flight_ready import (
    _check_goto_speed_limit,
    _check_outdoor_halss_ray_sampling,
)


def test_far_target_is_capped_at_two_mps():
    vx, vy = _limited_xy_velocity(3.0, 4.0, 2.0, 1.0)
    assert math.isclose(math.hypot(vx, vy), 2.0, abs_tol=1e-9)
    assert math.isclose(vx, 1.2, abs_tol=1e-9)
    assert math.isclose(vy, 1.6, abs_tol=1e-9)


def test_near_target_slows_proportionally():
    vx, vy = _limited_xy_velocity(0.1, 0.0, 2.0, 1.0)
    assert math.isclose(vx, 0.1, abs_tol=1e-9)
    assert math.isclose(vy, 0.0, abs_tol=1e-9)


def test_zero_distance_commands_zero_velocity():
    assert _limited_xy_velocity(0.0, 0.0, 2.0, 1.0) == (0.0, 0.0)


def test_outdoor_gate_rejects_limit_above_two_mps():
    ok, message = _check_goto_speed_limit({
        "localization": {"mode": "gps_px4_fastlio_perception"},
        "global_prior": {
            "goto_max_horizontal_speed_mps": 2.1,
            "goto_horizontal_kp_s": 1.0,
        },
    })
    assert not ok
    assert "(0, 2.0]" in message


def test_outdoor_gate_requires_fixed_halss_ray_sampling():
    cfg = {
        "localization": {"mode": "gps_px4_fastlio_perception"},
        "perception": {
            "halss_pinhole_ray_sampling_enabled": True,
            "halss_pinhole_ray_grid_res": 64,
        },
    }
    ok, message = _check_outdoor_halss_ray_sampling(cfg)
    assert ok, message
    cfg["perception"]["halss_pinhole_ray_sampling_enabled"] = False
    ok, message = _check_outdoor_halss_ray_sampling(cfg)
    assert not ok
    assert "fixed pinhole ray sampling" in message


if __name__ == "__main__":
    test_far_target_is_capped_at_two_mps()
    test_near_target_slows_proportionally()
    test_zero_distance_commands_zero_velocity()
    test_outdoor_gate_rejects_limit_above_two_mps()
    test_outdoor_gate_requires_fixed_halss_ray_sampling()
    print("GOTO speed limit tests: PASS")
