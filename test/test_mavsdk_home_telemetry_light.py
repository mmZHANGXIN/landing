#!/usr/bin/env python3
"""Lightweight MAVSDK home-telemetry source contract tests."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _source():
    return (ROOT / "control" / "mavsdk_controller.py").read_text(encoding="utf-8")


def test_home_telemetry_flags_and_waiter_exist():
    text = _source()
    assert "self._ned_ready = False" in text
    assert "self._gps_ready = False" in text
    assert "self._attitude_ready = False" in text
    assert "async def wait_for_home" in text
    assert "Home telemetry not ready" in text
    assert "async def wait_for_local_pose" in text
    assert "Local telemetry not ready" in text


def test_zero_gps_is_not_accepted_as_home():
    text = _source()
    assert "def _valid_lat_lon" in text
    assert "abs(lat) < 1e-9 and abs(lon) < 1e-9" in text
    assert "self._gps_ready = True" in text


def main():
    test_home_telemetry_flags_and_waiter_exist()
    test_zero_gps_is_not_accepted_as_home()
    print("=== Lightweight MAVSDK home telemetry contract ===")
    print("  OK MAVSDK tracks NED/GPS/attitude readiness and exposes wait_for_home")
    print("  OK zero GPS is rejected before global safe-point NED conversion")
    print("=== ALL PASSED ===")


if __name__ == "__main__":
    main()
