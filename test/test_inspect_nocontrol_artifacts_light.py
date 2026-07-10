#!/usr/bin/env python3
"""Lightweight tests for no-control artifact inspector."""

import math
import struct
import subprocess
import sys
import tempfile
import zlib
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PYTHON = sys.executable


def _npy_header(shape, descr="|u1"):
    meta = {
        "descr": descr,
        "fortran_order": False,
        "shape": shape,
    }
    header = repr(meta)
    prefix_len = 10
    padding = 16 - ((prefix_len + len(header) + 1) % 16)
    header = header + (" " * padding) + "\n"
    raw = header.encode("latin1")
    return b"\x93NUMPY\x01\x00" + struct.pack("<H", len(raw)) + raw


def _pack_values(values, descr):
    if values is None:
        return b""
    if not isinstance(values, (list, tuple)):
        values = [values]
    fmt_by_descr = {
        "<f4": "<f",
        "<f8": "<d",
        "<i4": "<i",
        "|u1": "<B",
    }
    fmt = fmt_by_descr[descr]
    return b"".join(struct.pack(fmt, value) for value in values)


def _write_npz(path: Path, missing_key=None, sync_ms=25.0,
               v_ned=None, action_id=3):
    yaw = math.radians(10.0)
    if v_ned is None:
        v_ned = [-math.sin(yaw), math.cos(yaw), 0.0]
    keys = {
        "sparse_depth": ((128, 128), "<f4", None),
        "valid_mask": ((128, 128), "|u1", None),
        "dense_depth": ((128, 128), "<f4", None),
        "sem_map": ((128, 128), "|u1", None),
        "binary_semantic_vis": ((128, 128, 3), "|u1", None),
        "yaw_rad": ((), "<f4", yaw),
        "cloud_odom_sync_ms": ((), "<f4", sync_ms),
        "pose_xyz": ((3,), "<f4", None),
        "rpy": ((3,), "<f4", None),
        "action_id": ((), "<i4", action_id),
        "v_body": ((3,), "<f4", [0.0, 1.0, 0.0]),
        "v_ned": ((3,), "<f4", v_ned),
    }
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for key, (shape, descr, values) in keys.items():
            if key == missing_key:
                continue
            zf.writestr(f"{key}.npy", _npy_header(shape, descr) + _pack_values(values, descr))


def _png_chunk(kind, data):
    return (
        struct.pack(">I", len(data))
        + kind
        + data
        + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)
    )


def _write_png(path: Path, width=2, height=2):
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0)
    raw_rows = b"".join(b"\x00" + (b"\x00" * width) for _ in range(height))
    data = zlib.compress(raw_rows)
    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(b"IHDR", ihdr)
        + _png_chunk(b"IDAT", data)
        + _png_chunk(b"IEND", b"")
    )


def _run(frame_dir: Path, *args):
    return subprocess.run(
        [PYTHON, str(ROOT / "inspect_nocontrol_artifacts_light.py"), str(frame_dir), *args],
        cwd=str(ROOT),
        text=True,
        capture_output=True,
    )


def test_good_artifacts_pass():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _write_npz(root / "000000_calib_frame.npz")
        _write_png(root / "000000_binary_semantic.png")
        _write_png(root / "000000_depth.png")
        result = _run(root, "--require-frames")
        text = result.stdout + result.stderr
        assert result.returncode == 0, text
        assert "[OK] no-control artifacts accepted" in text


def test_missing_key_fails():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _write_npz(root / "000000_calib_frame.npz", missing_key="binary_semantic_vis")
        result = _run(root)
        text = result.stdout + result.stderr
        assert result.returncode != 0, text
        assert "missing keys" in text
    assert "binary_semantic_vis" in text


def test_missing_sync_key_fails():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _write_npz(root / "000000_calib_frame.npz", missing_key="cloud_odom_sync_ms")
        result = _run(root)
        text = result.stdout + result.stderr
        assert result.returncode != 0, text
        assert "missing keys" in text
        assert "cloud_odom_sync_ms" in text


def test_sync_budget_fails():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _write_npz(root / "000000_calib_frame.npz", sync_ms=150.0)
        result = _run(root, "--max-sync-ms", "100")
        text = result.stdout + result.stderr
        assert result.returncode != 0, text
        assert "cloud_odom_sync_ms 150.0ms > 100.0ms" in text


def test_yaw_transform_fails():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _write_npz(root / "000000_calib_frame.npz", v_ned=[1.0, 0.0, 0.0])
        result = _run(root, "--max-yaw-transform-error", "0.15")
        text = result.stdout + result.stderr
        assert result.returncode != 0, text
        assert "yaw transform error" in text


def test_invalid_action_id_fails():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _write_npz(root / "000000_calib_frame.npz", action_id=12)
        result = _run(root)
        text = result.stdout + result.stderr
        assert result.returncode != 0, text
        assert "action_id 12 is outside 0..9" in text


def main():
    test_good_artifacts_pass()
    test_missing_key_fails()
    test_missing_sync_key_fails()
    test_sync_budget_fails()
    test_yaw_transform_fails()
    test_invalid_action_id_fails()
    print("=== Lightweight no-control artifact acceptance ===")
    print("  OK good artifacts pass")
    print("  OK missing key fails")
    print("  OK missing sync key fails")
    print("  OK stale cloud/odom sync fails")
    print("  OK yaw transform mismatch fails")
    print("  OK invalid action_id fails")
    print("=== ALL PASSED ===")


if __name__ == "__main__":
    main()
