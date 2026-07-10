#!/usr/bin/env python3
"""Lightweight tests for saved binary semantic NPZ/PNG comparison."""

import struct
import subprocess
import sys
import tempfile
import zlib
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PYTHON = sys.executable


def _npy_uint8(shape, data):
    meta = {
        "descr": "|u1",
        "fortran_order": False,
        "shape": shape,
    }
    header = repr(meta)
    padding = 16 - ((10 + len(header) + 1) % 16)
    header = (header + " " * padding + "\n").encode("latin1")
    return b"\x93NUMPY\x01\x00" + struct.pack("<H", len(header)) + header + bytes(data)


def _write_npz(path: Path, width=4, height=3, data=None):
    if data is None:
        data = [0, 255, 0, 255] * height
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("binary_semantic_vis.npy", _npy_uint8((height, width), data))


def _png_chunk(kind, data):
    return (
        struct.pack(">I", len(data))
        + kind
        + data
        + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)
    )


def _write_gray_png(path: Path, width=4, height=3, data=None):
    if data is None:
        data = [0, 255, 0, 255] * height
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0)
    rows = []
    for y in range(height):
        row = bytes(data[y * width:(y + 1) * width])
        rows.append(b"\x00" + row)
    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(b"IHDR", ihdr)
        + _png_chunk(b"IDAT", zlib.compress(b"".join(rows)))
        + _png_chunk(b"IEND", b"")
    )


def _run(*args):
    return subprocess.run(
        [PYTHON, str(ROOT / "compare_saved_binary_semantic_light.py"), *args],
        cwd=str(ROOT),
        text=True,
        capture_output=True,
    )


def test_matching_npz_png_pass():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        data = [0, 255, 0, 255] * 3
        _write_npz(root / "000000_calib_frame.npz", data=data)
        _write_gray_png(root / "000000_binary_semantic.png", data=data)
        result = _run("--frame-dir", str(root), "--grayscale")
        text = result.stdout + result.stderr
        assert result.returncode == 0, text
        assert "[OK] saved binary semantic matches" in text


def test_mismatched_npz_png_fails():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        npz_data = [0, 255, 0, 255] * 3
        png_data = list(npz_data)
        png_data[5] = 0
        _write_npz(root / "000000_calib_frame.npz", data=npz_data)
        _write_gray_png(root / "000000_binary_semantic.png", data=png_data)
        result = _run("--frame-dir", str(root), "--grayscale")
        text = result.stdout + result.stderr
        assert result.returncode != 0, text
        assert "max_abs_diff" in text
        assert "[FAIL]" in text


def main():
    test_matching_npz_png_pass()
    test_mismatched_npz_png_fails()
    print("=== Lightweight saved binary semantic comparison ===")
    print("  OK matching NPZ/PNG passes")
    print("  OK mismatched NPZ/PNG fails")
    print("=== ALL PASSED ===")


if __name__ == "__main__":
    main()
