#!/usr/bin/env python3
"""Lightweight tests for exact HALSS binary visualization comparison."""

import tempfile
import zlib
from pathlib import Path

from compare_halss_visualization_light import (
    Image,
    compare_images,
    load_image,
    resize_nearest,
    to_grayscale,
)


def write_pgm(path: Path, width: int, height: int, pixels: bytes):
    path.write_bytes(f"P5\n{width} {height}\n255\n".encode("ascii") + pixels)


def write_png_rgb(path: Path, width: int, height: int, pixels: bytes):
    import struct

    def chunk(kind: bytes, payload: bytes) -> bytes:
        crc = zlib.crc32(kind + payload) & 0xFFFFFFFF
        return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", crc)

    raw = bytearray()
    stride = width * 3
    for y in range(height):
        raw.append(0)
        raw.extend(pixels[y * stride:(y + 1) * stride])
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", ihdr)
        + chunk(b"IDAT", zlib.compress(bytes(raw)))
        + chunk(b"IEND", b"")
    )


def test_exact_pgm_match_passes(tmp: Path):
    ref_path = tmp / "ref.pgm"
    cand_path = tmp / "cand.pgm"
    pixels = bytes([0, 255, 0, 255, 255, 0])
    write_pgm(ref_path, 3, 2, pixels)
    write_pgm(cand_path, 3, 2, pixels)
    metrics = compare_images(load_image(str(ref_path)), load_image(str(cand_path)))
    assert metrics["exact_match"] is True
    assert metrics["max_abs_diff"] == 0
    assert metrics["mean_abs_diff"] == 0.0


def test_single_pixel_difference_fails_strict_gate(tmp: Path):
    ref_path = tmp / "ref.pgm"
    cand_path = tmp / "cand.pgm"
    write_pgm(ref_path, 3, 2, bytes([0, 255, 0, 255, 255, 0]))
    write_pgm(cand_path, 3, 2, bytes([0, 255, 0, 254, 255, 0]))
    metrics = compare_images(load_image(str(ref_path)), load_image(str(cand_path)))
    assert metrics["exact_match"] is False
    assert metrics["max_abs_diff"] == 1
    assert abs(metrics["mean_abs_diff"] - (1.0 / 6.0)) <= 1e-9


def test_equal_rgb_png_converts_to_gray(tmp: Path):
    png_path = tmp / "candidate.png"
    pixels_rgb = bytes([
        0, 0, 0,
        255, 255, 255,
        127, 127, 127,
        255, 255, 255,
    ])
    write_png_rgb(png_path, 2, 2, pixels_rgb)
    gray = to_grayscale(load_image(str(png_path)))
    assert gray.width == 2
    assert gray.height == 2
    assert gray.channels == 1
    assert gray.pixels == bytes([0, 255, 127, 255])


def test_nearest_resize_matches_cv2_style_indexing():
    src = Image(2, 2, 1, bytes([1, 2, 3, 4]))
    resized = resize_nearest(src, 4, 4)
    assert resized.pixels == bytes([
        1, 1, 2, 2,
        1, 1, 2, 2,
        3, 3, 4, 4,
        3, 3, 4, 4,
    ])


def main():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        test_exact_pgm_match_passes(tmp)
        test_single_pixel_difference_fails_strict_gate(tmp)
        test_equal_rgb_png_converts_to_gray(tmp)
    test_nearest_resize_matches_cv2_style_indexing()
    print("=== Lightweight HALSS visualization acceptance ===")
    print("  OK exact PGM match passes")
    print("  OK one-pixel mismatch fails strict gate")
    print("  OK RGB PNG grayscale conversion")
    print("  OK nearest-neighbor resize")
    print("=== ALL PASSED ===")


if __name__ == "__main__":
    main()
