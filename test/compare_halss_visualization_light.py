#!/usr/bin/env python3
"""Stdlib-only pixel comparison for HALSS binary semantic frames.

Supports 8-bit PNG and binary PGM (P5). This keeps the HALSS visualization
acceptance runnable on machines without OpenCV or NumPy.
"""

import argparse
import struct
import sys
import zlib
from dataclasses import dataclass
from pathlib import Path


PNG_SIG = b"\x89PNG\r\n\x1a\n"


@dataclass
class Image:
    width: int
    height: int
    channels: int
    pixels: bytes


def _read_pgm(path: Path) -> Image:
    data = path.read_bytes()
    idx = 0

    def next_token():
        nonlocal idx
        while idx < len(data) and data[idx] in b" \t\r\n":
            idx += 1
        if idx < len(data) and data[idx] == ord("#"):
            while idx < len(data) and data[idx] not in b"\r\n":
                idx += 1
            return next_token()
        start = idx
        while idx < len(data) and data[idx] not in b" \t\r\n":
            idx += 1
        return data[start:idx].decode("ascii")

    magic = next_token()
    if magic != "P5":
        raise ValueError(f"{path}: only binary PGM P5 is supported")
    width = int(next_token())
    height = int(next_token())
    maxval = int(next_token())
    if maxval != 255:
        raise ValueError(f"{path}: only 8-bit PGM maxval=255 is supported")
    while idx < len(data) and data[idx] in b" \t\r\n":
        idx += 1
    pixels = data[idx:]
    expected = width * height
    if len(pixels) != expected:
        raise ValueError(f"{path}: expected {expected} pixels, got {len(pixels)}")
    return Image(width, height, 1, pixels)


def _paeth(a: int, b: int, c: int) -> int:
    p = a + b - c
    pa = abs(p - a)
    pb = abs(p - b)
    pc = abs(p - c)
    if pa <= pb and pa <= pc:
        return a
    if pb <= pc:
        return b
    return c


def _unfilter_png(raw: bytes, width: int, height: int, channels: int) -> bytes:
    stride = width * channels
    bpp = channels
    out = bytearray(height * stride)
    pos = 0
    for row in range(height):
        if pos >= len(raw):
            raise ValueError("truncated PNG scanline data")
        filter_type = raw[pos]
        pos += 1
        scan = bytearray(raw[pos:pos + stride])
        pos += stride
        prev_row = out[(row - 1) * stride:row * stride] if row > 0 else None
        for i in range(stride):
            left = scan[i - bpp] if i >= bpp else 0
            up = prev_row[i] if prev_row is not None else 0
            up_left = prev_row[i - bpp] if prev_row is not None and i >= bpp else 0
            if filter_type == 0:
                value = scan[i]
            elif filter_type == 1:
                value = scan[i] + left
            elif filter_type == 2:
                value = scan[i] + up
            elif filter_type == 3:
                value = scan[i] + ((left + up) // 2)
            elif filter_type == 4:
                value = scan[i] + _paeth(left, up, up_left)
            else:
                raise ValueError(f"unsupported PNG filter type {filter_type}")
            scan[i] = value & 0xFF
        out[row * stride:(row + 1) * stride] = scan
    return bytes(out)


def _read_png(path: Path) -> Image:
    data = path.read_bytes()
    if not data.startswith(PNG_SIG):
        raise ValueError(f"{path}: not a PNG file")
    pos = len(PNG_SIG)
    width = height = bit_depth = color_type = None
    idat = bytearray()
    while pos + 8 <= len(data):
        length = struct.unpack(">I", data[pos:pos + 4])[0]
        chunk_type = data[pos + 4:pos + 8]
        chunk_data = data[pos + 8:pos + 8 + length]
        pos += 12 + length
        if chunk_type == b"IHDR":
            width, height, bit_depth, color_type, comp, filt, interlace = struct.unpack(
                ">IIBBBBB", chunk_data
            )
            if bit_depth != 8:
                raise ValueError(f"{path}: only 8-bit PNG is supported")
            if color_type not in (0, 2, 4, 6):
                raise ValueError(f"{path}: unsupported PNG color type {color_type}")
            if comp != 0 or filt != 0 or interlace != 0:
                raise ValueError(f"{path}: unsupported PNG compression/filter/interlace")
        elif chunk_type == b"IDAT":
            idat.extend(chunk_data)
        elif chunk_type == b"IEND":
            break
    if width is None or not idat:
        raise ValueError(f"{path}: missing IHDR or IDAT")

    channels_by_type = {0: 1, 2: 3, 4: 2, 6: 4}
    channels = channels_by_type[color_type]
    raw = zlib.decompress(bytes(idat))
    pixels = _unfilter_png(raw, width, height, channels)
    if color_type == 4:
        pixels = bytes(pixels[i] for i in range(0, len(pixels), 2))
        channels = 1
    elif color_type == 6:
        rgb = bytearray(width * height * 3)
        j = 0
        for i in range(0, len(pixels), 4):
            rgb[j:j + 3] = pixels[i:i + 3]
            j += 3
        pixels = bytes(rgb)
        channels = 3
    return Image(width, height, channels, pixels)


def load_image(path_arg: str) -> Image:
    path = Path(path_arg)
    if not path.exists():
        raise FileNotFoundError(path)
    head = path.read_bytes()[:8]
    if head.startswith(PNG_SIG):
        return _read_png(path)
    if head.startswith(b"P5"):
        return _read_pgm(path)
    raise ValueError(f"{path}: supported formats are 8-bit PNG and PGM P5")


def to_grayscale(img: Image) -> Image:
    if img.channels == 1:
        return img
    gray = bytearray(img.width * img.height)
    src = img.pixels
    for i in range(img.width * img.height):
        r = src[i * img.channels]
        g = src[i * img.channels + 1]
        b = src[i * img.channels + 2]
        if r == g == b:
            gray[i] = r
        else:
            gray[i] = int(round(0.299 * r + 0.587 * g + 0.114 * b))
    return Image(img.width, img.height, 1, bytes(gray))


def resize_nearest(img: Image, width: int, height: int) -> Image:
    if img.width == width and img.height == height:
        return img
    out = bytearray(width * height * img.channels)
    for y in range(height):
        src_y = min(img.height - 1, int(y * img.height / height))
        for x in range(width):
            src_x = min(img.width - 1, int(x * img.width / width))
            src = (src_y * img.width + src_x) * img.channels
            dst = (y * width + x) * img.channels
            out[dst:dst + img.channels] = img.pixels[src:src + img.channels]
    return Image(width, height, img.channels, bytes(out))


def compare_images(reference: Image, candidate: Image):
    if reference.channels != candidate.channels:
        raise ValueError(
            f"channel mismatch: reference={reference.channels}, candidate={candidate.channels}"
        )
    if reference.width != candidate.width or reference.height != candidate.height:
        candidate = resize_nearest(candidate, reference.width, reference.height)
    total = reference.width * reference.height * reference.channels
    diffs = [abs(reference.pixels[i] - candidate.pixels[i]) for i in range(total)]
    max_abs = max(diffs) if diffs else 0
    mean_abs = sum(diffs) / float(total or 1)
    mismatch_ratio = sum(1 for value in diffs if value > 0) / float(total or 1)
    exact = reference.pixels == candidate.pixels
    return {
        "shape": (reference.height, reference.width, reference.channels),
        "exact_match": exact,
        "mean_abs_diff": mean_abs,
        "max_abs_diff": max_abs,
        "mismatch_ratio": mismatch_ratio,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Stdlib-only pixel-level HALSS visualization comparison"
    )
    parser.add_argument("--reference", required=True, help="Reference HALSS PNG/PGM")
    parser.add_argument("--candidate", required=True, help="Current binary semantic PNG/PGM")
    parser.add_argument("--grayscale", action="store_true", help="Compare as grayscale")
    parser.add_argument("--max-mean-abs-diff", type=float, default=0.0)
    parser.add_argument("--max-pixel-diff", type=int, default=0)
    args = parser.parse_args(argv)

    ref = load_image(args.reference)
    cand = load_image(args.candidate)
    if args.grayscale:
        ref = to_grayscale(ref)
        cand = to_grayscale(cand)
    metrics = compare_images(ref, cand)

    print(f"reference={args.reference}")
    print(f"candidate={args.candidate}")
    print(f"shape={metrics['shape']}")
    print(f"exact_match={metrics['exact_match']}")
    print(f"mean_abs_diff={metrics['mean_abs_diff']:.6f}")
    print(f"max_abs_diff={metrics['max_abs_diff']}")
    print(f"mismatch_ratio={metrics['mismatch_ratio']:.6f}")

    passed = (
        metrics["mean_abs_diff"] <= args.max_mean_abs_diff
        and metrics["max_abs_diff"] <= args.max_pixel_diff
    )
    if passed:
        print("[OK] HALSS visualization comparison passed")
        return 0
    print("[FAIL] HALSS visualization comparison failed")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
