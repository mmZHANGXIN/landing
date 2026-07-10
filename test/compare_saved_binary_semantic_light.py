#!/usr/bin/env python3
"""Compare saved binary semantic PNGs against NPZ ``binary_semantic_vis``.

This verifies that the frame saved/displayed by ``test_live_nocontrol.py`` is
the HALSS binary visualization stored in the same-frame calibration NPZ. It is
stdlib-only and supports uint8 NPY arrays inside NPZ files.
"""

from __future__ import annotations

import argparse
import ast
import struct
import zipfile
from pathlib import Path

from compare_halss_visualization_light import Image, compare_images, load_image, to_grayscale


def _parse_npy(raw: bytes):
    if not raw.startswith(b"\x93NUMPY"):
        raise ValueError("missing NPY magic")
    major = raw[6]
    minor = raw[7]
    if (major, minor) == (1, 0):
        header_len = struct.unpack("<H", raw[8:10])[0]
        header_start = 10
    elif major in (2, 3):
        header_len = struct.unpack("<I", raw[8:12])[0]
        header_start = 12
    else:
        raise ValueError(f"unsupported NPY version {major}.{minor}")
    header_end = header_start + header_len
    encoding = "latin1" if major < 3 else "utf-8"
    header = raw[header_start:header_end].decode(encoding).strip()
    meta = ast.literal_eval(header)
    shape = tuple(meta["shape"])
    descr = meta["descr"]
    fortran_order = bool(meta["fortran_order"])
    data = raw[header_end:]
    return shape, descr, fortran_order, data


def load_binary_semantic_npz(path: Path) -> Image:
    with zipfile.ZipFile(path) as zf:
        member = None
        for name in zf.namelist():
            if Path(name).stem == "binary_semantic_vis" and name.endswith(".npy"):
                member = name
                break
        if member is None:
            raise ValueError(f"{path}: binary_semantic_vis.npy is missing")
        shape, descr, fortran_order, data = _parse_npy(zf.read(member))

    if fortran_order:
        raise ValueError(f"{path}: fortran_order arrays are not supported")
    if descr not in ("|u1", "uint8"):
        raise ValueError(f"{path}: binary_semantic_vis dtype must be uint8, got {descr!r}")
    if len(shape) == 2:
        height, width = shape
        channels = 1
    elif len(shape) == 3 and shape[2] in (1, 3):
        height, width, channels = shape
    else:
        raise ValueError(f"{path}: unsupported binary_semantic_vis shape {shape}")
    expected = height * width * channels
    if len(data) < expected:
        raise ValueError(f"{path}: truncated binary_semantic_vis data")
    return Image(width=width, height=height, channels=channels, pixels=data[:expected])


def _prefix(path: Path, suffix: str):
    name = path.name
    if not name.endswith(suffix):
        return None
    return name[: -len(suffix)]


def _compare_pair(npz_path: Path, png_path: Path, grayscale: bool):
    reference = load_binary_semantic_npz(npz_path)
    candidate = load_image(str(png_path))
    if grayscale:
        reference = to_grayscale(reference)
        candidate = to_grayscale(candidate)
    return compare_images(reference, candidate)


def compare_dir(frame_dir: Path, max_pairs: int, grayscale: bool):
    npz_files = sorted(frame_dir.glob("*_calib_frame.npz"))
    png_files = sorted(frame_dir.glob("*_binary_semantic.png"))
    npz_by_prefix = {_prefix(path, "_calib_frame.npz"): path for path in npz_files}
    png_by_prefix = {_prefix(path, "_binary_semantic.png"): path for path in png_files}
    prefixes = sorted(set(npz_by_prefix) & set(png_by_prefix))
    if max_pairs > 0:
        prefixes = prefixes[:max_pairs]
    results = []
    for prefix in prefixes:
        metrics = _compare_pair(npz_by_prefix[prefix], png_by_prefix[prefix], grayscale)
        results.append({
            "prefix": prefix,
            "npz": str(npz_by_prefix[prefix]),
            "png": str(png_by_prefix[prefix]),
            "metrics": metrics,
        })
    return {
        "npz_count": len(npz_files),
        "png_count": len(png_files),
        "pair_count": len(prefixes),
        "results": results,
    }


def main():
    parser = argparse.ArgumentParser(description="Compare saved binary semantic PNG to NPZ")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--frame-dir", help="Directory containing saved no-control frames")
    group.add_argument("--npz", help="Single *_calib_frame.npz")
    parser.add_argument("--png", help="Single *_binary_semantic.png, required with --npz")
    parser.add_argument("--grayscale", action="store_true",
                        help="Compare grayscale values; recommended for HALSS binary maps")
    parser.add_argument("--max-pairs", type=int, default=10)
    parser.add_argument("--max-mean-abs-diff", type=float, default=0.0)
    parser.add_argument("--max-pixel-diff", type=int, default=0)
    args = parser.parse_args()

    if args.npz:
        if not args.png:
            parser.error("--png is required when --npz is used")
        result = {
            "npz_count": 1,
            "png_count": 1,
            "pair_count": 1,
            "results": [{
                "prefix": Path(args.npz).name,
                "npz": args.npz,
                "png": args.png,
                "metrics": _compare_pair(Path(args.npz), Path(args.png), args.grayscale),
            }],
        }
    else:
        result = compare_dir(Path(args.frame_dir), args.max_pairs, args.grayscale)

    print("Saved binary semantic comparison summary")
    print(f"  npz_count: {result['npz_count']}")
    print(f"  png_count: {result['png_count']}")
    print(f"  compared_pairs: {result['pair_count']}")

    failures = []
    if result["pair_count"] <= 0:
        failures.append("no matching *_calib_frame.npz and *_binary_semantic.png pairs found")
    for item in result["results"]:
        metrics = item["metrics"]
        print(
            f"  pair={item['prefix']} shape={metrics['shape']} "
            f"exact={metrics['exact_match']} mean={metrics['mean_abs_diff']:.6f} "
            f"max={metrics['max_abs_diff']} mismatch={metrics['mismatch_ratio']:.6f}"
        )
        if metrics["mean_abs_diff"] > args.max_mean_abs_diff:
            failures.append(
                f"{item['prefix']}: mean_abs_diff {metrics['mean_abs_diff']:.6f} "
                f"> {args.max_mean_abs_diff:.6f}"
            )
        if metrics["max_abs_diff"] > args.max_pixel_diff:
            failures.append(
                f"{item['prefix']}: max_abs_diff {metrics['max_abs_diff']} "
                f"> {args.max_pixel_diff}"
            )

    if failures:
        for failure in failures:
            print(f"[FAIL] {failure}")
        return 1
    print("[OK] saved binary semantic matches NPZ binary_semantic_vis")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
