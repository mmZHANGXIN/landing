#!/usr/bin/env python3
"""Lightweight acceptance for saved no-control frame artifacts.

This script intentionally uses only the Python standard library. It checks the
files produced by ``test_live_nocontrol.py --save-raw-arrays`` and, optionally,
``--save-frames`` without importing NumPy/OpenCV/Torch.
"""

import argparse
import ast
import math
import struct
import zipfile
from pathlib import Path


REQUIRED_NPZ_KEYS = (
    "sparse_depth",
    "valid_mask",
    "dense_depth",
    "sem_map",
    "binary_semantic_vis",
    "yaw_rad",
    "cloud_odom_sync_ms",
    "pose_xyz",
    "rpy",
    "action_id",
    "v_body",
    "v_ned",
)

SMALL_VALUE_KEYS = {
    "yaw_rad",
    "cloud_odom_sync_ms",
    "action_id",
    "v_body",
    "v_ned",
}


def _parse_npy_header(raw: bytes):
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
    if len(raw) < header_end:
        raise ValueError("truncated NPY header")
    encoding = "latin1" if major < 3 else "utf-8"
    header = raw[header_start:header_end].decode(encoding).strip()
    meta = ast.literal_eval(header)
    shape = tuple(meta.get("shape", ()))
    return {
        "descr": meta.get("descr"),
        "fortran_order": bool(meta.get("fortran_order")),
        "shape": shape,
        "data_offset": header_end,
    }


def _shape_count(shape):
    count = 1
    for dim in shape:
        count *= int(dim)
    return count


def _struct_format(descr):
    if not isinstance(descr, str) or len(descr) < 2:
        raise ValueError(f"unsupported dtype descriptor {descr!r}")
    endian = descr[0]
    code = descr[1:]
    if endian == "|":
        prefix = "<"
    elif endian in ("<", ">", "="):
        prefix = "<" if endian == "=" else endian
    else:
        raise ValueError(f"unsupported dtype endian {descr!r}")
    formats = {
        "f4": "f",
        "f8": "d",
        "i1": "b",
        "u1": "B",
        "i2": "h",
        "u2": "H",
        "i4": "i",
        "u4": "I",
        "i8": "q",
        "u8": "Q",
    }
    if code not in formats:
        raise ValueError(f"unsupported dtype descriptor {descr!r}")
    return prefix + formats[code], struct.calcsize(prefix + formats[code])


def _read_small_values(raw: bytes, meta: dict, max_items: int = 3):
    shape = meta["shape"]
    count = _shape_count(shape)
    if count > max_items:
        return None
    if meta.get("fortran_order"):
        raise ValueError("Fortran-order NPY values are not supported")
    fmt, item_size = _struct_format(meta["descr"])
    start = meta["data_offset"]
    end = start + count * item_size
    if len(raw) < end:
        raise ValueError("truncated NPY data")
    values = []
    for idx in range(count):
        item = raw[start + idx * item_size:start + (idx + 1) * item_size]
        values.append(struct.unpack(fmt, item)[0])
    return values


def _inspect_npz(path: Path):
    arrays = {}
    with zipfile.ZipFile(path) as zf:
        for member in zf.namelist():
            if not member.endswith(".npy"):
                continue
            key = Path(member).stem
            with zf.open(member) as fp:
                raw = fp.read()
            meta = _parse_npy_header(raw)
            if key in SMALL_VALUE_KEYS:
                try:
                    meta["values"] = _read_small_values(raw, meta)
                except Exception as exc:
                    meta["value_error"] = str(exc)
            arrays[key] = meta
    return arrays


def _as_scalar(arrays, key):
    values = arrays[key].get("values")
    if values is None or len(values) != 1:
        raise ValueError(f"{key} value is unavailable")
    return values[0]


def _as_vector(arrays, key, expected_len=3):
    values = arrays[key].get("values")
    if values is None or len(values) != expected_len:
        raise ValueError(f"{key} vector value is unavailable")
    return [float(value) for value in values]


def _is_finite_number(value):
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def _png_size(path: Path):
    with path.open("rb") as fp:
        sig = fp.read(8)
        if sig != b"\x89PNG\r\n\x1a\n":
            raise ValueError("not a PNG")
        length = struct.unpack(">I", fp.read(4))[0]
        chunk_type = fp.read(4)
        if chunk_type != b"IHDR" or length < 8:
            raise ValueError("PNG missing IHDR")
        width, height = struct.unpack(">II", fp.read(8))
    return width, height


def _check_raw_npz(path: Path, grid_size: int,
                   max_sync_ms: float, max_yaw_transform_error: float):
    failures = []
    try:
        arrays = _inspect_npz(path)
    except Exception as exc:
        return [f"{path.name}: invalid NPZ/NPY payload ({exc})"], {}
    missing = [key for key in REQUIRED_NPZ_KEYS if key not in arrays]
    if missing:
        failures.append(f"{path.name}: missing keys {missing}")
        return failures, arrays

    expected_grid = {
        "sparse_depth",
        "valid_mask",
        "dense_depth",
        "sem_map",
    }
    for key in expected_grid:
        shape = arrays[key]["shape"]
        if shape != (grid_size, grid_size):
            failures.append(f"{path.name}: {key} shape {shape} != {(grid_size, grid_size)}")

    binary_shape = arrays["binary_semantic_vis"]["shape"]
    if binary_shape not in ((grid_size, grid_size), (grid_size, grid_size, 3)):
        failures.append(
            f"{path.name}: binary_semantic_vis shape {binary_shape} is not "
            f"{(grid_size, grid_size)} or {(grid_size, grid_size, 3)}"
        )

    vector_shapes = {
        "pose_xyz": (3,),
        "rpy": (3,),
        "v_body": (3,),
        "v_ned": (3,),
    }
    for key, expected in vector_shapes.items():
        shape = arrays[key]["shape"]
        if shape != expected:
            failures.append(f"{path.name}: {key} shape {shape} != {expected}")

    for key in ("yaw_rad", "cloud_odom_sync_ms", "action_id"):
        shape = arrays[key]["shape"]
        if shape not in ((), (1,)):
            failures.append(f"{path.name}: {key} shape {shape} is not scalar")

    for key in SMALL_VALUE_KEYS:
        error = arrays[key].get("value_error")
        if error:
            failures.append(f"{path.name}: {key} value cannot be read ({error})")

    if not failures:
        try:
            yaw_rad = float(_as_scalar(arrays, "yaw_rad"))
            sync_ms = float(_as_scalar(arrays, "cloud_odom_sync_ms"))
            action_id = int(_as_scalar(arrays, "action_id"))
            v_body = _as_vector(arrays, "v_body")
            v_ned = _as_vector(arrays, "v_ned")
        except Exception as exc:
            failures.append(f"{path.name}: scalar/vector diagnostic values unavailable ({exc})")
            return failures, arrays

        values_to_check = [yaw_rad, sync_ms, float(action_id), *v_body, *v_ned]
        if not all(_is_finite_number(value) for value in values_to_check):
            failures.append(f"{path.name}: diagnostic scalar/vector values contain NaN/inf")

        if not (0 <= action_id <= 9):
            failures.append(f"{path.name}: action_id {action_id} is outside 0..9")

        if sync_ms < 0.0 or sync_ms > max_sync_ms:
            failures.append(
                f"{path.name}: cloud_odom_sync_ms {sync_ms:.1f}ms > {max_sync_ms:.1f}ms"
            )

        c = math.cos(yaw_rad)
        s = math.sin(yaw_rad)
        expected = [
            c * v_body[0] - s * v_body[1],
            s * v_body[0] + c * v_body[1],
            v_body[2],
        ]
        yaw_error = math.sqrt(sum((expected[idx] - v_ned[idx]) ** 2 for idx in range(3)))
        if yaw_error > max_yaw_transform_error:
            failures.append(
                f"{path.name}: yaw transform error {yaw_error:.3f} > "
                f"{max_yaw_transform_error:.3f}"
            )

    return failures, arrays


def _prefix(path: Path, suffix: str):
    name = path.name
    if not name.endswith(suffix):
        return None
    return name[: -len(suffix)]


def inspect_artifacts(frame_dir: Path, grid_size: int, min_raw: int,
                      min_frame_pairs: int, max_sync_ms: float,
                      max_yaw_transform_error: float):
    raw_files = sorted(frame_dir.glob("*_calib_frame.npz"))
    sem_files = sorted(frame_dir.glob("*_binary_semantic.png"))
    depth_files = sorted(frame_dir.glob("*_depth.png"))

    sem_by_prefix = {_prefix(path, "_binary_semantic.png"): path for path in sem_files}
    depth_by_prefix = {_prefix(path, "_depth.png"): path for path in depth_files}
    frame_pair_prefixes = sorted(set(sem_by_prefix) & set(depth_by_prefix))
    raw_prefixes = {_prefix(path, "_calib_frame.npz") for path in raw_files}

    failures = []
    if len(raw_files) < min_raw:
        failures.append(f"only {len(raw_files)} raw NPZ files; need {min_raw}")
    if len(frame_pair_prefixes) < min_frame_pairs:
        failures.append(f"only {len(frame_pair_prefixes)} frame pairs; need {min_frame_pairs}")

    inspected_raw = 0
    for path in raw_files[: max(min_raw, 1)]:
        raw_failures, _arrays = _check_raw_npz(
            path,
            grid_size,
            max_sync_ms=max_sync_ms,
            max_yaw_transform_error=max_yaw_transform_error,
        )
        failures.extend(raw_failures)
        inspected_raw += 1

    inspected_pairs = 0
    for prefix in frame_pair_prefixes[: max(min_frame_pairs, 1)]:
        for label, path in (
            ("binary semantic", sem_by_prefix[prefix]),
            ("depth", depth_by_prefix[prefix]),
        ):
            try:
                width, height = _png_size(path)
            except Exception as exc:
                failures.append(f"{path.name}: invalid {label} PNG ({exc})")
                continue
            if width <= 0 or height <= 0:
                failures.append(f"{path.name}: invalid {label} PNG size {width}x{height}")
        inspected_pairs += 1

    if raw_files and frame_pair_prefixes:
        matched = raw_prefixes & set(frame_pair_prefixes)
        if not matched:
            failures.append("raw NPZ prefixes do not match saved frame prefixes")

    return {
        "raw_files": raw_files,
        "semantic_pngs": sem_files,
        "depth_pngs": depth_files,
        "frame_pair_prefixes": frame_pair_prefixes,
        "inspected_raw": inspected_raw,
        "inspected_pairs": inspected_pairs,
        "failures": failures,
    }


def main():
    parser = argparse.ArgumentParser(description="Inspect no-control saved artifacts")
    parser.add_argument("frame_dir", help="Directory containing experiments/frames outputs")
    parser.add_argument("--grid-size", type=int, default=128)
    parser.add_argument("--min-raw", type=int, default=1,
                        help="Minimum *_calib_frame.npz files required")
    parser.add_argument("--min-frame-pairs", type=int, default=0,
                        help="Minimum matching *_binary_semantic.png and *_depth.png pairs")
    parser.add_argument("--require-frames", action="store_true",
                        help="Require at least one binary semantic/depth PNG pair")
    parser.add_argument("--max-sync-ms", type=float, default=100.0,
                        help="Fail if a saved cloud/odom sync scalar exceeds this value")
    parser.add_argument("--max-yaw-transform-error", type=float, default=0.15,
                        help="Fail if saved v_ned differs from yaw-rotated v_body")
    args = parser.parse_args()

    min_frame_pairs = max(args.min_frame_pairs, 1) if args.require_frames else args.min_frame_pairs
    result = inspect_artifacts(
        Path(args.frame_dir),
        grid_size=args.grid_size,
        min_raw=args.min_raw,
        min_frame_pairs=min_frame_pairs,
        max_sync_ms=args.max_sync_ms,
        max_yaw_transform_error=args.max_yaw_transform_error,
    )

    print("No-control artifact acceptance summary")
    print(f"  raw_npz: {len(result['raw_files'])}")
    print(f"  binary_semantic_png: {len(result['semantic_pngs'])}")
    print(f"  depth_png: {len(result['depth_pngs'])}")
    print(f"  frame_pairs: {len(result['frame_pair_prefixes'])}")
    print(f"  inspected_raw: {result['inspected_raw']}")
    print(f"  inspected_pairs: {result['inspected_pairs']}")

    if result["failures"]:
        for failure in result["failures"]:
            print(f"[FAIL] {failure}")
        return 1
    print("[OK] no-control artifacts accepted")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
