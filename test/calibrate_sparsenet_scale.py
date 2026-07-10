#!/usr/bin/env python3
"""Estimate SparseNet output_scale from one saved calibration frame.

The converted TF checkpoint does not contain the original bias variables, so
its normalized output can be attenuated. This utility estimates a scalar
multiplier in encoded depth space:

  inverse_unit: encoded = 1 - depth_m / dmax
  unit:         encoded = depth_m / dmax

It does not edit the experiment config unless the operator copies the printed
value into depth_completion.output_scale.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _load_config(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _resolve(path_value):
    if path_value is None:
        return None
    path = Path(path_value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _load_array(path_value, key=None):
    path = Path(path_value)
    if path.suffix.lower() == ".npz":
        data = np.load(path)
        if key is None and len(data.files) == 1:
            key = data.files[0]
        if key is None:
            raise ValueError(f"NPZ input requires a key: {path}. Available: {list(data.keys())}")
        if key not in data:
            raise KeyError(f"Missing key '{key}' in {path}. Available: {list(data.keys())}")
        return data[key]
    if path.suffix.lower() == ".npy":
        return np.load(path)
    raise ValueError(f"Unsupported array file: {path}. Use .npy or .npz")


def _load_frame(args):
    if args.input:
        path = Path(args.input)
        data = np.load(path)
        if "sparse_depth" not in data:
            raise KeyError(f"{path} does not contain sparse_depth")
        sparse_depth = data["sparse_depth"].astype(np.float32)
        valid_mask = data["valid_mask"].astype(bool) if "valid_mask" in data else None
        reference_depth = data["reference_depth"].astype(np.float32) if "reference_depth" in data else None
        return sparse_depth, valid_mask, reference_depth

    if not args.sparse_depth:
        raise ValueError("Provide --input <frame.npz> or --sparse-depth <array.npy>")
    sparse_depth = _load_array(args.sparse_depth).astype(np.float32)
    valid_mask = _load_array(args.valid_mask).astype(bool) if args.valid_mask else None
    reference_depth = _load_array(args.reference_depth).astype(np.float32) if args.reference_depth else None
    return sparse_depth, valid_mask, reference_depth


def _encode_depth(depth_m: np.ndarray, mode: str, dmax: float) -> np.ndarray:
    depth = np.nan_to_num(depth_m, nan=dmax, posinf=dmax, neginf=0.0)
    depth = np.clip(depth, 0.0, dmax).astype(np.float32)
    if mode == "inverse_unit":
        return np.clip(1.0 - depth / dmax, 0.0, 1.0)
    if mode == "unit":
        return np.clip(depth / dmax, 0.0, 1.0)
    raise ValueError(f"Unsupported depth_completion.input_encoding: {mode}")


def _decode_depth(encoded: np.ndarray, mode: str, dmax: float) -> np.ndarray:
    encoded = np.clip(encoded, 0.0, 1.0).astype(np.float32)
    if mode == "inverse_unit":
        return (1.0 - encoded) * dmax
    if mode == "unit":
        return encoded * dmax
    raise ValueError(f"Unsupported depth_completion.input_encoding: {mode}")


def _summarize_error(pred_depth, target_depth, mask):
    err = np.abs(pred_depth[mask] - target_depth[mask])
    return {
        "median_abs_m": float(np.median(err)),
        "mean_abs_m": float(np.mean(err)),
        "p90_abs_m": float(np.percentile(err, 90)),
    }


def estimate_scale(args):
    cfg = _load_config(Path(args.config))
    dcfg = dict(cfg["depth_completion"])
    dmax = float(dcfg.get("dmax", cfg["depth_projection"].get("max_range", 30.0)))
    encoding = dcfg.get("input_encoding", "inverse_unit")

    sparse_depth, valid_mask, embedded_reference = _load_frame(args)
    if valid_mask is None:
        valid_mask = np.isfinite(sparse_depth) & (sparse_depth > 0.01) & (sparse_depth < dmax)
    valid_mask = valid_mask.astype(bool)

    if args.reference_depth:
        reference_depth = _load_array(args.reference_depth, args.reference_key).astype(np.float32)
    else:
        reference_depth = embedded_reference

    dcfg["weight_path"] = str(_resolve(dcfg.get("weight_path")))
    dcfg["output_scale"] = None

    from perception.sparse_depth_completion import DepthCompletion

    dc = DepthCompletion(dcfg)
    dense_unscaled = dc.complete(sparse_depth, valid_mask)
    pred_encoded = _encode_depth(dense_unscaled, encoding, dmax)

    if reference_depth is not None:
        target_depth = reference_depth.astype(np.float32)
        fit_mask = np.isfinite(target_depth) & (target_depth > 0.01) & (target_depth < dmax)
        source = "reference_depth"
        if not args.fit_all_reference:
            fit_mask &= valid_mask
    elif args.target_depth_m is not None:
        target_depth = np.full_like(sparse_depth, float(args.target_depth_m), dtype=np.float32)
        fit_mask = valid_mask.copy()
        source = f"target_depth_m={args.target_depth_m:.3f}"
    else:
        target_depth = sparse_depth.astype(np.float32)
        fit_mask = valid_mask.copy()
        source = "sparse_depth_observed_pixels"

    target_encoded = _encode_depth(target_depth, encoding, dmax)
    ratio_mask = (
        fit_mask
        & np.isfinite(pred_encoded)
        & np.isfinite(target_encoded)
        & (pred_encoded > args.min_encoded)
        & (target_encoded > args.min_encoded)
    )
    n = int(np.count_nonzero(ratio_mask))
    if n < args.min_pixels:
        raise RuntimeError(
            f"Only {n} usable calibration pixels. Need at least {args.min_pixels}."
        )

    ratios = target_encoded[ratio_mask] / pred_encoded[ratio_mask]
    scale = float(np.median(ratios))
    scaled_encoded = np.clip(pred_encoded * scale, 0.0, 1.0)
    dense_scaled = _decode_depth(scaled_encoded, encoding, dmax)

    raw_error = _summarize_error(dense_unscaled, target_depth, ratio_mask)
    scaled_error = _summarize_error(dense_scaled, target_depth, ratio_mask)
    report = {
        "source": source,
        "encoding": encoding,
        "dmax": dmax,
        "pixels_used": n,
        "valid_ratio": float(np.mean(valid_mask)),
        "suggested_output_scale": scale,
        "ratio_p10": float(np.percentile(ratios, 10)),
        "ratio_p50": float(np.percentile(ratios, 50)),
        "ratio_p90": float(np.percentile(ratios, 90)),
        "raw_error": raw_error,
        "scaled_error": scaled_error,
    }
    return report


def main():
    parser = argparse.ArgumentParser(description="Calibrate depth_completion.output_scale")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "config" / "experiment_config.yaml"))
    parser.add_argument("--input", help="NPZ saved by test_live_nocontrol.py with save_raw_arrays=true")
    parser.add_argument("--sparse-depth", help="Sparse depth .npy, meters")
    parser.add_argument("--valid-mask", help="Optional valid mask .npy")
    parser.add_argument("--reference-depth", help="Optional reference dense depth .npy/.npz, meters")
    parser.add_argument("--reference-key", default="reference_depth",
                        help="Array key when --reference-depth is an NPZ")
    parser.add_argument("--target-depth-m", type=float,
                        help="Known flat-plane depth in meters; otherwise use sparse observed pixels")
    parser.add_argument("--fit-all-reference", action="store_true",
                        help="When reference depth is given, fit all valid reference pixels instead of sparse overlap")
    parser.add_argument("--min-encoded", type=float, default=1e-5)
    parser.add_argument("--min-pixels", type=int, default=100)
    parser.add_argument("--out-json", help="Optional path to write the calibration report")
    args = parser.parse_args()

    report = estimate_scale(args)
    print("\nSparseNet output_scale calibration")
    print(f"  source: {report['source']}")
    print(f"  encoding: {report['encoding']}  dmax={report['dmax']:.3f}m")
    print(f"  pixels_used: {report['pixels_used']}  valid_ratio={report['valid_ratio']:.4f}")
    print(
        "  ratio p10/p50/p90: "
        f"{report['ratio_p10']:.4g} / {report['ratio_p50']:.4g} / {report['ratio_p90']:.4g}"
    )
    print(f"\nSuggested config:")
    print(f"  depth_completion.output_scale: {report['suggested_output_scale']:.6g}")
    print("\nMedian absolute error at fit pixels:")
    print(f"  raw:    {report['raw_error']['median_abs_m']:.3f} m")
    print(f"  scaled: {report['scaled_error']['median_abs_m']:.3f} m")

    if report["suggested_output_scale"] > 100.0:
        print("\nWARNING: suggested scale is very large; check encoding and calibration depth.")
    if args.out_json:
        out_path = Path(args.out_json)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"\nWrote report: {out_path}")


if __name__ == "__main__":
    main()
