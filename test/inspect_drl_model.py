#!/usr/bin/env python3
"""Inspect SB3 policy metadata without importing torch/stable-baselines3."""

import argparse
import json
import re
import sys
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent


def _extract_scalar(pattern: str, text: str):
    match = re.search(pattern, text, flags=re.DOTALL)
    return match.group(1) if match else None


def inspect_sb3_zip(path_value) -> dict:
    path = Path(path_value)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    result = {
        "path": str(path),
        "exists": path.is_file(),
        "is_zip": False,
        "shape": None,
        "low_repr": None,
        "high_repr": None,
        "normalize_images": None,
        "features_extractor": None,
        "net_arch": None,
        "warnings": [],
        "failures": [],
    }
    if not result["exists"]:
        result["failures"].append(f"missing policy file: {path}")
        return result

    try:
        with zipfile.ZipFile(path, "r") as zf:
            result["is_zip"] = True
            raw = zf.read("data").decode("utf-8", errors="replace")
    except Exception as exc:
        result["failures"].append(f"cannot read SB3 zip metadata: {exc}")
        return result

    shape_match = re.search(r'"_shape"\s*:\s*\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]', raw)
    if shape_match:
        result["shape"] = tuple(int(item) for item in shape_match.groups())
    result["low_repr"] = _extract_scalar(r'"low_repr"\s*:\s*"([^"]+)"', raw)
    result["high_repr"] = _extract_scalar(r'"high_repr"\s*:\s*"([^"]+)"', raw)
    norm_match = re.search(r'"normalize_images"\s*:\s*(true|false)', raw)
    if norm_match:
        result["normalize_images"] = norm_match.group(1) == "true"
    feat_match = re.search(r'"features_extractor_class"\s*:\s*"([^"]+)"', raw)
    if feat_match:
        result["features_extractor"] = feat_match.group(1)

    pi_match = re.search(r'"net_arch"\s*:\s*\{\s*"pi"\s*:\s*\[([^\]]*)\]\s*,\s*"vf"\s*:\s*\[([^\]]*)\]', raw)
    if pi_match:
        def parse_list(value):
            return [int(x.strip()) for x in value.split(",") if x.strip()]
        result["net_arch"] = {"pi": parse_list(pi_match.group(1)), "vf": parse_list(pi_match.group(2))}

    shape = result["shape"]
    if shape != (128, 128, 2):
        result["warnings"].append(f"expected HWC observation shape (128,128,2), got {shape}")
    if result["normalize_images"] is not False:
        result["warnings"].append(
            f"expected normalize_images=false for pre-encoded 0..1 observations, got {result['normalize_images']}"
        )
    if result["high_repr"] != "1.0":
        result["warnings"].append(
            f"expected converted SB3 observation high_repr=1.0, got {result['high_repr']}"
        )
    feature_name = result["features_extractor"] or ""
    if "SB2CNN" not in feature_name:
        result["warnings"].append(
            f"expected custom SB2CNN feature extractor, got {result['features_extractor']}"
        )
    if result["net_arch"] and result["net_arch"] != {"pi": [64, 64], "vf": [64, 32]}:
        result["warnings"].append(f"unexpected net_arch: {result['net_arch']}")
    return result


def format_report(result: dict) -> str:
    lines = [
        f"path: {result['path']}",
        f"exists: {result['exists']}  zip: {result['is_zip']}",
        f"shape: {result['shape']}",
        f"low/high: {result['low_repr']} / {result['high_repr']}",
        f"normalize_images: {result['normalize_images']}",
        f"features_extractor: {result['features_extractor']}",
        f"net_arch: {result['net_arch']}",
    ]
    for warning in result["warnings"]:
        lines.append(f"[WARN] {warning}")
    for failure in result["failures"]:
        lines.append(f"[FAIL] {failure}")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Inspect SB3 DRL model metadata")
    parser.add_argument("model", nargs="?", default=str(PROJECT_ROOT / "weights" / "last_step_model_sb3.zip"))
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    result = inspect_sb3_zip(args.model)
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(format_report(result))
    return 1 if result["failures"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
