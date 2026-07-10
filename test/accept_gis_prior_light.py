#!/usr/bin/env python3
"""Lightweight acceptance for saved GIS nine-grid prior summaries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load_latest(path: Path):
    if path.is_dir():
        files = sorted(path.glob("global_prior_*.json"))
        if not files:
            raise FileNotFoundError(f"{path}: no global_prior_*.json files")
        return files[-1], json.loads(files[-1].read_text(encoding="utf-8"))
    return path, json.loads(path.read_text(encoding="utf-8"))


def _parse_bounds(value):
    if value is None:
        return None
    parts = [float(part) for part in value.split(",")]
    if len(parts) != 4:
        raise ValueError("--bounds must be lon_left,lat_bottom,lon_right,lat_top")
    lon_left, lat_bottom, lon_right, lat_top = parts
    if lon_left >= lon_right or lat_bottom >= lat_top:
        raise ValueError("--bounds are not ordered")
    return lon_left, lat_bottom, lon_right, lat_top


def _risk_grid_ok(grid):
    return (
        isinstance(grid, list)
        and len(grid) == 3
        and all(isinstance(row, list) and len(row) == 3 for row in grid)
    )


def _best_cell(grid):
    best = None
    for row, values in enumerate(grid):
        for col, value in enumerate(values):
            risk = float(value)
            if best is None or risk < best["risk"]:
                best = {"row": row, "col": col, "risk": risk}
    return best


def evaluate(data, bounds=None, require_gps=True, max_min_risk=None,
             require_traceability=True):
    failures = []
    source_image = data.get("source_image_path")
    source_sem_mask = data.get("source_sem_mask_path")
    image_size = data.get("image_size_px")
    segmentation_source = data.get("segmentation_source")

    if require_traceability:
        if not source_image:
            failures.append("source_image_path is required for GIS traceability")
        if not (isinstance(image_size, list) and len(image_size) == 2):
            failures.append("image_size_px must be [width, height]")
        if bounds is None and data.get("bounds") is None:
            failures.append("bounds are required for GIS traceability")
        if not source_sem_mask and not segmentation_source:
            failures.append(
                "source_sem_mask_path or segmentation_source is required for GIS traceability"
            )

    grid = data.get("risk_grid")
    if not _risk_grid_ok(grid):
        failures.append("risk_grid must be a 3x3 matrix")
        best = None
    else:
        best = _best_cell(grid)

    min_risk = data.get("min_risk")
    if min_risk is None:
        failures.append("min_risk is missing")
    elif best is not None and abs(float(min_risk) - best["risk"]) > 1e-6:
        failures.append(
            f"min_risk {float(min_risk):.6f} does not match risk_grid best {best['risk']:.6f}"
        )
    elif max_min_risk is not None and float(min_risk) > max_min_risk:
        failures.append(f"min_risk {float(min_risk):.6f} > {max_min_risk:.6f}")

    recorded_best_cell = data.get("best_cell")
    if recorded_best_cell is not None:
        if not (isinstance(recorded_best_cell, list) and len(recorded_best_cell) == 2):
            failures.append("best_cell must be [row, col]")
        elif best is not None:
            row, col = int(recorded_best_cell[0]), int(recorded_best_cell[1])
            if row != best["row"] or col != best["col"]:
                failures.append(
                    f"best_cell {recorded_best_cell} does not match risk_grid best "
                    f"[{best['row']}, {best['col']}]"
                )

    cell_bounds = data.get("best_cell_bounds_px")
    if cell_bounds is not None and not (isinstance(cell_bounds, list) and len(cell_bounds) == 4):
        failures.append("best_cell_bounds_px must be [x0, y0, x1, y1]")

    px = data.get("best_center_px")
    if not (isinstance(px, list) and len(px) == 2):
        failures.append("best_center_px must be [x, y]")
    elif cell_bounds is not None and isinstance(cell_bounds, list) and len(cell_bounds) == 4:
        x, y = float(px[0]), float(px[1])
        x0, y0, x1, y1 = [float(value) for value in cell_bounds]
        if not (x0 <= x <= x1 and y0 <= y <= y1):
            failures.append("best_center_px is outside best_cell_bounds_px")

    gps = data.get("best_center_gps")
    has_gps = data.get("has_gps_target")
    if require_gps:
        if has_gps is not True:
            failures.append("has_gps_target must be true")
        if not (isinstance(gps, list) and len(gps) == 2):
            failures.append("best_center_gps must be [lat, lon]")
    elif gps is not None and not (isinstance(gps, list) and len(gps) == 2):
        failures.append("best_center_gps must be null or [lat, lon]")

    if bounds is not None and isinstance(gps, list) and len(gps) == 2:
        lon_left, lat_bottom, lon_right, lat_top = bounds
        recorded_bounds = data.get("bounds")
        if recorded_bounds is not None:
            if not (isinstance(recorded_bounds, list) and len(recorded_bounds) == 4):
                failures.append("bounds must be [lon_left, lat_bottom, lon_right, lat_top]")
            else:
                for idx, (actual, expected) in enumerate(zip(recorded_bounds, bounds)):
                    if abs(float(actual) - float(expected)) > 1e-9:
                        failures.append(
                            f"recorded bounds index {idx}={float(actual):.9f} "
                            f"does not match expected {float(expected):.9f}"
                        )
        lat, lon = float(gps[0]), float(gps[1])
        if not (lat_bottom <= lat <= lat_top):
            failures.append(f"best_center_gps latitude {lat:.7f} outside bounds")
        if not (lon_left <= lon <= lon_right):
            failures.append(f"best_center_gps longitude {lon:.7f} outside bounds")

    return failures, {
        "best_cell": best,
        "recorded_best_cell": recorded_best_cell,
        "min_risk": None if min_risk is None else float(min_risk),
        "best_center_px": px,
        "best_center_gps": gps,
        "has_gps_target": has_gps,
        "bounds": data.get("bounds"),
        "source_image_path": source_image,
        "source_sem_mask_path": source_sem_mask,
        "segmentation_source": segmentation_source,
        "image_size_px": image_size,
    }


def main():
    parser = argparse.ArgumentParser(description="Accept saved GIS global-prior JSON")
    parser.add_argument("path", help="global_prior_*.json file or directory containing one")
    parser.add_argument("--bounds", default=None,
                        help="lon_left,lat_bottom,lon_right,lat_top")
    parser.add_argument("--allow-missing-gps", action="store_true",
                        help="Allow best_center_gps=null for negative tests")
    parser.add_argument("--allow-missing-traceability", action="store_true",
                        help="Allow old/negative fixtures without source image, mask/model, or image size")
    parser.add_argument("--max-min-risk", type=float, default=None)
    args = parser.parse_args()

    bounds = _parse_bounds(args.bounds)
    source, data = _load_latest(Path(args.path))
    failures, summary = evaluate(
        data,
        bounds=bounds,
        require_gps=not args.allow_missing_gps,
        max_min_risk=args.max_min_risk,
        require_traceability=not args.allow_missing_traceability,
    )

    print("GIS prior acceptance summary")
    print(f"  source: {source}")
    print(f"  best_cell: {summary['best_cell']}")
    print(f"  recorded_best_cell: {summary['recorded_best_cell']}")
    print(f"  min_risk: {summary['min_risk']}")
    print(f"  best_center_px: {summary['best_center_px']}")
    print(f"  best_center_gps: {summary['best_center_gps']}")
    print(f"  has_gps_target: {summary['has_gps_target']}")
    print(f"  bounds: {summary['bounds']}")
    print(f"  source_image_path: {summary['source_image_path']}")
    print(f"  source_sem_mask_path: {summary['source_sem_mask_path']}")
    print(f"  segmentation_source: {summary['segmentation_source']}")
    print(f"  image_size_px: {summary['image_size_px']}")

    if failures:
        for failure in failures:
            print(f"[FAIL] {failure}")
        return 1
    print("[OK] GIS global prior accepted")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
