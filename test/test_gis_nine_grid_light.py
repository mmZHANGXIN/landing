#!/usr/bin/env python3
"""Lightweight GIS nine-grid acceptance test without numpy/cv2.

This keeps the global-prior contract testable on machines that do not have the
Orin perception stack installed. The full image/mask path is still covered by
``test_gis_nine_grid.py`` on Orin.
"""

RISK_LUT = {
    1: 0.0,  # Terrain / safest
    0: 1.0,  # Pavement
    5: 2.0,  # Vegetation
    4: 3.0,  # Building
}
DEFAULT_RISK = 4.0


def _make_mask(size=300):
    mask = [[9 for _ in range(size)] for _ in range(size)]
    # Put the only safe block in the center cell. With a 300x300 image and a
    # 3x3 grid, this should make cell (row=1, col=1) the unique lowest-risk cell.
    for y in range(100, 200):
        for x in range(100, 200):
            mask[y][x] = 1
    return mask


def _cell_mean_risk(mask, x0, x1, y0, y1):
    total = 0.0
    count = 0
    for y in range(y0, y1):
        row = mask[y]
        for x in range(x0, x1):
            total += RISK_LUT.get(row[x], DEFAULT_RISK)
            count += 1
    return total / count


def assess_nine_grid(mask):
    h = len(mask)
    w = len(mask[0])
    ch = h // 3
    cw = w // 3
    risk_grid = []
    best = {"risk": float("inf"), "row": 0, "col": 0, "cx": w // 2, "cy": h // 2}
    for row in range(3):
        risk_row = []
        for col in range(3):
            y0 = row * ch
            y1 = (row + 1) * ch if row < 2 else h
            x0 = col * cw
            x1 = (col + 1) * cw if col < 2 else w
            risk = _cell_mean_risk(mask, x0, x1, y0, y1)
            risk_row.append(risk)
            if risk < best["risk"]:
                best = {
                    "risk": risk,
                    "row": row,
                    "col": col,
                    "cx": (x0 + x1) // 2,
                    "cy": (y0 + y1) // 2,
                }
        risk_grid.append(risk_row)
    return best, risk_grid


def pixel_to_gps(px, py, bounds, img_size):
    lon_left, lat_bottom, lon_right, lat_top = bounds
    width, height = img_size
    lon = lon_left + (px / width) * (lon_right - lon_left)
    lat = lat_top - (py / height) * (lat_top - lat_bottom)
    return lat, lon


def main():
    mask = _make_mask()
    best, risk_grid = assess_nine_grid(mask)
    assert best["row"] == 1 and best["col"] == 1, best
    assert abs(best["risk"] - 0.0) <= 1e-9, best
    assert len(risk_grid) == 3 and all(len(row) == 3 for row in risk_grid), risk_grid

    bounds = (113.90, 22.72, 113.92, 22.74)
    lat, lon = pixel_to_gps(best["cx"], best["cy"], bounds, (300, 300))
    assert 22.72 <= lat <= 22.74, lat
    assert 113.90 <= lon <= 113.92, lon
    assert abs(lat - 22.73) <= 1e-6, lat
    assert abs(lon - 113.91) <= 1e-6, lon

    print("=== Lightweight GIS nine-grid acceptance ===")
    print("  OK risk_grid shape=3x3")
    print("  OK best_cell=(1,1) risk=0.0 center_px=(150,150)")
    print(f"  OK best_center_gps=({lat:.6f}, {lon:.6f})")
    print("=== ALL PASSED ===")


if __name__ == "__main__":
    main()
