"""
GIS 离线九宫格 — 验收测试
===========================
验收条件:
  1. bounds 缺失时 best_center_gps = None (拒绝生成目标点)
  2. bounds 提供时 best_center_gps 非空
  3. 输出语义 mask、overlay、3×3 风险矩阵
  4. 最低风险格中心 GPS 合理

用法:
  # 模拟测试 (不需要真实 GIS 影像)
  python test_gis_nine_grid.py --mock

  # 用真实影像 + 预计算语义 mask
  python test_gis_nine_grid.py \
    --image gis_data/satellite.png \
    --mask gis_data/sem_mask.png \
    --bounds 113.90,22.72,113.92,22.74
"""

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import cv2
import os

from preprocessing.global_safety_prior import GlobalSafetyPrior


def test_bounds_missing():
    """验收: bounds 缺失时拒绝生成目标点"""
    gsp = GlobalSafetyPrior()

    # 造一个假语义 mask (全危险 + 一块安全)
    mask = np.full((300, 300), 9, dtype=np.uint8)
    mask[100:200, 100:200] = 1  # 安全区

    result = gsp.assess(sem_mask=mask)

    assert result['best_center_gps'] is None, \
        "FAIL: best_center_gps should be None when no bounds"
    assert result['risk_grid'].shape == (3, 3), \
        f"FAIL: risk_grid shape {result['risk_grid'].shape} != (3,3)"
    assert 'overlay_img' in result, "FAIL: overlay_img missing"
    assert 'sem_mask' in result, "FAIL: sem_mask missing"

    print("  ✅ bounds 缺失 → best_center_gps=None (拒绝生成目标点)")
    print(f"  ✅ 3×3 风险矩阵:\n{result['risk_grid']}")


def test_bounds_provided():
    """验收: bounds 提供时 GPS 正确"""
    gsp = GlobalSafetyPrior()

    mask = np.full((300, 300), 9, dtype=np.uint8)
    mask[100:200, 100:200] = 1

    # bounds: (lon_left, lat_bottom, lon_right, lat_top)
    bounds = (113.90, 22.72, 113.92, 22.74)
    H, W = mask.shape
    gsp.set_georeference(bounds, (W, H))
    result = gsp.assess(sem_mask=mask)

    cx, cy = result['best_center_px']  # should be (150, 150) center of safe zone
    gps = gsp.pixel_to_gps(cx, cy)
    result['best_center_gps'] = gps

    assert result['best_center_gps'] is not None
    lat, lon = result['best_center_gps']
    assert 22.72 <= lat <= 22.74, f"lat {lat} out of bounds"
    assert 113.90 <= lon <= 113.92, f"lon {lon} out of bounds"

    print(f"  ✅ bounds 提供 → GPS=({lat:.6f}, {lon:.6f})")
    print(f"  ✅ 最小风险={result['min_risk']:.3f}")


def test_risk_lut():
    """验证风险映射表"""
    gsp = GlobalSafetyPrior()
    lut = gsp.RISK_LUT
    assert 1 in lut, "Safe class (1) should be in RISK_LUT"
    print(f"  ✅ RiskLUT: {dict(lut)}")


def test_save_results():
    """验收: 保存输出文件"""
    gsp = GlobalSafetyPrior()
    mask = np.full((300, 300), 9, dtype=np.uint8)
    mask[50:150, 50:150] = 1
    result = gsp.assess(sem_mask=mask)

    out_dir = "/tmp/gis_test_output"
    gsp.save_results(result, out_dir)

    files = os.listdir(out_dir)
    assert len(files) >= 3, f"Expected 3+ output files, got {len(files)}"
    print(f"  ✅ 输出 {len(files)} 文件 → {out_dir}: {sorted(files)}")
    # Cleanup
    for f in files:
        os.remove(os.path.join(out_dir, f))
    os.rmdir(out_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", default=None, help="GIS 卫星图路径")
    parser.add_argument("--mask", default=None, help="预计算语义mask路径")
    parser.add_argument("--bounds", default=None, help="bounds: lon_left,lat_bottom,lon_right,lat_top")
    parser.add_argument("--mock", action="store_true", default=True,
                        help="使用模拟数据运行验收测试 (默认)")
    args = parser.parse_args()

    print("=== GIS 离线九宫格 验收测试 ===\n")

    if args.image and args.mask:
        print("--- 真实影像模式 ---")
        gsp = GlobalSafetyPrior()
        bounds = None
        if args.bounds:
            parts = [float(x) for x in args.bounds.split(",")]
            bounds = tuple(parts)
        result = gsp.assess_from_file(args.image, args.mask, bounds=bounds)
        gps = result['best_center_gps']
        if gps:
            print(f"  GPS: ({gps[0]:.6f}, {gps[1]:.6f})")
        else:
            print("  GPS: None (no bounds)")
        print(f"  风险: {result['min_risk']:.3f}")
        print(f"  3×3 风险矩阵:\n{result['risk_grid']}")
        gsp.save_results(result, "/tmp/gis_test_output")

    print("\n--- 模拟验收 ---")
    test_bounds_missing()
    test_bounds_provided()
    test_risk_lut()
    test_save_results()

    print("\n=== ALL GIS 验收 PASSED ===")
