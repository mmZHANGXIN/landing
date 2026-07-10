#!/usr/bin/env python3
"""Compare HALSS reference visualization with current binary semantic output."""

import argparse
import sys

import cv2
import numpy as np


def _load(path, grayscale):
    flag = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
    img = cv2.imread(path, flag)
    if img is None:
        raise FileNotFoundError(path)
    return img


def _resize_like(img, ref):
    if img.shape[:2] == ref.shape[:2]:
        return img
    return cv2.resize(img, (ref.shape[1], ref.shape[0]), interpolation=cv2.INTER_NEAREST)


def _as_compare_space(img, grayscale):
    if grayscale and img.ndim == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def main():
    parser = argparse.ArgumentParser(description="Pixel-level HALSS visualization comparison")
    parser.add_argument("--reference", required=True, help="Reference HALSS image from notebook/original HALSS")
    parser.add_argument("--candidate", required=True, help="Current saved binary_semantic frame")
    parser.add_argument("--grayscale", action="store_true", help="Compare as grayscale")
    parser.add_argument("--max-mean-abs-diff", type=float, default=0.0)
    parser.add_argument("--max-pixel-diff", type=int, default=0)
    args = parser.parse_args()

    ref = _load(args.reference, args.grayscale)
    cand = _load(args.candidate, args.grayscale)
    cand = _resize_like(cand, ref)
    ref = _as_compare_space(ref, args.grayscale)
    cand = _as_compare_space(cand, args.grayscale)

    diff = np.abs(ref.astype(np.int16) - cand.astype(np.int16))
    mean_abs = float(diff.mean())
    max_abs = int(diff.max())
    exact = bool(np.array_equal(ref, cand))
    mismatch_ratio = float(np.mean(diff > 0))

    print(f"reference={args.reference}")
    print(f"candidate={args.candidate}")
    print(f"shape={ref.shape}")
    print(f"exact_match={exact}")
    print(f"mean_abs_diff={mean_abs:.6f}")
    print(f"max_abs_diff={max_abs}")
    print(f"mismatch_ratio={mismatch_ratio:.6f}")

    passed = mean_abs <= args.max_mean_abs_diff and max_abs <= args.max_pixel_diff
    if passed:
        print("[OK] HALSS visualization comparison passed")
        return 0
    print("[FAIL] HALSS visualization comparison failed")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
