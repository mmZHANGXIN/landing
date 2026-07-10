#!/usr/bin/env python3
"""CUDA DepthProjector parity test against the NumPy perspective path.

Run this on the Jetson Orin environment. It intentionally fails if PyTorch CUDA
is unavailable because the flight experiment requires GPU depth projection.
"""

import sys

import numpy as np
import torch

from perception.depth_projection import DepthProjector


def _projector(backend):
    return DepthProjector(
        img_width=128,
        img_height=128,
        max_range=30.0,
        mode="perspective",
        backend=backend,
        fx=64.0,
        fy=64.0,
        cx=63.5,
        cy=63.5,
    )


def _assert_same(points, pose, label):
    ref = _projector("numpy").project(points, pose)
    got = _projector("torch_cuda").project(points, pose)
    if ref.shape != got.shape:
        raise AssertionError(f"{label}: shape mismatch ref={ref.shape} got={got.shape}")
    diff = np.max(np.abs(ref - got))
    if not np.allclose(ref, got, atol=1e-5):
        raise AssertionError(f"{label}: max depth diff={diff}")
    print(f"  OK {label}: valid={int(np.sum(got < 30.0))} max_diff={diff:.2e}")


def main():
    if not torch.cuda.is_available():
        print("CUDA is not available; torch_cuda depth projection cannot be accepted.")
        sys.exit(2)

    pose0 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    points0 = np.array([
        [0.0, 0.0, 8.0],
        [0.0, 0.0, 5.0],
        [1.0, 0.0, 5.0],
        [0.0, 1.0, 5.0],
        [0.0, 0.0, -1.0],
        [100.0, 0.0, 5.0],
        [0.0, 0.0, 30.0],
    ], dtype=np.float32)
    _assert_same(points0, pose0, "center/body-axis/z-buffer/invalid")

    pose1 = np.array([10.0, 20.0, 3.0, 0.0, 0.0, np.pi / 2.0], dtype=np.float32)
    points1 = np.array([
        [10.0, 20.0, 8.0],
        [10.0, 21.0, 8.0],
        [11.0, 20.0, 8.0],
    ], dtype=np.float32)
    _assert_same(points1, pose1, "translated yawed pose")

    point_single = np.array([[0.0, 0.0, 5.0]], dtype=np.float32)
    _assert_same(point_single, pose0, "single-point frame")

    print("=== CUDA depth projection parity PASSED ===")


if __name__ == "__main__":
    main()
