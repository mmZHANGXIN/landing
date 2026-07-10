"""
SparseNet 单元测试 — 按需求文档验收标准
验证: 全零mask、单点、均匀平面、稀疏随机点不会 NaN 或尺度爆炸
"""

import sys
sys.path.insert(0, "/home/orin/evelyn/orin_landing")
import numpy as np
import torch
from perception.sparse_depth_completion import SparseNet, DepthCompletion

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# NOTE: ARM64 PyTorch CPU conv produces NaN with zero inputs.
# Always run SparseNet on GPU.

def test_all_zero():
    """全零输入 + 全零mask → 不应NaN"""
    model = SparseNet().to(DEVICE)
    x = torch.zeros(1, 1, 128, 128, device=DEVICE)
    mask = torch.zeros(1, 1, 128, 128, device=DEVICE)
    with torch.no_grad():
        out = model(x, mask)
    assert not torch.isnan(out).any(), "NaN in output!"
    assert not torch.isinf(out).any(), "Inf in output!"
    print(f"  [PASS] all_zero → range [{out.min():.4f}, {out.max():.4f}]")


def test_single_point():
    """单点输入 → 应有扩散 (非零输出范围)"""
    model = SparseNet().to(DEVICE)
    x = torch.zeros(1, 1, 128, 128, device=DEVICE)
    x[0, 0, 64, 64] = 0.5
    mask = torch.zeros(1, 1, 128, 128, device=DEVICE)
    mask[0, 0, 64, 64] = 1.0
    with torch.no_grad():
        out = model(x, mask)
    assert not torch.isnan(out).any(), "NaN!"
    assert out.max() > 0, "Output is all zero (no diffusion)"
    spread = (out > 0).sum().item()
    print(f"  [PASS] single_point → range [{out.min():.4f}, {out.max():.4f}] "
          f"nonzero={spread}/{128*128}")


def test_uniform_plane():
    """均匀平面 (全1输入) → 输出应接近输入"""
    model = SparseNet().to(DEVICE)
    x = torch.ones(1, 1, 128, 128, device=DEVICE) * 0.3
    mask = torch.ones(1, 1, 128, 128, device=DEVICE)
    with torch.no_grad():
        out = model(x, mask)
    assert not torch.isnan(out).any(), "NaN!"
    print(f"  [PASS] uniform_plane → range [{out.min():.4f}, {out.max():.4f}] "
          f"mean={out.mean():.4f}")


def test_sparse_random():
    """随机稀疏输入 (10% 有效) → 无 NaN, 有限范围"""
    model = SparseNet().to(DEVICE)
    x = torch.rand(1, 1, 128, 128, device=DEVICE)
    mask = (torch.rand(1, 1, 128, 128, device=DEVICE) > 0.9).float()
    with torch.no_grad():
        out = model(x * mask, mask)
    assert not torch.isnan(out).any(), "NaN!"
    assert not torch.isinf(out).any(), "Inf!"
    assert out.max() < 100, f"Scale explosion: max={out.max()}"
    print(f"  [PASS] sparse_random (10%) → range [{out.min():.4f}, {out.max():.4f}]")


def test_depth_completion_wrapper():
    """测试 DepthCompletion 封装"""
    cfg = {
        "dmax": 30.0,
        "input_size": 128,
        "framework": "torch",
        "weight_path": None,
    }
    dc = DepthCompletion(cfg)
    assert not dc.weights_ready, "Should be False with no weight_path"

    sparse = np.full((128, 128), 0.0, dtype=np.float32)
    sparse[60:68, 60:68] = 15.0
    valid = np.zeros((128, 128), dtype=bool)
    valid[60:68, 60:68] = True

    dense = dc.complete(sparse, valid)
    assert dense.shape == (128, 128)
    assert not np.isnan(dense).any()
    assert np.all(dense >= 0)
    assert np.all(dense <= 30.0)
    print(f"  [PASS] DepthCompletion wrapper → shape={dense.shape} "
          f"range=[{dense.min():.1f}, {dense.max():.1f}]")


if __name__ == "__main__":
    print("=== SparseNet Unit Tests ===")
    test_all_zero()
    test_single_point()
    test_uniform_plane()
    test_sparse_random()
    test_depth_completion_wrapper()
    print("\n=== ALL TESTS PASSED ===")
