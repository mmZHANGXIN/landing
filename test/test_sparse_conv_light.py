#!/usr/bin/env python3
"""Lightweight sparsity-invariant convolution acceptance without numpy/torch."""

DMAX = 30.0
EPS = 1e-9


def encode_inverse(depth_m):
    depth_m = max(0.0, min(DMAX, float(depth_m)))
    return max(0.0, min(1.0, 1.0 - depth_m / DMAX))


def decode_inverse(encoded):
    encoded = max(0.0, min(1.0, float(encoded)))
    return (1.0 - encoded) * DMAX


def sparse_conv3x3(x, mask, weights, bias=0.0):
    h = len(x)
    w = len(x[0])
    out = [[0.0 for _ in range(w)] for _ in range(h)]
    for y in range(h):
        for x_idx in range(w):
            numerator = 0.0
            count = 0.0
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    yy = y + dy
                    xx = x_idx + dx
                    if 0 <= yy < h and 0 <= xx < w and mask[yy][xx]:
                        numerator += x[yy][xx] * weights[dy + 1][dx + 1]
                        count += 1.0
            out[y][x_idx] = numerator / (count + EPS) + bias if count > 0.0 else bias
    return out


def max_pool_mask3x3(mask):
    h = len(mask)
    w = len(mask[0])
    out = [[0 for _ in range(w)] for _ in range(h)]
    for y in range(h):
        for x in range(w):
            val = 0
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    yy = y + dy
                    xx = x + dx
                    if 0 <= yy < h and 0 <= xx < w and mask[yy][xx]:
                        val = 1
            out[y][x] = val
    return out


def test_inverse_depth_encoding():
    assert encode_inverse(DMAX) == 0.0
    assert encode_inverse(0.0) == 1.0
    assert abs(encode_inverse(15.0) - 0.5) <= 1e-9
    assert abs(decode_inverse(0.5) - 15.0) <= 1e-9


def test_sparse_conv_normalizes_by_valid_count():
    x = [[0.0 for _ in range(5)] for _ in range(5)]
    mask = [[0 for _ in range(5)] for _ in range(5)]
    x[2][2] = 0.6
    x[2][3] = 0.3
    mask[2][2] = 1
    mask[2][3] = 1
    weights = [[1.0 for _ in range(3)] for _ in range(3)]
    out = sparse_conv3x3(x, mask, weights, bias=0.0)
    assert abs(out[2][2] - 0.45) <= 1e-6, out[2][2]
    assert abs(out[2][3] - 0.45) <= 1e-6, out[2][3]


def test_all_zero_mask_is_finite_and_bias_only():
    x = [[1.0 for _ in range(5)] for _ in range(5)]
    mask = [[0 for _ in range(5)] for _ in range(5)]
    weights = [[1.0 for _ in range(3)] for _ in range(3)]
    out = sparse_conv3x3(x, mask, weights, bias=0.1)
    assert all(abs(value - 0.1) <= 1e-9 for row in out for value in row)


def test_mask_max_pool_expands_valid_region():
    mask = [[0 for _ in range(5)] for _ in range(5)]
    mask[2][2] = 1
    pooled = max_pool_mask3x3(mask)
    assert sum(sum(row) for row in pooled) == 9, pooled
    assert pooled[1][1] == 1 and pooled[3][3] == 1
    assert pooled[0][0] == 0 and pooled[4][4] == 0


def test_output_scale_preserves_inverse_depth_semantics():
    unscaled_encoded = 0.02
    scaled_encoded = min(1.0, unscaled_encoded * 20.0)
    unscaled_depth = decode_inverse(unscaled_encoded)
    scaled_depth = decode_inverse(scaled_encoded)
    assert unscaled_depth > scaled_depth, (unscaled_depth, scaled_depth)
    assert abs(unscaled_depth - 29.4) <= 1e-6
    assert abs(scaled_depth - 18.0) <= 1e-6


def main():
    test_inverse_depth_encoding()
    test_sparse_conv_normalizes_by_valid_count()
    test_all_zero_mask_is_finite_and_bias_only()
    test_mask_max_pool_expands_valid_region()
    test_output_scale_preserves_inverse_depth_semantics()
    print("=== Lightweight sparse convolution acceptance ===")
    print("  OK inverse depth encoding/decoding")
    print("  OK sparse conv divides by valid count")
    print("  OK all-zero mask remains finite")
    print("  OK mask max-pool expands valid region")
    print("  OK output_scale preserves inverse-depth direction")
    print("=== ALL PASSED ===")


if __name__ == "__main__":
    main()
