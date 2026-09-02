#!/usr/bin/env python3
"""ReplayVisualizer 深度图显示单元测试 (主图固定 0~30m + 局部对比度自动量程).

覆盖 (对应规格 Test Plan):
  - 主窗口: 0/15/30 m 稳定映射为黑/中灰/白, 30 m 填充恒为纯白
  - 不同帧深度分布不同: 主窗口相同距离灰度恒定, 局部窗口量程跟随变化
  - 局部窗口近黑远白且纯灰度 (无彩色轮廓)
  - 自动量程排除 unknown / 30 m 填充值
  - 空有效区域 / 有效点过少时回退固定 0~30 m 量程, 不抛异常
  - 显示函数不修改输入深度数组
  - ReplayVisualizer.update() 端到端: 主/局部窗口均可见且为灰阶色条
  - legacy_inferno 模式保留旧版 inferno 色条 (主窗口), 局部窗口仍为灰阶
  - --no-local-depth 关闭局部窗口, 主窗口不受影响
  - --depth-display-mode / --no-local-depth 与可视化参数键均已声明

无 cv2 的开发环境 (如 macOS) 下使用最小 cv2 stub (no-op + 保内容缩放/色表);
Orin 完整环境下自动使用真实 cv2.
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


# ──────────────────────────────────────────────
# 无 cv2 时的最小 stub (Orin 上真实 cv2 存在, 自动跳过)
# ──────────────────────────────────────────────
def _install_cv2_stub():
    if "cv2" in sys.modules:
        return
    try:
        import cv2  # noqa: F401 — 真实 OpenCV 存在则使用, 绝不装 stub
        return
    except ImportError:
        pass
    import types

    def _resize(img, size, *_a, **_k):
        # 最近邻索引映射缩放 (stub 不追求精确插值, 只保留像素内容)
        arr = np.asarray(img)
        h, w = int(size[1]), int(size[0])
        ih, iw = arr.shape[:2]
        rows = np.minimum((np.arange(h) * ih) // max(h, 1), ih - 1)
        cols = np.minimum((np.arange(w) * iw) // max(w, 1), iw - 1)
        return np.ascontiguousarray(arr[np.ix_(rows, cols)])

    def _apply_colormap(img, _cmap):
        # inferno 近似: 黑→橙黄, 保证与灰阶可区分
        g = np.asarray(img, dtype=np.float32)
        return np.clip(np.stack([g * 0.9 + 20, g * 0.5, g * 0.2], axis=2),
                       0, 255).astype(np.uint8)

    stub = types.ModuleType("cv2")
    stub.INTER_NEAREST = 1
    stub.WINDOW_NORMAL = 0
    stub.COLOR_BGR2GRAY = 6
    stub.COLOR_GRAY2BGR = 8
    stub.COLORMAP_INFERNO = 4
    stub.FONT_HERSHEY_SIMPLEX = 0
    def _cvt_color(img, code):
        arr = np.asarray(img)
        if code == 8 and arr.ndim == 2:  # COLOR_GRAY2BGR
            return np.repeat(arr[..., None], 3, axis=2)
        if code == 6 and arr.ndim == 3:  # COLOR_BGR2GRAY
            return arr[..., 0]
        return arr

    stub.namedWindow = lambda *_a, **_k: None
    stub.resizeWindow = lambda *_a, **_k: None
    stub.moveWindow = lambda *_a, **_k: None
    stub.imshow = lambda *_a, **_k: None
    stub.waitKey = lambda *_a, **_k: 0
    stub.destroyAllWindows = lambda *_a, **_k: None
    stub.cvtColor = _cvt_color
    stub.putText = lambda *_a, **_k: None
    stub.resize = _resize
    stub.applyColorMap = _apply_colormap
    sys.modules["cv2"] = stub


_install_cv2_stub()

_spec = importlib.util.spec_from_file_location(
    "replay_bag_offline", ROOT / "scripts" / "replay_bag_offline.py")
_replay = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_replay)
ReplayVisualizer = _replay.ReplayVisualizer
render_depth_fixed_gray = _replay.render_depth_fixed_gray
render_depth_local_contrast = _replay.render_depth_local_contrast

# 捕获 imshow 的窗口图像 (stub no-op 无法验证内容)
_captured = {}


def _record_imshow(name, img):
    _captured[name] = img


sys.modules["cv2"].imshow = _record_imshow


# ──────────────────────────────────────────────
# 测试数据构造
# ──────────────────────────────────────────────
def _flat_depth(size=128, value=2.0):
    return np.full((size, size), value, dtype=np.float32)


def _bump_depth(size=128, base=2.0, bump=1.0, r0=60, r1=68):
    """平坦地面 + 一个凸起障碍 (更近, 深度突变 ~1m)."""
    d = _flat_depth(size, base)
    d[r0:r1, r0:r1] = bump
    return d


def _ones_mask(size=128):
    return np.ones((size, size), dtype=bool)


def _sem_map(size=128):
    sem = np.full((size, size), 128, dtype=np.uint8)
    sem[30:90, 30:90] = 255
    sem[60:70, 60:70] = 0
    return sem


# ──────────────────────────────────────────────
# 测试用例
# ──────────────────────────────────────────────
def test_main_window_fixed_mapping_0_15_30():
    """主窗口固定量程: 0 m→黑, 15 m→中灰, 30 m→纯白."""
    depth = _flat_depth(value=30.0)
    depth[32, 32] = 0.0
    depth[32, 64] = 15.0
    bgr = render_depth_fixed_gray(depth)
    gray = bgr[..., 0]
    assert gray[32, 32] == 0, f"0m 应映射为纯黑: {gray[32, 32]}"
    assert abs(int(gray[32, 64]) - 127.5) <= 1.0, f"15m 应为中灰: {gray[32, 64]}"
    assert gray[32, 96] == 255, "30m 填充应恒为纯白"


def test_main_window_independent_of_local_auto_range():
    """不同帧深度分布不同: 主窗口相同距离灰度恒定, 局部窗口量程跟随变化."""
    frame_a = _flat_depth(value=2.0)
    frame_b = frame_a.copy()
    frame_b[:, 64:] = 8.0  # 另一帧右半分布不同
    ga = render_depth_fixed_gray(frame_a)[..., 0]
    gb = render_depth_fixed_gray(frame_b)[..., 0]
    assert ga[32, 32] == gb[32, 32], "相同距离 (2.0m) 主窗口灰度应恒定"
    assert ga[32, 32] == ga[32, 96], "同一帧相同距离灰度相同"
    assert gb[32, 96] > gb[32, 32], "8m 应比 2m 亮"
    _, ia = render_depth_local_contrast(frame_a, _ones_mask())
    _, ib = render_depth_local_contrast(frame_b, _ones_mask())
    assert abs(ia["far_m"] - 2.25) < 1e-3, f"局部量程 A far={ia['far_m']}"
    assert abs(ib["far_m"] - 8.3) < 1e-3, f"局部量程 B far={ib['far_m']}"


def test_local_window_near_darker_than_far():
    """局部窗口近黑远白, 且为纯灰度 (无彩色轮廓)."""
    depth = _flat_depth(value=5.0)
    depth[:, :64] = 0.5
    bgr, info = render_depth_local_contrast(depth, _ones_mask())
    assert info["auto_range"] is True
    gray = bgr[..., 0]
    assert gray[:, :64].mean() < gray[:, 64:].mean(), "近处应比远处更黑"
    assert gray[32, 32] < gray[32, 96]
    assert gray[32, 96] > 200, "远端应接近白"
    assert (bgr[..., 0] == bgr[..., 1]).all() and (bgr[..., 1] == bgr[..., 2]).all(), \
        "局部窗口应纯灰度"


def test_local_window_excludes_dmax_fill_and_unknown():
    """自动量程排除 unknown / 30 m 填充值."""
    # 情况 1: 全有效掩码, 一半是 30m 填充 → 量程应落在 2m 附近
    depth = _flat_depth(value=2.0)
    depth[:, 96:] = 30.0
    _, info = render_depth_local_contrast(depth, _ones_mask())
    assert info["auto_range"] is True
    assert info["far_m"] < 10.0, f"30m 填充值被计入量程: far_m={info['far_m']}"
    # 情况 2: unknown 区域用掩码排除
    depth2 = _flat_depth(value=3.0)
    depth2[100:, :] = 30.0
    mask2 = np.zeros((128, 128), dtype=bool)
    mask2[:100, :100] = True
    _, info2 = render_depth_local_contrast(depth2, mask2)
    assert info2["auto_range"] is True
    assert info2["far_m"] < 10.0, f"unknown 被计入量程: far_m={info2['far_m']}"


def test_fallback_when_few_valid():
    """有效点过少 / 空有效区域回退固定 0~30 m 量程."""
    depth = _flat_depth(value=5.0)
    mask = np.zeros((128, 128), dtype=bool)
    mask[0, :10] = True  # 10 个有效点 < min_valid(32)
    bgr, info = render_depth_local_contrast(depth, mask)
    assert info["auto_range"] is False
    assert info["near_m"] == 0.0 and info["far_m"] == 30.0
    assert bgr.dtype == np.uint8 and bgr.shape == (128, 128, 3)
    # 空有效区域同样回退且不崩溃
    _, info2 = render_depth_local_contrast(depth, np.zeros((128, 128), bool))
    assert info2["auto_range"] is False


def test_render_does_not_modify_inputs():
    """显示函数不修改输入深度数组和掩码."""
    depth = _bump_depth()
    mask = _ones_mask()
    d0, m0 = depth.copy(), mask.copy()
    render_depth_fixed_gray(depth)
    render_depth_local_contrast(depth, mask)
    assert np.array_equal(depth, d0), "输入深度数组被修改"
    assert np.array_equal(mask, m0), "输入掩码被修改"
    # NaN 输入同样安全 (被替换为 vmax 显示值, 原数组不变)
    depth[64, 64] = np.nan
    d1 = depth.copy()
    render_depth_fixed_gray(depth)
    render_depth_local_contrast(depth, mask)
    np.testing.assert_allclose(depth, d1, equal_nan=True)


def test_visualizer_main_and_local_windows_end_to_end():
    """update() 端到端: 主窗口 (0-30m) 与局部窗口均可见且为灰阶色条."""
    vis = ReplayVisualizer(show_pointcloud=False)  # 默认 fixed_gray + 局部窗口
    _captured.clear()
    vis.update(_bump_depth(), _sem_map(), None,
               semantic_valid_mask=_ones_mask())
    main_img = _captured.get("2.Depth Map (0-30m)")
    assert main_img is not None, "未捕获主深度窗口图像"
    assert main_img.shape == (vis._disp_h, vis.display_width + 50, 3)
    bar = main_img[:, vis.display_width + 5:vis.display_width + 45]
    assert (bar[..., 0] == bar[..., 1]).all() and (bar[..., 1] == bar[..., 2]).all()
    assert bar[0, 0, 0] > bar[-1, 0, 0], "主窗口色条应上白 (30m) 下黑 (0m)"
    local_img = _captured.get("3.Local Depth Contrast")
    assert local_img is not None, "未捕获局部对比度窗口图像"
    assert local_img.shape == main_img.shape
    lbar = local_img[:, vis.display_width + 5:vis.display_width + 45]
    assert (lbar[..., 0] == lbar[..., 1]).all() and (lbar[..., 1] == lbar[..., 2]).all()
    assert lbar[0, 0, 0] > lbar[-1, 0, 0], "局部窗口色条应上白 (far) 下黑 (near)"
    # 窗口图像本身纯灰度 (无彩色轮廓)
    assert (local_img[..., 0] == local_img[..., 1]).all()
    assert (local_img[..., 1] == local_img[..., 2]).all()
    assert (main_img[..., 0] == main_img[..., 1]).all()
    vis.close()


def test_visualizer_legacy_inferno_mode():
    """legacy_inferno 主窗口保留旧版 inferno 色条, 局部窗口仍为灰阶."""
    vis = ReplayVisualizer(show_pointcloud=False, depth_display_mode="legacy_inferno")
    _captured.clear()
    vis.update(_bump_depth(), _sem_map(), None)
    main_img = _captured.get("2.Depth Map (0-30m)")
    assert main_img is not None
    bar = main_img[:, vis.display_width + 5:vis.display_width + 45]
    assert (bar[..., 0] != bar[..., 1]).any(), "legacy 色条应为 inferno 彩色"
    local_img = _captured.get("3.Local Depth Contrast")
    assert local_img is not None, "局部窗口应始终显示"
    lbar = local_img[:, vis.display_width + 5:vis.display_width + 45]
    assert (lbar[..., 0] == lbar[..., 1]).all(), "局部窗口应为灰阶"
    vis.close()


def test_no_local_depth_disables_local_window():
    """--no-local-depth (show_local_depth=False) 关闭局部窗口, 主窗口不受影响."""
    vis = ReplayVisualizer(show_pointcloud=False, show_local_depth=False)
    _captured.clear()
    vis.update(_bump_depth(), _sem_map(), None)
    assert _captured.get("2.Depth Map (0-30m)") is not None
    assert _captured.get("3.Local Depth Contrast") is None, "局部窗口应被关闭"
    vis.close()


def test_cli_flag_and_config_keys_declared():
    """--depth-display-mode / --no-local-depth 与可视化参数键均已声明."""
    script = (ROOT / "scripts" / "replay_bag_offline.py").read_text(encoding="utf-8")
    assert 'add_argument("--depth-display-mode"' in script
    assert 'add_argument("--no-local-depth"' in script
    assert "fixed_gray" in script and "legacy_inferno" in script
    assert 'vis_cfg.get("depth_display_pct_low", 2.0)' in script
    assert 'vis_cfg.get("depth_display_min_span_m", 0.5)' in script
    assert 'vis_cfg.get("show_local_depth", True)' in script
    assert "semantic_valid_mask=semantic_valid_mask" in script
    assert '"2.Depth Map (0-30m)"' in script
    assert '"3.Local Depth Contrast"' in script


def main():
    tests = [fn for name, fn in sorted(globals().items())
             if name.startswith("test_") and callable(fn)]
    for fn in tests:
        fn()
    cv2_backend = "real" if "cv2" in sys.modules and hasattr(sys.modules["cv2"], "imread") \
        else "stub"
    print(f"=== test_replay_depth_display: {len(tests)} tests PASSED "
          f"(cv2={cv2_backend}) ===")


if __name__ == "__main__":
    main()
