#!/usr/bin/env python3
"""ReplayVisualizer 3D 去畸变点云窗口单元测试 (Matplotlib Agg 非交互后端).

覆盖:
  - update() 能接收 (N,3) 点云、空点云和 None, 且不抛异常
  - 窗口使用的正是 halss_pts (坐标未被改写, 按 z 值着色)
  - show_pointcloud=False / Matplotlib 不可用时优雅降级
  - close() 同时关闭 Matplotlib 图和 OpenCV 窗口
  - --no-pointcloud 与 visualization.show_pointcloud 配置均已声明

无 cv2 的开发环境 (如 macOS) 下使用最小 cv2 stub, 仅覆盖 update() 用到的 API;
Orin 完整环境 (opencv-python) 下自动使用真实 cv2.
"""

import importlib.util
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # 非交互后端, 须在 import pyplot 之前设置
import matplotlib.pyplot as plt
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
        h, w = int(size[1]), int(size[0])
        if np.asarray(img).ndim == 3:
            return np.zeros((h, w, 3), dtype=np.uint8)
        return np.zeros((h, w), dtype=np.uint8)

    def _apply_colormap(img, _cmap):
        return np.repeat(np.asarray(img)[..., None], 3, axis=2)

    stub = types.ModuleType("cv2")
    stub.INTER_NEAREST = 1
    stub.WINDOW_NORMAL = 0
    stub.COLOR_BGR2GRAY = 6
    stub.COLOR_GRAY2BGR = 8
    stub.COLORMAP_INFERNO = 4
    stub.FONT_HERSHEY_SIMPLEX = 0
    stub.namedWindow = lambda *_a, **_k: None
    stub.resizeWindow = lambda *_a, **_k: None
    stub.moveWindow = lambda *_a, **_k: None
    stub.imshow = lambda *_a, **_k: None
    stub.waitKey = lambda *_a, **_k: 0
    stub.destroyAllWindows = lambda *_a, **_k: None
    def _cvt_color(img, code):
        arr = np.asarray(img)
        if code == 8 and arr.ndim == 2:  # COLOR_GRAY2BGR
            return np.repeat(arr[..., None], 3, axis=2)
        if code == 6 and arr.ndim == 3:  # COLOR_BGR2GRAY
            return arr[..., 0]
        return arr

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

# 生产代码优先 TkAgg 交互后端; 测试固定使用 Agg 非交互后端模拟更新
ReplayVisualizer._import_matplotlib = staticmethod(lambda: plt)


# ──────────────────────────────────────────────
# 测试数据构造
# ──────────────────────────────────────────────
def _depth_map(size=128):
    return np.linspace(0.5, 25.0, size * size).reshape(size, size).astype(np.float32)


def _sem_map(size=128):
    sem = np.full((size, size), 128, dtype=np.uint8)
    sem[30:90, 30:90] = 255
    sem[60:70, 60:70] = 0
    return sem


def _sample_cloud(n=5000, seed=7):
    """level-body 风格点云: x/y ∈ ±5m, z down 0.5~25.5m."""
    rng = np.random.default_rng(seed)
    pts = rng.uniform(-5.0, 5.0, size=(n, 3)).astype(np.float32)
    pts[:, 2] = np.abs(pts[:, 2]) * 5.0 + 0.5
    return pts


def _make_vis(show_pointcloud=True):
    return ReplayVisualizer(show_pointcloud=show_pointcloud)


# ──────────────────────────────────────────────
# 测试用例
# ──────────────────────────────────────────────
def test_update_accepts_full_cloud():
    vis = _make_vis()
    cloud = _sample_cloud()
    vis.update(_depth_map(), _sem_map(), None,
               point_cloud=cloud, cloud_stamp=162.5)
    assert vis._pc_scatter is not None
    assert vis._pc_fig.get_label() == "4.Deskewed Point Cloud", "点云窗口应重新编号为 4"
    title = vis._pc_ax.get_title()
    assert f"{len(cloud)} pts" in title
    assert "162.500" in title
    vis.close()


def test_update_accepts_empty_cloud_and_none():
    vis = _make_vis()
    vis.update(_depth_map(), _sem_map(), None,
               point_cloud=_sample_cloud(100), cloud_stamp=1.0)
    assert vis._pc_scatter is not None
    # 空点云 → 清空散点但保留窗口
    vis.update(_depth_map(), _sem_map(), None,
               point_cloud=np.empty((0, 3), dtype=np.float32), cloud_stamp=2.0)
    assert vis._pc_scatter is None
    assert vis._pc_fig is not None
    # None → 同样安全, 标题显示 0 pts
    vis.update(_depth_map(), _sem_map(), None, point_cloud=None, cloud_stamp=None)
    assert vis._pc_scatter is None
    assert "0 pts" in vis._pc_ax.get_title()
    vis.close()


def test_single_point_and_small_clouds_no_crash():
    vis = _make_vis()
    for cloud in (np.array([[0.1, 0.2, 3.0]], dtype=np.float32),
                  np.full((1, 3), np.nan, dtype=np.float32),
                  None):
        vis.update(_depth_map(), _sem_map(), None, point_cloud=cloud, cloud_stamp=3.0)
    vis.close()


def test_window_uses_halss_pts_unchanged():
    vis = _make_vis()
    cloud = _sample_cloud(2000, seed=11)
    cloud_copy = cloud.copy()
    vis.update(_depth_map(), _sem_map(), None, point_cloud=cloud, cloud_stamp=42.0)
    off_x, off_y, off_z = vis._pc_scatter._offsets3d
    assert np.array_equal(off_x, cloud[:, 0]), "x 坐标被改写"
    assert np.array_equal(off_y, cloud[:, 1]), "y 坐标被改写"
    assert np.array_equal(off_z, cloud[:, 2]), "z 坐标被改写"
    assert np.array_equal(cloud, cloud_copy), "输入数组被改写"
    # 按 z 值着色
    assert np.allclose(np.asarray(vis._pc_scatter.get_array()), cloud[:, 2])
    # 坐标轴与 level-body 定义一致
    assert vis._pc_ax.get_xlabel().startswith("x forward")
    assert vis._pc_ax.get_ylabel().startswith("y lateral")
    assert vis._pc_ax.get_zlabel().startswith("z down")
    vis.close()


def test_show_pointcloud_false_skips_matplotlib():
    vis = ReplayVisualizer(show_pointcloud=False)
    assert vis._mpl is None and vis._pc_fig is None
    vis.update(_depth_map(), _sem_map(), None,
               point_cloud=_sample_cloud(10), cloud_stamp=1.0)
    assert vis._pc_scatter is None
    vis.close()


def test_matplotlib_unavailable_degrades_gracefully():
    def _boom(self):  # noqa: ANN001
        raise ImportError("matplotlib broken")

    # 注意: 类属性访问会解开 staticmethod, 恢复时必须重新包装
    ReplayVisualizer._import_matplotlib = _boom
    try:
        vis = ReplayVisualizer(show_pointcloud=True)
        assert vis._mpl is None
        # 3D 窗口关闭, 但 update() 不抛异常
        vis.update(_depth_map(), _sem_map(), None,
                   point_cloud=_sample_cloud(100), cloud_stamp=5.0)
        vis.close()
    finally:
        ReplayVisualizer._import_matplotlib = staticmethod(lambda: plt)


def test_close_closes_figure():
    vis = _make_vis()
    vis.update(_depth_map(), _sem_map(), None,
               point_cloud=_sample_cloud(50), cloud_stamp=9.0)
    fig = vis._pc_fig
    assert fig is not None
    vis.close()
    assert vis._pc_fig is None
    assert not plt.fignum_exists(fig.number)


def test_cli_flag_and_config_key_declared():
    script = (ROOT / "scripts" / "replay_bag_offline.py").read_text(encoding="utf-8")
    assert 'add_argument("--no-pointcloud"' in script
    assert 'vis_cfg.get("show_pointcloud", True)' in script
    # 目录布局因机器而异: Orin 为 experiments/runs/<name>, 开发机为 experiments/<name>
    candidates = [
        ROOT.parent / "experiments" / "runs" / "20260807_162946_orin_landing"
        / "experiment_config_snapshot.yaml",
        ROOT.parent / "experiments" / "20260807_162946_orin_landing"
        / "experiment_config_snapshot.yaml",
    ]
    text = next((p.read_text(encoding="utf-8") for p in candidates if p.exists()), None)
    assert text is not None, "找不到 experiment_config_snapshot.yaml (两种布局均未命中)"
    assert "show_pointcloud: true" in text


def main():
    tests = [fn for name, fn in sorted(globals().items())
             if name.startswith("test_") and callable(fn)]
    for fn in tests:
        fn()
    plt.close("all")
    cv2_backend = "real" if "cv2" in sys.modules and hasattr(sys.modules["cv2"], "imread") \
        else "stub"
    print(f"=== test_replay_pointcloud_vis: {len(tests)} tests PASSED "
          f"(cv2={cv2_backend}, matplotlib Agg) ===")


if __name__ == "__main__":
    main()
