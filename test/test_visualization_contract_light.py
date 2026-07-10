#!/usr/bin/env python3
"""Lightweight source-level checks for visualization contract."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_no_control_window_title_and_nearest():
    text = (ROOT / "test_live_nocontrol.py").read_text(encoding="utf-8")
    assert 'set_window_title("PointCloud Canvas + Semantic + Depth (NN-fill + ONNX DRL)")' in text
    assert 'self.ax_sem.set_title("Semantic (white=safe, black=danger)")' in text
    assert 'self.ax_depth.set_title("Rendered Depth (NN Fill)")' in text
    assert 'interpolation="nearest"' in text


def test_cv2_visualizer_uses_halss_passthrough_and_nearest_resize():
    text = (ROOT / "visualization" / "display.py").read_text(encoding="utf-8")
    assert 'cfg.get("binary_semantic_window_title", "binary semantic")' in text
    assert "binary_semantic_vis" in text
    assert "cv2.INTER_NEAREST" in text


def test_config_declares_binary_semantic_window_title():
    text = (ROOT / "config" / "experiment_config.yaml").read_text(encoding="utf-8")
    assert 'binary_semantic_window_title: "binary semantic"' in text


def main():
    test_no_control_window_title_and_nearest()
    test_cv2_visualizer_uses_halss_passthrough_and_nearest_resize()
    test_config_declares_binary_semantic_window_title()
    print("=== Lightweight visualization contract acceptance ===")
    print("  OK no-control window title matches ONNX FAST-LIO view")
    print("  OK config declares binary semantic window title")
    print("  OK no-control semantic/depth imshow use nearest interpolation")
    print("  OK CV2 visualizer uses HALSS passthrough with nearest resize")
    print("=== ALL PASSED ===")


if __name__ == "__main__":
    main()
