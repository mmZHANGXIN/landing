"""Detect repeated DRL actions and save the frame that caused the warning."""

import logging
import math
from pathlib import Path


class ActionCollapseMonitor:
    """Warn when the policy repeatedly selects one action on live data."""

    def __init__(self, cfg: dict = None, logger: logging.Logger = None):
        cfg = cfg or {}
        self.enabled = bool(cfg.get("action_collapse_monitor", True))
        self.window = int(cfg.get("action_collapse_window", 20))
        self.min_confidence = float(cfg.get("action_collapse_min_confidence", 0.0))
        self.save_snapshot = bool(cfg.get("action_collapse_save_snapshot", True))
        self.save_dir = Path(cfg.get("save_dir", "experiments/frames"))
        self.logger = logger or logging.getLogger("ActionCollapseMonitor")

        self._last_action = None
        self._run_len = 0
        self._warned_for_action = None
        self._snapshot_count = 0

    def observe(self, seq: int, action_id: int, action_name: str, rl_info: dict,
                frame_arrays: dict = None, action_names=None) -> bool:
        """Update monitor state.

        Returns True when a new collapse warning was emitted.
        """
        if not self.enabled or self.window <= 1:
            return False

        if self._last_action == action_id:
            self._run_len += 1
        else:
            self._last_action = action_id
            self._run_len = 1
            self._warned_for_action = None

        if self._run_len < self.window or self._warned_for_action == action_id:
            return False

        confidence = rl_info.get("confidence")
        if confidence is not None and confidence < self.min_confidence:
            return False

        self._warned_for_action = action_id
        snapshot = None
        if self.save_snapshot and frame_arrays:
            snapshot = self._save_snapshot(seq, action_id, frame_arrays)

        self.logger.warning(
            "DRL action collapse: action=%d(%s) repeated %d frames | "
            "depth_norm=%.3f/%.3f/%.3f sem_norm=%.3f/%.3f/%.3f conf=%s top=%s%s",
            action_id, action_name, self._run_len,
            float(rl_info.get("depth_norm_min", math.nan)),
            float(rl_info.get("depth_norm_mean", math.nan)),
            float(rl_info.get("depth_norm_max", math.nan)),
            float(rl_info.get("sem_norm_min", math.nan)),
            float(rl_info.get("sem_norm_mean", math.nan)),
            float(rl_info.get("sem_norm_max", math.nan)),
            "n/a" if confidence is None else f"{float(confidence):.3f}",
            _top_probs(rl_info.get("action_probs"), action_names),
            "" if snapshot is None else f" | saved={snapshot}",
        )
        return True

    def _save_snapshot(self, seq: int, action_id: int, frame_arrays: dict) -> Path:
        import numpy as np

        self.save_dir.mkdir(parents=True, exist_ok=True)
        path = self.save_dir / f"{seq:06d}_action_collapse_a{action_id}_{self._snapshot_count}.npz"
        payload = {}
        for key, value in frame_arrays.items():
            if value is None:
                continue
            payload[key] = np.asarray(value)
        np.savez_compressed(path, **payload)
        self._snapshot_count += 1
        return path


def _top_probs(probs, action_names=None, k: int = 3) -> str:
    if probs is None:
        return "p=n/a"
    pairs = sorted(enumerate(probs), key=lambda item: item[1], reverse=True)[:k]
    if action_names is None:
        return "p=" + ",".join(f"{idx}:{float(prob):.3f}" for idx, prob in pairs)
    return "p=" + ",".join(
        f"{idx}:{action_names[idx]}:{float(prob):.3f}" for idx, prob in pairs
    )
