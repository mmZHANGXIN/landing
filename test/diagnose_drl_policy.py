#!/usr/bin/env python3
"""DRL policy observation sweep.

Use this before flight when the live pipeline keeps returning one action. It
feeds controlled depth/semantic observations into the loaded SB3 policy and
prints action probabilities plus encoded observation statistics.
"""

import argparse
import glob
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_ACTION_NAMES = ["HOVER", "N", "NW", "W", "SW", "S", "SE", "E", "NE", "DESCEND"]


def _load_config(path):
    import yaml
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _resolve(path_value):
    path = Path(path_value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _top_probs(probs, action_names, k=3):
    if probs is None:
        return "p=n/a"
    pairs = sorted(enumerate(probs), key=lambda item: item[1], reverse=True)[:k]
    return "p=" + ",".join(f"{idx}:{action_names[idx]}:{prob:.3f}" for idx, prob in pairs)


def _case_maps(name, h, w, dmax, rng):
    import numpy as np

    safe = np.ones((h, w), dtype=np.float32)
    danger = np.full((h, w), 9.0, dtype=np.float32)
    checker = np.where((np.indices((h, w)).sum(axis=0) % 2) == 0, 1.0, 9.0).astype(np.float32)

    if name == "near_safe":
        return np.full((h, w), 2.0, np.float32), safe
    if name == "mid_safe":
        return np.full((h, w), dmax * 0.5, np.float32), safe
    if name == "far_safe":
        return np.full((h, w), dmax, np.float32), safe
    if name == "mid_danger":
        return np.full((h, w), dmax * 0.5, np.float32), danger
    if name == "gradient_safe":
        depth = np.tile(np.linspace(0.0, dmax, w, dtype=np.float32), (h, 1))
        return depth, safe
    if name == "gradient_danger":
        depth = np.tile(np.linspace(0.0, dmax, w, dtype=np.float32), (h, 1))
        return depth, danger
    if name == "checker_mid":
        return np.full((h, w), dmax * 0.5, np.float32), checker
    if name == "random":
        depth = rng.uniform(0.0, dmax, (h, w)).astype(np.float32)
        sem = rng.choice([1.0, 9.0], size=(h, w), p=[0.6, 0.4]).astype(np.float32)
        return depth, sem
    raise ValueError(f"Unknown case: {name}")


def _load_frame(path):
    import numpy as np

    data = np.load(path)
    if "dense_depth" in data:
        depth = data["dense_depth"].astype(np.float32)
    elif "sparse_depth" in data:
        dmax = float(np.nanmax(data["sparse_depth"]))
        depth = np.nan_to_num(data["sparse_depth"], nan=dmax).astype(np.float32)
    else:
        raise KeyError(f"{path} has no dense_depth or sparse_depth")
    if "sem_map" not in data:
        raise KeyError(f"{path} has no sem_map")
    sem = data["sem_map"].astype(np.float32)
    return depth, sem


def _expand_frame_paths(args):
    paths = []
    if args.frame:
        paths.extend(args.frame)
    if args.frame_glob:
        for pattern in args.frame_glob:
            matches = sorted(glob.glob(pattern))
            if not matches:
                raise FileNotFoundError(f"--frame-glob matched no files: {pattern}")
            paths.extend(matches)
    seen = set()
    unique = []
    for item in paths:
        key = str(Path(item))
        if key in seen:
            continue
        seen.add(key)
        unique.append(item)
    return unique


def _run_sweep(agent, items, action_names):
    actions = []
    rows = []
    print("case,action,confidence,top3,depth_norm(min/mean/max),sem_norm(min/mean/max)")
    for name, depth, sem in items:
        action, info = agent.predict_with_info(depth, sem)
        actions.append(action)
        conf = info.get("confidence")
        conf_s = "n/a" if conf is None else f"{conf:.3f}"
        print(
            f"{name},{action}({action_names[action]}),{conf_s},"
            f"{_top_probs(info.get('action_probs'), action_names)},"
            f"{info['depth_norm_min']:.3f}/{info['depth_norm_mean']:.3f}/{info['depth_norm_max']:.3f},"
            f"{info['sem_norm_min']:.3f}/{info['sem_norm_mean']:.3f}/{info['sem_norm_max']:.3f}"
        )
        rows.append({
            "case": name,
            "action": action,
            "action_name": action_names[action],
            "confidence": conf,
            "top3": _top_probs(info.get("action_probs"), action_names),
            "depth_norm_min": info["depth_norm_min"],
            "depth_norm_mean": info["depth_norm_mean"],
            "depth_norm_max": info["depth_norm_max"],
            "sem_norm_min": info["sem_norm_min"],
            "sem_norm_mean": info["sem_norm_mean"],
            "sem_norm_max": info["sem_norm_max"],
        })
    return actions, rows


def _mode_pairs(args, obs_cfg):
    if args.scan_modes:
        return [
            ("unit", "unit"),
            ("meters_div255", "gray_unit"),
            ("meters", "raw"),
            ("inverse_unit", "unit"),
            ("unit", "gray_unit"),
        ]
    depth_mode = args.depth_mode or obs_cfg.get("depth_norm_mode", "meters_div255")
    sem_mode = args.sem_mode or obs_cfg.get("semantic_norm_mode", "gray_unit")
    return [(depth_mode, sem_mode)]


def main():
    parser = argparse.ArgumentParser(description="Diagnose SB3 DRL policy action distribution")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "config" / "experiment_config.yaml"))
    parser.add_argument("--depth-mode", default=None,
                        choices=["unit", "inverse_unit", "meters", "meters_div255"],
                        help="Override observation.depth_norm_mode")
    parser.add_argument("--sem-mode", default=None,
                        choices=["unit", "raw", "gray_unit", "gray_raw"],
                        help="Override observation.semantic_norm_mode")
    parser.add_argument("--frame", action="append",
                        help="Saved *_calib_frame.npz from test_live_nocontrol.py; may be repeated")
    parser.add_argument("--frame-glob", action="append",
                        help="Glob for saved *_calib_frame.npz live frames; may be repeated")
    parser.add_argument("--scan-modes", action="store_true",
                        help="Run common depth/semantic encoding pairs")
    parser.add_argument("--cases", default="near_safe,mid_safe,far_safe,mid_danger,gradient_safe,gradient_danger,checker_mid,random")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-json", default=None,
                        help="Optional path to write machine-readable diagnosis report")
    parser.add_argument("--fail-on-collapse", action="store_true",
                        help="Exit nonzero if any checked encoding maps all inputs to one action")
    parser.add_argument("--allow-cpu-policy", action="store_true",
                        help="Bench/debug only: allow SB3 policy inference on CPU even if config requires GPU")
    args = parser.parse_args()

    cfg = _load_config(Path(args.config))
    obs_cfg = cfg["observation"]
    depth_cfg = cfg["depth_projection"]
    decision_cfg = cfg["decision"]
    uav_cfg = cfg.get("uav", {})

    h = int(obs_cfg["img_height"])
    w = int(obs_cfg["img_width"])
    dmax = float(depth_cfg.get("max_range", 30.0))

    from control.action_decomposer import ActionDecomposer
    from rl.rl_agent import RLAgent

    try:
        decomposer = ActionDecomposer(uav_cfg)
        action_names = list(decomposer.action_names)
        action_frame = decomposer.action_frame
        action_lateral_sign = decomposer.action_lateral_sign
    except Exception as exc:
        print(f"[WARN] Failed to load ActionDecomposer mapping from config: {exc}")
        action_names = DEFAULT_ACTION_NAMES
        action_frame = uav_cfg.get("action_frame", "body")
        action_lateral_sign = int(uav_cfg.get("action_lateral_sign", -1))

    items = []
    frame_paths = _expand_frame_paths(args)
    if frame_paths:
        for frame_path in frame_paths:
            depth, sem = _load_frame(frame_path)
            items.append((Path(frame_path).name, depth, sem))
    else:
        import numpy as np

        rng = np.random.default_rng(args.seed)
        for case in [item.strip() for item in args.cases.split(",") if item.strip()]:
            depth, sem = _case_maps(case, h, w, dmax, rng)
            items.append((case, depth, sem))

    print(f"Policy: {_resolve(decision_cfg['policy_weights_path'])}")
    print(f"Input set: {'live frames' if args.frame else 'synthetic cases'} ({len(items)} items)")
    print(
        "Action mapping: "
        f"frame={action_frame} lateral_sign={action_lateral_sign} act3={action_names[3]}"
    )

    overall_actions = {}
    report = {
        "policy": str(_resolve(decision_cfg["policy_weights_path"])),
        "input_set": "live frames" if args.frame else "synthetic cases",
        "items": [name for name, _, _ in items],
        "action_frame": action_frame,
        "action_lateral_sign": action_lateral_sign,
        "action_names": action_names,
        "encodings": [],
        "collapsed_encodings": [],
    }
    for depth_mode, sem_mode in _mode_pairs(args, obs_cfg):
        print(f"\nEncoding: depth={depth_mode}, semantic={sem_mode}, dmax={dmax}")
        agent = RLAgent(
            str(_resolve(decision_cfg["policy_weights_path"])),
            img_size=(w, h),
            dmax=dmax,
            depth_norm_mode=depth_mode,
            semantic_norm_mode=sem_mode,
            require_gpu=bool(decision_cfg.get("require_gpu", True)) and not args.allow_cpu_policy,
        )
        if agent.model is None:
            print("[FAIL] DRL model failed to load; dummy policy would hide the real issue")
            return 2
        actions, rows = _run_sweep(agent, items, action_names)
        overall_actions[(depth_mode, sem_mode)] = actions

        unique = sorted(set(actions))
        collapsed = len(unique) == 1
        encoding_report = {
            "depth_mode": depth_mode,
            "semantic_mode": sem_mode,
            "unique_actions": unique,
            "unique_action_names": [action_names[action] for action in unique],
            "collapsed": collapsed,
            "rows": rows,
        }
        report["encodings"].append(encoding_report)
        if len(unique) == 1:
            print(f"[WARN] All inputs selected action {unique[0]}({action_names[unique[0]]}).")
            report["collapsed_encodings"].append({
                "depth_mode": depth_mode,
                "semantic_mode": sem_mode,
                "action": unique[0],
                "action_name": action_names[unique[0]],
            })
        else:
            named = ", ".join(f"{a}({action_names[a]})" for a in unique)
            print(f"[OK] Inputs produced multiple actions: {named}")

    if len(overall_actions) > 1:
        print("\nMode summary:")
        for (depth_mode, sem_mode), actions in overall_actions.items():
            uniq = ",".join(f"{a}({action_names[a]})" for a in sorted(set(actions)))
            print(f"  depth={depth_mode:13s} semantic={sem_mode:9s}: {uniq}")
    if args.out_json:
        out_path = Path(args.out_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"\nWrote JSON report: {out_path}")
    if args.fail_on_collapse and report["collapsed_encodings"]:
        print("[FAIL] One or more encodings collapsed to a single action.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
