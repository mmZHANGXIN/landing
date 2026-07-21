#!/usr/bin/env python3
"""Preflight checks for the Orin landing experiment.

This script is intentionally conservative: it validates the pieces that must be
true before connecting the control loop to a real flight controller.
"""

import argparse
import ast
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from diagnostics.flight_ready import iter_flight_ready_checks


class CheckReport:
    def __init__(self):
        self.failed = 0
        self.warned = 0

    def ok(self, msg):
        print(f"[OK] {msg}")

    def warn(self, msg):
        self.warned += 1
        print(f"[WARN] {msg}")

    def fail(self, msg):
        self.failed += 1
        print(f"[FAIL] {msg}")


def _load_config(path: Path):
    path = path.resolve()
    try:
        import yaml
        with open(path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
    except ModuleNotFoundError as exc:
        if exc.name != "yaml":
            raise
        cfg = _load_simple_yaml(path)
    parent = cfg.pop("extends", None)
    if parent:
        parent_path = Path(parent)
        if not parent_path.is_absolute():
            parent_path = path.parent / parent_path
        return _merge_dicts(_load_config(parent_path), cfg)
    return cfg


def _merge_dicts(base: dict, override: dict) -> dict:
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _merge_dicts(base[key], value)
        else:
            base[key] = value
    return base


def _strip_yaml_comment(line: str) -> str:
    in_single = False
    in_double = False
    escaped = False
    for idx, char in enumerate(line):
        if char == "\\" and in_double:
            escaped = not escaped
            continue
        if char == "'" and not in_double:
            in_single = not in_single
        elif char == '"' and not in_single and not escaped:
            in_double = not in_double
        elif char == "#" and not in_single and not in_double:
            return line[:idx]
        escaped = False
    return line


def _parse_simple_yaml_scalar(value: str):
    value = value.strip()
    if value in ("", "null", "Null", "NULL", "~"):
        return None
    if value in ("true", "True", "TRUE"):
        return True
    if value in ("false", "False", "FALSE"):
        return False
    if value.startswith(('"', "'", "[")):
        return ast.literal_eval(value)
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def _next_yaml_content(lines, start_idx):
    for raw in lines[start_idx:]:
        line = _strip_yaml_comment(raw).rstrip()
        if line.strip():
            return len(line) - len(line.lstrip(" ")), line.strip()
    return None, None


def _load_simple_yaml(path: Path):
    """Tiny fallback parser for this repository's experiment_config.yaml."""
    lines = path.read_text(encoding="utf-8").splitlines()
    root = {}
    stack = [(-1, root)]

    for idx, raw in enumerate(lines):
        line = _strip_yaml_comment(raw).rstrip()
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip(" "))
        content = line.strip()
        while indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]

        if content.startswith("- "):
            if not isinstance(parent, list):
                raise ValueError(f"Unsupported YAML list at line {idx + 1}: {raw}")
            parent.append(_parse_simple_yaml_scalar(content[2:]))
            continue

        if ":" not in content:
            raise ValueError(f"Unsupported YAML line {idx + 1}: {raw}")
        key, value = content.split(":", 1)
        key = key.strip()
        value = value.strip()
        if value:
            parent[key] = _parse_simple_yaml_scalar(value)
            continue

        next_indent, next_content = _next_yaml_content(lines, idx + 1)
        child = [] if next_indent is not None and next_indent > indent and next_content.startswith("- ") else {}
        parent[key] = child
        stack.append((indent, child))

    return root


def _resolve(path_value):
    if path_value is None:
        return None
    path = Path(path_value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def check_config(cfg, report):
    required = [
        ("runtime",),
        ("observation", "img_width"),
        ("observation", "img_height"),
        ("perception", "halss_weight_path"),
        ("perception", "halss_backend"),
        ("perception", "require_gpu"),
        ("depth_projection", "mode"),
        ("depth_projection", "backend"),
        ("depth_projection", "max_range"),
        ("decision", "backend"),
        ("decision", "onnx_model_path"),
        ("decision", "require_gpu"),
        ("uav", "action_frame"),
        ("uav", "action_lateral_sign"),
        ("uav", "yaw_rate_rad_s"),
        ("visualization", "binary_semantic_window_title"),
    ]
    for keys in required:
        cur = cfg
        for key in keys:
            if key not in cur:
                report.fail(f"Missing config key: {'.'.join(keys)}")
                break
            cur = cur[key]
        else:
            report.ok(f"Config key present: {'.'.join(keys)}")

    if cfg["depth_projection"].get("mode") != "training_camera":
        report.warn("depth_projection.mode is not the active 'training_camera' path")
    if cfg["depth_projection"].get("backend") != "numpy_opencv_nn_fill":
        report.warn("depth_projection.backend is not the active OpenCV NN-fill path")
    if cfg["perception"].get("halss_backend") != "bayesian_unet":
        report.warn("perception.halss_backend is not 'bayesian_unet'; HALSS Bayesian path is disabled")
    if cfg["perception"].get("require_gpu") is not True:
        report.warn("perception.require_gpu is not true; HALSS may fall back to CPU")
    if cfg["decision"].get("require_gpu") is not True:
        report.warn("decision.require_gpu is not true; DRL policy may fall back to CPU")
    obs_cfg = cfg["observation"]
    if obs_cfg.get("depth_norm_mode") != "raw_meters_graph_scaled":
        report.warn(
            "observation.depth_norm_mode must feed raw metres because ONNX input/truediv "
            "already performs the original SB2 /255"
        )
    if obs_cfg.get("semantic_norm_mode") != "raw_gray_graph_scaled":
        report.warn(
            "observation.semantic_norm_mode must feed raw grayscale because ONNX "
            "input/truediv already performs /255"
        )
    if cfg["uav"].get("action_frame") != "body":
        report.warn("uav.action_frame is not 'body'; yaw-fault body-frame compensation is disabled")
    action_lateral_sign = cfg["uav"].get("action_lateral_sign")
    if action_lateral_sign != -1:
        report.warn(
            "uav.action_lateral_sign is not -1; this no longer matches the original "
            "DeepRL quadrotor_env.py action mapping"
        )
    if cfg.get("visualization", {}).get("binary_semantic_window_title") != "binary semantic":
        report.warn("visualization.binary_semantic_window_title is not exactly 'binary semantic'")


def check_flight_ready_config(cfg, report):
    """Strict gates for the full real-flight experiment objective."""
    for ok, message in iter_flight_ready_checks(cfg):
        if ok:
            report.ok(message)
        else:
            report.fail(message)


def check_files(cfg, report):
    for label, path_value in [
        ("HALSS weight", cfg["perception"].get("halss_weight_path")),
        ("ONNX DRL policy", cfg["decision"].get("onnx_model_path")),
        ("ONNX metadata", cfg["decision"].get("onnx_meta_path")),
    ]:
        path = _resolve(path_value)
        if path and path.is_file():
            report.ok(f"{label} exists: {path}")
        else:
            report.fail(f"{label} missing: {path}")

    gp = cfg.get("global_prior", {})
    if gp.get("enabled"):
        if str(gp.get("mode", "gps")).lower() == "local_body_offset":
            offset = gp.get("local_body_offset_m")
            if isinstance(offset, (list, tuple)) and len(offset) >= 3:
                report.ok(f"Global prior uses indoor body offset {offset[:3]}")
            else:
                report.fail("Indoor global prior requires local_body_offset_m [forward,right,up]")
        elif gp.get("target_lat") is not None and gp.get("target_lon") is not None:
            if gp.get("target_source") == "gis":
                report.ok("Global prior uses GIS-derived configured GPS target")
            else:
                report.warn(
                    "Global prior target_lat/lon is configured but target_source is not 'gis'; "
                    "strict flight gates will reject it"
                )
        elif gp.get("image_path") and gp.get("bounds"):
            path = _resolve(gp.get("image_path"))
            if path and path.is_file():
                report.ok("Global prior GIS image and bounds configured")
            else:
                report.fail(f"Global prior image missing: {path}")
        else:
            report.fail("global_prior.enabled=true but no target_lat/lon or image_path+bounds")
    else:
        report.warn("global_prior.enabled=false; pipeline will skip GIS safe-area guidance")


def check_drl_metadata(cfg, report):
    try:
        import json
        path = _resolve(cfg["decision"].get("onnx_meta_path"))
        result = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        report.warn(f"DRL metadata inspection failed: {exc}")
        return
    shape = result.get("onnx_input_shape")
    normalization = str(result.get("normalization", ""))
    if shape != [-1, 128, 128, 2]:
        report.fail(f"DRL metadata input shape mismatch: {shape}")
    elif "255" not in normalization:
        report.fail(f"DRL metadata normalization is not obs/255: {normalization!r}")
    else:
        report.ok(f"ONNX metadata matches input contract: shape={shape}, {normalization}")


def check_cuda(report, required: bool = True):
    try:
        import torch
    except Exception as exc:
        if required:
            report.fail(f"torch import failed: {exc}")
        else:
            report.warn(f"Skipping CUDA check: torch import failed ({exc})")
        return None
    if torch.cuda.is_available():
        report.ok(f"CUDA available: {torch.cuda.get_device_name(0)}")
    else:
        if required:
            report.fail("CUDA is not available; flight inference is denied")
        else:
            report.warn("Skipping CUDA requirement: CUDA is not available on this host")
    return torch


def check_depth_projection(cfg, report):
    dcfg = cfg["depth_projection"]
    grid = int(dcfg.get("grid_cells", 0))
    dmax = float(dcfg.get("max_range", 0.0))
    if grid == 128 and dmax > 0.0:
        report.ok("Active BEV/NN-fill depth configuration sanity check passed")
    else:
        report.fail(f"Depth configuration invalid: grid_cells={grid}, max_range={dmax}")


def check_action_decomposer(cfg, report):
    import math
    from control.action_decomposer import ActionDecomposer

    dec = ActionDecomposer(cfg["uav"])
    _, v_ned, yr = dec.decompose(3, math.radians(90.0))
    lateral_sign = int(cfg["uav"].get("action_lateral_sign", -1))
    if cfg["uav"].get("action_frame", "body") == "body":
        expected = [-float(lateral_sign), 0.0, 0.0]
    else:
        expected = [0.0, float(lateral_sign), 0.0]
    if all(abs(float(a) - float(b)) <= 1e-5 for a, b in zip(v_ned, expected)):
        report.ok(
            f"Action decomposer frame={dec.action_frame} "
            f"lateral_sign={dec.action_lateral_sign} yaw compensation passed"
        )
    else:
        report.fail(f"Action decomposer mismatch: v_ned={v_ned}, expected={expected}")
    report.ok(f"Configured yaw_rate_rad_s={yr:.3f}")


def check_models(cfg, report, torch):
    if torch is None or not torch.cuda.is_available():
        report.warn("Skipping model load checks because CUDA/torch is unavailable")
        return

    try:
        import onnxruntime as ort
        session = ort.InferenceSession(str(_resolve(cfg["decision"]["onnx_model_path"])))
        input_shape = session.get_inputs()[0].shape
        if len(input_shape) == 4:
            report.ok(f"ONNX DRL policy loads: input={input_shape}")
        else:
            report.fail(f"Unexpected ONNX input shape: {input_shape}")
    except Exception as exc:
        report.fail(f"ONNX DRL model check failed: {exc}")


def main():
    parser = argparse.ArgumentParser(description="Orin landing preflight checks")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "config" / "experiment_config.yaml"))
    parser.add_argument("--skip-model-load", action="store_true",
                        help="Only check paths/config/geometry; skip CUDA model initialization")
    parser.add_argument("--flight-ready", action="store_true",
                        help="Enable strict gates required before connecting the real flight loop")
    args = parser.parse_args()

    report = CheckReport()
    cfg_path = Path(args.config)
    if not cfg_path.is_file():
        report.fail(f"Config not found: {cfg_path}")
        sys.exit(2)

    try:
        cfg = _load_config(cfg_path)
        report.ok(f"Config loaded: {cfg_path}")
    except Exception as exc:
        report.fail(f"Config load failed: {exc}")
        sys.exit(2)

    check_config(cfg, report)
    check_files(cfg, report)
    if args.flight_ready:
        check_flight_ready_config(cfg, report)
    check_drl_metadata(cfg, report)
    optional_runtime_checks = [
        ("depth projection geometry", check_depth_projection),
        ("action decomposer", check_action_decomposer),
    ]
    for label, check in optional_runtime_checks:
        try:
            check(cfg, report)
        except (ImportError, ModuleNotFoundError) as exc:
            if args.skip_model_load:
                report.warn(f"Skipping {label}: missing dependency ({exc})")
            else:
                report.fail(f"{label} check failed: missing dependency ({exc})")
        except RuntimeError as exc:
            if args.skip_model_load:
                report.warn(f"Skipping {label}: runtime unavailable ({exc})")
            else:
                report.fail(f"{label} check failed: {exc}")

    torch = check_cuda(report, required=not args.skip_model_load)
    if not args.skip_model_load:
        check_models(cfg, report, torch)

    print(f"\nPreflight summary: failed={report.failed}, warnings={report.warned}")
    sys.exit(1 if report.failed else 0)


if __name__ == "__main__":
    main()
