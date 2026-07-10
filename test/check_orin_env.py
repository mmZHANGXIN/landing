#!/usr/bin/env python3
"""Orin runtime environment compatibility checks."""

from __future__ import annotations

import argparse
import importlib
import os
import platform
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parent


@dataclass
class EnvCheck:
    key: str
    required: bool
    ok: bool
    detail: str
    fix: str


def _run_text(cmd):
    try:
        proc = subprocess.run(cmd, text=True, capture_output=True, timeout=5)
    except Exception as exc:
        return False, str(exc)
    text = (proc.stdout or proc.stderr or "").strip().splitlines()
    first = text[0] if text else f"exit={proc.returncode}"
    return proc.returncode == 0, first


def _module_version(name, attr="__version__"):
    try:
        module = importlib.import_module(name)
    except Exception as exc:
        return False, f"import failed: {exc}"
    return True, str(getattr(module, attr, "import ok"))


def _check_jetpack(required=True):
    path = Path("/etc/nv_tegra_release")
    if path.exists():
        return EnvCheck("jetpack", required, True, path.read_text(errors="replace").strip(), "Install JetPack 5.x on Jetson Orin.")
    return EnvCheck("jetpack", required, False, "/etc/nv_tegra_release missing", "Run on Jetson Orin with JetPack 5.x.")


def _check_command(key, command, required, fix):
    exe = command[0]
    if shutil.which(exe) is None:
        return EnvCheck(key, required, False, f"{exe} not found", fix)
    ok, detail = _run_text(command)
    return EnvCheck(key, required, ok, detail, fix)


def _check_ros_setup(required=True):
    candidates = [
        Path("/opt/ros/galactic/setup.bash"),
        Path("/opt/ros/humble/setup.bash"),
    ]
    found = [str(path) for path in candidates if path.exists()]
    if found:
        return EnvCheck("ros_setup", required, True, ", ".join(found), "Install ROS2 Galactic and source setup.bash.")
    return EnvCheck("ros_setup", required, False, "no /opt/ros/{galactic,humble}/setup.bash", "Install/source ROS2 Galactic for FAST-LIO.")


def _check_torch_cuda(required=True):
    ok, detail = _module_version("torch")
    if not ok:
        return EnvCheck("torch_cuda", required, False, detail, "Install Jetson PyTorch with CUDA support.")
    try:
        import torch
        cuda_ok = bool(torch.cuda.is_available())
        if cuda_ok:
            detail = f"torch={torch.__version__}, cuda=True, device={torch.cuda.get_device_name(0)}"
        else:
            detail = f"torch={torch.__version__}, cuda=False"
        return EnvCheck("torch_cuda", required, cuda_ok, detail, "Install Jetson PyTorch wheel and confirm CUDA is visible.")
    except Exception as exc:
        return EnvCheck("torch_cuda", required, False, str(exc), "Install Jetson PyTorch wheel and confirm CUDA is visible.")


def _check_cv2_gui(required=True):
    ok, detail = _module_version("cv2")
    if not ok:
        return EnvCheck("opencv", required, False, detail, "Install opencv-python or system OpenCV.")
    display = os.environ.get("DISPLAY")
    gui_hint = "DISPLAY set" if display else "DISPLAY missing"
    if required and not display:
        return EnvCheck(
            "opencv",
            required,
            False,
            f"cv2={detail}, {gui_hint}",
            "Export DISPLAY=:0 or run from a graphical session for live binary semantic/depth windows.",
        )
    return EnvCheck("opencv", required, True, f"cv2={detail}, {gui_hint}", "Install OpenCV and export DISPLAY=:0 for live windows.")


def _check_import(key, module, required, fix):
    ok, detail = _module_version(module)
    return EnvCheck(key, required, ok, detail, fix)


def collect_checks(args):
    return [
        _check_jetpack(required=args.require_jetson),
        _check_command("nvcc", ["nvcc", "--version"], args.strict, "Install CUDA toolkit from JetPack."),
        _check_command("ros2_cli", ["ros2", "--help"], args.strict, "Install/source ROS2 Galactic."),
        _check_ros_setup(required=args.strict),
        _check_import("rclpy", "rclpy", args.strict, "Source ROS2 setup.bash before running Python."),
        _check_torch_cuda(required=args.strict),
        _check_cv2_gui(required=args.strict),
        _check_import("numpy", "numpy", args.strict, "Install numpy compatible with Jetson PyTorch."),
        _check_import("yaml", "yaml", args.strict, "Install PyYAML."),
        _check_import("stable_baselines3", "stable_baselines3", args.strict, "Install stable-baselines3."),
        _check_import("mavsdk", "mavsdk", args.strict, "Install mavsdk."),
    ]


def _print_checks(checks):
    for check in checks:
        if check.ok:
            status = "OK"
        elif check.required:
            status = "FAIL"
        else:
            status = "WARN"
        print(f"[{status}] {check.key}: {check.detail}")
        if status != "OK":
            print(f"       fix: {check.fix}")


def _write_markdown(path: Path, checks):
    path = path if path.is_absolute() else ROOT / path
    path.parent.mkdir(parents=True, exist_ok=True)
    failures = [check.key for check in checks if check.required and not check.ok]
    lines = [
        "# Orin Environment Check",
        "",
        f"- generated_at: `{datetime.now().isoformat(timespec='seconds')}`",
        f"- python: `{sys.version.split()[0]}`",
        f"- platform: `{platform.platform()}`",
        f"- required_failures: `{len(failures)}`",
        f"- failed_keys: `{', '.join(failures) if failures else 'none'}`",
        "",
        "| Key | Required | Status | Detail | Fix |",
        "| --- | --- | --- | --- | --- |",
    ]
    for check in checks:
        status = "PASS" if check.ok else ("FAIL" if check.required else "WARN")
        lines.append(
            "| {key} | {required} | {status} | {detail} | {fix} |".format(
                key=check.key,
                required=str(check.required),
                status=status,
                detail=check.detail.replace("|", "/"),
                fix=check.fix.replace("|", "/"),
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main():
    parser = argparse.ArgumentParser(description="Check Jetson Orin runtime compatibility")
    parser.add_argument("--strict", action="store_true",
                        help="Fail when required runtime dependencies are unavailable")
    parser.add_argument("--require-jetson", action="store_true",
                        help="Require /etc/nv_tegra_release to exist")
    parser.add_argument("--out-md", default=None,
                        help="Optional Markdown report path")
    args = parser.parse_args()

    checks = collect_checks(args)
    _print_checks(checks)
    failures = [check.key for check in checks if check.required and not check.ok]
    print("")
    print(f"required_failures: {len(failures)}")
    if failures:
        print("failed_keys: " + ", ".join(failures))
    if args.out_md:
        out_path = _write_markdown(Path(args.out_md), checks)
        try:
            rel = out_path.relative_to(ROOT)
        except ValueError:
            rel = out_path
        print(f"markdown_report: {rel}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
