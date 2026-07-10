#!/usr/bin/env python3
"""Lightweight tests for Orin environment check script."""

import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PYTHON = sys.executable


def test_non_strict_runs_on_non_orin_and_writes_markdown():
    with tempfile.TemporaryDirectory() as tmp:
        report = Path(tmp) / "orin_env.md"
        result = subprocess.run(
            [
                PYTHON,
                str(ROOT / "check_orin_env.py"),
                "--out-md",
                str(report),
            ],
            cwd=str(ROOT),
            text=True,
            capture_output=True,
        )
        text = result.stdout + result.stderr
        assert result.returncode == 0, text
        assert "required_failures: 0" in text
        assert "markdown_report:" in text
        md = report.read_text(encoding="utf-8")
        assert "# Orin Environment Check" in md
        assert "torch_cuda" in md
        assert "stable_baselines3" in md


def test_require_jetson_fails_on_non_jetson():
    if Path("/etc/nv_tegra_release").exists():
        return
    result = subprocess.run(
        [
            PYTHON,
            str(ROOT / "check_orin_env.py"),
            "--require-jetson",
        ],
        cwd=str(ROOT),
        text=True,
        capture_output=True,
    )
    text = result.stdout + result.stderr
    assert result.returncode != 0, text
    assert "jetpack" in text
    assert "required_failures:" in text


def test_strict_display_requirement_is_documented():
    text = (ROOT / "check_orin_env.py").read_text(encoding="utf-8")
    assert "DISPLAY missing" in text
    assert "live binary semantic/depth windows" in text


def main():
    test_non_strict_runs_on_non_orin_and_writes_markdown()
    test_require_jetson_fails_on_non_jetson()
    test_strict_display_requirement_is_documented()
    print("=== Lightweight Orin environment check ===")
    print("  OK non-strict environment scan writes Markdown")
    print("  OK require-jetson fails on non-Jetson hosts")
    print("  OK strict DISPLAY requirement is documented")
    print("=== ALL PASSED ===")


if __name__ == "__main__":
    main()
