#!/usr/bin/env python3
"""Lightweight tests for the experiment requirement audit."""

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PYTHON = sys.executable


def _run(*args):
    return subprocess.run(
        [PYTHON, str(ROOT / "audit_experiment_requirements_light.py"), *args],
        cwd=str(ROOT),
        text=True,
        capture_output=True,
    )


def test_local_audit_passes_without_claiming_field_completion():
    result = _run("--strict-local")
    text = result.stdout + result.stderr
    assert result.returncode == 0, text
    assert "summary:" in text
    assert "LOCAL_FAIL=0" in text
    assert "FIELD_REQUIRED" in text
    assert "closed_loop_field_readiness" in text


def test_strict_field_fails_until_real_evidence_exists():
    result = _run("--strict-field")
    text = result.stdout + result.stderr
    assert result.returncode != 0, text
    assert "FIELD_REQUIRED" in text
    assert "field_missing:" in text


def test_markdown_audit_is_written():
    out = Path("/private/tmp/orinlanding_requirement_audit.md")
    if out.exists():
        out.unlink()
    result = _run("--out-md", str(out))
    text = result.stdout + result.stderr
    assert result.returncode == 0, text
    assert "markdown_report:" in text
    md = out.read_text(encoding="utf-8")
    assert "# Experiment Requirement Audit" in md
    assert "| closed_loop_field_readiness | FIELD_REQUIRED |" in md


def main():
    test_local_audit_passes_without_claiming_field_completion()
    test_strict_field_fails_until_real_evidence_exists()
    test_markdown_audit_is_written()
    print("=== Lightweight experiment requirement audit ===")
    print("  OK local audit passes without claiming field completion")
    print("  OK strict-field fails until real evidence exists")
    print("  OK markdown audit report is written")
    print("=== ALL PASSED ===")


if __name__ == "__main__":
    main()
