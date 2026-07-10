#!/usr/bin/env python3
"""Source-level checks that online inference modules fail closed on CPU fallback."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _read(relpath):
    return (ROOT / relpath).read_text(encoding="utf-8")


def test_halss_requires_gpu_by_default():
    text = _read("perception/halss_bayesian.py")
    assert 'cfg.get("require_gpu", True)' in text
    assert "Bayesian UNet CPU fallback is denied for flight" in text
    assert "require_gpu={self.require_gpu}" in text


def test_rl_agent_can_fail_closed_on_cpu_fallback():
    text = _read("rl/rl_agent.py")
    assert "require_gpu: bool = False" in text
    assert "SB3/PyTorch policy CPU fallback is denied for flight" in text
    assert "FAILED to load required GPU policy" in text
    assert "require_gpu={self.require_gpu}" in text


def test_runtime_entrypoints_pass_gpu_requirements():
    pipeline = _read("pipeline.py")
    nocontrol = _read("test_live_nocontrol.py")
    preflight = _read("preflight_check.py")
    diagnose = _read("test/diagnose_drl_policy.py")

    assert 'require_gpu=bool(dec_cfg.get("require_gpu", True))' in pipeline
    assert "ONNXDRL(" in nocontrol
    assert "onnx model missing" in nocontrol
    assert 'require_gpu=bool(cfg["decision"].get("require_gpu", True))' in preflight
    assert "--allow-cpu-policy" in diagnose
    assert 'decision_cfg.get("require_gpu", True)' in diagnose


def test_strict_flight_gate_rejects_disabled_module_gpu_requirements():
    text = _read("diagnostics/flight_ready.py")
    assert "perception.require_gpu must be true" in text
    assert "decision.require_gpu must be true" in text
    assert "HALSS and DRL GPU fallback disabled" in text


def main():
    test_halss_requires_gpu_by_default()
    test_rl_agent_can_fail_closed_on_cpu_fallback()
    test_runtime_entrypoints_pass_gpu_requirements()
    test_strict_flight_gate_rejects_disabled_module_gpu_requirements()
    print("=== Lightweight GPU fail-closed contract ===")
    print("  OK HALSS Bayesian rejects CPU fallback by default")
    print("  OK RLAgent can reject CPU/dummy fallback when required")
    print("  OK pipeline/preflight/diagnostics pass GPU requirements and no-control requires ONNX")
    print("  OK strict flight gate rejects disabled module GPU requirements")
    print("=== ALL PASSED ===")


if __name__ == "__main__":
    main()
