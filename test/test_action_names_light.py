#!/usr/bin/env python3
"""Lightweight checks that diagnostics use the configured action names."""

from control.action_decomposer import ActionDecomposer
from diagnose_drl_policy import DEFAULT_ACTION_NAMES, _top_probs as diag_top_probs
from diagnostics.action_monitor import _top_probs as monitor_top_probs


def test_default_mapping_matches_original_deeprl():
    dec = ActionDecomposer({"action_lateral_sign": -1})
    assert dec.action_names == DEFAULT_ACTION_NAMES
    assert dec.action_id_to_name(3) == "W"
    assert dec.action_id_to_name(7) == "E"


def test_mirror_mapping_is_explicit():
    dec = ActionDecomposer({"action_lateral_sign": 1})
    assert dec.action_id_to_name(3) == "E"
    assert dec.action_id_to_name(7) == "W"


def test_top_probs_use_runtime_names():
    original = ActionDecomposer({"action_lateral_sign": -1}).action_names
    mirror = ActionDecomposer({"action_lateral_sign": 1}).action_names
    probs = [0.0] * 10
    probs[3] = 0.7
    probs[7] = 0.2
    probs[1] = 0.1
    assert diag_top_probs(probs, original) == "p=3:W:0.700,7:E:0.200,1:N:0.100"
    assert diag_top_probs(probs, mirror) == "p=3:E:0.700,7:W:0.200,1:N:0.100"
    assert monitor_top_probs(probs, original) == "p=3:W:0.700,7:E:0.200,1:N:0.100"


def main():
    test_default_mapping_matches_original_deeprl()
    test_mirror_mapping_is_explicit()
    test_top_probs_use_runtime_names()
    print("=== Lightweight action-name acceptance ===")
    print("  OK default action_lateral_sign=-1: act3=W, act7=E")
    print("  OK mirror action_lateral_sign=+1: act3=E, act7=W")
    print("  OK DRL diagnostic top-prob labels use runtime names")
    print("=== ALL PASSED ===")


if __name__ == "__main__":
    main()
