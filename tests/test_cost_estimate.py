"""Tests for cost estimation correctness."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "experiments"))
from config import MODEL_PRICING, STAGES

N_PASS = 0
N_FAIL = 0


def assert_true(condition, label):
    global N_PASS, N_FAIL
    if condition:
        N_PASS += 1
    else:
        print(f"FAIL: {label}")
        N_FAIL += 1


def assert_close(actual, expected, label, tol=0.01):
    assert_true(
        abs(actual - expected) < tol,
        f"{label}: expected {expected}, got {actual}",
    )


def compute_calls(stage_cfg, n_items=None):
    """Compute total API calls for a stage."""
    items = n_items or stage_cfg["n_items"]
    variants = stage_cfg["n_prompt_variants"]
    models = len(stage_cfg["models"])
    temps = len(stage_cfg["temperatures"])
    reps = stage_cfg["n_replications"]
    return items * variants * models * temps * reps


def test_smoke_calls():
    """Smoke: 10 items × 1 variant × 1 model × 1 temp × 3 reps = 30."""
    cfg = STAGES["smoke"]
    calls = compute_calls(cfg)
    assert_true(calls == 30, f"smoke calls: expected 30, got {calls}")


def test_pilot_calls():
    """Pilot: 30 items × 3 variants × 3 models × 3 temps × 5 reps = 4050."""
    cfg = STAGES["pilot"]
    calls = compute_calls(cfg)
    assert_true(calls == 4050, f"pilot calls: expected 4050, got {calls}")


def test_full_calls():
    """Full: 150 items × 5 variants × 3 models × 3 temps × 8 reps = 54000."""
    cfg = STAGES["full"]
    calls = compute_calls(cfg)
    assert_true(calls == 54000, f"full calls: expected 54000, got {calls}")


def test_pricing_present():
    """All judge models have pricing entries."""
    for stage_name, cfg in STAGES.items():
        for model in cfg["models"]:
            assert_true(
                model in MODEL_PRICING,
                f"{stage_name}: {model} in MODEL_PRICING",
            )


def test_full_cost_estimate():
    """Rough cost estimate for full stage (Likert, ~500 input tokens)."""
    cfg = STAGES["full"]
    calls_per_model = 54000 / 3
    avg_input = 500
    avg_output = 5

    total = 0
    for model in cfg["models"]:
        p = MODEL_PRICING[model]
        cost = (
            calls_per_model * avg_input * p["input"] / 1e6
            + calls_per_model * avg_output * p["output"] / 1e6
        )
        total += cost

    # Should be roughly $25-35 for Likert (single method)
    assert_true(10 < total < 60, f"full Likert cost: ${total:.2f} in expected range")


if __name__ == "__main__":
    test_smoke_calls()
    test_pilot_calls()
    test_full_calls()
    test_pricing_present()
    test_full_cost_estimate()
    print(f"\n{N_PASS} passed, {N_FAIL} failed")
    if N_FAIL > 0:
        sys.exit(1)
    print("All tests passed.")
