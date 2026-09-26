"""Tests for curate_items.py output correctness."""

import sys
from pathlib import Path

import pandas as pd

# Add experiments to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "experiments"))
from curate_items import (
    EVALUATED_MODELS,
    LIKERT_ITEMS_PER_DIM,
    N_DIMENSIONS,
    PAIRWISE_PROMPTS_PER_DIM,
)

N_PASS = 0
N_FAIL = 0


def assert_true(condition, label):
    global N_PASS, N_FAIL
    if condition:
        N_PASS += 1
    else:
        print(f"FAIL: {label}")
        N_FAIL += 1


def test_likert_items():
    path = Path("data/items_likert.csv")
    if not path.exists():
        print(f"SKIP: {path} not found (run curate_items.py first)")
        return

    df = pd.read_csv(path)

    # Correct total count
    expected = LIKERT_ITEMS_PER_DIM * N_DIMENSIONS
    assert_true(len(df) == expected, f"likert count: expected {expected}, got {len(df)}")

    # Required columns
    for col in ["item_id", "category", "evaluated_model", "response_text", "prompt_text"]:
        assert_true(col in df.columns, f"likert has column '{col}'")

    # No duplicate item_ids
    assert_true(df["item_id"].nunique() == len(df), "likert: no duplicate item_ids")

    # Balanced categories
    cat_counts = df["category"].value_counts()
    assert_true(
        all(cat_counts == LIKERT_ITEMS_PER_DIM),
        f"likert: {LIKERT_ITEMS_PER_DIM} per category, got {cat_counts.to_dict()}",
    )

    # 5 categories
    assert_true(df["category"].nunique() == N_DIMENSIONS, "likert: 5 categories")

    # All evaluated models represented
    models_present = set(df["evaluated_model"].unique())
    for m in EVALUATED_MODELS:
        assert_true(m in models_present, f"likert: model {m} present")

    # No empty responses
    assert_true(df["response_text"].notna().all(), "likert: no NA responses")


def test_pairwise_items():
    path = Path("data/items_pairwise.csv")
    if not path.exists():
        print(f"SKIP: {path} not found (run curate_items.py first)")
        return

    df = pd.read_csv(path)

    # Correct total count: 5 prompts × 6 pairs × 5 dims = 150
    expected = PAIRWISE_PROMPTS_PER_DIM * 6 * N_DIMENSIONS
    assert_true(len(df) == expected, f"pairwise count: expected {expected}, got {len(df)}")

    # Required columns
    for col in ["item_id", "category", "model_a", "model_b", "response_a", "response_b"]:
        assert_true(col in df.columns, f"pairwise has column '{col}'")

    # No duplicate item_ids
    assert_true(df["item_id"].nunique() == len(df), "pairwise: no duplicate item_ids")

    # Balanced categories
    cat_counts = df["category"].value_counts()
    expected_per_dim = PAIRWISE_PROMPTS_PER_DIM * 6
    assert_true(
        all(cat_counts == expected_per_dim),
        f"pairwise: {expected_per_dim} per category, got {cat_counts.to_dict()}",
    )

    # Presentation order is recorded
    assert_true("true_order" in df.columns, "pairwise: has true_order column")

    # model_a != model_b for every row
    assert_true((df["model_a"] != df["model_b"]).all(), "pairwise: model_a != model_b")


if __name__ == "__main__":
    test_likert_items()
    test_pairwise_items()
    print(f"\n{N_PASS} passed, {N_FAIL} failed")
    if N_FAIL > 0:
        sys.exit(1)
    print("All tests passed.")
