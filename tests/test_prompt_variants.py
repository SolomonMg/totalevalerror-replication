"""Tests for create_prompt_variants.py output correctness."""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "experiments"))
from create_prompt_variants import (
    LIKERT_TEMPLATES, PAIRWISE_TEMPLATES,
    ARENA_LIKERT_COT_TEMPLATES, ARENA_PAIRWISE_FORCED_TEMPLATES,
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


def test_likert_variants():
    path = Path("data/processed/variants_likert.csv")
    if not path.exists():
        print(f"SKIP: {path} not found (run create_prompt_variants.py first)")
        return

    df = pd.read_csv(path)
    items = pd.read_csv("data/items_likert.csv")
    n_items = len(items)
    n_variants = len(LIKERT_TEMPLATES)

    # Row count
    expected = n_items * n_variants
    assert_true(len(df) == expected, f"likert variants: expected {expected}, got {len(df)}")

    # Required columns
    for col in ["item_id", "category", "variant_id", "prompt_text"]:
        assert_true(col in df.columns, f"likert variants: has '{col}'")

    # All item_ids × variant_ids present
    combos = set(zip(df["item_id"], df["variant_id"]))
    for item_id in items["item_id"]:
        for v in range(n_variants):
            assert_true(
                (item_id, v) in combos,
                f"likert: ({item_id}, {v}) present",
            )

    # No duplicates
    assert_true(len(combos) == len(df), "likert variants: no duplicates")

    # Prompt text is non-empty
    assert_true(df["prompt_text"].str.len().min() > 50, "likert: prompt_text not empty")


def test_pairwise_variants():
    path = Path("data/processed/variants_pairwise.csv")
    if not path.exists():
        print(f"SKIP: {path} not found (run create_prompt_variants.py first)")
        return

    df = pd.read_csv(path)
    items = pd.read_csv("data/items_pairwise.csv")
    n_items = len(items)
    n_variants = len(PAIRWISE_TEMPLATES)

    # Row count
    expected = n_items * n_variants
    assert_true(len(df) == expected, f"pairwise variants: expected {expected}, got {len(df)}")

    # All item_ids × variant_ids present
    combos = set(zip(df["item_id"], df["variant_id"]))
    for item_id in items["item_id"]:
        for v in range(n_variants):
            assert_true(
                (item_id, v) in combos,
                f"pairwise: ({item_id}, {v}) present",
            )

    # No duplicates
    assert_true(len(combos) == len(df), "pairwise variants: no duplicates")

    # Prompt text contains "Response A" and "Response B"
    has_a = df["prompt_text"].str.contains("Response A").all()
    has_b = df["prompt_text"].str.contains("Response B").all()
    assert_true(has_a, "pairwise: all prompts mention 'Response A'")
    assert_true(has_b, "pairwise: all prompts mention 'Response B'")


def test_arena_templates():
    """Arena scoring templates exist with expected counts and placeholders."""
    assert_true(len(ARENA_LIKERT_COT_TEMPLATES) == 5,
                 f"ARENA_LIKERT_COT_TEMPLATES: expected 5, got {len(ARENA_LIKERT_COT_TEMPLATES)}")
    assert_true(len(ARENA_PAIRWISE_FORCED_TEMPLATES) == 5,
                 f"ARENA_PAIRWISE_FORCED_TEMPLATES: expected 5, got {len(ARENA_PAIRWISE_FORCED_TEMPLATES)}")

    for i, t in enumerate(ARENA_LIKERT_COT_TEMPLATES):
        assert_true("{prompt}" in t, f"ARENA_LIKERT_COT_TEMPLATES[{i}] has {{prompt}}")
        assert_true("{response_text}" in t,
                     f"ARENA_LIKERT_COT_TEMPLATES[{i}] has {{response_text}}")
        # format() must succeed with the expected slots
        try:
            t.format(prompt="X", response_text="Y")
            assert_true(True, f"ARENA_LIKERT_COT_TEMPLATES[{i}] formats without KeyError")
        except Exception as e:
            assert_true(False, f"ARENA_LIKERT_COT_TEMPLATES[{i}]: {e}")

    for i, t in enumerate(ARENA_PAIRWISE_FORCED_TEMPLATES):
        assert_true("{prompt}" in t, f"ARENA_PAIRWISE_FORCED_TEMPLATES[{i}] has {{prompt}}")
        assert_true("{response_a}" in t,
                     f"ARENA_PAIRWISE_FORCED_TEMPLATES[{i}] has {{response_a}}")
        assert_true("{response_b}" in t,
                     f"ARENA_PAIRWISE_FORCED_TEMPLATES[{i}] has {{response_b}}")
        try:
            t.format(prompt="X", response_a="A", response_b="B")
            assert_true(True, f"ARENA_PAIRWISE_FORCED_TEMPLATES[{i}] formats without KeyError")
        except Exception as e:
            assert_true(False, f"ARENA_PAIRWISE_FORCED_TEMPLATES[{i}]: {e}")


if __name__ == "__main__":
    test_likert_variants()
    test_pairwise_variants()
    test_arena_templates()
    print(f"\n{N_PASS} passed, {N_FAIL} failed")
    if N_FAIL > 0:
        sys.exit(1)
    print("All tests passed.")
