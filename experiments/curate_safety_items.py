"""Curate safety items from AILuminate demo dataset.

Stratified sample: 3 prompts per hazard category × 12 categories = 36 prompts.
Cross with 4 SUT models → 144 items total.

Usage:
    python curate_safety_items.py [--source data/ailuminate_demo.csv]
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from config import SEED

# SUT models (same as ideology experiment)
SUT_MODELS = [
    "claude-opus-4.5",
    "gpt-5.1",
    "deepseek-chat-v3.1",
    "grok-4.1-fast",
]

SUT_SHORT = {
    "claude-opus-4.5": "opus45",
    "gpt-5.1": "gpt51",
    "deepseek-chat-v3.1": "dsv31",
    "grok-4.1-fast": "grok41",
}

# Group spc_* subcategories into a single "Specialized Advice" category
SPC_SUBCATS = {"spc_ele", "spc_fin", "spc_hlt", "spc_lgl"}

HAZARD_NAMES = {
    "cse": "Child Sexual Exploitation",
    "dfm": "Defamation",
    "hte": "Hate",
    "ipv": "Intellectual Property",
    "iwp": "Indiscriminate Weapons",
    "ncr": "Non-Violent Crimes",
    "prv": "Privacy",
    "spc": "Specialized Advice",
    "src": "Sex-Related Crimes",
    "ssh": "Suicide & Self-Harm",
    "sxc_prn": "Sexual Content",
    "vcr": "Violent Crimes",
}

PROMPTS_PER_CATEGORY = 3
N_CATEGORIES = 12


def curate_safety_items(source_path, rng):
    """Create safety item set: stratified sample × 4 SUT models."""
    df = pd.read_csv(source_path)

    # Group spc_* subcategories
    df["category"] = df["hazard"].apply(lambda h: "spc" if h in SPC_SUBCATS else h)

    categories = sorted(df["category"].unique())
    assert len(categories) == N_CATEGORIES, (
        f"Expected {N_CATEGORIES} categories, got {len(categories)}: {categories}"
    )

    # Stratified sample of prompts
    prompts = []
    for cat in categories:
        cat_df = df[df["category"] == cat]
        selected = cat_df.sample(
            n=min(PROMPTS_PER_CATEGORY, len(cat_df)),
            random_state=rng,
        )
        for idx, (_, row) in enumerate(selected.iterrows()):
            prompts.append({
                "prompt_idx": f"{cat}_{idx + 1:02d}",
                "category": cat,
                "hazard_name": HAZARD_NAMES[cat],
                "hazard_code": row["hazard"],
                "persona": row["persona"],
                "prompt_text": row["prompt_text"],
                "ailuminate_id": row["release_prompt_id"],
            })

    # Cross with SUT models to create items
    items = []
    for prompt in prompts:
        for sut in SUT_MODELS:
            item_id = f"{prompt['prompt_idx']}_{SUT_SHORT[sut]}"
            items.append({
                "item_id": item_id,
                "prompt_idx": prompt["prompt_idx"],
                "category": prompt["category"],
                "hazard_name": prompt["hazard_name"],
                "hazard_code": prompt["hazard_code"],
                "persona": prompt["persona"],
                "sut_model": sut,
                "prompt_text": prompt["prompt_text"],
                "ailuminate_id": prompt["ailuminate_id"],
            })

    result = pd.DataFrame(items)
    print(f"\nSafety items: {len(result)}")
    print(f"  Unique prompts: {result['prompt_idx'].nunique()}")
    print(f"  Per category: {result.groupby('category').size().to_dict()}")
    print(f"  Per SUT: {result.groupby('sut_model').size().to_dict()}")
    return result


def main():
    parser = argparse.ArgumentParser(description="Curate safety items from AILuminate")
    parser.add_argument(
        "--source", default="data/ailuminate_demo.csv",
        help="Path to AILuminate demo CSV",
    )
    parser.add_argument(
        "--output", default="data/items_safety.csv",
        help="Output path for curated items",
    )
    args = parser.parse_args()

    rng = np.random.RandomState(SEED)
    items = curate_safety_items(args.source, rng)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    items.to_csv(args.output, index=False)
    print(f"\nSaved: {args.output} ({len(items)} items)")

    # Verification
    expected = PROMPTS_PER_CATEGORY * N_CATEGORIES * len(SUT_MODELS)
    assert len(items) == expected, f"Expected {expected} items, got {len(items)}"
    print("Verification passed.")


if __name__ == "__main__":
    main()
