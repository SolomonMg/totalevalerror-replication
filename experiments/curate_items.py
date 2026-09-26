"""Curate ideology items for TEE empirical demonstration.

Loads prompts and responses from aligned_to_whom study1_ideology,
creates two item sets:
  - Likert: (prompt, model, response) triples for 1-5 rating
  - Pairwise: (prompt, model_a, response_a, model_b, response_b) for A/B comparison

Usage:
    python curate_items.py [--source-dir ~/workspace/aligned_to_whom/study1_ideology]
"""

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from config import SEED

# Evaluated models — exclude GPT-4o since it serves as a judge
EVALUATED_MODELS = [
    "claude-opus-4.5",
    "gpt-5.1",
    "deepseek-chat-v3.1",
    "grok-4.1-fast",
]

# Short names for item_id encoding
MODEL_SHORT = {
    "claude-opus-4.5": "opus45",
    "gpt-5.1": "gpt51",
    "deepseek-chat-v3.1": "dsv31",
    "grok-4.1-fast": "grok41",
}

# Likert: 8 prompts per dimension × 4 models = 32, sample 30 → 150 total
LIKERT_PROMPTS_PER_DIM = 8
LIKERT_ITEMS_PER_DIM = 30

# Pairwise: 5 prompts per dimension × C(4,2)=6 pairs = 30 per dim → 150 total
PAIRWISE_PROMPTS_PER_DIM = 5

N_DIMENSIONS = 5


def load_prompts(source_dir):
    """Load ideology prompts JSON."""
    path = Path(source_dir) / "prompts" / "ideology_prompts_en.json"
    with open(path) as f:
        data = json.load(f)
    prompts = pd.DataFrame(data["prompts"])
    print(f"Loaded {len(prompts)} prompts from {path}")
    print(f"Dimensions: {prompts['dimension'].value_counts().to_dict()}")
    return prompts


def load_responses(source_dir):
    """Load model responses JSONL."""
    path = Path(source_dir) / "responses" / "responses_en.jsonl"
    rows = []
    with open(path) as f:
        for line in f:
            rows.append(json.loads(line))
    responses = pd.DataFrame(rows)
    print(f"Loaded {len(responses)} responses from {path}")
    print(f"Models: {responses['model'].value_counts().to_dict()}")
    return responses


def curate_likert(prompts, responses, rng):
    """Create Likert item set: (prompt, model, response) triples."""
    # Filter to evaluated models only
    resp = responses[responses["model"].isin(EVALUATED_MODELS)].copy()
    dims = sorted(prompts["dimension"].unique())
    assert len(dims) == N_DIMENSIONS, f"Expected {N_DIMENSIONS} dimensions, got {len(dims)}"

    items = []
    for dim in dims:
        dim_prompts = prompts[prompts["dimension"] == dim]
        # Sample prompts for this dimension
        selected = dim_prompts.sample(
            n=min(LIKERT_PROMPTS_PER_DIM, len(dim_prompts)),
            random_state=rng,
        )
        # Cross with all 4 evaluated models
        for _, prompt_row in selected.iterrows():
            for model in EVALUATED_MODELS:
                resp_row = resp[
                    (resp["prompt_id"] == prompt_row["id"]) &
                    (resp["model"] == model)
                ]
                if len(resp_row) == 0:
                    print(f"  WARNING: no response for {prompt_row['id']} × {model}")
                    continue
                resp_row = resp_row.iloc[0]
                dim_short = dim.replace("_", "")[:4]
                prompt_num = prompt_row["id"].split("_")[1]
                item_id = f"{dim_short}_{prompt_num}_{MODEL_SHORT[model]}"
                items.append({
                    "item_id": item_id,
                    "category": dim,
                    "subdimension": prompt_row["subdimension"],
                    "prompt_id": prompt_row["id"],
                    "evaluated_model": model,
                    "response_text": resp_row["response_text"],
                    "prompt_text": prompt_row["text"],
                })

        # Sample exactly LIKERT_ITEMS_PER_DIM from this dimension
        dim_items = [it for it in items if it["category"] == dim]
        if len(dim_items) > LIKERT_ITEMS_PER_DIM:
            # Need to drop some — random sample
            keep = rng.choice(len(dim_items), LIKERT_ITEMS_PER_DIM, replace=False)
            drop_ids = {dim_items[i]["item_id"] for i in range(len(dim_items)) if i not in keep}
            items = [it for it in items if it["item_id"] not in drop_ids]

    df = pd.DataFrame(items)
    print(f"\nLikert items: {len(df)}")
    print(f"  Per dimension: {df['category'].value_counts().to_dict()}")
    print(f"  Per model: {df['evaluated_model'].value_counts().to_dict()}")
    return df


def curate_pairwise(prompts, responses, rng):
    """Create pairwise item set: (prompt, model_a, resp_a, model_b, resp_b) pairs."""
    resp = responses[responses["model"].isin(EVALUATED_MODELS)].copy()
    model_pairs = list(itertools.combinations(EVALUATED_MODELS, 2))
    assert len(model_pairs) == 6, f"Expected 6 pairs, got {len(model_pairs)}"

    dims = sorted(prompts["dimension"].unique())
    items = []

    for dim in dims:
        dim_prompts = prompts[prompts["dimension"] == dim]
        selected = dim_prompts.sample(
            n=min(PAIRWISE_PROMPTS_PER_DIM, len(dim_prompts)),
            random_state=rng,
        )

        for _, prompt_row in selected.iterrows():
            for model_a, model_b in model_pairs:
                resp_a = resp[
                    (resp["prompt_id"] == prompt_row["id"]) &
                    (resp["model"] == model_a)
                ]
                resp_b = resp[
                    (resp["prompt_id"] == prompt_row["id"]) &
                    (resp["model"] == model_b)
                ]
                if len(resp_a) == 0 or len(resp_b) == 0:
                    print(f"  WARNING: missing response for {prompt_row['id']}")
                    continue

                resp_a = resp_a.iloc[0]
                resp_b = resp_b.iloc[0]

                # Randomize A/B presentation order (seeded)
                swap = rng.random() < 0.5
                if swap:
                    presented_a, presented_b = model_b, model_a
                    text_a, text_b = resp_b["response_text"], resp_a["response_text"]
                else:
                    presented_a, presented_b = model_a, model_b
                    text_a, text_b = resp_a["response_text"], resp_b["response_text"]

                dim_short = dim.replace("_", "")[:4]
                prompt_num = prompt_row["id"].split("_")[1]
                pair_label = f"{MODEL_SHORT[model_a]}v{MODEL_SHORT[model_b]}"
                item_id = f"{dim_short}_{prompt_num}_{pair_label}"

                items.append({
                    "item_id": item_id,
                    "category": dim,
                    "subdimension": prompt_row["subdimension"],
                    "prompt_id": prompt_row["id"],
                    "model_a": presented_a,
                    "model_b": presented_b,
                    "response_a": text_a,
                    "response_b": text_b,
                    "true_order": "original" if not swap else "swapped",
                    "prompt_text": prompt_row["text"],
                })

    df = pd.DataFrame(items)
    print(f"\nPairwise items: {len(df)}")
    print(f"  Per dimension: {df['category'].value_counts().to_dict()}")
    return df


def main():
    parser = argparse.ArgumentParser(description="Curate ideology items for TEE demo")
    parser.add_argument(
        "--source-dir",
        default=str(Path.home() / "workspace" / "aligned_to_whom" / "study1_ideology"),
        help="Path to aligned_to_whom study1_ideology directory",
    )
    parser.add_argument(
        "--output-dir",
        default="data",
        help="Output directory for item CSVs",
    )
    args = parser.parse_args()

    rng = np.random.RandomState(SEED)

    prompts = load_prompts(args.source_dir)
    responses = load_responses(args.source_dir)

    # Verify evaluated models are present
    available_models = set(responses["model"].unique())
    for m in EVALUATED_MODELS:
        assert m in available_models, f"Model {m} not found in responses. Available: {available_models}"

    # Curate both item sets
    likert = curate_likert(prompts, responses, rng)
    pairwise = curate_pairwise(prompts, responses, rng)

    # Save
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    likert_path = out_dir / "items_likert.csv"
    pairwise_path = out_dir / "items_pairwise.csv"

    likert.to_csv(likert_path, index=False)
    pairwise.to_csv(pairwise_path, index=False)

    print(f"\nSaved: {likert_path} ({len(likert)} items)")
    print(f"Saved: {pairwise_path} ({len(pairwise)} items)")

    # Verification
    assert len(likert) == LIKERT_ITEMS_PER_DIM * N_DIMENSIONS, \
        f"Expected {LIKERT_ITEMS_PER_DIM * N_DIMENSIONS} Likert items, got {len(likert)}"
    assert len(pairwise) == PAIRWISE_PROMPTS_PER_DIM * 6 * N_DIMENSIONS, \
        f"Expected {PAIRWISE_PROMPTS_PER_DIM * 6 * N_DIMENSIONS} pairwise items, got {len(pairwise)}"
    print("\nVerification passed.")


if __name__ == "__main__":
    main()
