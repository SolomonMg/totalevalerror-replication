"""Run the TEE empirical demonstration.

Queries judge LLMs to score ideology responses across items, prompt variants,
temperatures, and judge models. Supports two scoring methods:
  - likert: Rate political conservatism 1-5
  - pairwise: Compare two responses, output A or B

Usage:
    python run_demo.py --stage smoke --scoring likert
    python run_demo.py --stage pilot --scoring pairwise
    python run_demo.py --stage full --scoring likert
"""

import argparse
import json
import random
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from config import (
    MAX_RETRIES,
    MAX_TOKENS,
    MODEL_PRICING,
    REQUEST_DELAY_S,
    RETRY_BASE_DELAY_S,
    SEED,
    STAGES,
)
from query_openrouter import get_client, query_model, save_results

# System prompts for judge models
SYSTEM_PROMPTS = {
    "likert": (
        "You are a political science research assistant. Your task is to rate "
        "policy responses on a left-right ideological scale. Respond with only "
        "a single number."
    ),
    "pairwise": (
        "You are a political science research assistant. Your task is to compare "
        "two policy responses and identify which is more politically conservative. "
        "Respond with only 'A' or 'B'."
    ),
}


def load_items_and_variants(scoring, stage_cfg):
    """Load items and prompt variants for the specified scoring method."""
    items_path = f"data/items_{scoring}.csv"
    variants_path = f"data/processed/variants_{scoring}.csv"

    items = pd.read_csv(items_path)
    n_items = min(stage_cfg["n_items"], len(items))

    # Stratified sample by category
    if "category" in items.columns and items["category"].nunique() > 1:
        n_per_cat = max(1, n_items // items["category"].nunique())
        sampled = []
        for _, grp in items.groupby("category"):
            sampled.append(grp.sample(min(len(grp), n_per_cat), random_state=SEED))
        items = pd.concat(sampled).head(n_items).reset_index(drop=True)
    else:
        items = items.sample(n_items, random_state=SEED)

    item_ids = set(items["item_id"].tolist())

    # Load prompt variants
    if Path(variants_path).exists() and stage_cfg["n_prompt_variants"] > 1:
        variants = pd.read_csv(variants_path)
        variants = variants[variants["item_id"].isin(item_ids)]
        variants = variants[variants["variant_id"] < stage_cfg["n_prompt_variants"]]
    else:
        # Use original items with variant_id 0 only
        variants = items[["item_id", "category"]].copy()
        variants["variant_id"] = 0
        # Build prompt text from the first template
        if scoring == "likert":
            from create_prompt_variants import LIKERT_TEMPLATES
            template = LIKERT_TEMPLATES[0]
            variants["prompt_text"] = items["response_text"].apply(
                lambda r: template.format(response_text=r)
            )
        else:
            from create_prompt_variants import PAIRWISE_TEMPLATES
            template = PAIRWISE_TEMPLATES[0]
            variants["prompt_text"] = items.apply(
                lambda row: template.format(
                    response_a=row["response_a"],
                    response_b=row["response_b"],
                ),
                axis=1,
            )

    return items, variants


def estimate_cost(variants, stage_cfg, scoring):
    """Estimate total API cost for the run."""
    n_calls = (
        len(variants)
        * len(stage_cfg["models"])
        * len(stage_cfg["temperatures"])
        * stage_cfg["n_replications"]
    )
    # Likert: ~500 input tokens, ~5 output tokens
    # Pairwise: ~900 input tokens, ~5 output tokens
    avg_input_tokens = 500 if scoring == "likert" else 900
    avg_output_tokens = 5

    total_cost = 0
    for model in stage_cfg["models"]:
        pricing = MODEL_PRICING.get(model, {"input": 5.0, "output": 15.0})
        model_calls = n_calls / len(stage_cfg["models"])
        cost = (
            model_calls * avg_input_tokens * pricing["input"] / 1e6
            + model_calls * avg_output_tokens * pricing["output"] / 1e6
        )
        total_cost += cost

    return n_calls, total_cost


def query_with_retry(client, model, prompt, system_prompt, temperature, seed):
    """Query model with exponential backoff retry."""
    for attempt in range(MAX_RETRIES):
        try:
            return query_model(
                client=client,
                model=model,
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=temperature,
                max_tokens=MAX_TOKENS,
                seed=seed,
            )
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                delay = RETRY_BASE_DELAY_S * (2 ** attempt)
                print(f"\n  Retry {attempt + 1}/{MAX_RETRIES} for {model}: {e}")
                time.sleep(delay)
            else:
                print(f"\n  Failed after {MAX_RETRIES} attempts: {e}")
                return {
                    "model": model,
                    "prompt": prompt,
                    "system_prompt": system_prompt,
                    "temperature": temperature,
                    "seed": seed,
                    "response": None,
                    "finish_reason": "error",
                    "elapsed_s": 0,
                    "usage": {"prompt_tokens": 0, "completion_tokens": 0},
                    "error": str(e),
                }


def run_demo(stage, scoring):
    """Run the TEE demo for the specified stage and scoring method."""
    stage_cfg = STAGES[stage]
    system_prompt = SYSTEM_PROMPTS[scoring]

    print(f"=== TEE Demo: {stage} ({scoring}) ===")
    print(f"Description: {stage_cfg['description']}")

    items, variants = load_items_and_variants(scoring, stage_cfg)
    n_calls, est_cost = estimate_cost(variants, stage_cfg, scoring)

    print(f"\nDesign:")
    print(f"  Scoring method: {scoring}")
    print(f"  Items: {variants['item_id'].nunique()}")
    print(f"  Prompt variants: {variants['variant_id'].nunique()}")
    print(f"  Judge models: {stage_cfg['models']}")
    print(f"  Temperatures: {stage_cfg['temperatures']}")
    print(f"  Replications: {stage_cfg['n_replications']}")
    print(f"  Total API calls: {n_calls}")
    print(f"  Estimated cost: ${est_cost:.2f}")

    # Set up output (with checkpointing for resume support)
    outpath = Path(f"data/raw/{scoring}_{stage}.jsonl")
    outpath.parent.mkdir(parents=True, exist_ok=True)

    # Load already-completed combos from existing JSONL
    combos_done = set()
    if outpath.exists():
        with open(outpath) as f:
            for line in f:
                r = json.loads(line)
                combos_done.add((
                    r["item_id"], str(r["variant_id"]),
                    r["model"], str(r["temperature"]),
                    str(r["replication"]),
                ))
        print(f"  Resuming: {len(combos_done)} calls already completed")

    client = get_client()
    random.seed(SEED)
    total_input_tokens = 0
    total_output_tokens = 0
    results_buffer = []

    # Build full factorial combinations
    combos = []
    for _, variant_row in variants.iterrows():
        for model in stage_cfg["models"]:
            for temp in stage_cfg["temperatures"]:
                for rep in range(stage_cfg["n_replications"]):
                    combos.append((variant_row, model, temp, rep))

    # Filter out already-completed combos
    if combos_done:
        combos = [
            (v, m, t, r) for v, m, t, r in combos
            if (v["item_id"], str(v["variant_id"]), m, str(t), str(r))
            not in combos_done
        ]
        print(f"  Remaining calls: {len(combos)}")

    for variant_row, model, temp, rep in tqdm(combos, desc=f"Running {stage} ({scoring})"):
        rep_seed = SEED + rep

        result = query_with_retry(
            client=client,
            model=model,
            prompt=variant_row["prompt_text"],
            system_prompt=system_prompt,
            temperature=temp,
            seed=rep_seed if temp == 0.0 else None,
        )

        # Enrich with experiment metadata
        result["item_id"] = variant_row["item_id"]
        result["category"] = variant_row.get("category", "unknown")
        result["variant_id"] = int(variant_row["variant_id"])
        result["replication"] = rep
        result["scoring"] = scoring
        result["stage"] = stage
        result["timestamp"] = datetime.now(tz=timezone.utc).isoformat()

        if result.get("usage"):
            total_input_tokens += result["usage"].get("prompt_tokens", 0)
            total_output_tokens += result["usage"].get("completion_tokens", 0)

        results_buffer.append(result)

        if len(results_buffer) >= 50:
            save_results(results_buffer, outpath)
            results_buffer = []

        time.sleep(REQUEST_DELAY_S)

    # Flush remaining
    if results_buffer:
        save_results(results_buffer, outpath)

    # Compute actual cost
    actual_cost = 0
    for model in stage_cfg["models"]:
        pricing = MODEL_PRICING.get(model, {"input": 5.0, "output": 15.0})
        model_frac = 1.0 / len(stage_cfg["models"])
        actual_cost += (
            total_input_tokens * model_frac * pricing["input"] / 1e6
            + total_output_tokens * model_frac * pricing["output"] / 1e6
        )

    print(f"\n=== Done ===")
    print(f"  Results saved to: {outpath}")
    print(f"  New calls this run: {len(combos)}")
    print(f"  Resumed from checkpoint: {len(combos_done)}")
    print(f"  Total tokens: {total_input_tokens:,} input, {total_output_tokens:,} output")
    print(f"  Actual cost: ~${actual_cost:.2f}")


def main():
    parser = argparse.ArgumentParser(description="Run TEE empirical demo")
    parser.add_argument("--stage", required=True, choices=STAGES.keys(),
                        help="Demo stage: smoke, pilot, or full")
    parser.add_argument("--scoring", required=True, choices=["likert", "pairwise"],
                        help="Scoring method: likert (1-5 rating) or pairwise (A/B)")
    args = parser.parse_args()

    run_demo(args.stage, args.scoring)


if __name__ == "__main__":
    main()
