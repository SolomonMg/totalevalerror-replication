"""Run the MMLU multiple-choice experiment for TLE variance decomposition.

Queries SUT models on MMLU items across prompt variants, temperatures, and
replications. Each response is scored inline by parsing the first A/B/C/D
letter and comparing to the correct answer.

Uses concurrent requests (ThreadPoolExecutor) for throughput.

Design: 200 items x 5 variants x 3 temps x 3 models x 8 reps = 72,000 calls (full).

Usage:
    python run_mmlu.py --stage smoke
    python run_mmlu.py --stage pilot
    python run_mmlu.py --stage full
    python run_mmlu.py --stage full --workers 50
"""

import argparse
import json
import random
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from config import (
    MAX_RETRIES,
    MODEL_PRICING,
    RETRY_BASE_DELAY_S,
    SEED,
    TEMPERATURES,
)
from query_openrouter import get_client, query_model

# --- MMLU SUT models ---
MMLU_MODELS = {
    "gemini-flash": "google/gemini-2.0-flash-001",
    "deepseek-v3.1": "deepseek/deepseek-chat-v3.1",
    "gpt-4o": "openai/gpt-4o",
}

# --- Stage configurations ---
MMLU_STAGES = {
    "smoke": {
        "n_items": 20,
        "models": ["google/gemini-2.0-flash-001"],
        "temperatures": [0.7],
        "n_prompt_variants": 1,
        "n_replications": 3,
        "description": "Verify MMLU infrastructure, parsing, scoring",
    },
    "pilot": {
        "n_items": 80,
        "models": list(MMLU_MODELS.values()),
        "temperatures": TEMPERATURES,
        "n_prompt_variants": 3,
        "n_replications": 5,
        "description": "MMLU variance decomposition feasibility check",
    },
    "full": {
        "n_items": 200,
        "models": list(MMLU_MODELS.values()),
        "temperatures": TEMPERATURES,
        "n_prompt_variants": 5,
        "n_replications": 8,
        "description": "Full MMLU variance decomposition for publication",
    },
}

SYSTEM_PROMPT = "You are a helpful assistant."
MAX_TOKENS_MMLU = 8  # Only need a single letter

# Thread-safe writer
_write_lock = threading.Lock()

# Strict answer parser (never guesses from prose); the analysis re-parses with
# analysis/lib_parse_mmlu.R, which mirrors it.
from mmlu_parse import parse_answer  # noqa: E402


def load_variants(stage_cfg):
    """Load MMLU items and prompt variants from the design matrix CSV.

    Returns a DataFrame filtered to the stage's n_items and n_prompt_variants.
    """
    variants_path = Path("data/processed/variants_mmlu.csv")
    if not variants_path.exists():
        raise FileNotFoundError(
            f"{variants_path} not found. Run create_mmlu_variants.py first:\n"
            "  python curate_mmlu_items.py\n"
            "  python create_mmlu_variants.py"
        )

    variants = pd.read_csv(variants_path)

    # Filter to requested number of prompt variants
    # variant_id is like "v_0", "v_1", ...; sort to get the first N
    all_variant_ids = sorted(variants["variant_id"].unique())
    keep_variants = set(all_variant_ids[: stage_cfg["n_prompt_variants"]])
    variants = variants[variants["variant_id"].isin(keep_variants)]

    # Stratified item subsample if needed
    n_items = stage_cfg["n_items"]
    unique_items = variants["item_id"].unique()
    if n_items < len(unique_items):
        # Stratified sample by category
        items_meta = variants[["item_id", "category"]].drop_duplicates()
        n_per_cat = max(1, n_items // items_meta["category"].nunique())
        sampled_ids = []
        for _, grp in items_meta.groupby("category"):
            sampled_ids.extend(
                grp.sample(min(len(grp), n_per_cat), random_state=SEED)["item_id"].tolist()
            )
        sampled_ids = sampled_ids[:n_items]
        variants = variants[variants["item_id"].isin(set(sampled_ids))]

    return variants.reset_index(drop=True)


def estimate_cost(variants, stage_cfg):
    """Estimate total API cost for the experiment."""
    n_calls = (
        len(variants)
        * len(stage_cfg["models"])
        * len(stage_cfg["temperatures"])
        * stage_cfg["n_replications"]
    )
    # MMLU prompts are short; responses even shorter (single letter)
    avg_input_tokens = 200
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


def query_with_retry(client, model, prompt, temperature, seed):
    """Query model with exponential backoff retry."""
    for attempt in range(MAX_RETRIES):
        try:
            return query_model(
                client=client,
                model=model,
                prompt=prompt,
                system_prompt=SYSTEM_PROMPT,
                temperature=temperature,
                max_tokens=MAX_TOKENS_MMLU,
                seed=seed,
            )
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                delay = RETRY_BASE_DELAY_S * (2 ** attempt)
                time.sleep(delay)
            else:
                return {
                    "model": model,
                    "prompt": prompt,
                    "system_prompt": SYSTEM_PROMPT,
                    "temperature": temperature,
                    "seed": seed,
                    "response": None,
                    "finish_reason": "error",
                    "elapsed_s": 0,
                    "usage": {"prompt_tokens": 0, "completion_tokens": 0},
                    "error": str(e),
                }


def process_one(combo, client, outpath, stage_name):
    """Process a single MMLU query: call API, score, write to JSONL."""
    variant_row, model, temp, rep = combo
    rep_seed = SEED + rep

    result = query_with_retry(
        client=client,
        model=model,
        prompt=variant_row["prompt_text"],
        temperature=temp,
        seed=rep_seed if temp == 0.0 else None,
    )

    # Scoring: parse answer and compare to correct
    parsed_answer = parse_answer(result.get("response"))
    correct_answer = variant_row["correct_answer"]
    score = int(parsed_answer == correct_answer) if parsed_answer else 0

    result["item_id"] = variant_row["item_id"]
    result["category"] = variant_row["category"]
    result["subcategory"] = variant_row["subcategory"]
    result["variant_id"] = variant_row["variant_id"]
    result["replication"] = rep
    result["correct_answer"] = correct_answer
    result["parsed_answer"] = parsed_answer
    result["score"] = score
    result["scoring"] = "mmlu"
    result["stage"] = stage_name
    result["timestamp"] = datetime.now(tz=timezone.utc).isoformat()

    # Thread-safe write
    with _write_lock:
        with open(outpath, "a") as f:
            f.write(json.dumps(result) + "\n")

    return result


def run_mmlu(stage, n_workers):
    """Run the MMLU experiment with concurrent requests."""
    stage_cfg = MMLU_STAGES[stage]

    print(f"=== TLE MMLU Experiment: {stage} (parallel, {n_workers} workers) ===")
    print(f"Description: {stage_cfg['description']}")

    variants = load_variants(stage_cfg)
    n_calls, est_cost = estimate_cost(variants, stage_cfg)

    print(f"\nDesign:")
    print(f"  Items: {variants['item_id'].nunique()}")
    print(f"  Prompt variants: {variants['variant_id'].nunique()}")
    print(f"  SUT models: {stage_cfg['models']}")
    print(f"  Temperatures: {stage_cfg['temperatures']}")
    print(f"  Replications: {stage_cfg['n_replications']}")
    print(f"  Total API calls: {n_calls}")
    print(f"  Estimated cost: ${est_cost:.2f}")

    # Confirm before proceeding
    confirm = input("\nProceed? [y/N] ").strip().lower()
    if confirm != "y":
        print("Aborted.")
        return

    # Set up output with checkpointing
    outpath = Path(f"data/raw/mmlu_{stage}.jsonl")
    outpath.parent.mkdir(parents=True, exist_ok=True)

    combos_done = set()
    if outpath.exists():
        with open(outpath) as f:
            for line in f:
                r = json.loads(line)
                if r.get("response") is not None:
                    combos_done.add((
                        r["item_id"], str(r["variant_id"]),
                        r["model"], str(r["temperature"]),
                        str(r["replication"]),
                    ))
        print(f"  Resuming: {len(combos_done)} valid calls already completed")

    client = get_client()
    random.seed(SEED)

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

    if len(combos) == 0:
        print("All calls already completed.")
        return

    # Shuffle to spread load across providers
    random.shuffle(combos)

    total_input_tokens = 0
    total_output_tokens = 0
    n_errors = 0
    n_correct = 0
    n_scored = 0

    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = {
            executor.submit(process_one, combo, client, outpath, stage): combo
            for combo in combos
        }

        for future in tqdm(as_completed(futures), total=len(futures), desc=f"MMLU {stage}"):
            result = future.result()
            if result.get("usage"):
                total_input_tokens += result["usage"].get("prompt_tokens", 0)
                total_output_tokens += result["usage"].get("completion_tokens", 0)
            if result.get("error"):
                n_errors += 1
            if result.get("parsed_answer") is not None:
                n_scored += 1
                n_correct += result.get("score", 0)

    # Compute actual cost
    actual_cost = 0
    for model in stage_cfg["models"]:
        pricing = MODEL_PRICING.get(model, {"input": 5.0, "output": 15.0})
        model_frac = 1.0 / len(stage_cfg["models"])
        actual_cost += (
            total_input_tokens * model_frac * pricing["input"] / 1e6
            + total_output_tokens * model_frac * pricing["output"] / 1e6
        )

    accuracy = n_correct / n_scored if n_scored > 0 else 0

    print(f"\n=== Done ===")
    print(f"  Results saved to: {outpath}")
    print(f"  New calls this run: {len(combos)}")
    print(f"  Resumed from checkpoint: {len(combos_done)}")
    print(f"  Errors: {n_errors}")
    print(f"  Accuracy: {n_correct}/{n_scored} ({accuracy:.1%})")
    print(f"  Total tokens: {total_input_tokens:,} input, {total_output_tokens:,} output")
    print(f"  Actual cost: ~${actual_cost:.2f}")


def main():
    parser = argparse.ArgumentParser(description="Run TLE MMLU experiment")
    parser.add_argument("--stage", required=True, choices=MMLU_STAGES.keys(),
                        help="Stage: smoke, pilot, or full")
    parser.add_argument("--workers", type=int, default=30,
                        help="Number of concurrent workers (default: 30)")
    args = parser.parse_args()

    run_mmlu(args.stage, args.workers)


if __name__ == "__main__":
    main()
