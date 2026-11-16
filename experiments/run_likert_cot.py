"""Run the Likert ideology experiment with the CoT + JSON prompt.

Matches the pairwise CoT protocol (same judges, 3 reps, 3 temps, 5 variants)
so the scoring-method comparison isolates rating-format differences.

Design:
  - 150 items x 5 CoT prompt variants x 3 temperatures x 3 replications
  - 3 judges: openai/gpt-oss-120b, google/gemini-2.0-flash-001,
              deepseek/deepseek-chat-v3.1
  - JSON output: {"reasoning": "...", "score": <1-5>}
  - 20,250 total calls, ~$3 via OpenRouter

Usage:
    python experiments/run_likert_cot.py [--dry-run]
"""

import argparse
import json
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from config import MAX_RETRIES, MODEL_PRICING, RETRY_BASE_DELAY_S, SEED
from create_prompt_variants import LIKERT_COT_TEMPLATES
from query_openrouter import get_client, query_model, save_results

PROJECT_ROOT = Path(__file__).resolve().parent.parent

JUDGES = [
    "openai/gpt-oss-120b",
    "google/gemini-2.0-flash-001",
    "deepseek/deepseek-chat-v3.1",
]
TEMPERATURES = [0.0, 0.7, 1.0]
N_REPLICATIONS = 3
N_PROMPT_VARIANTS = len(LIKERT_COT_TEMPLATES)  # 5
MAX_TOKENS_COT = 500  # reasoning sentence + JSON wrapper
N_WORKERS = 12


def build_variants(items_df):
    rows = []
    for _, item in items_df.iterrows():
        for v_id, tmpl in enumerate(LIKERT_COT_TEMPLATES):
            prompt_text = tmpl.format(response_text=item["response_text"])
            rows.append({
                "item_id": item["item_id"],
                "category": item["category"],
                "variant_id": v_id,
                "prompt_text": prompt_text,
            })
    return pd.DataFrame(rows)


def estimate_cost(n_items):
    n_calls = n_items * N_PROMPT_VARIANTS * len(TEMPERATURES) * N_REPLICATIONS * len(JUDGES)
    avg_in = 1000  # Likert is shorter (one response, not two)
    avg_out = 100
    total = 0.0
    calls_per_judge = n_calls // len(JUDGES)
    for j in JUDGES:
        p = MODEL_PRICING.get(j, {"input": 5.0, "output": 15.0})
        total += calls_per_judge * avg_in * p["input"] / 1e6
        total += calls_per_judge * avg_out * p["output"] / 1e6
    return n_calls, total


def query_with_retry(client, model, prompt, temperature, seed):
    for attempt in range(MAX_RETRIES):
        try:
            return query_model(client=client, model=model, prompt=prompt,
                               system_prompt="", temperature=temperature,
                               max_tokens=MAX_TOKENS_COT, seed=seed)
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_BASE_DELAY_S * (2 ** attempt))
            else:
                return {
                    "model": model, "prompt": prompt, "system_prompt": "",
                    "temperature": temperature, "seed": seed,
                    "response": None, "finish_reason": "error",
                    "elapsed_s": 0,
                    "usage": {"prompt_tokens": 0, "completion_tokens": 0},
                    "error": str(e),
                }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--items", default="data/items_likert.csv")
    parser.add_argument("--out", default="data/raw/likert_cot.jsonl")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    items_path = PROJECT_ROOT / args.items
    outpath = PROJECT_ROOT / args.out

    items = pd.read_csv(items_path)
    print(f"Loaded {len(items)} items from {items_path}")
    print(f"  Per category: {items['category'].value_counts().to_dict()}")

    variants = build_variants(items)
    n_calls, est_cost = estimate_cost(len(items))

    print(f"\nDesign:")
    print(f"  Items: {len(items)}")
    print(f"  Prompt variants: {N_PROMPT_VARIANTS}")
    print(f"  Temperatures: {TEMPERATURES}")
    print(f"  Replications: {N_REPLICATIONS}")
    print(f"  Judges: {JUDGES}")
    print(f"  Total calls: {n_calls:,}")
    print(f"  Estimated cost: ~${est_cost:.2f}")

    if args.dry_run:
        print("\n[dry-run] Exiting.")
        return

    outpath.parent.mkdir(parents=True, exist_ok=True)

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
    total_in = 0
    total_out = 0
    buffer = []

    combos = []
    for _, var_row in variants.iterrows():
        for judge in JUDGES:
            for temp in TEMPERATURES:
                for rep in range(N_REPLICATIONS):
                    combos.append((var_row, judge, temp, rep))

    if combos_done:
        combos = [
            (v, m, t, r) for v, m, t, r in combos
            if (v["item_id"], str(v["variant_id"]), m, str(t), str(r))
            not in combos_done
        ]
        print(f"  Remaining calls: {len(combos):,}")

    write_lock = threading.Lock()

    def process_one(combo):
        var_row, model, temp, rep = combo
        rep_seed = SEED + rep
        result = query_with_retry(
            client=client, model=model, prompt=var_row["prompt_text"],
            temperature=temp, seed=rep_seed if temp == 0.0 else None,
        )
        result["item_id"]     = var_row["item_id"]
        result["category"]    = var_row.get("category", "unknown")
        result["variant_id"]  = int(var_row["variant_id"])
        result["replication"] = rep
        result["scoring"]     = "likert_cot"
        result["stage"]       = "full"
        result["timestamp"]   = datetime.now(tz=timezone.utc).isoformat()
        return result

    with ThreadPoolExecutor(max_workers=N_WORKERS) as ex:
        futures = [ex.submit(process_one, c) for c in combos]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="likert_cot"):
            result = fut.result()
            if result.get("usage"):
                total_in  += result["usage"].get("prompt_tokens", 0)
                total_out += result["usage"].get("completion_tokens", 0)
            buffer.append(result)
            if len(buffer) >= 50:
                with write_lock:
                    save_results(buffer, outpath)
                    buffer = []

    if buffer:
        save_results(buffer, outpath)

    actual = 0.0
    for j in JUDGES:
        p = MODEL_PRICING.get(j, {"input": 5.0, "output": 15.0})
        frac = 1.0 / len(JUDGES)
        actual += total_in * frac * p["input"] / 1e6
        actual += total_out * frac * p["output"] / 1e6

    print("\n=== Done ===")
    print(f"  Results: {outpath}")
    print(f"  New calls: {len(combos):,}")
    print(f"  Resumed: {len(combos_done):,}")
    print(f"  Tokens: {total_in:,} in, {total_out:,} out")
    print(f"  Actual cost: ~${actual:.2f}")


if __name__ == "__main__":
    main()
