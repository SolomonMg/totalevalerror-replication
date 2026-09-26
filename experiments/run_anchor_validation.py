"""Score anchor-persona responses with the matched CoT pipelines.

Three modes:
  --mode likert          Score each (persona, prompt) response on 1-5 Likert.
                         Input: data/processed/anchor_likert_input.csv
                         Output: data/raw/anchor_likert.jsonl
  --mode pairwise_tie    Pairwise with TIE option (Davidson BT downstream).
                         Input: data/processed/anchor_pairwise_input.csv
                         Output: data/raw/anchor_pairwise_tie.jsonl
  --mode pairwise_forced Pairwise forced A/B (standard BT downstream).
                         Output: data/raw/anchor_pairwise_forced.jsonl

Design: 3 judges (gpt-oss-120b, gemini-2.0-flash, deepseek-v3.1), 1 prompt
variant, T=0.7, R=1. Pairwise modes run both orderings per pair.
"""

import argparse
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from config import MAX_RETRIES, MODEL_PRICING, RETRY_BASE_DELAY_S, SEED
from create_prompt_variants import (
    ANCHOR_LIKERT_TEMPLATE,
    ANCHOR_PAIRWISE_TIE_TEMPLATE,
    ANCHOR_PAIRWISE_FORCED_TEMPLATE,
)
from query_openrouter import get_client, query_model, save_results

PROJECT_ROOT = Path(__file__).resolve().parent.parent

JUDGES = [
    "openai/gpt-oss-120b",
    "google/gemini-2.0-flash-001",
    "deepseek/deepseek-chat-v3.1",
]
TEMPERATURE = 0.7
MAX_TOKENS = 500
N_WORKERS = 12


def build_likert_combos(df):
    combos = []
    for _, row in df.iterrows():
        prompt = ANCHOR_LIKERT_TEMPLATE.format(response_text=row["response_text"])
        for judge in JUDGES:
            combos.append({
                "prompt": prompt, "judge": judge,
                "meta": {
                    "item_id": row["item_id"],
                    "prompt_id": row["prompt_id"],
                    "dimension": row["dimension"],
                    "anchor_label": row["anchor_label"],
                    "anchor_profile": row["anchor_profile"],
                },
            })
    return combos


def build_pairwise_combos(df, template):
    combos = []
    for _, row in df.iterrows():
        # listed: A=response_a, B=response_b
        p_listed = template.format(response_a=row["response_a"],
                                   response_b=row["response_b"])
        # swapped: A=response_b, B=response_a
        p_swapped = template.format(response_a=row["response_b"],
                                    response_b=row["response_a"])
        for judge in JUDGES:
            for order, prompt in (("listed", p_listed), ("swapped", p_swapped)):
                combos.append({
                    "prompt": prompt, "judge": judge,
                    "meta": {
                        "item_id": row["item_id"],
                        "prompt_id": row["prompt_id"],
                        "dimension": row["dimension"],
                        "anchor_a": row["anchor_a"],
                        "anchor_b": row["anchor_b"],
                        "order": order,
                    },
                })
    return combos


def query_with_retry(client, model, prompt, temperature, seed):
    for attempt in range(MAX_RETRIES):
        try:
            return query_model(client=client, model=model, prompt=prompt,
                               system_prompt="", temperature=temperature,
                               max_tokens=MAX_TOKENS, seed=seed)
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
    parser.add_argument("--mode", required=True,
                        choices=["likert", "pairwise_tie", "pairwise_forced"])
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.mode == "likert":
        input_csv = PROJECT_ROOT / "data/processed/anchor_likert_input.csv"
        out_jsonl = PROJECT_ROOT / "data/raw/anchor_likert.jsonl"
        df = pd.read_csv(input_csv)
        combos = build_likert_combos(df)
    else:
        input_csv = PROJECT_ROOT / "data/processed/anchor_pairwise_input.csv"
        tmpl = (ANCHOR_PAIRWISE_TIE_TEMPLATE if args.mode == "pairwise_tie"
                else ANCHOR_PAIRWISE_FORCED_TEMPLATE)
        out_jsonl = PROJECT_ROOT / f"data/raw/anchor_{args.mode}.jsonl"
        df = pd.read_csv(input_csv)
        combos = build_pairwise_combos(df, tmpl)

    print(f"Mode: {args.mode}")
    print(f"Input: {input_csv} ({len(df)} rows)")
    print(f"Output: {out_jsonl}")
    print(f"Calls: {len(combos):,}")

    # Rough cost estimate
    avg_in = 1200
    avg_out = 150
    est_cost = 0.0
    per_judge = len(combos) // len(JUDGES)
    for j in JUDGES:
        p = MODEL_PRICING.get(j, {"input": 5.0, "output": 15.0})
        est_cost += per_judge * avg_in * p["input"] / 1e6
        est_cost += per_judge * avg_out * p["output"] / 1e6
    print(f"Estimated cost: ~${est_cost:.2f}")

    if args.dry_run:
        print("[dry-run] Exiting.")
        return

    out_jsonl.parent.mkdir(parents=True, exist_ok=True)

    # Resume support
    combos_done = set()
    if out_jsonl.exists():
        with open(out_jsonl) as f:
            for line in f:
                r = json.loads(line)
                key = (r["meta"]["item_id"], r["model"],
                       r["meta"].get("order", "n/a"))
                combos_done.add(key)
        print(f"  Resuming: {len(combos_done)} done")

    todo = [c for c in combos
            if (c["meta"]["item_id"], c["judge"],
                c["meta"].get("order", "n/a")) not in combos_done]
    print(f"  Remaining: {len(todo):,}")

    client = get_client()
    buffer = []
    total_in = 0
    total_out = 0
    write_lock = threading.Lock()

    def process_one(combo):
        result = query_with_retry(
            client=client, model=combo["judge"], prompt=combo["prompt"],
            temperature=TEMPERATURE, seed=SEED)
        result["meta"] = combo["meta"]
        result["mode"] = args.mode
        result["timestamp"] = datetime.now(tz=timezone.utc).isoformat()
        return result

    with ThreadPoolExecutor(max_workers=N_WORKERS) as ex:
        futures = [ex.submit(process_one, c) for c in todo]
        for fut in tqdm(as_completed(futures), total=len(futures), desc=args.mode):
            r = fut.result()
            if r.get("usage"):
                total_in += r["usage"].get("prompt_tokens", 0)
                total_out += r["usage"].get("completion_tokens", 0)
            buffer.append(r)
            if len(buffer) >= 50:
                with write_lock:
                    save_results(buffer, out_jsonl)
                    buffer = []
    if buffer:
        save_results(buffer, out_jsonl)

    actual = 0.0
    for j in JUDGES:
        p = MODEL_PRICING.get(j, {"input": 5.0, "output": 15.0})
        frac = 1.0 / len(JUDGES)
        actual += total_in * frac * p["input"] / 1e6
        actual += total_out * frac * p["output"] / 1e6
    print(f"\n=== Done ===")
    print(f"  New calls: {len(todo):,}  |  Tokens: {total_in:,}/{total_out:,}")
    print(f"  Actual cost: ~${actual:.2f}")


if __name__ == "__main__":
    main()
