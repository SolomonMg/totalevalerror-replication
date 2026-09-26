"""Score Chatbot Arena battles with the CoT Likert and pairwise pipelines.

Two modes:
  --mode likert            Score each response (A and B) on 1-5 quality using all
                           5 ARENA_LIKERT_COT_TEMPLATES across 3 judges.
                           Calls per battle: 2 responses x 3 judges x 5 variants = 30.
  --mode pairwise_forced   Forced A/B using all 5 ARENA_PAIRWISE_FORCED_TEMPLATES,
                           both orderings, 3 judges.
                           Calls per battle: 2 orderings x 3 judges x 5 variants = 30.

Input: data/processed/arena_battles_scored_input.csv
Output: data/raw/arena_{mode}.jsonl

The single-judge baselines are extracted *from* the TEE run (fixing
judge=gpt-oss-120b, variant=0), so no separate runs are needed.
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
    ARENA_LIKERT_COT_TEMPLATES,
    ARENA_PAIRWISE_FORCED_TEMPLATES,
)
from query_openrouter import get_client, query_model, save_results

PROJECT_ROOT = Path(__file__).resolve().parent.parent

JUDGES = [
    "openai/gpt-oss-120b",
    "google/gemini-2.0-flash-001",
    "deepseek/deepseek-chat-v3.1",
]
TEMPERATURE = 1.0  # per plan; real Arena scoring varies outputs
MAX_TOKENS = 2000  # gpt-oss-120b pairwise reasoning + output can burn 800+; OpenRouter
                   # provider routing sometimes caps at 400 silently, so overshoot.
N_WORKERS = 16

STAGE_BATTLES = {
    "smoke": 5,     # ~1 min, < $0.01
    "pilot": 20,    # ~5 min, ~$0.60 per mode
    "full":  None,  # all
}


def build_likert_combos(df):
    combos = []
    for _, row in df.iterrows():
        for resp_side in ("a", "b"):
            resp_text = row[f"response_{resp_side}"]
            for v_id, tmpl in enumerate(ARENA_LIKERT_COT_TEMPLATES):
                prompt = tmpl.format(prompt=row["prompt"], response_text=resp_text)
                for judge in JUDGES:
                    combos.append({
                        "prompt": prompt, "judge": judge,
                        "meta": {
                            "battle_id": row["battle_id"],
                            "category_llm": row.get("category_llm", ""),
                            "winner": row["winner"],
                            "response_side": resp_side,
                            "variant_id": v_id,
                        },
                    })
    return combos


def build_pairwise_combos(df):
    combos = []
    for _, row in df.iterrows():
        # Two orderings: listed = (response_a, response_b) in A/B slots;
        # swapped = (response_b, response_a) in A/B slots.
        for v_id, tmpl in enumerate(ARENA_PAIRWISE_FORCED_TEMPLATES):
            prompt_listed = tmpl.format(
                prompt=row["prompt"],
                response_a=row["response_a"],
                response_b=row["response_b"],
            )
            prompt_swapped = tmpl.format(
                prompt=row["prompt"],
                response_a=row["response_b"],
                response_b=row["response_a"],
            )
            for judge in JUDGES:
                for order, p in (("listed", prompt_listed), ("swapped", prompt_swapped)):
                    combos.append({
                        "prompt": p, "judge": judge,
                        "meta": {
                            "battle_id": row["battle_id"],
                            "category_llm": row.get("category_llm", ""),
                            "winner": row["winner"],
                            "order": order,
                            "variant_id": v_id,
                        },
                    })
    return combos


def query_with_retry(client, model, prompt, temperature, seed):
    for attempt in range(MAX_RETRIES):
        try:
            return query_model(
                client=client, model=model, prompt=prompt,
                system_prompt="", temperature=temperature,
                max_tokens=MAX_TOKENS, seed=seed,
            )
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


def combo_key(meta, judge):
    if "response_side" in meta:
        return (meta["battle_id"], meta["response_side"],
                str(meta["variant_id"]), judge)
    return (meta["battle_id"], meta["order"],
            str(meta["variant_id"]), judge)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True, choices=["likert", "pairwise_forced"])
    parser.add_argument("--stage", default="full", choices=list(STAGE_BATTLES.keys()),
                         help="smoke (5 battles), pilot (20), or full (all)")
    parser.add_argument("--input", default="data/processed/arena_battles_scored_input.csv")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    input_csv = PROJECT_ROOT / args.input
    stage_suffix = "" if args.stage == "full" else f"_{args.stage}"
    out_jsonl = PROJECT_ROOT / f"data/raw/arena_{args.mode}{stage_suffix}.jsonl"

    df = pd.read_csv(input_csv)
    n_cap = STAGE_BATTLES[args.stage]
    if n_cap is not None:
        # Stratified-by-category sample, seeded. Take roughly n_cap/len(categories) per cat.
        cats = df["category_llm"].unique()
        per_cat = max(1, n_cap // len(cats))
        pieces = []
        for c in cats:
            g = df[df["category_llm"] == c]
            pieces.append(g.sample(min(per_cat, len(g)), random_state=42))
        df = pd.concat(pieces, ignore_index=True)
        print(f"Stage: {args.stage} ({args.stage}: {n_cap} battles target -> {len(df)} actual)")

    print(f"Mode: {args.mode} | Input: {input_csv} ({len(df):,} battles)")
    print(f"Output: {out_jsonl}")

    if args.mode == "likert":
        combos = build_likert_combos(df)
    else:
        combos = build_pairwise_combos(df)
    print(f"Calls: {len(combos):,}")

    avg_in = 2500 if args.mode == "pairwise_forced" else 1500
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

    # Resume
    combos_done = set()
    if out_jsonl.exists():
        with open(out_jsonl) as f:
            for line in f:
                r = json.loads(line)
                combos_done.add(combo_key(r["meta"], r["model"]))
        print(f"  Resuming: {len(combos_done):,} done")

    todo = [c for c in combos if combo_key(c["meta"], c["judge"]) not in combos_done]
    print(f"  Remaining: {len(todo):,}")

    client = get_client()
    buffer = []
    total_in = 0
    total_out = 0
    write_lock = threading.Lock()

    def process_one(c):
        result = query_with_retry(
            client=client, model=c["judge"], prompt=c["prompt"],
            temperature=TEMPERATURE, seed=SEED)
        result["meta"] = c["meta"]
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
            if len(buffer) >= 100:
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
