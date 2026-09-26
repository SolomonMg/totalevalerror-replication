"""Classify each Arena battle prompt into one of 4 target task categories.

Target categories:
  creative_writing, persuasion, coding, factual_qa, other

Uses openai/gpt-oss-120b via OpenRouter: cheap, fast, JSON output.
Inputs: data/raw/arena_battles_candidates.parquet
Outputs:
  - data/processed/arena_battles_categorized.csv  (all candidates with tags)
  - data/processed/arena_battles_scored_input.csv (N_PER_CAT per target category)

Usage:
  python experiments/classify_arena_tasks.py [--n-per-cat 375]
"""

import argparse
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from config import MAX_RETRIES, RETRY_BASE_DELAY_S, SEED
from query_openrouter import get_client, query_model

PROJECT_ROOT = Path(__file__).resolve().parent.parent

CLASSIFIER_MODEL = "openai/gpt-oss-120b"
N_WORKERS = 32

CLASSIFY_TEMPLATE = (
    "Classify the task category of a user prompt. Read the prompt and pick ONE category.\n\n"
    "Categories:\n"
    "  creative_writing: fiction, poetry, storytelling, dialogue for entertainment, world-building\n"
    "  persuasion: essays arguing a position, marketing/advertising copy, rhetorical speeches, convincing text, debates, opinion pieces\n"
    "  coding: writing, debugging, explaining, analyzing code; programming questions; technical software tasks\n"
    "  factual_qa: factual information requests, knowledge lookups, definitions, explanations of concepts or events\n"
    "  other: anything else (math, role-play, casual chat, translation, summarization of neutral text, etc.)\n\n"
    "Respond with ONLY a single-line JSON object. No prose, no markdown, no reasoning.\n"
    "Format exactly: {{\"category\": \"<one of the five labels>\"}}\n\n"
    "Prompt:\n{prompt}"
)


def classify_one(client, prompt_text):
    # Truncate prompt for classification — category is usually inferable from
    # the first 1-2 sentences. Speeds up tokenization + reduces classifier cost.
    user_prompt = CLASSIFY_TEMPLATE.format(prompt=prompt_text[:1200])
    for attempt in range(MAX_RETRIES):
        try:
            r = query_model(
                client=client,
                model=CLASSIFIER_MODEL,
                prompt=user_prompt,
                system_prompt="",
                temperature=0.0,
                max_tokens=200,   # gpt-oss-120b is verbose; 200 absorbs its CoT padding
                seed=SEED,
            )
            resp = r.get("response") or ""
            try:
                j = json.loads(resp)
                cat = str(j.get("category", "")).strip().lower()
            except Exception:
                # Fallback: substring scan
                u = resp.lower()
                for c in ("creative_writing", "persuasion", "coding", "factual_qa", "other"):
                    if c in u:
                        cat = c
                        break
                else:
                    cat = "other"
            if cat not in ("creative_writing", "persuasion", "coding", "factual_qa", "other"):
                cat = "other"
            return cat, resp, r.get("usage", {})
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_BASE_DELAY_S * (2 ** attempt))
            else:
                return "other", f"ERROR: {e}", {}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", default="data/raw/arena_battles_candidates.parquet")
    parser.add_argument("--out-all", default="data/processed/arena_battles_categorized.csv")
    parser.add_argument("--out-final", default="data/processed/arena_battles_scored_input.csv")
    parser.add_argument("--n-per-cat", type=int, default=375,
                         help="Battles per target category in final output")
    args = parser.parse_args()

    candidates_path = PROJECT_ROOT / args.candidates
    out_all = PROJECT_ROOT / args.out_all
    out_final = PROJECT_ROOT / args.out_final
    out_all.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(candidates_path)
    print(f"Loaded {len(df):,} candidates from {candidates_path}")

    # Resume support: if out_all already exists, only classify missing rows
    done_ids = set()
    if out_all.exists():
        prev = pd.read_csv(out_all)
        done_ids = set(prev["battle_id"].astype(str))
        print(f"  Resuming from {len(done_ids):,} already-classified battles")

    todo = df[~df["battle_id"].astype(str).isin(done_ids)].copy()
    print(f"  {len(todo):,} battles to classify")

    client = get_client()
    results = []
    lock = threading.Lock()

    def work(row):
        cat, raw, usage = classify_one(client, row["prompt"])
        return {"battle_id": row["battle_id"], "category_llm": cat,
                "category_raw": raw[:200],
                "tokens_in": usage.get("prompt_tokens", 0),
                "tokens_out": usage.get("completion_tokens", 0)}

    if len(todo) > 0:
        total_in = 0
        total_out = 0
        with ThreadPoolExecutor(max_workers=N_WORKERS) as ex:
            futures = [ex.submit(work, r) for _, r in todo.iterrows()]
            for fut in tqdm(as_completed(futures), total=len(futures), desc="classify"):
                r = fut.result()
                results.append(r)
                total_in += r["tokens_in"]
                total_out += r["tokens_out"]
        cost = total_in * 0.039 / 1e6 + total_out * 0.19 / 1e6
        print(f"\n  Classification done. Tokens: {total_in:,} in / {total_out:,} out. Cost ~${cost:.3f}")

    # Combine with existing
    new_df = pd.DataFrame(results)
    if out_all.exists() and len(done_ids) > 0:
        prev = pd.read_csv(out_all)
        all_tagged = pd.concat([prev, new_df], ignore_index=True)
    else:
        all_tagged = new_df
    all_tagged = all_tagged.drop_duplicates(subset="battle_id", keep="last")

    # Join with candidates
    merged = df.merge(all_tagged[["battle_id", "category_llm"]],
                      on="battle_id", how="left")

    print("\nCategory distribution:")
    print(merged["category_llm"].value_counts())

    # Save full tagged set
    all_tagged.to_csv(out_all, index=False)
    print(f"\nWrote {out_all} ({len(all_tagged):,} rows)")

    # Sample N_PER_CAT per target category
    targets = ["creative_writing", "persuasion", "coding", "factual_qa"]
    final_pieces = []
    for cat in targets:
        subset = merged[merged["category_llm"] == cat]
        n_take = min(args.n_per_cat, len(subset))
        if n_take < args.n_per_cat:
            print(f"  WARNING: only {n_take} battles for {cat} (requested {args.n_per_cat})")
        final_pieces.append(subset.sample(n_take, random_state=SEED))
    final_df = pd.concat(final_pieces, ignore_index=True)

    # Balance check: make sure winner distribution isn't wildly skewed
    print("\nFinal sample winner distribution by category:")
    print(pd.crosstab(final_df["category_llm"], final_df["winner"]))

    final_df.to_csv(out_final, index=False)
    print(f"\nWrote {out_final} ({len(final_df):,} rows)")


if __name__ == "__main__":
    main()
