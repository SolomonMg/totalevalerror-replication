"""Fetch Chatbot Arena human-preference battles for the scoring-pipeline demo.

Downloads `lmarena-ai/arena-human-preference-100k`, filters to single-turn English
battles with non-empty responses and a valid winner, samples a candidate pool
with stratification by the dataset's built-in category_tag features, and writes
a Parquet file that classify_arena_tasks.py consumes.

Output schema:
  battle_id, prompt, response_a, response_b, model_a, model_b, winner,
  is_code, is_refusal, is_creative, is_math, has_domain_knowledge, has_if

Usage:
  python experiments/fetch_arena_battles.py [--n-candidates 3000]
"""

import argparse
import random
from pathlib import Path

import pandas as pd
from datasets import load_dataset
from tqdm import tqdm

from config import SEED

PROJECT_ROOT = Path(__file__).resolve().parent.parent

SEED_RANDOM = random.Random(SEED)


def extract_single_turn_text(conversation):
    """Arena records conversation as a list of {role, content} dicts. Return user
    turn and assistant response for single-turn battles; None if not exactly one user+assistant turn."""
    if not isinstance(conversation, list) or len(conversation) < 2:
        return None, None
    if conversation[0].get("role") != "user":
        return None, None
    if conversation[1].get("role") != "assistant":
        return None, None
    return conversation[0].get("content", ""), conversation[1].get("content", "")


def extract_category_flags(category_tag, is_code):
    """Extract per-battle category flags from the dataset's existing tagging."""
    flags = {
        "is_code": bool(is_code),
        "is_creative": False,
        "is_math": False,
        "has_domain_knowledge": False,
        "has_if": False,
    }
    if isinstance(category_tag, dict):
        crit = category_tag.get("criteria_v0.1", {})
        if isinstance(crit, dict):
            flags["is_creative"] = bool(crit.get("creativity", False))
            flags["has_domain_knowledge"] = bool(crit.get("domain_knowledge", False))
        math_v = category_tag.get("math_v0.1", {})
        if isinstance(math_v, dict):
            flags["is_math"] = bool(math_v.get("math", False))
        if_v = category_tag.get("if_v0.1", {})
        if isinstance(if_v, dict):
            flags["has_if"] = bool(if_v.get("if", False))
    return flags


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-candidates", type=int, default=3000,
                         help="Number of candidate battles to keep (before LLM categorization)")
    parser.add_argument("--out", default="data/raw/arena_battles_candidates.parquet")
    parser.add_argument("--max-prompt-chars", type=int, default=4000,
                         help="Skip battles with prompts longer than this (saves scoring cost)")
    parser.add_argument("--min-response-chars", type=int, default=40,
                         help="Skip battles with extremely short responses (likely degenerate)")
    parser.add_argument("--max-response-chars", type=int, default=6000,
                         help="Skip battles with responses longer than this (prompt-length pressure)")
    args = parser.parse_args()

    outpath = PROJECT_ROOT / args.out
    outpath.parent.mkdir(parents=True, exist_ok=True)

    print(f"Streaming arena-human-preference-100k ...")
    ds = load_dataset("lmarena-ai/arena-human-preference-100k",
                       split="train", streaming=True)

    kept = []
    seen = 0
    for rec in tqdm(ds, desc="scanning"):
        seen += 1
        if rec.get("language") != "English":
            continue
        if rec.get("turn") != 1:
            continue
        if rec.get("winner") not in ("model_a", "model_b", "tie", "tie (bothbad)"):
            continue
        if rec.get("is_refusal"):
            continue

        prompt_a, resp_a = extract_single_turn_text(rec.get("conversation_a", []))
        prompt_b, resp_b = extract_single_turn_text(rec.get("conversation_b", []))
        if prompt_a is None or prompt_b is None:
            continue
        # Both sides should share the same prompt in Arena; cheap sanity check.
        prompt = prompt_a

        if not prompt or not resp_a or not resp_b:
            continue
        if len(prompt) > args.max_prompt_chars:
            continue
        if len(resp_a) < args.min_response_chars or len(resp_b) < args.min_response_chars:
            continue
        if len(resp_a) > args.max_response_chars or len(resp_b) > args.max_response_chars:
            continue

        flags = extract_category_flags(rec.get("category_tag"), rec.get("is_code"))

        kept.append({
            "battle_id": rec["question_id"],
            "prompt": prompt,
            "response_a": resp_a,
            "response_b": resp_b,
            "model_a": rec["model_a"],
            "model_b": rec["model_b"],
            "winner": rec["winner"],
            **flags,
        })

        # Stop once we have 2x the candidate target (we'll sub-sample). Avoids
        # streaming the whole 100k unnecessarily.
        if len(kept) >= args.n_candidates * 2:
            break

    print(f"Scanned {seen:,} records; kept {len(kept):,} passing filters.")

    df = pd.DataFrame(kept)

    # Stratified sample to get balanced category flags in the candidate pool.
    # We want candidates that over-represent our 4 target categories so that
    # LLM classification is efficient. Stratify by (is_code, is_creative).
    df["_stratum"] = (df["is_code"].astype(int) * 2 + df["is_creative"].astype(int))
    # Within each stratum, sample proportionally up to n_candidates total.
    n_total = min(args.n_candidates, len(df))
    per_stratum = max(50, n_total // 4)
    pieces = []
    for s, g in df.groupby("_stratum", group_keys=False):
        pieces.append(g.sample(min(per_stratum, len(g)), random_state=SEED))
    sampled = pd.concat(pieces, ignore_index=True)
    if "_stratum" in sampled.columns:
        sampled = sampled.drop(columns="_stratum")

    # Shuffle and truncate
    sampled = sampled.sample(frac=1, random_state=SEED).head(n_total).reset_index(drop=True)

    print(f"\nCandidate pool: {len(sampled):,}")
    print("Winner distribution:", sampled["winner"].value_counts().to_dict())
    print("Category flags (from dataset):")
    for col in ["is_code", "is_creative", "is_math", "has_domain_knowledge", "has_if"]:
        print(f"  {col}: {sampled[col].sum():,}")

    sampled.to_parquet(outpath, index=False)
    print(f"\nWrote {outpath} ({len(sampled):,} rows, {outpath.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
