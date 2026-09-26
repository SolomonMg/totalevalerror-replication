"""Generate prompt paraphrases for TEE variance decomposition.

Takes an input CSV of items (item_id, topic, prompt_text) and generates
K-1 semantically equivalent paraphrases per item using a cheap LLM.

Usage:
    python generate_prompts.py --input data/items.csv --k 3 [--output data/processed/prompt_variants.csv]
"""

import argparse
import json
import time
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from config import (
    MAX_RETRIES,
    PARAPHRASE_MODEL,
    REQUEST_DELAY_S,
    RETRY_BASE_DELAY_S,
    SEED,
)
from query_openrouter import get_client, query_model

SYSTEM_PROMPT = (
    "You are a precise paraphrasing assistant. "
    "Rephrase the given evaluation prompt to be semantically identical "
    "but use different wording. Do not change the meaning, task, or "
    "expected response format. Return ONLY the rephrased prompt, "
    "nothing else."
)


def generate_paraphrase(client, original_prompt, variant_num):
    """Generate a single paraphrase of the original prompt."""
    user_msg = (
        f"Rephrase this evaluation prompt (variant {variant_num}):\n\n"
        f"{original_prompt}"
    )
    for attempt in range(MAX_RETRIES):
        try:
            result = query_model(
                client=client,
                model=PARAPHRASE_MODEL,
                prompt=user_msg,
                system_prompt=SYSTEM_PROMPT,
                temperature=0.9,  # Higher temp for diverse paraphrases
                max_tokens=512,
                seed=SEED + variant_num,
            )
            return result["response"].strip()
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                delay = RETRY_BASE_DELAY_S * (2 ** attempt)
                print(f"  Retry {attempt + 1}/{MAX_RETRIES} after error: {e}")
                time.sleep(delay)
            else:
                print(f"  Failed after {MAX_RETRIES} attempts: {e}")
                return None


def main():
    parser = argparse.ArgumentParser(description="Generate prompt paraphrases")
    parser.add_argument("--input", required=True, help="Input CSV (item_id, topic, prompt_text)")
    parser.add_argument("--k", type=int, default=3, help="Total number of variants per item (including original)")
    parser.add_argument("--output", default="data/processed/prompt_variants.csv", help="Output CSV path")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    assert all(col in df.columns for col in ["item_id", "topic", "prompt_text"]), \
        "Input CSV must have columns: item_id, topic, prompt_text"

    client = get_client()
    rows = []

    for _, item in tqdm(df.iterrows(), total=len(df), desc="Generating paraphrases"):
        # Variant 0 = original
        rows.append({
            "item_id": item["item_id"],
            "topic": item["topic"],
            "variant_id": 0,
            "prompt_text": item["prompt_text"],
            "is_original": True,
        })

        # Generate K-1 paraphrases
        for v in range(1, args.k):
            paraphrase = generate_paraphrase(client, item["prompt_text"], v)
            if paraphrase:
                rows.append({
                    "item_id": item["item_id"],
                    "topic": item["topic"],
                    "variant_id": v,
                    "prompt_text": paraphrase,
                    "is_original": False,
                })
            time.sleep(REQUEST_DELAY_S)

    out_df = pd.DataFrame(rows)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)
    print(f"\nSaved {len(out_df)} prompt variants to {out_path}")
    print(f"  Items: {out_df['item_id'].nunique()}, Variants per item: {args.k}")


if __name__ == "__main__":
    main()
