"""Generate SUT (system-under-test) responses to AILuminate safety prompts.

Each of 4 frontier models responds to each curated safety prompt once.
These responses become the items that judge models evaluate.

Usage:
    python generate_sut_responses.py [--items data/items_safety.csv]
"""

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from config import (
    MAX_RETRIES,
    REQUEST_DELAY_S,
    RETRY_BASE_DELAY_S,
    SEED,
    MODEL_PRICING,
)
from query_openrouter import get_client, query_model, save_results

# OpenRouter model IDs for SUTs
SUT_OPENROUTER_IDS = {
    "claude-opus-4.5": "anthropic/claude-opus-4-5",
    "gpt-5.1": "openai/gpt-5.1",
    "deepseek-chat-v3.1": "deepseek/deepseek-chat-v3.1",
    "grok-4.1-fast": "x-ai/grok-4.1-fast",
}

SUT_MAX_TOKENS = 1024  # SUTs generate substantive responses


def query_with_retry(client, model, prompt, temperature, seed):
    """Query model with exponential backoff retry."""
    for attempt in range(MAX_RETRIES):
        try:
            return query_model(
                client=client,
                model=model,
                prompt=prompt,
                system_prompt="You are a helpful AI assistant.",
                temperature=temperature,
                max_tokens=SUT_MAX_TOKENS,
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
                    "temperature": temperature,
                    "seed": seed,
                    "response": None,
                    "finish_reason": "error",
                    "elapsed_s": 0,
                    "usage": {"prompt_tokens": 0, "completion_tokens": 0},
                    "error": str(e),
                }


def main():
    parser = argparse.ArgumentParser(description="Generate SUT responses to safety prompts")
    parser.add_argument("--items", default="data/items_safety.csv")
    parser.add_argument("--output-jsonl", default="data/raw/sut_responses_safety.jsonl")
    parser.add_argument("--output-csv", default="data/sut_responses_safety.csv")
    parser.add_argument("--temperature", type=float, default=0.7)
    args = parser.parse_args()

    items = pd.read_csv(args.items)
    print(f"Loaded {len(items)} items from {args.items}")

    # Checkpointing: load already-completed items
    jsonl_path = Path(args.output_jsonl)
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    done_ids = set()
    if jsonl_path.exists():
        with open(jsonl_path) as f:
            for line in f:
                r = json.loads(line)
                if r.get("response"):
                    done_ids.add(r["item_id"])
        print(f"  Resuming: {len(done_ids)} responses already collected")

    remaining = items[~items["item_id"].isin(done_ids)]
    print(f"  Remaining: {len(remaining)} calls")

    if len(remaining) == 0:
        print("All responses already collected.")
    else:
        client = get_client()
        results_buffer = []
        total_input_tokens = 0
        total_output_tokens = 0

        for _, row in tqdm(remaining.iterrows(), total=len(remaining), desc="SUT responses"):
            sut_model = row["sut_model"]
            openrouter_id = SUT_OPENROUTER_IDS[sut_model]

            result = query_with_retry(
                client=client,
                model=openrouter_id,
                prompt=row["prompt_text"],
                temperature=args.temperature,
                seed=SEED,
            )

            result["item_id"] = row["item_id"]
            result["category"] = row["category"]
            result["sut_model"] = sut_model
            result["prompt_idx"] = row["prompt_idx"]
            result["timestamp"] = datetime.now(tz=timezone.utc).isoformat()

            if result.get("usage"):
                total_input_tokens += result["usage"].get("prompt_tokens", 0)
                total_output_tokens += result["usage"].get("completion_tokens", 0)

            results_buffer.append(result)

            if len(results_buffer) >= 20:
                save_results(results_buffer, jsonl_path)
                results_buffer = []

            time.sleep(REQUEST_DELAY_S)

        if results_buffer:
            save_results(results_buffer, jsonl_path)

        print(f"\nTokens: {total_input_tokens:,} input, {total_output_tokens:,} output")

    # Build clean CSV from all JSONL results
    rows = []
    with open(jsonl_path) as f:
        for line in f:
            r = json.loads(line)
            rows.append({
                "item_id": r["item_id"],
                "sut_model": r.get("sut_model", ""),
                "sut_response": r.get("response", ""),
                "response_tokens": r.get("usage", {}).get("completion_tokens", 0),
            })

    csv_df = pd.DataFrame(rows)
    csv_df.to_csv(args.output_csv, index=False)
    print(f"\nSaved: {args.output_csv} ({len(csv_df)} responses)")

    # Check for failures
    n_failed = csv_df["sut_response"].isna().sum() + (csv_df["sut_response"] == "").sum()
    if n_failed > 0:
        print(f"  WARNING: {n_failed} failed responses")


if __name__ == "__main__":
    main()
