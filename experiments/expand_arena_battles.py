"""Greedy (model x category) stratified sampler for the Arena expansion.

Reads:
  data/processed/arena_battles_categorized.csv  (battle_id -> category_llm)
  data/raw/arena_battles_candidates.parquet     (battle_id -> prompt/models/etc)
  data/processed/arena_battles_scored_input.csv (existing scored battles)

Writes:
  data/processed/arena_battles_scored_input.csv (appends new rows)
  data/processed/arena_expansion_log.csv        (per-cell count summary)

Sampling strategy (user decision, see plan file):
  - Persuasion: include every available Persuasion battle (ceiling cap).
  - Other three categories: greedy coverage-maximizing sampler targeting
    F_other (default 40) battles per (model, category) cell.
    At each iteration, score each candidate battle as
    max(0, F - current[m_a]) + max(0, F - current[m_b]) and pick the
    highest-scoring battle (tiebreak: earliest battle_id for reproducibility).
    Stop when no under-filled cells remain or pool is exhausted.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SEED = 42
TARGET_CATEGORIES = ("creative_writing", "persuasion", "coding", "factual_qa")


def load_full_candidate_pool():
    cat = pd.read_csv(PROJECT_ROOT / "data/processed/arena_battles_categorized.csv")
    cands = pd.read_parquet(PROJECT_ROOT / "data/raw/arena_battles_candidates.parquet")
    df = cands.merge(cat[["battle_id", "category_llm"]], on="battle_id", how="inner")
    df = df[df["category_llm"].isin(TARGET_CATEGORIES)].reset_index(drop=True)
    return df


def greedy_sample(pool_df, current_counts, floor, category):
    """Greedy coverage sampler within a single category pool.

    pool_df must have columns (battle_id, model_a, model_b).
    current_counts is a dict {model: current_count} BEFORE expansion.
    floor is the target count per model within the category.
    Returns list of battle_ids to add.
    """
    # Copy state
    counts = dict(current_counts)
    # Precompute a stable index order (by battle_id) for deterministic tiebreak
    pool = pool_df.sort_values("battle_id").reset_index(drop=True).copy()
    pool["_available"] = True

    # Precompute deficits; update as we go
    def deficit(m):
        return max(0, floor - counts.get(m, 0))

    selected = []
    iteration = 0
    while True:
        iteration += 1
        # Score each available battle. Vectorized.
        active = pool[pool["_available"]]
        if len(active) == 0:
            break
        # Score = deficit(a) + deficit(b)
        # Need current deficits; build a fast lookup
        deficits = {}
        for m in pd.concat([active["model_a"], active["model_b"]]).unique():
            deficits[m] = deficit(m)
        scores = active["model_a"].map(deficits) + active["model_b"].map(deficits)
        max_score = scores.max()
        if max_score == 0:
            # No more deficits; all cells met or unreachable
            break
        # Pick first (lowest battle_id) among the top-scored battles
        winners = active[scores == max_score]
        pick = winners.iloc[0]
        selected.append(pick["battle_id"])
        counts[pick["model_a"]] = counts.get(pick["model_a"], 0) + 1
        counts[pick["model_b"]] = counts.get(pick["model_b"], 0) + 1
        pool.loc[pool["battle_id"] == pick["battle_id"], "_available"] = False

        if iteration % 200 == 0:
            remaining_deficit = sum(max(0, floor - v) for v in counts.values())
            print(f"  [{category}] iter={iteration} selected={len(selected)} "
                  f"sum_deficit={remaining_deficit}")
    return selected, counts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--floor-other", type=int, default=40,
                         help="Target per-(model, category) cell floor for non-Persuasion categories.")
    parser.add_argument("--dry-run", action="store_true",
                         help="Compute sample but don't write out.")
    args = parser.parse_args()

    np.random.seed(SEED)

    full_pool = load_full_candidate_pool()
    existing = pd.read_csv(PROJECT_ROOT / "data/processed/arena_battles_scored_input.csv")
    existing_ids = set(existing["battle_id"].astype(str))

    print(f"Full categorized pool: {len(full_pool):,}")
    print(f"Already in scored input:  {len(existing_ids):,}")
    print()
    print("Category breakdown in full pool:")
    print(full_pool["category_llm"].value_counts().to_string())
    print()

    # Per-category per-model current counts (from existing)
    existing_tagged = full_pool[full_pool["battle_id"].astype(str).isin(existing_ids)].copy()
    all_selected = []  # battle_ids to add

    for cat in TARGET_CATEGORIES:
        cat_pool = full_pool[full_pool["category_llm"] == cat].copy()
        cat_available = cat_pool[~cat_pool["battle_id"].astype(str).isin(existing_ids)].copy()
        # Current per-model counts from existing scored battles in this category
        cat_existing = existing_tagged[existing_tagged["category_llm"] == cat]
        cur_counts = pd.concat([cat_existing["model_a"], cat_existing["model_b"]]).value_counts().to_dict()

        print(f"--- {cat} ---")
        print(f"  Full category pool: {len(cat_pool)}")
        print(f"  Already scored:     {len(cat_existing)}")
        print(f"  Available to add:   {len(cat_available)}")

        if cat == "persuasion":
            # Include every available Persuasion battle (ceiling cap).
            picked = cat_available["battle_id"].astype(str).tolist()
            cur = dict(cur_counts)
            for _, row in cat_available.iterrows():
                cur[row["model_a"]] = cur.get(row["model_a"], 0) + 1
                cur[row["model_b"]] = cur.get(row["model_b"], 0) + 1
            print(f"  Strategy: take all available. Added: {len(picked)}")
            all_selected.extend([(bid, cat) for bid in picked])
            # Log stats
            final_counts = cur
        else:
            print(f"  Strategy: greedy to floor F={args.floor_other}.")
            picked, final_counts = greedy_sample(
                cat_available[["battle_id", "model_a", "model_b"]],
                cur_counts, args.floor_other, cat
            )
            print(f"  Added: {len(picked)}")
            all_selected.extend([(bid, cat) for bid in picked])

        # Log under-filled cells
        models_all = set(list(cur_counts.keys()) + list(final_counts.keys()))
        below_floor = []
        floor = 0 if cat == "persuasion" else args.floor_other
        for m in models_all:
            n = final_counts.get(m, 0)
            if n < floor:
                below_floor.append((m, n))
        if below_floor:
            below_floor.sort(key=lambda x: x[1])
            print(f"  Models below floor {floor}: {len(below_floor)}")
            for m, n in below_floor[:10]:
                print(f"    {m}: {n}")
            if len(below_floor) > 10:
                print(f"    ... and {len(below_floor) - 10} more")
        print()

    # Build the rows to append
    new_ids = set(bid for bid, _ in all_selected)
    new_rows = full_pool[full_pool["battle_id"].astype(str).isin(new_ids)].copy()
    # Match existing schema
    schema_cols = list(existing.columns)
    for c in schema_cols:
        if c not in new_rows.columns:
            new_rows[c] = np.nan
    new_rows = new_rows[schema_cols]
    print(f"Total new battles to score: {len(new_rows):,}")
    print(f"Existing + new:             {len(existing) + len(new_rows):,}")

    # Per-category summary after expansion
    cat_counts_after = pd.concat([
        existing["category_llm"].value_counts().rename("before"),
        new_rows["category_llm"].value_counts().rename("new")
    ], axis=1).fillna(0).astype(int)
    cat_counts_after["after"] = cat_counts_after["before"] + cat_counts_after["new"]
    print("\nPer-category battle counts:")
    print(cat_counts_after.to_string())

    if args.dry_run:
        print("\n--dry-run: NOT writing output")
        return

    # Write expanded scored_input.csv
    merged = pd.concat([existing, new_rows], ignore_index=True)
    out_scored = PROJECT_ROOT / "data/processed/arena_battles_scored_input.csv"
    merged.to_csv(out_scored, index=False)
    print(f"\nWrote {out_scored} ({len(merged):,} rows)")

    # Log
    log_out = PROJECT_ROOT / "data/processed/arena_expansion_log.csv"
    cat_counts_after.reset_index().rename(columns={"index": "category"}).to_csv(log_out, index=False)
    print(f"Wrote {log_out}")


if __name__ == "__main__":
    main()
