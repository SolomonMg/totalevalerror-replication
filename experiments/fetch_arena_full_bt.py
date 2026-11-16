"""Fit a 'gold' Bradley-Terry leaderboard from every single-turn English
battle in lmarena-ai/arena-human-preference-100k.

Serves as a high-sample-size human baseline. The per-config BT leaderboards
in data/processed/arena_bt_rankings.csv were fit on a 1,500-battle subset;
this script fits on the full ~80k eligible battles so we can ask whether
pipeline BT tracks the underlying human ranking rather than just a 1,500-battle
reconstruction of it.

Filters match experiments/fetch_arena_battles.py: English, turn == 1,
winner in {model_a, model_b, tie, tie (bothbad)}, not is_refusal. No
length filters — those were applied only to make scoring tractable.

Output: data/processed/arena_full_bt.csv with columns
  model, bt_strength_full, n_battles_full
"""

from pathlib import Path

import numpy as np
import pandas as pd
from datasets import load_dataset
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def fit_bt_mm(pair_df, n_iter=5000, tol=1e-8):
    """MM algorithm (Hunter 2004). Returns dict model -> pi, normalized mean=1."""
    players = sorted(set(pair_df["p1"]) | set(pair_df["p2"]))
    K = len(players)
    idx = {p: i for i, p in enumerate(players)}
    W = np.zeros((K, K))
    for row in pair_df.itertuples(index=False):
        W[idx[row.p1], idx[row.p2]] += row.wins1
        W[idx[row.p2], idx[row.p1]] += row.wins2
    N = W + W.T
    w = W.sum(axis=1)
    pi = np.ones(K)
    for _ in range(n_iter):
        pi_new = np.zeros(K)
        for i in range(K):
            denom = 0.0
            for j in range(K):
                if i == j or N[i, j] == 0:
                    continue
                denom += N[i, j] / (pi[i] + pi[j])
            pi_new[i] = w[i] / denom if denom > 0 else pi[i]
        pi_new = pi_new / pi_new.mean()
        if np.max(np.abs(pi_new - pi)) < tol:
            break
        pi = pi_new
    return dict(zip(players, pi))


def main():
    ds = load_dataset(
        "lmarena-ai/arena-human-preference-100k",
        split="train",
        streaming=True,
    )
    records = []
    for rec in tqdm(ds, desc="scanning"):
        if rec.get("language") != "English":
            continue
        if rec.get("turn") != 1:
            continue
        w = rec.get("winner")
        if w not in ("model_a", "model_b", "tie", "tie (bothbad)"):
            continue
        if rec.get("is_refusal"):
            continue
        records.append(
            {"model_a": rec["model_a"], "model_b": rec["model_b"], "winner": w}
        )

    df = pd.DataFrame(records)
    print(f"\nEligible battles: {len(df):,}")
    print(f"Unique models:    {df[['model_a', 'model_b']].stack().nunique()}")

    # Canonical pair order (alphabetical) so wins aggregate correctly.
    df["p1"] = np.minimum(df["model_a"], df["model_b"])
    df["p2"] = np.maximum(df["model_a"], df["model_b"])
    w_map = {"model_a": 1.0, "model_b": 0.0, "tie": 0.5, "tie (bothbad)": 0.5}
    df["model_a_wins"] = df["winner"].map(w_map)
    df["p1_wins"] = np.where(
        df["model_a"] == df["p1"], df["model_a_wins"], 1 - df["model_a_wins"]
    )
    pairs = (
        df.groupby(["p1", "p2"])
        .agg(wins1=("p1_wins", "sum"), n=("p1_wins", "count"))
        .reset_index()
    )
    pairs["wins2"] = pairs["n"] - pairs["wins1"]

    print(f"Unique pairs:     {len(pairs):,}")
    pi = fit_bt_mm(pairs[["p1", "p2", "wins1", "wins2"]])

    # Per-model battle counts for the output.
    counts = pd.concat([df["model_a"], df["model_b"]]).value_counts().to_dict()

    out = pd.DataFrame(
        {
            "model": list(pi.keys()),
            "bt_strength_full": list(pi.values()),
            "n_battles_full": [int(counts.get(m, 0)) for m in pi.keys()],
        }
    ).sort_values("bt_strength_full", ascending=False, ignore_index=True)

    outpath = PROJECT_ROOT / "data/processed/arena_full_bt.csv"
    out.to_csv(outpath, index=False)
    print(f"\nWrote {outpath} ({len(out)} models)")
    print(out.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
