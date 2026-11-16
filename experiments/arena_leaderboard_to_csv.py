"""Convert Arena's leaderboard history (Parquet) to CSV for R, keeping the 'overall' category.

Source: https://huggingface.co/datasets/lmarena-ai/leaderboard-dataset (CC-BY-4.0), downloaded to
data/raw/external/arena_leaderboard_dataset/{text,text_style_control}/full-00000-of-00001.parquet (MANIFEST.txt).
Output: data/processed/arena_lb_overall_{text,text_style_control}.csv (one row per model x publish date).
Run: python3 experiments/arena_leaderboard_to_csv.py
"""
from pathlib import Path

import pyarrow.compute as pc
import pyarrow.csv as pcsv
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw/external/arena_leaderboard_dataset"
OUT = ROOT / "data/processed"

for cfg in ("text", "text_style_control"):
    t = pq.read_table(RAW / cfg / "full-00000-of-00001.parquet")
    t = t.filter(pc.equal(t["category"], "overall"))
    pcsv.write_csv(t, OUT / f"arena_lb_overall_{cfg}.csv")
    print(f"{cfg}: {t.num_rows} overall rows -> data/processed/arena_lb_overall_{cfg}.csv")
