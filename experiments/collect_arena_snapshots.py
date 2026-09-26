"""
Collect daily Arena AI leaderboard snapshots from api.wulong.dev.

Pulls text, code, and vision leaderboards for all available dates,
saves to data/processed/arena_daily_elo.csv.

Usage:
    python experiments/collect_arena_snapshots.py
"""

import json
import csv
import urllib.request
from datetime import date, timedelta
from pathlib import Path

API_BASE = "https://api.wulong.dev/arena-ai-leaderboards/v1/leaderboard"
LEADERBOARDS = ["text", "code", "vision"]
START_DATE = date(2026, 3, 20)
END_DATE = date.today()
OUTPUT = Path("data/processed/arena_daily_elo.csv")

def fetch_leaderboard(name: str, d: date) -> list[dict]:
    url = f"{API_BASE}?name={name}&date={d.isoformat()}"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "TEE-research/1.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())
    except Exception as e:
        print(f"  SKIP {name} {d}: {e}")
        return []

    if "meta" not in data or "models" not in data:
        return []

    rows = []
    for m in data["models"]:
        rows.append({
            "leaderboard": name,
            "date": d.isoformat(),
            "rank": m.get("rank"),
            "model": m.get("model"),
            "vendor": m.get("vendor"),
            "license": m.get("license"),
            "elo": m.get("score"),
            "ci": m.get("ci"),
            "votes": m.get("votes"),
        })
    return rows

def main():
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    all_rows = []
    d = START_DATE
    while d <= END_DATE:
        print(f"Fetching {d.isoformat()}...")
        for lb in LEADERBOARDS:
            rows = fetch_leaderboard(lb, d)
            all_rows.extend(rows)
            print(f"  {lb}: {len(rows)} models")
        d += timedelta(days=1)

    if not all_rows:
        print("No data collected!")
        return

    fieldnames = ["leaderboard", "date", "rank", "model", "vendor", "license",
                  "elo", "ci", "votes"]
    with open(OUTPUT, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    n_dates = len(set(r["date"] for r in all_rows))
    n_models = len(set(r["model"] for r in all_rows))
    print(f"\nSaved {len(all_rows)} rows to {OUTPUT}")
    print(f"  {n_dates} dates, {n_models} unique models, {len(LEADERBOARDS)} leaderboards")

if __name__ == "__main__":
    main()
