"""Deterministic admissibility check for every MMLU prompt-variant set.

Applies structural() verbatim from rebuttal_neurips2026/gen_mmlu_variant_sets.py:66-70:
a variant must keep the input placeholders and a letter-only answer instruction.
The main-text set predates that pipeline and was never checked; this script checks
every set that was collected (SI si:prompt_admissibility, si:mmlu_parsing).

Output: data/processed/mmlu_variant_structural_check.csv (set, variant_id, passes, issues)
Usage:  experiments/.venv/bin/python experiments/check_mmlu_variant_structure.py   (from project root)
"""
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))
from create_mmlu_variants import MMLU_TEMPLATES  # noqa: E402

PLACEHOLDERS = ("{question}", "{A}", "{B}", "{C}", "{D}")


def structural(t):
    issues = [p for p in PLACEHOLDERS if p not in t]
    if "letter" not in t.lower() and "a, b, c" not in t.lower():
        issues.append("no letter-only instruction")
    return issues


R = ROOT / "rebuttal_neurips2026"
fin = json.loads((R / "final_templates.json").read_text())
sets = {
    "main_text": MMLU_TEMPLATES,
    "narrow": fin["mmlu_narrow"],
    "broad_wording": fin["mmlu_broad"],
    "broadplus": json.loads((R / "mmlu_broadplus_templates.json").read_text()),
}
rows = [dict(set=s, variant_id=f"v_{i}", passes=not structural(t), issues="; ".join(structural(t)))
        for s, ts in sets.items() for i, t in enumerate(ts)]
out = ROOT / "data/processed/mmlu_variant_structural_check.csv"
with out.open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
print(f"{len(rows)} variants checked; failures: {[r for r in rows if not r['passes']]}")
