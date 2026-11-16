"""Finalize variant templates and build the 4 scoring design CSVs (rebuttal Analysis A).

- Regenerates the MMLU NARROW set to 4 DISTINCT paraphrases (fixes the duplicate).
- Loads safety narrow/broad and mmlu broad templates from the pre-screen JSONs.
- Writes design CSVs (item x variant x filled prompt_text) to rebuttal_neurips2026/designs/:
    safety_strict.csv, safety_broadword.csv, mmlu_strict.csv, mmlu_broadword.csv
Nothing scored here.
"""
import json
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
DESIGNS = HERE / "designs"
DESIGNS.mkdir(exist_ok=True)
sys.path.insert(0, str(HERE.parent / "experiments"))
from config import PARAPHRASE_MODEL, SEED  # noqa: E402
from query_openrouter import get_client, query_model  # noqa: E402
from create_mmlu_variants import MMLU_TEMPLATES  # noqa: E402

ROOT = HERE.parent

# ---- 1. Regenerate MMLU narrow to 4 DISTINCT paraphrases ----
NARROW_SYSTEM = (
    "You are a precise paraphrasing assistant for evaluation prompts. Rewrite the given "
    "multiple-choice instruction using different wording, keeping the meaning EXACTLY the "
    "same. You MUST preserve: (1) the task -- answer a 4-option multiple-choice question by "
    "returning ONLY the letter (A, B, C, or D); (2) the placeholders {question}, {A}, {B}, "
    "{C}, {D} verbatim, each option on its own line labeled A/B/C/D. Do not add reasoning "
    "requests, hints, or persona framing. Return ONLY the rewritten template."
)
MMLU_SEED = MMLU_TEMPLATES[0]
PLACE = ("{question}", "{A}", "{B}", "{C}", "{D}")


def mmlu_structural_ok(t):
    return all(p in t for p in PLACE) and ("letter" in t.lower() or "a, b, c" in t.lower())


def regen_mmlu_narrow(client):
    cands, seen = [], set()
    for k in range(12):
        r = query_model(client, PARAPHRASE_MODEL, MMLU_SEED, system_prompt=NARROW_SYSTEM,
                        temperature=0.8, max_tokens=512, seed=SEED + 300 + k)
        t = r["response"].strip()
        key = t.split("\n")[0].strip().lower()
        if key in seen or not mmlu_structural_ok(t):
            continue
        seen.add(key)
        cands.append(t)
        if len(cands) == 4:
            break
    if len(cands) < 4:
        raise RuntimeError(f"only {len(cands)} distinct MMLU narrow paraphrases")
    return [MMLU_SEED] + cands


# ---- 2. Build design CSVs ----
def build_safety(templates, out_name):
    items = pd.read_csv(ROOT / "data/items_safety.csv")
    resp = pd.read_csv(ROOT / "data/sut_responses_safety.csv")
    merged = items.merge(resp[["item_id", "sut_response"]], on="item_id")
    rows = []
    for _, it in merged.iterrows():
        for vid, tmpl in enumerate(templates):
            rows.append({
                "item_id": it["item_id"], "category": it["category"], "variant_id": vid,
                "prompt_text": tmpl.format(prompt_text=it["prompt_text"],
                                           response_text=it["sut_response"]),
            })
    df = pd.DataFrame(rows)
    df.to_csv(DESIGNS / out_name, index=False)
    print(f"  {out_name}: {len(df)} rows ({df.item_id.nunique()} items x {len(templates)} variants)")


def build_mmlu(templates, out_name):
    items = pd.read_csv(ROOT / "data/items_mmlu.csv")
    rows = []
    for _, it in items.iterrows():
        for vid, tmpl in enumerate(templates):
            rows.append({
                "item_id": it["item_id"], "category": it["category"],
                "subcategory": it["subcategory"], "variant_id": f"v_{vid}",
                "prompt_text": tmpl.format(question=it["question"], A=it["choice_A"],
                                           B=it["choice_B"], C=it["choice_C"], D=it["choice_D"]),
                "correct_answer": it["correct_answer"],
            })
    df = pd.DataFrame(rows)
    df.to_csv(DESIGNS / out_name, index=False)
    print(f"  {out_name}: {len(df)} rows ({df.item_id.nunique()} items x {len(templates)} variants)")


def main():
    client = get_client()

    safety_narrow = [v["template"] for v in
                     json.loads((HERE / "safety_strict_variants_candidate.json").read_text())]
    safety_axis = json.loads((HERE / "safety_diversity_axis.json").read_text())
    safety_broad = safety_axis["sets"]["broad_wording"]["templates"]

    mmlu_axis = json.loads((HERE / "mmlu_variant_sets.json").read_text())
    mmlu_broad = mmlu_axis["sets"]["broad_wording"]["templates"]

    print("Regenerating MMLU narrow (4 distinct)...")
    mmlu_narrow = regen_mmlu_narrow(client)
    for i, t in enumerate(mmlu_narrow):
        print(f"  N-V{i}: {t.splitlines()[0]}")
    # persist finalized templates
    (HERE / "final_templates.json").write_text(json.dumps({
        "safety_narrow": safety_narrow, "safety_broad": safety_broad,
        "mmlu_narrow": mmlu_narrow, "mmlu_broad": mmlu_broad,
    }, indent=2))

    print("\nBuilding design CSVs:")
    build_safety(safety_narrow, "safety_strict.csv")
    build_safety(safety_broad, "safety_broadword.csv")
    build_mmlu(mmlu_narrow, "mmlu_strict.csv")
    build_mmlu(mmlu_broad, "mmlu_broadword.csv")
    print("\nDone. Designs in rebuttal_neurips2026/designs/. Nothing scored.")


if __name__ == "__main__":
    main()
