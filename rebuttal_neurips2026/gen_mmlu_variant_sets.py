"""Generate + pre-screen MMLU narrow & broad-wording variant sets (rebuttal Analysis A).

MMLU "variant" = the SUT question-instruction. Seed V0 = MMLU_TEMPLATES[0] (Standard).
  NARROW        : strict paraphrases (conservative wording change, same task)
  BROAD-WORDING : aggressive paraphrases (max wording/structure change, same task)
  BROAD-CONSTRUCT: existing MMLU_TEMPLATES (framing: Standard/Exam/Expert/Minimal/Analytical)

Preserves placeholders {question},{A},{B},{C},{D} and the "answer with only the letter" task.
Uses a STRONG construct-equivalence judge (tries gpt-4o -> opus -> deepseek) with a
discriminating rubric. Embeds instruction text (placeholders stripped) for the cosine
negative-finding. NOTHING scored here. Cost ~$0.10.

Output: rebuttal_neurips2026/mmlu_variant_sets.json
"""
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "experiments"))
from config import PARAPHRASE_MODEL, SEED  # noqa: E402
from query_openrouter import get_client, query_model  # noqa: E402
from create_mmlu_variants import MMLU_TEMPLATES  # noqa: E402

EQUIV_MODELS = ["openai/gpt-4o", "anthropic/claude-opus-4-5", "deepseek/deepseek-chat-v3.1"]
N_BROAD_CANDIDATES = 8
N_SELECT = 4
SEED_TEMPLATE = MMLU_TEMPLATES[0]
PLACEHOLDERS = ("{question}", "{A}", "{B}", "{C}", "{D}")

NARROW_SYSTEM = (
    "You are a precise paraphrasing assistant for evaluation prompts. Rewrite the given "
    "multiple-choice instruction using different wording, keeping the meaning EXACTLY the "
    "same. You MUST preserve: (1) the task -- answer a 4-option multiple-choice question by "
    "returning ONLY the letter (A, B, C, or D); (2) the placeholders {question}, {A}, {B}, "
    "{C}, {D} verbatim, each option on its own line labeled A/B/C/D. Do not add reasoning "
    "requests, hints, or persona framing. Return ONLY the rewritten template."
)
BROAD_SYSTEM = (
    "You are a paraphrasing assistant for evaluation prompts. Rewrite the given multiple-"
    "choice instruction using MAXIMALLY different vocabulary, sentence structure, and "
    "register, so it reads as differently as possible -- BUT the meaning must stay IDENTICAL. "
    "You MUST preserve: (1) the task -- answer a 4-option multiple-choice question by "
    "returning ONLY the letter (A, B, C, or D); (2) the placeholders {question}, {A}, {B}, "
    "{C}, {D} verbatim, each option on its own line labeled A/B/C/D. Do NOT add reasoning "
    "requests, expert/exam persona framing, or hints that change the task. Return ONLY the "
    "rewritten template."
)
EQUIV_SYSTEM = (
    "You decide whether two instructions pose the SAME measurement task. Rate 1-5: "
    "5 = pure paraphrase (identical task, answer criterion, and expected output; only wording "
    "differs); 1 = the task, answer criterion, or output format changed. Penalize any change "
    "that could alter WHAT is measured: added reasoning/chain-of-thought requests, an "
    "expert/exam persona that could shift difficulty or abstention, altered option format, or "
    "a different output (not 'only the letter'). Respond with ONLY a single integer 1-5."
)


def strip_ph(t):
    for p in PLACEHOLDERS:
        t = t.replace(p, " ")
    return t


def structural(t):
    issues = [p for p in PLACEHOLDERS if p not in t]
    if "letter" not in t.lower() and "a, b, c" not in t.lower():
        issues.append("no letter-only instruction")
    return issues


def pick_equiv_model(client):
    for m in EQUIV_MODELS:
        try:
            query_model(client, m, "Reply with 5", system_prompt="Reply only a digit.",
                        temperature=0.0, max_tokens=4, seed=SEED)
            return m
        except Exception:  # noqa: BLE001
            continue
    return None


def equiv(client, model, t):
    p = f"SEED:\n{SEED_TEMPLATE}\n\nCANDIDATE:\n{t}\n\nInteger 1-5 only:"
    try:
        r = query_model(client, model, p, system_prompt=EQUIV_SYSTEM,
                        temperature=0.0, max_tokens=8, seed=SEED)
        d = [c for c in (r["response"] or "") if c.isdigit()]
        return int(d[0]) if d else None
    except Exception:  # noqa: BLE001
        return None


def load_embedder():
    from sentence_transformers import SentenceTransformer
    for name in ("all-mpnet-base-v2", "all-MiniLM-L6-v2"):
        try:
            return SentenceTransformer(name), name
        except Exception:  # noqa: BLE001
            continue
    raise RuntimeError("no embedder")


def gen(client, system, temp, base):
    r = query_model(client, PARAPHRASE_MODEL, SEED_TEMPLATE, system_prompt=system,
                    temperature=temp, max_tokens=512, seed=base)
    return r["response"].strip()


def main():
    client = get_client()
    model, emb_name = load_embedder()
    eqm = pick_equiv_model(client)
    print(f"Embedder: {emb_name} | equivalence judge: {eqm}\n")

    def emb(texts):
        return model.encode([strip_ph(t) for t in texts], normalize_embeddings=True)

    def mean_pair_cos(texts):
        e = emb(texts)
        s = [float(np.dot(e[i], e[j])) for i, j in combinations(range(len(e)), 2)]
        return round(float(np.mean(s)), 4), round(float(min(s)), 4)

    # NARROW
    narrow = [SEED_TEMPLATE] + [gen(client, NARROW_SYSTEM, 0.9, SEED + k) for k in range(1, 5)]
    narrow_eq = [equiv(client, eqm, t) for t in narrow]
    narrow_st = [structural(t) for t in narrow]

    # BROAD-WORDING: generate 8, embed, select 4 most seed-distant that pass equiv>=4 + structural
    cands = [gen(client, BROAD_SYSTEM, 1.0, SEED + 200 + k) for k in range(N_BROAD_CANDIDATES)]
    e_all = model.encode([strip_ph(SEED_TEMPLATE)] + [strip_ph(t) for t in cands],
                         normalize_embeddings=True)
    seed_e = e_all[0]
    scored = []
    for t, ev in zip(cands, e_all[1:]):
        scored.append({"template": t, "cos_to_seed": round(float(np.dot(seed_e, ev)), 4),
                       "equiv": equiv(client, eqm, t), "structural": structural(t)})
    valid = [c for c in scored if (c["equiv"] or 0) >= 4 and not c["structural"]]
    valid.sort(key=lambda c: c["cos_to_seed"])
    selected = valid[:N_SELECT]
    broad = [SEED_TEMPLATE] + [c["template"] for c in selected]

    out = {"embedder": emb_name, "equiv_judge": eqm, "sets": {}}
    for name, texts, eqs, sts in (("narrow", narrow, narrow_eq, narrow_st),):
        mp, mn = mean_pair_cos(texts)
        out["sets"][name] = {"mean_pairwise_cosine": mp, "min_pairwise_cosine": mn,
                             "equiv": eqs, "structural": sts, "templates": texts}
    mp, mn = mean_pair_cos(broad)
    out["sets"]["broad_wording"] = {"mean_pairwise_cosine": mp, "min_pairwise_cosine": mn,
                                    "templates": broad, "selection": scored}
    mp, mn = mean_pair_cos(list(MMLU_TEMPLATES))
    out["sets"]["broad_construct"] = {"mean_pairwise_cosine": mp, "min_pairwise_cosine": mn,
                                      "templates": list(MMLU_TEMPLATES)}
    (HERE / "mmlu_variant_sets.json").write_text(json.dumps(out, indent=2))

    print("=" * 78)
    print("MMLU DIVERSITY AXIS (within-set mean pairwise cosine) + construct-equivalence")
    print("=" * 78)
    for n in ("narrow", "broad_wording", "broad_construct"):
        s = out["sets"][n]
        print(f"  {n:16s} mean_cos={s['mean_pairwise_cosine']:.3f} min_cos={s['min_pairwise_cosine']:.3f}")
    print(f"\nNARROW equiv scores (should be ~5): {narrow_eq}  structural: {narrow_st}")
    print("\n--- NARROW set ---")
    for i, t in enumerate(narrow):
        print(f"\nN-V{i}: {t}")
    print("\n--- BROAD-WORDING selection (sorted by seed-distance) ---")
    for c in sorted(scored, key=lambda x: x["cos_to_seed"]):
        mk = " <== SELECTED" if c in selected else ""
        print(f"  cos_to_seed={c['cos_to_seed']:.3f} equiv={c['equiv']} struct={c['structural']}{mk}")
    print("\n--- BROAD-WORDING final set ---")
    for i, t in enumerate(broad):
        print(f"\nBW-V{i}: {t}")
    print("\nSaved rebuttal_neurips2026/mmlu_variant_sets.json  (nothing scored)")


if __name__ == "__main__":
    main()
