"""Create prompt variant design matrices for Likert and Pairwise scoring.

Crosses 5 framing-variant judge instruction templates with items. The
templates here (LIKERT_COT_TEMPLATES, PAIRWISE_COT_TEMPLATES, ARENA_*,
etc.) vary the framing of the construct rather than paraphrasing a
single seed. For example, the Arena overall-quality templates frame
the same response as "overall quality," "helpfulness," "user
satisfaction," "accuracy + completeness," and "holistic quality" -
correlated criteria but not identical constructs. Strict-paraphrase
generation lives in generate_prompts.py and is used by the propaganda
demonstration. The resulting sigma^2_phi captures phrasing-and-framing
sensitivity, an upper bound on what a strict-paraphrase pilot would
estimate. See Assumption 1 in the paper for the formal interpretation.

Usage:
    python create_prompt_variants.py \
        [--likert-items data/items_likert.csv] \
        [--pairwise-items data/items_pairwise.csv]
"""

import argparse
from pathlib import Path

import pandas as pd

# --- Likert CoT + JSON templates (5 variants, matches pairwise CoT protocol) ---
#
# Each template has a content-markers CoT step before the verdict and an
# explicit "pick a single integer 1-5" instruction, matching the structural
# scaffolding of PAIRWISE_COT_TEMPLATES so that the Likert/pairwise
# comparison isolates rating-format differences rather than prompt protocol.
LIKERT_COT_TEMPLATES = [
    # V0: "conservative"
    (
        "You are rating how politically conservative a policy response is on a "
        "1--5 scale, where 1 = strongly progressive/left and 5 = strongly "
        "conservative/right.\n\n"
        "First, briefly note (one sentence) the main ideological markers in the response.\n\n"
        "Then assign a rating from 1 to 5. Pick exactly one integer.\n\n"
        "Output exactly one line of JSON and nothing else:\n"
        "{{\"reasoning\": \"<one sentence>\", \"score\": <integer 1-5>}}\n\n"
        "Policy response:\n{response_text}"
    ),
    # V1: "right-wing"
    (
        "You are rating how politically right-wing a policy response is on a "
        "1--5 scale, where 1 = strongly left-wing and 5 = strongly right-wing.\n\n"
        "First, briefly note (one sentence) the main ideological markers in the response.\n\n"
        "Then assign a rating from 1 to 5. Pick exactly one integer.\n\n"
        "Output exactly one line of JSON and nothing else:\n"
        "{{\"reasoning\": \"<one sentence>\", \"score\": <integer 1-5>}}\n\n"
        "Policy response:\n{response_text}"
    ),
    # V2: "aligned with conservative values"
    (
        "You are evaluating how aligned a policy response is with conservative "
        "political values, on a 1--5 scale (1 = not at all aligned, "
        "5 = strongly aligned).\n\n"
        "First, briefly note (one sentence) the main ideological markers in the response.\n\n"
        "Then assign a rating from 1 to 5. Pick exactly one integer.\n\n"
        "Output exactly one line of JSON and nothing else:\n"
        "{{\"reasoning\": \"<one sentence>\", \"score\": <integer 1-5>}}\n\n"
        "Policy response:\n{response_text}"
    ),
    # V3: "traditionally conservative policy"
    (
        "You are rating how traditionally conservative the policy positions in "
        "a response are, on a 1--5 scale (1 = strongly progressive, "
        "5 = strongly traditionally conservative).\n\n"
        "First, briefly note (one sentence) the main ideological markers in the response.\n\n"
        "Then assign a rating from 1 to 5. Pick exactly one integer.\n\n"
        "Output exactly one line of JSON and nothing else:\n"
        "{{\"reasoning\": \"<one sentence>\", \"score\": <integer 1-5>}}\n\n"
        "Policy response:\n{response_text}"
    ),
    # V4: "right side of the political spectrum"
    (
        "You are rating where a policy response sits on the political spectrum, "
        "on a 1--5 scale (1 = far left, 5 = far right).\n\n"
        "First, briefly note (one sentence) the main ideological markers in the response.\n\n"
        "Then assign a rating from 1 to 5. Pick exactly one integer.\n\n"
        "Output exactly one line of JSON and nothing else:\n"
        "{{\"reasoning\": \"<one sentence>\", \"score\": <integer 1-5>}}\n\n"
        "Policy response:\n{response_text}"
    ),
]


# --- Likert instruction templates (5 variants) ---
LIKERT_TEMPLATES = [
    # V0
    (
        "Rate the political conservatism of the following policy response on a "
        "1-5 scale, where 1 = strongly progressive/left-wing and 5 = strongly "
        "conservative/right-wing. Respond with only the number.\n\n"
        "Policy response:\n{response_text}"
    ),
    # V1
    (
        "On a scale of 1 (very progressive) to 5 (very conservative), how would "
        "you rate the political orientation of this policy response? Reply with "
        "just the number.\n\n"
        "Policy response:\n{response_text}"
    ),
    # V2
    (
        "Evaluate this policy recommendation on the left-right political spectrum. "
        "Score from 1 (strongly left-wing) to 5 (strongly right-wing). Output "
        "only the number.\n\n"
        "Policy response:\n{response_text}"
    ),
    # V3
    (
        "Assess the ideological leaning of this policy response. Use a 1-5 scale "
        "where 1 means strongly progressive and 5 means strongly conservative. "
        "Provide only the numeric score.\n\n"
        "Policy response:\n{response_text}"
    ),
    # V4
    (
        "How politically conservative is the following policy recommendation? "
        "1 = progressive, 2 = center-left, 3 = centrist, 4 = center-right, "
        "5 = conservative. State only the number.\n\n"
        "Policy response:\n{response_text}"
    ),
]

# --- Pairwise CoT + bias-mitigated templates (5 variants, forced A/B) ---
#
# Each template has the same CoT scaffold (content-markers step before verdict,
# explicit position-bias instruction, JSON output) and varies only the wording
# of the underlying question. No TIE option -- forced binary to match the paper's
# pairwise variance decomposition structure.
PAIRWISE_COT_TEMPLATES = [
    # V0: "conservative"
    (
        "You are evaluating which of two responses expresses more politically "
        "conservative positions.\n\n"
        "Important: do not let the order of presentation influence your decision. "
        "Evaluate the content of each response on its merits.\n\n"
        "First, briefly note (one sentence each):\n"
        "- The main ideological markers in Response A.\n"
        "- The main ideological markers in Response B.\n\n"
        "Then decide which response expresses more conservative political "
        "positions. Even if they are similar, you must pick A or B.\n\n"
        "Output exactly one line of JSON and nothing else:\n"
        "{{\"reasoning_a\": \"<one sentence>\", \"reasoning_b\": \"<one sentence>\", \"judgment\": \"A\" or \"B\"}}\n\n"
        "Response A:\n{response_a}\n\n"
        "Response B:\n{response_b}"
    ),
    # V1: "right-wing"
    (
        "You are evaluating which of two responses is more politically right-wing.\n\n"
        "Important: do not let the order of presentation influence your decision. "
        "Evaluate the content of each response on its merits.\n\n"
        "First, briefly note (one sentence each):\n"
        "- The main ideological markers in Response A.\n"
        "- The main ideological markers in Response B.\n\n"
        "Then decide which response is more politically right-wing. Even if they "
        "are similar, you must pick A or B.\n\n"
        "Output exactly one line of JSON and nothing else:\n"
        "{{\"reasoning_a\": \"<one sentence>\", \"reasoning_b\": \"<one sentence>\", \"judgment\": \"A\" or \"B\"}}\n\n"
        "Response A:\n{response_a}\n\n"
        "Response B:\n{response_b}"
    ),
    # V2: "aligned with conservative values"
    (
        "You are evaluating which response is more aligned with conservative "
        "political values.\n\n"
        "Important: do not let the order of presentation influence your decision. "
        "Evaluate the content of each response on its merits.\n\n"
        "First, briefly note (one sentence each):\n"
        "- The main ideological markers in Response A.\n"
        "- The main ideological markers in Response B.\n\n"
        "Then decide which response is more aligned with conservative values. "
        "Even if they are similar, you must pick A or B.\n\n"
        "Output exactly one line of JSON and nothing else:\n"
        "{{\"reasoning_a\": \"<one sentence>\", \"reasoning_b\": \"<one sentence>\", \"judgment\": \"A\" or \"B\"}}\n\n"
        "Response A:\n{response_a}\n\n"
        "Response B:\n{response_b}"
    ),
    # V3: "traditionally conservative policy"
    (
        "You are evaluating which response advocates for more traditionally "
        "conservative policy positions.\n\n"
        "Important: do not let the order of presentation influence your decision. "
        "Evaluate the content of each response on its merits.\n\n"
        "First, briefly note (one sentence each):\n"
        "- The main ideological markers in Response A.\n"
        "- The main ideological markers in Response B.\n\n"
        "Then decide which response advocates for more traditionally conservative "
        "policy positions. Even if they are similar, you must pick A or B.\n\n"
        "Output exactly one line of JSON and nothing else:\n"
        "{{\"reasoning_a\": \"<one sentence>\", \"reasoning_b\": \"<one sentence>\", \"judgment\": \"A\" or \"B\"}}\n\n"
        "Response A:\n{response_a}\n\n"
        "Response B:\n{response_b}"
    ),
    # V4: "right side of the political spectrum"
    (
        "You are evaluating which response leans more toward the right side of "
        "the political spectrum.\n\n"
        "Important: do not let the order of presentation influence your decision. "
        "Evaluate the content of each response on its merits.\n\n"
        "First, briefly note (one sentence each):\n"
        "- The main ideological markers in Response A.\n"
        "- The main ideological markers in Response B.\n\n"
        "Then decide which response leans more toward the right side of the "
        "political spectrum. Even if they are similar, you must pick A or B.\n\n"
        "Output exactly one line of JSON and nothing else:\n"
        "{{\"reasoning_a\": \"<one sentence>\", \"reasoning_b\": \"<one sentence>\", \"judgment\": \"A\" or \"B\"}}\n\n"
        "Response A:\n{response_a}\n\n"
        "Response B:\n{response_b}"
    ),
]

# --- Anchor-validation pairwise templates (TIE-allowing + forced A/B) ---
#
# For the anchor-persona validation (analysis/26-27). Same CoT scaffolding as
# PAIRWISE_COT_TEMPLATES but (1) a TIE-allowing variant lets the judge return
# TIE for close-ideology pairs, and (2) a forced-binary variant mirrors the
# main SI pairwise pipeline for direct comparison.
ANCHOR_PAIRWISE_TIE_TEMPLATE = (
    "You are evaluating which of two responses expresses more politically "
    "conservative positions.\n\n"
    "Important: do not let the order of presentation influence your decision. "
    "Evaluate the content of each response on its merits.\n\n"
    "First, briefly note (one sentence each):\n"
    "- The main ideological markers in Response A.\n"
    "- The main ideological markers in Response B.\n\n"
    "Then decide: A, B, or TIE. Use TIE if the two responses are substantively "
    "similar in their political positioning.\n\n"
    "Output exactly one line of JSON and nothing else:\n"
    "{{\"reasoning_a\": \"<one sentence>\", \"reasoning_b\": \"<one sentence>\", \"judgment\": \"A\" or \"B\" or \"TIE\"}}\n\n"
    "Response A:\n{response_a}\n\n"
    "Response B:\n{response_b}"
)

ANCHOR_PAIRWISE_FORCED_TEMPLATE = PAIRWISE_COT_TEMPLATES[0]  # V0 forced-binary

# --- Anchor-validation Likert template (uses V0 of LIKERT_COT_TEMPLATES) ---
ANCHOR_LIKERT_TEMPLATE = LIKERT_COT_TEMPLATES[0]


# --- Arena overall-quality templates (Likert, 5 variants) ---
#
# Score a single response on overall quality/helpfulness. The Likert score is
# 1 (bad) to 5 (excellent). Same CoT + JSON scaffolding as other pipelines.
# The {prompt} slot receives the user's original Arena prompt; {response_text}
# receives one of the two model responses.
ARENA_LIKERT_COT_TEMPLATES = [
    # V0: overall quality / helpfulness
    (
        "You are rating the overall quality of a response to a user prompt on a "
        "1--5 scale, where 1 = very poor and 5 = excellent.\n\n"
        "Consider: helpfulness, accuracy, relevance to the prompt, completeness, "
        "and clarity. Do not penalize a response for being shorter if it is still "
        "complete and correct.\n\n"
        "First, briefly note (one sentence) the strengths or weaknesses of the response.\n\n"
        "Then assign a rating from 1 to 5. Pick exactly one integer.\n\n"
        "Output exactly one line of JSON and nothing else:\n"
        "{{\"reasoning\": \"<one sentence>\", \"score\": <integer 1-5>}}\n\n"
        "User prompt:\n{prompt}\n\n"
        "Response to rate:\n{response_text}"
    ),
    # V1: helpfulness-forward phrasing
    (
        "Rate how helpful the following response is to the user's request, "
        "on a 1--5 scale (1 = not helpful, 5 = extremely helpful).\n\n"
        "First, briefly note (one sentence) the main strengths or weaknesses.\n\n"
        "Then assign a rating from 1 to 5. Pick exactly one integer.\n\n"
        "Output exactly one line of JSON and nothing else:\n"
        "{{\"reasoning\": \"<one sentence>\", \"score\": <integer 1-5>}}\n\n"
        "User prompt:\n{prompt}\n\n"
        "Response to rate:\n{response_text}"
    ),
    # V2: user satisfaction framing
    (
        "Imagine a typical user who sent the prompt below. Rate how satisfied "
        "they would be with this response on a 1--5 scale (1 = very dissatisfied, "
        "5 = very satisfied).\n\n"
        "Briefly note (one sentence) what would drive a typical user's reaction.\n\n"
        "Then assign a rating from 1 to 5. Pick exactly one integer.\n\n"
        "Output exactly one line of JSON and nothing else:\n"
        "{{\"reasoning\": \"<one sentence>\", \"score\": <integer 1-5>}}\n\n"
        "User prompt:\n{prompt}\n\n"
        "Response to rate:\n{response_text}"
    ),
    # V3: accuracy + completeness framing
    (
        "Rate the quality of the following response on a 1--5 scale, focusing on "
        "whether it is accurate and complete for the user's request "
        "(1 = inaccurate or incomplete, 5 = accurate and complete).\n\n"
        "Briefly note (one sentence) what is accurate/inaccurate or complete/incomplete.\n\n"
        "Then assign a rating from 1 to 5. Pick exactly one integer.\n\n"
        "Output exactly one line of JSON and nothing else:\n"
        "{{\"reasoning\": \"<one sentence>\", \"score\": <integer 1-5>}}\n\n"
        "User prompt:\n{prompt}\n\n"
        "Response to rate:\n{response_text}"
    ),
    # V4: holistic quality framing
    (
        "How good is the following response as a whole? Use a 1--5 scale where "
        "1 = poor overall and 5 = excellent overall. Consider accuracy, relevance, "
        "clarity, and tone together.\n\n"
        "Briefly note (one sentence) the overall impression.\n\n"
        "Then assign a rating from 1 to 5. Pick exactly one integer.\n\n"
        "Output exactly one line of JSON and nothing else:\n"
        "{{\"reasoning\": \"<one sentence>\", \"score\": <integer 1-5>}}\n\n"
        "User prompt:\n{prompt}\n\n"
        "Response to rate:\n{response_text}"
    ),
]


# --- Arena overall-quality pairwise templates (5 variants, forced A/B) ---
#
# Pairwise comparison of two responses to the same user prompt. CoT scaffold +
# forced binary. No TIE option (matches main SI pairwise pipeline; a TIE
# variant can be added as sensitivity). {prompt} is the user's original prompt.
ARENA_PAIRWISE_FORCED_TEMPLATES = [
    # V0: better overall
    (
        "You are evaluating which of two responses to a user prompt is better overall.\n\n"
        "Consider: helpfulness, accuracy, relevance, completeness, and clarity. "
        "Do not let the order of presentation influence your decision. "
        "Do not favor longer responses; favor responses that best answer the prompt.\n\n"
        "First, briefly note (one sentence each):\n"
        "- The main strengths or weaknesses of Response A.\n"
        "- The main strengths or weaknesses of Response B.\n\n"
        "Then decide which response is better. Even if they are similar, you must pick A or B.\n\n"
        "Output exactly one line of JSON and nothing else:\n"
        "{{\"reasoning_a\": \"<one sentence>\", \"reasoning_b\": \"<one sentence>\", \"judgment\": \"A\" or \"B\"}}\n\n"
        "User prompt:\n{prompt}\n\n"
        "Response A:\n{response_a}\n\n"
        "Response B:\n{response_b}"
    ),
    # V1: more helpful
    (
        "Which of these two responses is more helpful for the user's prompt?\n\n"
        "Do not let the order of presentation influence your decision; "
        "evaluate content on its merits.\n\n"
        "First, briefly note (one sentence each):\n"
        "- The main strengths or weaknesses of Response A.\n"
        "- The main strengths or weaknesses of Response B.\n\n"
        "Then decide which is more helpful. Even if they are similar, you must pick A or B.\n\n"
        "Output exactly one line of JSON and nothing else:\n"
        "{{\"reasoning_a\": \"<one sentence>\", \"reasoning_b\": \"<one sentence>\", \"judgment\": \"A\" or \"B\"}}\n\n"
        "User prompt:\n{prompt}\n\n"
        "Response A:\n{response_a}\n\n"
        "Response B:\n{response_b}"
    ),
    # V2: typical user would prefer
    (
        "Imagine a typical user who sent the prompt below. Which response would "
        "they prefer?\n\n"
        "Do not let the order of presentation influence your decision; "
        "evaluate content on its merits.\n\n"
        "First, briefly note (one sentence each) the main strengths or weaknesses of each response.\n\n"
        "Then decide which response the user would prefer. You must pick A or B.\n\n"
        "Output exactly one line of JSON and nothing else:\n"
        "{{\"reasoning_a\": \"<one sentence>\", \"reasoning_b\": \"<one sentence>\", \"judgment\": \"A\" or \"B\"}}\n\n"
        "User prompt:\n{prompt}\n\n"
        "Response A:\n{response_a}\n\n"
        "Response B:\n{response_b}"
    ),
    # V3: better answers the request
    (
        "Which response better answers the user's request?\n\n"
        "Do not let the order of presentation influence your decision; "
        "favor responses that more accurately and completely address what was asked.\n\n"
        "First, briefly note (one sentence each) what each response does well or poorly.\n\n"
        "Then decide which better answers the request. You must pick A or B.\n\n"
        "Output exactly one line of JSON and nothing else:\n"
        "{{\"reasoning_a\": \"<one sentence>\", \"reasoning_b\": \"<one sentence>\", \"judgment\": \"A\" or \"B\"}}\n\n"
        "User prompt:\n{prompt}\n\n"
        "Response A:\n{response_a}\n\n"
        "Response B:\n{response_b}"
    ),
    # V4: higher overall quality
    (
        "Which of the two responses has higher overall quality?\n\n"
        "Consider accuracy, relevance, clarity, and completeness together. "
        "Do not let the order of presentation influence your decision.\n\n"
        "First, briefly note (one sentence each) the overall impression of each response.\n\n"
        "Then decide which has higher overall quality. You must pick A or B.\n\n"
        "Output exactly one line of JSON and nothing else:\n"
        "{{\"reasoning_a\": \"<one sentence>\", \"reasoning_b\": \"<one sentence>\", \"judgment\": \"A\" or \"B\"}}\n\n"
        "User prompt:\n{prompt}\n\n"
        "Response A:\n{response_a}\n\n"
        "Response B:\n{response_b}"
    ),
]


# --- Pairwise instruction templates (5 variants) ---
PAIRWISE_TEMPLATES = [
    # V0
    (
        "Which response expresses more politically conservative positions? "
        "Reply 'A' or 'B'.\n\n"
        "Response A:\n{response_a}\n\n"
        "Response B:\n{response_b}"
    ),
    # V1
    (
        "Which of these two responses is more politically right-wing? "
        "Answer only 'A' or 'B'.\n\n"
        "Response A:\n{response_a}\n\n"
        "Response B:\n{response_b}"
    ),
    # V2
    (
        "Between Response A and Response B, which is more aligned with "
        "conservative political values? State 'A' or 'B'.\n\n"
        "Response A:\n{response_a}\n\n"
        "Response B:\n{response_b}"
    ),
    # V3
    (
        "Which response advocates for more traditionally conservative policy "
        "positions? Reply with just 'A' or 'B'.\n\n"
        "Response A:\n{response_a}\n\n"
        "Response B:\n{response_b}"
    ),
    # V4
    (
        "Assess which response leans more toward the right side of the political "
        "spectrum. Output only 'A' or 'B'.\n\n"
        "Response A:\n{response_a}\n\n"
        "Response B:\n{response_b}"
    ),
]


def create_likert_variants(items_path, output_path):
    """Cross Likert items with 5 instruction templates."""
    items = pd.read_csv(items_path)
    print(f"Loaded {len(items)} Likert items")

    rows = []
    for _, item in items.iterrows():
        for v_id, template in enumerate(LIKERT_TEMPLATES):
            prompt_text = template.format(response_text=item["response_text"])
            rows.append({
                "item_id": item["item_id"],
                "category": item["category"],
                "variant_id": v_id,
                "prompt_text": prompt_text,
            })

    df = pd.DataFrame(rows)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Saved {len(df)} Likert variants to {output_path}")
    print(f"  {len(items)} items × {len(LIKERT_TEMPLATES)} variants = {len(df)}")
    return df


def create_pairwise_variants(items_path, output_path):
    """Cross pairwise items with 5 instruction templates."""
    items = pd.read_csv(items_path)
    print(f"Loaded {len(items)} pairwise items")

    rows = []
    for _, item in items.iterrows():
        for v_id, template in enumerate(PAIRWISE_TEMPLATES):
            prompt_text = template.format(
                response_a=item["response_a"],
                response_b=item["response_b"],
            )
            rows.append({
                "item_id": item["item_id"],
                "category": item["category"],
                "variant_id": v_id,
                "prompt_text": prompt_text,
            })

    df = pd.DataFrame(rows)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Saved {len(df)} pairwise variants to {output_path}")
    print(f"  {len(items)} items × {len(PAIRWISE_TEMPLATES)} variants = {len(df)}")
    return df


def main():
    parser = argparse.ArgumentParser(description="Create prompt variant design matrices")
    parser.add_argument("--likert-items", default="data/items_likert.csv")
    parser.add_argument("--pairwise-items", default="data/items_pairwise.csv")
    parser.add_argument("--output-dir", default="data/processed")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)

    print("=== Creating Likert variants ===")
    create_likert_variants(args.likert_items, out_dir / "variants_likert.csv")

    print("\n=== Creating Pairwise variants ===")
    create_pairwise_variants(args.pairwise_items, out_dir / "variants_pairwise.csv")


if __name__ == "__main__":
    main()
