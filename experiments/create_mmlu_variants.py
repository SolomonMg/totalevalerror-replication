"""Create MMLU multiple-choice prompt variants for TLE experiment.

5 framing-variant instruction templates for multiple-choice QA. Templates
vary the framing (Standard, Exam, Expert, Minimal, Analytical) and option
delimiters while preserving the core task: select the correct letter
from four choices. These are NOT strict semantic paraphrases of a single
seed; the framings can shift refusal/abstention rates on borderline
items. The resulting sigma^2_phi captures phrasing-and-framing sensitivity,
an upper bound on what a strict-paraphrase pilot would estimate. See
Assumption 1 in the paper for the formal interpretation.

Usage:
    python create_mmlu_variants.py [--items data/items_mmlu.csv] \
        [--n_variants 5] [--output data/processed/variants_mmlu.csv]
"""

import argparse
from pathlib import Path

import pandas as pd


MMLU_TEMPLATES = [
    # V0: Standard
    (
        "Answer the following multiple-choice question. Reply with only the "
        "letter (A, B, C, or D).\n\n"
        "Question: {question}\n"
        "A. {A}\n"
        "B. {B}\n"
        "C. {C}\n"
        "D. {D}"
    ),
    # V1: Exam framing
    (
        "You are taking a test. Select the correct answer for the question "
        "below. Respond with just the letter.\n\n"
        "Q: {question}\n"
        "(A) {A}\n"
        "(B) {B}\n"
        "(C) {C}\n"
        "(D) {D}"
    ),
    # V2: Expert framing
    (
        "As a knowledgeable expert, identify the correct answer to this "
        "question. Output only the letter of the correct choice.\n\n"
        "{question}\n"
        "Options:\n"
        "A) {A}\n"
        "B) {B}\n"
        "C) {C}\n"
        "D) {D}"
    ),
    # V3: Minimal
    (
        "{question}\n"
        "A. {A}\n"
        "B. {B}\n"
        "C. {C}\n"
        "D. {D}\n\n"
        "Answer:"
    ),
    # V4: Analytical
    (
        "Read the following question carefully and select the best answer "
        "from the options provided. State only the letter.\n\n"
        "Question: {question}\n"
        "Choices:\n"
        "a) {A}\n"
        "b) {B}\n"
        "c) {C}\n"
        "d) {D}"
    ),
]


def create_mmlu_variants(items_path, output_path, n_variants):
    """Cross MMLU items with instruction templates to build design matrix."""
    items = pd.read_csv(items_path)
    print(f"Loaded {len(items)} MMLU items from {items_path}")

    templates = MMLU_TEMPLATES[:n_variants]

    rows = []
    for _, item in items.iterrows():
        for v_id, template in enumerate(templates):
            prompt_text = template.format(
                question=item["question"],
                A=item["choice_A"],
                B=item["choice_B"],
                C=item["choice_C"],
                D=item["choice_D"],
            )
            rows.append({
                "item_id": item["item_id"],
                "category": item["category"],
                "subcategory": item["subcategory"],
                "variant_id": f"v_{v_id}",
                "prompt_text": prompt_text,
                "correct_answer": item["correct_answer"],
            })

    df = pd.DataFrame(rows)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Saved {len(df)} MMLU variants to {output_path}")
    print(f"  {len(items)} items x {n_variants} variants = {len(df)}")
    return df


def main():
    parser = argparse.ArgumentParser(
        description="Create MMLU multiple-choice prompt variants"
    )
    parser.add_argument("--items", default="data/items_mmlu.csv")
    parser.add_argument("--n_variants", type=int, default=5,
                        help="Number of templates to use (1-5, default: 5)")
    parser.add_argument("--output", default="data/processed/variants_mmlu.csv")
    args = parser.parse_args()

    if not 1 <= args.n_variants <= len(MMLU_TEMPLATES):
        parser.error(
            f"--n_variants must be between 1 and {len(MMLU_TEMPLATES)}"
        )

    create_mmlu_variants(args.items, args.output, args.n_variants)


if __name__ == "__main__":
    main()
