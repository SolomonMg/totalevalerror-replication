"""Curate MMLU items for TLE empirical demonstration.

Stratified sample from MMLU test set across 4 broad categories
(STEM, Humanities, Social Sciences, Other), with 2 specific subjects
per category for subcategory variation.

Usage:
    python curate_mmlu_items.py [--n_items 200] [--output-dir data]
"""

import argparse
import io
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from config import SEED

# --- Subject selection by broad category ---
CATEGORY_SUBJECTS = {
    "STEM": ["abstract_algebra", "college_physics"],
    "Humanities": ["philosophy", "world_religions"],
    "Social Sciences": ["us_foreign_policy", "sociology"],
    "Other": ["professional_law", "miscellaneous"],
}

# Answer index (0-3) to letter mapping (for GitHub CSV fallback)
ANSWER_MAP = {0: "A", 1: "B", 2: "C", 3: "D"}


def load_via_datasets():
    """Load MMLU test set via HuggingFace datasets library.

    Returns dict mapping subject name -> DataFrame with columns:
        question, A, B, C, D, answer (letter A-D)
    Only loads the subjects we need.
    """
    from datasets import load_dataset

    needed_subjects = set()
    for subjects in CATEGORY_SUBJECTS.values():
        needed_subjects.update(subjects)

    ds = load_dataset("cais/mmlu", "all", split="test")
    df_all = ds.to_pandas()

    # cais/mmlu has columns: question, choices (list), answer (int 0-3), subject
    subject_dfs = {}
    for subject in needed_subjects:
        mask = df_all["subject"] == subject
        sub = df_all[mask].copy()
        if len(sub) == 0:
            raise ValueError(f"Subject '{subject}' not found in dataset. "
                             f"Available: {sorted(df_all['subject'].unique())[:20]}...")

        # Expand choices list into separate columns
        choices = pd.DataFrame(sub["choices"].tolist(), columns=["A", "B", "C", "D"])
        choices.index = sub.index
        sub = sub.drop(columns=["choices"])
        sub = pd.concat([sub, choices], axis=1)
        sub["answer"] = sub["answer"].map(ANSWER_MAP)
        subject_dfs[subject] = sub.reset_index(drop=True)

    return subject_dfs


def load_via_github():
    """Fallback: download MMLU test CSVs from hendrycks/test GitHub repo.

    Returns dict mapping subject name -> DataFrame with columns:
        question, A, B, C, D, answer (letter A-D)
    """
    base_url = "https://raw.githubusercontent.com/hendrycks/test/master/test/"
    needed_subjects = set()
    for subjects in CATEGORY_SUBJECTS.values():
        needed_subjects.update(subjects)

    subject_dfs = {}
    for subject in sorted(needed_subjects):
        url = f"{base_url}{subject}_test.csv"
        print(f"  Downloading {subject} from {url}")
        try:
            response = urllib.request.urlopen(url)
            content = response.read().decode("utf-8")
        except Exception as e:
            raise RuntimeError(f"Failed to download {url}: {e}")

        # CSV has no header: question, A, B, C, D, answer
        df = pd.read_csv(
            io.StringIO(content),
            header=None,
            names=["question", "A", "B", "C", "D", "answer"],
        )
        subject_dfs[subject] = df

    return subject_dfs


def load_subjects():
    """Load MMLU subjects, trying datasets library first, then GitHub fallback."""
    try:
        print("Loading MMLU via HuggingFace datasets library...")
        return load_via_datasets()
    except ImportError:
        print("datasets library not available, falling back to GitHub download...")
        return load_via_github()


def curate_items(subject_dfs, n_items, rng):
    """Stratified sample across broad categories and subjects.

    Samples n_items/4 per broad category, split evenly across subjects
    within each category.
    """
    n_categories = len(CATEGORY_SUBJECTS)
    per_category = n_items // n_categories
    remainder = n_items % n_categories

    items = []
    item_counter = 0

    for cat_idx, (category, subjects) in enumerate(CATEGORY_SUBJECTS.items()):
        # Distribute remainder across first few categories
        cat_n = per_category + (1 if cat_idx < remainder else 0)
        n_subjects = len(subjects)
        per_subject = cat_n // n_subjects
        sub_remainder = cat_n % n_subjects

        for sub_idx, subject in enumerate(subjects):
            sub_n = per_subject + (1 if sub_idx < sub_remainder else 0)
            df = subject_dfs[subject]

            if sub_n > len(df):
                print(f"  WARNING: requested {sub_n} items from {subject} "
                      f"but only {len(df)} available; taking all")
                sub_n = len(df)

            sampled = df.sample(n=sub_n, random_state=rng)

            for _, row in sampled.iterrows():
                item_counter += 1
                items.append({
                    "item_id": f"mmlu_{item_counter:04d}",
                    "category": category,
                    "subcategory": subject,
                    "question": row["question"],
                    "choice_A": row["A"],
                    "choice_B": row["B"],
                    "choice_C": row["C"],
                    "choice_D": row["D"],
                    "correct_answer": row["answer"],
                })

    return pd.DataFrame(items)


def main():
    parser = argparse.ArgumentParser(description="Curate MMLU items for TLE demo")
    parser.add_argument(
        "--n_items",
        type=int,
        default=200,
        help="Total number of items to sample (default: 200)",
    )
    parser.add_argument(
        "--output-dir",
        default="data",
        help="Output directory for item CSV",
    )
    args = parser.parse_args()

    rng = np.random.RandomState(SEED)

    subject_dfs = load_subjects()

    # Print available counts
    print("\nAvailable items per subject:")
    for category, subjects in CATEGORY_SUBJECTS.items():
        for subject in subjects:
            n = len(subject_dfs[subject])
            print(f"  {category} / {subject}: {n}")

    # Curate
    df = curate_items(subject_dfs, args.n_items, rng)

    # Save
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "items_mmlu.csv"
    df.to_csv(out_path, index=False)

    # Summary
    print(f"\nSaved: {out_path} ({len(df)} items)")
    print(f"\nItems per category:")
    print(df["category"].value_counts().to_string())
    print(f"\nItems per subcategory:")
    print(df.groupby(["category", "subcategory"]).size().to_string())

    # Verification
    assert len(df) == args.n_items, \
        f"Expected {args.n_items} items, got {len(df)}"
    assert set(df["category"]) == set(CATEGORY_SUBJECTS.keys()), \
        f"Missing categories: {set(CATEGORY_SUBJECTS.keys()) - set(df['category'])}"
    assert df["correct_answer"].isin(["A", "B", "C", "D"]).all(), \
        "Invalid correct_answer values found"
    print("\nVerification passed.")


if __name__ == "__main__":
    main()
