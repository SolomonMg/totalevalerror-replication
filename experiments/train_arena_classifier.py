"""Train a local category classifier on existing LLM-labeled Arena prompts.

Input: data/processed/arena_battles_categorized.csv (11,957 labeled battles)
       joined with data/raw/arena_battles_candidates.parquet for prompt text.

Classifier: TF-IDF (char 2-5 + word 1-2) -> LinearSVC(class_weight='balanced').
Holds out 20% stratified test split; reports per-class precision/recall/F1.
Saves model + vectorizer for inference in classify_arena_full_svm.py.

Success criteria: macro F1 >= 0.75 and Persuasion F1 >= 0.75.
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SEED = 42

CATEGORIES = ("creative_writing", "persuasion", "coding", "factual_qa", "other")


def load_labeled():
    cat = pd.read_csv(PROJECT_ROOT / "data/processed/arena_battles_categorized.csv")
    cands = pd.read_parquet(PROJECT_ROOT / "data/raw/arena_battles_candidates.parquet")
    df = cat.merge(cands[["battle_id", "prompt"]], on="battle_id", how="left")
    df = df.dropna(subset=["prompt", "category_llm"])
    df = df[df["category_llm"].isin(CATEGORIES)].reset_index(drop=True)
    # Classifier saw only first 1200 chars during labeling; train on same view.
    df["prompt_trunc"] = df["prompt"].str.slice(0, 1200)
    return df


def main():
    df = load_labeled()
    print(f"Training set: {len(df):,} labeled prompts")
    print("Class distribution:")
    print(df["category_llm"].value_counts().to_string())
    print()

    X = df["prompt_trunc"].astype(str).tolist()
    y = df["category_llm"].astype(str).tolist()
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED,
    )
    print(f"Train: {len(X_tr):,}   Test: {len(X_te):,}")

    word_vec = TfidfVectorizer(
        analyzer="word", ngram_range=(1, 2), min_df=3, max_df=0.95, sublinear_tf=True
    )
    char_vec = TfidfVectorizer(
        analyzer="char_wb", ngram_range=(2, 5), min_df=3, max_df=0.95, sublinear_tf=True
    )
    features = FeatureUnion([("word", word_vec), ("char", char_vec)])
    clf = LinearSVC(class_weight="balanced", C=1.0, random_state=SEED, max_iter=5000)
    pipe = Pipeline([("features", features), ("clf", clf)])

    print("\nFitting SVM pipeline ...")
    pipe.fit(X_tr, y_tr)

    y_pred = pipe.predict(X_te)
    print("\n=== Held-out test report ===")
    print(classification_report(y_te, y_pred, digits=3, labels=list(CATEGORIES)))

    # Compute per-class F1 via classification_report dict for decision logging.
    report = classification_report(y_te, y_pred, labels=list(CATEGORIES), output_dict=True)
    persuasion_f1 = report["persuasion"]["f1-score"]
    macro_f1 = report["macro avg"]["f1-score"]
    print(f"Macro F1:      {macro_f1:.3f}")
    print(f"Persuasion F1: {persuasion_f1:.3f}")

    if persuasion_f1 >= 0.75 and macro_f1 >= 0.75:
        print("=> Meets success criteria. SVM labels can be used directly.")
    elif persuasion_f1 >= 0.65:
        print("=> Persuasion F1 in 0.65-0.75 range. Consider gpt-oss fallback on "
              "items with SVM decision margin < 0.3.")
    else:
        print("=> Persuasion F1 < 0.65. Fallback to DistilBERT fine-tune recommended.")

    out_dir = PROJECT_ROOT / "data/processed/arena_classifier"
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, out_dir / "svm_pipeline.joblib")
    print(f"\nSaved {out_dir / 'svm_pipeline.joblib'}")

    # Also save test-set predictions + confidence margins for spot-check later.
    # decision_function for LinearSVC returns one-vs-rest margins per class.
    margins = pipe.decision_function(X_te)
    classes = pipe.classes_
    # Best-class margin minus second-best: a simple "confidence" proxy
    sorted_m = np.sort(margins, axis=1)
    conf = sorted_m[:, -1] - sorted_m[:, -2]
    pd.DataFrame({
        "prompt_trunc": X_te,
        "y_true": list(y_te),
        "y_pred": list(y_pred),
        "confidence": conf,
    }).to_csv(out_dir / "test_predictions.csv", index=False)
    print(f"Saved {out_dir / 'test_predictions.csv'}")


if __name__ == "__main__":
    main()
