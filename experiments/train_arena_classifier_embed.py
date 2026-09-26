"""Train a category classifier using sentence embeddings + logistic regression.

This is the fallback when TF-IDF + LinearSVC underperforms on Persuasion
(F1 < 0.65). Sentence embeddings from a pretrained model capture semantic
structure in Persuasion (argumentation, persuasion tone) that char/word
n-grams miss.

Model: sentence-transformers/all-MiniLM-L6-v2 (22M params, 384-dim).
Classifier: LogisticRegression(class_weight='balanced', C=1.0).

Reports per-class precision/recall/F1 on 20% stratified held-out split.
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SEED = 42
CATEGORIES = ("creative_writing", "persuasion", "coding", "factual_qa", "other")

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def load_labeled():
    cat = pd.read_csv(PROJECT_ROOT / "data/processed/arena_battles_categorized.csv")
    cands = pd.read_parquet(PROJECT_ROOT / "data/raw/arena_battles_candidates.parquet")
    df = cat.merge(cands[["battle_id", "prompt"]], on="battle_id", how="left")
    df = df.dropna(subset=["prompt", "category_llm"])
    df = df[df["category_llm"].isin(CATEGORIES)].reset_index(drop=True)
    df["prompt_trunc"] = df["prompt"].str.slice(0, 1200)
    return df


def pick_device():
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def main():
    df = load_labeled()
    print(f"Training set: {len(df):,} labeled prompts")
    print(df["category_llm"].value_counts().to_string())
    print()

    X = df["prompt_trunc"].astype(str).tolist()
    y = df["category_llm"].astype(str).tolist()

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED,
    )
    print(f"Train: {len(X_tr):,}   Test: {len(X_te):,}")

    device = pick_device()
    print(f"\nDevice: {device}")
    print(f"Loading {EMBED_MODEL} ...")
    embed = SentenceTransformer(EMBED_MODEL, device=device)

    print("Encoding train prompts ...")
    E_tr = embed.encode(X_tr, batch_size=64, show_progress_bar=True,
                        convert_to_numpy=True, normalize_embeddings=True)
    print("Encoding test prompts ...")
    E_te = embed.encode(X_te, batch_size=64, show_progress_bar=True,
                        convert_to_numpy=True, normalize_embeddings=True)

    print("\nFitting LogisticRegression ...")
    clf = LogisticRegression(
        class_weight="balanced", C=1.0, max_iter=2000,
        random_state=SEED,
    )
    clf.fit(E_tr, y_tr)

    y_pred = clf.predict(E_te)
    print("\n=== Held-out test report ===")
    print(classification_report(y_te, y_pred, digits=3, labels=list(CATEGORIES)))

    report = classification_report(y_te, y_pred, labels=list(CATEGORIES), output_dict=True)
    persuasion_f1 = report["persuasion"]["f1-score"]
    macro_f1 = report["macro avg"]["f1-score"]
    print(f"Macro F1:      {macro_f1:.3f}")
    print(f"Persuasion F1: {persuasion_f1:.3f}")

    if persuasion_f1 >= 0.75 and macro_f1 >= 0.75:
        print("=> Meets success criteria. Embeddings + LogReg sufficient.")
    elif persuasion_f1 >= 0.65:
        print("=> Persuasion F1 in 0.65-0.75 range. Consider gpt-oss fallback on "
              "low-confidence items (margin < 0.3).")
    else:
        print("=> Persuasion F1 < 0.65. Consider DistilBERT fine-tune.")

    out_dir = PROJECT_ROOT / "data/processed/arena_classifier"
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, out_dir / "embed_logreg.joblib")
    print(f"\nSaved {out_dir / 'embed_logreg.joblib'}")

    # Save also the test-set predictions with predict_proba for spot check + confidence.
    proba = clf.predict_proba(E_te)
    top_prob = proba.max(axis=1)
    pd.DataFrame({
        "prompt_trunc": X_te,
        "y_true": list(y_te),
        "y_pred": list(y_pred),
        "top_prob": top_prob,
    }).to_csv(out_dir / "test_predictions_embed.csv", index=False)
    print(f"Saved {out_dir / 'test_predictions_embed.csv'}")


if __name__ == "__main__":
    main()
