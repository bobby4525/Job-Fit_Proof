"""Reproducible reference implementation of the CareerFit TF-IDF baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from evaluation import evaluate_predictions

TEXT_COLUMNS = [
    "requirement_text",
    "requirement_type",
    "priority",
    "target_role",
    "years_experience",
    "education_level",
    "skills",
    "english_level",
    "mandarin_level",
    "domain_experience",
    "location",
    "taiwan_work_authorization",
    "notes",
]


def build_model_text(df: pd.DataFrame) -> pd.Series:
    missing = [c for c in TEXT_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected columns: {missing}")
    return df[TEXT_COLUMNS].fillna("").astype(str).agg(" | ".join, axis=1)


def load_dataset(pairs_path: Path, candidates_path: Path) -> pd.DataFrame:
    pairs = pd.read_csv(pairs_path)
    candidates = pd.read_csv(candidates_path)
    df = pairs.merge(candidates, on="candidate_id", how="left", validate="many_to_one")
    if "split" not in df.columns or "analytical_seed_label" not in df.columns:
        raise ValueError("Dataset must contain split and analytical_seed_label columns.")
    df["model_text"] = build_model_text(df)
    return df


def train_and_evaluate(df: pd.DataFrame) -> tuple[Pipeline, dict]:
    train = df[df["split"] == "train"].copy()
    validation = df[df["split"] == "validation"].copy()
    test = df[df["split"] == "test"].copy()

    model = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    ngram_range=(1, 2),
                    min_df=2,
                    max_features=60000,
                    sublinear_tf=True,
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )

    model.fit(train["model_text"], train["analytical_seed_label"])

    results = {}
    for name, split_df in [("validation", validation), ("test", test)]:
        pred = model.predict(split_df["model_text"])
        results[name] = evaluate_predictions(split_df["analytical_seed_label"], pred)

    return model, results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("results/tfidf_reference_metrics.json"))
    args = parser.parse_args()

    df = load_dataset(args.pairs, args.candidates)
    _, results = train_and_evaluate(df)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print("Validation accuracy:", round(results["validation"]["accuracy"], 4))
    print("Validation macro-F1:", round(results["validation"]["macro_f1"], 4))
    print("Test accuracy:", round(results["test"]["accuracy"], 4))
    print("Test macro-F1:", round(results["test"]["macro_f1"], 4))


if __name__ == "__main__":
    main()
