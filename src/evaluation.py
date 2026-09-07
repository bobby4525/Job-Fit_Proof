"""Shared evaluation utilities for CareerFit experiments."""

from __future__ import annotations

from typing import Iterable

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from label_schema import LABELS


def evaluate_predictions(y_true: Iterable[str], y_pred: Iterable[str]) -> dict:
    y_true = list(y_true)
    y_pred = list(y_pred)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=LABELS, average="weighted", zero_division=0)),
        "classification_report": classification_report(
            y_true, y_pred, labels=LABELS, output_dict=True, zero_division=0
        ),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=LABELS).tolist(),
    }


def per_class_frame(metrics: dict) -> pd.DataFrame:
    report = metrics["classification_report"]
    rows = []
    for label in LABELS:
        item = report[label]
        rows.append(
            {
                "label": label,
                "precision": item["precision"],
                "recall": item["recall"],
                "f1": item["f1-score"],
                "support": item["support"],
            }
        )
    return pd.DataFrame(rows)
