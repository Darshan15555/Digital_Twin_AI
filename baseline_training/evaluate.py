from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def find_best_f1_threshold(y_true, y_proba) -> tuple[float, float]:
    thresholds = np.arange(0.05, 0.96, 0.01)
    best_threshold = 0.5
    best_f1 = -1.0
    for threshold in thresholds:
        preds = (y_proba >= threshold).astype(int)
        score = f1_score(y_true, preds, zero_division=0)
        if score > best_f1:
            best_f1 = float(score)
            best_threshold = float(threshold)
    return best_threshold, best_f1


def metrics_at_threshold(y_true, y_proba, threshold: float) -> dict[str, float]:
    preds = (y_proba >= threshold).astype(int)
    return {
        "threshold": float(threshold),
        "f1": float(f1_score(y_true, preds, zero_division=0)),
        "precision": float(precision_score(y_true, preds, zero_division=0)),
        "recall": float(recall_score(y_true, preds, zero_division=0)),
    }


def evaluate_split(y_true, y_proba) -> dict[str, float | dict[str, float]]:
    auroc = float(roc_auc_score(y_true, y_proba)) if len(np.unique(y_true)) > 1 else float("nan")
    best_threshold, best_f1 = find_best_f1_threshold(y_true, y_proba)
    return {
        "auroc": auroc,
        "default_0_5": metrics_at_threshold(y_true, y_proba, 0.5),
        "best_f1": metrics_at_threshold(y_true, y_proba, best_threshold),
        "best_f1_value": float(best_f1),
    }


def evaluate_model(model, X_val, y_val, X_test, y_test) -> dict[str, object]:
    val_proba = model.predict_proba(X_val)[:, 1]
    val_metrics = evaluate_split(y_val, val_proba)

    threshold = float(val_metrics["best_f1"]["threshold"])
    test_proba = model.predict_proba(X_test)[:, 1]
    test_default = metrics_at_threshold(y_test, test_proba, 0.5)
    test_best = metrics_at_threshold(y_test, test_proba, threshold)
    test_auroc = float(roc_auc_score(y_test, test_proba)) if len(np.unique(y_test)) > 1 else float("nan")

    return {
        "validation": val_metrics,
        "test": {
            "auroc": test_auroc,
            "default_0_5": test_default,
            "best_f1_threshold_from_val": test_best,
        },
    }


def select_best_model(report_by_model: dict[str, dict]) -> str:
    if not report_by_model:
        raise ValueError("No model reports provided for selection")
    ranked = sorted(
        report_by_model.items(),
        key=lambda item: (
            item[1]["validation"]["auroc"],
            item[1]["validation"]["best_f1_value"],
        ),
        reverse=True,
    )
    return ranked[0][0]

