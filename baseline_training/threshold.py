from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score


def metrics_at_threshold(y_true, y_proba, threshold: float) -> dict[str, float]:
    y_true_arr = np.asarray(y_true).astype(int)
    y_proba_arr = np.asarray(y_proba).astype(float)
    preds = (y_proba_arr >= float(threshold)).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true_arr, preds, labels=[0, 1]).ravel()
    sensitivity = float(tp / max(tp + fn, 1))
    specificity = float(tn / max(tn + fp, 1))

    return {
        "threshold": float(threshold),
        "f1": float(f1_score(y_true_arr, preds, zero_division=0)),
        "precision": float(precision_score(y_true_arr, preds, zero_division=0)),
        "recall": float(recall_score(y_true_arr, preds, zero_division=0)),
        "sensitivity": sensitivity,
        "specificity": specificity,
        "youden_index": float(sensitivity + specificity - 1.0),
    }


def threshold_scan(y_true, y_proba, *, step: float = 0.01) -> pd.DataFrame:
    thresholds = np.arange(step, 1.0, step)
    rows = [metrics_at_threshold(y_true, y_proba, float(thr)) for thr in thresholds]
    return pd.DataFrame(rows)


def optimize_thresholds(
    y_true,
    y_proba,
    *,
    sensitivity_target: float = 0.90,
    step: float = 0.01,
) -> dict[str, dict[str, float | str | bool]]:
    scan = threshold_scan(y_true, y_proba, step=step)
    if scan.empty:
        raise ValueError("Threshold scan failed: no thresholds evaluated")

    best_f1_row = scan.sort_values(["f1", "precision", "recall"], ascending=False).iloc[0].to_dict()
    best_youden_row = scan.sort_values(["youden_index", "f1"], ascending=False).iloc[0].to_dict()

    high_sens_df = scan[scan["sensitivity"] >= float(sensitivity_target)]
    sensitivity_met = not high_sens_df.empty
    if sensitivity_met:
        high_sens_row = (
            high_sens_df.sort_values(["precision", "f1", "threshold"], ascending=[False, False, True]).iloc[0].to_dict()
        )
    else:
        high_sens_row = scan.sort_values(["sensitivity", "precision"], ascending=False).iloc[0].to_dict()

    return {
        "max_f1": {
            **best_f1_row,
            "rule": "Maximize F1",
        },
        "high_sensitivity": {
            **high_sens_row,
            "rule": f"Sensitivity >= {sensitivity_target:.2f}",
            "target_met": bool(sensitivity_met),
            "target_value": float(sensitivity_target),
        },
        "balanced_youden": {
            **best_youden_row,
            "rule": "Maximize Youden index (TPR - FPR)",
        },
    }


def evaluate_threshold_options(y_true, y_proba, threshold_options: dict[str, dict[str, float | str | bool]]) -> dict[str, dict]:
    evaluation: dict[str, dict] = {}
    for name, payload in threshold_options.items():
        threshold = float(payload["threshold"])
        evaluation[name] = metrics_at_threshold(y_true, y_proba, threshold)
    return evaluation
