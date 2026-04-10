from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from models.common import DATA_DIR, ensure_directories, setup_logger, write_json

log = setup_logger("models.evaluation", "models/evaluation.log")


def _save_plot(path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


def evaluate_model(model, X, y_true, meta_df, model_name: str, split: str, threshold: float = 0.5) -> dict:
    ensure_directories()
    output_dir = DATA_DIR / f"eval_{model_name}_{split}"
    output_dir.mkdir(parents=True, exist_ok=True)
    y_proba = model.predict_proba(X)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)

    unique_classes = np.unique(np.asarray(y_true))
    auroc = roc_auc_score(y_true, y_proba) if len(unique_classes) > 1 else float("nan")
    auprc = average_precision_score(y_true, y_proba) if len(unique_classes) > 1 else float("nan")
    f1 = f1_score(y_true, y_pred, zero_division=0)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    brier = brier_score_loss(y_true, y_proba)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    ppv = tp / (tp + fp) if (tp + fp) else 0.0
    npv = tn / (tn + fn) if (tn + fn) else 0.0
    if len(unique_classes) > 1:
        fpr, tpr, roc_thresholds = roc_curve(y_true, y_proba)
        pr_prec, pr_rec, pr_thresholds = precision_recall_curve(y_true, y_proba)
        frac_pos, mean_pred = calibration_curve(y_true, y_proba, n_bins=10, strategy="quantile")
    else:
        fpr, tpr, roc_thresholds = np.array([0.0, 1.0]), np.array([0.0, 1.0]), np.array([1.0, 0.0])
        pr_prec, pr_rec, pr_thresholds = np.array([np.mean(y_true)]), np.array([1.0]), np.array([])
        frac_pos, mean_pred = np.array([np.mean(y_true)]), np.array([float(np.mean(y_proba))])
    if len(mean_pred) > 1:
        slope, intercept = np.polyfit(mean_pred, frac_pos, deg=1)
    else:
        slope, intercept = 1.0, 0.0
    ece = float(np.abs(frac_pos - mean_pred).mean()) if len(mean_pred) else 0.0

    patient_agg = pd.DataFrame({
        "patientunitstayid": meta_df["patientunitstayid"].values,
        "y_true": np.asarray(y_true),
        "y_proba": y_proba,
    }).groupby("patientunitstayid", as_index=False).agg(
        y_true_patient=("y_true", "max"),
        y_proba_max=("y_proba", "max"),
        y_proba_mean=("y_proba", "mean"),
        y_proba_last=("y_proba", "last"),
    )
    patient_unique = patient_agg["y_true_patient"].nunique()
    patient_auroc = roc_auc_score(patient_agg["y_true_patient"], patient_agg["y_proba_max"]) if patient_unique > 1 else float("nan")
    patient_auroc_mean = roc_auc_score(patient_agg["y_true_patient"], patient_agg["y_proba_mean"]) if patient_unique > 1 else float("nan")
    patient_auprc = average_precision_score(patient_agg["y_true_patient"], patient_agg["y_proba_max"]) if patient_unique > 1 else float("nan")

    top_features = []
    if hasattr(model, "feature_importances_"):
        importance_df = pd.DataFrame({"feature": X.columns, "importance": model.feature_importances_}).sort_values("importance", ascending=False)
        top_features = importance_df.head(20).to_dict("records")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns

        if hasattr(model, "feature_importances_"):
            plt.figure(figsize=(10, 6))
            sns.barplot(data=importance_df.head(20), x="importance", y="feature", orient="h")
            plt.title(f"Top Feature Importance: {model_name}")
            _save_plot(output_dir / "feature_importance.png")

        plt.figure(figsize=(7, 6))
        plt.plot(fpr, tpr, label=f"AUROC={auroc:.3f}")
        plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC Curve: {model_name} ({split})")
        plt.legend()
        _save_plot(output_dir / "roc_curve.png")

        plt.figure(figsize=(7, 6))
        plt.plot(pr_rec, pr_prec, label=f"AUPRC={auprc:.3f}")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title(f"PR Curve: {model_name} ({split})")
        plt.legend()
        _save_plot(output_dir / "pr_curve.png")

        plt.figure(figsize=(7, 6))
        plt.plot(mean_pred, frac_pos, marker="o")
        plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
        plt.xlabel("Mean Predicted Value")
        plt.ylabel("Fraction of Positives")
        plt.title(f"Calibration Plot: {model_name} ({split})")
        _save_plot(output_dir / "calibration_plot.png")

        plt.figure(figsize=(6, 5))
        sns.heatmap(np.array([[tn, fp], [fn, tp]]), annot=True, fmt="d", cmap="Blues", xticklabels=["Pred 0", "Pred 1"], yticklabels=["True 0", "True 1"])
        plt.title(f"Confusion Matrix: {model_name} ({split})")
        _save_plot(output_dir / "confusion_matrix.png")

        plt.figure(figsize=(8, 5))
        sns.histplot(y_proba[np.asarray(y_true) == 0], color="steelblue", label="Negative", kde=True, stat="density", bins=30)
        sns.histplot(y_proba[np.asarray(y_true) == 1], color="firebrick", label="Positive", kde=True, stat="density", bins=30)
        plt.legend()
        plt.title(f"Score Distribution: {model_name} ({split})")
        _save_plot(output_dir / "score_distribution.png")
    except ImportError:
        log.warning("matplotlib/seaborn not installed; skipping evaluation plots for %s", model_name)

    results = {
        "model_name": model_name,
        "split": split,
        "threshold": float(threshold),
        "n_samples": int(len(y_true)),
        "n_patients": int(patient_agg["patientunitstayid"].nunique()),
        "positive_rate": float(np.mean(y_true)),
        "auroc": float(auroc),
        "auprc": float(auprc),
        "f1": float(f1),
        "precision": float(precision),
        "sensitivity": float(recall),
        "specificity": float(specificity),
        "ppv": float(ppv),
        "npv": float(npv),
        "brier_score": float(brier),
        "calibration_slope": float(slope),
        "calibration_intercept": float(intercept),
        "calibration_ece": float(ece),
        "patient_auroc": float(patient_auroc),
        "patient_auroc_mean": float(patient_auroc_mean),
        "patient_auprc": float(patient_auprc),
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "top_features": top_features,
    }
    write_json(output_dir / "metrics.json", results)
    return results


def tune_threshold(y_true, y_proba, patient_ids) -> dict:
    del patient_ids
    thresholds_to_test = np.arange(0.05, 0.95, 0.01)
    f1_scores = [f1_score(y_true, (y_proba >= threshold).astype(int), zero_division=0) for threshold in thresholds_to_test]
    max_f1_threshold = float(thresholds_to_test[int(np.argmax(f1_scores))])
    max_f1_value = float(np.max(f1_scores))
    fpr, tpr, roc_thresh = roc_curve(y_true, y_proba)
    finite_mask = np.isfinite(roc_thresh)
    fpr_finite = fpr[finite_mask]
    tpr_finite = tpr[finite_mask]
    roc_thresh_finite = roc_thresh[finite_mask]
    sens_90_mask = tpr_finite >= 0.90
    max_sens90_threshold = float(roc_thresh_finite[sens_90_mask][-1]) if sens_90_mask.any() else float(thresholds_to_test[0])
    spec_90_mask = (1 - fpr_finite) >= 0.90
    max_spec90_threshold = float(roc_thresh_finite[spec_90_mask][0]) if spec_90_mask.any() else 0.5
    balanced_threshold = float(roc_thresh_finite[int(np.argmax(tpr_finite - fpr_finite))]) if len(roc_thresh_finite) else 0.5

    for name, threshold in [
        ("Max F1", max_f1_threshold),
        ("High Sensitivity (>=90%)", max_sens90_threshold),
        ("High Specificity (>=90%)", max_spec90_threshold),
        ("Balanced (Youden J)", balanced_threshold),
        ("Default", 0.5),
    ]:
        preds = (y_proba >= threshold).astype(int)
        sens = recall_score(y_true, preds, zero_division=0)
        spec = precision_score(1 - np.asarray(y_true), 1 - preds, zero_division=0)
        f1 = f1_score(y_true, preds, zero_division=0)
        log.info("%s threshold %.3f | Sens=%.3f Spec=%.3f F1=%.3f", name, threshold, sens, spec, f1)

    return {
        "max_f1": max_f1_threshold,
        "max_f1_value": max_f1_value,
        "max_sensitivity_90": max_sens90_threshold,
        "max_specificity_90": max_spec90_threshold,
        "balanced": balanced_threshold,
        "default": 0.5,
        "recommended": max_f1_threshold,
        "clinical_note": "Use max_sensitivity_90 for screening and max_specificity_90 for high-confidence action thresholds.",
    }


def save_all_artifacts(model, model_name: str, preprocessor, thresholds: dict, feature_names: list[str], train_results: dict, test_results: dict, both_results: dict) -> None:
    import pickle

    ensure_directories()
    model_path = Path("models/artifacts") / f"{model_name}.pkl"
    with model_path.open("wb") as handle:
        pickle.dump(model, handle)

    write_json(Path("models/artifacts/feature_list.json"), {"features": feature_names, "count": len(feature_names)})

    config_path = Path("models/artifacts/preprocessing_config.json")
    config_payload = {}
    if config_path.exists():
        config_payload = json.loads(config_path.read_text(encoding="utf-8"))
    if "preprocessors" not in config_payload:
        if config_payload.get("feature_names"):
            config_payload = {
                "default_target": config_payload.get("target", preprocessor.target),
                "preprocessors": {
                    config_payload.get("target", preprocessor.target): config_payload,
                },
            }
        else:
            config_payload = {"default_target": preprocessor.target, "preprocessors": {}}
    config_payload["preprocessors"][preprocessor.target] = preprocessor.export_config()
    write_json(config_path, config_payload)

    threshold_path = Path("models/artifacts/thresholds.json")
    threshold_payload = {}
    if threshold_path.exists():
        threshold_payload = json.loads(threshold_path.read_text(encoding="utf-8"))
    task = "mortality" if "mortality" in model_name else "deterioration"
    threshold_payload[task] = thresholds
    write_json(threshold_path, threshold_payload)

    metadata_path = Path("models/artifacts/model_metadata.json")
    metadata_payload = {}
    if metadata_path.exists():
        metadata_payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata_payload[task] = {
        "model_name": model_name,
        "task": task,
        "trained_at": pd.Timestamp.now().isoformat(),
        "model_type": type(model).__name__,
        "n_features": len(feature_names),
        "feature_names": feature_names,
        "performance": {"validation": train_results, "test": test_results, "comparison": both_results},
        "thresholds": thresholds,
        "recommended_threshold": thresholds["max_f1"],
        "clinical_thresholds": {
            "screening": thresholds["max_sensitivity_90"],
            "treatment_decision": thresholds["max_specificity_90"],
        },
        "preprocessor_target": preprocessor.target,
    }
    write_json(metadata_path, metadata_payload)
    log.info("Saved artifacts for %s", model_name)
