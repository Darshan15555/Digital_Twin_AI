from __future__ import annotations

import argparse
import logging
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from baseline_training.analysis import (
    conceptual_comparison,
    extract_feature_importance,
    measure_inference_latency,
    model_size_mb,
    optimization_suggestions,
    run_robustness_checks,
    summarize_feature_importance,
)
from baseline_training.bootstrap_artifact import ensure_artifact, ensure_dataset
from baseline_training.threshold import evaluate_threshold_options, metrics_at_threshold, optimize_thresholds
from baseline_training.utils import (
    FEATURE_COLS,
    PATIENT_COL,
    TARGET_COL,
    ensure_dir,
    load_dataset,
    set_global_seed,
    split_dataset,
    split_integrity_report,
    write_json,
)


log = logging.getLogger("baseline_optimize")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Optimize and validate baseline model on HR/SpO2/BP features.")
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=Path("data/parquet/minimal_hr_spo2_bp.parquet"),
        help="Evaluation dataset path (.parquet or .csv).",
    )
    parser.add_argument(
        "--artifact-path",
        type=Path,
        default=Path("artifacts/baseline_training/best_model.joblib"),
        help="Trained model artifact path from baseline_training/train.py.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/baseline_training"),
        help="Directory to write thresholds.json, evaluation_report.json, feature_importance.csv.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-size", type=float, default=0.70)
    parser.add_argument("--val-size", type=float, default=0.15)
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument("--sensitivity-target", type=float, default=0.90)
    parser.add_argument(
        "--row-level-split",
        action="store_true",
        help="Disable patient-level splitting and split rows directly.",
    )
    parser.add_argument("--latency-runs", type=int, default=200)
    return parser.parse_args()


def _validate_columns(df: pd.DataFrame, feature_cols: list[str]) -> None:
    required = {PATIENT_COL, TARGET_COL, *feature_cols}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Dataset missing required columns for optimization: {missing}")


def _load_artifact(artifact_path: Path) -> tuple[object, str, list[str], dict[str, float]]:
    if not artifact_path.exists():
        raise FileNotFoundError(f"Model artifact not found: {artifact_path}")

    bundle = joblib.load(artifact_path)
    if isinstance(bundle, dict) and "model" in bundle:
        model = bundle["model"]
        model_name = str(bundle.get("model_name", type(model).__name__))
        features = bundle.get("features") or FEATURE_COLS
        medians = bundle.get("imputer_medians") or {}
        features = [str(col) for col in features]
        medians_map = {str(k): float(v) for k, v in medians.items()}
        return model, model_name, features, medians_map

    model = bundle
    return model, type(model).__name__, FEATURE_COLS.copy(), {}


def _prepare_xy(df: pd.DataFrame, feature_cols: list[str]) -> tuple[pd.DataFrame, pd.Series]:
    X = df[feature_cols].copy()
    y = pd.to_numeric(df[TARGET_COL], errors="coerce").fillna(0).astype("int8")
    return X, y


def _derive_fill_values(X_train_raw: pd.DataFrame, feature_cols: list[str], artifact_medians: dict[str, float]) -> dict[str, float]:
    fill_values: dict[str, float] = {}
    for col in feature_cols:
        value = artifact_medians.get(col)
        if value is not None and np.isfinite(value):
            fill_values[col] = float(value)
            continue

        numeric = pd.to_numeric(X_train_raw[col], errors="coerce")
        median = float(numeric.median()) if numeric.notna().any() else 0.0
        fill_values[col] = median
    return fill_values


def _apply_medians(X_raw: pd.DataFrame, feature_cols: list[str], fill_values: dict[str, float]) -> pd.DataFrame:
    X = X_raw[feature_cols].copy()
    for col in feature_cols:
        X[col] = pd.to_numeric(X[col], errors="coerce").fillna(float(fill_values[col]))
    return X.astype(float)


def _auroc(y_true, y_proba) -> float:
    y = np.asarray(y_true)
    if np.unique(y).shape[0] < 2:
        return float("nan")
    return float(roc_auc_score(y, y_proba))


def _split_metrics(y_true, y_proba, threshold: float = 0.5) -> dict[str, float]:
    payload = metrics_at_threshold(y_true, y_proba, threshold)
    payload["auroc"] = _auroc(y_true, y_proba)
    return payload


def _class_balance(y: pd.Series) -> dict[str, float | int]:
    positives = int((y == 1).sum())
    negatives = int((y == 0).sum())
    total = int(len(y))
    return {
        "rows": total,
        "positive_count": positives,
        "negative_count": negatives,
        "positive_rate": float(positives / max(total, 1)),
        "negative_to_positive_ratio": float(negatives / max(positives, 1)),
    }


def _overfit_report(train_metrics: dict[str, float], test_metrics: dict[str, float]) -> dict[str, float | bool]:
    auroc_gap = float(train_metrics["auroc"] - test_metrics["auroc"]) if np.isfinite(train_metrics["auroc"]) and np.isfinite(test_metrics["auroc"]) else float("nan")
    f1_gap = float(train_metrics["f1"] - test_metrics["f1"])
    return {
        "auroc_gap_train_minus_test": auroc_gap,
        "f1_gap_train_minus_test": f1_gap,
        "potential_overfit_auroc": bool(np.isfinite(auroc_gap) and auroc_gap > 0.05),
        "potential_overfit_f1": bool(f1_gap > 0.05),
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    args = parse_args()
    set_global_seed(args.seed)
    ensure_dir(args.output_dir)

    if not args.dataset_path.exists():
        args.dataset_path = ensure_dataset(args.dataset_path, seed=args.seed)
        log.info("Dataset was missing. Auto-generated fallback dataset at %s", args.dataset_path)
    if not args.artifact_path.exists():
        args.artifact_path = ensure_artifact(args.artifact_path, args.dataset_path, seed=args.seed)
        log.info("Model artifact was missing. Auto-generated fallback artifact at %s", args.artifact_path)

    model, model_name, feature_cols, artifact_medians = _load_artifact(args.artifact_path)
    df = load_dataset(args.dataset_path)
    _validate_columns(df, feature_cols)
    log.info("Loaded dataset rows=%d cols=%d | model=%s", df.shape[0], df.shape[1], model_name)

    train_df, val_df, test_df = split_dataset(
        df,
        train_size=args.train_size,
        val_size=args.val_size,
        test_size=args.test_size,
        random_state=args.seed,
        patient_level=not args.row_level_split,
    )
    split_report = split_integrity_report(train_df, val_df, test_df)

    X_train_raw, y_train = _prepare_xy(train_df, feature_cols)
    X_val_raw, y_val = _prepare_xy(val_df, feature_cols)
    X_test_raw, y_test = _prepare_xy(test_df, feature_cols)

    fill_values = _derive_fill_values(X_train_raw, feature_cols, artifact_medians)
    X_train = _apply_medians(X_train_raw, feature_cols, fill_values)
    X_val = _apply_medians(X_val_raw, feature_cols, fill_values)
    X_test = _apply_medians(X_test_raw, feature_cols, fill_values)

    train_proba = model.predict_proba(X_train)[:, 1]
    val_proba = model.predict_proba(X_val)[:, 1]
    test_proba = model.predict_proba(X_test)[:, 1]

    train_default = _split_metrics(y_train, train_proba, threshold=0.5)
    val_default = _split_metrics(y_val, val_proba, threshold=0.5)
    test_default = _split_metrics(y_test, test_proba, threshold=0.5)

    threshold_options = optimize_thresholds(
        y_val,
        val_proba,
        sensitivity_target=args.sensitivity_target,
        step=0.01,
    )
    threshold_eval_test = evaluate_threshold_options(y_test, test_proba, threshold_options)
    selected_threshold = float(threshold_options["max_f1"]["threshold"])

    feature_importance_df = extract_feature_importance(model, feature_cols)
    feature_importance_path = args.output_dir / "feature_importance.csv"
    feature_importance_df.to_csv(feature_importance_path, index=False)
    feature_importance_summary = summarize_feature_importance(feature_importance_df)

    latency_report = measure_inference_latency(model, X_test, runs=args.latency_runs)
    artifact_size = model_size_mb(args.artifact_path)
    perf_suggestions = optimization_suggestions(model, latency_report, artifact_size)

    transform_fn = lambda raw_df: _apply_medians(raw_df, feature_cols, fill_values)
    robustness = run_robustness_checks(
        model,
        feature_cols=feature_cols,
        medians=fill_values,
        transform_fn=transform_fn,
        threshold=selected_threshold,
    )

    thresholds_payload = {
        "model_name": model_name,
        "selected_threshold": selected_threshold,
        "validation_threshold_options": threshold_options,
        "test_metrics_at_options": threshold_eval_test,
    }
    write_json(args.output_dir / "thresholds.json", thresholds_payload)

    report = {
        "run_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_path": str(args.dataset_path),
        "artifact_path": str(args.artifact_path),
        "model_name": model_name,
        "features": feature_cols,
        "split_report": split_report,
        "class_balance": {
            "train": _class_balance(y_train),
            "validation": _class_balance(y_val),
            "test": _class_balance(y_test),
        },
        "metrics_default_threshold_0_5": {
            "train": train_default,
            "validation": val_default,
            "test": test_default,
        },
        "overfitting_check": _overfit_report(train_default, test_default),
        "threshold_tuning": threshold_options,
        "feature_importance_summary": feature_importance_summary,
        "performance": {
            "single_row_latency_ms": latency_report,
            "artifact_size_mb": artifact_size,
            "suggestions": perf_suggestions,
        },
        "robustness_checks": robustness,
        "comparison_insights": conceptual_comparison(),
    }
    write_json(args.output_dir / "evaluation_report.json", report)

    log.info("Wrote: %s", args.output_dir / "thresholds.json")
    log.info("Wrote: %s", args.output_dir / "evaluation_report.json")
    log.info("Wrote: %s", feature_importance_path)


if __name__ == "__main__":
    main()
