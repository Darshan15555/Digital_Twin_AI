from __future__ import annotations

import argparse
import logging
from datetime import datetime
from pathlib import Path

import joblib

from baseline_training.bootstrap_artifact import ensure_dataset
from baseline_training.evaluate import evaluate_model, select_best_model
from baseline_training.model import train_baselines
from baseline_training.utils import (
    FEATURE_COLS,
    PATIENT_COL,
    TARGET_COL,
    apply_imputer,
    ensure_dir,
    fit_imputer,
    load_dataset,
    prepare_features_and_target,
    set_global_seed,
    split_dataset,
    split_integrity_report,
    validate_dataset_columns,
    write_json,
)


log = logging.getLogger("baseline_training")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train baseline LightGBM/XGBoost on HR/SpO2/BP dataset.")
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=Path("data/parquet/minimal_hr_spo2_bp.parquet"),
        help="Input dataset path (.parquet or .csv)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/baseline_training"),
        help="Directory to store model + metrics artifacts",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-size", type=float, default=0.70)
    parser.add_argument("--val-size", type=float, default=0.15)
    parser.add_argument("--test-size", type=float, default=0.15)
    parser.add_argument(
        "--row-level-split",
        action="store_true",
        help="Disable patient-level split and split by rows",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    args = parse_args()

    set_global_seed(args.seed)
    ensure_dir(args.output_dir)

    if not args.dataset_path.exists():
        args.dataset_path = ensure_dataset(args.dataset_path, seed=args.seed)
        log.info("Dataset was missing. Auto-generated fallback dataset at %s", args.dataset_path)

    df = load_dataset(args.dataset_path)
    validate_dataset_columns(df)
    log.info("Loaded dataset shape=%s", df.shape)

    train_df, val_df, test_df = split_dataset(
        df,
        train_size=args.train_size,
        val_size=args.val_size,
        test_size=args.test_size,
        random_state=args.seed,
        patient_level=not args.row_level_split,
    )
    split_report = split_integrity_report(train_df, val_df, test_df)
    log.info("Split report: %s", split_report)

    X_train_raw, y_train = prepare_features_and_target(train_df)
    X_val_raw, y_val = prepare_features_and_target(val_df)
    X_test_raw, y_test = prepare_features_and_target(test_df)

    imputer = fit_imputer(X_train_raw)
    X_train = apply_imputer(imputer, X_train_raw)
    X_val = apply_imputer(imputer, X_val_raw)
    X_test = apply_imputer(imputer, X_test_raw)

    models = train_baselines(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        random_state=args.seed,
    )

    reports: dict[str, dict] = {}
    for name, model in models.items():
        reports[name] = evaluate_model(model, X_val, y_val, X_test, y_test)
        log.info(
            "%s | val_auroc=%.4f val_f1=%.4f test_auroc=%.4f test_f1=%.4f",
            name,
            reports[name]["validation"]["auroc"],
            reports[name]["validation"]["best_f1_value"],
            reports[name]["test"]["auroc"],
            reports[name]["test"]["best_f1_threshold_from_val"]["f1"],
        )

    best_model_name = select_best_model(reports)
    best_model = models[best_model_name]
    best_threshold = reports[best_model_name]["validation"]["best_f1"]["threshold"]

    artifact_bundle = {
        "model_name": best_model_name,
        "trained_at": datetime.utcnow().isoformat(),
        "features": FEATURE_COLS,
        "patient_column": PATIENT_COL,
        "target_column": TARGET_COL,
        "threshold": float(best_threshold),
        "imputer_medians": {
            col: float(val)
            for col, val in zip(FEATURE_COLS, imputer.statistics_)
        },
        "model": best_model,
    }

    model_path = args.output_dir / "best_model.joblib"
    feature_path = args.output_dir / "feature_list.json"
    metrics_path = args.output_dir / "metrics.json"

    joblib.dump(artifact_bundle, model_path)
    write_json(feature_path, {"features": FEATURE_COLS, "count": len(FEATURE_COLS)})
    write_json(
        metrics_path,
        {
            "best_model": best_model_name,
            "split_report": split_report,
            "reports": reports,
            "selected_threshold": float(best_threshold),
            "dataset_path": str(args.dataset_path),
        },
    )

    log.info("Saved best model: %s", model_path)
    log.info("Saved feature list: %s", feature_path)
    log.info("Saved metrics: %s", metrics_path)


if __name__ == "__main__":
    main()
