from __future__ import annotations

import sys

import pandas as pd

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.common import setup_logger
from models.evaluation.evaluator import evaluate_model, save_all_artifacts, tune_threshold
from models.preprocessing.build_preprocessor import ICUPreprocessor

log = setup_logger("models.train_mortality", "models/training_mortality.log")


def _load_splits() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return (
        pd.read_parquet("data/ts_train.parquet"),
        pd.read_parquet("data/ts_val.parquet"),
        pd.read_parquet("data/ts_test.parquet"),
    )


def train_mortality_models(target: str = "label_hospital_mortality") -> dict:
    import lightgbm as lgb
    import xgboost as xgb

    train, val, test = _load_splits()
    preprocessor = ICUPreprocessor(target=target, exclude_leaky=True)
    X_train, y_train, X_val, y_val, X_test, y_test, train_df, val_df, test_df = preprocessor.fit_transform(train, val, test)
    log.info("Train %s pos_rate=%.3f", X_train.shape, y_train.mean())
    log.info("Val   %s pos_rate=%.3f", X_val.shape, y_val.mean())
    log.info("Test  %s pos_rate=%.3f", X_test.shape, y_test.mean())

    neg = int((y_train == 0).sum())
    pos = int((y_train == 1).sum())
    scale_pos_weight = neg / max(pos, 1)
    log.info("Class ratio neg/pos: %.2f", scale_pos_weight)

    lgb_model = lgb.LGBMClassifier(
        objective="binary",
        metric=["binary_logloss", "auc"],
        boosting_type="gbdt",
        n_estimators=1000,
        learning_rate=0.05,
        num_leaves=63,
        max_depth=-1,
        min_child_samples=50,
        feature_fraction=0.8,
        bagging_fraction=0.8,
        bagging_freq=5,
        scale_pos_weight=scale_pos_weight,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )
    lgb_model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False), lgb.log_evaluation(period=100)],
    )

    xgb_model = xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric=["logloss", "auc"],
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=6,
        min_child_weight=10,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        gamma=1.0,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
        tree_method="hist",
        device="cpu",
    )
    xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=100)

    lgb_val_results = evaluate_model(lgb_model, X_val, y_val, val_df[["patientunitstayid", "window_id"]], model_name="lightgbm_mortality", split="validation")
    xgb_val_results = evaluate_model(xgb_model, X_val, y_val, val_df[["patientunitstayid", "window_id"]], model_name="xgboost_mortality", split="validation")
    if lgb_val_results["auroc"] >= xgb_val_results["auroc"]:
        best_model = lgb_model
        best_name = "lgbm"
        best_val_results = lgb_val_results
    else:
        best_model = xgb_model
        best_name = "xgb"
        best_val_results = xgb_val_results
    log.info("Selected best model: %s", best_name)

    val_proba = best_model.predict_proba(X_val)[:, 1]
    thresholds = tune_threshold(y_val, val_proba, val_df["patientunitstayid"])
    test_results = evaluate_model(
        best_model,
        X_test,
        y_test,
        test_df[["patientunitstayid", "window_id", target]],
        model_name=f"{best_name}_mortality",
        split="test",
        threshold=thresholds["max_f1"],
    )

    save_all_artifacts(
        model=best_model,
        model_name=f"mortality_{best_name}",
        preprocessor=preprocessor,
        thresholds=thresholds,
        feature_names=preprocessor.feature_names_,
        train_results=best_val_results,
        test_results=test_results,
        both_results={"lightgbm": lgb_val_results, "xgboost": xgb_val_results},
    )
    log.info("Final mortality test AUROC=%.4f AUPRC=%.4f F1=%.4f", test_results["auroc"], test_results["auprc"], test_results["f1"])
    return {"best_model": best_name, "validation": best_val_results, "test": test_results, "thresholds": thresholds}


if __name__ == "__main__":
    train_mortality_models()
