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

log = setup_logger("models.train_deterioration", "models/training_deterioration.log")


def train_deterioration_models() -> dict:
    import lightgbm as lgb

    train = pd.read_parquet("data/ts_train.parquet")
    val = pd.read_parquet("data/ts_val.parquet")
    test = pd.read_parquet("data/ts_test.parquet")
    train = train[train["window_id"] < 5].copy()
    val = val[val["window_id"] < 5].copy()
    test = test[test["window_id"] < 5].copy()

    preprocessor = ICUPreprocessor(target="label_deterioration_next", exclude_leaky=True)
    X_train, y_train, X_val, y_val, X_test, y_test, train_df, val_df, test_df = preprocessor.fit_transform(train, val, test)
    neg = int((y_train == 0).sum())
    pos = int((y_train == 1).sum())
    scale_pos_weight = neg / max(pos, 1)

    model = lgb.LGBMClassifier(
        objective="binary",
        metric=["binary_logloss", "auc"],
        boosting_type="gbdt",
        n_estimators=1000,
        learning_rate=0.05,
        num_leaves=31,
        max_depth=-1,
        min_child_samples=100,
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
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False), lgb.log_evaluation(period=100)],
    )
    val_results = evaluate_model(model, X_val, y_val, val_df[["patientunitstayid", "window_id"]], model_name="lightgbm_deterioration", split="validation")
    thresholds = tune_threshold(y_val, model.predict_proba(X_val)[:, 1], val_df["patientunitstayid"])
    test_results = evaluate_model(
        model,
        X_test,
        y_test,
        test_df[["patientunitstayid", "window_id", "label_deterioration_next"]],
        model_name="lightgbm_deterioration",
        split="test",
        threshold=thresholds["max_f1"],
    )
    save_all_artifacts(
        model=model,
        model_name="deterioration_lgbm",
        preprocessor=preprocessor,
        thresholds=thresholds,
        feature_names=preprocessor.feature_names_,
        train_results=val_results,
        test_results=test_results,
        both_results={"lightgbm": val_results},
    )
    return {"validation": val_results, "test": test_results, "thresholds": thresholds}


if __name__ == "__main__":
    train_deterioration_models()
