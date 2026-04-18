from __future__ import annotations

import lightgbm as lgb
import xgboost as xgb
from sklearn.base import ClassifierMixin


def class_weight_ratio(y_train) -> float:
    positives = int((y_train == 1).sum())
    negatives = int((y_train == 0).sum())
    return negatives / max(positives, 1)


def build_lightgbm(*, scale_pos_weight: float, random_state: int) -> lgb.LGBMClassifier:
    return lgb.LGBMClassifier(
        objective="binary",
        n_estimators=180,
        learning_rate=0.05,
        num_leaves=31,
        max_depth=-1,
        min_child_samples=50,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.0,
        reg_lambda=1.0,
        scale_pos_weight=scale_pos_weight,
        random_state=random_state,
        n_jobs=-1,
        verbose=-1,
    )


def build_xgboost(*, scale_pos_weight: float, random_state: int) -> xgb.XGBClassifier:
    return xgb.XGBClassifier(
        objective="binary:logistic",
        n_estimators=180,
        learning_rate=0.05,
        max_depth=6,
        min_child_weight=5,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.0,
        reg_lambda=1.0,
        gamma=0.0,
        scale_pos_weight=scale_pos_weight,
        random_state=random_state,
        n_jobs=-1,
        tree_method="hist",
        eval_metric="auc",
    )


def train_lightgbm(
    model: lgb.LGBMClassifier,
    X_train,
    y_train,
    X_val,
    y_val,
) -> lgb.LGBMClassifier:
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(period=0),
        ],
    )
    return model


def train_xgboost(
    model: xgb.XGBClassifier,
    X_train,
    y_train,
    X_val,
    y_val,
) -> xgb.XGBClassifier:
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )
    return model


def train_baselines(
    *,
    X_train,
    y_train,
    X_val,
    y_val,
    random_state: int,
) -> dict[str, ClassifierMixin]:
    spw = class_weight_ratio(y_train)
    lgbm = build_lightgbm(scale_pos_weight=spw, random_state=random_state)
    xgbm = build_xgboost(scale_pos_weight=spw, random_state=random_state)

    trained_lgbm = train_lightgbm(lgbm, X_train, y_train, X_val, y_val)
    trained_xgbm = train_xgboost(xgbm, X_train, y_train, X_val, y_val)
    return {
        "lightgbm": trained_lgbm,
        "xgboost": trained_xgbm,
    }
