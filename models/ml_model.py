"""
Mortality, LOS, and sepsis model utilities.
"""

from __future__ import annotations

import json
import logging
import pickle

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from config.config import (
    EARLY_MORTALITY_THRESHOLD,
    FEATURE_NAMES_PATH,
    LOS_MODEL_PATH,
    MODEL_METRICS_PATH,
    MORTALITY_BASELINE_MODEL_PATH,
    MORTALITY_MODEL_PATH,
    MORTALITY_SCALER_PATH,
    MORTALITY_XGB_MODEL_PATH,
    POSTGRES_EARLY_FEATURE_NAMES_PATH,
    POSTGRES_EARLY_IMPUTER_PATH,
    POSTGRES_EARLY_MODEL_PATH,
)

log = logging.getLogger(__name__)

MODEL_FEATURES = [
    "age",
    "gender_enc",
    "admissionweight",
    "icu_los_hours",
    "apachescore",
    "predictedhospitalmortality",
    "heartrate_mean",
    "heartrate_max",
    "heartrate_min",
    "heartrate_std",
    "sao2_mean",
    "sao2_min",
    "resp_mean",
    "resp_max",
    "sbp_mean",
    "sbp_min",
    "temp_mean",
    "temp_max",
    "cvp_mean",
    "nibp_systolic_mean",
    "nibp_mean_mean",
    "creatinine_max",
    "glucose_max",
    "glucose_min",
    "lactate_max",
    "hemoglobin_min",
    "wbc_max",
    "potassium_min",
    "potassium_max",
    "bicarbonate_min",
    "bun_max",
    "gcs_min",
    "on_vasopressor",
    "vasopressor_duration_hours",
    "fluid_balance_24h",
    "urine_output_per_hour",
    "on_ventilator",
    "peep_mean",
    "fio2_mean",
    "has_diabetes",
    "has_chf",
    "has_copd",
    "has_ckd",
    "has_hypertension",
    "has_immunosuppression",
    "qsofa_score",
    "sofa_approx_score",
]

EARLY_HIGH_RISK_THRESHOLD = 0.70


def model_is_trained() -> bool:
    return (
        MORTALITY_MODEL_PATH.exists()
        and MORTALITY_SCALER_PATH.exists()
        and FEATURE_NAMES_PATH.exists()
        and LOS_MODEL_PATH.exists()
    )


def _prepare_features(df: pd.DataFrame) -> tuple[pd.DataFrame, SimpleImputer, list[str]]:
    available = [col for col in MODEL_FEATURES if col in df.columns]
    X = df[available].copy()
    imputer = SimpleImputer(strategy="median")
    X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=available, index=X.index)
    return X_imputed, imputer, available


def train_models(df: pd.DataFrame) -> dict:
    df = df.copy()
    df = df.dropna(subset=["hospital_mortality", "icu_los_hours"])

    X, imputer, feature_names = _prepare_features(df)
    y_cls = df["hospital_mortality"].astype(int)
    y_los = df["icu_los_hours"].astype(float)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_cls, test_size=0.2, random_state=42, stratify=y_cls
    )

    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    rf_proba = rf.predict_proba(X_test)[:, 1]
    rf_pred = (rf_proba >= 0.5).astype(int)

    baseline = LogisticRegression(max_iter=1000, class_weight="balanced")
    baseline.fit(X_train, y_train)

    xgb_model = None
    try:
        from xgboost import XGBClassifier

        xgb_model = XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
            random_state=42,
        )
        xgb_model.fit(X_train, y_train)
    except Exception as exc:
        log.warning("XGBoost unavailable: %s", exc)

    X_los_train, X_los_test, y_los_train, y_los_test = train_test_split(
        X, y_los, test_size=0.2, random_state=42
    )
    los_model = RandomForestRegressor(
        n_estimators=200,
        max_depth=10,
        random_state=42,
        n_jobs=-1,
    )
    los_model.fit(X_los_train, y_los_train)
    los_pred = los_model.predict(X_los_test)

    metrics = {
        "mortality_rf": {
            "auc_roc": float(roc_auc_score(y_test, rf_proba)),
            "auprc": float(average_precision_score(y_test, rf_proba)),
            "f1": float(f1_score(y_test, rf_pred)),
            "recall": float(recall_score(y_test, rf_pred)),
        },
        "los_rf": {
            "mae": float(mean_absolute_error(y_los_test, los_pred)),
            "rmse": float(np.sqrt(mean_squared_error(y_los_test, los_pred))),
        },
    }

    with open(MORTALITY_MODEL_PATH, "wb") as handle:
        pickle.dump(rf, handle)
    with open(MORTALITY_BASELINE_MODEL_PATH, "wb") as handle:
        pickle.dump(baseline, handle)
    if xgb_model is not None:
        with open(MORTALITY_XGB_MODEL_PATH, "wb") as handle:
            pickle.dump(xgb_model, handle)
    with open(MORTALITY_SCALER_PATH, "wb") as handle:
        pickle.dump(imputer, handle)
    with open(LOS_MODEL_PATH, "wb") as handle:
        pickle.dump(los_model, handle)
    with open(FEATURE_NAMES_PATH, "wb") as handle:
        pickle.dump(feature_names, handle)
    MODEL_METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    return metrics


def _load_pickle(path):
    with open(path, "rb") as handle:
        return pickle.load(handle)


def load_models():
    rf = _load_pickle(MORTALITY_MODEL_PATH)
    imputer = _load_pickle(MORTALITY_SCALER_PATH)
    los = _load_pickle(LOS_MODEL_PATH)
    features = _load_pickle(FEATURE_NAMES_PATH)
    return rf, imputer, los, features


def _prepare_row(row: dict, imputer: SimpleImputer, feature_names: list[str]) -> pd.DataFrame:
    df = pd.DataFrame([row])
    for column in feature_names:
        if column not in df.columns:
            df[column] = np.nan
    transformed = pd.DataFrame(imputer.transform(df[feature_names]), columns=feature_names)
    return transformed


def predict_mortality(row: dict) -> dict:
    rf, imputer, _, features = load_models()
    X = _prepare_row(row, imputer, features)
    score = float(rf.predict_proba(X)[0, 1])
    pred = int(score >= 0.5)
    importances = sorted(zip(features, rf.feature_importances_), key=lambda item: item[1], reverse=True)[:5]
    return {
        "risk_score": round(score, 4),
        "prediction": pred,
        "risk_label": "HIGH" if score >= 0.7 else "MODERATE" if score >= 0.4 else "LOW",
        "top_features": [(name, round(float(value), 4)) for name, value in importances],
    }


def early_model_is_trained() -> bool:
    return (
        POSTGRES_EARLY_MODEL_PATH.exists()
        and POSTGRES_EARLY_IMPUTER_PATH.exists()
        and POSTGRES_EARLY_FEATURE_NAMES_PATH.exists()
    )


def load_early_mortality_model():
    model = _load_pickle(POSTGRES_EARLY_MODEL_PATH)
    imputer = _load_pickle(POSTGRES_EARLY_IMPUTER_PATH)
    features = _load_pickle(POSTGRES_EARLY_FEATURE_NAMES_PATH)
    return model, imputer, features


def _build_early_operating_mode(threshold: float) -> str:
    if abs(threshold - 0.60) < 1e-9:
        return "balanced_0.60"
    if abs(threshold - 0.40) < 1e-9:
        return "high_recall_0.40"
    if abs(threshold - 0.65) < 1e-9:
        return "higher_precision_0.65"
    return f"custom_{threshold:.2f}"


def _early_recommended_action(score: float, threshold: float) -> str:
    if score >= EARLY_HIGH_RISK_THRESHOLD:
        return "urgent review"
    if score >= threshold:
        return "clinical review recommended"
    if score >= max(0.40, threshold - 0.20):
        return "monitor closely"
    return "routine monitoring"


def predict_early_mortality(row: dict, threshold: float | None = None) -> dict:
    model, imputer, features = load_early_mortality_model()
    active_threshold = EARLY_MORTALITY_THRESHOLD if threshold is None else threshold
    X = _prepare_row(row, imputer, features)
    score = float(model.predict_proba(X)[0, 1])
    pred = int(score >= active_threshold)
    risk_label = "HIGH" if score >= EARLY_HIGH_RISK_THRESHOLD else "MODERATE" if score >= active_threshold else "LOW"
    importances = sorted(zip(features, model.feature_importances_), key=lambda item: item[1], reverse=True)[:5]
    return {
        "risk_score": round(score, 4),
        "prediction": pred,
        "threshold": round(float(active_threshold), 2),
        "risk_label": risk_label,
        "alert": bool(pred),
        "operating_mode": _build_early_operating_mode(float(active_threshold)),
        "recommended_action": _early_recommended_action(score, float(active_threshold)),
        "top_features": [(name, round(float(value), 4)) for name, value in importances],
        "model_variant": "early_random_forest",
    }


def predict_los(row: dict) -> dict:
    _, imputer, los_model, features = load_models()
    X = _prepare_row(row, imputer, features)
    hours = float(los_model.predict(X)[0])
    return {
        "predicted_icu_los_hours": round(hours, 2),
        "predicted_icu_los_days": round(hours / 24.0, 2),
    }


def predict_sepsis(row: dict) -> dict:
    resp = float(row.get("resp_mean") or 0)
    sbp = float(row.get("sbp_min") or row.get("sbp_mean") or 999)
    gcs = float(row.get("gcs_min") or 15)
    lactate = float(row.get("lactate_max") or 0)
    creatinine = float(row.get("creatinine_max") or 0)
    sao2 = float(row.get("sao2_min") or 100)

    qsofa = int(resp >= 22) + int(sbp <= 100) + int(gcs < 15)
    sofa_approx = qsofa + int(lactate > 2) + int(creatinine > 2) + int(sao2 < 92)
    sepsis_risk = int(qsofa >= 2 or sofa_approx >= 4)
    return {
        "qsofa_score": qsofa,
        "sofa_approx_score": sofa_approx,
        "sepsis_risk": sepsis_risk,
        "interpretation": "High sepsis risk" if sepsis_risk else "Lower sepsis risk",
    }
