"""
ml_model.py
ICU mortality prediction model.
Uses Random Forest for good performance without deep learning.
Model is trained once and saved — reloaded on subsequent runs.
"""

import pandas as pd
import numpy as np
import pickle
import logging
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from config.config import MORTALITY_MODEL_PATH, SCALER_PATH, FEATURES_PATH

log = logging.getLogger(__name__)

FEATURE_COLS = [
    "age", "admissionheight", "admissionweight",
    "apachescore", "predictedhospitalmortality",
    "hr_mean", "hr_std", "sao2_mean", "sao2_min",
    "resp_mean", "sbp_mean", "dbp_mean", "temp_mean",
    "gender_enc", "unittype_enc"
]

TARGET_COL = "hospital_mortality"


def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["gender_enc"] = (df["gender"].str.lower() == "male").astype(float)
    unit_map = {v: i for i, v in enumerate(df["unittype"].fillna("unknown").unique())}
    df["unittype_enc"] = df["unittype"].fillna("unknown").map(unit_map).fillna(0)
    return df


def prepare_X_y(df: pd.DataFrame):
    df = encode_features(df)
    available = [c for c in FEATURE_COLS if c in df.columns]
    X = df[available].copy()
    X = X.fillna(X.median(numeric_only=True))
    y = df[TARGET_COL].fillna(0).astype(int) if TARGET_COL in df.columns else None
    return X, y, available


def train_model(ml_df: pd.DataFrame):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_auc_score, classification_report

    log.info("Training mortality prediction model...")

    df = ml_df.dropna(subset=[TARGET_COL])
    X, y, feature_names = prepare_X_y(df)

    log.info(f"Dataset: {X.shape}, positive rate: {y.mean():.3f}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # Random Forest (no need to scale, but we keep scaler for LR)
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=8,
        min_samples_leaf=10,
        n_jobs=-1,
        random_state=42,
        class_weight="balanced"
    )
    rf.fit(X_train, y_train)

    y_pred_proba = rf.predict_proba(X_test)[:, 1]
    y_pred = rf.predict(X_test)
    auc = roc_auc_score(y_test, y_pred_proba)
    log.info(f"Random Forest AUC: {auc:.4f}")
    log.info("\n" + classification_report(y_test, y_pred))

    # Save model + scaler + feature list
    with open(MORTALITY_MODEL_PATH, "wb") as f:
        pickle.dump(rf, f)
    with open(SCALER_PATH, "wb") as f:
        pickle.dump(scaler, f)
    with open(FEATURES_PATH, "wb") as f:
        pickle.dump(feature_names, f)

    log.info(f"Model saved → {MORTALITY_MODEL_PATH}")
    return rf, scaler, feature_names, auc


def load_model():
    with open(MORTALITY_MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)
    with open(FEATURES_PATH, "rb") as f:
        feature_names = pickle.load(f)
    return model, scaler, feature_names


def model_is_trained() -> bool:
    return MORTALITY_MODEL_PATH.exists() and FEATURES_PATH.exists()


def predict_patient(patient_row: dict) -> dict:
    """Predict mortality risk for a single patient dict."""
    model, scaler, feature_names = load_model()

    row_df = pd.DataFrame([patient_row])
    row_df = encode_features(row_df)

    # Fill missing features with 0
    for col in feature_names:
        if col not in row_df.columns:
            row_df[col] = 0.0

    X = row_df[feature_names].fillna(0)

    risk_score = float(model.predict_proba(X)[0][1])
    prediction = int(model.predict(X)[0])

    if risk_score >= 0.7:
        risk_label = "HIGH"
    elif risk_score >= 0.4:
        risk_label = "MODERATE"
    else:
        risk_label = "LOW"

    # Feature importances
    importances = dict(zip(feature_names, model.feature_importances_))
    top_features = sorted(importances.items(), key=lambda x: -x[1])[:5]

    return {
        "risk_score": round(risk_score, 4),
        "risk_label": risk_label,
        "prediction": prediction,
        "top_features": top_features
    }


def digital_twin_forecast(vitals_df: pd.DataFrame, steps: int = 6) -> dict:
    """
    Simple digital twin: forecast next N vital readings using rolling average.
    Returns predicted values for each vital column.
    """
    vital_cols = ["heartrate", "respiration", "sao2", "systemicsystolic", "temperature"]
    result = {}

    for col in vital_cols:
        series = vitals_df[col].dropna()
        if len(series) < 3:
            result[col] = []
            continue

        # Use exponentially weighted moving average for forecast
        ewm_val = series.ewm(span=5).mean().iloc[-1]
        std_val = series.tail(10).std()
        if pd.isna(std_val):
            std_val = 0

        # Simple forecast: mean ± slight trend
        last_vals = series.tail(5).values
        trend = (last_vals[-1] - last_vals[0]) / max(len(last_vals) - 1, 1)

        forecast = []
        current = ewm_val
        for i in range(1, steps + 1):
            noise = np.random.normal(0, std_val * 0.2)
            predicted = current + (trend * 0.3) + noise
            forecast.append(round(float(predicted), 2))
            current = predicted

        result[col] = forecast

    return result


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # Quick test with dummy data
    if model_is_trained():
        result = predict_patient({
            "age": 65, "apachescore": 50, "hr_mean": 90,
            "sao2_mean": 94, "sbp_mean": 110, "gender": "Male"
        })
        print("Prediction:", result)
    else:
        print("Model not yet trained. Run setup.py first.")
