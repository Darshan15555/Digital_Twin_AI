from __future__ import annotations

import logging

import pandas as pd

from backend.services.patient_service import get_patient_by_id
from config.settings import settings
from models.digital_twin import deterioration_score, forecast_vitals, simulate_what_if
from models.ml_model import _prepare_row, early_model_is_trained, load_early_mortality_model, load_models, load_postgres_full_mortality_model, model_is_trained, postgres_full_model_is_trained, predict_sepsis as model_predict_sepsis

log = logging.getLogger(__name__)


def predict_mortality(pid: int) -> dict:
    not_ready = {"risk_score": 0.0, "risk_label": "UNKNOWN", "prediction": 0, "top_features": [], "interpretation": "ML model not trained yet. Run training pipeline first.", "model_ready": False}
    try:
        patient = get_patient_by_id(pid)
        if patient is None:
            return not_ready

        if model_is_trained():
            model, imputer, _, features = load_models()
        elif postgres_full_model_is_trained():
            model, imputer, features = load_postgres_full_mortality_model()
        else:
            return not_ready

        if hasattr(model, "n_jobs"):
            model.n_jobs = 1
        X = _prepare_row(patient, imputer, features)
        score = float(model.predict_proba(X)[0, 1])
        prediction = int(score >= 0.5)
        importances = sorted(zip(features, model.feature_importances_), key=lambda item: item[1], reverse=True)[:8] if hasattr(model, "feature_importances_") else []
        result = {"risk_score": round(score, 4), "risk_label": "HIGH" if score >= 0.7 else "MODERATE" if score >= 0.4 else "LOW", "prediction": prediction, "top_features": [(name, round(float(value), 4)) for name, value in importances]}
        label = result.get("risk_label", "LOW")
        return {
            "risk_score": result.get("risk_score", 0.0),
            "risk_label": label,
            "prediction": result.get("prediction", 0),
            "top_features": result.get("top_features", []),
            "interpretation": {"HIGH": "High mortality risk. Intensify monitoring and treatment.", "MODERATE": "Moderate risk. Continue standard ICU protocols.", "LOW": "Low risk indicators currently present."}.get(label, "Mortality score unavailable."),
            "model_ready": True,
        }
    except Exception as exc:
        log.error("predict_mortality(%s) error: %s", pid, exc, exc_info=True)
        return not_ready


def predict_early_mortality(pid: int) -> dict:
    default = {"risk_score": 0.0, "prediction": 0, "threshold": settings.EARLY_MORTALITY_THRESHOLD, "risk_label": "UNKNOWN", "alert": False, "operating_mode": "balanced_0.60", "recommended_action": "model unavailable", "top_features": [], "model_variant": "early_random_forest", "model_ready": False}
    try:
        patient = get_patient_by_id(pid)
        if patient is None or not early_model_is_trained():
            return default
        model, imputer, features = load_early_mortality_model()
        if hasattr(model, "n_jobs"):
            model.n_jobs = 1
        X = _prepare_row(patient, imputer, features)
        score = float(model.predict_proba(X)[0, 1])
        threshold = settings.EARLY_MORTALITY_THRESHOLD
        prediction = int(score >= threshold)
        importances = sorted(zip(features, model.feature_importances_), key=lambda item: item[1], reverse=True)[:8] if hasattr(model, "feature_importances_") else []
        return {
            "risk_score": round(score, 4),
            "prediction": prediction,
            "threshold": threshold,
            "risk_label": "HIGH" if score >= 0.7 else "MODERATE" if score >= threshold else "LOW",
            "alert": bool(prediction),
            "operating_mode": "balanced_0.60" if abs(threshold - 0.60) < 1e-9 else f"custom_{threshold:.2f}",
            "recommended_action": "urgent review" if score >= 0.7 else "clinical review recommended" if score >= threshold else "monitor closely" if score >= 0.4 else "routine monitoring",
            "top_features": [(name, round(float(value), 4)) for name, value in importances],
            "model_variant": "early_random_forest",
            "model_ready": True,
        }
    except Exception as exc:
        log.error("predict_early_mortality(%s) error: %s", pid, exc, exc_info=True)
        return default


def predict_los(pid: int) -> dict:
    default = {"los_hours": 0.0, "los_days": 0.0, "confidence_interval": [0.0, 0.0], "unit_avg_los_hours": None, "model_ready": False, "predicted_icu_los_hours": 0.0, "predicted_icu_los_days": 0.0, "note": "LOS model unavailable"}
    try:
        patient = get_patient_by_id(pid)
        if patient is None:
            return default
        if model_is_trained():
            _, _, los_model, _ = load_models()
            if hasattr(los_model, "n_jobs"):
                los_model.n_jobs = 1
            _, imputer, _, features = load_models()
            X = _prepare_row(patient, imputer, features)
            hours = float(los_model.predict(X)[0])
            days = hours / 24.0
            return {"los_hours": round(hours, 1), "los_days": round(days, 1), "confidence_interval": [round(hours * 0.65, 1), round(hours * 1.55, 1)], "unit_avg_los_hours": None, "model_ready": True, "predicted_icu_los_hours": round(hours, 1), "predicted_icu_los_days": round(days, 1)}
        hours = float(patient.get("icu_los_hours") or 24.0)
        days = hours / 24.0
        default.update({"los_hours": round(hours, 1), "los_days": round(days, 1), "confidence_interval": [round(hours * 0.6, 1), round(hours * 1.4, 1)], "predicted_icu_los_hours": round(hours, 1), "predicted_icu_los_days": round(days, 1), "note": "Estimate based on existing ICU LOS feature"})
        return default
    except Exception as exc:
        log.error("predict_los(%s) error: %s", pid, exc, exc_info=True)
        return default


def predict_sepsis(pid: int) -> dict:
    default = {"qsofa_score": 0, "sofa_approx": None, "sofa_approx_score": None, "sepsis_risk": False, "risk_label": "LOW RISK", "criteria_met": {"resp_rate": False, "sbp": False, "gcs": False}, "interpretation": "Lower sepsis risk"}
    try:
        patient = get_patient_by_id(pid)
        if patient is None:
            return default
        result = model_predict_sepsis(patient)
        qsofa = int(result.get("qsofa_score", 0))
        default.update({"qsofa_score": qsofa, "sofa_approx": result.get("sofa_approx_score"), "sofa_approx_score": result.get("sofa_approx_score"), "sepsis_risk": bool(result.get("sepsis_risk", 0)), "risk_label": "SEPSIS RISK — INITIATE PROTOCOL" if qsofa >= 2 else "ELEVATED" if qsofa == 1 else "LOW RISK", "criteria_met": {"resp_rate": bool((patient.get("resp_mean") or 0) >= 22), "sbp": bool((patient.get("sbp_min") or patient.get("sbp_mean") or 999) <= 100), "gcs": bool((patient.get("gcs_min") or 15) < 15)}, "interpretation": result.get("interpretation", "Lower sepsis risk")})
        return default
    except Exception as exc:
        log.error("predict_sepsis(%s) error: %s", pid, exc, exc_info=True)
        return default


def get_digital_twin(pid: int, steps: int = 12) -> dict:
    default = {"patient_id": pid, "steps": steps, "future_offsets": [], "forecast": {}, "deterioration_score": 0.0, "deterioration_label": "Stable"}
    try:
        from backend.services.clinical_service import get_vitals

        vitals_payload = get_vitals(pid, max_points=settings.MAX_VITALS_PER_PATIENT)
        periodic = pd.DataFrame(vitals_payload.get("periodic", []))
        patient = get_patient_by_id(pid)
        if periodic.empty or patient is None:
            return default
        for column in ["heartrate", "respiration", "sao2", "systemicsystolic", "temperature"]:
            if column in periodic.columns:
                periodic[column] = pd.to_numeric(periodic[column], errors="coerce")
        forecast = forecast_vitals(periodic, steps=min(steps, 12))
        future_offsets = []
        if "observationoffset" in periodic.columns:
            offsets = pd.to_numeric(periodic["observationoffset"], errors="coerce").dropna()
            last_offset = float(offsets.max()) if not offsets.empty else 0.0
            future_offsets = [round(last_offset + (idx + 1) * 5, 1) for idx in range(min(steps, 12))]
        deterioration = deterioration_score(patient)
        return {"patient_id": pid, "steps": min(steps, 12), "future_offsets": future_offsets, "forecast": forecast, "simulated_forecast": simulate_what_if(forecast), "deterioration_score": float(deterioration.get("score", 0.0)), "deterioration_label": deterioration.get("label", "Stable"), "deterioration": deterioration}
    except Exception as exc:
        log.error("get_digital_twin(%s) error: %s", pid, exc, exc_info=True)
        return default
