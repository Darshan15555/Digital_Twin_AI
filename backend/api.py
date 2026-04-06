"""
FastAPI backend for the local ICU analytics system.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import create_engine

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from config.config import API_HOST, API_PORT, DATABASE_URL
from models.digital_twin import ALERT_THRESHOLDS, deterioration_score, forecast_vitals, simulate_what_if
from models.ml_model import (
    early_model_is_trained,
    model_is_trained,
    predict_early_mortality,
    predict_los,
    predict_mortality,
    predict_sepsis,
)
from utils.db_manager import (
    analytics_query,
    get_patient,
    get_patient_table,
    get_patients_filtered,
    get_stats,
    update_prediction_fields,
)

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)
pg_engine = create_engine(DATABASE_URL, future=True, pool_pre_ping=True) if DATABASE_URL.startswith("postgresql") else None

app = FastAPI(title="ICU Analytics API", version="2.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def to_json(df: pd.DataFrame) -> list[dict]:
    if df.empty:
        return []
    return df.where(pd.notnull(df), None).to_dict(orient="records")


def load_patient_row(patient_id: int) -> dict:
    patient_df = get_patient(patient_id)
    if patient_df.empty:
        raise HTTPException(status_code=404, detail=f"Patient {patient_id} not found")
    return patient_df.iloc[0].to_dict()


def load_postgres_patient_row(patient_id: int) -> dict:
    if pg_engine is None:
        raise HTTPException(status_code=503, detail="PostgreSQL database is not configured for early mortality predictions.")

    query = """
        SELECT *
        FROM ml_prep.ml_dataset
        WHERE patientunitstayid = %(patient_id)s
    """
    patient_df = pd.read_sql(query, pg_engine, params={"patient_id": patient_id})
    if patient_df.empty:
        raise HTTPException(status_code=404, detail=f"Patient {patient_id} not found in ml_prep.ml_dataset")
    return patient_df.iloc[0].to_dict()


@app.get("/")
def root():
    return {"status": "ok", "system": "ICU Analytics API"}


@app.get("/stats")
def stats():
    payload = get_stats()
    payload["model_ready"] = model_is_trained()
    return payload


@app.get("/patients")
def list_patients(
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
    search: Optional[str] = None,
    unit: Optional[str] = None,
    high_risk_only: bool = False,
    age_min: Optional[int] = None,
    age_max: Optional[int] = None,
    on_vasopressor: Optional[bool] = None,
    on_ventilator: Optional[bool] = None,
):
    df = get_patients_filtered(
        limit=limit,
        offset=offset,
        search=search,
        unit=unit,
        risk_only=high_risk_only,
        age_min=age_min,
        age_max=age_max,
        on_vasopressor=on_vasopressor,
        on_ventilator=on_ventilator,
    )
    return {"patients": to_json(df), "count": len(df)}


@app.get("/patients/{patient_id}")
def patient_record(patient_id: int):
    return load_patient_row(patient_id)


@app.get("/patients/{patient_id}/vitals")
def patient_vitals(patient_id: int):
    periodic = get_patient_table("vitals_periodic", patient_id, "observationoffset")
    aperiodic = get_patient_table("vitals_aperiodic", patient_id, "observationoffset")
    return {
        "periodic": to_json(periodic),
        "aperiodic": to_json(aperiodic),
    }


@app.get("/patients/{patient_id}/labs")
def patient_labs(patient_id: int):
    return {"labs": to_json(get_patient_table("labs", patient_id, "labresultoffset"))}


@app.get("/patients/{patient_id}/diagnosis")
def patient_diagnosis(patient_id: int):
    return {"diagnosis": to_json(get_patient_table("diagnosis", patient_id, "diagnosisoffset"))}


@app.get("/patients/{patient_id}/treatments")
def patient_treatments(patient_id: int):
    return {"treatments": to_json(get_patient_table("treatments", patient_id, "treatmentoffset"))}


@app.get("/patients/{patient_id}/medications")
def patient_medications(patient_id: int):
    meds = get_patient_table("medications", patient_id, "drugstartoffset")
    infusions = get_patient_table("infusions", patient_id, "infusionoffset")
    return {"medications": to_json(meds), "infusions": to_json(infusions)}


@app.get("/patients/{patient_id}/fluid-balance")
def patient_fluid_balance(patient_id: int):
    summary = get_patient_table("intake_output_summary", patient_id)
    raw = get_patient_table("intake_output_summary", patient_id)
    return {"summary": to_json(summary), "timeline": to_json(raw)}


@app.get("/patients/{patient_id}/ventilation")
def patient_ventilation(patient_id: int):
    return {"ventilation": to_json(get_patient_table("respiratory_care", patient_id, "respcarestatusoffset"))}


@app.get("/patients/{patient_id}/comorbidities")
def patient_comorbidities(patient_id: int):
    return {"comorbidities": to_json(get_patient_table("comorbidities", patient_id))}


@app.get("/patients/{patient_id}/predict/mortality")
def patient_predict_mortality(patient_id: int):
    if not model_is_trained():
        raise HTTPException(status_code=503, detail="Models not trained. Run setup.py first.")
    row = load_patient_row(patient_id)
    result = predict_mortality(row)
    update_prediction_fields(
        patient_id,
        {
            "predicted_mortality_risk": result["risk_score"],
            "predicted_mortality_label": result["risk_label"],
        },
    )
    return result


@app.get("/patients/{patient_id}/predict/early-mortality")
def patient_predict_early_mortality(patient_id: int):
    if not early_model_is_trained():
        raise HTTPException(status_code=503, detail="Early mortality model not trained. Run train_postgres_model.py first.")
    row = load_postgres_patient_row(patient_id)
    result = predict_early_mortality(row)
    return result


@app.get("/patients/{patient_id}/predict/los")
def patient_predict_los(patient_id: int):
    if not model_is_trained():
        raise HTTPException(status_code=503, detail="Models not trained. Run setup.py first.")
    row = load_patient_row(patient_id)
    return predict_los(row)


@app.get("/patients/{patient_id}/predict/sepsis")
def patient_predict_sepsis(patient_id: int):
    row = load_patient_row(patient_id)
    return predict_sepsis(row)


@app.get("/patients/{patient_id}/digital-twin")
def patient_digital_twin(
    patient_id: int,
    steps: int = Query(6, ge=3, le=12),
    fio2_delta: float = 0.0,
    vasopressor_delta: float = 0.0,
):
    row = load_patient_row(patient_id)
    periodic = get_patient_table("vitals_periodic", patient_id, "observationoffset")
    labs = get_patient_table("labs", patient_id, "labresultoffset")
    nurse = get_patient_table("nurse_charting", patient_id, "nursingchartoffset")
    if periodic.empty:
        raise HTTPException(status_code=404, detail="No periodic vitals found")

    forecast = forecast_vitals(periodic, steps=steps)
    simulated = simulate_what_if(forecast, fio2_delta=fio2_delta, vasopressor_delta=vasopressor_delta)
    deterioration = deterioration_score(row)
    return {
        "forecast": forecast,
        "simulated_forecast": simulated,
        "deterioration": deterioration,
        "alert_thresholds": ALERT_THRESHOLDS,
        "latest_labs": to_json(labs.tail(20)),
        "latest_nurse_charting": to_json(nurse.tail(20)),
    }


@app.get("/analytics/unit-breakdown")
def analytics_unit_breakdown():
    df = analytics_query(
        """
        SELECT unittype,
               COUNT(*) AS patient_count,
               AVG(hospital_mortality) * 100 AS mortality_pct,
               AVG(apachescore) AS avg_apache,
               AVG(icu_los_hours) AS avg_los_hours
        FROM ml_dataset
        GROUP BY unittype
        ORDER BY patient_count DESC
        """
    )
    return to_json(df)


@app.get("/analytics/mortality-by-age")
def analytics_mortality_by_age():
    df = analytics_query("SELECT age, hospital_mortality FROM ml_dataset WHERE age IS NOT NULL")
    if df.empty:
        return []
    df["age_group"] = pd.cut(
        df["age"],
        bins=[0, 18, 40, 60, 75, 120],
        labels=["<18", "18-40", "40-60", "60-75", "75+"],
        include_lowest=True,
    )
    result = (
        df.groupby("age_group", observed=True)
        .agg(count=("hospital_mortality", "count"), mortality_pct=("hospital_mortality", lambda s: s.mean() * 100))
        .reset_index()
    )
    return to_json(result)


@app.get("/analytics/top-diagnoses")
def analytics_top_diagnoses(limit: int = 15):
    df = analytics_query(
        f"""
        SELECT diagnosisstring, COUNT(*) AS count
        FROM diagnosis
        GROUP BY diagnosisstring
        ORDER BY count DESC
        LIMIT {int(limit)}
        """
    )
    return to_json(df)


@app.get("/analytics/vasopressor-usage")
def analytics_vasopressor_usage():
    df = analytics_query(
        """
        SELECT unittype, AVG(on_vasopressor) * 100 AS vasopressor_pct
        FROM ml_dataset
        GROUP BY unittype
        ORDER BY vasopressor_pct DESC
        """
    )
    return to_json(df)


@app.get("/analytics/ventilator-usage")
def analytics_ventilator_usage():
    df = analytics_query(
        """
        SELECT unittype, AVG(on_ventilator) * 100 AS ventilator_pct
        FROM ml_dataset
        GROUP BY unittype
        ORDER BY ventilator_pct DESC
        """
    )
    return to_json(df)


@app.get("/analytics/fluid-balance")
def analytics_fluid_balance():
    df = analytics_query(
        """
        SELECT hospital_mortality,
               AVG(fluid_balance_24h) AS avg_fluid_balance,
               AVG(urine_output_per_hour) AS avg_urine_output_per_hour
        FROM ml_dataset
        GROUP BY hospital_mortality
        """
    )
    return to_json(df)


if __name__ == "__main__":
    uvicorn.run(app, host=API_HOST, port=API_PORT)
