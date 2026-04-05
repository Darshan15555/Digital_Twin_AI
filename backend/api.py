"""
api.py
FastAPI backend for ICU Analytics System.
Provides REST endpoints for the Streamlit dashboard.
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
import uvicorn
import logging
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.db_manager import (
    get_patient_list, get_patient_by_id,
    get_vitals_by_patient, get_labs_by_patient,
    get_diagnosis_by_patient, get_treatments_by_patient,
    get_stats, count_patients, update_ml_scores
)
from models.ml_model import predict_patient, digital_twin_forecast, model_is_trained
from config.config import API_HOST, API_PORT

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

app = FastAPI(
    title="ICU Analytics API",
    description="Local ICU analytics system powered by eICU dataset",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def df_to_json(df: pd.DataFrame) -> list:
    import math
    """Convert dataframe to JSON-safe list of dicts."""
    records = df.where(pd.notnull(df), None).to_dict(orient="records")
    cleaned = []
    for row in records:
        clean_row = {k: (None if isinstance(v, float) and (math.isnan(v) or math.isinf(v)) else v) for k, v in row.items()}
        cleaned.append(clean_row)
    return cleaned


@app.get("/")
def root():
    return {"status": "ok", "system": "ICU Analytics API", "version": "1.0.0"}


@app.get("/stats")
def system_stats():
    """Overall ICU statistics."""
    try:
        stats = get_stats()
        return {
            "total_patients": int(stats["total_patients"]),
            "mortality_rate": round(float(stats["mortality_rate"] or 0) * 100, 2),
            "avg_age": round(float(stats["avg_age"] or 0), 1),
            "avg_apache_score": round(float(stats["avg_apache"] or 0), 1),
            "model_ready": model_is_trained()
        }
    except Exception as e:
        log.error(f"Stats error: {e}")
        raise HTTPException(500, str(e))


@app.get("/patients")
def list_patients(
    limit: int = Query(100, le=500),
    offset: int = 0,
    search: Optional[str] = None,
    high_risk_only: bool = False
):
    """Get list of patients with optional filters."""
    try:
        from utils.db_manager import get_conn
        import sqlite3
        conn = get_conn()

        query = "SELECT * FROM patients WHERE 1=1"
        params = []

        if search:
            try:
                pid = int(search)
                query += " AND patientunitstayid = ?"
                params.append(pid)
            except ValueError:
                query += " AND (unittype LIKE ? OR gender LIKE ?)"
                params.extend([f"%{search}%", f"%{search}%"])

        if high_risk_only:
            query += " AND (ml_risk_score >= 0.5 OR apachescore >= 60)"

        query += f" LIMIT {limit} OFFSET {offset}"
        df = pd.read_sql(query, conn, params=params)
        conn.close()
        return {"patients": df_to_json(df), "total": len(df)}
    except Exception as e:
        log.error(f"Patient list error: {e}")
        raise HTTPException(500, str(e))


@app.get("/patients/{patient_id}")
def get_patient(patient_id: int):
    """Get detailed info for a single patient."""
    df = get_patient_by_id(patient_id)
    if df.empty:
        raise HTTPException(404, f"Patient {patient_id} not found")
    return df_to_json(df)[0]


@app.get("/patients/{patient_id}/vitals")
def patient_vitals(patient_id: int, max_points: int = 200):
    """Get vitals time-series for a patient."""
    df = get_vitals_by_patient(patient_id)
    if df.empty:
        return {"vitals": [], "count": 0}

    # Downsample if too many points
    if len(df) > max_points:
        step = len(df) // max_points
        df = df.iloc[::step]

    return {"vitals": df_to_json(df), "count": len(df)}


@app.get("/patients/{patient_id}/labs")
def patient_labs(patient_id: int):
    """Get lab results for a patient."""
    df = get_labs_by_patient(patient_id)
    return {"labs": df_to_json(df), "count": len(df)}


@app.get("/patients/{patient_id}/diagnosis")
def patient_diagnosis(patient_id: int):
    """Get diagnoses for a patient."""
    df = get_diagnosis_by_patient(patient_id)
    return {"diagnosis": df_to_json(df)}


@app.get("/patients/{patient_id}/treatments")
def patient_treatments(patient_id: int):
    """Get treatments for a patient."""
    df = get_treatments_by_patient(patient_id)
    return {"treatments": df_to_json(df)}


@app.get("/patients/{patient_id}/predict")
def predict_outcome(patient_id: int):
    """Predict ICU mortality risk for a patient."""
    if not model_is_trained():
        raise HTTPException(503, "ML model not trained yet. Run setup.py first.")

    patient_df = get_patient_by_id(patient_id)
    if patient_df.empty:
        raise HTTPException(404, f"Patient {patient_id} not found")

    patient_dict = patient_df.iloc[0].to_dict()
    result = predict_patient(patient_dict)

    # Cache prediction in DB
    update_ml_scores(patient_id, result["risk_score"], result["prediction"])

    return {
        "patient_id": patient_id,
        **result
    }


@app.get("/patients/{patient_id}/digital-twin")
def digital_twin(patient_id: int, steps: int = 6):
    """Generate digital twin forecast for patient vitals."""
    vitals_df = get_vitals_by_patient(patient_id)
    if vitals_df.empty:
        raise HTTPException(404, "No vitals data available for this patient")

    forecast = digital_twin_forecast(vitals_df, steps=steps)
    last_offset = vitals_df["observationoffset"].max() if "observationoffset" in vitals_df.columns else 0

    # Build future time points (each step = 5 minutes)
    future_offsets = [last_offset + (i + 1) * 5 for i in range(steps)]

    return {
        "patient_id": patient_id,
        "steps": steps,
        "future_offsets": future_offsets,
        "forecast": forecast,
        "note": "Forecast using exponential weighted moving average"
    }


@app.get("/analytics/unit-breakdown")
def unit_breakdown():
    """Get patient distribution by ICU unit type."""
    from utils.db_manager import get_conn
    conn = get_conn()
    df = pd.read_sql(
        "SELECT unittype, COUNT(*) as count, AVG(hospital_mortality)*100 as mortality_pct, AVG(apachescore) as avg_apache FROM patients GROUP BY unittype ORDER BY count DESC",
        conn
    )
    conn.close()
    return df_to_json(df)


@app.get("/analytics/mortality-by-age")
def mortality_by_age():
    """Get mortality rate by age group."""
    from utils.db_manager import get_conn
    conn = get_conn()
    df = pd.read_sql("SELECT age, hospital_mortality FROM patients WHERE age IS NOT NULL", conn)
    conn.close()
    df["age_group"] = pd.cut(df["age"], bins=[0, 18, 40, 60, 75, 120],
                              labels=["<18", "18-40", "40-60", "60-75", "75+"])
    result = df.groupby("age_group", observed=True).agg(
        count=("hospital_mortality", "count"),
        mortality_pct=("hospital_mortality", lambda x: round(x.mean() * 100, 2))
    ).reset_index()
    return df_to_json(result)


@app.get("/analytics/top-diagnoses")
def top_diagnoses(limit: int = 15):
    """Get most common diagnoses."""
    from utils.db_manager import get_conn
    conn = get_conn()
    df = pd.read_sql(
        f"SELECT diagnosisstring, COUNT(*) as count FROM diagnosis GROUP BY diagnosisstring ORDER BY count DESC LIMIT {limit}",
        conn
    )
    conn.close()
    return df_to_json(df)


if __name__ == "__main__":
    log.info(f"Starting ICU Analytics API on {API_HOST}:{API_PORT}")
    uvicorn.run("api:app", host=API_HOST, port=API_PORT, reload=False)
