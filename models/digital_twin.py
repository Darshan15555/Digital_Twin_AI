"""
Digital twin forecasting and deterioration scoring.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

ALERT_THRESHOLDS = {
    "heartrate": {"low": 40, "high": 150},
    "sao2": {"low": 90, "high": None},
    "systemicsystolic": {"low": 80, "high": None},
    "lactate": {"low": None, "high": 4},
    "gcs": {"low": 8, "high": None},
}


def forecast_vitals(vitals_df: pd.DataFrame, steps: int = 6, recent_window: int = 12) -> dict:
    forecast = {}
    vital_columns = ["heartrate", "sao2", "respiration", "systemicsystolic", "temperature"]

    for column in vital_columns:
        if column not in vitals_df.columns:
            continue
        series = vitals_df[column].dropna()
        if len(series) < 3:
            forecast[column] = {"values": [], "lower": [], "upper": []}
            continue

        recent = series.tail(recent_window)
        ewma_value = recent.ewm(span=min(5, len(recent))).mean().iloc[-1]
        trend = 0.0
        if len(recent) > 1:
            trend = (recent.iloc[-1] - recent.iloc[0]) / (len(recent) - 1)
        sigma = recent.std()
        if pd.isna(sigma):
            sigma = 0.0

        values = []
        lower = []
        upper = []
        current = ewma_value
        for _ in range(steps):
            current = current + (trend * 0.3)
            values.append(round(float(current), 2))
            lower.append(round(float(current - 1.5 * sigma), 2))
            upper.append(round(float(current + 1.5 * sigma), 2))

        forecast[column] = {"values": values, "lower": lower, "upper": upper}

    return forecast


def deterioration_score(patient_row: dict) -> dict:
    penalties = 0
    hr = patient_row.get("heartrate_mean")
    sao2 = patient_row.get("sao2_min")
    sbp = patient_row.get("sbp_min")
    lactate = patient_row.get("lactate_max")
    gcs = patient_row.get("gcs_min")

    if hr is not None and (hr < 50 or hr > 130):
        penalties += 1
    if sao2 is not None and sao2 < 92:
        penalties += 1
    if sbp is not None and sbp < 90:
        penalties += 1
    if lactate is not None and lactate > 4:
        penalties += 1
    if gcs is not None and gcs < 8:
        penalties += 2

    if penalties >= 4:
        label = "Critical"
    elif penalties >= 2:
        label = "Deteriorating"
    else:
        label = "Stable"

    return {"score": penalties, "label": label}


def simulate_what_if(current_forecast: dict, fio2_delta: float = 0.0, vasopressor_delta: float = 0.0) -> dict:
    adjusted = {}
    for key, payload in current_forecast.items():
        values = payload.get("values", [])
        if not values:
            adjusted[key] = payload
            continue
        updated = values[:]
        if key == "sao2":
            updated = [round(value + (fio2_delta * 8), 2) for value in updated]
        if key == "systemicsystolic":
            updated = [round(value + (vasopressor_delta * 12), 2) for value in updated]
        adjusted[key] = {
            "values": updated,
            "lower": payload.get("lower", []),
            "upper": payload.get("upper", []),
        }
    return adjusted
