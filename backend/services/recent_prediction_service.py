from __future__ import annotations

import logging
import re
import sqlite3
from typing import Any

import numpy as np
import pandas as pd

from config.settings import settings
from backend.services.prediction_service import predict_deterioration, predict_mortality
from training.steps.step3_load_labs import LAB_ALIASES, LAB_RANGES
from training.steps.step6_aggregate import aggregate_interventions, aggregate_labs, aggregate_vitals
from training.steps.step8_compute_scores import compute_all_scores
from training.utils.range_filters import filter_numeric_range, normalize_fio2

log = logging.getLogger(__name__)

WINDOW_MINUTES = 240
MAX_WINDOWS = 6


def _sqlite_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(str(settings.db_path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def _sqlite_table_exists(conn: sqlite3.Connection, table_name: str) -> bool:
    query = "SELECT name FROM sqlite_master WHERE type='table' AND name=?"
    return conn.execute(query, (table_name,)).fetchone() is not None


def _sqlite_df(conn: sqlite3.Connection, sql: str, params: tuple = ()) -> pd.DataFrame:
    return pd.read_sql_query(sql, conn, params=params)


def _clean_text(series: pd.Series) -> pd.Series:
    return (
        series.fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
        .str.replace(r"[_/]+", " ", regex=True)
        .str.replace(r"\s+", " ", regex=True)
        .str.replace(r"[^\w\s(),.-]+", "", regex=True)
        .str.strip()
    )


LAB_NAME_MAP = {
    alias: canonical
    for canonical, aliases in LAB_ALIASES.items()
    for alias in aliases
}


def _normalize_lab_name(series: pd.Series) -> pd.Series:
    normalized = _clean_text(series).replace(LAB_NAME_MAP)
    direct_mask = normalized.isin(LAB_RANGES)
    if direct_mask.all():
        return normalized
    for canonical, aliases in LAB_ALIASES.items():
        alias_pattern = "|".join(re.escape(alias) for alias in aliases)
        match_mask = ~direct_mask & normalized.str.contains(rf"\b(?:{alias_pattern})\b", regex=True, na=False)
        normalized = normalized.where(~match_mask, canonical)
        direct_mask = normalized.isin(LAB_RANGES)
    return normalized


def _build_patient_static_row(patient: dict[str, Any], patient_id: int) -> pd.DataFrame:
    age = patient.get("age")
    height = patient.get("admissionheight", patient.get("admissionheight_cm"))
    weight = patient.get("admissionweight", patient.get("admissionweight_kg"))
    bmi = None
    try:
        if height and weight:
            bmi = float(weight) / ((float(height) / 100.0) ** 2)
    except Exception:
        bmi = None

    gender = str(patient.get("gender", "")).strip().lower()
    ethnicity = str(patient.get("ethnicity", "")).strip().lower()
    unit = str(patient.get("unittype", "")).strip().lower()
    source = str(patient.get("unitadmitsource", "")).strip().lower()
    diagnosis = patient.get("apacheadmissiondx")

    return pd.DataFrame(
        [
            {
                "patientunitstayid": patient_id,
                "age": age,
                "admissionweight_kg": weight,
                "admissionheight_cm": height,
                "bmi": bmi,
                "hospitaladmit_offset_h": abs(float(patient.get("hospitaladmitoffset_hours", 0) or 0)),
                "gender_male": int(gender == "male"),
                "ethnicity_white": int("white" in ethnicity),
                "ethnicity_black": int("black" in ethnicity),
                "ethnicity_hispanic": int("hispanic" in ethnicity),
                "unittype_micu": int("micu" in unit),
                "unittype_sicu": int("sicu" in unit),
                "unittype_ccu": int("ccu" in unit),
                "unittype_csicu": int("csicu" in unit),
                "unittype_neuro": int("neuro" in unit),
                "unittype_cardiac": int("cardiac" in unit),
                "unitsource_ed": int("ed" in source or "emergency" in source),
                "unitsource_floor": int("floor" in source),
                "unitsource_or": int(source == "operating room" or source == "or"),
                "apache_diagnosis": diagnosis,
            }
        ]
    )


def _fetch_recent_sources(patient_id: int, hours_back: int) -> tuple[dict[str, pd.DataFrame], float]:
    minutes_back = int(hours_back * 60)
    source_maxima: list[float] = []

    queries = {
        "vitals_periodic": ("SELECT * FROM vitals_periodic WHERE patientunitstayid = ?", "observationoffset"),
        "vitals_aperiodic": ("SELECT * FROM vitals_aperiodic WHERE patientunitstayid = ?", "observationoffset"),
        "labs": ("SELECT * FROM labs WHERE patientunitstayid = ?", "labresultoffset"),
        "infusions": ("SELECT * FROM infusions WHERE patientunitstayid = ?", "infusionoffset"),
        "respiratory_care": ("SELECT * FROM respiratory_care WHERE patientunitstayid = ?", "respcarestatusoffset"),
        "nurse_charting": ("SELECT * FROM nurse_charting WHERE patientunitstayid = ?", "nursingchartoffset"),
    }

    raw: dict[str, pd.DataFrame] = {}
    with _sqlite_conn() as conn:
        for table_name, (sql, offset_col) in queries.items():
            if not _sqlite_table_exists(conn, table_name):
                raw[table_name] = pd.DataFrame()
                continue
            df = _sqlite_df(conn, sql, (patient_id,))
            if df.empty:
                raw[table_name] = df
                continue
            df[offset_col] = pd.to_numeric(df[offset_col], errors="coerce")
            df = df.dropna(subset=[offset_col]).copy()
            raw[table_name] = df
            if not df.empty:
                source_maxima.append(float(df[offset_col].max()))

    anchor_offset = max(source_maxima) if source_maxima else 0.0
    min_offset = max(0.0, anchor_offset - minutes_back)

    filtered: dict[str, pd.DataFrame] = {}
    for table_name, df in raw.items():
        if df.empty:
            filtered[table_name] = df
            continue
        offset_col = {
            "vitals_periodic": "observationoffset",
            "vitals_aperiodic": "observationoffset",
            "labs": "labresultoffset",
            "infusions": "infusionoffset",
            "respiratory_care": "respcarestatusoffset",
            "nurse_charting": "nursingchartoffset",
        }[table_name]
        subset = df[df[offset_col].between(min_offset, anchor_offset, inclusive="both")].copy()
        subset[offset_col] = subset[offset_col] - min_offset
        filtered[table_name] = subset

    return filtered, anchor_offset


def _prepare_vitals(periodic: pd.DataFrame, aperiodic: pd.DataFrame) -> dict[str, pd.DataFrame]:
    periodic = periodic.copy()
    aperiodic = aperiodic.copy()
    if "systemicsystolic" not in periodic.columns:
        periodic["systemicsystolic"] = np.nan
    if "systemicdiastolic" not in periodic.columns:
        periodic["systemicdiastolic"] = np.nan
    periodic["map_estimate"] = (
        pd.to_numeric(periodic["systemicsystolic"], errors="coerce")
        + 2 * pd.to_numeric(periodic["systemicdiastolic"], errors="coerce")
    ) / 3.0
    return {"periodic": periodic, "aperiodic": aperiodic}


def _prepare_labs(labs: pd.DataFrame) -> pd.DataFrame:
    labs = labs.copy()
    if labs.empty:
        return pd.DataFrame(columns=["patientunitstayid", "labresultoffset", "labname", "labresult"])
    labs["labname"] = _normalize_lab_name(labs["labname"])
    labs = labs[labs["labname"].isin(LAB_RANGES)].copy()
    labs["labresult"] = pd.to_numeric(labs["labresult"], errors="coerce")
    for lab_name, (lower, upper) in LAB_RANGES.items():
        mask = labs["labname"].eq(lab_name)
        if mask.any():
            labs.loc[mask, "labresult"] = filter_numeric_range(labs.loc[mask, "labresult"], lower, upper)
    return labs.dropna(subset=["labresult"])[["patientunitstayid", "labresultoffset", "labname", "labresult"]]


def _prepare_interventions(raw: dict[str, pd.DataFrame]) -> dict[str, object]:
    infusion = raw["infusions"].copy()
    if not infusion.empty:
        infusion["drugname"] = _clean_text(infusion["drugname"])
        infusion["drugrate"] = pd.to_numeric(infusion["drugrate"], errors="coerce")
        infusion["is_vasopressor"] = pd.to_numeric(infusion.get("on_vasopressor"), errors="coerce").fillna(0).astype("int8")
        infusion["epinephrine_active"] = infusion["drugname"].str.contains("epinephrine|adrenaline", regex=True, na=False).astype("int8")
        infusion["dopamine_active"] = infusion["drugname"].str.contains("dopamine", regex=True, na=False).astype("int8")
        infusion["vasopressin_active"] = infusion["drugname"].str.contains("vasopressin", regex=True, na=False).astype("int8")
        infusion["norepinephrine_active"] = infusion["drugname"].str.contains("norepinephrine|levophed|noradrenaline", regex=True, na=False).astype("int8")
        infusion = infusion[["patientunitstayid", "infusionoffset", "drugname", "drugrate", "is_vasopressor", "epinephrine_active", "dopamine_active", "vasopressin_active", "norepinephrine_active"]]
    else:
        infusion = pd.DataFrame(columns=["patientunitstayid", "infusionoffset", "drugname", "drugrate", "is_vasopressor", "epinephrine_active", "dopamine_active", "vasopressin_active", "norepinephrine_active"])

    resp_care = raw["respiratory_care"].copy()
    if not resp_care.empty:
        resp_care["airwaytype"] = _clean_text(resp_care.get("airwaytype"))
        resp_care["peep"] = pd.to_numeric(resp_care.get("peep"), errors="coerce")
        resp_care["fio2_inline"] = normalize_fio2(resp_care.get("fio2"))
        resp_care["on_ventilator"] = pd.to_numeric(resp_care.get("on_ventilator"), errors="coerce").fillna(0).astype("int8")
        resp_care = resp_care[["patientunitstayid", "respcarestatusoffset", "airwaytype", "peep", "fio2_inline", "on_ventilator"]]
    else:
        resp_care = pd.DataFrame(columns=["patientunitstayid", "respcarestatusoffset", "airwaytype", "peep", "fio2_inline", "on_ventilator"])

    io_df = pd.DataFrame(columns=["patientunitstayid", "intakeoutputoffset", "cellvaluenumeric", "io_type"])

    nurse = raw["nurse_charting"].copy()
    if not nurse.empty:
        nurse["nursingchartoffset"] = pd.to_numeric(nurse["nursingchartoffset"], errors="coerce")
        metric_name = _clean_text(nurse.get("metric_name"))
        metric_map = {
            "gcs total": "gcs_total",
            "gcs motor": "gcs_motor",
            "gcs verbal": "gcs_verbal",
            "gcs eye": "gcs_eye",
            "pain score": "pain_score",
            "rass score": "rass_score",
            "cam icu": "cam_icu",
        }
        nurse["charttype"] = metric_name.replace(metric_map)
        nurse["nursingchartvalue"] = pd.to_numeric(nurse.get("metric_value"), errors="coerce")
        nurse = nurse[nurse["charttype"].isin(metric_map.values())][["patientunitstayid", "nursingchartoffset", "charttype", "nursingchartvalue"]].dropna(subset=["nursingchartoffset"])
    else:
        nurse = pd.DataFrame(columns=["patientunitstayid", "nursingchartoffset", "charttype", "nursingchartvalue"])

    return {
        "infusion": infusion,
        "respiratory": {"care": resp_care, "charting": pd.DataFrame(columns=["patientunitstayid", "respchartoffset", "fio2"])},
        "intake_output": io_df,
        "nurse_charting": nurse,
    }


def build_recent_feature_row(patient_id: int, hours_back: int = 8) -> dict[str, Any]:
    with _sqlite_conn() as conn:
        patient_df = _sqlite_df(conn, "SELECT * FROM patients WHERE patientunitstayid = ?", (patient_id,))
    patient = patient_df.iloc[0].to_dict() if not patient_df.empty else None
    if not patient:
        return {"error": f"Patient {patient_id} not found"}

    raw, anchor_offset = _fetch_recent_sources(patient_id, hours_back)
    if all(df.empty for df in raw.values()):
        return {"error": "No recent clinical data available for this patient"}

    window_count = max(1, min(MAX_WINDOWS, int(np.ceil(hours_back * 60 / WINDOW_MINUTES))))
    skeleton = pd.DataFrame(
        {
            "patientunitstayid": [patient_id] * window_count,
            "window_id": list(range(window_count)),
            "window_start_hour": [i * 4 for i in range(window_count)],
            "window_end_hour": [(i + 1) * 4 for i in range(window_count)],
        }
    )

    vitals = _prepare_vitals(raw["vitals_periodic"], raw["vitals_aperiodic"])
    labs = _prepare_labs(raw["labs"])
    interventions = _prepare_interventions(raw)
    patient_static = _build_patient_static_row(patient, patient_id)

    features = aggregate_vitals(skeleton, vitals["periodic"], vitals["aperiodic"])
    features = aggregate_interventions(features, interventions)
    features = aggregate_labs(features, labs)
    features = features.merge(patient_static, on="patientunitstayid", how="left")
    if "urine_rate_mlhr" not in features.columns:
        features["urine_rate_mlhr"] = np.nan
    if "fio2_mean" not in features.columns:
        features["fio2_mean"] = np.nan
    if "po2_value" not in features.columns:
        features["po2_value"] = np.nan
    features["oliguria"] = (features["urine_rate_mlhr"] < (0.5 * features["admissionweight_kg"])).fillna(False).astype("int8")
    features["pf_ratio"] = features["po2_value"] / features["fio2_mean"].replace({0: np.nan})
    features = compute_all_scores(features)
    features = features.sort_values(["patientunitstayid", "window_id"]).reset_index(drop=True)
    final_row = features.iloc[-1].dropna().to_dict() if not features.empty else {}

    return {
        "patient_id": patient_id,
        "hours_back": hours_back,
        "anchor_offset_minutes": anchor_offset,
        "window_count": window_count,
        "feature_row": final_row,
        "feature_frame": features,
    }


def predict_from_recent_history(patient_id: int, hours_back: int = 8, threshold_type: str = "max_f1") -> dict[str, Any]:
    try:
        built = build_recent_feature_row(patient_id, hours_back=hours_back)
        if built.get("error"):
            return built
        feature_row = built["feature_row"]
        current_window = int(built["feature_frame"]["window_id"].max()) if not built["feature_frame"].empty else 0
        mortality = predict_mortality(feature_row, threshold_type=threshold_type)
        deterioration = predict_deterioration(feature_row, window_id=min(current_window, 4), threshold_type=threshold_type)
        return {
            "patient_id": patient_id,
            "hours_back": hours_back,
            "window_count": built["window_count"],
            "current_window_id": current_window,
            "mortality": mortality,
            "deterioration": deterioration,
            "used_features": sorted(feature_row.keys()),
        }
    except Exception as exc:
        log.error("predict_from_recent_history(%s) error: %s", patient_id, exc, exc_info=True)
        return {"error": str(exc), "patient_id": patient_id, "hours_back": hours_back}
