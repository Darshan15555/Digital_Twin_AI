from __future__ import annotations

import logging

import pandas as pd

from backend.dependencies import IS_POSTGRES, get_table_columns, query_df, table_exists
from backend.services.patient_service import get_patient_by_id
from utils.sanitize import sanitize_df

log = logging.getLogger(__name__)


def get_vitals(pid: int, max_points: int = 400) -> dict:
    try:
        if IS_POSTGRES:
            df = query_df(
                """
                SELECT patientunitstayid, observationoffset, heartrate, respiration, sao2,
                       systemicsystolic, systemicdiastolic, temperature, systemicmean AS cvp
                FROM raw_eicu.vital_periodic
                WHERE CAST(patientunitstayid AS text) = :patient_id
                ORDER BY observationoffset
                """,
                {"patient_id": str(pid)},
            )
        else:
            table_name = "vitals" if table_exists("vitals") else "vitals_periodic"
            if not table_exists(table_name):
                return {"vitals": [], "periodic": [], "aperiodic": [], "count": 0}
            available_cols = get_table_columns(table_name)
            wanted = ["observationoffset", "heartrate", "respiration", "sao2", "systemicsystolic", "systemicdiastolic", "temperature", "cvp"]
            select_cols = ["patientunitstayid"] + [col for col in wanted if col in available_cols]
            df = query_df(f"SELECT {', '.join(select_cols)} FROM {table_name} WHERE patientunitstayid = ? ORDER BY observationoffset", (pid,))
        if df.empty:
            return {"vitals": [], "periodic": [], "aperiodic": [], "count": 0}
        if len(df) > max_points:
            df = df.iloc[:: max(1, len(df) // max_points)].copy()
        if "offset_hours" not in df.columns and "observationoffset" in df.columns:
            df["offset_hours"] = pd.to_numeric(df["observationoffset"], errors="coerce") / 60.0
        rows = sanitize_df(df)
        return {"vitals": rows, "periodic": rows, "aperiodic": [], "count": len(rows)}
    except Exception as exc:
        log.error("get_vitals(%s) error: %s", pid, exc, exc_info=True)
        return {"vitals": [], "periodic": [], "aperiodic": [], "count": 0}


def get_labs(pid: int) -> dict:
    try:
        if IS_POSTGRES:
            df = query_df(
                """
                SELECT labresultoffset, labname, labresult
                FROM raw_eicu.lab_subset
                WHERE CAST(patientunitstayid AS text) = :patient_id
                ORDER BY labresultoffset
                """,
                {"patient_id": str(pid)},
            )
        else:
            if not table_exists("labs"):
                return {"labs": [], "count": 0}
            df = query_df("SELECT labresultoffset, labname, labresult FROM labs WHERE patientunitstayid = ? ORDER BY labresultoffset", (pid,))
        if not df.empty and "offset_hours" not in df.columns:
            df["offset_hours"] = pd.to_numeric(df["labresultoffset"], errors="coerce") / 60.0
        rows = sanitize_df(df)
        return {"labs": rows, "count": len(rows)}
    except Exception as exc:
        log.error("get_labs(%s) error: %s", pid, exc, exc_info=True)
        return {"labs": [], "count": 0}


def get_diagnosis(pid: int) -> dict:
    try:
        if IS_POSTGRES:
            df = query_df(
                """
                SELECT
                    admitdxenteredoffset AS diagnosisoffset,
                    admitdxname AS diagnosisstring,
                    NULL::text AS icd9code,
                    NULL::text AS diagnosispriority
                FROM raw_eicu.admission_dx
                WHERE CAST(patientunitstayid AS text) = :patient_id
                ORDER BY admitdxenteredoffset
                """,
                {"patient_id": str(pid)},
            )
        else:
            if not table_exists("diagnosis"):
                return {"diagnosis": []}
            available = get_table_columns("diagnosis")
            selected = ", ".join([col for col in ["diagnosisoffset", "diagnosisstring", "icd9code", "diagnosispriority"] if col in available] or ["*"])
            df = query_df(f"SELECT {selected} FROM diagnosis WHERE patientunitstayid = ? ORDER BY diagnosisoffset", (pid,))
        return {"diagnosis": sanitize_df(df)}
    except Exception as exc:
        log.error("get_diagnosis(%s) error: %s", pid, exc, exc_info=True)
        return {"diagnosis": []}


def get_treatments(pid: int) -> dict:
    try:
        if IS_POSTGRES or not table_exists("treatments"):
            return {"treatments": []}
        return {"treatments": sanitize_df(query_df("SELECT * FROM treatments WHERE patientunitstayid = ? ORDER BY treatmentoffset", (pid,)))}
    except Exception as exc:
        log.error("get_treatments(%s) error: %s", pid, exc, exc_info=True)
        return {"treatments": []}


def get_medications(pid: int) -> dict:
    try:
        if IS_POSTGRES:
            return {"medications": [], "infusions": [], "has_vasopressor": False}
        meds = sanitize_df(query_df("SELECT * FROM medications WHERE patientunitstayid = ? ORDER BY drugstartoffset", (pid,))) if table_exists("medications") else []
        infusions = sanitize_df(query_df("SELECT * FROM infusions WHERE patientunitstayid = ? ORDER BY infusionoffset", (pid,))) if table_exists("infusions") else []
        return {"medications": meds, "infusions": infusions, "has_vasopressor": False}
    except Exception as exc:
        log.error("get_medications(%s) error: %s", pid, exc, exc_info=True)
        return {"medications": [], "infusions": [], "has_vasopressor": False}


def get_fluid_balance(pid: int) -> dict:
    empty = {"total_intake": None, "total_output": None, "fluid_balance": None, "urine_per_hour": None, "hourly": [], "summary": [], "timeline": []}
    try:
        patient = get_patient_by_id(pid)
        if not patient:
            return empty
        payload = {"total_intake": None, "total_output": None, "fluid_balance": patient.get("fluid_balance_24h"), "urine_per_hour": patient.get("urine_output_per_hour"), "hourly": []}
        payload["summary"] = [payload.copy()]
        payload["timeline"] = []
        return payload
    except Exception as exc:
        log.error("get_fluid_balance(%s) error: %s", pid, exc, exc_info=True)
        return empty


def get_ventilation(pid: int) -> dict:
    empty = {"on_ventilator": False, "airwaytype": None, "peep_mean": None, "fio2_mean": None, "duration_hours": None, "records": [], "ventilation": []}
    try:
        patient = get_patient_by_id(pid)
        if not patient:
            return empty
        payload = {"on_ventilator": bool(patient.get("on_ventilator", 0)), "airwaytype": None, "peep_mean": patient.get("peep_mean"), "fio2_mean": patient.get("fio2_mean"), "duration_hours": None, "records": []}
        payload["ventilation"] = payload["records"]
        return payload
    except Exception as exc:
        log.error("get_ventilation(%s) error: %s", pid, exc, exc_info=True)
        return empty


def get_comorbidities(pid: int) -> dict:
    default = {"has_diabetes": False, "has_chf": False, "has_copd": False, "has_ckd": False, "has_hypertension": False, "has_immunosuppression": False, "comorbidities": []}
    try:
        patient = get_patient_by_id(pid)
        if not patient:
            return default
        result = {
            "has_diabetes": bool(patient.get("has_diabetes", 0)),
            "has_chf": bool(patient.get("has_chf", 0)),
            "has_copd": bool(patient.get("has_copd", 0)),
            "has_ckd": bool(patient.get("has_ckd", 0)),
            "has_hypertension": bool(patient.get("has_hypertension", 0)),
            "has_immunosuppression": bool(patient.get("has_immunosuppression", 0)),
        }
        result["comorbidities"] = [result]
        return result
    except Exception as exc:
        log.error("get_comorbidities(%s) error: %s", pid, exc, exc_info=True)
        return default
