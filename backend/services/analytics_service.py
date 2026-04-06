from __future__ import annotations

import logging

import pandas as pd

from backend.dependencies import IS_POSTGRES, get_table_columns, query_df, table_exists
from utils.sanitize import sanitize_df

log = logging.getLogger(__name__)


def get_unit_breakdown() -> list:
    try:
        if IS_POSTGRES:
            return []
        available = get_table_columns("patients")
        if "unittype" not in available:
            return []
        parts = ["unittype", "COUNT(*) AS count"]
        if "hospital_mortality" in available:
            parts.append("ROUND(AVG(CAST(hospital_mortality AS FLOAT))*100,2) AS mortality_pct")
        if "apachescore" in available:
            parts.append("ROUND(AVG(apachescore),1) AS avg_apache")
        return sanitize_df(query_df(f"SELECT {', '.join(parts)} FROM patients WHERE unittype IS NOT NULL GROUP BY unittype ORDER BY count DESC"))
    except Exception as exc:
        log.error("get_unit_breakdown error: %s", exc, exc_info=True)
        return []


def get_mortality_by_age() -> list:
    try:
        if IS_POSTGRES:
            df = query_df(
                """
                SELECT
                    CASE WHEN p.age = '> 89' THEN 90 ELSE NULLIF(p.age, '')::numeric END AS age,
                    m.hospital_mortality
                FROM ml_prep.ml_dataset m
                LEFT JOIN raw_eicu.patient p
                    ON p.patientunitstayid = m.patientunitstayid
                WHERE p.age IS NOT NULL
                """
            )
        else:
            if "age" not in get_table_columns("patients"):
                return []
            df = query_df("SELECT age, hospital_mortality FROM patients WHERE age IS NOT NULL AND age > 0 AND age < 120")
        if df.empty:
            return []
        df["age"] = pd.to_numeric(df["age"], errors="coerce")
        df = df.dropna(subset=["age"])
        df["age_group"] = pd.cut(df["age"], bins=[0, 18, 40, 60, 75, 120], labels=["<18", "18-40", "40-60", "60-75", "75+"])
        result = df.groupby("age_group", observed=True).agg(count=("age", "count"), mortality_pct=("hospital_mortality", lambda x: pd.to_numeric(x, errors="coerce").mean() * 100)).reset_index()
        result["age_group"] = result["age_group"].astype(str)
        return sanitize_df(result)
    except Exception as exc:
        log.error("get_mortality_by_age error: %s", exc, exc_info=True)
        return []


def get_top_diagnoses(limit: int = 20) -> list:
    try:
        if IS_POSTGRES:
            return sanitize_df(
                query_df(
                    """
                    SELECT admitdxname AS diagnosisstring, COUNT(*) AS count
                    FROM raw_eicu.admission_dx
                    WHERE admitdxname IS NOT NULL
                    GROUP BY admitdxname
                    ORDER BY count DESC
                    LIMIT :limit
                    """,
                    {"limit": limit},
                )
            )
        diag_table = "diagnosis" if table_exists("diagnosis") else "diagnoses" if table_exists("diagnoses") else None
        if not diag_table:
            return []
        available = get_table_columns(diag_table)
        diag_col = next((c for c in ["diagnosisstring", "diagnosis", "diagnosisname"] if c in available), None)
        if not diag_col:
            return []
        return sanitize_df(query_df(f"SELECT {diag_col} AS diagnosisstring, COUNT(*) AS count FROM {diag_table} WHERE {diag_col} IS NOT NULL GROUP BY {diag_col} ORDER BY count DESC LIMIT ?", (limit,)))
    except Exception as exc:
        log.error("get_top_diagnoses error: %s", exc, exc_info=True)
        return []


def get_vasopressor_usage() -> list:
    try:
        return []
    except Exception as exc:
        log.error("get_vasopressor_usage error: %s", exc, exc_info=True)
        return []


def get_ventilator_usage() -> list:
    try:
        return []
    except Exception as exc:
        log.error("get_ventilator_usage error: %s", exc, exc_info=True)
        return []


def get_fluid_balance_by_outcome() -> list:
    try:
        return []
    except Exception as exc:
        log.error("get_fluid_balance_by_outcome error: %s", exc, exc_info=True)
        return []
