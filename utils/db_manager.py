"""
SQLite helpers for the ICU analytics system.
"""

from __future__ import annotations

import logging
import sqlite3

import pandas as pd

from config.config import SQLITE_DB_PATH

log = logging.getLogger(__name__)

INDEXED_TABLES = [
    "patients",
    "vitals_periodic",
    "vitals_aperiodic",
    "labs",
    "diagnosis",
    "treatments",
    "medications",
    "infusions",
    "nurse_charting",
    "respiratory_care",
    "intake_output_summary",
    "comorbidities",
    "ml_dataset",
]


def get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(str(SQLITE_DB_PATH), check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    return conn


def init_db() -> None:
    conn = get_conn()
    conn.close()
    log.info("SQLite database ready at %s", SQLITE_DB_PATH)


def write_table(name: str, df: pd.DataFrame, if_exists: str = "replace") -> None:
    conn = get_conn()
    df.to_sql(name, conn, if_exists=if_exists, index=False)
    if "patientunitstayid" in df.columns:
        conn.execute(f"CREATE INDEX IF NOT EXISTS idx_{name}_pid ON {name}(patientunitstayid)")
    conn.commit()
    conn.close()
    log.info("Saved %s with %s rows", name, f"{len(df):,}")


def write_processed_tables(tables: dict[str, pd.DataFrame]) -> None:
    for name, df in tables.items():
        write_table(name, df)


def table_count(name: str) -> int:
    conn = get_conn()
    count = conn.execute(f"SELECT COUNT(*) FROM {name}").fetchone()[0]
    conn.close()
    return int(count)


def query_df(sql: str, params: tuple | list | None = None) -> pd.DataFrame:
    conn = get_conn()
    df = pd.read_sql(sql, conn, params=params)
    conn.close()
    return df


def get_stats() -> dict:
    stats = query_df(
        """
        SELECT
            COUNT(*) AS total_patients,
            AVG(hospital_mortality) AS mortality_rate,
            AVG(apachescore) AS avg_apache_score,
            AVG(icu_los_hours) AS avg_icu_los_hours
        FROM ml_dataset
        """
    ).iloc[0].to_dict()
    return stats


def get_patients_filtered(
    limit: int = 100,
    offset: int = 0,
    search: str | None = None,
    unit: str | None = None,
    risk_only: bool = False,
    age_min: int | None = None,
    age_max: int | None = None,
    on_vasopressor: bool | None = None,
    on_ventilator: bool | None = None,
) -> pd.DataFrame:
    sql = "SELECT * FROM ml_dataset WHERE 1=1"
    params: list = []
    if search:
        if search.isdigit():
            sql += " AND patientunitstayid = ?"
            params.append(int(search))
        else:
            sql += " AND (unittype LIKE ? OR apacheadmissiondx LIKE ? OR gender LIKE ?)"
            like = f"%{search}%"
            params.extend([like, like, like])
    if unit:
        sql += " AND unittype = ?"
        params.append(unit)
    if risk_only:
        sql += " AND (predicted_mortality_risk >= 0.5 OR qsofa_score >= 2 OR sepsis_risk = 1)"
    if age_min is not None:
        sql += " AND age >= ?"
        params.append(age_min)
    if age_max is not None:
        sql += " AND age <= ?"
        params.append(age_max)
    if on_vasopressor is not None:
        sql += " AND on_vasopressor = ?"
        params.append(int(on_vasopressor))
    if on_ventilator is not None:
        sql += " AND on_ventilator = ?"
        params.append(int(on_ventilator))
    sql += " ORDER BY patientunitstayid LIMIT ? OFFSET ?"
    params.extend([limit, offset])
    return query_df(sql, tuple(params))


def get_patient(patient_id: int) -> pd.DataFrame:
    return query_df("SELECT * FROM ml_dataset WHERE patientunitstayid = ?", (patient_id,))


def get_patient_table(table: str, patient_id: int, order_by: str | None = None) -> pd.DataFrame:
    sql = f"SELECT * FROM {table} WHERE patientunitstayid = ?"
    if order_by:
        sql += f" ORDER BY {order_by}"
    return query_df(sql, (patient_id,))


def update_prediction_fields(patient_id: int, fields: dict[str, float | int | str]) -> None:
    assignments = ", ".join(f"{key} = ?" for key in fields)
    params = list(fields.values()) + [patient_id]
    conn = get_conn()
    conn.execute(f"UPDATE ml_dataset SET {assignments} WHERE patientunitstayid = ?", params)
    conn.commit()
    conn.close()


def analytics_query(sql: str) -> pd.DataFrame:
    return query_df(sql)
