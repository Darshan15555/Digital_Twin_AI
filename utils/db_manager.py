"""
Database helpers for the ICU analytics system.
"""

from __future__ import annotations

import logging
import sqlite3

import pandas as pd
from sqlalchemy import create_engine

from config.config import DATABASE_URL, SQLITE_DB_PATH

log = logging.getLogger(__name__)

POSTGRES_ENABLED = DATABASE_URL.startswith("postgresql")
pg_engine = create_engine(DATABASE_URL, future=True, pool_pre_ping=True) if POSTGRES_ENABLED else None

POSTGRES_TABLE_MAP = {
    "ml_dataset": '"ml_prep"."ml_dataset"',
    "patients": '"raw_eicu"."patient"',
    "vitals_periodic": '"raw_eicu"."vital_periodic"',
    "labs": '"raw_eicu"."lab_subset"',
    "diagnosis": '"raw_eicu"."admission_dx"',
}

EMPTY_PATIENT_TABLES = {
    "vitals_aperiodic",
    "treatments",
    "medications",
    "infusions",
    "nurse_charting",
    "respiratory_care",
    "intake_output_summary",
    "comorbidities",
}


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


def query_df(sql: str, params: tuple | list | dict | None = None) -> pd.DataFrame:
    if POSTGRES_ENABLED and pg_engine is not None:
        return pd.read_sql(sql, pg_engine, params=params)
    conn = get_conn()
    df = pd.read_sql(sql, conn, params=params)
    conn.close()
    return df


def get_stats() -> dict:
    if POSTGRES_ENABLED:
        stats = query_df(
            """
            SELECT
                COUNT(*) AS total_patients,
                AVG(m.hospital_mortality) AS mortality_rate,
                AVG(NULLIF(p.age, '> 89')::numeric) AS avg_age,
                NULL::numeric AS avg_apache_score,
                AVG(m.icu_los_hours) AS avg_icu_los_hours,
                0::bigint AS total_on_vasopressor,
                0::bigint AS total_on_ventilator,
                COUNT(*) FILTER (WHERE m.qsofa_score >= 2) AS total_sepsis_risk
            FROM ml_prep.ml_dataset m
            LEFT JOIN raw_eicu.patient p
                ON p.patientunitstayid = m.patientunitstayid
            """
        ).iloc[0].to_dict()
        return stats

    return query_df(
        """
        SELECT
            COUNT(*) AS total_patients,
            AVG(hospital_mortality) AS mortality_rate,
            AVG(apachescore) AS avg_apache_score,
            AVG(icu_los_hours) AS avg_icu_los_hours
        FROM ml_dataset
        """
    ).iloc[0].to_dict()


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
    if POSTGRES_ENABLED:
        sql = """
            SELECT
                m.patientunitstayid,
                CASE WHEN p.age = '> 89' THEN 90 ELSE NULLIF(p.age, '')::numeric END AS age,
                p.gender,
                p.ethnicity,
                NULL::text AS unittype,
                NULL::numeric AS apachescore,
                m.hospital_mortality,
                m.icu_los_hours,
                m.qsofa_score,
                m.admissionweight,
                0::integer AS on_vasopressor,
                0::integer AS on_ventilator,
                0::integer AS sepsis_risk,
                NULL::numeric AS fluid_balance_24h,
                NULL::numeric AS predicted_mortality_risk
            FROM ml_prep.ml_dataset m
            LEFT JOIN raw_eicu.patient p
                ON p.patientunitstayid = m.patientunitstayid
            WHERE 1=1
        """
        params: dict[str, object] = {"limit": limit, "offset": offset}
        if search:
            if search.isdigit():
                sql += " AND m.patientunitstayid = %(patient_id)s"
                params["patient_id"] = int(search)
            else:
                sql += " AND (COALESCE(p.gender, '') ILIKE %(search)s OR COALESCE(p.ethnicity, '') ILIKE %(search)s OR COALESCE(p.apacheadmissiondx, '') ILIKE %(search)s)"
                params["search"] = f"%{search}%"
        if age_min is not None:
            sql += " AND (CASE WHEN p.age = '> 89' THEN 90 ELSE NULLIF(p.age, '')::numeric END) >= %(age_min)s"
            params["age_min"] = age_min
        if age_max is not None:
            sql += " AND (CASE WHEN p.age = '> 89' THEN 90 ELSE NULLIF(p.age, '')::numeric END) <= %(age_max)s"
            params["age_max"] = age_max
        if risk_only:
            sql += " AND m.qsofa_score >= 2"
        sql += " ORDER BY m.patientunitstayid LIMIT %(limit)s OFFSET %(offset)s"
        return query_df(sql, params)

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
    if POSTGRES_ENABLED:
        return query_df(
            """
            SELECT
                m.*,
                CASE WHEN p.age = '> 89' THEN 90 ELSE NULLIF(p.age, '')::numeric END AS age,
                p.gender,
                p.ethnicity,
                p.apacheadmissiondx,
                p.hospitaladmitsource,
                p.unitadmitsource,
                NULL::text AS unittype,
                NULL::numeric AS apachescore,
                0::integer AS on_vasopressor,
                0::integer AS on_ventilator,
                0::integer AS sepsis_risk,
                NULL::numeric AS fluid_balance_24h,
                NULL::numeric AS predicted_mortality_risk
            FROM ml_prep.ml_dataset m
            LEFT JOIN raw_eicu.patient p
                ON p.patientunitstayid = m.patientunitstayid
            WHERE m.patientunitstayid = %(patient_id)s
            """,
            {"patient_id": patient_id},
        )
    return query_df("SELECT * FROM ml_dataset WHERE patientunitstayid = ?", (patient_id,))


def get_patient_table(table: str, patient_id: int, order_by: str | None = None) -> pd.DataFrame:
    if POSTGRES_ENABLED:
        if table in EMPTY_PATIENT_TABLES:
            return pd.DataFrame()
        mapped_table = POSTGRES_TABLE_MAP.get(table)
        if not mapped_table:
            return pd.DataFrame()
        sql = f"SELECT * FROM {mapped_table} WHERE patientunitstayid = %(patient_id)s"
        if order_by:
            sql += f" ORDER BY {order_by}"
        return query_df(sql, {"patient_id": patient_id})

    sql = f"SELECT * FROM {table} WHERE patientunitstayid = ?"
    if order_by:
        sql += f" ORDER BY {order_by}"
    return query_df(sql, (patient_id,))


def update_prediction_fields(patient_id: int, fields: dict[str, float | int | str]) -> None:
    if POSTGRES_ENABLED:
        log.info("Skipping prediction field update for PostgreSQL-backed read-only dataset | patient_id=%s", patient_id)
        return
    assignments = ", ".join(f"{key} = ?" for key in fields)
    params = list(fields.values()) + [patient_id]
    conn = get_conn()
    conn.execute(f"UPDATE ml_dataset SET {assignments} WHERE patientunitstayid = ?", params)
    conn.commit()
    conn.close()


def analytics_query(sql: str) -> pd.DataFrame:
    if not POSTGRES_ENABLED:
        return query_df(sql)
    lowered = " ".join(sql.lower().split())
    if "from ml_dataset" in lowered and "group by unittype" in lowered:
        return pd.DataFrame(columns=["unittype", "patient_count", "mortality_pct", "avg_apache", "avg_los_hours", "vasopressor_pct", "ventilator_pct"])
    if "select age, hospital_mortality from ml_dataset" in lowered:
        return query_df(
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
    if "from diagnosis" in lowered:
        return query_df(
            """
            SELECT admitdxname AS diagnosisstring, COUNT(*) AS count
            FROM raw_eicu.admission_dx
            GROUP BY admitdxname
            ORDER BY count DESC
            LIMIT 15
            """
        )
    if "avg(fluid_balance_24h)" in lowered:
        return pd.DataFrame(
            columns=["hospital_mortality", "avg_fluid_balance", "avg_urine_output_per_hour"]
        )
    return pd.DataFrame()
