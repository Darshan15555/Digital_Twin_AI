from __future__ import annotations

import logging
from pathlib import Path

from backend.dependencies import IS_POSTGRES, get_table_columns, query_df
from config.settings import settings
from utils.sanitize import safe_float, safe_int, sanitize_df, sanitize_row

log = logging.getLogger(__name__)


def get_available_patient_columns() -> list[str]:
    return get_table_columns("ml_dataset", "ml_prep") if IS_POSTGRES else get_table_columns("patients")


def build_patient_select(requested_cols: list[str]) -> str:
    available = get_available_patient_columns()
    select_cols = [column for column in requested_cols if column in available]
    return ", ".join(select_cols or ["patientunitstayid"])


def get_system_stats() -> dict:
    try:
        if IS_POSTGRES:
            df = query_df(
                """
                SELECT
                    COUNT(*) AS total_patients,
                    AVG(m.hospital_mortality) * 100 AS mortality_rate,
                    AVG(CASE WHEN p.age = '> 89' THEN 90 ELSE NULLIF(p.age, '')::numeric END) AS avg_age,
                    NULL::numeric AS avg_apache_score,
                    AVG(m.icu_los_hours) AS avg_icu_los_hours,
                    0::bigint AS total_on_vasopressor,
                    0::bigint AS total_on_ventilator,
                    COUNT(*) FILTER (WHERE m.qsofa_score >= 2) AS total_sepsis_risk
                FROM ml_prep.ml_dataset m
                LEFT JOIN raw_eicu.patient p
                    ON p.patientunitstayid = m.patientunitstayid
                """
            )
            row = df.iloc[0].to_dict() if not df.empty else {}
            return {
                "total_patients": safe_int(row.get("total_patients"), 0),
                "mortality_rate": safe_float(row.get("mortality_rate"), 0.0),
                "avg_age": safe_float(row.get("avg_age"), 0.0),
                "avg_apache_score": safe_float(row.get("avg_apache_score"), 0.0) or 0.0,
                "avg_icu_los_hours": safe_float(row.get("avg_icu_los_hours"), 0.0),
                "total_on_vasopressor": safe_int(row.get("total_on_vasopressor"), 0),
                "total_on_ventilator": safe_int(row.get("total_on_ventilator"), 0),
                "total_sepsis_risk": safe_int(row.get("total_sepsis_risk"), 0),
                "model_ready": Path(settings.resolve_path(settings.EARLY_MORTALITY_MODEL_PATH)).exists() or Path(settings.resolve_path(settings.MORTALITY_MODEL_PATH)).exists(),
            }

        available = get_available_patient_columns()
        stats = {}
        df = query_df("SELECT COUNT(*) AS n FROM patients")
        stats["total_patients"] = safe_int(df.iloc[0]["n"], 0) if not df.empty else 0
        stats["mortality_rate"] = 0.0
        if "hospital_mortality" in available:
            df = query_df("SELECT AVG(CAST(hospital_mortality AS FLOAT)) * 100 AS r FROM patients")
            stats["mortality_rate"] = safe_float(df.iloc[0]["r"] if not df.empty else None, 0.0)
        stats["avg_age"] = 0.0
        if "age" in available:
            df = query_df("SELECT AVG(age) AS a FROM patients WHERE age IS NOT NULL AND age > 0")
            stats["avg_age"] = safe_float(df.iloc[0]["a"] if not df.empty else None, 0.0)
        stats["avg_apache_score"] = 0.0
        if "apachescore" in available:
            df = query_df("SELECT AVG(apachescore) AS s FROM patients WHERE apachescore IS NOT NULL")
            stats["avg_apache_score"] = safe_float(df.iloc[0]["s"] if not df.empty else None, 0.0)
        stats["avg_icu_los_hours"] = 0.0
        los_col = next((c for c in ["icu_los_hours", "unitdischargeoffset"] if c in available), None)
        if los_col:
            df = query_df(f"SELECT AVG({los_col}) AS l FROM patients WHERE {los_col} IS NOT NULL AND {los_col} > 0")
            raw_los = safe_float(df.iloc[0]["l"] if not df.empty else None, 0.0)
            stats["avg_icu_los_hours"] = raw_los / 60 if los_col == "unitdischargeoffset" else raw_los
        stats["total_on_vasopressor"] = 0
        stats["total_on_ventilator"] = 0
        stats["total_sepsis_risk"] = 0
        stats["model_ready"] = Path(settings.resolve_path(settings.MORTALITY_MODEL_PATH)).exists()
        return stats
    except Exception as exc:
        log.error("get_system_stats error: %s", exc, exc_info=True)
        return {"total_patients": 0, "mortality_rate": 0.0, "avg_age": 0.0, "avg_apache_score": 0.0, "avg_icu_los_hours": 0.0, "total_on_vasopressor": 0, "total_on_ventilator": 0, "total_sepsis_risk": 0, "model_ready": False, "error": str(exc)}


def get_patient_list(limit: int = 100, offset: int = 0, search: str | None = None, unit_type: str | None = None, high_risk_only: bool = False, on_vasopressor: bool = False, on_ventilator: bool = False, age_min: float | None = None, age_max: float | None = None) -> dict:
    try:
        if IS_POSTGRES:
            sql = """
                SELECT
                    m.patientunitstayid,
                    CASE WHEN p.age = '> 89' THEN 90 ELSE NULLIF(p.age, '')::numeric END AS age,
                    p.gender,
                    p.ethnicity,
                    NULL::text AS unittype,
                    NULL::numeric AS apachescore,
                    m.hospital_mortality,
                    NULL::numeric AS ml_risk_score,
                    NULL::integer AS ml_prediction,
                    NULL::numeric AS los_prediction_hours,
                    m.qsofa_score,
                    CASE WHEN m.qsofa_score >= 2 THEN 1 ELSE 0 END AS sepsis_risk,
                    0::integer AS on_vasopressor,
                    0::integer AS on_ventilator,
                    NULL::numeric AS fluid_balance_24h,
                    0::integer AS has_diabetes,
                    0::integer AS has_chf,
                    0::integer AS has_copd,
                    0::integer AS has_ckd,
                    0::integer AS has_hypertension,
                    m.icu_los_hours
                FROM ml_prep.ml_dataset m
                LEFT JOIN raw_eicu.patient p
                    ON p.patientunitstayid = m.patientunitstayid
                WHERE 1=1
            """
            params: dict[str, object] = {"limit": limit, "offset": offset}
            if search:
                if search.isdigit():
                    sql += " AND m.patientunitstayid = :patient_id"
                    params["patient_id"] = int(search)
                else:
                    sql += " AND (COALESCE(p.gender, '') ILIKE :search OR COALESCE(p.ethnicity, '') ILIKE :search OR COALESCE(p.apacheadmissiondx, '') ILIKE :search)"
                    params["search"] = f"%{search}%"
            if age_min is not None:
                sql += " AND (CASE WHEN p.age = '> 89' THEN 90 ELSE NULLIF(p.age, '')::numeric END) >= :age_min"
                params["age_min"] = age_min
            if age_max is not None:
                sql += " AND (CASE WHEN p.age = '> 89' THEN 90 ELSE NULLIF(p.age, '')::numeric END) <= :age_max"
                params["age_max"] = age_max
            if high_risk_only:
                sql += " AND m.qsofa_score >= 2"
            count_df = query_df(f"SELECT COUNT(*) AS n FROM ({sql}) base", params)
            total = safe_int(count_df.iloc[0]["n"] if not count_df.empty else 0, 0)
            sql += " ORDER BY m.patientunitstayid LIMIT :limit OFFSET :offset"
            return {"patients": sanitize_df(query_df(sql, params)), "total": total}

        available = get_available_patient_columns()
        conditions = ["1=1"]
        params: list = []
        if search:
            try:
                conditions.append("patientunitstayid = ?")
                params.append(int(search))
            except ValueError:
                if "unittype" in available:
                    conditions.append("(unittype LIKE ? OR gender LIKE ?)")
                    params.extend([f"%{search}%", f"%{search}%"])
        if unit_type and "unittype" in available:
            conditions.append("unittype = ?")
            params.append(unit_type)
        if high_risk_only and "qsofa_score" in available:
            conditions.append("qsofa_score >= 2")
        if on_vasopressor and "on_vasopressor" in available:
            conditions.append("on_vasopressor = 1")
        if on_ventilator and "on_ventilator" in available:
            conditions.append("on_ventilator = 1")
        if age_min is not None and "age" in available:
            conditions.append("age >= ?")
            params.append(age_min)
        if age_max is not None and "age" in available:
            conditions.append("age <= ?")
            params.append(age_max)
        where_clause = " AND ".join(conditions)
        count_df = query_df(f"SELECT COUNT(*) AS n FROM patients WHERE {where_clause}", tuple(params))
        total = safe_int(count_df.iloc[0]["n"] if not count_df.empty else 0, 0)
        df = query_df(f"SELECT * FROM patients WHERE {where_clause} ORDER BY patientunitstayid LIMIT ? OFFSET ?", tuple(params + [limit, offset]))
        return {"patients": sanitize_df(df), "total": total}
    except Exception as exc:
        log.error("get_patient_list error: %s", exc, exc_info=True)
        return {"patients": [], "total": 0, "error": str(exc)}


def get_patient_by_id(pid: int) -> dict | None:
    try:
        if IS_POSTGRES:
            df = query_df(
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
                    CASE WHEN m.qsofa_score >= 2 THEN 1 ELSE 0 END AS sepsis_risk,
                    NULL::numeric AS fluid_balance_24h
                FROM ml_prep.ml_dataset m
                LEFT JOIN raw_eicu.patient p
                    ON p.patientunitstayid = m.patientunitstayid
                WHERE m.patientunitstayid = :patient_id
                """,
                {"patient_id": pid},
            )
        else:
            df = query_df("SELECT * FROM patients WHERE patientunitstayid = ?", (pid,))
        return None if df.empty else sanitize_row(df.iloc[0].to_dict())
    except Exception as exc:
        log.error("get_patient_by_id(%s) error: %s", pid, exc, exc_info=True)
        return None
