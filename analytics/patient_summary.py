from __future__ import annotations

import argparse
import logging
import os
import time
from dataclasses import dataclass

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from ingestion.config import PipelineConfig


log = logging.getLogger(__name__)


def safe_numeric(column_ref: str) -> str:
    return (
        "CASE "
        f"WHEN {column_ref} IS NULL THEN NULL "
        f"WHEN trim({column_ref}::text) IN ('', 'nan', 'None', '<NA>') THEN NULL "
        f"WHEN trim({column_ref}::text) ~ '^-?[0-9]+(\\.[0-9]+)?$' THEN trim({column_ref}::text)::double precision "
        "ELSE NULL "
        "END"
    )


def patient_summary_sql(raw_schema: str) -> str:
    return f"""
WITH patient_base AS (
    SELECT
        patientunitstayid,
        patienthealthsystemstayid,
        hospitalid,
        gender,
        age AS age_raw,
        CASE
            WHEN age IS NULL THEN NULL
            WHEN trim(age::text) IN ('', 'nan', 'None', '<NA>') THEN NULL
            WHEN trim(age::text) IN ('> 89', '>89') THEN 90
            WHEN trim(age::text) ~ '^[0-9]+(\\.[0-9]+)?$' THEN round((trim(age::text))::numeric)::int
            ELSE NULL
        END AS age_years,
        ethnicity,
        {safe_numeric("admissionheight")} AS admissionheight,
        {safe_numeric("admissionweight")} AS admissionweight,
        hospitaladmitsource,
        hospitaldischargestatus,
        unitadmitsource,
        unitdischargestatus,
        {safe_numeric("unitdischargeoffset")} AS unitdischargeoffset,
        {safe_numeric("hospitaldischargeoffset")} AS hospitaldischargeoffset,
        apacheadmissiondx
    FROM "{raw_schema}"."patient"
),
lab_summary AS (
    SELECT
        patientunitstayid,
        COUNT(*) AS lab_measurement_count,
        COUNT(DISTINCT lower(labname)) AS distinct_lab_name_count,
        MIN({safe_numeric("labresultoffset")}) AS first_lab_offset,
        MAX({safe_numeric("labresultoffset")}) AS last_lab_offset,
        AVG({safe_numeric("labresult")}) AS avg_lab_result_all,
        MAX({safe_numeric("labresult")}) AS max_lab_result_all,
        AVG({safe_numeric("labresult")}) FILTER (WHERE lower(labname) LIKE 'creatinine%') AS creatinine_avg,
        MAX({safe_numeric("labresult")}) FILTER (WHERE lower(labname) LIKE 'creatinine%') AS creatinine_max,
        AVG({safe_numeric("labresult")}) FILTER (WHERE lower(labname) LIKE 'glucose%') AS glucose_avg,
        MAX({safe_numeric("labresult")}) FILTER (WHERE lower(labname) LIKE 'glucose%') AS glucose_max,
        AVG({safe_numeric("labresult")}) FILTER (WHERE lower(labname) LIKE 'potassium%') AS potassium_avg,
        AVG({safe_numeric("labresult")}) FILTER (WHERE lower(labname) LIKE 'sodium%') AS sodium_avg,
        AVG({safe_numeric("labresult")}) FILTER (WHERE lower(labname) LIKE 'wbc%') AS wbc_avg,
        AVG({safe_numeric("labresult")}) FILTER (WHERE lower(labname) LIKE 'hemoglobin%') AS hemoglobin_avg,
        AVG({safe_numeric("labresult")}) FILTER (WHERE lower(labname) LIKE 'platelet%') AS platelet_avg,
        AVG({safe_numeric("labresult")}) FILTER (WHERE lower(labname) LIKE 'bicarbonate%') AS bicarbonate_avg,
        AVG({safe_numeric("labresult")}) FILTER (WHERE lower(labname) LIKE 'bun%') AS bun_avg,
        AVG({safe_numeric("labresult")}) FILTER (WHERE lower(labname) LIKE 'lactate%') AS lactate_avg
    FROM "{raw_schema}"."lab"
    GROUP BY patientunitstayid
),
vital_summary AS (
    SELECT
        patientunitstayid,
        COUNT(*) AS vital_measurement_count,
        MIN({safe_numeric("observationoffset")}) AS first_vital_offset,
        MAX({safe_numeric("observationoffset")}) AS last_vital_offset,
        AVG({safe_numeric("temperature")}) AS avg_temperature,
        MIN({safe_numeric("temperature")}) AS min_temperature,
        MAX({safe_numeric("temperature")}) AS max_temperature,
        AVG({safe_numeric("sao2")}) AS avg_sao2,
        MIN({safe_numeric("sao2")}) AS min_sao2,
        AVG({safe_numeric("heartrate")}) AS avg_heartrate,
        MAX({safe_numeric("heartrate")}) AS max_heartrate,
        AVG({safe_numeric("respiration")}) AS avg_respiration,
        MAX({safe_numeric("respiration")}) AS max_respiration,
        AVG({safe_numeric("systemicsystolic")}) AS avg_systemic_systolic,
        MIN({safe_numeric("systemicsystolic")}) AS min_systemic_systolic,
        MAX({safe_numeric("systemicsystolic")}) AS max_systemic_systolic,
        AVG({safe_numeric("systemicdiastolic")}) AS avg_systemic_diastolic,
        AVG({safe_numeric("systemicmean")}) AS avg_systemic_mean
    FROM "{raw_schema}"."vital_periodic"
    GROUP BY patientunitstayid
),
admission_dx_summary AS (
    SELECT
        patientunitstayid,
        COUNT(*) AS admission_dx_count,
        MIN({safe_numeric("admitdxenteredoffset")}) AS first_admit_dx_offset,
        MAX({safe_numeric("admitdxenteredoffset")}) AS last_admit_dx_offset,
        STRING_AGG(DISTINCT admitdxname, ' | ' ORDER BY admitdxname) AS admit_dx_names
    FROM "{raw_schema}"."admission_dx"
    GROUP BY patientunitstayid
)
SELECT
    p.patientunitstayid,
    p.patienthealthsystemstayid,
    p.hospitalid,
    p.gender,
    p.age_raw,
    p.age_years,
    p.ethnicity,
    p.admissionheight,
    p.admissionweight,
    p.hospitaladmitsource,
    p.hospitaldischargestatus,
    p.unitadmitsource,
    p.unitdischargestatus,
    p.unitdischargeoffset,
    p.hospitaldischargeoffset,
    p.apacheadmissiondx,
    CASE
        WHEN lower(coalesce(p.hospitaldischargestatus, '')) LIKE '%%expired%%' THEN 1
        ELSE 0
    END AS hospital_mortality,
    CASE
        WHEN lower(coalesce(p.unitdischargestatus, '')) LIKE '%%expired%%' THEN 1
        ELSE 0
    END AS unit_mortality,
    round(p.unitdischargeoffset / 60.0, 2) AS icu_los_hours,
    round(p.hospitaldischargeoffset / 60.0, 2) AS hospital_los_hours,
    l.lab_measurement_count,
    l.distinct_lab_name_count,
    l.first_lab_offset,
    l.last_lab_offset,
    l.avg_lab_result_all,
    l.max_lab_result_all,
    l.creatinine_avg,
    l.creatinine_max,
    l.glucose_avg,
    l.glucose_max,
    l.potassium_avg,
    l.sodium_avg,
    l.wbc_avg,
    l.hemoglobin_avg,
    l.platelet_avg,
    l.bicarbonate_avg,
    l.bun_avg,
    l.lactate_avg,
    v.vital_measurement_count,
    v.first_vital_offset,
    v.last_vital_offset,
    v.avg_temperature,
    v.min_temperature,
    v.max_temperature,
    v.avg_sao2,
    v.min_sao2,
    v.avg_heartrate,
    v.max_heartrate,
    v.avg_respiration,
    v.max_respiration,
    v.avg_systemic_systolic,
    v.min_systemic_systolic,
    v.max_systemic_systolic,
    v.avg_systemic_diastolic,
    v.avg_systemic_mean,
    dx.admission_dx_count,
    dx.first_admit_dx_offset,
    dx.last_admit_dx_offset,
    dx.admit_dx_names
FROM patient_base p
LEFT JOIN lab_summary l
    ON p.patientunitstayid = l.patientunitstayid
LEFT JOIN vital_summary v
    ON p.patientunitstayid = v.patientunitstayid
LEFT JOIN admission_dx_summary dx
    ON p.patientunitstayid = dx.patientunitstayid
"""


@dataclass(slots=True)
class AnalyticsConfig:
    database_url: str
    raw_schema: str
    analytics_schema: str


def configure_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("analytics.log"),
        ],
    )


class PatientSummaryBuilder:
    def __init__(self, config: AnalyticsConfig) -> None:
        self.config = config
        self.engine: Engine = create_engine(config.database_url, future=True, pool_pre_ping=True)

    def build(self) -> int:
        started_at = time.perf_counter()
        with self.engine.begin() as connection:
            connection.execute(text(f'CREATE SCHEMA IF NOT EXISTS "{self.config.analytics_schema}"'))
            log.info(
                "Building analytics.patient_summary | raw_schema=%s analytics_schema=%s",
                self.config.raw_schema,
                self.config.analytics_schema,
            )
            connection.execute(text(f'DROP TABLE IF EXISTS "{self.config.analytics_schema}"."patient_summary"'))
            connection.execute(
                text(
                    f'CREATE TABLE "{self.config.analytics_schema}"."patient_summary" AS '
                    + patient_summary_sql(self.config.raw_schema)
                )
            )
            connection.execute(
                text(
                    f'CREATE INDEX IF NOT EXISTS "idx_patient_summary_pid" '
                    f'ON "{self.config.analytics_schema}"."patient_summary" ("patientunitstayid")'
                )
            )
            connection.execute(
                text(
                    f'CREATE INDEX IF NOT EXISTS "idx_patient_summary_hospital" '
                    f'ON "{self.config.analytics_schema}"."patient_summary" ("hospitalid")'
                )
            )
            connection.execute(text(f'ANALYZE "{self.config.analytics_schema}"."patient_summary"'))
            connection.execute(
                text(
                    "COMMENT ON TABLE "
                    f'"{self.config.analytics_schema}"."patient_summary" IS '
                    "'One-row-per-patient curated summary built from raw eICU landing tables.'"
                )
            )
            count = connection.execute(
                text(f'SELECT COUNT(*) FROM "{self.config.analytics_schema}"."patient_summary"')
            ).scalar_one()

        elapsed = time.perf_counter() - started_at
        log.info(
            "Built analytics.patient_summary successfully | rows=%s elapsed=%.2fs",
            f"{count:,}",
            elapsed,
        )
        return int(count)


def build_config() -> AnalyticsConfig:
    ingestion_config = PipelineConfig()
    return AnalyticsConfig(
        database_url=ingestion_config.database_url,
        raw_schema=ingestion_config.schema_name,
        analytics_schema=os.getenv("ANALYTICS_SCHEMA", "analytics"),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build analytics.patient_summary from raw_eicu tables.")
    parser.add_argument(
        "--analytics-schema",
        default=os.getenv("ANALYTICS_SCHEMA", "analytics"),
        help="Target schema for curated analytics tables.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = build_config()
    config.analytics_schema = args.analytics_schema
    configure_logging()
    builder = PatientSummaryBuilder(config)
    builder.build()


if __name__ == "__main__":
    main()
