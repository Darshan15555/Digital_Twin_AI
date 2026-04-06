from __future__ import annotations

import json
from pathlib import Path

from sqlalchemy import text
from sqlalchemy.engine import Engine


def build_report(engine: Engine, schema_name: str) -> dict:
    queries = {
        "patient_base_clean_rows": f'SELECT COUNT(*) FROM "{schema_name}"."patient_base_clean"',
        "lab_features_rows": f'SELECT COUNT(*) FROM "{schema_name}"."lab_features"',
        "vital_features_rows": f'SELECT COUNT(*) FROM "{schema_name}"."vital_features"',
        "diagnosis_features_rows": f'SELECT COUNT(*) FROM "{schema_name}"."diagnosis_features"',
        "ml_dataset_rows": f'SELECT COUNT(*) FROM "{schema_name}"."ml_dataset"',
        "ml_dataset_distinct_patients": f'SELECT COUNT(DISTINCT patientunitstayid) FROM "{schema_name}"."ml_dataset"',
    }
    report: dict[str, int] = {}
    with engine.begin() as connection:
        for name, sql in queries.items():
            report[name] = int(connection.execute(text(sql)).scalar_one())

        missing_summary = connection.execute(
            text(
                f"""
                SELECT json_build_object(
                    'admissionheight_nulls', SUM(CASE WHEN admissionheight IS NULL THEN 1 ELSE 0 END),
                    'admissionweight_nulls', SUM(CASE WHEN admissionweight IS NULL THEN 1 ELSE 0 END),
                    'creatinine_max_nulls', SUM(CASE WHEN creatinine_max IS NULL THEN 1 ELSE 0 END),
                    'heartrate_mean_nulls', SUM(CASE WHEN heartrate_mean IS NULL THEN 1 ELSE 0 END)
                )
                FROM "{schema_name}"."ml_dataset"
                """
            )
        ).scalar_one()
        report["missing_summary"] = missing_summary
    return report


def save_report(report: dict, output_path: str = "preprocessing_report.json") -> None:
    Path(output_path).write_text(json.dumps(report, indent=2), encoding="utf-8")
