from __future__ import annotations

import argparse
import logging
import time

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from preprocessing.config import PreprocessingConfig, load_preprocessing_config
from preprocessing.encoding import final_dataset_sql
from preprocessing.feature_engineering import (
    diagnosis_features_sql,
    lab_features_sql,
    patient_base_sql,
    vital_features_sql,
)
from preprocessing.report import build_report, save_report


log = logging.getLogger(__name__)


def configure_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("preprocessing.log"),
        ],
    )


class PreprocessingPipeline:
    def __init__(self, config: PreprocessingConfig) -> None:
        self.config = config
        self.engine: Engine = create_engine(config.database_url, future=True, pool_pre_ping=True)

    def run(self) -> dict:
        started_at = time.perf_counter()
        statements = [
            ('DROP TABLE IF EXISTS "{target_schema}"."ml_dataset"', "drop ml_dataset"),
            ('DROP TABLE IF EXISTS "{target_schema}"."diagnosis_features"', "drop diagnosis_features"),
            ('DROP TABLE IF EXISTS "{target_schema}"."vital_features"', "drop vital_features"),
            ('DROP TABLE IF EXISTS "{target_schema}"."lab_features"', "drop lab_features"),
            ('DROP TABLE IF EXISTS "{target_schema}"."patient_base_clean"', "drop patient_base_clean"),
            (patient_base_sql(self.config.raw_schema), "build patient_base_clean"),
            (lab_features_sql(self.config.raw_schema, self.config.lab_source_table), "build lab_features"),
            (vital_features_sql(self.config.raw_schema), "build vital_features"),
            (diagnosis_features_sql(self.config.raw_schema), "build diagnosis_features"),
            (final_dataset_sql(), "build ml_dataset"),
            ('CREATE INDEX IF NOT EXISTS "idx_ml_dataset_pid" ON "{target_schema}"."ml_dataset" ("patientunitstayid")', "index ml_dataset"),
            ('ANALYZE "{target_schema}"."ml_dataset"', "analyze ml_dataset"),
        ]

        with self.engine.begin() as connection:
            connection.execute(text(f'CREATE SCHEMA IF NOT EXISTS "{self.config.target_schema}"'))
            log.info(
                "Starting preprocessing | raw_schema=%s target_schema=%s lab_source_table=%s",
                self.config.raw_schema,
                self.config.target_schema,
                self.config.lab_source_table,
            )
            for sql_template, step in statements:
                log.info("Running step: %s", step)
                connection.execute(text(sql_template.format(target_schema=self.config.target_schema)))

        report = build_report(self.engine, self.config.target_schema)
        save_report(report)
        elapsed = time.perf_counter() - started_at
        log.info(
            "Preprocessing finished | target_schema=%s rows=%s elapsed=%.2fs",
            self.config.target_schema,
            f"{report['ml_dataset_rows']:,}",
            elapsed,
        )
        return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build ML-ready preprocessing tables in PostgreSQL.")
    parser.add_argument(
        "--target-schema",
        default=None,
        help="Target schema for preprocessed feature tables. Defaults to PREPROCESS_SCHEMA or ml_prep.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_preprocessing_config()
    if args.target_schema:
        config.target_schema = args.target_schema
    configure_logging()
    pipeline = PreprocessingPipeline(config)
    pipeline.run()


if __name__ == "__main__":
    main()
