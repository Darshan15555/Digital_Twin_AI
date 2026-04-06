from __future__ import annotations

import os
from dataclasses import dataclass

from ingestion.config import PipelineConfig


@dataclass(slots=True)
class PreprocessingConfig:
    database_url: str
    raw_schema: str
    target_schema: str
    lab_source_table: str
    missing_numeric_strategy: str = "median"
    missing_categorical_strategy: str = "unknown"
    outlier_strategy: str = "clip"
    outlier_method: str = "iqr"
    encoding_strategy: str = "one_hot"


def load_preprocessing_config() -> PreprocessingConfig:
    ingestion = PipelineConfig()
    return PreprocessingConfig(
        database_url=ingestion.database_url,
        raw_schema=ingestion.schema_name,
        target_schema=os.getenv("PREPROCESS_SCHEMA", "ml_prep"),
        lab_source_table=os.getenv("LAB_SOURCE_TABLE", "lab_subset"),
        missing_numeric_strategy=os.getenv("MISSING_NUMERIC_STRATEGY", "median").lower(),
        missing_categorical_strategy=os.getenv("MISSING_CATEGORICAL_STRATEGY", "unknown").lower(),
        outlier_strategy=os.getenv("OUTLIER_STRATEGY", "clip").lower(),
        outlier_method=os.getenv("OUTLIER_METHOD", "iqr").lower(),
        encoding_strategy=os.getenv("ENCODING_STRATEGY", "one_hot").lower(),
    )
