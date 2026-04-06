from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv


BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")


DEFAULT_IMPORTANT_FILES = [
    "patient.csv.gz",
    "admissionDx.csv.gz",
    "lab.csv.gz",
    "vitalPeriodic.csv.gz",
]


@dataclass(slots=True)
class PipelineConfig:
    data_path: Path = field(
        default_factory=lambda: Path(
            os.getenv(
                "DATA_PATH",
                os.getenv(
                    "EICU_RAW_PATH",
                    r"D:\physionet-data\eicu",
                ),
            )
        )
    )
    database_url: str = field(
        default_factory=lambda: os.getenv(
            "DATABASE_URL", "postgresql+psycopg://postgres:postgres@localhost:5432/icu_db"
        )
    )
    schema_name: str = field(default_factory=lambda: os.getenv("DB_SCHEMA", "raw_eicu"))
    chunk_size: int = field(default_factory=lambda: int(os.getenv("CHUNK_SIZE", "2000")))
    insert_batch_size: int = field(default_factory=lambda: int(os.getenv("INSERT_BATCH_SIZE", "10000")))
    max_workers: int = field(default_factory=lambda: max(1, int(os.getenv("MAX_WORKERS", "2"))))
    log_level: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))
    csv_fallback_tables: tuple[str, ...] = field(
        default_factory=lambda: tuple(
            item.strip().lower()
            for item in os.getenv("CSV_FALLBACK_TABLES", "lab.csv.gz,nurseCharting.csv.gz,vitalPeriodic.csv.gz").split(",")
            if item.strip()
        )
    )
    important_files: tuple[str, ...] = field(
        default_factory=lambda: tuple(
            item.strip()
            for item in os.getenv("IMPORTANT_FILES", ",".join(DEFAULT_IMPORTANT_FILES)).split(",")
            if item.strip()
        )
    )

    def validate(self) -> None:
        if not self.data_path.exists():
            raise FileNotFoundError(f"Configured DATA_PATH does not exist: {self.data_path}")
        if self.chunk_size <= 0:
            raise ValueError("CHUNK_SIZE must be greater than 0")
        if self.insert_batch_size <= 0:
            raise ValueError("INSERT_BATCH_SIZE must be greater than 0")
        if self.max_workers <= 0:
            raise ValueError("MAX_WORKERS must be greater than 0")
