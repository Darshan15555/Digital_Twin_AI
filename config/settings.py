from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv


class Settings:
    def __init__(self) -> None:
        base_dir = Path(__file__).resolve().parent.parent
        load_dotenv(base_dir / ".env")
        self.BASE_DIR = base_dir
        self.EICU_RAW_PATH = os.getenv("EICU_RAW_PATH", r"D:\physionet-data\eicu\eicu-collaborative-research-database-2.0")
        self.DATA_DIR = os.getenv("DATA_DIR", "data")
        self.PARQUET_DIR = os.getenv("PARQUET_DIR", "data/parquet")
        self.SQLITE_DB_PATH = os.getenv("SQLITE_DB_PATH", "data/icu_data.db")
        self.DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{self.SQLITE_DB_PATH}")
        self.MODELS_DIR = os.getenv("MODELS_DIR", "models")
        self.MORTALITY_MODEL_PATH = os.getenv("MORTALITY_MODEL_PATH", "models/mortality_rf.pkl")
        self.LOS_MODEL_PATH = os.getenv("LOS_MODEL_PATH", "models/los_rf.pkl")
        self.MORTALITY_SCALER_PATH = os.getenv("MORTALITY_SCALER_PATH", "models/mortality_scaler.pkl")
        self.MORTALITY_FEATURES_PATH = os.getenv("MORTALITY_FEATURES_PATH", "models/feature_names.pkl")
        self.EARLY_MORTALITY_MODEL_PATH = os.getenv("EARLY_MORTALITY_MODEL_PATH", "models/mortality_postgres_rf_early.pkl")
        self.EARLY_MORTALITY_IMPUTER_PATH = os.getenv("EARLY_MORTALITY_IMPUTER_PATH", "models/mortality_postgres_imputer_early.pkl")
        self.EARLY_MORTALITY_FEATURES_PATH = os.getenv("EARLY_MORTALITY_FEATURES_PATH", "models/feature_names_postgres_early.pkl")
        self.API_HOST = os.getenv("API_HOST", "127.0.0.1")
        self.API_PORT = int(os.getenv("API_PORT", "8000"))
        self.API_RELOAD = os.getenv("API_RELOAD", "false").lower() == "true"
        self.DEFAULT_PATIENT_LIMIT = int(os.getenv("DEFAULT_PATIENT_LIMIT", "100"))
        self.MAX_VITALS_PER_PATIENT = int(os.getenv("MAX_VITALS_PER_PATIENT", "400"))
        self.EARLY_MORTALITY_THRESHOLD = float(os.getenv("EARLY_MORTALITY_THRESHOLD", "0.60"))

    @property
    def db_path(self) -> Path:
        return (self.BASE_DIR / self.SQLITE_DB_PATH).resolve()

    @property
    def models_path(self) -> Path:
        return (self.BASE_DIR / self.MODELS_DIR).resolve()

    def resolve_path(self, raw_path: str) -> Path:
        return (self.BASE_DIR / raw_path).resolve()


settings = Settings()

