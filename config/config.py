import os
from pathlib import Path

# === PATHS ===
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"

# Raw eICU data path - UPDATE THIS to your local path
EICU_RAW_PATH = Path(r"D:\physionet-data\eicu\eicu-collaborative-research-database-2.0")

# Processed data
SQLITE_DB_PATH = DATA_DIR / "icu_data.db"
PARQUET_DIR = DATA_DIR / "parquet"

# Files to load
EICU_FILES = [
    "patient.csv.gz",
    "vitalPeriodic.csv.gz",
    "lab.csv.gz",
    "diagnosis.csv.gz",
    "treatment.csv.gz",
    "apachePatientResult.csv.gz",
]

# Model paths
MORTALITY_MODEL_PATH = MODELS_DIR / "mortality_model.pkl"
SCALER_PATH = MODELS_DIR / "scaler.pkl"
FEATURES_PATH = MODELS_DIR / "feature_names.pkl"

# Processing
CHUNK_SIZE = 50000
MAX_PATIENTS_CACHE = 200  # how many patients to store in processed DB
VITALS_SAMPLE_ROWS = 500000  # limit vitalPeriodic rows for efficiency

# API
API_HOST = "127.0.0.1"
API_PORT = 8000

# Ensure dirs exist
DATA_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)
PARQUET_DIR.mkdir(exist_ok=True)
