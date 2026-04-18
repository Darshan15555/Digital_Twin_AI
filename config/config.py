import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
PARQUET_DIR = DATA_DIR / "parquet"
ARTIFACTS_DIR = DATA_DIR / "artifacts"

load_dotenv(BASE_DIR / ".env")

EICU_RAW_PATH = Path(os.getenv("EICU_RAW_PATH", "data/parquet"))
SQLITE_DB_PATH = DATA_DIR / "icu_data.db"

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "50000"))
LAB_CHUNK_SIZE = int(os.getenv("LAB_CHUNK_SIZE", "5000"))
VITAL_PERIODIC_MAX_ROWS = 1_000_000
VITAL_APERIODIC_MAX_ROWS = 500_000
INTAKE_OUTPUT_MAX_ROWS = 500_000

API_HOST = os.getenv("API_HOST", "127.0.0.1")
API_PORT = int(os.getenv("API_PORT", "8000"))
DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{SQLITE_DB_PATH}")

FILE_NAMES = {
    "patient": "patient.csv.gz",
    "vital_periodic": "vitalPeriodic.csv.gz",
    "vital_aperiodic": "vitalAperiodic.csv.gz",
    "lab": "lab.csv.gz",
    "diagnosis": "diagnosis.csv.gz",
    "treatment": "treatment.csv.gz",
    "apache": "apachePatientResult.csv.gz",
    "nurse_charting": "nurseCharting.csv.gz",
    "medication": "medication.csv.gz",
    "infusion_drug": "infusionDrug.csv.gz",
    "intake_output": "intakeOutput.csv.gz",
    "respiratory_care": "respiratoryCare.csv.gz",
    "hospital": "hospital.csv.gz",
    "past_history": "pastHistory.csv.gz",
}

PRIMARY_TABLES = [
    "patients",
    "vitals_periodic",
    "vitals_aperiodic",
    "labs",
    "diagnosis",
    "treatments",
    "medications",
    "infusions",
    "respiratory_care",
    "nurse_charting",
]

DERIVED_TABLES = [
    "intake_output_summary",
    "comorbidities",
]

NURSE_TARGETS = {
    "glasgow coma score": "gcs",
    "pain score/goal": "pain_score",
    "delirium scale/score": "delirium_score",
}

VASOPRESSOR_KEYWORDS = [
    "norepinephrine",
    "epinephrine",
    "dopamine",
    "vasopressin",
    "phenylephrine",
    "dobutamine",
]

COMORBIDITY_MAP = {
    "diabetes mellitus": "has_diabetes",
    "congestive heart failure": "has_chf",
    "copd": "has_copd",
    "renal failure/insufficiency (chronic)": "has_ckd",
    "hypertension": "has_hypertension",
    "immunosuppression": "has_immunosuppression",
}

LAB_RANGES = {
    "creatinine": (0, 50),
    "glucose": (10, 2000),
    "potassium": (1, 10),
    "sodium": (100, 200),
    "lactate": (0, 30),
    "hemoglobin": (1, 25),
    "wbc": (0, 200),
    "platelets": (0, 2000),
    "bicarbonate": (5, 50),
    "bun": (1, 300),
    "alt": (0, 10000),
    "ast": (0, 10000),
    "inr": (0, 20),
    "pco2": (5, 150),
    "po2": (10, 700),
    "ph": (6.5, 8.0),
}

for directory in (DATA_DIR, MODELS_DIR, PARQUET_DIR, ARTIFACTS_DIR):
    directory.mkdir(parents=True, exist_ok=True)
