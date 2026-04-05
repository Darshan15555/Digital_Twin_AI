"""
db_manager.py
Manages SQLite database for processed ICU data.
"""

import sqlite3
import pandas as pd
import logging
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from config.config import SQLITE_DB_PATH

log = logging.getLogger(__name__)


def get_conn():
    conn = sqlite3.connect(str(SQLITE_DB_PATH), check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    return conn


def init_db():
    conn = get_conn()
    c = conn.cursor()

    c.execute("""
    CREATE TABLE IF NOT EXISTS patients (
        patientunitstayid INTEGER PRIMARY KEY,
        age REAL,
        gender TEXT,
        ethnicity TEXT,
        admissionheight REAL,
        admissionweight REAL,
        unittype TEXT,
        unitadmitsource TEXT,
        unitdischargestatus TEXT,
        hospitaldischargestatus TEXT,
        hospital_mortality INTEGER,
        apachescore REAL,
        predictedhospitalmortality REAL,
        unitdischargeoffset REAL,
        hr_mean REAL, hr_std REAL,
        sao2_mean REAL, sao2_min REAL,
        resp_mean REAL,
        sbp_mean REAL, dbp_mean REAL, temp_mean REAL,
        ml_risk_score REAL,
        ml_prediction INTEGER
    )
    """)

    c.execute("""
    CREATE TABLE IF NOT EXISTS vitals (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        patientunitstayid INTEGER,
        observationoffset REAL,
        heartrate REAL,
        respiration REAL,
        sao2 REAL,
        systemicsystolic REAL,
        systemicdiastolic REAL,
        temperature REAL,
        cvp REAL
    )
    """)

    c.execute("CREATE INDEX IF NOT EXISTS idx_vitals_pid ON vitals(patientunitstayid)")

    c.execute("""
    CREATE TABLE IF NOT EXISTS labs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        patientunitstayid INTEGER,
        labresultoffset REAL,
        labname TEXT,
        labresult REAL
    )
    """)

    c.execute("CREATE INDEX IF NOT EXISTS idx_labs_pid ON labs(patientunitstayid)")

    c.execute("""
    CREATE TABLE IF NOT EXISTS diagnosis (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        patientunitstayid INTEGER,
        diagnosisoffset REAL,
        diagnosisstring TEXT,
        icd9code TEXT,
        diagnosispriority TEXT
    )
    """)

    c.execute("""
    CREATE TABLE IF NOT EXISTS treatments (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        patientunitstayid INTEGER,
        treatmentoffset REAL,
        treatmentstring TEXT
    )
    """)

    conn.commit()
    conn.close()
    log.info("SQLite DB initialized")


def save_patients_to_db(df: pd.DataFrame):
    """Save patient summary to SQLite patients table."""
    conn = get_conn()
    cols = [
        "patientunitstayid", "age", "gender", "ethnicity",
        "admissionheight", "admissionweight", "unittype",
        "unitadmitsource", "unitdischargestatus", "hospitaldischargestatus",
        "hospital_mortality", "apachescore", "predictedhospitalmortality",
        "unitdischargeoffset",
        "hr_mean", "hr_std", "sao2_mean", "sao2_min",
        "resp_mean", "sbp_mean", "dbp_mean", "temp_mean"
    ]
    available = [c for c in cols if c in df.columns]
    # Always include ml columns (as NULL) so UPDATE works later
    df = df[available].copy()
    df["ml_risk_score"] = None
    df["ml_prediction"] = None
    df.to_sql("patients", conn, if_exists="replace", index=False)
    conn.close()
    log.info(f"Saved {len(df)} patients to SQLite")


def save_vitals_to_db(df: pd.DataFrame):
    conn = get_conn()
    df.to_sql("vitals", conn, if_exists="replace", index=False)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_vitals_pid ON vitals(patientunitstayid)")
    conn.commit()
    conn.close()
    log.info(f"Saved {len(df):,} vitals rows to SQLite")


def save_labs_to_db(df: pd.DataFrame):
    conn = get_conn()
    df[["patientunitstayid", "labresultoffset", "labname", "labresult"]].to_sql(
        "labs", conn, if_exists="replace", index=False
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_labs_pid ON labs(patientunitstayid)")
    conn.commit()
    conn.close()
    log.info(f"Saved {len(df):,} lab rows to SQLite")


def save_diagnosis_to_db(df: pd.DataFrame):
    conn = get_conn()
    cols = ["patientunitstayid", "diagnosisoffset", "diagnosisstring", "icd9code", "diagnosispriority"]
    available = [c for c in cols if c in df.columns]
    df[available].to_sql("diagnosis", conn, if_exists="replace", index=False)
    conn.commit()
    conn.close()


def save_treatments_to_db(df: pd.DataFrame):
    conn = get_conn()
    df[["patientunitstayid", "treatmentoffset", "treatmentstring"]].to_sql(
        "treatments", conn, if_exists="replace", index=False
    )
    conn.commit()
    conn.close()


def get_patient_list(limit=500, offset=0):
    conn = get_conn()
    df = pd.read_sql(
        f"SELECT * FROM patients ORDER BY patientunitstayid LIMIT {limit} OFFSET {offset}",
        conn
    )
    conn.close()
    return df


def get_patient_by_id(pid: int):
    conn = get_conn()
    df = pd.read_sql(
        "SELECT * FROM patients WHERE patientunitstayid = ?",
        conn, params=(pid,)
    )
    conn.close()
    return df


def get_vitals_by_patient(pid: int):
    conn = get_conn()
    df = pd.read_sql(
        "SELECT * FROM vitals WHERE patientunitstayid = ? ORDER BY observationoffset",
        conn, params=(pid,)
    )
    conn.close()
    return df


def get_labs_by_patient(pid: int):
    conn = get_conn()
    df = pd.read_sql(
        "SELECT * FROM labs WHERE patientunitstayid = ? ORDER BY labresultoffset",
        conn, params=(pid,)
    )
    conn.close()
    return df


def get_diagnosis_by_patient(pid: int):
    conn = get_conn()
    df = pd.read_sql(
        "SELECT * FROM diagnosis WHERE patientunitstayid = ?",
        conn, params=(pid,)
    )
    conn.close()
    return df


def get_treatments_by_patient(pid: int):
    conn = get_conn()
    df = pd.read_sql(
        "SELECT * FROM treatments WHERE patientunitstayid = ?",
        conn, params=(pid,)
    )
    conn.close()
    return df


def update_ml_scores(pid: int, risk_score: float, prediction: int):
    conn = get_conn()
    conn.execute(
        "UPDATE patients SET ml_risk_score = ?, ml_prediction = ? WHERE patientunitstayid = ?",
        (risk_score, prediction, pid)
    )
    conn.commit()
    conn.close()


def count_patients():
    conn = get_conn()
    c = conn.cursor()
    c.execute("SELECT COUNT(*) FROM patients")
    n = c.fetchone()[0]
    conn.close()
    return n


def get_stats():
    conn = get_conn()
    stats = {}
    stats["total_patients"] = pd.read_sql("SELECT COUNT(*) as n FROM patients", conn)["n"][0]
    stats["mortality_rate"] = pd.read_sql("SELECT AVG(hospital_mortality) as r FROM patients", conn)["r"][0]
    stats["avg_age"] = pd.read_sql("SELECT AVG(age) as a FROM patients", conn)["a"][0]
    stats["avg_apache"] = pd.read_sql("SELECT AVG(apachescore) as s FROM patients WHERE apachescore IS NOT NULL", conn)["s"][0]
    conn.close()
    return stats


if __name__ == "__main__":
    init_db()
    print("DB init OK")
