"""
data_loader.py
Efficiently loads eICU .csv.gz files using chunking.
Converts to Parquet for fast subsequent reads.
"""

import pandas as pd
import numpy as np
import sqlite3
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config.config import (
    EICU_RAW_PATH, PARQUET_DIR, SQLITE_DB_PATH,
    CHUNK_SIZE, VITALS_SAMPLE_ROWS, MAX_PATIENTS_CACHE
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)


def parquet_path(name: str) -> Path:
    return PARQUET_DIR / f"{name}.parquet"


def is_cached(name: str) -> bool:
    return parquet_path(name).exists()


def load_gz_chunked(filename: str, usecols=None, dtype=None, nrows=None) -> pd.DataFrame:
    """Load a .csv.gz file from eICU directory using chunking."""
    filepath = EICU_RAW_PATH / filename
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    log.info(f"Loading {filename} ...")
    chunks = []
    rows_loaded = 0

    reader = pd.read_csv(
        filepath,
        compression="gzip",
        chunksize=CHUNK_SIZE,
        usecols=usecols,
        dtype=dtype,
        low_memory=False
    )

    for chunk in reader:
        chunks.append(chunk)
        rows_loaded += len(chunk)
        if nrows and rows_loaded >= nrows:
            break

    df = pd.concat(chunks, ignore_index=True)
    if nrows:
        df = df.head(nrows)
    log.info(f"  → Loaded {len(df):,} rows from {filename}")
    return df


def load_patients() -> pd.DataFrame:
    name = "patient"
    if is_cached(name):
        log.info(f"Loading cached {name}.parquet")
        return pd.read_parquet(parquet_path(name))

    df = load_gz_chunked(
        "patient.csv.gz",
        usecols=[
            "patientunitstayid", "patienthealthsystemstayid",
            "age", "gender", "ethnicity", "admissionheight", "admissionweight",
            "unittype", "unitadmitsource", "unitdischargestatus",
            "hospitaldischargestatus", "apacheadmissiondx",
            "unitdischargeoffset", "hospitaladmitoffset"
        ]
    )

    # Clean age (eICU stores "> 89" as string)
    df["age"] = df["age"].replace("> 89", "90")
    df["age"] = pd.to_numeric(df["age"], errors="coerce")
    df["gender"] = df["gender"].fillna("Unknown")
    df["ethnicity"] = df["ethnicity"].fillna("Unknown")

    # Binary mortality label
    df["hospital_mortality"] = (
        df["hospitaldischargestatus"].str.lower() == "expired"
    ).astype(int)

    df.to_parquet(parquet_path(name), index=False)
    log.info(f"Saved {name}.parquet ({len(df):,} rows)")
    return df


def load_vitals(patient_ids=None) -> pd.DataFrame:
    name = "vitals"
    if is_cached(name) and patient_ids is None:
        log.info(f"Loading cached {name}.parquet")
        return pd.read_parquet(parquet_path(name))

    df = load_gz_chunked(
        "vitalPeriodic.csv.gz",
        usecols=[
            "patientunitstayid", "observationoffset",
            "heartrate", "respiration", "sao2",
            "systemicsystolic", "systemicdiastolic",
            "temperature", "cvp"
        ],
        nrows=VITALS_SAMPLE_ROWS
    )

    df = df.dropna(subset=["patientunitstayid"])
    df["patientunitstayid"] = df["patientunitstayid"].astype(int)

    if patient_ids is not None:
        df = df[df["patientunitstayid"].isin(patient_ids)]

    # Basic range filters (remove physiologically impossible values)
    df = df[
        (df["heartrate"].between(0, 300) | df["heartrate"].isna()) &
        (df["sao2"].between(50, 100) | df["sao2"].isna()) &
        (df["temperature"].between(25, 45) | df["temperature"].isna())
    ]

    if patient_ids is None:
        df.to_parquet(parquet_path(name), index=False)
        log.info(f"Saved {name}.parquet ({len(df):,} rows)")
    return df


def load_labs(patient_ids=None) -> pd.DataFrame:
    name = "labs"
    if is_cached(name) and patient_ids is None:
        log.info(f"Loading cached {name}.parquet")
        return pd.read_parquet(parquet_path(name))

    df = load_gz_chunked(
        "lab.csv.gz",
        usecols=[
            "patientunitstayid", "labresultoffset",
            "labname", "labresult", "labresulttext"
        ]
    )

    df["patientunitstayid"] = pd.to_numeric(df["patientunitstayid"], errors="coerce")
    df = df.dropna(subset=["patientunitstayid", "labresult"])
    df["patientunitstayid"] = df["patientunitstayid"].astype(int)
    df["labresult"] = pd.to_numeric(df["labresult"], errors="coerce")

    if patient_ids is not None:
        df = df[df["patientunitstayid"].isin(patient_ids)]

    if patient_ids is None:
        df.to_parquet(parquet_path(name), index=False)
        log.info(f"Saved {name}.parquet ({len(df):,} rows)")
    return df


def load_diagnosis() -> pd.DataFrame:
    name = "diagnosis"
    if is_cached(name):
        log.info(f"Loading cached {name}.parquet")
        return pd.read_parquet(parquet_path(name))

    df = load_gz_chunked(
        "diagnosis.csv.gz",
        usecols=[
            "patientunitstayid", "diagnosisoffset",
            "diagnosisstring", "icd9code", "diagnosispriority"
        ]
    )
    df["patientunitstayid"] = pd.to_numeric(df["patientunitstayid"], errors="coerce")
    df = df.dropna(subset=["patientunitstayid"])
    df["patientunitstayid"] = df["patientunitstayid"].astype(int)

    df.to_parquet(parquet_path(name), index=False)
    log.info(f"Saved {name}.parquet ({len(df):,} rows)")
    return df


def load_apache() -> pd.DataFrame:
    name = "apache"
    if is_cached(name):
        log.info(f"Loading cached {name}.parquet")
        return pd.read_parquet(parquet_path(name))

    df = load_gz_chunked(
        "apachePatientResult.csv.gz",
        usecols=[
            "patientunitstayid", "apachescore",
            "predictedhospitalmortality", "actualhospitalmortality",
            "preoperation"
        ]
    )
    df["patientunitstayid"] = pd.to_numeric(df["patientunitstayid"], errors="coerce")
    df = df.dropna(subset=["patientunitstayid"])
    df["patientunitstayid"] = df["patientunitstayid"].astype(int)

    df.to_parquet(parquet_path(name), index=False)
    log.info(f"Saved {name}.parquet ({len(df):,} rows)")
    return df


def load_treatment() -> pd.DataFrame:
    name = "treatment"
    if is_cached(name):
        log.info(f"Loading cached {name}.parquet")
        return pd.read_parquet(parquet_path(name))

    df = load_gz_chunked(
        "treatment.csv.gz",
        usecols=["patientunitstayid", "treatmentoffset", "treatmentstring"]
    )
    df["patientunitstayid"] = pd.to_numeric(df["patientunitstayid"], errors="coerce")
    df = df.dropna(subset=["patientunitstayid"])
    df["patientunitstayid"] = df["patientunitstayid"].astype(int)

    df.to_parquet(parquet_path(name), index=False)
    log.info(f"Saved {name}.parquet ({len(df):,} rows)")
    return df


def build_ml_dataset() -> pd.DataFrame:
    """Build a flat dataset for ML training by joining patients with Apache scores and vitals aggregates."""
    name = "ml_dataset"
    if is_cached(name):
        log.info(f"Loading cached {name}.parquet")
        return pd.read_parquet(parquet_path(name))

    log.info("Building ML dataset...")
    patients = load_patients()
    apache = load_apache()

    # Aggregate vitals per patient
    vitals = load_vitals()
    vitals_agg = vitals.groupby("patientunitstayid").agg(
        hr_mean=("heartrate", "mean"),
        hr_std=("heartrate", "std"),
        sao2_mean=("sao2", "mean"),
        sao2_min=("sao2", "min"),
        resp_mean=("respiration", "mean"),
        sbp_mean=("systemicsystolic", "mean"),
        dbp_mean=("systemicdiastolic", "mean"),
        temp_mean=("temperature", "mean"),
    ).reset_index()

    # Aggregate key labs per patient
    labs = load_labs()
    key_labs = ["creatinine", "glucose", "potassium", "sodium", "lactate", "hemoglobin", "wbc"]
    lab_pivot = (
        labs[labs["labname"].str.lower().isin(key_labs)]
        .groupby(["patientunitstayid", "labname"])["labresult"]
        .mean()
        .unstack()
        .reset_index()
    )
    lab_pivot.columns = ["patientunitstayid"] + [f"lab_{c}" for c in lab_pivot.columns[1:]]

    # Merge everything
    df = patients.merge(apache[["patientunitstayid", "apachescore", "predictedhospitalmortality"]], on="patientunitstayid", how="left")
    df = df.merge(vitals_agg, on="patientunitstayid", how="left")
    df = df.merge(lab_pivot, on="patientunitstayid", how="left")

    df.to_parquet(parquet_path(name), index=False)
    log.info(f"ML dataset ready: {len(df):,} rows, {df.shape[1]} columns")
    return df


if __name__ == "__main__":
    log.info("=== eICU Data Loading Pipeline ===")
    patients = load_patients()
    log.info(f"Patients: {len(patients):,}")
    vitals = load_vitals()
    log.info(f"Vitals rows: {len(vitals):,}")
    labs = load_labs()
    log.info(f"Labs rows: {len(labs):,}")
    diagnosis = load_diagnosis()
    log.info(f"Diagnosis rows: {len(diagnosis):,}")
    apache = load_apache()
    log.info(f"Apache rows: {len(apache):,}")
    treatment = load_treatment()
    log.info(f"Treatment rows: {len(treatment):,}")
    ml_df = build_ml_dataset()
    log.info(f"ML dataset: {ml_df.shape}")
    log.info("=== Data loading complete ===")
