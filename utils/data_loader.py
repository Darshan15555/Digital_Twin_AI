"""
Chunked eICU data loading and preprocessing with parquet caching.
"""

from __future__ import annotations

import csv
import gzip
import logging
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from config.config import (
    CHUNK_SIZE,
    COMORBIDITY_MAP,
    EICU_RAW_PATH,
    FILE_NAMES,
    INTAKE_OUTPUT_MAX_ROWS,
    LAB_CHUNK_SIZE,
    LAB_RANGES,
    NURSE_TARGETS,
    PARQUET_DIR,
    VASOPRESSOR_KEYWORDS,
    VITAL_APERIODIC_MAX_ROWS,
    VITAL_PERIODIC_MAX_ROWS,
)

log = logging.getLogger(__name__)


def parquet_path(name: str) -> Path:
    return PARQUET_DIR / f"{name}.parquet"


def has_cache(name: str) -> bool:
    return parquet_path(name).exists()


def load_cached(name: str) -> pd.DataFrame:
    return pd.read_parquet(parquet_path(name))


def save_cache(name: str, df: pd.DataFrame) -> pd.DataFrame:
    df.to_parquet(parquet_path(name), index=False)
    log.info("Saved %s parquet with %s rows", name, f"{len(df):,}")
    return df


def source_path(key: str) -> Path:
    configured = EICU_RAW_PATH / FILE_NAMES[key]
    if configured.exists():
        return configured
    if configured.suffix == ".gz":
        csv_fallback = configured.with_suffix("")
        if csv_fallback.exists():
            return csv_fallback
    return configured


def _csv_columns(key: str) -> list[str]:
    filepath = source_path(key)
    if not filepath.exists():
        raise FileNotFoundError(f"Missing source file: {filepath}")
    compression = "gzip" if filepath.suffix == ".gz" else None
    return pd.read_csv(filepath, compression=compression, nrows=0).columns.tolist()


def validate_source_data() -> list[str]:
    missing = [filename for key, filename in FILE_NAMES.items() if not source_path(key).exists()]
    return missing


def _read_csv_in_chunks(
    key: str,
    usecols: list[str],
    transform: Callable[[pd.DataFrame], pd.DataFrame | None],
    nrows: int | None = None,
    engine: str = "c",
    chunk_size: int | None = None,
) -> pd.DataFrame:
    filepath = source_path(key)
    if not filepath.exists():
        raise FileNotFoundError(f"Missing source file: {filepath}")

    log.info("Loading %s", filepath.name)
    chunks: list[pd.DataFrame] = []
    rows_loaded = 0

    read_csv_kwargs = {
        "filepath_or_buffer": filepath,
        "compression": "gzip" if filepath.suffix == ".gz" else None,
        "chunksize": chunk_size or CHUNK_SIZE,
        "usecols": usecols,
        "engine": engine,
        "on_bad_lines": "skip",
    }
    if engine != "python":
        read_csv_kwargs["low_memory"] = False

    reader = pd.read_csv(**read_csv_kwargs)

    for chunk in reader:
        if nrows is not None and rows_loaded >= nrows:
            break

        if nrows is not None and rows_loaded + len(chunk) > nrows:
            chunk = chunk.head(nrows - rows_loaded)

        cleaned = transform(chunk)
        if cleaned is not None and not cleaned.empty:
            chunks.append(cleaned)
            rows_loaded += len(chunk)
        else:
            rows_loaded += len(chunk)

    if not chunks:
        return pd.DataFrame(columns=usecols)

    return pd.concat(chunks, ignore_index=True)


def _to_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _filter_range(series: pd.Series, lower: float, upper: float) -> pd.Series:
    numeric = _to_numeric(series)
    return numeric.where(numeric.between(lower, upper))


def _read_gzip_dict_chunks(
    key: str,
    usecols: list[str],
    transform: Callable[[pd.DataFrame], pd.DataFrame | None],
    chunk_size: int,
) -> pd.DataFrame:
    filepath = source_path(key)
    if not filepath.exists():
        raise FileNotFoundError(f"Missing source file: {filepath}")

    log.info("Streaming %s", filepath.name)
    chunks: list[pd.DataFrame] = []
    buffer: list[dict[str, str]] = []

    open_fn = gzip.open if filepath.suffix == ".gz" else open
    with open_fn(filepath, mode="rt", newline="", encoding="utf-8", errors="ignore") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            return pd.DataFrame(columns=usecols)

        available_cols = [column for column in usecols if column in reader.fieldnames]
        for row in reader:
            buffer.append({column: row.get(column) for column in available_cols})
            if len(buffer) >= chunk_size:
                chunk = pd.DataFrame.from_records(buffer, columns=available_cols)
                cleaned = transform(chunk)
                if cleaned is not None and not cleaned.empty:
                    chunks.append(cleaned)
                buffer.clear()

    if buffer:
        chunk = pd.DataFrame.from_records(buffer, columns=usecols)
        cleaned = transform(chunk)
        if cleaned is not None and not cleaned.empty:
            chunks.append(cleaned)

    if not chunks:
        return pd.DataFrame(columns=usecols)

    return pd.concat(chunks, ignore_index=True)


def load_patients(force: bool = False) -> pd.DataFrame:
    if has_cache("patients") and not force:
        return load_cached("patients")

    usecols = [
        "patientunitstayid",
        "patienthealthsystemstayid",
        "hospitalid",
        "age",
        "gender",
        "ethnicity",
        "admissionheight",
        "admissionweight",
        "unittype",
        "unitadmitsource",
        "unitdischargestatus",
        "hospitaldischargestatus",
        "apacheadmissiondx",
        "unitdischargeoffset",
        "hospitaladmitoffset",
    ]

    def transform(chunk: pd.DataFrame) -> pd.DataFrame:
        chunk = chunk.copy()
        chunk["patientunitstayid"] = _to_numeric(chunk["patientunitstayid"])
        chunk = chunk.dropna(subset=["patientunitstayid"])
        chunk["patientunitstayid"] = chunk["patientunitstayid"].astype(int)
        chunk["hospitalid"] = _to_numeric(chunk["hospitalid"])
        chunk["age"] = chunk["age"].replace("> 89", 90)
        chunk["age"] = _to_numeric(chunk["age"])
        chunk["gender"] = chunk["gender"].fillna("Unknown").astype(str).str.strip()
        chunk["gender_enc"] = np.where(chunk["gender"].str.lower() == "male", 1, 0)
        chunk["ethnicity"] = chunk["ethnicity"].fillna("Unknown")
        chunk["admissionheight"] = _filter_range(chunk["admissionheight"], 100, 250)
        chunk["admissionweight"] = _filter_range(chunk["admissionweight"], 20, 400)
        chunk["hospital_mortality"] = (
            chunk["hospitaldischargestatus"].fillna("").str.lower().eq("expired").astype(int)
        )
        chunk["icu_los_hours"] = _to_numeric(chunk["unitdischargeoffset"]) / 60.0
        chunk["hospitaladmitoffset_hours"] = _to_numeric(chunk["hospitaladmitoffset"]) / 60.0
        return chunk

    df = _read_csv_in_chunks("patient", usecols, transform)
    return save_cache("patients", df)


def load_hospitals(force: bool = False) -> pd.DataFrame:
    if has_cache("hospitals") and not force:
        return load_cached("hospitals")

    usecols = ["hospitalid", "region", "numbedscategory", "teachingstatus"]

    def transform(chunk: pd.DataFrame) -> pd.DataFrame:
        chunk = chunk.copy()
        chunk["hospitalid"] = _to_numeric(chunk["hospitalid"])
        chunk = chunk.dropna(subset=["hospitalid"]).drop_duplicates(subset=["hospitalid"])
        chunk["hospitalid"] = chunk["hospitalid"].astype(int)
        return chunk

    df = _read_csv_in_chunks("hospital", usecols, transform)
    return save_cache("hospitals", df)


def load_apache(force: bool = False) -> pd.DataFrame:
    if has_cache("apache") and not force:
        return load_cached("apache")

    desired_cols = [
        "patientunitstayid",
        "apachescore",
        "predictedhospitalmortality",
        "actualhospitalmortality",
        "preoperation",
        "preopmi",
        "preopcardiaccath",
    ]
    available_cols = set(_csv_columns("apache"))
    usecols = [col for col in desired_cols if col in available_cols]

    def transform(chunk: pd.DataFrame) -> pd.DataFrame:
        chunk = chunk.copy()
        chunk["patientunitstayid"] = _to_numeric(chunk["patientunitstayid"])
        chunk = chunk.dropna(subset=["patientunitstayid"])
        chunk["patientunitstayid"] = chunk["patientunitstayid"].astype(int)
        chunk["apachescore"] = _to_numeric(chunk["apachescore"])
        chunk["predictedhospitalmortality"] = _to_numeric(chunk["predictedhospitalmortality"])
        if "preoperation" in chunk.columns:
            chunk["preoperation_flag"] = chunk["preoperation"].fillna("").astype(str).str.lower().eq("yes").astype(int)
        else:
            preop_cols = [col for col in ["preopmi", "preopcardiaccath"] if col in chunk.columns]
            if preop_cols:
                chunk["preoperation_flag"] = (
                    chunk[preop_cols]
                    .apply(pd.to_numeric, errors="coerce")
                    .fillna(0)
                    .max(axis=1)
                    .gt(0)
                    .astype(int)
                )
            else:
                chunk["preoperation_flag"] = 0
        return chunk

    df = _read_csv_in_chunks("apache", usecols, transform)
    return save_cache("apache", df)


def load_vitals_periodic(force: bool = False) -> pd.DataFrame:
    if has_cache("vitals_periodic") and not force:
        return load_cached("vitals_periodic")

    usecols = [
        "patientunitstayid",
        "observationoffset",
        "heartrate",
        "respiration",
        "sao2",
        "systemicsystolic",
        "systemicdiastolic",
        "temperature",
        "cvp",
    ]

    def transform(chunk: pd.DataFrame) -> pd.DataFrame:
        chunk = chunk.copy()
        chunk["patientunitstayid"] = _to_numeric(chunk["patientunitstayid"])
        chunk = chunk.dropna(subset=["patientunitstayid"])
        chunk["patientunitstayid"] = chunk["patientunitstayid"].astype(int)
        chunk["observationoffset"] = _to_numeric(chunk["observationoffset"])
        chunk["offset_hours"] = chunk["observationoffset"] / 60.0
        chunk["heartrate"] = _filter_range(chunk["heartrate"], 0, 300)
        chunk["respiration"] = _filter_range(chunk["respiration"], 0, 80)
        chunk["sao2"] = _filter_range(chunk["sao2"], 50, 100)
        chunk["systemicsystolic"] = _filter_range(chunk["systemicsystolic"], 40, 300)
        chunk["systemicdiastolic"] = _filter_range(chunk["systemicdiastolic"], 20, 200)
        temp = _to_numeric(chunk["temperature"])
        temp = np.where(temp > 45, (temp - 32) * 5.0 / 9.0, temp)
        chunk["temperature"] = pd.Series(temp, index=chunk.index).where(
            pd.Series(temp, index=chunk.index).between(25, 45)
        )
        chunk["cvp"] = _filter_range(chunk["cvp"], -5, 40)
        return chunk

    df = _read_csv_in_chunks("vital_periodic", usecols, transform, nrows=VITAL_PERIODIC_MAX_ROWS)
    return save_cache("vitals_periodic", df)


def load_vitals_aperiodic(force: bool = False) -> pd.DataFrame:
    if has_cache("vitals_aperiodic") and not force:
        return load_cached("vitals_aperiodic")

    usecols = [
        "patientunitstayid",
        "observationoffset",
        "noninvasivesystolic",
        "noninvasivemean",
        "noninvasivediastolic",
        "paop",
    ]

    def transform(chunk: pd.DataFrame) -> pd.DataFrame:
        chunk = chunk.copy()
        chunk["patientunitstayid"] = _to_numeric(chunk["patientunitstayid"])
        chunk = chunk.dropna(subset=["patientunitstayid"])
        chunk["patientunitstayid"] = chunk["patientunitstayid"].astype(int)
        chunk["observationoffset"] = _to_numeric(chunk["observationoffset"])
        chunk["offset_hours"] = chunk["observationoffset"] / 60.0
        chunk["noninvasivesystolic"] = _filter_range(chunk["noninvasivesystolic"], 40, 300)
        chunk["noninvasivemean"] = _filter_range(chunk["noninvasivemean"], 20, 200)
        chunk["noninvasivediastolic"] = _filter_range(chunk["noninvasivediastolic"], 20, 200)
        chunk["paop"] = _filter_range(chunk["paop"], 0, 40)
        return chunk

    df = _read_csv_in_chunks("vital_aperiodic", usecols, transform, nrows=VITAL_APERIODIC_MAX_ROWS)
    return save_cache("vitals_aperiodic", df)


def load_labs(force: bool = False) -> pd.DataFrame:
    if has_cache("labs") and not force:
        return load_cached("labs")

    usecols = ["patientunitstayid", "labresultoffset", "labname", "labresult"]

    def transform(chunk: pd.DataFrame) -> pd.DataFrame:
        chunk = chunk.copy()
        chunk["patientunitstayid"] = _to_numeric(chunk["patientunitstayid"])
        chunk = chunk.dropna(subset=["patientunitstayid"])
        chunk["patientunitstayid"] = chunk["patientunitstayid"].astype(int)
        chunk["labresultoffset"] = _to_numeric(chunk["labresultoffset"])
        chunk["offset_hours"] = chunk["labresultoffset"] / 60.0
        chunk["labname"] = chunk["labname"].fillna("").astype(str).str.strip().str.lower()
        chunk["labresult"] = _to_numeric(chunk["labresult"])
        chunk = chunk.dropna(subset=["labresult"])
        for name, (lower, upper) in LAB_RANGES.items():
            mask = chunk["labname"] == name
            chunk.loc[mask, "labresult"] = chunk.loc[mask, "labresult"].where(
                chunk.loc[mask, "labresult"].between(lower, upper)
            )
        chunk = chunk.dropna(subset=["labresult"])
        return chunk

    df = _read_gzip_dict_chunks("lab", usecols, transform, chunk_size=LAB_CHUNK_SIZE)
    return save_cache("labs", df)


def load_diagnosis(force: bool = False) -> pd.DataFrame:
    if has_cache("diagnosis") and not force:
        return load_cached("diagnosis")

    usecols = ["patientunitstayid", "diagnosisoffset", "diagnosisstring", "icd9code", "diagnosispriority"]

    def transform(chunk: pd.DataFrame) -> pd.DataFrame:
        chunk = chunk.copy()
        chunk["patientunitstayid"] = _to_numeric(chunk["patientunitstayid"])
        chunk = chunk.dropna(subset=["patientunitstayid"])
        chunk["patientunitstayid"] = chunk["patientunitstayid"].astype(int)
        chunk["diagnosisoffset"] = _to_numeric(chunk["diagnosisoffset"])
        chunk["offset_hours"] = chunk["diagnosisoffset"] / 60.0
        return chunk

    df = _read_csv_in_chunks("diagnosis", usecols, transform)
    return save_cache("diagnosis", df)


def load_treatments(force: bool = False) -> pd.DataFrame:
    if has_cache("treatments") and not force:
        return load_cached("treatments")

    usecols = ["patientunitstayid", "treatmentoffset", "treatmentstring"]

    def transform(chunk: pd.DataFrame) -> pd.DataFrame:
        chunk = chunk.copy()
        chunk["patientunitstayid"] = _to_numeric(chunk["patientunitstayid"])
        chunk = chunk.dropna(subset=["patientunitstayid"])
        chunk["patientunitstayid"] = chunk["patientunitstayid"].astype(int)
        chunk["treatmentoffset"] = _to_numeric(chunk["treatmentoffset"])
        chunk["offset_hours"] = chunk["treatmentoffset"] / 60.0
        return chunk

    df = _read_csv_in_chunks("treatment", usecols, transform)
    return save_cache("treatments", df)


def load_medications(force: bool = False) -> pd.DataFrame:
    if has_cache("medications") and not force:
        return load_cached("medications")

    usecols = [
        "patientunitstayid",
        "drugstartoffset",
        "drugstopoffset",
        "drugname",
        "dosage",
        "routeadmin",
        "frequency",
    ]

    def transform(chunk: pd.DataFrame) -> pd.DataFrame:
        chunk = chunk.copy()
        chunk["patientunitstayid"] = _to_numeric(chunk["patientunitstayid"])
        chunk = chunk.dropna(subset=["patientunitstayid"])
        chunk["patientunitstayid"] = chunk["patientunitstayid"].astype(int)
        chunk["drugstartoffset"] = _to_numeric(chunk["drugstartoffset"])
        chunk["drugstopoffset"] = _to_numeric(chunk["drugstopoffset"])
        chunk["offset_hours"] = chunk["drugstartoffset"] / 60.0
        chunk["drugname"] = chunk["drugname"].fillna("").astype(str).str.strip()
        return chunk

    df = _read_csv_in_chunks("medication", usecols, transform)
    return save_cache("medications", df)


def load_infusions(force: bool = False) -> pd.DataFrame:
    if has_cache("infusions") and not force:
        return load_cached("infusions")

    usecols = [
        "patientunitstayid",
        "infusionoffset",
        "drugname",
        "drugrate",
    ]

    def transform(chunk: pd.DataFrame) -> pd.DataFrame:
        chunk = chunk.copy()
        chunk["patientunitstayid"] = _to_numeric(chunk["patientunitstayid"])
        chunk = chunk.dropna(subset=["patientunitstayid"])
        chunk["patientunitstayid"] = chunk["patientunitstayid"].astype(int)
        chunk["infusionoffset"] = _to_numeric(chunk["infusionoffset"])
        chunk["offset_hours"] = chunk["infusionoffset"] / 60.0
        chunk["drugname"] = chunk["drugname"].fillna("").astype(str).str.strip().str.lower()
        chunk["drugrate"] = _to_numeric(chunk["drugrate"])
        chunk = chunk[(chunk["drugrate"].isna()) | (chunk["drugrate"] >= 0)]
        chunk["on_vasopressor"] = chunk["drugname"].apply(
            lambda value: int(any(keyword in value for keyword in VASOPRESSOR_KEYWORDS))
        )
        return chunk

    df = _read_csv_in_chunks("infusion_drug", usecols, transform)
    return save_cache("infusions", df)


def load_nurse_charting(force: bool = False) -> pd.DataFrame:
    if has_cache("nurse_charting") and not force:
        return load_cached("nurse_charting")

    usecols = [
        "patientunitstayid",
        "nursingchartoffset",
        "nursingchartcelltypevalname",
        "nursingchartvalue",
    ]

    def transform(chunk: pd.DataFrame) -> pd.DataFrame:
        chunk = chunk.copy()
        chunk["nursingchartcelltypevalname"] = (
            chunk["nursingchartcelltypevalname"].fillna("").astype(str).str.strip().str.lower()
        )
        chunk = chunk[chunk["nursingchartcelltypevalname"].isin(NURSE_TARGETS.keys())]
        if chunk.empty:
            return chunk
        chunk["patientunitstayid"] = _to_numeric(chunk["patientunitstayid"])
        chunk = chunk.dropna(subset=["patientunitstayid"])
        chunk["patientunitstayid"] = chunk["patientunitstayid"].astype(int)
        chunk["nursingchartoffset"] = _to_numeric(chunk["nursingchartoffset"])
        chunk["offset_hours"] = chunk["nursingchartoffset"] / 60.0
        chunk["metric_name"] = chunk["nursingchartcelltypevalname"].map(NURSE_TARGETS)
        chunk["metric_value"] = _to_numeric(chunk["nursingchartvalue"])
        gcs_mask = chunk["metric_name"] == "gcs"
        chunk.loc[gcs_mask, "metric_value"] = chunk.loc[gcs_mask, "metric_value"].where(
            chunk.loc[gcs_mask, "metric_value"].between(3, 15)
        )
        pain_mask = chunk["metric_name"] == "pain_score"
        chunk.loc[pain_mask, "metric_value"] = chunk.loc[pain_mask, "metric_value"].where(
            chunk.loc[pain_mask, "metric_value"].between(0, 10)
        )
        result = chunk.dropna(subset=["metric_value"])[
            [
                "patientunitstayid",
                "nursingchartoffset",
                "offset_hours",
                "metric_name",
                "metric_value",
            ]
        ]
        return result

    df = _read_csv_in_chunks("nurse_charting", usecols, transform)
    if df.empty:
        df = pd.DataFrame(
            columns=[
                "patientunitstayid",
                "nursingchartoffset",
                "offset_hours",
                "metric_name",
                "metric_value",
            ]
        )
    return save_cache("nurse_charting", df)


def load_intake_output(force: bool = False) -> pd.DataFrame:
    if has_cache("intake_output_raw") and not force:
        return load_cached("intake_output_raw")

    usecols = ["patientunitstayid", "intakeoutputoffset", "celllabel", "cellvaluenumeric"]
    allowed = {"volume in", "volume out", "urine output"}

    def transform(chunk: pd.DataFrame) -> pd.DataFrame:
        chunk = chunk.copy()
        chunk["celllabel"] = chunk["celllabel"].fillna("").astype(str).str.strip().str.lower()
        chunk = chunk[chunk["celllabel"].isin(allowed)]
        if chunk.empty:
            return chunk
        chunk["patientunitstayid"] = _to_numeric(chunk["patientunitstayid"])
        chunk = chunk.dropna(subset=["patientunitstayid"])
        chunk["patientunitstayid"] = chunk["patientunitstayid"].astype(int)
        chunk["intakeoutputoffset"] = _to_numeric(chunk["intakeoutputoffset"])
        chunk["offset_hours"] = chunk["intakeoutputoffset"] / 60.0
        chunk["cellvaluenumeric"] = _to_numeric(chunk["cellvaluenumeric"])
        chunk = chunk[chunk["cellvaluenumeric"] > 0]
        return chunk

    df = _read_csv_in_chunks("intake_output", usecols, transform, nrows=INTAKE_OUTPUT_MAX_ROWS)
    return save_cache("intake_output_raw", df)


def load_respiratory_care(force: bool = False) -> pd.DataFrame:
    if has_cache("respiratory_care") and not force:
        return load_cached("respiratory_care")

    available_cols = set(_csv_columns("respiratory_care"))
    column_aliases = {
        "patientunitstayid": ["patientunitstayid"],
        "respcarestatusoffset": ["respcarestatusoffset"],
        "priorventday1": ["priorventday1", "priorventstartoffset"],
        "priorventday2": ["priorventday2", "priorventendoffset"],
        "airwaytype": ["airwaytype"],
        "airwaysize": ["airwaysize"],
        "apneainterval": ["apneainterval", "setapneainterval"],
        "peep": ["peep", "peeplimit"],
        "fio2": ["fio2", "setapneafio2"],
    }
    selected_columns: dict[str, str] = {}
    for target, candidates in column_aliases.items():
        source = next((name for name in candidates if name in available_cols), None)
        if source is not None:
            selected_columns[target] = source

    usecols = list(dict.fromkeys(selected_columns.values()))

    def transform(chunk: pd.DataFrame) -> pd.DataFrame:
        chunk = chunk.copy()
        for target, source in selected_columns.items():
            if target != source and source in chunk.columns:
                chunk[target] = chunk[source]
        for target in column_aliases:
            if target not in chunk.columns:
                chunk[target] = np.nan

        chunk["patientunitstayid"] = _to_numeric(chunk["patientunitstayid"])
        chunk = chunk.dropna(subset=["patientunitstayid"])
        chunk["patientunitstayid"] = chunk["patientunitstayid"].astype(int)
        chunk["respcarestatusoffset"] = _to_numeric(chunk["respcarestatusoffset"])
        chunk["offset_hours"] = chunk["respcarestatusoffset"] / 60.0
        chunk["priorventday1"] = _to_numeric(chunk["priorventday1"]).fillna(0).gt(0).astype(int)
        chunk["priorventday2"] = _to_numeric(chunk["priorventday2"]).fillna(0).gt(0).astype(int)
        chunk["airwaytype"] = chunk["airwaytype"].replace("", np.nan)
        chunk["airwaysize"] = _to_numeric(chunk["airwaysize"])
        chunk["apneainterval"] = _to_numeric(chunk["apneainterval"])
        chunk["peep"] = _filter_range(chunk["peep"], 0, 30)
        fio2 = _to_numeric(chunk["fio2"])
        fio2 = np.where(fio2 > 1, fio2 / 100.0, fio2)
        chunk["fio2"] = pd.Series(fio2, index=chunk.index).where(pd.Series(fio2, index=chunk.index).between(0.21, 1.0))
        chunk["on_ventilator"] = chunk["airwaytype"].notna().astype(int)
        return chunk[
            [
                "patientunitstayid",
                "respcarestatusoffset",
                "offset_hours",
                "priorventday1",
                "priorventday2",
                "airwaytype",
                "airwaysize",
                "apneainterval",
                "peep",
                "fio2",
                "on_ventilator",
            ]
        ]

    df = _read_csv_in_chunks("respiratory_care", usecols, transform)
    return save_cache("respiratory_care", df)


def load_past_history(force: bool = False) -> pd.DataFrame:
    if has_cache("past_history_flags") and not force:
        return load_cached("past_history_flags")

    usecols = ["patientunitstayid", "pasthistoryvalue"]

    def transform(chunk: pd.DataFrame) -> pd.DataFrame:
        chunk = chunk.copy()
        chunk["patientunitstayid"] = _to_numeric(chunk["patientunitstayid"])
        chunk = chunk.dropna(subset=["patientunitstayid"])
        chunk["patientunitstayid"] = chunk["patientunitstayid"].astype(int)
        chunk["pasthistoryvalue"] = chunk["pasthistoryvalue"].fillna("").astype(str).str.strip()
        chunk["history_lower"] = chunk["pasthistoryvalue"].str.lower()
        rows = []
        for key, column in COMORBIDITY_MAP.items():
            subset = chunk[chunk["history_lower"].str.contains(key, na=False, regex=False)][["patientunitstayid"]].copy()
            if subset.empty:
                continue
            subset[column] = 1
            rows.append(subset)
        if not rows:
            return pd.DataFrame(columns=["patientunitstayid", *COMORBIDITY_MAP.values()])
        merged = pd.concat(rows, ignore_index=True)
        return merged

    df = _read_csv_in_chunks("past_history", usecols, transform)
    if df.empty:
        df = pd.DataFrame(columns=["patientunitstayid", *COMORBIDITY_MAP.values()])
    grouped = df.groupby("patientunitstayid", as_index=False).max()
    for column in COMORBIDITY_MAP.values():
        if column not in grouped.columns:
            grouped[column] = 0
    return save_cache("past_history_flags", grouped[["patientunitstayid", *COMORBIDITY_MAP.values()]])


def build_intake_output_summary(force: bool = False) -> pd.DataFrame:
    if has_cache("intake_output_summary") and not force:
        return load_cached("intake_output_summary")

    io_df = load_intake_output(force=force)
    patients = load_patients(force=force)[["patientunitstayid", "icu_los_hours"]]

    if io_df.empty:
        empty = patients.copy()
        empty["total_intake"] = 0.0
        empty["total_output"] = 0.0
        empty["total_urine_output"] = 0.0
        empty["fluid_balance"] = 0.0
        empty["urine_output_per_hour"] = 0.0
        return save_cache("intake_output_summary", empty)

    io_df = io_df.copy()
    io_df["intake_value"] = np.where(io_df["celllabel"] == "volume in", io_df["cellvaluenumeric"], 0.0)
    io_df["output_value"] = np.where(io_df["celllabel"] == "volume out", io_df["cellvaluenumeric"], 0.0)
    io_df["urine_value"] = np.where(io_df["celllabel"] == "urine output", io_df["cellvaluenumeric"], 0.0)

    summary = (
        io_df.groupby("patientunitstayid", as_index=False)
        .agg(
            total_intake=("intake_value", "sum"),
            total_output=("output_value", "sum"),
            total_urine_output=("urine_value", "sum"),
        )
    )
    summary["fluid_balance"] = summary["total_intake"] - summary["total_output"]
    summary = summary.merge(patients, on="patientunitstayid", how="left")
    summary["urine_output_per_hour"] = summary["total_urine_output"] / summary["icu_los_hours"].replace({0: np.nan})
    summary["urine_output_per_hour"] = summary["urine_output_per_hour"].fillna(0)
    return save_cache("intake_output_summary", summary)


def load_all_processed_tables(force: bool = False) -> dict[str, pd.DataFrame]:
    return {
        "patients": load_patients(force=force),
        "hospitals": load_hospitals(force=force),
        "apache": load_apache(force=force),
        "vitals_periodic": load_vitals_periodic(force=force),
        "vitals_aperiodic": load_vitals_aperiodic(force=force),
        "labs": load_labs(force=force),
        "diagnosis": load_diagnosis(force=force),
        "treatments": load_treatments(force=force),
        "medications": load_medications(force=force),
        "infusions": load_infusions(force=force),
        "nurse_charting": load_nurse_charting(force=force),
        "intake_output_summary": build_intake_output_summary(force=force),
        "respiratory_care": load_respiratory_care(force=force),
        "comorbidities": load_past_history(force=force),
    }
