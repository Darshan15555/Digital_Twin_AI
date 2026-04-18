from __future__ import annotations

import numpy as np
import pandas as pd


REQUIRED_OUTPUT_COLUMNS = [
    "patient_id",
    "hr",
    "spo2",
    "bp_sys",
    "bp_dia",
    "bp_mean",
]

ALL_OUTPUT_COLUMNS = [
    "patient_id",
    "offset_min",
    "hr",
    "spo2",
    "bp_sys",
    "bp_dia",
    "bp_mean",
]


def clean_vital_periodic_chunk(chunk: pd.DataFrame, *, first_24h_only: bool = True) -> pd.DataFrame:
    if chunk.empty:
        return pd.DataFrame(columns=ALL_OUTPUT_COLUMNS)

    working = chunk.rename(
        columns={
            "patientunitstayid": "patient_id",
            "observationoffset": "offset_min",
            "heartrate": "hr",
            "sao2": "spo2",
            "systemicsystolic": "bp_sys",
            "systemicdiastolic": "bp_dia",
            "systemicmean": "bp_mean",
        }
    ).copy()

    for col in ["patient_id", "offset_min", "hr", "spo2", "bp_sys", "bp_dia", "bp_mean"]:
        if col not in working.columns:
            working[col] = np.nan

    numeric_cols = ["patient_id", "offset_min", "hr", "spo2", "bp_sys", "bp_dia", "bp_mean"]
    for col in numeric_cols:
        if col in working.columns:
            working[col] = pd.to_numeric(working[col], errors="coerce")

    working = working.dropna(subset=["patient_id", "offset_min"])
    if first_24h_only:
        working = working[working["offset_min"].between(0, 1439, inclusive="both")]
    if working.empty:
        return pd.DataFrame(columns=ALL_OUTPUT_COLUMNS)

    working["hr"] = working["hr"].where(working["hr"].between(30, 220, inclusive="both"))
    working["spo2"] = working["spo2"].where(working["spo2"].between(50, 100, inclusive="both"))
    working["bp_sys"] = working["bp_sys"].where(working["bp_sys"].between(40, 300, inclusive="both"))
    working["bp_dia"] = working["bp_dia"].where(working["bp_dia"].between(20, 220, inclusive="both"))
    working["bp_mean"] = working["bp_mean"].where(working["bp_mean"].between(20, 250, inclusive="both"))

    invalid_bp_pair = (
        working["bp_sys"].notna()
        & working["bp_dia"].notna()
        & (working["bp_sys"] <= working["bp_dia"])
    )
    working.loc[invalid_bp_pair, ["bp_sys", "bp_dia", "bp_mean"]] = np.nan

    computed_mean = (working["bp_sys"] + 2.0 * working["bp_dia"]) / 3.0
    working["bp_mean"] = working["bp_mean"].fillna(computed_mean)
    working["bp_mean"] = working["bp_mean"].where(working["bp_mean"].between(20, 250, inclusive="both"))

    working = working.dropna(subset=["hr", "spo2", "bp_sys", "bp_dia", "bp_mean"], how="all")
    if working.empty:
        return pd.DataFrame(columns=ALL_OUTPUT_COLUMNS)

    working["patient_id"] = working["patient_id"].astype("int64")
    working["offset_min"] = working["offset_min"].astype("int16")
    for col in ["hr", "spo2", "bp_sys", "bp_dia", "bp_mean"]:
        working[col] = working[col].astype("float32")

    return working[ALL_OUTPUT_COLUMNS]


def clean_vital_aperiodic_chunk(chunk: pd.DataFrame, *, first_24h_only: bool = True) -> pd.DataFrame:
    if chunk.empty:
        return pd.DataFrame(columns=ALL_OUTPUT_COLUMNS)

    working = chunk.rename(
        columns={
            "patientunitstayid": "patient_id",
            "observationoffset": "offset_min",
            "noninvasivesystolic": "bp_sys",
            "noninvasivediastolic": "bp_dia",
            "noninvasivemean": "bp_mean",
        }
    ).copy()
    for col in ["patient_id", "offset_min", "bp_sys", "bp_dia", "bp_mean"]:
        if col not in working.columns:
            working[col] = np.nan
    working["hr"] = np.nan
    working["spo2"] = np.nan

    for col in ["patient_id", "offset_min", "bp_sys", "bp_dia", "bp_mean"]:
        working[col] = pd.to_numeric(working[col], errors="coerce")

    working = working.dropna(subset=["patient_id", "offset_min"])
    if first_24h_only:
        working = working[working["offset_min"].between(0, 1439, inclusive="both")]
    if working.empty:
        return pd.DataFrame(columns=ALL_OUTPUT_COLUMNS)

    working["bp_sys"] = working["bp_sys"].where(working["bp_sys"].between(40, 300, inclusive="both"))
    working["bp_dia"] = working["bp_dia"].where(working["bp_dia"].between(20, 220, inclusive="both"))
    working["bp_mean"] = working["bp_mean"].where(working["bp_mean"].between(20, 250, inclusive="both"))

    invalid_bp_pair = (
        working["bp_sys"].notna()
        & working["bp_dia"].notna()
        & (working["bp_sys"] <= working["bp_dia"])
    )
    working.loc[invalid_bp_pair, ["bp_sys", "bp_dia", "bp_mean"]] = np.nan
    computed_mean = (working["bp_sys"] + 2.0 * working["bp_dia"]) / 3.0
    working["bp_mean"] = working["bp_mean"].fillna(computed_mean)

    working = working.dropna(subset=["bp_sys", "bp_dia", "bp_mean"], how="all")
    if working.empty:
        return pd.DataFrame(columns=ALL_OUTPUT_COLUMNS)

    working["patient_id"] = working["patient_id"].astype("int64")
    working["offset_min"] = working["offset_min"].astype("int16")
    for col in ["hr", "spo2", "bp_sys", "bp_dia", "bp_mean"]:
        working[col] = working[col].astype("float32")

    return working[ALL_OUTPUT_COLUMNS]


def attach_target_label(frame: pd.DataFrame, label_map: pd.Series | None) -> pd.DataFrame:
    if frame.empty or label_map is None or label_map.empty:
        return frame
    with_label = frame.copy()
    with_label["target_label"] = with_label["patient_id"].map(label_map).astype("float32")
    return with_label


def handle_missing(frame: pd.DataFrame, strategy: str = "drop") -> pd.DataFrame:
    if frame.empty:
        return frame.copy()

    valid = frame.copy()
    target_cols = ["hr", "spo2", "bp_sys", "bp_dia", "bp_mean"]
    strategy = strategy.lower().strip()

    if strategy == "drop":
        valid = valid.dropna(subset=target_cols, how="any")
        return valid.reset_index(drop=True)

    if strategy in {"median", "mean"}:
        for col in target_cols:
            value = valid[col].median() if strategy == "median" else valid[col].mean()
            valid[col] = valid[col].fillna(value)
        return valid.reset_index(drop=True)

    raise ValueError(f"Unsupported missing-value strategy: {strategy}")


def aggregate_mean_per_patient(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()

    numeric_aggs = {
        "hr": "mean",
        "spo2": "mean",
        "bp_sys": "mean",
        "bp_dia": "mean",
        "bp_mean": "mean",
    }

    cols = ["patient_id", *numeric_aggs]
    if "target_label" in frame.columns:
        cols.append("target_label")
        numeric_aggs["target_label"] = "max"

    grouped = frame[cols].groupby("patient_id", as_index=False).agg(numeric_aggs)
    for col in ["hr", "spo2", "bp_sys", "bp_dia", "bp_mean"]:
        grouped[col] = grouped[col].astype("float32")
    if "target_label" in grouped.columns:
        grouped["target_label"] = grouped["target_label"].astype("int8")
    return grouped
