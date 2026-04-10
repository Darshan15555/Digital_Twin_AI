from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from training.utils.range_filters import to_float32, to_int_type

log = logging.getLogger(__name__)

WINDOWS_PER_PATIENT = 6
SPLIT_RANDOM_SEED = 42


def _sample_patients_to_budget(df: pd.DataFrame, max_rows: int) -> pd.DataFrame:
    if max_rows <= 0 or len(df) <= max_rows:
        return df.copy()

    patient_counts = (
        df.groupby("patientunitstayid", observed=True)
        .size()
        .rename("row_count")
        .reset_index()
        .sort_values(["row_count", "patientunitstayid"])
    )

    rng = np.random.default_rng(SPLIT_RANDOM_SEED)
    shuffled_ids = patient_counts["patientunitstayid"].to_numpy().copy()
    rng.shuffle(shuffled_ids)

    count_map = patient_counts.set_index("patientunitstayid")["row_count"]
    kept_ids: list[int] = []
    used_rows = 0

    for patient_id in shuffled_ids:
        row_count = int(count_map.loc[patient_id])
        if kept_ids and used_rows + row_count > max_rows:
            continue
        if not kept_ids and row_count > max_rows:
            kept_ids.append(int(patient_id))
            break
        kept_ids.append(int(patient_id))
        used_rows += row_count
        if used_rows >= max_rows:
            break

    return df[df["patientunitstayid"].isin(kept_ids)].copy()


def _stratified_patient_split(patient_df: pd.DataFrame) -> dict[str, list[int]]:
    rng = np.random.default_rng(SPLIT_RANDOM_SEED)
    splits: dict[str, list[int]] = {"train": [], "val": [], "test": []}

    for _, group in patient_df.groupby("label_hospital_mortality", observed=True, dropna=False):
        ids = group["patientunitstayid"].drop_duplicates().to_numpy().copy()
        rng.shuffle(ids)
        n_patients = len(ids)
        if n_patients == 1:
            splits["train"].extend(ids.tolist())
            continue

        n_train = max(1, int(round(n_patients * 0.70)))
        n_val = int(round(n_patients * 0.15))
        if n_train + n_val >= n_patients:
            n_val = max(0, n_patients - n_train - 1)
        n_test = n_patients - n_train - n_val

        if n_test == 0 and n_patients >= 3:
            if n_val > 1:
                n_val -= 1
            else:
                n_train -= 1
            n_test = 1

        splits["train"].extend(ids[:n_train].tolist())
        splits["val"].extend(ids[n_train:n_train + n_val].tolist())
        splits["test"].extend(ids[n_train + n_val:n_train + n_val + n_test].tolist())

    return {key: sorted(set(value)) for key, value in splits.items()}


def _optimize_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    optimized = df.copy()

    float_cols = optimized.select_dtypes(include=["float64"]).columns.tolist()
    to_float32(optimized, float_cols)

    binary_cols: list[str] = []
    for column in optimized.columns:
        if column == "patientunitstayid":
            continue
        series = optimized[column].dropna()
        if len(series) == 0:
            continue
        if series.isin([0, 1]).all():
            binary_cols.append(column)

    nullable_float_binary = {"label_deterioration_next"}
    to_int_type(optimized, [column for column in binary_cols if column not in nullable_float_binary], "int8")

    count_cols = [
        column
        for column in optimized.columns
        if column not in binary_cols
        and (
            column.endswith("_count")
            or column.endswith("_score")
            or column in {"window_id", "hr_obs_count", "num_vasopressors"}
        )
    ]
    to_int_type(optimized, count_cols, "int16")

    if "patientunitstayid" in optimized.columns:
        optimized["patientunitstayid"] = pd.to_numeric(optimized["patientunitstayid"], errors="coerce").astype("int32")
    if "window_id" in optimized.columns:
        optimized["window_id"] = pd.to_numeric(optimized["window_id"], errors="coerce").astype("int8")
    if "label_deterioration_next" in optimized.columns:
        optimized["label_deterioration_next"] = pd.to_numeric(
            optimized["label_deterioration_next"], errors="coerce"
        ).astype("float32")

    return optimized


def _build_metadata(df: pd.DataFrame, label_cols: list[str], splits: dict[str, list[int]]) -> dict[str, object]:
    feature_cols = [column for column in df.columns if not column.startswith("label_")]
    split_rows = {
        split_name: int(df["patientunitstayid"].isin(patient_ids).sum())
        for split_name, patient_ids in splits.items()
    }
    split_patients = {split_name: int(len(patient_ids)) for split_name, patient_ids in splits.items()}

    metadata = {
        "created_at": datetime.now().isoformat(),
        "total_rows": int(len(df)),
        "total_patients": int(df["patientunitstayid"].nunique()),
        "total_windows": int(len(df)),
        "avg_windows_per_patient": float(len(df) / max(df["patientunitstayid"].nunique(), 1)),
        "expected_windows_per_patient": WINDOWS_PER_PATIENT,
        "mortality_rate": float(df["label_hospital_mortality"].mean()) if "label_hospital_mortality" in df.columns else None,
        "icu_mortality_rate": float(df["label_icu_mortality"].mean()) if "label_icu_mortality" in df.columns else None,
        "deterioration_rate": float(df["label_deterioration_next"].mean()) if "label_deterioration_next" in df.columns else None,
        "features": feature_cols,
        "labels": label_cols,
        "window_coverage": {str(int(k)): int(v) for k, v in df["window_id"].value_counts().sort_index().items()},
        "missingness_report": {column: float(df[column].isna().mean()) for column in df.columns},
        "splits": {
            split_name: {"patients": split_patients[split_name], "rows": split_rows[split_name]}
            for split_name in ["train", "val", "test"]
        },
        "dtypes": {column: str(dtype) for column, dtype in df.dtypes.items()},
    }
    return metadata


def save_dataset(df: pd.DataFrame, output_path: Path, max_rows: int) -> dict[str, object]:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = _sample_patients_to_budget(df, max_rows)
    df = df.sort_values(["patientunitstayid", "window_id"]).reset_index(drop=True)

    index_cols = ["patientunitstayid", "window_id", "window_start_hour", "window_end_hour"]
    label_cols = [column for column in df.columns if column.startswith("label_")]
    ordered_cols = index_cols + [column for column in df.columns if column not in index_cols + label_cols] + label_cols
    df = df.loc[:, ordered_cols]

    df = _optimize_dtypes(df)

    df.to_parquet(output_path, index=False, compression="snappy")

    patient_labels = df[["patientunitstayid", "label_hospital_mortality"]].drop_duplicates()
    splits = _stratified_patient_split(patient_labels)
    split_files: dict[str, str] = {}

    for split_name, ids in splits.items():
        split_path = output_path.parent / f"ts_{split_name}.parquet"
        df[df["patientunitstayid"].isin(ids)].to_parquet(split_path, index=False, compression="snappy")
        split_files[split_name] = split_path.name

    metadata = _build_metadata(df, label_cols, splits)
    metadata["output_file"] = output_path.name
    metadata["split_files"] = split_files

    (output_path.parent / "ts_dataset_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (output_path.parent / "ts_splits.json").write_text(json.dumps(splits, indent=2), encoding="utf-8")

    feature_cols = [column for column in df.columns if not column.startswith("label_")]
    missingness = df[feature_cols].isnull().mean().sort_values(ascending=False)

    log.info("Total rows: %s", f"{len(df):,}")
    log.info("Total patients: %s", f"{df['patientunitstayid'].nunique():,}")
    log.info("Total features: %s", len(feature_cols))
    log.info("Total label columns: %s", len(label_cols))
    if "label_hospital_mortality" in df.columns:
        log.info("Mortality rate: %.2f%%", df["label_hospital_mortality"].mean() * 100.0)
    if "label_deterioration_next" in df.columns:
        log.info("Deterioration rate: %.2f%%", df["label_deterioration_next"].mean() * 100.0)
    log.info("Avg windows per patient: %.2f", len(df) / max(df["patientunitstayid"].nunique(), 1))
    for column, pct in missingness.head(20).items():
        log.info("Missingness | %s: %.1f%%", column, pct * 100.0)
    log.info("File size: %.1f MB", output_path.stat().st_size / 1e6)

    return {
        "shape": tuple(df.shape),
        "mortality_rate": float(df["label_hospital_mortality"].mean()) if "label_hospital_mortality" in df.columns else float("nan"),
        "patients": int(df["patientunitstayid"].nunique()),
        "avg_windows": float(len(df) / max(df["patientunitstayid"].nunique(), 1)),
    }