from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


WINDOW_SIZE_MIN = 240
MAX_WINDOWS = 6
MAX_WINDOW_ID = MAX_WINDOWS - 1


def resolve_eicu_path(preferred: str | Path) -> Path:
    path = Path(preferred)
    if path.exists():
        return path
    fallback = Path(r"D:\physionet-data\eicu")
    if fallback.exists():
        return fallback
    raise FileNotFoundError(f"Unable to find eICU data path: {path}")


def resolve_source_file(eicu_path: Path, stem: str) -> Path:
    candidates = [eicu_path / f"{stem}.csv", eicu_path / f"{stem}.csv.gz"]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Missing source file for {stem} under {eicu_path}")


def read_csv_kwargs(source: Path) -> dict[str, object]:
    return {"compression": "gzip"} if source.suffix == ".gz" else {}


def filter_numeric_range(series: pd.Series, lower: float, upper: float) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    return numeric.where(numeric.between(lower, upper))


def normalize_temperature(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce").astype("float64")
    fahrenheit_mask = numeric > 45
    numeric = numeric.where(~fahrenheit_mask, (numeric - 32.0) * 5.0 / 9.0)
    return numeric.where(numeric.between(25, 45))


def normalize_fio2(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce").astype("float64")
    numeric = numeric.where(numeric <= 1.5, numeric / 100.0)
    return numeric.where(numeric.between(0.21, 1.0))


def compute_window_id(offset_series: pd.Series, allow_negative: bool = False) -> pd.Series:
    offsets = pd.to_numeric(offset_series, errors="coerce")
    if allow_negative:
        offsets = offsets.clip(lower=-60, upper=1439)
        window_ids = (offsets.clip(lower=0) // WINDOW_SIZE_MIN).clip(0, MAX_WINDOW_ID)
    else:
        offsets = offsets.clip(lower=0, upper=1439)
        window_ids = (offsets // WINDOW_SIZE_MIN).clip(0, MAX_WINDOW_ID)
    return window_ids.astype("Int8")


def to_float32(df: pd.DataFrame, columns: list[str]) -> None:
    for column in columns:
        if column in df.columns:
            df[column] = df[column].astype("float32")


def to_int_type(df: pd.DataFrame, columns: list[str], dtype: str) -> None:
    for column in columns:
        if column in df.columns:
            df[column] = df[column].fillna(0).astype(dtype)
