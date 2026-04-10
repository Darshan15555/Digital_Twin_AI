from __future__ import annotations

import numpy as np
import pandas as pd

JOIN_KEYS = ["patientunitstayid", "window_id"]


def merge_window_features(base: pd.DataFrame, feature_df: pd.DataFrame) -> pd.DataFrame:
    if base.empty:
        return feature_df.copy()
    if feature_df is None or feature_df.empty:
        return base.copy()

    feature_columns = [column for column in feature_df.columns if column not in JOIN_KEYS]
    if feature_df.duplicated(subset=JOIN_KEYS).any():
        aggregations = {column: "last" for column in feature_columns}
        feature_df = feature_df.groupby(JOIN_KEYS, as_index=False, observed=True).agg(aggregations)

    merged = base.merge(feature_df, on=JOIN_KEYS, how="left", sort=False)
    return merged


def add_missing_columns(df: pd.DataFrame, columns: list[str], fill_value=np.nan) -> pd.DataFrame:
    result = df.copy()
    missing_columns = [column for column in columns if column not in result.columns]
    for column in missing_columns:
        result[column] = fill_value
    return result


def assign_fixed_windows(df: pd.DataFrame, offset_col: str, window_size_min: int = 240, n_windows: int = 6) -> pd.DataFrame:
    if df.empty or offset_col not in df.columns:
        return df.copy()

    working = df.copy()
    working[offset_col] = pd.to_numeric(working[offset_col], errors="coerce")
    working = working[working[offset_col].between(0, window_size_min * n_windows - 1, inclusive="both")].copy()
    if working.empty:
        return working

    working["window_id"] = (working[offset_col] // window_size_min).astype("int8")
    return working[working["window_id"].between(0, n_windows - 1, inclusive="both")].copy()