from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupShuffleSplit, train_test_split


PATIENT_COL = "patient_id"
TARGET_COL = "target_label"
FEATURE_COLS = ["hr", "spo2", "bp_sys", "bp_dia", "bp_mean"]


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def load_dataset(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported dataset format: {path}. Expected .parquet or .csv")


def validate_dataset_columns(df: pd.DataFrame) -> None:
    required = {PATIENT_COL, TARGET_COL, *FEATURE_COLS}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Dataset missing required columns: {missing}")


def _patient_level_split(
    df: pd.DataFrame,
    *,
    train_size: float,
    val_size: float,
    test_size: float,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not np.isclose(train_size + val_size + test_size, 1.0):
        raise ValueError("train_size + val_size + test_size must sum to 1.0")

    groups = df[PATIENT_COL].astype("int64")
    splitter_train = GroupShuffleSplit(n_splits=1, train_size=train_size, random_state=random_state)
    train_idx, temp_idx = next(splitter_train.split(df, groups=groups))
    train_df = df.iloc[train_idx].copy()
    temp_df = df.iloc[temp_idx].copy()

    relative_val_size = val_size / (val_size + test_size)
    splitter_val = GroupShuffleSplit(n_splits=1, train_size=relative_val_size, random_state=random_state + 1)
    val_idx, test_idx = next(splitter_val.split(temp_df, groups=temp_df[PATIENT_COL].astype("int64")))

    val_df = temp_df.iloc[val_idx].copy()
    test_df = temp_df.iloc[test_idx].copy()
    return train_df, val_df, test_df


def _row_level_split(
    df: pd.DataFrame,
    *,
    train_size: float,
    val_size: float,
    test_size: float,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not np.isclose(train_size + val_size + test_size, 1.0):
        raise ValueError("train_size + val_size + test_size must sum to 1.0")

    train_df, temp_df = train_test_split(
        df,
        train_size=train_size,
        random_state=random_state,
        stratify=df[TARGET_COL] if df[TARGET_COL].nunique() > 1 else None,
    )
    relative_val_size = val_size / (val_size + test_size)
    val_df, test_df = train_test_split(
        temp_df,
        train_size=relative_val_size,
        random_state=random_state + 1,
        stratify=temp_df[TARGET_COL] if temp_df[TARGET_COL].nunique() > 1 else None,
    )
    return train_df.copy(), val_df.copy(), test_df.copy()


def split_dataset(
    df: pd.DataFrame,
    *,
    train_size: float = 0.70,
    val_size: float = 0.15,
    test_size: float = 0.15,
    random_state: int = 42,
    patient_level: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if patient_level and PATIENT_COL in df.columns:
        return _patient_level_split(
            df,
            train_size=train_size,
            val_size=val_size,
            test_size=test_size,
            random_state=random_state,
        )
    return _row_level_split(
        df,
        train_size=train_size,
        val_size=val_size,
        test_size=test_size,
        random_state=random_state,
    )


def split_integrity_report(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> dict[str, object]:
    train_patients = set(train_df[PATIENT_COL].astype("int64").tolist())
    val_patients = set(val_df[PATIENT_COL].astype("int64").tolist())
    test_patients = set(test_df[PATIENT_COL].astype("int64").tolist())
    return {
        "train_rows": int(len(train_df)),
        "val_rows": int(len(val_df)),
        "test_rows": int(len(test_df)),
        "train_patients": int(len(train_patients)),
        "val_patients": int(len(val_patients)),
        "test_patients": int(len(test_patients)),
        "overlap_train_val": int(len(train_patients & val_patients)),
        "overlap_train_test": int(len(train_patients & test_patients)),
        "overlap_val_test": int(len(val_patients & test_patients)),
        "target_rate_train": float(train_df[TARGET_COL].mean()),
        "target_rate_val": float(val_df[TARGET_COL].mean()),
        "target_rate_test": float(test_df[TARGET_COL].mean()),
    }


def prepare_features_and_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    X = df[FEATURE_COLS].copy()
    y = pd.to_numeric(df[TARGET_COL], errors="coerce").fillna(0).astype("int8")
    return X, y


def fit_imputer(X_train: pd.DataFrame) -> SimpleImputer:
    imputer = SimpleImputer(strategy="median")
    imputer.fit(X_train[FEATURE_COLS])
    return imputer


def apply_imputer(imputer: SimpleImputer, X: pd.DataFrame) -> pd.DataFrame:
    transformed = imputer.transform(X[FEATURE_COLS])
    return pd.DataFrame(transformed, columns=FEATURE_COLS, index=X.index)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

