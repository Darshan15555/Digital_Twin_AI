from __future__ import annotations

import numpy as np
import pandas as pd


DETERIORATION_REQUIRED_CURRENT_COLUMNS = ["sao2_mean", "sbp_mean", "gcs_total", "lactate_value", "vasopressor_active"]
DETERIORATION_REQUIRED_NEXT_COLUMNS = [
    "next_sao2_mean",
    "next_sbp_mean",
    "next_gcs_total",
    "next_lactate_value",
    "next_vasopressor_active",
]


def _as_binary(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(0).gt(0).astype("int8")


def _add_patient_outcome_labels(features_df: pd.DataFrame, patients_df: pd.DataFrame) -> pd.DataFrame:
    outcomes = patients_df[
        ["patientunitstayid", "hospital_mortality", "icu_mortality", "icu_los_hours"]
    ].copy()
    outcomes["label_hospital_mortality"] = _as_binary(outcomes["hospital_mortality"])
    outcomes["label_icu_mortality"] = _as_binary(outcomes["icu_mortality"])
    outcomes["label_icu_los_hours"] = pd.to_numeric(outcomes["icu_los_hours"], errors="coerce").astype("float32")
    outcomes["label_icu_los_days"] = (outcomes["label_icu_los_hours"] / 24.0).astype("float32")
    outcomes["label_prolonged_los"] = outcomes["label_icu_los_hours"].gt(168).astype("int8")
    outcomes["label_short_los"] = outcomes["label_icu_los_hours"].lt(48).astype("int8")

    label_cols = [column for column in outcomes.columns if column.startswith("label_")]
    return features_df.merge(outcomes[["patientunitstayid"] + label_cols], on="patientunitstayid", how="left")


def _compute_mortality_horizon_labels(df: pd.DataFrame) -> pd.DataFrame:
    death_time = np.where(df["label_icu_mortality"].eq(1), df["label_icu_los_hours"], np.inf)
    hours_until_death = death_time - pd.to_numeric(df["window_end_hour"], errors="coerce")
    df["label_mortality_24h"] = ((hours_until_death >= 0) & (hours_until_death <= 24)).astype("int8")
    df["label_mortality_48h"] = ((hours_until_death >= 0) & (hours_until_death <= 48)).astype("int8")
    df["label_mortality_72h"] = ((hours_until_death >= 0) & (hours_until_death <= 72)).astype("int8")
    return df


def _compute_deterioration_next_label(df: pd.DataFrame) -> pd.DataFrame:
    ordered = df.sort_values(["patientunitstayid", "window_id"]).copy()
    group = ordered.groupby("patientunitstayid", observed=True)

    shift_map = {
        "sao2_mean": "next_sao2_mean",
        "sbp_mean": "next_sbp_mean",
        "gcs_total": "next_gcs_total",
        "lactate_value": "next_lactate_value",
        "vasopressor_active": "next_vasopressor_active",
    }
    for current_col, next_col in shift_map.items():
        if current_col in ordered.columns:
            ordered[next_col] = group[current_col].shift(-1)
        else:
            ordered[next_col] = np.nan

    deterioration = (
        ((ordered["sao2_mean"] - ordered["next_sao2_mean"]) > 5)
        | ((ordered["sbp_mean"] - ordered["next_sbp_mean"]) > 20)
        | ((ordered["gcs_total"] - ordered["next_gcs_total"]) > 2)
        | ((ordered["next_lactate_value"] - ordered["lactate_value"]) > 1.0)
        | (ordered["next_vasopressor_active"] > ordered["vasopressor_active"])
    )

    current_available = ordered[DETERIORATION_REQUIRED_CURRENT_COLUMNS].notna().any(axis=1)
    next_available = ordered[DETERIORATION_REQUIRED_NEXT_COLUMNS].notna().any(axis=1)
    ordered["label_deterioration_next"] = deterioration.where(current_available & next_available).astype("float32")

    last_window = group["window_id"].transform("max")
    ordered.loc[ordered["window_id"].eq(last_window), "label_deterioration_next"] = np.nan
    ordered = ordered.drop(columns=DETERIORATION_REQUIRED_NEXT_COLUMNS)
    return ordered


def compute_all_labels(
    features_df: pd.DataFrame,
    patients_df: pd.DataFrame,
    vitals_df,
    labs_df,
) -> pd.DataFrame:
    del vitals_df, labs_df

    labeled = _add_patient_outcome_labels(features_df.copy(), patients_df)
    labeled = _compute_mortality_horizon_labels(labeled)
    labeled = _compute_deterioration_next_label(labeled)

    return labeled.sort_values(["patientunitstayid", "window_id"]).reset_index(drop=True)