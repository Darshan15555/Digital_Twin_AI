"""
Feature engineering for patient-level ICU prediction datasets.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from config.config import COMORBIDITY_MAP
from utils.data_loader import (
    build_intake_output_summary,
    has_cache,
    load_apache,
    load_cached,
    load_hospitals,
    load_infusions,
    load_labs,
    load_nurse_charting,
    load_patients,
    load_past_history,
    load_respiratory_care,
    load_vitals_aperiodic,
    load_vitals_periodic,
    save_cache,
)

log = logging.getLogger(__name__)


def _first_24h(df: pd.DataFrame, offset_col: str) -> pd.DataFrame:
    return df[(df[offset_col] >= 0) & (df[offset_col] <= 1440)].copy()


def _agg_named(df: pd.DataFrame, group_col: str, agg_spec: dict[str, tuple[str, str]]) -> pd.DataFrame:
    if df.empty:
        columns = [group_col, *agg_spec.keys()]
        return pd.DataFrame(columns=columns)
    grouped = df.groupby(group_col).agg(**agg_spec).reset_index()
    return grouped


def build_feature_dataset(force: bool = False) -> pd.DataFrame:
    if has_cache("ml_dataset") and not force:
        return load_cached("ml_dataset")

    log.info("Building patient-level ML dataset")

    patients = load_patients(force=force)
    hospitals = load_hospitals(force=force)
    apache = load_apache(force=force)
    periodic = _first_24h(load_vitals_periodic(force=force), "observationoffset")
    aperiodic = _first_24h(load_vitals_aperiodic(force=force), "observationoffset")
    labs = _first_24h(load_labs(force=force), "labresultoffset")
    nurse = _first_24h(load_nurse_charting(force=force), "nursingchartoffset")
    infusions = _first_24h(load_infusions(force=force), "infusionoffset")
    intake_summary = build_intake_output_summary(force=force)
    resp = _first_24h(load_respiratory_care(force=force), "respcarestatusoffset")
    comorbidities = load_past_history(force=force)

    periodic_features = _agg_named(
        periodic,
        "patientunitstayid",
        {
            "heartrate_mean": ("heartrate", "mean"),
            "heartrate_max": ("heartrate", "max"),
            "heartrate_min": ("heartrate", "min"),
            "heartrate_std": ("heartrate", "std"),
            "sao2_mean": ("sao2", "mean"),
            "sao2_min": ("sao2", "min"),
            "resp_mean": ("respiration", "mean"),
            "resp_max": ("respiration", "max"),
            "sbp_mean": ("systemicsystolic", "mean"),
            "sbp_min": ("systemicsystolic", "min"),
            "temp_mean": ("temperature", "mean"),
            "temp_max": ("temperature", "max"),
            "cvp_mean": ("cvp", "mean"),
        },
    )

    aperiodic_features = _agg_named(
        aperiodic,
        "patientunitstayid",
        {
            "nibp_systolic_mean": ("noninvasivesystolic", "mean"),
            "nibp_mean_mean": ("noninvasivemean", "mean"),
        },
    )

    lab_targets = {
        "creatinine": ("creatinine_max", "max"),
        "glucose": ("glucose_max", "max"),
        "lactate": ("lactate_max", "max"),
        "hemoglobin": ("hemoglobin_min", "min"),
        "wbc": ("wbc_max", "max"),
        "potassium": [("potassium_min", "min"), ("potassium_max", "max")],
        "bicarbonate": ("bicarbonate_min", "min"),
        "bun": ("bun_max", "max"),
        "platelets": ("platelets_max", "max"),
    }
    lab_frames = []
    for lab_name, spec in lab_targets.items():
        sub = labs[labs["labname"] == lab_name]
        if isinstance(spec, list):
            frame = sub.groupby("patientunitstayid")["labresult"].agg(
                **{name: op for name, op in spec}
            ).reset_index()
        else:
            frame = sub.groupby("patientunitstayid")["labresult"].agg(**{spec[0]: spec[1]}).reset_index()
        lab_frames.append(frame)
    labs_features = patients[["patientunitstayid"]].copy()
    labs_features = labs_features.drop_duplicates()
    for frame in lab_frames:
        labs_features = labs_features.merge(frame, on="patientunitstayid", how="left")

    glucose_min = (
        labs[labs["labname"] == "glucose"]
        .groupby("patientunitstayid")["labresult"]
        .min()
        .reset_index(name="glucose_min")
    )
    labs_features = labs_features.merge(glucose_min, on="patientunitstayid", how="left")

    gcs_min = (
        nurse[nurse["metric_name"] == "gcs"]
        .groupby("patientunitstayid")["metric_value"]
        .min()
        .reset_index(name="gcs_min")
    )

    infusion_features = (
        infusions.groupby("patientunitstayid")
        .agg(
            on_vasopressor=("on_vasopressor", "max"),
            vasopressor_records=("on_vasopressor", "sum"),
        )
        .reset_index()
    )
    infusion_features["vasopressor_duration_hours"] = infusion_features["vasopressor_records"] * (5.0 / 60.0)
    infusion_features = infusion_features.drop(columns=["vasopressor_records"])

    fluid_features = intake_summary[[
        "patientunitstayid",
        "fluid_balance",
        "urine_output_per_hour",
    ]].rename(columns={"fluid_balance": "fluid_balance_24h"})

    resp_features = _agg_named(
        resp,
        "patientunitstayid",
        {
            "on_ventilator": ("on_ventilator", "max"),
            "peep_mean": ("peep", "mean"),
            "fio2_mean": ("fio2", "mean"),
            "priorventday1_flag": ("priorventday1", "max"),
            "priorventday2_flag": ("priorventday2", "max"),
        },
    )

    base = patients.merge(hospitals, on="hospitalid", how="left")
    base = base.merge(
        apache[["patientunitstayid", "apachescore", "predictedhospitalmortality", "preoperation_flag"]],
        on="patientunitstayid",
        how="left",
    )
    base = base.merge(periodic_features, on="patientunitstayid", how="left")
    base = base.merge(aperiodic_features, on="patientunitstayid", how="left")
    base = base.merge(labs_features, on="patientunitstayid", how="left")
    base = base.merge(gcs_min, on="patientunitstayid", how="left")
    base = base.merge(infusion_features, on="patientunitstayid", how="left")
    base = base.merge(fluid_features, on="patientunitstayid", how="left")
    base = base.merge(resp_features, on="patientunitstayid", how="left")
    base = base.merge(comorbidities, on="patientunitstayid", how="left")

    for column in COMORBIDITY_MAP.values():
        if column not in base.columns:
            base[column] = 0

    binary_defaults = [
        "on_vasopressor",
        "on_ventilator",
        "priorventday1_flag",
        "priorventday2_flag",
        *COMORBIDITY_MAP.values(),
    ]
    for column in binary_defaults:
        if column in base.columns:
            base[column] = base[column].fillna(0).astype(int)

    base["qsofa_score"] = (
        (base["resp_mean"].fillna(0) >= 22).astype(int)
        + (base["sbp_min"].fillna(999) <= 100).astype(int)
        + (base["gcs_min"].fillna(15) < 15).astype(int)
    )
    platelets_penalty = (
        (base["platelets_max"].fillna(9999) < 150).astype(int) if "platelets_max" in base.columns else 0
    )
    sofa_points = (
        (base["lactate_max"].fillna(0) > 2).astype(int)
        + (base["sao2_min"].fillna(100) < 92).astype(int)
        + (base["creatinine_max"].fillna(0) > 2).astype(int)
        + platelets_penalty
    )
    base["sofa_approx_score"] = base["qsofa_score"] + sofa_points
    base["sepsis_risk"] = (base["qsofa_score"] >= 2).astype(int)

    base["los_days"] = base["icu_los_hours"] / 24.0
    base["predicted_mortality_risk"] = np.nan
    base["predicted_mortality_label"] = None
    return save_cache("ml_dataset", base)
