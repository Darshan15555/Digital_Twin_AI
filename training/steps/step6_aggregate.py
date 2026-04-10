from __future__ import annotations

import numpy as np
import pandas as pd

from training.utils.carry_forward import carry_forward_with_limit
from training.utils.range_filters import compute_window_id
from training.utils.window_aggregator import add_missing_columns, merge_window_features

LAB_VALUE_COLUMNS = [
    "lactate_value",
    "creatinine_value",
    "glucose_value",
    "hemoglobin_value",
    "wbc_value",
    "potassium_value",
    "sodium_value",
    "bicarbonate_value",
    "bun_value",
    "platelets_value",
    "inr_value",
    "ph_value",
    "pco2_value",
    "po2_value",
    "alt_value",
    "ast_value",
]


def aggregate_vitals(skeleton: pd.DataFrame, vitals_periodic: pd.DataFrame, vitals_aperiodic: pd.DataFrame) -> pd.DataFrame:
    periodic = vitals_periodic.copy()
    periodic["window_id"] = compute_window_id(periodic["observationoffset"]).astype("int8")
    periodic_agg = periodic.groupby(["patientunitstayid", "window_id"], observed=True).agg(
        hr_mean=("heartrate", "mean"),
        hr_min=("heartrate", "min"),
        hr_max=("heartrate", "max"),
        hr_std=("heartrate", "std"),
        hr_obs_count=("heartrate", "count"),
        sao2_mean=("sao2", "mean"),
        sao2_min=("sao2", "min"),
        sao2_std=("sao2", "std"),
        sao2_below_90=("sao2", lambda x: int((x < 90).any())),
        resp_mean=("respiration", "mean"),
        resp_max=("respiration", "max"),
        resp_std=("respiration", "std"),
        resp_above_25=("respiration", lambda x: int((x > 25).any())),
        sbp_mean=("systemicsystolic", "mean"),
        sbp_min=("systemicsystolic", "min"),
        sbp_std=("systemicsystolic", "std"),
        sbp_below_90=("systemicsystolic", lambda x: int((x < 90).any())),
        dbp_mean=("systemicdiastolic", "mean"),
        map_mean=("map_estimate", "mean"),
        map_below_65=("map_estimate", lambda x: int((x < 65).any())),
        temp_mean=("temperature", "mean"),
        temp_max=("temperature", "max"),
        temp_fever=("temperature", lambda x: int((x > 38.3).any())),
        temp_hypothermia=("temperature", lambda x: int((x < 36.0).any())),
        cvp_mean=("cvp", "mean"),
    ).reset_index()

    aperiodic = vitals_aperiodic.copy()
    aperiodic["window_id"] = compute_window_id(aperiodic["observationoffset"]).astype("int8")
    aperiodic_agg = aperiodic.groupby(["patientunitstayid", "window_id"], observed=True).agg(
        nibp_systolic_mean=("noninvasivesystolic", "mean"),
        nibp_mean_mean=("noninvasivemean", "mean"),
        nibp_below_90=("noninvasivesystolic", lambda x: int((x < 90).any())),
    ).reset_index()

    merged = merge_window_features(skeleton, periodic_agg)
    return merge_window_features(merged, aperiodic_agg)


def _recompute_lab_flags(df: pd.DataFrame) -> pd.DataFrame:
    df["lactate_above_2"] = (df["lactate_value"] > 2).fillna(False).astype("int8")
    df["lactate_above_4"] = (df["lactate_value"] > 4).fillna(False).astype("int8")
    df["creatinine_above_2"] = (df["creatinine_value"] > 2).fillna(False).astype("int8")
    df["glucose_hypoglycemia"] = (df["glucose_value"] < 70).fillna(False).astype("int8")
    df["glucose_hyperglycemia"] = (df["glucose_value"] > 180).fillna(False).astype("int8")
    df["hemoglobin_low"] = (df["hemoglobin_value"] < 7).fillna(False).astype("int8")
    df["wbc_high"] = (df["wbc_value"] > 12).fillna(False).astype("int8")
    df["wbc_low"] = (df["wbc_value"] < 4).fillna(False).astype("int8")
    df["potassium_abnormal"] = ((df["potassium_value"] < 3.5) | (df["potassium_value"] > 5.5)).fillna(False).astype("int8")
    df["bicarbonate_low"] = (df["bicarbonate_value"] < 18).fillna(False).astype("int8")
    df["platelets_low"] = (df["platelets_value"] < 100).fillna(False).astype("int8")
    df["inr_high"] = (df["inr_value"] > 1.5).fillna(False).astype("int8")
    df["ph_acidosis"] = (df["ph_value"] < 7.35).fillna(False).astype("int8")
    df["ph_alkalosis"] = (df["ph_value"] > 7.45).fillna(False).astype("int8")
    return df


def aggregate_labs(skeleton: pd.DataFrame, labs_df: pd.DataFrame) -> pd.DataFrame:
    labs = labs_df.copy()
    labs["window_id"] = compute_window_id(labs["labresultoffset"], allow_negative=True).astype("int8")
    last_labs = (
        labs.sort_values("labresultoffset")
        .groupby(["patientunitstayid", "window_id", "labname"], observed=True)["labresult"]
        .last()
        .reset_index()
    )
    wide = last_labs.pivot_table(index=["patientunitstayid", "window_id"], columns="labname", values="labresult", aggfunc="last").reset_index()
    wide.columns = [f"{column}_value" if column not in {"patientunitstayid", "window_id"} else column for column in wide.columns]

    measured = last_labs.copy()
    measured["measured"] = 1
    measured_wide = measured.pivot_table(index=["patientunitstayid", "window_id"], columns="labname", values="measured", aggfunc="max", fill_value=0).reset_index()
    measured_wide.columns = [f"{column}_measured" if column not in {"patientunitstayid", "window_id"} else column for column in measured_wide.columns]

    merged = merge_window_features(skeleton, wide)
    merged = merge_window_features(merged, measured_wide)
    expected_measured = [column.replace("_value", "_measured") for column in LAB_VALUE_COLUMNS]
    merged = add_missing_columns(merged, LAB_VALUE_COLUMNS, np.nan)
    merged = add_missing_columns(merged, expected_measured, 0)
    merged = carry_forward_with_limit(merged, "patientunitstayid", "window_id", LAB_VALUE_COLUMNS, limit=3)
    merged[expected_measured] = merged[expected_measured].fillna(0).astype("int8")
    return _recompute_lab_flags(merged)


def aggregate_interventions(skeleton: pd.DataFrame, interventions: dict[str, object]) -> pd.DataFrame:
    infusion = interventions["infusion"].copy()
    infusion["window_id"] = compute_window_id(infusion["infusionoffset"]).astype("int8")
    infusion_agg = infusion.groupby(["patientunitstayid", "window_id"], observed=True).agg(
        vasopressor_active=("is_vasopressor", "max"),
        epinephrine_active=("epinephrine_active", "max"),
        dopamine_active=("dopamine_active", "max"),
        vasopressin_active=("vasopressin_active", "max"),
        num_vasopressors=("drugname", lambda x: x[x.notna()].nunique()),
    ).reset_index()
    norepi = infusion.loc[infusion.get("norepinephrine_active", 0).eq(1)].groupby(["patientunitstayid", "window_id"], observed=True)["drugrate"].max().rename("norepinephrine_dose").reset_index()
    infusion_agg = infusion_agg.merge(norepi, on=["patientunitstayid", "window_id"], how="left")

    resp_care = interventions["respiratory"]["care"].copy()
    resp_care["window_id"] = compute_window_id(resp_care["respcarestatusoffset"]).astype("int8")
    resp_agg = resp_care.groupby(["patientunitstayid", "window_id"], observed=True).agg(
        ventilator_active=("on_ventilator", "max"),
        peep_mean=("peep", "mean"),
        fio2_mean=("fio2_inline", "mean"),
        peep_above_10=("peep", lambda x: int((x > 10).any())),
    ).reset_index()

    resp_chart = interventions["respiratory"]["charting"].copy()
    resp_chart["window_id"] = compute_window_id(resp_chart["respchartoffset"]).astype("int8")
    chart_fio2 = resp_chart.groupby(["patientunitstayid", "window_id"], observed=True)["fio2"].mean().rename("fio2_chart_mean").reset_index()
    resp_agg = resp_agg.merge(chart_fio2, on=["patientunitstayid", "window_id"], how="left")
    resp_agg["fio2_mean"] = resp_agg["fio2_chart_mean"].combine_first(resp_agg["fio2_mean"])
    resp_agg["fio2_above_60"] = (resp_agg["fio2_mean"] > 0.60).fillna(False).astype("int8")
    resp_agg = resp_agg.drop(columns=["fio2_chart_mean"])

    io_df = interventions["intake_output"].copy()
    io_df["window_id"] = compute_window_id(io_df["intakeoutputoffset"]).astype("int8")
    io_agg = io_df.groupby(["patientunitstayid", "window_id", "io_type"], observed=True)["cellvaluenumeric"].sum().unstack(fill_value=0).reset_index()
    io_agg = add_missing_columns(io_agg, ["intake", "output", "urine"], 0.0)
    io_agg = io_agg.rename(columns={"intake": "fluid_intake_window", "output": "fluid_output_window", "urine": "urine_output_window"})
    io_agg["fluid_balance_window"] = io_agg["fluid_intake_window"] - io_agg["fluid_output_window"]
    io_agg["urine_rate_mlhr"] = io_agg["urine_output_window"] / 4.0

    nurse = interventions["nurse_charting"].copy()
    nurse["window_id"] = compute_window_id(nurse["nursingchartoffset"]).astype("int8")
    nurse = nurse.sort_values("nursingchartoffset")
    nurse_wide = nurse.pivot_table(index=["patientunitstayid", "window_id"], columns="charttype", values="nursingchartvalue", aggfunc="last").reset_index()
    nurse_wide = add_missing_columns(nurse_wide, ["gcs_total", "gcs_motor", "gcs_verbal", "gcs_eye", "pain_score", "rass_score", "cam_icu"], np.nan)
    gcs_present = nurse_wide[["gcs_motor", "gcs_verbal", "gcs_eye"]].notna().all(axis=1)
    gcs_sum = nurse_wide[["gcs_motor", "gcs_verbal", "gcs_eye"]].sum(axis=1, min_count=3)
    invalid_sum = gcs_present & nurse_wide["gcs_total"].notna() & ((gcs_sum - nurse_wide["gcs_total"]).abs() > 1)
    nurse_wide.loc[invalid_sum, ["gcs_total", "gcs_motor", "gcs_verbal", "gcs_eye"]] = np.nan
    nurse_wide["gcs_below_8"] = (nurse_wide["gcs_total"] < 8).fillna(False).astype("int8")
    nurse_wide["gcs_measured"] = nurse_wide["gcs_total"].notna().astype("int8")
    nurse_wide["cam_icu_positive"] = (nurse_wide["cam_icu"] > 0).fillna(False).astype("int8")

    merged = merge_window_features(skeleton, infusion_agg)
    merged = merge_window_features(merged, resp_agg)
    merged = merge_window_features(merged, io_agg)
    return merge_window_features(merged, nurse_wide)


def aggregate_all_features(
    windows_df: pd.DataFrame,
    vitals_df: dict[str, pd.DataFrame],
    labs_df: pd.DataFrame,
    interventions: dict[str, object],
    patients_df: pd.DataFrame,
) -> pd.DataFrame:
    features = aggregate_vitals(windows_df, vitals_df["periodic"], vitals_df["aperiodic"])
    features = aggregate_interventions(features, interventions)
    features = aggregate_labs(features, labs_df)
    features = features.merge(patients_df, on="patientunitstayid", how="left")
    features["oliguria"] = (features["urine_rate_mlhr"] < (0.5 * features["admissionweight_kg"])).fillna(False).astype("int8")
    features["pf_ratio"] = features["po2_value"] / features["fio2_mean"].replace({0: np.nan})
    features = features[features["hr_obs_count"].fillna(0) > 0].copy()
    return features.sort_values(["patientunitstayid", "window_id"]).reset_index(drop=True)
