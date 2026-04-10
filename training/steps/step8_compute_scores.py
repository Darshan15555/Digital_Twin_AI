from __future__ import annotations

import numpy as np
import pandas as pd


DELTA_FEATURES = {
    "hr_mean": "hr_delta",
    "sao2_mean": "sao2_delta",
    "sbp_mean": "sbp_delta",
    "resp_mean": "resp_delta",
    "temp_mean": "temp_delta",
    "map_mean": "map_delta",
    "gcs_total": "gcs_delta",
    "lactate_value": "lactate_delta",
    "creatinine_value": "creatinine_delta",
    "pf_ratio": "pf_ratio_delta",
}

WINDOW_HOURS = 4.0


def _safe_numeric(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype="float32")
    return pd.to_numeric(df[column], errors="coerce")


def _news_component_resp(series: pd.Series) -> pd.Series:
    return pd.Series(
        np.select(
            [series <= 8, series.between(9, 11), series.between(21, 24), series >= 25],
            [3, 1, 2, 3],
            default=0,
        ),
        index=series.index,
    )


def _news_component_sao2(series: pd.Series) -> pd.Series:
    return pd.Series(
        np.select([series <= 91, series.isin([92, 93]), series.isin([94, 95])], [3, 2, 1], default=0),
        index=series.index,
    )


def _news_component_temp(series: pd.Series) -> pd.Series:
    return pd.Series(
        np.select(
            [series <= 35.0, series.between(35.1, 36.0), series.between(38.1, 39.0), series >= 39.1],
            [3, 1, 1, 2],
            default=0,
        ),
        index=series.index,
    )


def _news_component_sbp(series: pd.Series) -> pd.Series:
    return pd.Series(
        np.select([series <= 90, series.between(91, 100), series.between(101, 110), series >= 220], [3, 2, 1, 3], default=0),
        index=series.index,
    )


def _news_component_hr(series: pd.Series) -> pd.Series:
    return pd.Series(
        np.select(
            [series <= 40, series.between(41, 50), series.between(91, 110), series.between(111, 130), series >= 131],
            [3, 1, 1, 2, 3],
            default=0,
        ),
        index=series.index,
    )


def _compute_deltas(ordered: pd.DataFrame) -> pd.DataFrame:
    group = ordered.groupby("patientunitstayid", observed=True)
    for source_col, delta_col in DELTA_FEATURES.items():
        ordered[delta_col] = group[source_col].diff() if source_col in ordered.columns else np.nan
    return ordered


def _compute_cumulative_features(ordered: pd.DataFrame) -> pd.DataFrame:
    group = ordered.groupby("patientunitstayid", observed=True)

    ordered["cumulative_fluid_balance"] = group["fluid_balance_window"].cumsum() if "fluid_balance_window" in ordered.columns else np.nan
    ordered["cumulative_fluid_intake"] = group["fluid_intake_window"].cumsum() if "fluid_intake_window" in ordered.columns else np.nan
    ordered["cumulative_fluid_output"] = group["fluid_output_window"].cumsum() if "fluid_output_window" in ordered.columns else np.nan
    ordered["cumulative_urine_output"] = group["urine_output_window"].cumsum() if "urine_output_window" in ordered.columns else np.nan
    ordered["cumulative_vasopressor_h"] = (
        group["vasopressor_active"].cumsum() * WINDOW_HOURS if "vasopressor_active" in ordered.columns else np.nan
    )
    ordered["cumulative_ventilator_h"] = (
        group["ventilator_active"].cumsum() * WINDOW_HOURS if "ventilator_active" in ordered.columns else np.nan
    )
    ordered["max_lactate_so_far"] = group["lactate_value"].cummax() if "lactate_value" in ordered.columns else np.nan
    ordered["min_sao2_so_far"] = group["sao2_min"].cummin() if "sao2_min" in ordered.columns else np.nan
    ordered["min_sbp_so_far"] = group["sbp_min"].cummin() if "sbp_min" in ordered.columns else np.nan
    ordered["min_map_so_far"] = group["map_mean"].cummin() if "map_mean" in ordered.columns else np.nan
    ordered["min_gcs_so_far"] = group["gcs_total"].cummin() if "gcs_total" in ordered.columns else np.nan
    ordered["max_creatinine_so_far"] = group["creatinine_value"].cummax() if "creatinine_value" in ordered.columns else np.nan
    return ordered


def _compute_qsofa(ordered: pd.DataFrame) -> pd.DataFrame:
    resp = _safe_numeric(ordered, "resp_mean")
    sbp = _safe_numeric(ordered, "sbp_mean")
    gcs = _safe_numeric(ordered, "gcs_total")

    qsofa = ((resp >= 22).astype("int8") + (sbp <= 100).astype("int8") + (gcs < 15).astype("int8")).astype("float32")
    qsofa_missing = resp.isna() & sbp.isna() & gcs.isna()
    ordered["qsofa_score"] = qsofa.mask(qsofa_missing)
    ordered["qsofa_positive"] = ordered["qsofa_score"].ge(2).fillna(False).astype("int8")
    return ordered


def _compute_sofa(ordered: pd.DataFrame) -> pd.DataFrame:
    pf_ratio = _safe_numeric(ordered, "pf_ratio")
    map_mean = _safe_numeric(ordered, "map_mean")
    vasopressor = _safe_numeric(ordered, "vasopressor_active")
    creatinine = _safe_numeric(ordered, "creatinine_value")
    gcs = _safe_numeric(ordered, "gcs_total")
    platelets = _safe_numeric(ordered, "platelets_value")
    bilirubin = _safe_numeric(ordered, "bilirubin_value")

    resp_sofa = pd.Series(
        np.select([pf_ratio < 100, pf_ratio < 200, pf_ratio < 300, pf_ratio < 400], [4, 3, 2, 1], default=0),
        index=ordered.index,
    )
    cv_sofa = pd.Series(
        np.select([vasopressor >= 1, map_mean < 70], [2, 1], default=0),
        index=ordered.index,
    )
    renal_sofa = pd.Series(
        np.select([creatinine >= 5.0, creatinine >= 3.5, creatinine >= 2.0, creatinine >= 1.2], [4, 3, 2, 1], default=0),
        index=ordered.index,
    )
    neuro_sofa = pd.Series(
        np.select([gcs < 6, gcs < 10, gcs < 13, gcs < 15], [4, 3, 2, 1], default=0),
        index=ordered.index,
    )
    coag_sofa = pd.Series(
        np.select([platelets < 20, platelets < 50, platelets < 100, platelets < 150], [4, 3, 2, 1], default=0),
        index=ordered.index,
    )
    liver_sofa = pd.Series(
        np.select([bilirubin >= 12.0, bilirubin >= 6.0, bilirubin >= 2.0, bilirubin >= 1.2], [4, 3, 2, 1], default=0),
        index=ordered.index,
    )

    components = pd.DataFrame(
        {
            "sofa_respiration": resp_sofa,
            "sofa_cardiovascular": cv_sofa,
            "sofa_renal": renal_sofa,
            "sofa_neurologic": neuro_sofa,
            "sofa_coagulation": coag_sofa,
            "sofa_liver": liver_sofa,
        },
        index=ordered.index,
    )
    ordered[list(components.columns)] = components.astype("float32")
    sofa_missing = pd.concat([pf_ratio, map_mean, vasopressor, creatinine, gcs, platelets, bilirubin], axis=1).isna().all(axis=1)
    ordered["sofa_score"] = components.sum(axis=1).astype("float32").mask(sofa_missing)
    ordered["sofa_partial"] = ordered["sofa_score"]
    ordered["sofa_high_risk"] = ordered["sofa_score"].ge(8).fillna(False).astype("int8")
    return ordered


def _compute_sirs(ordered: pd.DataFrame) -> pd.DataFrame:
    temp = _safe_numeric(ordered, "temp_mean")
    hr = _safe_numeric(ordered, "hr_mean")
    resp = _safe_numeric(ordered, "resp_mean")
    pco2 = _safe_numeric(ordered, "pco2_value")
    wbc = _safe_numeric(ordered, "wbc_value")

    ordered["sirs_score"] = (
        ((temp > 38.0) | (temp < 36.0)).astype("int8")
        + (hr > 90).astype("int8")
        + ((resp > 20) | (pco2 < 32)).astype("int8")
        + ((wbc > 12) | (wbc < 4)).astype("int8")
    ).astype("float32")
    sirs_missing = pd.concat([temp, hr, resp, pco2, wbc], axis=1).isna().all(axis=1)
    ordered["sirs_score"] = ordered["sirs_score"].mask(sirs_missing)
    ordered["sirs_positive"] = ordered["sirs_score"].ge(2).fillna(False).astype("int8")
    return ordered


def _compute_news(ordered: pd.DataFrame) -> pd.DataFrame:
    resp = _safe_numeric(ordered, "resp_mean")
    sao2 = _safe_numeric(ordered, "sao2_mean")
    sbp = _safe_numeric(ordered, "sbp_mean")
    hr = _safe_numeric(ordered, "hr_mean")
    temp = _safe_numeric(ordered, "temp_mean")
    gcs = _safe_numeric(ordered, "gcs_total")

    news = (
        _news_component_resp(resp)
        + _news_component_sao2(sao2)
        + _news_component_sbp(sbp)
        + _news_component_hr(hr)
        + _news_component_temp(temp)
        + (gcs < 15).fillna(False).astype("int8") * 3
    ).astype("float32")
    news_missing = pd.concat([resp, sao2, sbp, hr, temp, gcs], axis=1).isna().all(axis=1)
    ordered["news_score"] = news.mask(news_missing)
    ordered["news_high_risk"] = ordered["news_score"].ge(7).fillna(False).astype("int8")
    return ordered


def _compute_additional_scores(ordered: pd.DataFrame) -> pd.DataFrame:
    hr = _safe_numeric(ordered, "hr_mean")
    sbp = _safe_numeric(ordered, "sbp_mean")
    lactate = _safe_numeric(ordered, "lactate_value")

    ordered["shock_index"] = (hr / sbp.replace({0: np.nan})).clip(0, 5).astype("float32")
    ordered["lactate_shock_flag"] = ((ordered["shock_index"] >= 1.0) & (lactate >= 2.0)).fillna(False).astype("int8")
    return ordered


def compute_all_scores(df: pd.DataFrame) -> pd.DataFrame:
    ordered = df.sort_values(["patientunitstayid", "window_id"]).copy()
    ordered = _compute_deltas(ordered)
    ordered = _compute_cumulative_features(ordered)
    ordered = _compute_qsofa(ordered)
    ordered = _compute_sofa(ordered)
    ordered = _compute_sirs(ordered)
    ordered = _compute_news(ordered)
    ordered = _compute_additional_scores(ordered)
    return ordered