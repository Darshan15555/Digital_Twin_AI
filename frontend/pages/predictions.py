from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from frontend.components.cards import metric_card, risk_badge
from frontend.components.charts import BASE_LAYOUT

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS_DIR = PROJECT_ROOT / "models" / "artifacts"
TRAIN_PATH = PROJECT_ROOT / "data" / "ts_train.parquet"
SIMILAR_PATIENT_K = 3

RECOMMENDED_MORTALITY_COLUMNS = [
    "window_id",
    "age",
    "hr_mean",
    "sao2_mean",
    "resp_mean",
    "sbp_mean",
    "temp_mean",
    "gcs_total",
    "lactate_value",
    "creatinine_value",
    "ph_value",
    "fio2_mean",
    "vasopressor_active",
    "nibp_below_90",
    "sao2_below_90",
    "diag_sepsis",
    "diag_respiratory",
    "apache_diagnosis",
]

ACTIVE_DETERIORATION = "ACTIVE_DETERIORATION"

STATE_ORDER = ["STABLE", "EARLY_DETERIORATION", ACTIVE_DETERIORATION, "SHOCK", "RECOVERY"]
STATE_COLORS = {
    "STABLE": "#00E5A0",
    "EARLY_DETERIORATION": "#FFB830",
    "ACTIVE_DETERIORATION": "#FF7A30",
    "SHOCK": "#FF4560",
    "RECOVERY": "#60A5FA",
}

STATE_ALIASES = {
    "ACTIVE_DETIORATION": ACTIVE_DETERIORATION,
    "ACTIVE_DTERIORATION": ACTIVE_DETERIORATION,
}


@st.cache_resource
def _load_predictor():
    from models.inference.predictor import ICUPredictor

    return ICUPredictor()


@st.cache_data
def _load_prep_config() -> dict:
    config_path = ARTIFACTS_DIR / "preprocessing_config.json"
    if not config_path.exists():
        return {}
    return json.loads(config_path.read_text(encoding="utf-8"))


@st.cache_data
def _load_similar_reference() -> pd.DataFrame:
    columns = [
        "patientunitstayid",
        "window_id",
        "label_hospital_mortality",
        "label_mortality_24h",
        "label_mortality_48h",
        "label_mortality_72h",
        "label_deterioration_next",
        "hr_mean",
        "sao2_mean",
        "resp_mean",
        "sbp_mean",
        "map_mean",
        "temp_mean",
        "gcs_total",
        "lactate_value",
        "creatinine_value",
        "vasopressor_active",
    ]
    try:
        df = pd.read_parquet(TRAIN_PATH, columns=columns)
    except Exception:
        fallback = pd.read_parquet(TRAIN_PATH)
        available = [col for col in columns if col in fallback.columns]
        df = fallback[available].copy()
    if "map_mean" not in df.columns and "sbp_mean" in df.columns:
        df["map_mean"] = pd.to_numeric(df["sbp_mean"], errors="coerce")
    shock_like = (
        (pd.to_numeric(df.get("map_mean"), errors="coerce") < 65)
        | (pd.to_numeric(df.get("lactate_value"), errors="coerce") > 4.0)
        | (pd.to_numeric(df.get("label_deterioration_next"), errors="coerce") == 1)
        | (pd.to_numeric(df.get("vasopressor_active"), errors="coerce") > 0)
    )
    df["critical_event_hour"] = np.where(shock_like, pd.to_numeric(df.get("window_id"), errors="coerce") * 4.0, np.nan)
    patient_summary = (
        df.groupby("patientunitstayid", as_index=False)
        .agg(
            final_hr=("hr_mean", "last"),
            final_sao2=("sao2_mean", "last"),
            final_resp=("resp_mean", "last"),
            final_sbp=("sbp_mean", "last"),
            final_temp=("temp_mean", "last"),
            final_gcs=("gcs_total", "last"),
            final_lactate=("lactate_value", "last"),
            final_creatinine=("creatinine_value", "last"),
            max_vasopressor=("vasopressor_active", "max"),
            mortality=("label_hospital_mortality", "max"),
            mortality_24h=("label_mortality_24h", "max"),
            mortality_48h=("label_mortality_48h", "max"),
            mortality_72h=("label_mortality_72h", "max"),
            median_map=("map_mean", "median"),
            first_critical_event_hour=("critical_event_hour", "min"),
            windows=("window_id", "max"),
        )
    )
    patient_summary["critical_event_observed"] = patient_summary["first_critical_event_hour"].notna().astype(int)
    return patient_summary.fillna(patient_summary.median(numeric_only=True))


def _predict_dataframe(df: pd.DataFrame, threshold_type: str) -> pd.DataFrame:
    predictor = _load_predictor()
    results: list[dict] = []
    for _, row in df.iterrows():
        payload = {key: value for key, value in row.to_dict().items() if pd.notna(value)}
        pred = predictor.predict_mortality(payload, threshold_type=threshold_type)
        results.append(
            {
                **row.to_dict(),
                "mortality_risk_score": pred.get("risk_score"),
                "mortality_risk_label": pred.get("risk_label"),
                "mortality_prediction": pred.get("prediction"),
                "mortality_threshold": pred.get("threshold_used"),
                "model_ready": pred.get("model_ready"),
                "error": pred.get("error"),
            }
        )
    results_df = pd.DataFrame(results)
    if "window_id" in results_df.columns:
        results_df["window_id"] = pd.to_numeric(results_df["window_id"], errors="coerce")
        results_df = results_df.sort_values("window_id").reset_index(drop=True)
    return results_df


def _sample_template() -> pd.DataFrame:
    rows = [
        {"window_id": 0, "age": 74, "hr_mean": 98, "sao2_mean": 94, "resp_mean": 24, "sbp_mean": 105, "temp_mean": 38.2, "gcs_total": 14, "lactate_value": 2.1, "creatinine_value": 1.4, "ph_value": 7.36, "fio2_mean": 0.40, "vasopressor_active": 0, "nibp_below_90": 0, "sao2_below_90": 0, "diag_sepsis": 1, "diag_respiratory": 1, "apache_diagnosis": "Sepsis, pulmonary"},
        {"window_id": 1, "age": 74, "hr_mean": 105, "sao2_mean": 92, "resp_mean": 26, "sbp_mean": 98, "temp_mean": 38.5, "gcs_total": 13, "lactate_value": 2.8, "creatinine_value": 1.5, "ph_value": 7.33, "fio2_mean": 0.50, "vasopressor_active": 0, "nibp_below_90": 0, "sao2_below_90": 0, "diag_sepsis": 1, "diag_respiratory": 1, "apache_diagnosis": "Sepsis, pulmonary"},
        {"window_id": 2, "age": 74, "hr_mean": 112, "sao2_mean": 91, "resp_mean": 28, "sbp_mean": 88, "temp_mean": 38.8, "gcs_total": 12, "lactate_value": 3.6, "creatinine_value": 1.6, "ph_value": 7.31, "fio2_mean": 0.60, "vasopressor_active": 1, "nibp_below_90": 1, "sao2_below_90": 0, "diag_sepsis": 1, "diag_respiratory": 1, "apache_diagnosis": "Sepsis, pulmonary"},
        {"window_id": 3, "age": 74, "hr_mean": 118, "sao2_mean": 89, "resp_mean": 30, "sbp_mean": 84, "temp_mean": 39.1, "gcs_total": 11, "lactate_value": 4.5, "creatinine_value": 1.8, "ph_value": 7.28, "fio2_mean": 0.80, "vasopressor_active": 1, "nibp_below_90": 1, "sao2_below_90": 1, "diag_sepsis": 1, "diag_respiratory": 1, "apache_diagnosis": "Sepsis, pulmonary"},
        {"window_id": 4, "age": 74, "hr_mean": 122, "sao2_mean": 88, "resp_mean": 32, "sbp_mean": 82, "temp_mean": 38.6, "gcs_total": 10, "lactate_value": 5.8, "creatinine_value": 2.1, "ph_value": 7.24, "fio2_mean": 0.90, "vasopressor_active": 1, "nibp_below_90": 1, "sao2_below_90": 1, "diag_sepsis": 1, "diag_respiratory": 1, "apache_diagnosis": "Sepsis, pulmonary"},
        {"window_id": 5, "age": 74, "hr_mean": 128, "sao2_mean": 86, "resp_mean": 34, "sbp_mean": 78, "temp_mean": 37.8, "gcs_total": 9, "lactate_value": 7.2, "creatinine_value": 2.4, "ph_value": 7.19, "fio2_mean": 1.00, "vasopressor_active": 1, "nibp_below_90": 1, "sao2_below_90": 1, "diag_sepsis": 1, "diag_respiratory": 1, "apache_diagnosis": "Sepsis, pulmonary"},
    ]
    return pd.DataFrame(rows, columns=RECOMMENDED_MORTALITY_COLUMNS)


def _classify_state(row: pd.Series, prev_state: str | None, prev_row: pd.Series | None) -> str:
    hr = pd.to_numeric(row.get("hr_mean"), errors="coerce")
    sao2 = pd.to_numeric(row.get("sao2_mean"), errors="coerce")
    sbp = pd.to_numeric(row.get("sbp_mean"), errors="coerce")
    lactate = pd.to_numeric(row.get("lactate_value"), errors="coerce")
    resp = pd.to_numeric(row.get("resp_mean"), errors="coerce")
    gcs = pd.to_numeric(row.get("gcs_total"), errors="coerce")
    vasopressor = int(row.get("vasopressor_active", 0) or 0)

    qsofa = int((resp >= 22) if pd.notna(resp) else 0) + int((sbp <= 100) if pd.notna(sbp) else 0) + int((gcs < 15) if pd.notna(gcs) else 0)
    if ((pd.notna(sbp) and sbp < 90) or int(row.get("nibp_below_90", 0) or 0) == 1) and vasopressor == 1 and pd.notna(lactate) and lactate > 4:
        return "SHOCK"
    if qsofa >= 2 or (prev_row is not None and pd.notna(lactate) and pd.notna(pd.to_numeric(prev_row.get("lactate_value"), errors="coerce")) and lactate > pd.to_numeric(prev_row.get("lactate_value"), errors="coerce")):
        return ACTIVE_DETERIORATION

    trending_flags = 0
    if prev_row is not None:
        prev_hr = pd.to_numeric(prev_row.get("hr_mean"), errors="coerce")
        prev_sao2 = pd.to_numeric(prev_row.get("sao2_mean"), errors="coerce")
        prev_sbp = pd.to_numeric(prev_row.get("sbp_mean"), errors="coerce")
        prev_resp = pd.to_numeric(prev_row.get("resp_mean"), errors="coerce")
        if pd.notna(hr) and pd.notna(prev_hr) and hr > prev_hr:
            trending_flags += 1
        if pd.notna(sao2) and pd.notna(prev_sao2) and sao2 < prev_sao2:
            trending_flags += 1
        if pd.notna(sbp) and pd.notna(prev_sbp) and sbp < prev_sbp:
            trending_flags += 1
        if pd.notna(resp) and pd.notna(prev_resp) and resp > prev_resp:
            trending_flags += 1
        if prev_state in {ACTIVE_DETERIORATION, "SHOCK"}:
            improving = 0
            if pd.notna(hr) and pd.notna(prev_hr) and hr < prev_hr:
                improving += 1
            if pd.notna(sao2) and pd.notna(prev_sao2) and sao2 > prev_sao2:
                improving += 1
            if pd.notna(sbp) and pd.notna(prev_sbp) and sbp > prev_sbp:
                improving += 1
            if improving >= 2:
                return "RECOVERY"

    if trending_flags >= 2 or qsofa == 1:
        return "EARLY_DETERIORATION"
    if (pd.notna(hr) and 60 <= hr <= 100) and (pd.notna(sao2) and sao2 > 95) and (pd.notna(sbp) and sbp > 90) and (pd.notna(lactate) and lactate < 2):
        return "STABLE"
    return "EARLY_DETERIORATION"


def _add_states(results_df: pd.DataFrame) -> pd.DataFrame:
    df = results_df.copy()
    states = []
    prev_state = None
    prev_row = None
    for _, row in df.iterrows():
        state = _classify_state(row, prev_state, prev_row)
        states.append(state)
        prev_state = state
        prev_row = row
    df["clinical_state"] = states
    df["clinical_state"] = df["clinical_state"].replace(STATE_ALIASES)
    return df


def _transition_summary(df: pd.DataFrame, similar_df: pd.DataFrame | None = None) -> dict:
    if df.empty:
        return {}
    current_state = STATE_ALIASES.get(df["clinical_state"].iloc[-1], df["clinical_state"].iloc[-1])
    dwell_windows = 1
    for state in reversed(df["clinical_state"].tolist()[:-1]):
        if STATE_ALIASES.get(state, state) == current_state:
            dwell_windows += 1
        else:
            break
    risk = float(df["mortality_risk_score"].iloc[-1]) if "mortality_risk_score" in df.columns else 0.0
    if similar_df is None:
        similar_df = _find_similar_patients(df, top_k=25)
    transitions = compute_clinical_transition_probabilities(df, similar_df)
    next_state = max(transitions, key=transitions.get)
    shock_probability = float(transitions.get("SHOCK", 0.0))
    if current_state == "SHOCK" or risk >= 0.85 or shock_probability >= 0.60:
        time_to_critical = "within 4 hours"
    elif risk >= 0.65 or shock_probability >= 0.35:
        time_to_critical = "within 8 hours"
    else:
        time_to_critical = "monitor over next 12 hours"

    severity_note = (
        "Immediate high-risk physiology detected."
        if current_state == "SHOCK"
        else "Patient is worsening and may become critical soon."
        if current_state in {"EARLY_DETERIORATION", ACTIVE_DETERIORATION}
        else "Patient is relatively stable at current trajectory."
    )
    confidence = display_prediction_confidence(risk, len(similar_df))

    return {
        "current_state": current_state,
        "dwell_hours": int(dwell_windows * 4),
        "next_state": next_state,
        "next_state_probability": float(transitions[next_state]),
        "transition_probabilities": transitions,
        "time_to_critical": time_to_critical,
        "severity_note": severity_note,
        "confidence_note": confidence,
        "similar_count": int(len(similar_df)),
    }


def _state_trajectory_chart(df: pd.DataFrame) -> go.Figure:
    plot_df = df.copy()
    plot_df["clinical_state"] = plot_df["clinical_state"].replace(STATE_ALIASES)
    plot_df["state_index"] = plot_df["clinical_state"].map({state: idx for idx, state in enumerate(STATE_ORDER)})
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=plot_df["window_id"],
            y=plot_df["state_index"],
            mode="lines+markers+text",
            text=plot_df["clinical_state"],
            textposition="top center",
            line={"color": "#00C8FF", "width": 3},
            marker={"size": 12, "color": [STATE_COLORS.get(s, "#A78BFA") for s in plot_df["clinical_state"]]},
            name="Clinical State",
        )
    )
    fig.update_layout(height=340, **BASE_LAYOUT, yaxis={"tickmode": "array", "tickvals": list(range(len(STATE_ORDER))), "ticktext": STATE_ORDER}, xaxis_title="4-hour Window", yaxis_title="Clinical State")
    return fig


def _risk_trajectory_chart(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["window_id"], y=df["mortality_risk_score"], mode="lines+markers", line={"color": "#FF4560", "width": 3}, name="Mortality Risk"))
    fig.update_layout(height=300, **BASE_LAYOUT, xaxis_title="4-hour Window", yaxis_title="Predicted Mortality Risk")
    return fig


def _vitals_trajectory_chart(df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for col, color, label in [("hr_mean", "#FFB830", "HR"), ("sao2_mean", "#00E5A0", "SpO2"), ("sbp_mean", "#60A5FA", "SBP"), ("lactate_value", "#FF4560", "Lactate")]:
        if col in df.columns:
            fig.add_trace(go.Scatter(x=df["window_id"], y=df[col], mode="lines+markers", name=label, line={"color": color}))
    fig.update_layout(height=320, **BASE_LAYOUT, xaxis_title="4-hour Window", yaxis_title="Value")
    return fig


def _find_similar_patients(df: pd.DataFrame, top_k: int = SIMILAR_PATIENT_K) -> pd.DataFrame:
    ref = _load_similar_reference().copy()
    final = df.iloc[-1]
    query = pd.Series(
        {
            "final_hr": pd.to_numeric(final.get("hr_mean"), errors="coerce"),
            "final_sao2": pd.to_numeric(final.get("sao2_mean"), errors="coerce"),
            "final_resp": pd.to_numeric(final.get("resp_mean"), errors="coerce"),
            "final_sbp": pd.to_numeric(final.get("sbp_mean"), errors="coerce"),
            "final_temp": pd.to_numeric(final.get("temp_mean"), errors="coerce"),
            "final_gcs": pd.to_numeric(final.get("gcs_total"), errors="coerce"),
            "final_lactate": pd.to_numeric(final.get("lactate_value"), errors="coerce"),
            "final_creatinine": pd.to_numeric(final.get("creatinine_value"), errors="coerce"),
            "max_vasopressor": pd.to_numeric(df.get("vasopressor_active", pd.Series([0])).max(), errors="coerce"),
            "median_map": pd.to_numeric(df.get("map_mean", df.get("sbp_mean", pd.Series([np.nan]))).tail(3).median(), errors="coerce"),
            "windows": pd.to_numeric(df.get("window_id", pd.Series([0])).max(), errors="coerce"),
        }
    )
    numeric_cols = query.index.tolist()
    ref_numeric = ref[numeric_cols].copy()
    z_ref = (ref_numeric - ref_numeric.mean()) / ref_numeric.std().replace({0: 1})
    z_query = (query[numeric_cols] - ref_numeric.mean()) / ref_numeric.std().replace({0: 1})
    ref["distance"] = np.sqrt(((z_ref - z_query) ** 2).sum(axis=1))
    cols = [
        "patientunitstayid",
        "mortality",
        "mortality_24h",
        "mortality_48h",
        "mortality_72h",
        "distance",
        "first_critical_event_hour",
        "critical_event_observed",
        "final_lactate",
        "final_sbp",
        "final_sao2",
        "max_vasopressor",
    ]
    return ref.sort_values("distance").head(top_k)[cols]


def _normalize_probabilities(probs: dict[str, float]) -> dict[str, float]:
    clipped = {state: max(0.0, float(value)) for state, value in probs.items()}
    total = sum(clipped.values())
    if total <= 0:
        return {state: 1.0 / len(STATE_ORDER) for state in STATE_ORDER}
    return {state: clipped.get(state, 0.0) / total for state in STATE_ORDER}


def compute_clinical_transition_probabilities(patient_df: pd.DataFrame, historical_data: pd.DataFrame) -> dict[str, float]:
    probs = {state: 0.0 for state in STATE_ORDER}
    current_state = STATE_ALIASES.get(str(patient_df.iloc[-1]["clinical_state"]), str(patient_df.iloc[-1]["clinical_state"]))

    risk_series = pd.to_numeric(patient_df.get("mortality_risk_score", pd.Series(dtype=float)), errors="coerce").dropna()
    mortality_trend = float(risk_series.tail(3).diff().mean()) if len(risk_series) >= 2 else 0.0

    map_series = pd.to_numeric(patient_df.get("map_mean", pd.Series(dtype=float)), errors="coerce")
    if map_series.dropna().empty and "sbp_mean" in patient_df.columns:
        map_series = pd.to_numeric(patient_df["sbp_mean"], errors="coerce")
    map_trend = float(map_series.tail(3).diff().mean()) if map_series.dropna().shape[0] >= 2 else 0.0

    lactate_series = pd.to_numeric(patient_df.get("lactate_value", pd.Series(dtype=float)), errors="coerce")
    lactate_trend = float(lactate_series.tail(3).diff().mean()) if lactate_series.dropna().shape[0] >= 2 else 0.0

    gcs_series = pd.to_numeric(patient_df.get("gcs_total", pd.Series(dtype=float)), errors="coerce")
    gcs_trend = float(gcs_series.tail(3).diff().mean()) if gcs_series.dropna().shape[0] >= 2 else 0.0

    similar_mortality = float(historical_data["mortality"].mean()) if not historical_data.empty else 0.5
    similar_shock_like = (
        float(((historical_data["final_lactate"] > 4.0) | (historical_data["final_sbp"] < 90.0) | (historical_data["max_vasopressor"] > 0)).mean())
        if not historical_data.empty
        else 0.5
    )

    if current_state == "SHOCK":
        probs.update({"SHOCK": 0.58, ACTIVE_DETERIORATION: 0.18, "RECOVERY": 0.24})
    elif current_state == ACTIVE_DETERIORATION:
        probs.update({ACTIVE_DETERIORATION: 0.45, "SHOCK": 0.32, "RECOVERY": 0.15, "EARLY_DETERIORATION": 0.08})
    elif current_state == "EARLY_DETERIORATION":
        probs.update({"EARLY_DETERIORATION": 0.34, ACTIVE_DETERIORATION: 0.36, "RECOVERY": 0.20, "SHOCK": 0.10})
    elif current_state == "RECOVERY":
        probs.update({"RECOVERY": 0.40, "STABLE": 0.37, "EARLY_DETERIORATION": 0.18, ACTIVE_DETERIORATION: 0.05})
    else:
        probs.update({"STABLE": 0.62, "EARLY_DETERIORATION": 0.22, "RECOVERY": 0.10, ACTIVE_DETERIORATION: 0.05, "SHOCK": 0.01})

    if mortality_trend > 0.03:
        probs["SHOCK"] += 0.12
        probs[ACTIVE_DETERIORATION] += 0.08
        probs["RECOVERY"] -= 0.10
    elif mortality_trend < -0.03:
        probs["RECOVERY"] += 0.12
        probs["SHOCK"] -= 0.08
        probs[ACTIVE_DETERIORATION] -= 0.04

    if lactate_trend > 0.40:
        probs["SHOCK"] += 0.15
        probs["RECOVERY"] -= 0.08
    elif lactate_trend < -0.20:
        probs["RECOVERY"] += 0.10
        probs["SHOCK"] -= 0.06

    if map_trend < -2.5:
        probs["SHOCK"] += 0.10
    elif map_trend > 2.5:
        probs["RECOVERY"] += 0.08

    if gcs_trend < -0.6:
        probs[ACTIVE_DETERIORATION] += 0.08
        probs["SHOCK"] += 0.05
    elif gcs_trend > 0.5:
        probs["RECOVERY"] += 0.07

    probs["SHOCK"] += 0.12 * max(0.0, similar_mortality - 0.5)
    probs["RECOVERY"] += 0.10 * max(0.0, 0.5 - similar_mortality)
    probs["SHOCK"] += 0.10 * max(0.0, similar_shock_like - 0.5)
    probs["STABLE"] += 0.08 * max(0.0, 0.5 - similar_shock_like)

    return _normalize_probabilities(probs)


def get_clinical_recommendations(state: str, mortality_risk: float, vitals_trend: dict[str, float]) -> dict:
    recommendations = {
        "SHOCK": {
            "immediate_actions": [
                "Initiate vasopressor support when MAP < 65 mmHg.",
                "Place arterial/central access for close hemodynamic monitoring.",
                "Trend lactate hourly and target a downward trajectory.",
                "Ensure broad-spectrum infection coverage and source control review.",
                "Escalate to ICU senior team immediately.",
            ],
            "monitoring": "Continuous arterial pressure and high-frequency reassessment.",
            "escalation_criteria": "Persistent hypotension or rising lactate despite support.",
        },
        ACTIVE_DETERIORATION: {
            "immediate_actions": [
                "Increase monitoring intensity and repeat key labs early.",
                "Reassess perfusion status and fluid responsiveness.",
                "Screen for infectious, cardiogenic, and respiratory drivers.",
                "Notify senior clinician for early escalation planning.",
            ],
            "monitoring": "Hourly vitals with frequent trend review.",
            "escalation_criteria": "Transition to SHOCK signs: lactate rising and BP falling.",
        },
        "EARLY_DETERIORATION": {
            "immediate_actions": [
                "Tighten observation interval and verify data quality.",
                "Repeat lactate/ABG based on trajectory.",
                "Review medications, fluids, and oxygen strategy.",
            ],
            "monitoring": "Vital trends every 30-60 minutes.",
            "escalation_criteria": "Any sustained multi-parameter worsening over next window.",
        },
        "RECOVERY": {
            "immediate_actions": [
                "Continue current plan while watching for relapse signals.",
                "Step down interventions cautiously, not abruptly.",
            ],
            "monitoring": "Trend-focused monitoring with relapse watch.",
            "escalation_criteria": "Any reversal in BP, lactate, mental status, or oxygenation.",
        },
        "STABLE": {
            "immediate_actions": [
                "Maintain standard ICU protocol and routine reassessment.",
                "Watch early warning trends to catch deterioration promptly.",
            ],
            "monitoring": "Standard protocol interval monitoring.",
            "escalation_criteria": "Two or more worsening trend markers in a single window.",
        },
    }
    payload = recommendations.get(state, recommendations["EARLY_DETERIORATION"]).copy()
    payload["risk_context"] = (
        f"Mortality risk is {mortality_risk*100:.1f}%. "
        f"Lactate trend {vitals_trend.get('lactate_trend', 0.0):+.2f}, "
        f"MAP/SBP trend {vitals_trend.get('map_trend', 0.0):+.2f}."
    )
    return payload


def display_prediction_confidence(mortality_risk: float, n_similar_patients: int) -> str:
    if n_similar_patients < 10:
        confidence = "LOW (limited similar cases)"
    elif n_similar_patients < 50:
        confidence = "MEDIUM"
    else:
        confidence = "HIGH"
    return f"Confidence: {confidence} | Based on {n_similar_patients} similar patients | Risk {mortality_risk*100:.1f}%"


def _clip_probability(value: float) -> float:
    return float(np.clip(float(value), 0.0, 0.999))


def _estimate_risk_progression(df: pd.DataFrame, transition_probabilities: dict[str, float]) -> dict[str, float]:
    risk_series = pd.to_numeric(df.get("mortality_risk_score", pd.Series(dtype=float)), errors="coerce").dropna()
    current = float(risk_series.iloc[-1]) if not risk_series.empty else 0.0
    slope = float(risk_series.tail(3).diff().mean()) if len(risk_series) >= 2 else 0.0
    shock_p = float(transition_probabilities.get("SHOCK", 0.0))
    active_p = float(transition_probabilities.get(ACTIVE_DETERIORATION, 0.0))

    drift = slope + (0.06 * (shock_p - 0.25)) + (0.03 * (active_p - 0.25))
    if abs(drift) < 0.006:
        drift = 0.012 if current >= 0.75 else 0.005

    risk_4h = _clip_probability(current + drift)
    risk_8h = _clip_probability(risk_4h + (drift * 0.85))
    risk_12h = _clip_probability(risk_8h + (drift * 0.70))

    return {
        "now": _clip_probability(current),
        "4h": risk_4h,
        "8h": risk_8h,
        "12h": risk_12h,
    }


def _estimate_critical_window(risk_projection: dict[str, float]) -> str:
    if risk_projection.get("now", 0.0) >= 0.90:
        return "Now to 2 hours"
    if risk_projection.get("4h", 0.0) >= 0.90:
        return "2 to 4 hours"
    if risk_projection.get("8h", 0.0) >= 0.90:
        return "4 to 8 hours"
    if risk_projection.get("12h", 0.0) >= 0.90:
        return "8 to 12 hours"
    return "12 to 24 hours"


def _estimate_survival_timeline(current_risk: float, risk_projection: dict[str, float], similar_df: pd.DataFrame) -> dict[str, float]:
    n_similar = int(len(similar_df))
    historical_24 = float(1.0 - similar_df["mortality_24h"].mean()) if "mortality_24h" in similar_df.columns and not similar_df.empty else np.nan
    historical_48 = float(1.0 - similar_df["mortality_48h"].mean()) if "mortality_48h" in similar_df.columns and not similar_df.empty else np.nan
    historical_72 = float(1.0 - similar_df["mortality_72h"].mean()) if "mortality_72h" in similar_df.columns and not similar_df.empty else np.nan
    historical_hospital = float(1.0 - similar_df["mortality"].mean()) if "mortality" in similar_df.columns and not similar_df.empty else np.nan

    model_survival_24h = _clip_probability(1.0 - current_risk)
    if np.isnan(historical_24):
        survival_24h = model_survival_24h
    else:
        # Conservative rule: do not exceed either model or matched-cohort 24h survival.
        survival_24h = _clip_probability(min(model_survival_24h, historical_24))

    # 4h should not be lower than 24h survival.
    short_term = _clip_probability(1.0 - risk_projection.get("4h", current_risk))
    survival_4h = max(short_term, survival_24h)

    # High-risk trajectories require steeper long-horizon survival decay.
    # risk_severity: 0 at <=0.5 risk, 1 at >=0.9 risk.
    risk_severity = float(np.clip((current_risk - 0.5) / 0.4, 0.0, 1.0))
    decay_48 = 0.75 - (0.30 * risk_severity)   # 0.45..0.75
    decay_7d = 0.70 - (0.40 * risk_severity)   # 0.30..0.70
    decay_30d = 0.65 - (0.35 * risk_severity)  # 0.30..0.65

    model_survival_48h = _clip_probability(survival_24h * decay_48)
    model_survival_7d = _clip_probability(model_survival_48h * decay_7d)
    model_survival_30d = _clip_probability(model_survival_7d * decay_30d)

    if not np.isnan(historical_48):
        survival_48h = _clip_probability(min(model_survival_48h, historical_48))
    else:
        survival_48h = model_survival_48h

    if not np.isnan(historical_72):
        survival_7d = _clip_probability(min(model_survival_7d, historical_72))
    else:
        survival_7d = model_survival_7d

    if not np.isnan(historical_hospital):
        survival_30d = _clip_probability(min(model_survival_30d, historical_hospital * decay_30d))
    else:
        survival_30d = model_survival_30d

    # Enforce monotonicity by horizon: 4h >= 24h >= 48h >= 7d >= 30d.
    survival_24h = min(survival_24h, survival_4h)
    survival_48h = min(survival_48h, survival_24h)
    survival_7d = min(survival_7d, survival_48h)
    survival_30d = min(survival_30d, survival_7d)

    note_parts: list[str] = []
    if n_similar < 10:
        note_parts.append("Longer-term estimates are based on few similar patients; interpret cautiously.")
    if not np.isnan(historical_24):
        note_parts.append(
            f"Model 24h survival {model_survival_24h*100:.1f}% vs similar-cohort 24h survival {historical_24*100:.1f}%."
        )

    return {
        "4h": _clip_probability(survival_4h),
        "24h": _clip_probability(survival_24h),
        "48h": _clip_probability(survival_48h),
        "7d": _clip_probability(survival_7d),
        "30d": _clip_probability(survival_30d),
        "note": " ".join(note_parts).strip(),
    }


def _risk_progression_table(risk_projection: dict[str, float]) -> pd.DataFrame:
    now = risk_projection.get("now", 0.0)
    return pd.DataFrame(
        [
            {"time_horizon": "Now", "mortality_risk": now, "delta_vs_now": 0.0, "interpretation": "Current model-estimated risk"},
            {"time_horizon": "4 hours", "mortality_risk": risk_projection.get("4h", now), "delta_vs_now": risk_projection.get("4h", now) - now, "interpretation": "Immediate near-term trajectory"},
            {"time_horizon": "8 hours", "mortality_risk": risk_projection.get("8h", now), "delta_vs_now": risk_projection.get("8h", now) - now, "interpretation": "Short-horizon risk evolution"},
            {"time_horizon": "12 hours", "mortality_risk": risk_projection.get("12h", now), "delta_vs_now": risk_projection.get("12h", now) - now, "interpretation": "Extended acute window estimate"},
        ]
    )


def _survival_timeline_table(survival_timeline: dict[str, float]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"time_horizon": "Next 4 hours", "survival_chance": survival_timeline.get("4h", 0.0), "suggested_action": "Immediate intervention and continuous reassessment"},
            {"time_horizon": "Next 24 hours", "survival_chance": survival_timeline.get("24h", 0.0), "suggested_action": "Aggressive care and rapid response-level monitoring"},
            {"time_horizon": "Next 7 days", "survival_chance": survival_timeline.get("7d", 0.0), "suggested_action": "Frequent goals-of-care and prognosis review"},
            {"time_horizon": "Next 30 days", "survival_chance": survival_timeline.get("30d", 0.0), "suggested_action": "Long-range prognostic planning and follow-up"},
        ]
    )


def _time_bound_actions(current_state: str, mortality_risk: float) -> dict[str, list[str]]:
    if mortality_risk >= 0.85 or current_state == "SHOCK":
        return {
            "within_30m": [
                "Start vasopressor support if MAP < 65 mmHg.",
                "Obtain arterial or central access for high-frequency monitoring.",
                "Order STAT lactate, ABG, and perfusion-focused labs.",
            ],
            "within_1h": [
                "Target MAP > 65 mmHg with protocolized hemodynamic support.",
                "Review fluid responsiveness and begin resuscitation when indicated.",
                "Re-check source control and antimicrobial coverage plans.",
            ],
            "within_2h": [
                "Repeat lactate and confirm downward trend goal.",
                "Escalate to ICU senior physician for immediate review.",
                "Prepare prognosis communication with family/caregiver team.",
            ],
        }
    if mortality_risk >= 0.65 or current_state in {"EARLY_DETERIORATION", ACTIVE_DETERIORATION}:
        return {
            "within_30m": [
                "Increase monitoring interval and validate all vital measurements.",
                "Repeat lactate and blood gas if trend is worsening.",
            ],
            "within_1h": [
                "Review oxygenation, perfusion, and hemodynamic support needs.",
                "Notify senior resident or duty intensivist for escalation planning.",
            ],
            "within_2h": [
                "Reassess trajectory against the projected risk timeline.",
                "Escalate to shock pathway if BP drops or lactate continues rising.",
            ],
        }
    return {
        "within_30m": [
            "Continue protocol care with trend-based monitoring.",
        ],
        "within_1h": [
            "Reassess for new deterioration markers.",
        ],
        "within_2h": [
            "Maintain monitoring cadence and update risk view after next window.",
        ],
    }


def render_page() -> None:
    st.markdown('<div class="section-header">Digital Twin Mortality Engine</div>', unsafe_allow_html=True)
    st.write("Upload a 4-hour window sequence for one patient. The trained mortality model scores each window, then the digital twin layer derives interpretable states, transitions, and similar historical outcomes.")

    if not ARTIFACTS_DIR.exists():
        st.error("Model artifacts folder is missing. Train the models first.")
        return

    predictor = _load_predictor()
    model_info = predictor.get_model_info()
    mortality = model_info.get("mortality_model")
    if not mortality:
        st.error("Mortality model artifacts are not available.")
        return

    top_cols = st.columns(3)
    with top_cols[0]:
        st.markdown(metric_card(f"{mortality.get('auroc', 0):.3f}", "Mortality AUROC", "Trained model performance", "cyan"), unsafe_allow_html=True)
    with top_cols[1]:
        st.markdown(metric_card(str(len(RECOMMENDED_MORTALITY_COLUMNS)), "Recommended Inputs", "Columns for uploaded trajectory CSV", "amber"), unsafe_allow_html=True)
    with top_cols[2]:
        st.markdown(metric_card("6", "Preferred Windows", "24h trajectory using 4h windows", "violet"), unsafe_allow_html=True)

    with st.expander("Input Guidance", expanded=True):
        st.write("Use one row per 4-hour window for the same patient. A 6-row CSV covering `window_id 0..5` gives the best digital-twin story.")
        st.dataframe(pd.DataFrame({"recommended_input_columns": RECOMMENDED_MORTALITY_COLUMNS}), use_container_width=True, hide_index=True)
        template_df = _sample_template()
        st.download_button("Download Digital Twin CSV Template", template_df.to_csv(index=False).encode("utf-8"), file_name="mortality_digital_twin_template.csv", mime="text/csv")

    threshold_type = st.selectbox("Threshold Type", ["max_f1", "balanced", "max_sensitivity_90", "max_specificity_90", "default"], index=0)
    uploaded = st.file_uploader("Upload Patient Trajectory CSV", type=["csv"])
    if uploaded is None:
        return

    try:
        input_df = pd.read_csv(uploaded)
    except Exception as exc:
        st.error(f"Could not read CSV: {exc}")
        return

    if input_df.empty:
        st.warning("Uploaded CSV is empty.")
        return
    if "window_id" not in input_df.columns:
        st.error("The digital twin CSV must include `window_id`.")
        return

    input_df["window_id"] = pd.to_numeric(input_df["window_id"], errors="coerce")
    input_df = input_df.dropna(subset=["window_id"]).sort_values("window_id").reset_index(drop=True)
    st.write("Uploaded trajectory preview")
    st.dataframe(input_df, use_container_width=True)

    if not st.button("Build Digital Twin", use_container_width=True):
        return

    with st.spinner("Scoring windows and building patient state trajectory..."):
        scored_df = _predict_dataframe(input_df, threshold_type=threshold_type)
        twin_df = _add_states(scored_df)
        similar_df = _find_similar_patients(twin_df, top_k=SIMILAR_PATIENT_K)
        summary = _transition_summary(twin_df, similar_df=similar_df)

    st.markdown('<div class="section-header">Twin Summary</div>', unsafe_allow_html=True)
    recent_risk = float(twin_df["mortality_risk_score"].iloc[-1]) if "mortality_risk_score" in twin_df.columns else 0.0
    risk_projection = _estimate_risk_progression(twin_df, summary.get("transition_probabilities", {}))
    critical_window = _estimate_critical_window(risk_projection)
    survival_timeline = _estimate_survival_timeline(recent_risk, risk_projection, similar_df)
    survival_24h = survival_timeline.get("24h", max(0.0, 1.0 - recent_risk))

    s1, s2, s3 = st.columns(3)
    with s1:
        st.markdown(metric_card(f"{survival_24h*100:.1f}%", "24h Survival Probability", "Based on model risk and similar patients", "cyan"), unsafe_allow_html=True)
    with s2:
        st.markdown(metric_card(f"{recent_risk*100:.1f}%", "24h Death Risk", "Mortality risk from latest 4-hour window", "red"), unsafe_allow_html=True)
    with s3:
        st.markdown(metric_card(critical_window, "Critical Window Estimate", "Expected time to severe instability", "amber"), unsafe_allow_html=True)

    narrative = (
        f"24-hour survival probability is {survival_24h*100:.1f}% (death risk {recent_risk*100:.1f}%). "
        f"Estimated critical deterioration window: {critical_window}. "
        f"{summary.get('severity_note', '')}"
    )
    st.info(narrative)
    st.caption(summary.get("confidence_note", "Confidence: unknown"))

    st.markdown("**Predicted Risk Progression (next 12 hours)**")
    risk_df = _risk_progression_table(risk_projection)
    baseline_pct = round(float(risk_projection.get("now", 0.0)) * 100.0, 1)
    risk_df["mortality_risk_pct"] = risk_df["mortality_risk"].map(lambda x: round(float(x) * 100.0, 1))
    risk_df["delta_vs_now_pct"] = risk_df["mortality_risk_pct"] - baseline_pct
    risk_df["mortality_risk"] = risk_df["mortality_risk_pct"].map(lambda x: f"{x:.1f}%")
    risk_df["delta_vs_now"] = risk_df["delta_vs_now_pct"].map(lambda x: f"{x:+.1f}%")
    risk_df = risk_df.drop(columns=["mortality_risk_pct", "delta_vs_now_pct"])
    st.dataframe(risk_df, use_container_width=True, hide_index=True)

    st.markdown("**Survival Timeline Estimate**")
    survival_df = _survival_timeline_table(survival_timeline)
    survival_df["survival_chance"] = survival_df["survival_chance"].map(lambda x: f"{x*100:.1f}%")
    st.dataframe(survival_df, use_container_width=True, hide_index=True)
    if survival_timeline.get("note"):
        st.caption(survival_timeline["note"])

    trends = {
        "map_trend": float(pd.to_numeric(twin_df.get("map_mean", twin_df.get("sbp_mean", pd.Series(dtype=float))), errors="coerce").tail(3).diff().mean() or 0.0),
        "lactate_trend": float(pd.to_numeric(twin_df.get("lactate_value", pd.Series(dtype=float)), errors="coerce").tail(3).diff().mean() or 0.0),
    }
    recs = get_clinical_recommendations(summary.get("current_state", "EARLY_DETERIORATION"), recent_risk, trends)
    st.markdown("**Time-Critical Actions (next 2 hours)**")
    actions = _time_bound_actions(summary.get("current_state", "EARLY_DETERIORATION"), recent_risk)
    st.write("Within 30 minutes")
    for item in actions.get("within_30m", []):
        st.write(f"- {item}")
    st.write("Within 1 hour")
    for item in actions.get("within_1h", []):
        st.write(f"- {item}")
    st.write("Within 2 hours")
    for item in actions.get("within_2h", []):
        st.write(f"- {item}")

    st.markdown("**Clinical Recommendations (state context)**")
    for item in recs.get("immediate_actions", []):
        st.write(f"- {item}")
    st.write(f"Monitoring: {recs.get('monitoring', 'N/A')}")
    st.write(f"Escalation: {recs.get('escalation_criteria', 'N/A')}")
    st.caption(recs.get("risk_context", ""))

    chart_cols = st.columns(2)
    with chart_cols[0]:
        st.plotly_chart(_state_trajectory_chart(twin_df), use_container_width=True)
    with chart_cols[1]:
        st.plotly_chart(_risk_trajectory_chart(twin_df), use_container_width=True)

    st.plotly_chart(_vitals_trajectory_chart(twin_df), use_container_width=True)

    st.markdown('<div class="section-header">State Transition Context</div>', unsafe_allow_html=True)
    transition_df = pd.DataFrame(
        [{"next_state": key, "probability": value} for key, value in summary.get("transition_probabilities", {}).items()]
    ).sort_values("probability", ascending=False)
    st.dataframe(transition_df, use_container_width=True, hide_index=True)

    st.markdown('<div class="section-header">Similar Patient Outcomes</div>', unsafe_allow_html=True)
    if similar_df.empty:
        st.warning("No similar historical patients found.")
    else:
        recovered = int((similar_df["mortality"] == 0).sum()) if "mortality" in similar_df.columns else 0
        progressed = int((similar_df["mortality"] == 1).sum()) if "mortality" in similar_df.columns else 0
        mortality_24h = float(similar_df["mortality_24h"].mean()) if "mortality_24h" in similar_df.columns else np.nan
        mortality_48h = float(similar_df["mortality_48h"].mean()) if "mortality_48h" in similar_df.columns else np.nan
        event_hours = pd.to_numeric(similar_df.get("first_critical_event_hour", pd.Series(dtype=float)), errors="coerce").dropna()
        median_event_hour = float(event_hours.median()) if not event_hours.empty else np.nan
        summary_line = (
            f"Matched {len(similar_df)} similar patients: {recovered} survived, {progressed} died."
            f" 24h mortality: {mortality_24h*100:.1f}%."
        )
        if not np.isnan(mortality_48h):
            summary_line += f" 48h mortality: {mortality_48h*100:.1f}%."
        if not np.isnan(median_event_hour):
            summary_line += f" Median first critical event around hour {median_event_hour:.0f}."
        st.write(summary_line)
        st.dataframe(similar_df, use_container_width=True, hide_index=True)

    st.markdown('<div class="section-header">Window-Level Twin Table</div>', unsafe_allow_html=True)
    st.dataframe(twin_df, use_container_width=True)
    st.download_button(
        "Download Twin Analysis CSV",
        twin_df.to_csv(index=False).encode("utf-8"),
        file_name="digital_twin_analysis.csv",
        mime="text/csv",
    )
