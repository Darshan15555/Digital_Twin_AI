from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

BASE_LAYOUT = {
    "paper_bgcolor": "rgba(0,0,0,0)",
    "plot_bgcolor": "rgba(15,23,36,0.9)",
    "font": {"family": "JetBrains Mono", "color": "#5A7A99", "size": 11},
    "margin": {"l": 50, "r": 20, "t": 45, "b": 40},
    "legend": {"bgcolor": "rgba(0,0,0,0)", "bordercolor": "rgba(255,255,255,0.05)"},
    "hoverlabel": {"bgcolor": "#16213A", "bordercolor": "#00C8FF", "font_color": "#E8F4FD", "font_family": "JetBrains Mono"},
}


def _finish(fig: go.Figure, height: int) -> go.Figure:
    fig.update_layout(height=height, **BASE_LAYOUT)
    fig.update_xaxes(gridcolor="rgba(0,200,255,0.06)", showline=False, zeroline=False)
    fig.update_yaxes(gridcolor="rgba(0,200,255,0.06)", showline=False, zeroline=False)
    return fig


def vitals_multiplot(df_vitals: pd.DataFrame) -> go.Figure:
    fig = make_subplots(rows=3, cols=2, subplot_titles=["Heart Rate", "SpO2", "Respiration", "Systolic BP", "Temperature", "CVP"])
    mapping = [
        ("heartrate", "#FF4560", (40, 150), 1, 1),
        ("sao2", "#00E5A0", (90, None), 1, 2),
        ("respiration", "#00C8FF", (8, 30), 2, 1),
        ("systemicsystolic", "#A78BFA", (80, 200), 2, 2),
        ("temperature", "#FFB830", (35.5, 38.5), 3, 1),
        ("cvp", "#60A5FA", (None, None), 3, 2),
    ]
    x = df_vitals.get("offset_hours", df_vitals.get("observationoffset", pd.Series(dtype=float)) / 60.0)
    for column, color, thresholds, row, col in mapping:
        if column not in df_vitals.columns:
            continue
        fig.add_trace(go.Scatter(x=x, y=df_vitals[column], mode="lines", line={"color": color, "width": 1.5}, showlegend=False), row=row, col=col)
        low, high = thresholds
        if low is not None:
            fig.add_hline(y=low, line_dash="dash", line_color="#FF4560", opacity=0.35, row=row, col=col)
        if high is not None:
            fig.add_hline(y=high, line_dash="dash", line_color="#FF4560", opacity=0.35, row=row, col=col)
    fig.update_xaxes(title_text="Hours from ICU Admission")
    return _finish(fig, 680)


def labs_chart(df_labs: pd.DataFrame, selected_labs: list[str]) -> go.Figure:
    plot_df = df_labs[df_labs["labname"].isin(selected_labs)].copy()
    plot_df["offset_hours"] = plot_df.get("offset_hours", plot_df["labresultoffset"] / 60.0)
    fig = px.line(plot_df, x="offset_hours", y="labresult", color="labname", markers=True)
    return _finish(fig, 380)


def risk_gauge(risk_score: float) -> go.Figure:
    score_pct = float(risk_score) * 100
    fig = go.Figure(go.Indicator(mode="gauge+number", value=score_pct, number={"suffix": "%"}, title={"text": "Mortality Risk"}, gauge={"axis": {"range": [0, 100]}, "bar": {"color": "#FF4560" if score_pct >= 65 else "#FFB830" if score_pct >= 35 else "#00E5A0"}, "steps": [{"range": [0, 35], "color": "rgba(0,229,160,0.18)"}, {"range": [35, 65], "color": "rgba(255,184,48,0.18)"}, {"range": [65, 100], "color": "rgba(255,69,96,0.18)"}], "threshold": {"line": {"color": "#E8F4FD", "width": 3}, "value": score_pct}}))
    return _finish(fig, 260)


def los_gauge(los_hours: float) -> go.Figure:
    fig = go.Figure(go.Indicator(mode="gauge+number", value=float(los_hours), number={"suffix": " h"}, title={"text": f"Length of Stay ({los_hours / 24.0:.1f} days)"}, gauge={"axis": {"range": [0, 336]}, "bar": {"color": "#A78BFA"}, "steps": [{"range": [0, 72], "color": "rgba(0,200,255,0.12)"}, {"range": [72, 168], "color": "rgba(167,139,250,0.18)"}, {"range": [168, 336], "color": "rgba(255,184,48,0.18)"}]}))
    return _finish(fig, 260)


def unit_breakdown_chart(data: pd.DataFrame) -> go.Figure:
    df = data.copy()
    df["unit_short"] = df["unittype"].astype(str).str.slice(0, 15)
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_bar(x=df["unit_short"], y=df["patient_count"], marker_color="#00C8FF", name="Patients")
    fig.add_scatter(x=df["unit_short"], y=df["mortality_pct"], marker_color="#FF4560", mode="lines+markers", name="Mortality %", secondary_y=True)
    return _finish(fig, 340)


def mortality_age_chart(data: pd.DataFrame) -> go.Figure:
    fig = px.bar(data, x="mortality_pct", y="age_group", orientation="h", color="mortality_pct", color_continuous_scale=["#00E5A0", "#FF4560"], text="count")
    return _finish(fig, 320)


def top_diagnoses_chart(data: pd.DataFrame) -> go.Figure:
    df = data.copy()
    df["diagnosis_short"] = df["diagnosisstring"].astype(str).str.slice(0, 55)
    fig = px.bar(df.sort_values("count"), x="count", y="diagnosis_short", orientation="h", color_discrete_sequence=["#00C8FF"])
    return _finish(fig, 420)


def apache_distribution_chart(survived_scores: pd.Series, expired_scores: pd.Series) -> go.Figure:
    fig = go.Figure()
    fig.add_histogram(x=survived_scores, nbinsx=20, name="Survived", marker_color="rgba(0,200,255,0.5)")
    fig.add_histogram(x=expired_scores, nbinsx=20, name="Expired", marker_color="rgba(255,69,96,0.5)")
    fig.update_layout(barmode="overlay")
    return _finish(fig, 320)


def fluid_balance_chart(hourly_data: pd.DataFrame) -> go.Figure:
    df = hourly_data.copy()
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    x = df.index if "hour" not in df.columns else df["hour"]
    if "intake" in df.columns:
        fig.add_bar(x=x, y=df["intake"], name="Intake", marker_color="#00C8FF")
    if "output" in df.columns:
        fig.add_bar(x=x, y=df["output"], name="Output", marker_color="#FFB830")
    if {"intake", "output"}.issubset(df.columns):
        fig.add_scatter(x=x, y=(df["intake"].fillna(0) - df["output"].fillna(0)).cumsum(), name="Running Balance", line={"color": "#A78BFA"}, secondary_y=True)
    fig.add_hline(y=0, line_color="rgba(255,255,255,0.3)", line_dash="dot", secondary_y=True)
    return _finish(fig, 340)


def vasopressor_ventilator_chart(vaso_data: pd.DataFrame, vent_data: pd.DataFrame) -> go.Figure:
    fig = make_subplots(rows=1, cols=2, subplot_titles=["Vasopressor Usage", "Ventilator Usage"])
    fig.add_bar(x=vaso_data.get("vasopressor_pct", []), y=vaso_data.get("unittype", []), orientation="h", marker_color="#FFB830", row=1, col=1)
    fig.add_bar(x=vent_data.get("ventilator_pct", []), y=vent_data.get("unittype", []), orientation="h", marker_color="#A78BFA", row=1, col=2)
    return _finish(fig, 300)


def digital_twin_chart(df_vitals: pd.DataFrame, forecast: dict, future_offsets: list[float], vital_name: str, color: str) -> go.Figure:
    fig = go.Figure()
    x = df_vitals.get("offset_hours", df_vitals.get("observationoffset", pd.Series(dtype=float)) / 60.0)
    if vital_name in df_vitals.columns:
        fig.add_scatter(x=x, y=df_vitals[vital_name], mode="lines", name="Historical", line={"color": color})
    values = forecast.get(vital_name, {}).get("values", [])
    lower = forecast.get(vital_name, {}).get("lower", [])
    upper = forecast.get(vital_name, {}).get("upper", [])
    if values:
        fig.add_scatter(x=future_offsets, y=values, mode="lines", name="Forecast", line={"color": "#E8F4FD", "dash": "dash"})
    if lower and upper:
        fig.add_scatter(x=future_offsets + future_offsets[::-1], y=upper + lower[::-1], fill="toself", line={"color": "rgba(0,0,0,0)"}, fillcolor="rgba(255,255,255,0.08)", name="Band")
    if len(x) > 0:
        fig.add_vline(x=float(x.iloc[-1]), line_dash="dash", line_color="#E8F4FD")
    return _finish(fig, 240)


def sepsis_radar_chart(criteria: dict) -> go.Figure:
    labels = ["Resp", "SBP", "GCS", "Lactate", "SOFA"]
    values = [criteria.get("resp_rate_score", 0), criteria.get("sbp_score", 0), criteria.get("gcs_score", 0), criteria.get("lactate_score", 0), criteria.get("sofa_score", 0)]
    fig = go.Figure(go.Scatterpolar(r=values, theta=labels, fill="toself", line={"color": "#FF4560"}))
    return _finish(fig, 300)


def feature_importance_chart(features: list) -> go.Figure:
    df = pd.DataFrame(features[:8], columns=["feature", "importance"])
    fig = px.bar(df.sort_values("importance"), x="importance", y="feature", orientation="h", color="importance", color_continuous_scale=["#6D5EF5", "#A78BFA"])
    return _finish(fig, 280)
