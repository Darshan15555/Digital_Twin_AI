"""
Streamlit dashboard for the ICU analytics system.
"""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
from plotly.subplots import make_subplots

from config.config import API_HOST, API_PORT

API_BASE = f"http://{API_HOST}:{API_PORT}"

st.set_page_config(page_title="ICU Analytics System", page_icon="ICU", layout="wide")


def api_get(path: str, params: dict | None = None):
    try:
        response = requests.get(f"{API_BASE}{path}", params=params, timeout=20)
        response.raise_for_status()
        return response.json()
    except Exception as exc:
        st.error(f"API request failed: {exc}")
        return None


@st.cache_data(ttl=60)
def get_stats():
    return api_get("/stats")


@st.cache_data(ttl=60)
def get_patients(**params):
    return api_get("/patients", params)


@st.cache_data(ttl=60)
def get_patient(patient_id: int):
    return api_get(f"/patients/{patient_id}")


@st.cache_data(ttl=60)
def get_json(path: str, params: dict | None = None):
    return api_get(path, params)


def page_overview():
    stats = get_stats()
    st.title("ICU Analytics Overview")
    if not stats:
        return

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Patients", f"{int(stats.get('total_patients', 0)):,}")
    c2.metric("Mortality %", f"{(stats.get('mortality_rate') or 0) * 100:.2f}")
    c3.metric("Avg APACHE", f"{stats.get('avg_apache_score') or 0:.1f}")
    c4.metric("Avg LOS (hrs)", f"{stats.get('avg_icu_los_hours') or 0:.1f}")

    unit_df = pd.DataFrame(get_json("/analytics/unit-breakdown") or [])
    age_df = pd.DataFrame(get_json("/analytics/mortality-by-age") or [])
    vaso_df = pd.DataFrame(get_json("/analytics/vasopressor-usage") or [])
    vent_df = pd.DataFrame(get_json("/analytics/ventilator-usage") or [])
    diag_df = pd.DataFrame(get_json("/analytics/top-diagnoses") or [])

    col1, col2 = st.columns(2)
    with col1:
        if not unit_df.empty:
            fig = px.bar(unit_df, x="unittype", y="patient_count", color="mortality_pct", title="Patients by Unit")
            st.plotly_chart(fig, use_container_width=True)
    with col2:
        if not age_df.empty:
            fig = px.bar(age_df, x="age_group", y="mortality_pct", title="Mortality by Age Group")
            st.plotly_chart(fig, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        if not vaso_df.empty:
            st.plotly_chart(px.bar(vaso_df, x="unittype", y="vasopressor_pct", title="Vasopressor Usage %"), use_container_width=True)
    with col4:
        if not vent_df.empty:
            st.plotly_chart(px.bar(vent_df, x="unittype", y="ventilator_pct", title="Ventilator Usage %"), use_container_width=True)

    if not diag_df.empty:
        st.plotly_chart(px.bar(diag_df, x="count", y="diagnosisstring", orientation="h", title="Top Diagnoses"), use_container_width=True)


def page_patient_search():
    st.title("Patient Search")
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    search = col1.text_input("Search")
    unit = col2.text_input("Unit Type")
    age_min = col3.number_input("Age Min", min_value=0, value=0)
    age_max = col4.number_input("Age Max", min_value=0, value=100)
    high_risk = col5.checkbox("High Risk Only")
    on_vent = col6.checkbox("On Ventilator")
    on_vaso = st.checkbox("On Vasopressor")

    payload = get_patients(
        limit=200,
        search=search or None,
        unit=unit or None,
        age_min=int(age_min),
        age_max=int(age_max),
        high_risk_only=high_risk,
        on_vasopressor=on_vaso,
        on_ventilator=on_vent,
    )
    df = pd.DataFrame((payload or {}).get("patients", []))
    if df.empty:
        st.info("No matching patients found.")
        return

    display_cols = [
        "patientunitstayid",
        "age",
        "gender",
        "unittype",
        "apachescore",
        "predicted_mortality_risk",
        "qsofa_score",
        "hospital_mortality",
        "on_vasopressor",
        "on_ventilator",
    ]
    existing = [col for col in display_cols if col in df.columns]
    st.dataframe(df[existing], use_container_width=True, height=500)

    patient_id = st.number_input("Patient ID", min_value=0, step=1)
    if st.button("Open Patient"):
        st.session_state["patient_id"] = int(patient_id)


def _plot_vitals(periodic: pd.DataFrame):
    fig = make_subplots(rows=3, cols=2, subplot_titles=["HR", "SpO2", "SBP", "Temp", "RR", "CVP"])
    specs = [
        ("heartrate", 1, 1),
        ("sao2", 1, 2),
        ("systemicsystolic", 2, 1),
        ("temperature", 2, 2),
        ("respiration", 3, 1),
        ("cvp", 3, 2),
    ]
    x = periodic["offset_hours"] if "offset_hours" in periodic.columns else periodic["observationoffset"] / 60.0
    for column, row, col in specs:
        if column in periodic.columns:
            fig.add_trace(go.Scatter(x=x, y=periodic[column], mode="lines", name=column), row=row, col=col)
    fig.update_layout(height=700, title="Vital Timeline")
    st.plotly_chart(fig, use_container_width=True)


def page_patient_detail():
    st.title("Patient Detail")
    patient_id = int(st.session_state.get("patient_id", 0) or st.number_input("Patient ID", min_value=0, step=1))
    if not patient_id:
        st.info("Select a patient first.")
        return

    patient = get_patient(patient_id)
    if not patient:
        return
    st.subheader(f"Patient {patient_id}")
    st.json({k: patient[k] for k in list(patient.keys())[:15]})

    vitals = get_json(f"/patients/{patient_id}/vitals") or {}
    labs = pd.DataFrame((get_json(f"/patients/{patient_id}/labs") or {}).get("labs", []))
    diagnosis = pd.DataFrame((get_json(f"/patients/{patient_id}/diagnosis") or {}).get("diagnosis", []))
    treatments = pd.DataFrame((get_json(f"/patients/{patient_id}/treatments") or {}).get("treatments", []))
    meds = get_json(f"/patients/{patient_id}/medications") or {}
    fluid = get_json(f"/patients/{patient_id}/fluid-balance") or {}
    comorb = get_json(f"/patients/{patient_id}/comorbidities") or {}
    vent = pd.DataFrame((get_json(f"/patients/{patient_id}/ventilation") or {}).get("ventilation", []))

    tabs = st.tabs(["Vitals", "Labs", "Diagnosis", "Treatments", "Medications", "Fluid Balance", "Comorbidities", "Ventilation"])

    with tabs[0]:
        periodic = pd.DataFrame(vitals.get("periodic", []))
        if not periodic.empty:
            _plot_vitals(periodic)
    with tabs[1]:
        if not labs.empty:
            lab_options = sorted(labs["labname"].dropna().unique().tolist())
            selected = st.multiselect("Lab Tests", lab_options, default=lab_options[:4])
            plot_df = labs[labs["labname"].isin(selected)].copy()
            plot_df["offset_hours"] = plot_df["offset_hours"].fillna(plot_df["labresultoffset"] / 60.0)
            st.plotly_chart(px.line(plot_df, x="offset_hours", y="labresult", color="labname"), use_container_width=True)
            st.dataframe(labs, use_container_width=True)
    with tabs[2]:
        st.dataframe(diagnosis, use_container_width=True)
    with tabs[3]:
        st.dataframe(treatments, use_container_width=True)
    with tabs[4]:
        st.write("Regular Medications")
        st.dataframe(pd.DataFrame(meds.get("medications", [])), use_container_width=True)
        st.write("Infusions")
        st.dataframe(pd.DataFrame(meds.get("infusions", [])), use_container_width=True)
    with tabs[5]:
        st.dataframe(pd.DataFrame(fluid.get("summary", [])), use_container_width=True)
    with tabs[6]:
        st.dataframe(pd.DataFrame(comorb.get("comorbidities", [])), use_container_width=True)
    with tabs[7]:
        st.dataframe(vent, use_container_width=True)


def page_predictions():
    st.title("Predictions Panel")
    patient_id = int(st.session_state.get("patient_id", 0) or st.number_input("Patient ID", min_value=0, step=1))
    if not patient_id:
        st.info("Select a patient first.")
        return

    mortality = get_json(f"/patients/{patient_id}/predict/mortality")
    los = get_json(f"/patients/{patient_id}/predict/los")
    sepsis = get_json(f"/patients/{patient_id}/predict/sepsis")
    patient = get_patient(patient_id)

    col1, col2, col3, col4 = st.columns(4)
    if mortality:
        col1.metric("Mortality Risk", f"{mortality['risk_score'] * 100:.1f}%")
    if los:
        col2.metric("Predicted LOS", f"{los['predicted_icu_los_hours']:.1f} hrs")
    if sepsis:
        col3.metric("qSOFA", sepsis["qsofa_score"])
        col4.metric("Sepsis Risk", "High" if sepsis["sepsis_risk"] else "Low")

    if mortality:
        gauge = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=mortality["risk_score"] * 100,
                title={"text": "Mortality Risk"},
                gauge={"axis": {"range": [0, 100]}},
            )
        )
        st.plotly_chart(gauge, use_container_width=True)
        feat_df = pd.DataFrame(mortality["top_features"], columns=["feature", "importance"])
        st.plotly_chart(px.bar(feat_df, x="importance", y="feature", orientation="h"), use_container_width=True)
    if patient:
        st.json(
            {
                "deterioration_status": patient.get("predicted_mortality_label", "N/A"),
                "qsofa_score": patient.get("qsofa_score"),
                "sepsis_risk": patient.get("sepsis_risk"),
            }
        )


def page_analytics():
    st.title("Population Analytics")
    unit_df = pd.DataFrame(get_json("/analytics/unit-breakdown") or [])
    fluid_df = pd.DataFrame(get_json("/analytics/fluid-balance") or [])
    patients_df = pd.DataFrame((get_patients(limit=500) or {}).get("patients", []))

    if not unit_df.empty:
        st.plotly_chart(px.bar(unit_df, x="unittype", y="mortality_pct", color="avg_apache", title="Mortality by Unit"), use_container_width=True)

    if not patients_df.empty:
        if {"apachescore", "hospital_mortality"}.issubset(patients_df.columns):
            st.plotly_chart(px.box(patients_df, x="hospital_mortality", y="apachescore", title="APACHE by Outcome"), use_container_width=True)
        if {"fluid_balance_24h", "hospital_mortality"}.issubset(patients_df.columns):
            st.plotly_chart(
                px.scatter(patients_df, x="fluid_balance_24h", y="predictedhospitalmortality", color="hospital_mortality", title="Fluid Balance vs Mortality"),
                use_container_width=True,
            )
    if not fluid_df.empty:
        st.dataframe(fluid_df, use_container_width=True)


def page_digital_twin():
    st.title("Digital Twin")
    patient_id = int(st.session_state.get("patient_id", 0) or st.number_input("Patient ID", min_value=0, step=1))
    steps = st.slider("Forecast Steps", min_value=6, max_value=12, value=6)
    fio2_delta = st.slider("What-if FiO2 Change", min_value=-0.2, max_value=0.2, value=0.0, step=0.05)
    vaso_delta = st.slider("What-if Vasopressor Dose Change", min_value=-1.0, max_value=1.0, value=0.0, step=0.1)

    if not patient_id:
        st.info("Select a patient first.")
        return

    twin = get_json(
        f"/patients/{patient_id}/digital-twin",
        {"steps": steps, "fio2_delta": fio2_delta, "vasopressor_delta": vaso_delta},
    )
    vitals = get_json(f"/patients/{patient_id}/vitals") or {}
    periodic = pd.DataFrame(vitals.get("periodic", []))
    if twin is None or periodic.empty:
        return

    st.metric("Deterioration", twin["deterioration"]["label"])

    recent = periodic.tail(100).copy()
    recent["offset_hours"] = recent["offset_hours"].fillna(recent["observationoffset"] / 60.0)
    for column in ["heartrate", "sao2", "systemicsystolic", "temperature"]:
        if column not in recent.columns or column not in twin["forecast"]:
            continue
        base = twin["forecast"][column]
        sim = twin["simulated_forecast"][column]
        future_x = list(range(len(recent), len(recent) + len(base["values"])))
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=recent["offset_hours"], y=recent[column], mode="lines", name="Observed"))
        fig.add_trace(go.Scatter(x=future_x, y=base["values"], mode="lines+markers", name="Forecast"))
        fig.add_trace(go.Scatter(x=future_x, y=sim["values"], mode="lines+markers", name="What-if"))
        if base["lower"]:
            fig.add_trace(go.Scatter(x=future_x + future_x[::-1], y=base["upper"] + base["lower"][::-1], fill="toself", name="CI", line={"color": "rgba(0,0,0,0)"}))
        st.plotly_chart(fig, use_container_width=True)


PAGES = {
    "Overview": page_overview,
    "Patient Search": page_patient_search,
    "Patient Detail": page_patient_detail,
    "Predictions Panel": page_predictions,
    "Analytics": page_analytics,
    "Digital Twin": page_digital_twin,
}

with st.sidebar:
    st.header("ICU Analytics")
    page = st.radio("Page", list(PAGES.keys()))
    if st.button("Clear Cache"):
        st.cache_data.clear()
        st.rerun()

PAGES[page]()
