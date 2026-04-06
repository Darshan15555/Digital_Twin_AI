from __future__ import annotations

import pandas as pd
import streamlit as st

from frontend.api_client import fetch_analytics_age, fetch_analytics_diagnoses, fetch_analytics_fluid, fetch_analytics_units, fetch_analytics_vasopressor, fetch_analytics_ventilator, fetch_patients, fetch_stats
from frontend.components.cards import metric_card
from frontend.components.charts import fluid_balance_chart, mortality_age_chart, top_diagnoses_chart, unit_breakdown_chart, vasopressor_ventilator_chart


def render_page() -> None:
    st.markdown('<div class="section-header">System Overview</div>', unsafe_allow_html=True)
    with st.spinner("Loading ICU overview..."):
        stats = fetch_stats()
    if not stats:
        st.warning("Overview is unavailable because the backend did not return system stats.")
        return

    mortality_pct = (stats.get("mortality_rate") or 0) * 100
    cards = [
        (f"{int(stats.get('total_patients', 0)):,}", "Total Patients", "Rows available in current app dataset", "cyan"),
        (f"{mortality_pct:.2f}%", "Mortality Rate", "Outcome prevalence", "red" if mortality_pct > 15 else "amber" if mortality_pct > 8 else "green"),
        (f"{stats.get('avg_age') or '—'}", "Avg Age", "May be unavailable in current backend stats", "cyan"),
        (f"{(stats.get('avg_apache_score') or 0):.1f}", "Avg APACHE", "Severity baseline", "red" if (stats.get("avg_apache_score") or 0) > 70 else "amber" if (stats.get("avg_apache_score") or 0) > 50 else "cyan"),
        (f"{int(stats.get('total_on_vasopressor', 0)):,}", "On Vasopressor", "Field may be absent in current API", "amber"),
        (f"{int(stats.get('total_on_ventilator', 0)):,}", "On Ventilator", "Field may be absent in current API", "violet"),
    ]
    cols = st.columns(6)
    for col, payload in zip(cols, cards):
        with col:
            st.markdown(metric_card(*payload), unsafe_allow_html=True)

    units_df = pd.DataFrame(fetch_analytics_units() or [])
    age_df = pd.DataFrame(fetch_analytics_age() or [])
    vaso_df = pd.DataFrame(fetch_analytics_vasopressor() or [])
    vent_df = pd.DataFrame(fetch_analytics_ventilator() or [])
    fluid_df = pd.DataFrame(fetch_analytics_fluid() or [])
    diagnoses_df = pd.DataFrame(fetch_analytics_diagnoses() or [])

    col1, col2 = st.columns([3, 2])
    with col1:
        if not units_df.empty:
            st.plotly_chart(unit_breakdown_chart(units_df), use_container_width=True)
        else:
            st.info("Unit breakdown data is empty.")
    with col2:
        if not age_df.empty:
            st.plotly_chart(mortality_age_chart(age_df), use_container_width=True)
        else:
            st.info("Age mortality data is empty.")

    col3, col4 = st.columns(2)
    with col3:
        if not vaso_df.empty and not vent_df.empty:
            st.plotly_chart(vasopressor_ventilator_chart(vaso_df, vent_df), use_container_width=True)
        else:
            st.info("Intervention usage data is unavailable.")
    with col4:
        if not fluid_df.empty:
            plot_df = fluid_df.rename(columns={"hospital_mortality": "hour", "avg_fluid_balance": "intake", "avg_urine_output_per_hour": "output"})
            st.plotly_chart(fluid_balance_chart(plot_df), use_container_width=True)
        else:
            st.info("Fluid balance analytics are unavailable.")

    if not diagnoses_df.empty:
        st.plotly_chart(top_diagnoses_chart(diagnoses_df), use_container_width=True)

    st.markdown('<div class="section-header">Recent High-Risk Patients</div>', unsafe_allow_html=True)
    payload = fetch_patients(limit=10, high_risk=True) or {}
    patients_df = pd.DataFrame(payload.get("patients", []))
    if patients_df.empty:
        st.info("No high-risk patients available from the current API.")
        return
    display = patients_df[[c for c in ["patientunitstayid", "age", "gender", "unittype", "apachescore", "predicted_mortality_risk", "qsofa_score", "hospital_mortality"] if c in patients_df.columns]].copy()
    st.dataframe(display, use_container_width=True, hide_index=True)
    selected = st.number_input("Open patient from high-risk list", min_value=0, step=1, key="overview_patient_pick")
    if st.button("Go To Patient Detail", key="overview_go") and selected:
        st.session_state["selected_patient"] = int(selected)
        st.session_state["page"] = "👤  Patient Detail"
        st.rerun()

