from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from frontend.api_client import fetch_analytics_age, fetch_analytics_diagnoses, fetch_analytics_fluid, fetch_analytics_units, fetch_analytics_vasopressor, fetch_analytics_ventilator, fetch_patients
from frontend.components.charts import apache_distribution_chart, mortality_age_chart, top_diagnoses_chart, unit_breakdown_chart, vasopressor_ventilator_chart


def render_page() -> None:
    st.markdown('<div class="section-header">Population Analytics</div>', unsafe_allow_html=True)
    units_df = pd.DataFrame(fetch_analytics_units() or [])
    age_df = pd.DataFrame(fetch_analytics_age() or [])
    fluid_df = pd.DataFrame(fetch_analytics_fluid() or [])
    vaso_df = pd.DataFrame(fetch_analytics_vasopressor() or [])
    vent_df = pd.DataFrame(fetch_analytics_ventilator() or [])
    diag_df = pd.DataFrame(fetch_analytics_diagnoses() or [])
    patients_df = pd.DataFrame((fetch_patients(limit=500) or {}).get("patients", []))

    if not units_df.empty:
        st.plotly_chart(unit_breakdown_chart(units_df), use_container_width=True)

    row1a, row1b = st.columns(2)
    with row1a:
        if not age_df.empty:
            st.plotly_chart(mortality_age_chart(age_df), use_container_width=True)
    with row1b:
        if not patients_df.empty and "gender" in patients_df.columns:
            gender_df = patients_df.groupby("gender", dropna=False)["hospital_mortality"].agg(["count", "mean"]).reset_index()
            fig = px.pie(gender_df, names="gender", values="count", color="gender", color_discrete_sequence=["#00C8FF", "#A78BFA", "#5A7A99"])
            st.plotly_chart(fig, use_container_width=True)

    row2a, row2b = st.columns(2)
    with row2a:
        if not patients_df.empty and {"apachescore", "hospital_mortality"}.issubset(patients_df.columns):
            survived = patients_df.loc[patients_df["hospital_mortality"] == 0, "apachescore"]
            expired = patients_df.loc[patients_df["hospital_mortality"] == 1, "apachescore"]
            st.plotly_chart(apache_distribution_chart(survived, expired), use_container_width=True)
    with row2b:
        if not fluid_df.empty:
            fig = px.bar(fluid_df, x="hospital_mortality", y="avg_fluid_balance", color="hospital_mortality", color_discrete_sequence=["#00E5A0", "#FF4560"])
            st.plotly_chart(fig, use_container_width=True)

    if not vaso_df.empty and not vent_df.empty:
        st.plotly_chart(vasopressor_ventilator_chart(vaso_df, vent_df), use_container_width=True)

    if not diag_df.empty:
        st.plotly_chart(top_diagnoses_chart(diag_df), use_container_width=True)

    with st.expander("Lab Value Analysis"):
        st.info("Detailed lab distribution by outcome is not exposed by the current backend API yet.")

