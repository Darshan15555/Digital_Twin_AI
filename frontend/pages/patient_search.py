from __future__ import annotations

import math

import pandas as pd
import streamlit as st

from frontend.api_client import fetch_patients


def render_page() -> None:
    st.markdown('<div class="section-header">Patient Search</div>', unsafe_allow_html=True)
    page_num = st.session_state.get("patient_search_page", 1)
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    search = col1.text_input("Search", value="")
    unit = col2.text_input("Unit Type", value="")
    age_range = col3.slider("Age Range", 0, 100, (0, 100))
    high_risk = col4.toggle("High Risk", value=False)
    on_vaso = col5.toggle("Vasopressor", value=False)
    on_vent = col6.toggle("Ventilator", value=False)
    page_size = st.selectbox("Results per page", [50, 100, 200], index=1)

    offset = (page_num - 1) * page_size
    payload = fetch_patients(limit=page_size, offset=offset, search=search or None, unit_type=unit or None, high_risk=high_risk, on_vasopressor=on_vaso if on_vaso else None, on_ventilator=on_vent if on_vent else None, age_min=age_range[0], age_max=age_range[1]) or {}
    df = pd.DataFrame(payload.get("patients", []))
    total_count = payload.get("count", len(df))
    total_pages = max(1, math.ceil(max(total_count, 1) / page_size))

    st.caption(f"Showing {len(df)} of {total_count} patients")
    if df.empty:
        st.info("No patients match the current filters.")
    else:
        if "hospital_mortality" in df.columns:
            df["outcome_label"] = df["hospital_mortality"].map({1: "Expired", 0: "Survived"}).fillna("Unknown")
        if "on_vasopressor" in df.columns:
            df["vaso_label"] = df["on_vasopressor"].map({1: "Yes", 0: "No"}).fillna("No")
        if "on_ventilator" in df.columns:
            df["vent_label"] = df["on_ventilator"].map({1: "Yes", 0: "No"}).fillna("No")
        cols = [c for c in ["patientunitstayid", "age", "gender", "unittype", "apachescore", "icu_los_hours", "predicted_mortality_risk", "qsofa_score", "vaso_label", "vent_label", "sepsis_risk", "outcome_label"] if c in df.columns]
        st.dataframe(df[cols], use_container_width=True, hide_index=True)

    prev_col, page_col, next_col = st.columns([1, 2, 1])
    if prev_col.button("Prev", disabled=page_num <= 1):
        st.session_state["patient_search_page"] = page_num - 1
        st.rerun()
    page_col.markdown(f"<div class='table-caption'>Page {page_num} of {total_pages}</div>", unsafe_allow_html=True)
    if next_col.button("Next", disabled=page_num >= total_pages):
        st.session_state["patient_search_page"] = page_num + 1
        st.rerun()

    patient_id = st.number_input("Patient quick-select", min_value=0, step=1, key="patient_search_pick")
    if st.button("View Patient →") and patient_id:
        st.session_state["selected_patient"] = int(patient_id)
        st.session_state["page"] = "👤  Patient Detail"
        st.rerun()
