from __future__ import annotations

import pandas as pd
import streamlit as st

from frontend.api_client import fetch_comorbidities, fetch_diagnosis, fetch_fluid_balance, fetch_labs, fetch_medications, fetch_patient, fetch_predict_early_mortality, fetch_predict_los, fetch_predict_sepsis, fetch_treatments, fetch_ventilation, fetch_vitals
from frontend.components.cards import alert_banner, comorbidity_tags, patient_header_card, risk_badge, vital_status_dot
from frontend.components.charts import ensure_offset_hours, feature_importance_chart, fluid_balance_chart, labs_chart, los_gauge, risk_gauge, sepsis_radar_chart, vitals_multiplot


def render_page() -> None:
    st.markdown('<div class="section-header">Patient Detail</div>', unsafe_allow_html=True)
    default_pid = int(st.session_state.get("selected_patient", 0) or 0)
    patient_id = int(st.number_input("Patient ID", min_value=0, value=default_pid, step=1))
    if st.button("Load Patient") and patient_id:
        st.session_state["selected_patient"] = patient_id
    patient_id = int(st.session_state.get("selected_patient", patient_id) or 0)
    if not patient_id:
        st.info("Select a patient to inspect.")
        return

    patient = fetch_patient(patient_id)
    if not patient:
        st.warning("Patient record is unavailable from the current backend.")
        return

    st.markdown(patient_header_card(patient), unsafe_allow_html=True)
    early = fetch_predict_early_mortality(patient_id) or {}
    if (patient.get("qsofa_score") or 0) >= 2:
        st.markdown(alert_banner("Sepsis Risk: qSOFA >= 2", "critical"), unsafe_allow_html=True)
    if early.get("risk_score", 0) >= 0.7:
        st.markdown(alert_banner("High early mortality risk", "critical"), unsafe_allow_html=True)
    if patient.get("on_vasopressor"):
        st.markdown(alert_banner("Patient on vasopressor support", "warning"), unsafe_allow_html=True)

    vitals_payload = fetch_vitals(patient_id) or {}
    periodic = pd.DataFrame(vitals_payload.get("periodic", []))
    if not periodic.empty:
        periodic = ensure_offset_hours(periodic, "observationoffset")
    specs = [
        ("heartrate", "Heart Rate", (40, 55, 110, 150)),
        ("sao2", "SpO2", (90, 92, 100, 100)),
        ("respiration", "Resp", (8, 12, 24, 30)),
        ("systemicsystolic", "SBP", (80, 90, 160, 200)),
        ("temperature", "Temp", (35.5, 36.0, 38.0, 38.5)),
        ("cvp", "CVP", (0, 2, 12, 20)),
    ]
    for col, (field, label, thresholds) in zip(st.columns(6), specs):
        with col:
            value = None
            if not periodic.empty and field in periodic.columns:
                clean = pd.to_numeric(periodic[field], errors="coerce").dropna()
                if not clean.empty:
                    value = float(clean.iloc[-1])
            st.markdown(
                f"""
                <div class="vital-card">
                  <div class="vital-title">{label}</div>
                  <div class="vital-value">{'—' if value is None else f'{value:.1f}'}</div>
                  <div class="metric-subtext">{vital_status_dot(value, *thresholds)}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    tabs = st.tabs(["Vitals", "Labs", "Diagnosis & Treatment", "Medications", "Fluid Balance", "Ventilation", "History & Comorbidities", "Predictions"])

    with tabs[0]:
        if periodic.empty:
            st.info("No periodic vitals available.")
        else:
            st.plotly_chart(vitals_multiplot(periodic), use_container_width=True)
            summary = periodic[[c for c in ["heartrate", "sao2", "respiration", "systemicsystolic", "temperature", "cvp"] if c in periodic.columns]].describe().T
            st.dataframe(summary, use_container_width=True)

    with tabs[1]:
        labs = pd.DataFrame((fetch_labs(patient_id) or {}).get("labs", []))
        if labs.empty:
            st.info("No labs available.")
        else:
            options = sorted(labs["labname"].dropna().unique().tolist())
            selected = st.multiselect("Lab Tests", options, default=options[: min(4, len(options))])
            if selected:
                st.plotly_chart(labs_chart(labs, selected), use_container_width=True)
            st.dataframe(labs.groupby("labname")["labresult"].agg(["mean", "min", "max", "count"]).reset_index(), use_container_width=True)

    with tabs[2]:
        left, right = st.columns(2)
        with left:
            st.dataframe(pd.DataFrame((fetch_diagnosis(patient_id) or {}).get("diagnosis", [])), use_container_width=True)
        with right:
            st.dataframe(pd.DataFrame((fetch_treatments(patient_id) or {}).get("treatments", [])), use_container_width=True)

    with tabs[3]:
        meds = fetch_medications(patient_id) or {}
        st.write("Regular Medications")
        st.dataframe(pd.DataFrame(meds.get("medications", [])), use_container_width=True)
        st.write("Infusions")
        st.dataframe(pd.DataFrame(meds.get("infusions", [])), use_container_width=True)

    with tabs[4]:
        fluid = fetch_fluid_balance(patient_id) or {}
        summary_df = pd.DataFrame(fluid.get("summary", []))
        timeline_df = pd.DataFrame(fluid.get("timeline", []))
        k1, k2, k3, k4 = st.columns(4)
        if not summary_df.empty:
            row = summary_df.iloc[0]
            k1.metric("Total Intake", str(row.get("intake", row.get("total_intake", "—"))))
            k2.metric("Total Output", str(row.get("output", row.get("total_output", "—"))))
            k3.metric("Net Balance", str(row.get("net", row.get("fluid_balance", "—"))))
            k4.metric("Urine/hr", str(row.get("urine_output_per_hour", row.get("urine_per_hour", "—"))))
        if not timeline_df.empty:
            st.plotly_chart(fluid_balance_chart(timeline_df), use_container_width=True)
        else:
            st.info("Fluid timeline is unavailable.")

    with tabs[5]:
        vent = pd.DataFrame((fetch_ventilation(patient_id) or {}).get("ventilation", []))
        if vent.empty:
            st.info("Patient not on mechanical ventilation or no records available.")
        else:
            st.dataframe(vent, use_container_width=True)

    with tabs[6]:
        comorb_rows = (fetch_comorbidities(patient_id) or {}).get("comorbidities", [])
        comorb = comorb_rows[0] if comorb_rows else patient
        st.markdown(comorbidity_tags(comorb), unsafe_allow_html=True)
        st.json(comorb)

    with tabs[7]:
        result = fetch_predict_early_mortality(patient_id) or {}
        if result:
            left, right = st.columns([2, 3])
            with left:
                st.plotly_chart(risk_gauge(result["risk_score"]), use_container_width=True)
            with right:
                st.markdown(risk_badge(result["risk_label"]), unsafe_allow_html=True)
                st.write(f"Risk Score: {result['risk_score']:.3f} ({result['risk_score'] * 100:.1f}%)")
                st.write(f"Prediction: {'Alert triggered' if result['prediction'] else 'No alert'}")
                st.write(f"Action: {result['recommended_action']}")
                st.plotly_chart(feature_importance_chart(result["top_features"]), use_container_width=True)
        los_result = fetch_predict_los(patient_id) or {}
        if los_result:
            st.plotly_chart(los_gauge(los_result["predicted_icu_los_hours"]), use_container_width=True)
        sepsis = fetch_predict_sepsis(patient_id) or {}
        if sepsis:
            st.metric("qSOFA Score", int(sepsis.get("qsofa_score", 0)))
            criteria = {
                "resp_rate_score": 1 if sepsis.get("qsofa_score", 0) >= 1 else 0,
                "sbp_score": 1 if sepsis.get("qsofa_score", 0) >= 1 else 0,
                "gcs_score": 1 if sepsis.get("qsofa_score", 0) >= 1 else 0,
                "lactate_score": 1 if sepsis.get("sofa_approx_score", 0) >= 4 else 0,
                "sofa_score": sepsis.get("sofa_approx_score", 0),
            }
            st.plotly_chart(sepsis_radar_chart(criteria), use_container_width=True)
