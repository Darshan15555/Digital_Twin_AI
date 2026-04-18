from __future__ import annotations

import pandas as pd
import streamlit as st

from frontend.api_client import (
    fetch_comorbidities,
    fetch_diagnosis,
    fetch_fluid_balance,
    fetch_labs,
    fetch_medications,
    fetch_patient,
    fetch_treatments,
    fetch_ventilation,
    fetch_vitals,
    predict_vitals,
)
from frontend.components.cards import comorbidity_tags, patient_header_card, vital_status_dot
from frontend.components.charts import fluid_balance_chart, labs_chart, vitals_multiplot


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
    vitals_payload = fetch_vitals(patient_id) or {}
    periodic = pd.DataFrame(vitals_payload.get("periodic", []))
    if not periodic.empty:
        if "offset_hours" in periodic.columns:
            periodic["offset_hours"] = pd.to_numeric(periodic["offset_hours"], errors="coerce")
        elif "observationoffset" in periodic.columns:
            periodic["offset_hours"] = pd.to_numeric(periodic["observationoffset"], errors="coerce") / 60.0
        else:
            periodic["offset_hours"] = pd.NA

    labs_payload = fetch_labs(patient_id) or {}
    labs = pd.DataFrame(labs_payload.get("labs", []))

    st.markdown('<div class="section-header">Vital-Signs Prediction</div>', unsafe_allow_html=True)
    st.caption("Prediction based on vital signs.")
    in_progress_key = f"patient_prediction_in_progress_{patient_id}"
    if in_progress_key not in st.session_state:
        st.session_state[in_progress_key] = False
    status = st.empty()
    if st.session_state[in_progress_key]:
        status.info("Processing...")

    c1, c2, c3, c4 = st.columns([1, 1, 1, 1])
    hr_value = c1.number_input(
        "Heart Rate (beats per minute)",
        min_value=30.0,
        max_value=220.0,
        value=None,
        step=1.0,
        placeholder="Enter heart rate",
        key=f"patient_prediction_hr_{patient_id}",
    )
    c1.caption("How fast the heart is beating right now.")
    spo2_value = c2.number_input(
        "Oxygen Saturation (%)",
        min_value=70.0,
        max_value=100.0,
        value=None,
        step=1.0,
        placeholder="Enter oxygen saturation",
        key=f"patient_prediction_spo2_{patient_id}",
    )
    c2.caption("Percentage of oxygen in the blood.")
    bp_sys_value = c3.number_input(
        "Blood Pressure (Systolic)",
        min_value=40.0,
        max_value=300.0,
        value=None,
        step=1.0,
        placeholder="Enter systolic blood pressure",
        key=f"patient_prediction_bp_sys_{patient_id}",
    )
    c3.caption("Top blood pressure number during heart contraction.")
    bp_dia_value = c4.number_input(
        "Blood Pressure (Diastolic)",
        min_value=20.0,
        max_value=220.0,
        value=None,
        step=1.0,
        placeholder="Enter diastolic blood pressure",
        key=f"patient_prediction_bp_dia_{patient_id}",
    )
    c4.caption("Bottom blood pressure number between heart beats.")
    threshold = st.slider("Decision Threshold", min_value=0.0, max_value=1.0, value=0.5, step=0.01, key=f"prediction_threshold_{patient_id}")
    run_prediction = st.button(
        "Run Predict",
        key=f"run_prediction_{patient_id}",
        use_container_width=True,
        disabled=st.session_state[in_progress_key],
    )

    if run_prediction:
        if any(value is None for value in (hr_value, spo2_value, bp_sys_value, bp_dia_value)):
            st.warning("Please enter all required values")
            return
        st.session_state[in_progress_key] = True
        status.info("Processing...")
        with st.spinner("Processing..."):
            result = predict_vitals(
                hr=float(hr_value),
                spo2=float(spo2_value),
                bp_sys=float(bp_sys_value),
                bp_dia=float(bp_dia_value),
                threshold=float(threshold),
            ) or {}
        st.session_state[in_progress_key] = False
        if not result:
            st.warning("Prediction unavailable.")
            status.empty()
        elif isinstance(result, dict) and result.get("ok") is False:
            st.warning(str(result.get("message", "Prediction unavailable.")))
            status.empty()
        else:
            status.success("Result ready")
            prediction_value = int(result.get("prediction", 0))
            risk_category = "High Risk" if prediction_value == 1 else "Low Risk"
            probability = float(result.get("probability", 0.0))
            threshold_used = float(result.get("threshold_used", threshold))
            model_name = str(result.get("model_name", "N/A"))

            m1, m2, m3 = st.columns(3)
            m1.metric("Risk Category", risk_category)
            m2.metric("Risk Probability", f"{probability * 100:.1f}%")
            m3.metric("Threshold Used", f"{threshold_used * 100:.1f}%")
            st.info(f"Risk Probability: {probability * 100:.1f}% (chance of deterioration)")
            st.caption(f"Model used: {model_name}")

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

    tabs = st.tabs(["Vitals", "Labs", "Diagnosis & Treatment", "Medications", "Fluid Balance", "Ventilation", "History & Comorbidities"])

    with tabs[0]:
        if periodic.empty:
            st.info("No periodic vitals available.")
        else:
            st.plotly_chart(vitals_multiplot(periodic), use_container_width=True)
            summary = periodic[[c for c in ["heartrate", "sao2", "respiration", "systemicsystolic", "temperature", "cvp"] if c in periodic.columns]].describe().T
            st.dataframe(summary, use_container_width=True)

    with tabs[1]:
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


