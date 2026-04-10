from __future__ import annotations

import pandas as pd
import streamlit as st

from frontend.api_client import (
    fetch_comorbidities,
    fetch_diagnosis,
    fetch_fluid_balance,
    fetch_labs,
    fetch_medications,
    fetch_model_info,
    fetch_patient,
    fetch_treatments,
    fetch_ventilation,
    fetch_vitals,
    predict_recent_patient,
    predict_deterioration,
    predict_mortality,
)
from frontend.components.cards import comorbidity_tags, patient_header_card, risk_badge, vital_status_dot
from frontend.components.charts import fluid_balance_chart, labs_chart, vitals_multiplot


def _build_prediction_payload(patient: dict) -> dict:
    return {
        "age": patient.get("age"),
        "apache_score": patient.get("apache_score", patient.get("apachescore")),
        "apache_diagnosis": patient.get("apache_diagnosis", patient.get("apacheadmissiondx")),
        "hr_mean": patient.get("hr_mean"),
        "sao2_mean": patient.get("sao2_mean"),
        "sbp_mean": patient.get("sbp_mean"),
        "resp_mean": patient.get("resp_mean"),
        "temp_mean": patient.get("temp_mean"),
        "lactate_value": patient.get("lactate_value"),
        "creatinine_value": patient.get("creatinine_value"),
        "gcs_total": patient.get("gcs_total"),
        "vasopressor_active": patient.get("vasopressor_active", patient.get("on_vasopressor", 0)),
        "ventilator_active": patient.get("ventilator_active", patient.get("on_ventilator", 0)),
        "window_id": patient.get("window_id", 0),
    }


def _prediction_metric(label: str, score: float | None, status: str | None, note: str) -> None:
    pretty_score = "N/A" if score is None else f"{score:.3f}"
    badge = risk_badge(status or "LOW") if status else '<span class="risk-badge risk-low">UNKNOWN</span>'
    st.markdown(
        f"""
        <div class="metric-card">
          <div class="metric-label">{label}</div>
          <div class="metric-value">{pretty_score}</div>
          <div class="metric-subtext">{note}</div>
          <div style="margin-top:10px;">{badge}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


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
    model_info = fetch_model_info() or {}
    prediction_payload = _build_prediction_payload(patient)
    vitals_payload = fetch_vitals(patient_id) or {}
    periodic = pd.DataFrame(vitals_payload.get("periodic", []))
    if not periodic.empty:
        if "offset_hours" in periodic.columns:
            periodic["offset_hours"] = pd.to_numeric(periodic["offset_hours"], errors="coerce")
        elif "observationoffset" in periodic.columns:
            periodic["offset_hours"] = pd.to_numeric(periodic["observationoffset"], errors="coerce") / 60.0
        else:
            periodic["offset_hours"] = pd.NA
        latest_row = periodic.iloc[-1].to_dict()
        prediction_payload.update(
            {
                "hr_mean": latest_row.get("heartrate", prediction_payload.get("hr_mean")),
                "sao2_mean": latest_row.get("sao2", prediction_payload.get("sao2_mean")),
                "resp_mean": latest_row.get("respiration", prediction_payload.get("resp_mean")),
                "sbp_mean": latest_row.get("systemicsystolic", prediction_payload.get("sbp_mean")),
                "temp_mean": latest_row.get("temperature", prediction_payload.get("temp_mean")),
                "cvp_mean": latest_row.get("cvp", prediction_payload.get("cvp_mean")),
            }
        )

    labs_payload = fetch_labs(patient_id) or {}
    labs = pd.DataFrame(labs_payload.get("labs", []))
    if not labs.empty and {"labname", "labresult"}.issubset(labs.columns):
        latest_labs = (
            labs.dropna(subset=["labname"])
            .sort_values(by=[c for c in ["labresultoffset"] if c in labs.columns] or ["labname"])
            .groupby("labname", as_index=False)
            .tail(1)
        )
        lab_map = {
            "lactate": "lactate_value",
            "creatinine": "creatinine_value",
            "glucose": "glucose_value",
            "sodium": "sodium_value",
            "potassium": "potassium_value",
            "bicarbonate": "bicarbonate_value",
            "hemoglobin": "hemoglobin_value",
            "wbc": "wbc_value",
        }
        for _, row in latest_labs.iterrows():
            name = str(row.get("labname", "")).strip().lower()
            for token, feature_name in lab_map.items():
                if token in name:
                    prediction_payload[feature_name] = row.get("labresult", prediction_payload.get(feature_name))
                    break

    st.markdown('<div class="section-header">ML Predictions</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns([1, 1, 1, 1])
    threshold_type = c1.selectbox(
        "Threshold",
        ["max_f1", "balanced", "max_sensitivity_90", "max_specificity_90", "default"],
        index=0,
        key=f"prediction_threshold_{patient_id}",
    )
    hours_back = c2.selectbox("Recent Horizon", [4, 6, 8, 12], index=2, key=f"prediction_hours_{patient_id}")
    window_id = c3.selectbox(
        "Window",
        [0, 1, 2, 3, 4, 5],
        index=int(prediction_payload.get("window_id", 0) or 0),
        key=f"prediction_window_{patient_id}",
    )
    run_prediction = c4.button("Snapshot Predict", key=f"run_prediction_{patient_id}", use_container_width=True)
    run_recent_prediction = st.button("Recent TS Predict", key=f"run_recent_prediction_{patient_id}", use_container_width=True)
    st.caption(
        f"Loaded models: mortality={'yes' if model_info.get('mortality_model') else 'no'}, "
        f"deterioration={'yes' if model_info.get('deterioration_model') else 'no'}"
    )

    if run_prediction:
        mortality_result = predict_mortality(prediction_payload, threshold_type=threshold_type) or {}
        deterioration_result = predict_deterioration(prediction_payload, window_id=window_id, threshold_type=threshold_type) or {}
        r1, r2 = st.columns(2)
        with r1:
            _prediction_metric(
                "Mortality Risk",
                mortality_result.get("risk_score"),
                mortality_result.get("risk_label"),
                mortality_result.get("interpretation", "Prediction unavailable"),
            )
        with r2:
            deterioration_label = "HIGH" if deterioration_result.get("prediction") == 1 else "LOW"
            if deterioration_result.get("error"):
                deterioration_label = "UNKNOWN"
            _prediction_metric(
                "Deterioration Risk",
                deterioration_result.get("deterioration_risk"),
                deterioration_label,
                deterioration_result.get("error", f"Threshold: {deterioration_result.get('threshold_type', threshold_type)}"),
            )
        if mortality_result.get("top_features"):
            st.write("Top mortality drivers")
            st.dataframe(
                pd.DataFrame(mortality_result["top_features"], columns=["feature", "importance"]),
                use_container_width=True,
                hide_index=True,
            )

    if run_recent_prediction:
        recent_result = predict_recent_patient(patient_id, hours_back=hours_back, threshold_type=threshold_type) or {}
        if recent_result.get("error"):
            st.warning(recent_result["error"])
        else:
            st.caption(
                f"Built from last {recent_result.get('hours_back', hours_back)} hours "
                f"across {recent_result.get('window_count', 1)} window(s)."
            )
            rr1, rr2 = st.columns(2)
            mortality_recent = recent_result.get("mortality", {})
            deterioration_recent = recent_result.get("deterioration", {})
            with rr1:
                _prediction_metric(
                    "Recent-History Mortality",
                    mortality_recent.get("risk_score"),
                    mortality_recent.get("risk_label"),
                    mortality_recent.get("interpretation", "Prediction unavailable"),
                )
            with rr2:
                det_label = "HIGH" if deterioration_recent.get("prediction") == 1 else "LOW"
                if deterioration_recent.get("error"):
                    det_label = "UNKNOWN"
                _prediction_metric(
                    "Recent-History Deterioration",
                    deterioration_recent.get("deterioration_risk"),
                    det_label,
                    deterioration_recent.get("error", f"Current window: {recent_result.get('current_window_id', 0)}"),
                )

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


