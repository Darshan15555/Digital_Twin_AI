from __future__ import annotations

import pandas as pd
import streamlit as st

from frontend.api_client import fetch_digital_twin, fetch_vitals
from frontend.components.cards import alert_banner
from frontend.components.charts import digital_twin_chart


def render_page() -> None:
    st.markdown('<div class="section-header">Digital Twin - Patient Simulation</div>', unsafe_allow_html=True)
    patient_col, steps_col, band_col, zones_col = st.columns([2, 2, 1, 1])
    patient_id = int(patient_col.number_input("Patient ID", min_value=0, value=int(st.session_state.get("selected_patient", 0) or 0), step=1))
    steps = steps_col.select_slider("Forecast Steps", options=[6, 12, 18, 24], value=12)
    show_band = band_col.toggle("Confidence Band", value=True)
    zones_col.toggle("Alert Zones", value=True)
    fio2_delta = st.slider("FiO2 Adjustment", min_value=-0.2, max_value=0.2, value=0.0, step=0.05)
    vaso_delta = st.slider("Vasopressor Dose Modifier", min_value=-1.0, max_value=1.0, value=0.0, step=0.1)

    if not patient_id:
        st.info("Select a patient to generate the digital twin.")
        return
    st.session_state["selected_patient"] = patient_id

    if st.button("Generate Digital Twin"):
        with st.spinner("Running physiological forecast..."):
            st.session_state["digital_twin_payload"] = fetch_digital_twin(patient_id, steps=min(steps, 12), fio2_delta=fio2_delta, vasopressor_delta=vaso_delta)

    twin = st.session_state.get("digital_twin_payload")
    if not twin:
        st.info("Generate a digital twin forecast to view the simulation.")
        return

    label = twin.get("deterioration", {}).get("label", "Stable")
    level = "critical" if label.lower() == "critical" else "warning" if label.lower() == "deteriorating" else "info"
    st.markdown(alert_banner(f"Deterioration status: {label}", level), unsafe_allow_html=True)

    periodic = pd.DataFrame((fetch_vitals(patient_id) or {}).get("periodic", []))
    if periodic.empty:
        st.warning("Historical vitals are unavailable for charting.")
        return
    periodic["offset_hours"] = periodic.get("offset_hours", periodic["observationoffset"] / 60.0)
    last_x = float(periodic["offset_hours"].dropna().iloc[-1]) if not periodic["offset_hours"].dropna().empty else 0.0
    forecast_len = len(next(iter(twin.get("forecast", {}).values()), {}).get("values", []))
    future_offsets = [round(last_x + idx + 1, 2) for idx in range(forecast_len)]

    mapping = [("heartrate", "#FF4560"), ("sao2", "#00E5A0"), ("respiration", "#00C8FF"), ("systemicsystolic", "#A78BFA"), ("temperature", "#FFB830")]
    cols = st.columns(2)
    for idx, (field, color) in enumerate(mapping):
        with cols[idx % 2]:
            if field in twin.get("forecast", {}):
                forecast = twin["forecast"].copy()
                if not show_band:
                    forecast[field]["lower"] = []
                    forecast[field]["upper"] = []
                st.plotly_chart(digital_twin_chart(periodic.tail(50), forecast, future_offsets, field, color), use_container_width=True)

    with st.expander("View Forecast Data"):
        rows = []
        for i, offset in enumerate(future_offsets):
            row = {"time_step": i + 1, "offset_h": offset}
            for field, _ in mapping:
                values = twin.get("forecast", {}).get(field, {}).get("values", [])
                row[field] = values[i] if i < len(values) else None
            rows.append(row)
        st.dataframe(pd.DataFrame(rows), use_container_width=True)

