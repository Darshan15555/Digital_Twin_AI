from __future__ import annotations

import streamlit as st

from frontend.api_client import predict_vitals


def render_page() -> None:
    st.markdown('<div class="section-header">Predictions</div>', unsafe_allow_html=True)
    st.caption("Prediction based on vital signs.")
    in_progress_key = "predictions_in_progress"
    if in_progress_key not in st.session_state:
        st.session_state[in_progress_key] = False

    status = st.empty()
    if st.session_state[in_progress_key]:
        status.info("Processing...")

    c1, c2 = st.columns(2)
    with c1:
        hr = st.number_input(
            "Heart Rate (beats per minute)",
            min_value=30.0,
            max_value=220.0,
            value=None,
            step=1.0,
            placeholder="Enter heart rate",
        )
        st.caption("How fast the heart is beating right now.")
        spo2 = st.number_input(
            "Oxygen Saturation (%)",
            min_value=70.0,
            max_value=100.0,
            value=None,
            step=1.0,
            placeholder="Enter oxygen saturation",
        )
        st.caption("Percentage of oxygen in the blood.")
    with c2:
        bp_sys = st.number_input(
            "Blood Pressure (Systolic)",
            min_value=40.0,
            max_value=300.0,
            value=None,
            step=1.0,
            placeholder="Enter systolic blood pressure",
        )
        st.caption("Top blood pressure number during heart contraction.")
        bp_dia = st.number_input(
            "Blood Pressure (Diastolic)",
            min_value=20.0,
            max_value=220.0,
            value=None,
            step=1.0,
            placeholder="Enter diastolic blood pressure",
        )
        st.caption("Bottom blood pressure number between heart beats.")

    threshold = st.slider("Threshold", min_value=0.0, max_value=1.0, value=0.5, step=0.01)
    if st.button("Predict", use_container_width=True, disabled=st.session_state[in_progress_key]):
        if any(value is None for value in (hr, spo2, bp_sys, bp_dia)):
            st.warning("Please enter all required values")
            return
        st.session_state[in_progress_key] = True
        status.info("Processing...")
        with st.spinner("Processing..."):
            result = predict_vitals(hr=float(hr), spo2=float(spo2), bp_sys=float(bp_sys), bp_dia=float(bp_dia), threshold=threshold) or {}
        st.session_state[in_progress_key] = False
        if not result:
            st.error("Prediction request failed.")
            status.empty()
        elif isinstance(result, dict) and result.get("ok") is False:
            st.error(str(result.get("message", "Prediction request failed.")))
            status.empty()
        else:
            status.success("Result ready")
            prediction_value = int(result.get("prediction", 0))
            risk_category = "High Risk" if prediction_value == 1 else "Low Risk"
            probability = float(result.get("probability", 0.0))
            threshold_used = float(result.get("threshold_used", threshold))
            model_name = str(result.get("model_name", "N/A"))

            st.success(f"Prediction completed: {risk_category}")
            m1, m2, m3 = st.columns(3)
            m1.metric("Risk Category", risk_category)
            m2.metric("Risk Probability", f"{probability * 100:.1f}%")
            m3.metric("Threshold Used", f"{threshold_used * 100:.1f}%")
            st.info(f"Risk Probability: {probability * 100:.1f}% (chance of deterioration)")
            st.caption(f"Model used: {model_name}")
