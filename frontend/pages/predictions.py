from __future__ import annotations

import streamlit as st

from frontend.api_client import fetch_predict_early_mortality, fetch_predict_los, fetch_predict_mortality, fetch_predict_sepsis
from frontend.components.cards import risk_badge
from frontend.components.charts import feature_importance_chart, los_gauge, risk_gauge, sepsis_radar_chart


def render_page() -> None:
    st.markdown('<div class="section-header">AI Predictions & Risk Scoring</div>', unsafe_allow_html=True)
    patient_id = int(st.number_input("Patient ID", min_value=0, value=int(st.session_state.get("selected_patient", 0) or 0), step=1))
    if patient_id:
        st.session_state["selected_patient"] = patient_id
    if not patient_id:
        st.info("Select a patient to run predictions.")
        return

    with st.expander("Early Mortality Risk", expanded=True):
        if st.button("Run Early Mortality Model"):
            st.session_state["last_prediction"] = {"type": "early", "payload": fetch_predict_early_mortality(patient_id)}
        result = (st.session_state.get("last_prediction") or {}).get("payload") if (st.session_state.get("last_prediction") or {}).get("type") == "early" else fetch_predict_early_mortality(patient_id)
        if result:
            if not result.get("model_ready", False):
                st.warning("Early mortality model is not available. Run the training pipeline or restart the backend after training.")
            col1, col2 = st.columns([2, 3])
            with col1:
                st.plotly_chart(risk_gauge(result["risk_score"]), use_container_width=True)
            with col2:
                st.markdown(risk_badge(result["risk_label"]), unsafe_allow_html=True)
                st.write(f"Risk Score: {result['risk_score']:.3f} ({result['risk_score'] * 100:.1f}%)")
                st.write(f"Prediction: {'Mortality risk alert' if result['prediction'] else 'No mortality alert'}")
                st.write(f"Operating Mode: {result['operating_mode']}")
                st.write(f"Recommended Action: {result['recommended_action']}")
                st.plotly_chart(feature_importance_chart(result["top_features"]), use_container_width=True)

    with st.expander("Legacy Mortality Risk"):
        if st.button("Run Mortality Model"):
            st.session_state["last_prediction"] = {"type": "mortality", "payload": fetch_predict_mortality(patient_id)}
        result = (st.session_state.get("last_prediction") or {}).get("payload") if (st.session_state.get("last_prediction") or {}).get("type") == "mortality" else fetch_predict_mortality(patient_id)
        if result:
            if not result.get("model_ready", False):
                st.warning(result.get("interpretation", "Mortality model is not available."))
            c1, c2 = st.columns([2, 3])
            with c1:
                st.plotly_chart(risk_gauge(result["risk_score"]), use_container_width=True)
            with c2:
                st.markdown(risk_badge(result["risk_label"]), unsafe_allow_html=True)
                st.write(f"Risk Score: {result['risk_score']:.3f} ({result['risk_score'] * 100:.1f}%)")
                st.write(f"Prediction: {'Mortality predicted' if result['prediction'] else 'Survival predicted'}")
                st.plotly_chart(feature_importance_chart(result["top_features"]), use_container_width=True)

    with st.expander("Length of Stay"):
        if st.button("Run LOS Prediction"):
            st.session_state["last_prediction"] = {"type": "los", "payload": fetch_predict_los(patient_id)}
        result = (st.session_state.get("last_prediction") or {}).get("payload") if (st.session_state.get("last_prediction") or {}).get("type") == "los" else fetch_predict_los(patient_id)
        if result:
            if not result.get("model_ready", False):
                st.info(result.get("note", "LOS is currently an estimate, not a trained model prediction."))
            st.plotly_chart(los_gauge(result["predicted_icu_los_hours"]), use_container_width=True)
            st.write(f"Predicted LOS: {result['predicted_icu_los_hours']:.1f} hours ({result['predicted_icu_los_days']:.1f} days)")

    with st.expander("Sepsis Risk (qSOFA)"):
        if st.button("Calculate Sepsis Score"):
            st.session_state["last_prediction"] = {"type": "sepsis", "payload": fetch_predict_sepsis(patient_id)}
        result = (st.session_state.get("last_prediction") or {}).get("payload") if (st.session_state.get("last_prediction") or {}).get("type") == "sepsis" else fetch_predict_sepsis(patient_id)
        if result:
            score = int(result.get("qsofa_score", 0))
            left, right = st.columns([2, 3])
            with left:
                st.metric("qSOFA", score)
                st.write("SEPSIS RISK" if result.get("sepsis_risk") else "LOWER RISK")
            with right:
                criteria = {"resp_rate_score": 1 if score >= 1 else 0, "sbp_score": 1 if score >= 1 else 0, "gcs_score": 1 if score >= 1 else 0, "lactate_score": 1 if result.get("sofa_approx_score", 0) >= 4 else 0, "sofa_score": result.get("sofa_approx_score", 0)}
                st.plotly_chart(sepsis_radar_chart(criteria), use_container_width=True)
                st.write(f"Interpretation: {result.get('interpretation', 'Unavailable')}")
