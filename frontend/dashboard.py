from __future__ import annotations

from datetime import datetime
from pathlib import Path
import sys

import streamlit as st

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from frontend.api_client import fetch_stats
from frontend.components.styles import inject_styles
from frontend.pages import analytics, overview, patient_detail, patient_search, predictions

st.set_page_config(page_title="ICU Analytics System", page_icon="hospital", layout="wide", initial_sidebar_state="expanded")
inject_styles()

PAGES = {
    "Overview": overview.render_page,
    "Patient Search": patient_search.render_page,
    "Patient Detail": patient_detail.render_page,
    "Predictions": predictions.render_page,
    "Analytics": analytics.render_page,
}

if "page" not in st.session_state:
    st.session_state["page"] = "Overview"
if "selected_patient" not in st.session_state:
    st.session_state["selected_patient"] = 0

stats = fetch_stats()
online = stats is not None

with st.sidebar:
    st.markdown('<div class="sidebar-logo">ICU ANALYTICS</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-subtitle">eICU v2.0 · Local Instance</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="sidebar-subtitle"><span class="status-dot {"status-online" if online else "status-offline"}"></span>{"ONLINE" if online else "OFFLINE"}</div>',
        unsafe_allow_html=True,
    )
    st.divider()
    chosen = st.radio("Navigation", list(PAGES.keys()), index=list(PAGES.keys()).index(st.session_state["page"]))
    st.session_state["page"] = chosen
    st.divider()
    if stats:
        st.markdown(
            f"<div class='sidebar-subtitle'>Patients <span class='mono-text' style='color:#00C8FF'>{int(stats.get('total_patients', 0)):,}</span></div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"<div class='sidebar-subtitle'>Mortality <span class='mono-text' style='color:#00C8FF'>{(stats.get('mortality_rate') or 0) * 100:.1f}%</span></div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"<div class='sidebar-subtitle'>Avg APACHE <span class='mono-text' style='color:#00C8FF'>{(stats.get('avg_apache_score') or 0):.1f}</span></div>",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"<div class='sidebar-subtitle'>Avg LOS <span class='mono-text' style='color:#00C8FF'>{(stats.get('avg_icu_los_hours') or 0):.1f}h</span></div>",
            unsafe_allow_html=True,
        )
    st.divider()
    if st.button("Clear Cache"):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.rerun()
    st.markdown(f"<div class='sidebar-subtitle'>Last updated {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</div>", unsafe_allow_html=True)
    st.markdown('<div class="sidebar-subtitle">v2.1.0</div>', unsafe_allow_html=True)

PAGES[st.session_state["page"]]()
