from __future__ import annotations

from datetime import datetime
from pathlib import Path
import sys

import streamlit as st

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from frontend.api_client import fetch_stats, fetch_system_health
from frontend.components.styles import inject_styles
from frontend.pages import analytics, overview, patient_detail, patient_search, predictions

st.set_page_config(page_title="ICU Analytics System", page_icon="hospital", layout="wide", initial_sidebar_state="expanded")
inject_styles()

PAGE_ROUTES = {
    "overview": ("Overview", overview.render_page),
    "patient_search": ("Patient Search", patient_search.render_page),
    "patient_detail": ("Patient Detail", patient_detail.render_page),
    "predictions": ("Predictions", predictions.render_page),
    "analytics": ("Analytics", analytics.render_page),
}

LEGACY_PAGE_KEYS = {
    "Overview": "overview",
    "Patient Search": "patient_search",
    "Patient Detail": "patient_detail",
    "Predictions": "predictions",
    "Analytics": "analytics",
}

if "page" not in st.session_state:
    st.session_state["page"] = "overview"
st.session_state["page"] = LEGACY_PAGE_KEYS.get(st.session_state["page"], st.session_state["page"])
if st.session_state["page"] not in PAGE_ROUTES:
    st.session_state["page"] = "overview"
if "selected_patient" not in st.session_state:
    st.session_state["selected_patient"] = 0

stats = fetch_stats()
health = fetch_system_health()
raw_status = str((health or {}).get("status", "")).strip().lower()
if raw_status == "online":
    system_status = "ONLINE"
elif health:
    system_status = "PARTIAL"
else:
    system_status = "OFFLINE"
status_css = {"ONLINE": "status-online", "PARTIAL": "status-partial", "OFFLINE": "status-offline"}

with st.sidebar:
    st.markdown('<div class="sidebar-logo">ICU ANALYTICS</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-subtitle">eICU v2.0 · Local Instance</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="sidebar-subtitle"><span class="status-dot {status_css.get(system_status, "status-offline")}"></span>{system_status}</div>',
        unsafe_allow_html=True,
    )
    if health and system_status == "PARTIAL":
        st.markdown(
            f"<div class='sidebar-subtitle'>{health.get('reason', 'Prediction service is not fully ready')}</div>",
            unsafe_allow_html=True,
        )
    st.divider()
    page_keys = list(PAGE_ROUTES.keys())
    chosen = st.radio(
        "Navigation",
        page_keys,
        index=page_keys.index(st.session_state["page"]),
        format_func=lambda key: PAGE_ROUTES[key][0],
    )
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

PAGE_ROUTES[st.session_state["page"]][1]()
