"""
dashboard.py
Streamlit ICU Analytics Dashboard
Run with: streamlit run frontend/dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from config.config import API_HOST, API_PORT

API_BASE = f"http://{API_HOST}:{API_PORT}"

# ─── PAGE CONFIG ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ICU Analytics System",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── CUSTOM CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500&family=DM+Sans:wght@300;400;500;600&display=swap');

    :root {
        --bg: #0a0e1a;
        --surface: #111827;
        --surface2: #1a2234;
        --accent: #00d4ff;
        --accent2: #ff6b6b;
        --accent3: #4ade80;
        --text: #e2e8f0;
        --muted: #64748b;
    }

    .stApp { background-color: var(--bg); }

    .metric-card {
        background: var(--surface);
        border: 1px solid rgba(0, 212, 255, 0.15);
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        position: relative;
        overflow: hidden;
    }
    .metric-card::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 2px;
        background: linear-gradient(90deg, var(--accent), var(--accent2));
    }
    .metric-value {
        font-family: 'JetBrains Mono', monospace;
        font-size: 2.2rem;
        font-weight: 500;
        color: var(--accent);
    }
    .metric-label {
        font-family: 'DM Sans', sans-serif;
        font-size: 0.8rem;
        color: var(--muted);
        text-transform: uppercase;
        letter-spacing: 1.5px;
        margin-top: 4px;
    }

    .risk-badge-high {
        background: rgba(255, 107, 107, 0.15);
        border: 1px solid #ff6b6b;
        color: #ff6b6b;
        padding: 6px 16px;
        border-radius: 20px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.9rem;
        font-weight: 500;
    }
    .risk-badge-moderate {
        background: rgba(251, 191, 36, 0.15);
        border: 1px solid #fbbf24;
        color: #fbbf24;
        padding: 6px 16px;
        border-radius: 20px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.9rem;
        font-weight: 500;
    }
    .risk-badge-low {
        background: rgba(74, 222, 128, 0.15);
        border: 1px solid #4ade80;
        color: #4ade80;
        padding: 6px 16px;
        border-radius: 20px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.9rem;
        font-weight: 500;
    }

    .section-header {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.75rem;
        color: var(--accent);
        text-transform: uppercase;
        letter-spacing: 3px;
        margin-bottom: 16px;
        padding-bottom: 8px;
        border-bottom: 1px solid rgba(0, 212, 255, 0.2);
    }

    .patient-info-row {
        display: flex;
        gap: 16px;
        flex-wrap: wrap;
    }
    .info-chip {
        background: var(--surface2);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 8px;
        padding: 6px 14px;
        font-family: 'DM Sans', sans-serif;
        font-size: 0.85rem;
        color: var(--text);
    }
    .info-chip span { color: var(--muted); font-size: 0.75rem; }

    div[data-testid="stSidebar"] {
        background: var(--surface) !important;
        border-right: 1px solid rgba(0, 212, 255, 0.1);
    }
    .sidebar-title {
        font-family: 'JetBrains Mono', monospace;
        color: var(--accent);
        font-size: 1.1rem;
        font-weight: 500;
    }

    div[data-testid="metric-container"] {
        background: var(--surface);
        border: 1px solid rgba(0, 212, 255, 0.1);
        border-radius: 10px;
        padding: 12px;
    }
</style>
""", unsafe_allow_html=True)

PLOT_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(17,24,39,0.8)",
    font=dict(family="JetBrains Mono", color="#94a3b8", size=11),
    xaxis=dict(gridcolor="rgba(255,255,255,0.05)", showline=False, zeroline=False),
    yaxis=dict(gridcolor="rgba(255,255,255,0.05)", showline=False, zeroline=False),
    margin=dict(l=40, r=20, t=40, b=40),
    legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="rgba(255,255,255,0.1)")
)


# ─── API HELPERS ─────────────────────────────────────────────────────────────────
def api_get(path: str, params=None):
    try:
        r = requests.get(f"{API_BASE}{path}", params=params, timeout=10)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.ConnectionError:
        st.error("⚠️ Cannot connect to backend. Start the API: `python backend/api.py`")
        return None
    except Exception as e:
        st.error(f"API Error: {e}")
        return None


@st.cache_data(ttl=60)
def fetch_stats():
    return api_get("/stats")


@st.cache_data(ttl=30)
def fetch_patients(limit=200, search=None, high_risk=False):
    params = {"limit": limit, "high_risk_only": str(high_risk).lower()}
    if search:
        params["search"] = search
    return api_get("/patients", params=params)


@st.cache_data(ttl=60)
def fetch_patient(pid):
    return api_get(f"/patients/{pid}")


@st.cache_data(ttl=60)
def fetch_vitals(pid):
    return api_get(f"/patients/{pid}/vitals")


@st.cache_data(ttl=60)
def fetch_labs(pid):
    return api_get(f"/patients/{pid}/labs")


@st.cache_data(ttl=60)
def fetch_diagnosis(pid):
    return api_get(f"/patients/{pid}/diagnosis")


@st.cache_data(ttl=60)
def fetch_treatments(pid):
    return api_get(f"/patients/{pid}/treatments")


@st.cache_data(ttl=300)
def fetch_analytics_units():
    return api_get("/analytics/unit-breakdown")


@st.cache_data(ttl=300)
def fetch_analytics_age():
    return api_get("/analytics/mortality-by-age")


@st.cache_data(ttl=300)
def fetch_top_diagnoses():
    return api_get("/analytics/top-diagnoses")


# ─── SIDEBAR ─────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sidebar-title">🏥 ICU Analytics</div>', unsafe_allow_html=True)
    st.markdown('<div style="color:#64748b;font-size:0.75rem;font-family:\'JetBrains Mono\',monospace;margin-bottom:20px">eICU v2.0 · Local System</div>', unsafe_allow_html=True)

    page = st.radio(
        "Navigation",
        ["📊 Overview", "🔍 Patient Search", "👤 Patient Detail", "🧠 Analytics", "📡 Digital Twin"],
        label_visibility="collapsed"
    )

    st.divider()
    stats = fetch_stats()
    if stats:
        st.markdown(f"**Patients:** `{stats.get('total_patients', '—'):,}`")
        st.markdown(f"**Mortality:** `{stats.get('mortality_rate', '—')}%`")
        st.markdown(f"**Avg Age:** `{stats.get('avg_age', '—')}`")
        st.markdown(f"**APACHE Avg:** `{stats.get('avg_apache_score', '—')}`")
        model_status = "✅ Ready" if stats.get("model_ready") else "❌ Not trained"
        st.markdown(f"**ML Model:** {model_status}")

    st.divider()
    if st.button("🔄 Clear Cache"):
        st.cache_data.clear()
        st.rerun()

# ─── PAGES ────────────────────────────────────────────────────────────────────────

# ── OVERVIEW PAGE ──
if page == "📊 Overview":
    st.markdown("## 📊 ICU Overview")

    if stats:
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(f"""<div class="metric-card">
                <div class="metric-value">{stats.get('total_patients', 0):,}</div>
                <div class="metric-label">Total Patients</div>
            </div>""", unsafe_allow_html=True)
        with c2:
            st.markdown(f"""<div class="metric-card">
                <div class="metric-value">{stats.get('mortality_rate', 0)}%</div>
                <div class="metric-label">Mortality Rate</div>
            </div>""", unsafe_allow_html=True)
        with c3:
            st.markdown(f"""<div class="metric-card">
                <div class="metric-value">{stats.get('avg_age', 0)}</div>
                <div class="metric-label">Avg Age (yrs)</div>
            </div>""", unsafe_allow_html=True)
        with c4:
            st.markdown(f"""<div class="metric-card">
                <div class="metric-value">{stats.get('avg_apache_score', 0)}</div>
                <div class="metric-label">Avg APACHE Score</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-header">ICU Unit Breakdown</div>', unsafe_allow_html=True)
        units = fetch_analytics_units()
        if units:
            df_u = pd.DataFrame(units).head(10)
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=df_u["unittype"],
                y=df_u["count"],
                marker_color="#00d4ff",
                name="Patients"
            ))
            fig.add_trace(go.Scatter(
                x=df_u["unittype"],
                y=df_u["mortality_pct"],
                mode="lines+markers",
                yaxis="y2",
                name="Mortality %",
                line=dict(color="#ff6b6b", width=2),
                marker=dict(size=6)
            ))
            fig.update_layout(
                **PLOT_LAYOUT,
                yaxis2=dict(
                    overlaying="y", side="right",
                    gridcolor="rgba(0,0,0,0)",
                    title="Mortality %"
                ),
                height=320, title="Patients & Mortality by Unit"
            )
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown('<div class="section-header">Mortality by Age Group</div>', unsafe_allow_html=True)
        age_data = fetch_analytics_age()
        if age_data:
            df_a = pd.DataFrame(age_data)
            fig2 = go.Figure()
            fig2.add_trace(go.Bar(
                x=df_a["age_group"].astype(str),
                y=df_a["mortality_pct"],
                marker=dict(
                    color=df_a["mortality_pct"],
                    colorscale=[[0, "#4ade80"], [0.5, "#fbbf24"], [1, "#ff6b6b"]],
                    showscale=True
                )
            ))
            fig2.update_layout(**PLOT_LAYOUT, height=320, title="ICU Mortality by Age Group")
            st.plotly_chart(fig2, use_container_width=True)

    st.markdown('<div class="section-header">Top Diagnoses</div>', unsafe_allow_html=True)
    diag_data = fetch_top_diagnoses()
    if diag_data:
        df_d = pd.DataFrame(diag_data)
        fig3 = go.Figure(go.Bar(
            x=df_d["count"],
            y=df_d["diagnosisstring"].str[:50],
            orientation="h",
            marker_color="#00d4ff",
            marker_line=dict(width=0)
        ))
        fig3.update_layout(**PLOT_LAYOUT, height=400, title="Most Frequent Diagnoses")
        st.plotly_chart(fig3, use_container_width=True)


# ── PATIENT SEARCH ──
elif page == "🔍 Patient Search":
    st.markdown("## 🔍 Patient Search")

    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        search_q = st.text_input("Search by Patient ID, Unit Type, or Gender", placeholder="e.g. 141168")
    with col2:
        high_risk = st.checkbox("High Risk Only")
    with col3:
        limit = st.selectbox("Show", [50, 100, 200], index=1)

    data = fetch_patients(limit=limit, search=search_q if search_q else None, high_risk=high_risk)

    if data and data.get("patients"):
        df = pd.DataFrame(data["patients"])
        st.markdown(f"**{len(df)} patients found**")

        # Risk color coding
        def risk_color(score):
            if score is None or pd.isna(score):
                return "⬜"
            elif score >= 0.7:
                return "🔴"
            elif score >= 0.4:
                return "🟡"
            else:
                return "🟢"

        display_cols = ["patientunitstayid", "age", "gender", "unittype",
                        "apachescore", "hospital_mortality", "ml_risk_score"]
        available = [c for c in display_cols if c in df.columns]
        df_show = df[available].copy()

        if "ml_risk_score" in df_show.columns:
            df_show["risk"] = df_show["ml_risk_score"].apply(risk_color)

        if "hospital_mortality" in df_show.columns:
            df_show["outcome"] = df_show["hospital_mortality"].map({1: "💀 Expired", 0: "✅ Survived"})

        st.dataframe(df_show, use_container_width=True, height=400)

        st.markdown("---")
        selected_id = st.number_input("Enter Patient ID to view detail →", min_value=0, step=1)
        if selected_id and st.button("View Patient"):
            st.session_state["selected_patient"] = int(selected_id)
            st.session_state["go_to_detail"] = True
            st.rerun()
    else:
        st.info("No patients found. Make sure setup.py has been run.")


# ── PATIENT DETAIL ──
elif page == "👤 Patient Detail" or st.session_state.get("go_to_detail"):
    st.session_state["go_to_detail"] = False
    st.markdown("## 👤 Patient Detail")

    pid = st.session_state.get("selected_patient", None)
    pid_input = st.number_input("Patient Unit Stay ID", value=pid or 0, step=1, min_value=0)

    if pid_input:
        patient = fetch_patient(pid_input)

        if patient:
            st.session_state["selected_patient"] = pid_input

            # Patient header
            col_info, col_risk = st.columns([3, 1])
            with col_info:
                st.markdown(f"### Patient #{patient.get('patientunitstayid')}")
                chips = []
                if patient.get("age"):
                    chips.append(f"Age: {patient['age']}")
                if patient.get("gender"):
                    chips.append(f"Gender: {patient['gender']}")
                if patient.get("ethnicity"):
                    chips.append(f"Ethnicity: {patient['ethnicity']}")
                if patient.get("unittype"):
                    chips.append(f"Unit: {patient['unittype']}")
                if patient.get("apachescore"):
                    chips.append(f"APACHE: {patient['apachescore']}")
                st.markdown(" &nbsp;|&nbsp; ".join(f"**{c}**" for c in chips))

            with col_risk:
                risk = patient.get("ml_risk_score")
                if risk:
                    label = "HIGH" if risk >= 0.7 else "MODERATE" if risk >= 0.4 else "LOW"
                    badge_cls = f"risk-badge-{label.lower()}"
                    st.markdown(f'<br><div class="{badge_cls}">Risk: {label} ({risk:.0%})</div>', unsafe_allow_html=True)

            st.divider()

            # Tabs for different data
            tab1, tab2, tab3, tab4, tab5 = st.tabs(["📈 Vitals", "🧪 Labs", "🏥 Diagnosis", "💊 Treatment", "🤖 Prediction"])

            # ── VITALS TAB ──
            with tab1:
                vitals_data = fetch_vitals(pid_input)
                if vitals_data and vitals_data.get("vitals"):
                    df_v = pd.DataFrame(vitals_data["vitals"])
                    df_v["time_h"] = df_v["observationoffset"] / 60

                    fig = make_subplots(
                        rows=3, cols=2,
                        subplot_titles=["Heart Rate (bpm)", "SpO₂ (%)",
                                        "Respiration (rpm)", "Systolic BP (mmHg)",
                                        "Temperature (°C)", "CVP (mmHg)"]
                    )
                    plots = [
                        ("heartrate", 1, 1, "#00d4ff"),
                        ("sao2", 1, 2, "#4ade80"),
                        ("respiration", 2, 1, "#fbbf24"),
                        ("systemicsystolic", 2, 2, "#a78bfa"),
                        ("temperature", 3, 1, "#f97316"),
                        ("cvp", 3, 2, "#ec4899"),
                    ]
                    for col_name, row, col, color in plots:
                        if col_name in df_v.columns:
                            series = df_v[col_name].dropna()
                            if len(series) > 0:
                                fig.add_trace(go.Scatter(
                                    x=df_v.loc[series.index, "time_h"],
                                    y=series,
                                    mode="lines",
                                    line=dict(color=color, width=1.5),
                                    name=col_name,
                                    showlegend=False
                                ), row=row, col=col)

                    fig.update_layout(**PLOT_LAYOUT, height=600, title="Patient Vitals Over Time (Hours)")
                    st.plotly_chart(fig, use_container_width=True)
                    st.caption(f"Showing {len(df_v):,} observations")
                else:
                    st.info("No vitals data available for this patient.")

            # ── LABS TAB ──
            with tab2:
                labs_data = fetch_labs(pid_input)
                if labs_data and labs_data.get("labs"):
                    df_l = pd.DataFrame(labs_data["labs"])
                    df_l["time_h"] = df_l["labresultoffset"] / 60

                    key_labs = df_l["labname"].value_counts().head(8).index.tolist()
                    selected_labs = st.multiselect("Select lab tests to plot", key_labs, default=key_labs[:4])

                    if selected_labs:
                        fig_l = go.Figure()
                        colors_l = ["#00d4ff", "#4ade80", "#fbbf24", "#a78bfa", "#f97316", "#ec4899", "#60a5fa", "#34d399"]
                        for i, lab in enumerate(selected_labs):
                            sub = df_l[df_l["labname"] == lab].sort_values("time_h")
                            fig_l.add_trace(go.Scatter(
                                x=sub["time_h"], y=sub["labresult"],
                                mode="lines+markers",
                                name=lab,
                                line=dict(color=colors_l[i % len(colors_l)], width=2),
                                marker=dict(size=4)
                            ))
                        fig_l.update_layout(**PLOT_LAYOUT, height=350, title="Lab Results Over Time")
                        st.plotly_chart(fig_l, use_container_width=True)

                    # Summary table
                    lab_summary = df_l.groupby("labname")["labresult"].agg(["mean", "min", "max", "count"]).reset_index()
                    lab_summary.columns = ["Lab Test", "Mean", "Min", "Max", "Count"]
                    lab_summary = lab_summary.sort_values("Count", ascending=False)
                    st.dataframe(lab_summary, use_container_width=True)
                else:
                    st.info("No lab data available for this patient.")

            # ── DIAGNOSIS TAB ──
            with tab3:
                diag_data = fetch_diagnosis(pid_input)
                if diag_data and diag_data.get("diagnosis"):
                    df_diag = pd.DataFrame(diag_data["diagnosis"])
                    st.dataframe(
                        df_diag[["diagnosisoffset", "diagnosisstring", "icd9code", "diagnosispriority"]].rename(columns={
                            "diagnosisoffset": "Offset (min)",
                            "diagnosisstring": "Diagnosis",
                            "icd9code": "ICD9",
                            "diagnosispriority": "Priority"
                        }),
                        use_container_width=True
                    )
                else:
                    st.info("No diagnosis data available.")

            # ── TREATMENT TAB ──
            with tab4:
                treat_data = fetch_treatments(pid_input)
                if treat_data and treat_data.get("treatments"):
                    df_t = pd.DataFrame(treat_data["treatments"])
                    st.dataframe(
                        df_t[["treatmentoffset", "treatmentstring"]].rename(columns={
                            "treatmentoffset": "Offset (min)",
                            "treatmentstring": "Treatment"
                        }),
                        use_container_width=True
                    )
                else:
                    st.info("No treatment data available.")

            # ── PREDICTION TAB ──
            with tab5:
                st.markdown("### 🤖 Mortality Risk Prediction")
                if st.button("Run Prediction", type="primary"):
                    with st.spinner("Running ML model..."):
                        result = api_get(f"/patients/{pid_input}/predict")
                    if result:
                        score = result["risk_score"]
                        label = result["risk_label"]
                        badge_cls = f"risk-badge-{label.lower()}"

                        col_a, col_b = st.columns(2)
                        with col_a:
                            st.markdown(f'<br><div class="{badge_cls}" style="font-size:1.5rem;padding:14px 28px">⚠️ {label} RISK</div>', unsafe_allow_html=True)
                            st.markdown(f"**Risk Score:** `{score:.4f}` ({score:.1%})")
                            st.markdown(f"**Prediction:** {'Mortality predicted' if result['prediction'] else 'Survival predicted'}")

                        with col_b:
                            # Gauge chart
                            fig_g = go.Figure(go.Indicator(
                                mode="gauge+number",
                                value=score * 100,
                                domain={"x": [0, 1], "y": [0, 1]},
                                title={"text": "Risk Score", "font": {"color": "#94a3b8"}},
                                number={"suffix": "%", "font": {"color": "#e2e8f0"}},
                                gauge={
                                    "axis": {"range": [0, 100], "tickcolor": "#64748b"},
                                    "bar": {"color": "#ff6b6b" if score >= 0.7 else "#fbbf24" if score >= 0.4 else "#4ade80"},
                                    "bgcolor": "rgba(0,0,0,0)",
                                    "steps": [
                                        {"range": [0, 40], "color": "rgba(74,222,128,0.1)"},
                                        {"range": [40, 70], "color": "rgba(251,191,36,0.1)"},
                                        {"range": [70, 100], "color": "rgba(255,107,107,0.1)"},
                                    ],
                                    "threshold": {"line": {"color": "#ffffff", "width": 2}, "value": score * 100}
                                }
                            ))
                            fig_g.update_layout(**PLOT_LAYOUT, height=250)
                            st.plotly_chart(fig_g, use_container_width=True)

                        if result.get("top_features"):
                            st.markdown("**Top Predictive Features:**")
                            for feat, importance in result["top_features"]:
                                st.markdown(f"- `{feat}`: {importance:.3f} importance")
        else:
            st.warning("Patient not found. Enter a valid Patient Unit Stay ID.")


# ── ANALYTICS PAGE ──
elif page == "🧠 Analytics":
    st.markdown("## 🧠 Population Analytics")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="section-header">Unit Type Distribution</div>', unsafe_allow_html=True)
        units = fetch_analytics_units()
        if units:
            df_u = pd.DataFrame(units)
            fig = px.pie(df_u, values="count", names="unittype",
                         color_discrete_sequence=["#00d4ff", "#4ade80", "#fbbf24", "#f97316", "#a78bfa", "#ec4899", "#60a5fa"])
            fig.update_layout(**PLOT_LAYOUT, height=300)
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown('<div class="section-header">Mortality by Age Group</div>', unsafe_allow_html=True)
        age_data = fetch_analytics_age()
        if age_data:
            df_a = pd.DataFrame(age_data)
            fig2 = go.Figure(go.Bar(
                x=df_a["age_group"].astype(str),
                y=df_a["mortality_pct"],
                marker_color=["#4ade80", "#a3e635", "#fbbf24", "#f97316", "#ff6b6b"]
            ))
            fig2.update_layout(**PLOT_LAYOUT, height=300, title="Mortality % by Age Group")
            st.plotly_chart(fig2, use_container_width=True)

    st.markdown('<div class="section-header">APACHE Score vs Mortality</div>', unsafe_allow_html=True)
    data = fetch_patients(limit=500)
    if data and data.get("patients"):
        df = pd.DataFrame(data["patients"])
        if "apachescore" in df.columns and "hospital_mortality" in df.columns:
            df_plot = df.dropna(subset=["apachescore", "hospital_mortality"])
            df_plot["outcome_label"] = df_plot["hospital_mortality"].map({1: "Expired", 0: "Survived"})
            fig3 = px.box(
                df_plot, x="outcome_label", y="apachescore",
                color="outcome_label",
                color_discrete_map={"Expired": "#ff6b6b", "Survived": "#4ade80"}
            )
            fig3.update_layout(**PLOT_LAYOUT, height=350, showlegend=False)
            st.plotly_chart(fig3, use_container_width=True)

    st.markdown('<div class="section-header">Top Diagnoses</div>', unsafe_allow_html=True)
    diag_data = fetch_top_diagnoses()
    if diag_data:
        df_d = pd.DataFrame(diag_data)
        fig4 = go.Figure(go.Bar(
            x=df_d["count"],
            y=df_d["diagnosisstring"].str[:60],
            orientation="h",
            marker_color="#00d4ff"
        ))
        fig4.update_layout(**PLOT_LAYOUT, height=450, title="Most Common Diagnoses")
        st.plotly_chart(fig4, use_container_width=True)


# ── DIGITAL TWIN ──
elif page == "📡 Digital Twin":
    st.markdown("## 📡 Digital Twin — Vital Forecasting")
    st.info("The digital twin predicts future vital signs using exponential weighted moving average with trend analysis.")

    pid = st.session_state.get("selected_patient", None)
    pid_input = st.number_input("Patient Unit Stay ID", value=pid or 0, step=1, min_value=0)
    steps = st.slider("Forecast Steps (each = 5 min)", min_value=3, max_value=12, value=6)

    if pid_input and st.button("Generate Forecast", type="primary"):
        with st.spinner("Generating digital twin forecast..."):
            vitals_data = fetch_vitals(pid_input)
            forecast_data = api_get(f"/patients/{pid_input}/digital-twin", {"steps": steps})

        if vitals_data and forecast_data and vitals_data.get("vitals"):
            df_v = pd.DataFrame(vitals_data["vitals"])
            df_v["time_h"] = df_v["observationoffset"] / 60
            future_offsets = [o / 60 for o in forecast_data.get("future_offsets", [])]
            forecast = forecast_data.get("forecast", {})

            vital_map = {
                "heartrate": ("Heart Rate (bpm)", "#00d4ff"),
                "sao2": ("SpO₂ (%)", "#4ade80"),
                "respiration": ("Respiration (rpm)", "#fbbf24"),
                "systemicsystolic": ("Systolic BP (mmHg)", "#a78bfa"),
                "temperature": ("Temperature (°C)", "#f97316"),
            }

            for col_name, (label, color) in vital_map.items():
                if col_name not in df_v.columns or col_name not in forecast:
                    continue

                series = df_v[col_name].dropna()
                if len(series) < 3:
                    continue

                fig = go.Figure()

                # Historical - last 50 points
                tail = df_v[col_name].tail(50)
                tail_t = df_v["time_h"].loc[tail.index]

                fig.add_trace(go.Scatter(
                    x=tail_t, y=tail,
                    mode="lines",
                    name="Historical",
                    line=dict(color=color, width=2)
                ))

                # Connect to forecast
                connect_x = [tail_t.iloc[-1]] + future_offsets
                connect_y = [tail.iloc[-1]] + forecast[col_name]

                fig.add_trace(go.Scatter(
                    x=connect_x, y=connect_y,
                    mode="lines+markers",
                    name="Forecast",
                    line=dict(color="#ffffff", width=2, dash="dash"),
                    marker=dict(size=6, color="#ffffff", symbol="diamond")
                ))

                # Confidence band (±1 std of last 10 values)
                std_val = series.tail(10).std()
                upper = [v + std_val for v in forecast[col_name]]
                lower = [v - std_val for v in forecast[col_name]]

                fig.add_trace(go.Scatter(
                    x=future_offsets + future_offsets[::-1],
                    y=upper + lower[::-1],
                    fill="toself",
                    fillcolor=f"rgba(255,255,255,0.05)",
                    line=dict(color="rgba(0,0,0,0)"),
                    name="Confidence Band",
                    showlegend=False
                ))

                fig.add_vline(
                    x=tail_t.iloc[-1], line_dash="dot",
                    line_color="rgba(255,255,255,0.3)",
                    annotation_text="Now"
                )

                fig.update_layout(**PLOT_LAYOUT, height=220, title=f"Digital Twin: {label}",
                                  margin=dict(l=40, r=20, t=40, b=20))
                st.plotly_chart(fig, use_container_width=True)

            st.caption(f"⚡ {steps} steps forecast · each step ≈ 5 minutes · confidence band = ±1 SD of last 10 observations")
        else:
            st.warning("No data available. Make sure patient ID is valid and has vitals.")
