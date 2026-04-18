from __future__ import annotations

import streamlit as st


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500&family=Rajdhani:wght@400;600;700&display=swap');
        :root {
          --bg-deep: #070B14;
          --bg-surface: #0F1724;
          --bg-elevated: #16213A;
          --border: rgba(0, 200, 255, 0.12);
          --accent-cyan: #00C8FF;
          --accent-green: #00E5A0;
          --accent-amber: #FFB830;
          --accent-red: #FF4560;
          --accent-violet: #A78BFA;
          --text-primary: #E8F4FD;
          --text-muted: #5A7A99;
          --text-dim: #2D4A66;
        }
        .stApp {
          background: var(--bg-deep);
          color: var(--text-primary);
          background-image: radial-gradient(circle, rgba(0,200,255,0.04) 1px, transparent 1px);
          background-size: 28px 28px;
        }
        * { font-family: 'Inter', sans-serif; }
        h1, h2, h3, .section-header, .sidebar-logo { font-family: 'Rajdhani', sans-serif !important; }
        code, pre, .mono-text { font-family: 'JetBrains Mono', monospace !important; }
        [data-testid="stSidebar"] { background: var(--bg-surface); border-right: 1px solid var(--border); }
        [data-testid="stSidebar"] * { color: var(--text-primary); }
        .sidebar-logo { color: var(--accent-cyan); font-size: 1.4rem; font-weight: 700; letter-spacing: 0.08em; }
        .sidebar-subtitle { color: var(--text-muted); font-family: 'JetBrains Mono', monospace; font-size: 0.72rem; }
        .status-dot { display:inline-block; width:10px; height:10px; border-radius:50%; margin-right:8px; animation:pulse 1.8s infinite; }
        .status-online { background: var(--accent-green); box-shadow: 0 0 10px rgba(0,229,160,0.6); }
        .status-partial { background: var(--accent-amber); box-shadow: 0 0 10px rgba(255,184,48,0.6); }
        .status-offline { background: var(--accent-red); box-shadow: 0 0 10px rgba(255,69,96,0.6); }
        @keyframes pulse { 0% { transform: scale(1); opacity: 0.8; } 50% { transform: scale(1.2); opacity: 1; } 100% { transform: scale(1); opacity: 0.8; } }
        .metric-card { background: var(--bg-surface); border: 1px solid var(--border); border-radius: 12px; padding: 14px 16px 12px 16px; position: relative; overflow: hidden; min-height: 122px; transition: transform 0.2s ease; }
        .metric-card:hover { transform: translateY(-2px); }
        .metric-card::before { content: ""; position: absolute; top: 0; left: 0; right: 0; height: 2px; background: linear-gradient(90deg, #00C8FF, #A78BFA); }
        .metric-label { font-size: 0.72rem; color: var(--text-muted); text-transform: uppercase; letter-spacing: 1.8px; }
        .metric-value { font-family: 'Rajdhani', sans-serif; font-size: 2.2rem; line-height: 1.1; color: var(--accent-cyan); margin-top: 8px; }
        .metric-subtext { font-size: 0.8rem; color: var(--text-muted); margin-top: 8px; }
        .risk-badge { display:inline-block; padding:5px 14px; border-radius:20px; font-family:'JetBrains Mono', monospace; font-size:0.82rem; border:1px solid transparent; }
        .risk-critical, .risk-high { background: rgba(255,69,96,0.15); border-color: #FF4560; color: #FF4560; box-shadow: 0 0 8px rgba(255,69,96,0.3); }
        .risk-moderate { background: rgba(255,184,48,0.08); border-color: rgba(255,184,48,0.5); color: #FFB830; }
        .risk-low { background: rgba(0,229,160,0.15); border-color: #00E5A0; color: #00E5A0; }
        .section-header { font-size: 0.86rem; text-transform: uppercase; letter-spacing: 3px; color: var(--accent-cyan); border-bottom: 1px solid rgba(0,200,255,0.3); padding-bottom: 6px; margin: 12px 0 16px 0; }
        .alert-banner { width: 100%; border-left: 4px solid; border-radius: 10px; padding: 12px 14px; margin: 8px 0; background: rgba(255,255,255,0.03); }
        .alert-critical { border-color: var(--accent-red); background: rgba(255,69,96,0.08); }
        .alert-warning { border-color: var(--accent-amber); background: rgba(255,184,48,0.08); }
        .alert-info { border-color: var(--accent-cyan); background: rgba(0,200,255,0.08); }
        .comorbidity-tag { display:inline-block; padding:6px 12px; margin:4px 6px 4px 0; border-radius:999px; border:1px solid rgba(255,255,255,0.08); background: rgba(255,255,255,0.03); color: var(--text-muted); font-size: 0.84rem; }
        .comorbidity-tag.active { color: var(--accent-cyan); border-color: rgba(0,200,255,0.35); background: rgba(0,200,255,0.08); }
        .patient-header-card, .vital-card { background: var(--bg-surface); border: 1px solid var(--border); border-radius: 14px; padding: 16px; }
        .patient-header-title { color: var(--accent-cyan); font-family: 'Rajdhani', sans-serif; font-size: 1.4rem; font-weight: 700; }
        .vital-card { min-height: 112px; border-top: 3px solid var(--accent-cyan); }
        .vital-title { color: var(--text-muted); text-transform: uppercase; letter-spacing: 1.5px; font-size: 0.72rem; }
        .vital-value { font-family: 'JetBrains Mono', monospace; font-size: 1.8rem; color: var(--text-primary); margin: 8px 0 4px 0; }
        .table-caption { color: var(--text-muted); font-family: 'JetBrains Mono', monospace; font-size: 0.8rem; margin-bottom: 8px; }
        div[data-testid="stDataFrame"] { border: 1px solid var(--border); border-radius: 12px; overflow: hidden; }
        </style>
        """,
        unsafe_allow_html=True,
    )
