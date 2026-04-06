from __future__ import annotations

from html import escape


def metric_card(value: str, label: str, subtext: str = "", color: str = "cyan") -> str:
    color_map = {"cyan": "#00C8FF", "green": "#00E5A0", "amber": "#FFB830", "red": "#FF4560", "violet": "#A78BFA"}
    return f"""
    <div class="metric-card">
      <div class="metric-label">{escape(label)}</div>
      <div class="metric-value" style="color:{color_map.get(color, '#00C8FF')}">{escape(str(value))}</div>
      <div class="metric-subtext">{escape(subtext)}</div>
    </div>
    """


def risk_badge(label: str) -> str:
    normalized = (label or "LOW").upper()
    css_class = {"CRITICAL": "risk-critical", "HIGH": "risk-high", "MODERATE": "risk-moderate", "LOW": "risk-low"}.get(normalized, "risk-low")
    return f'<span class="risk-badge {css_class}">{escape(normalized)}</span>'


def vital_status_dot(value: float | int | None, low_crit: float, low_warn: float, high_warn: float, high_crit: float) -> str:
    if value is None:
        return "⚪ Unknown"
    if value <= low_crit or value >= high_crit:
        return "🔴 Critical"
    if value <= low_warn or value >= high_warn:
        return "🟠 Warning"
    return "🟢 Stable"


def comorbidity_tags(comorbidities_dict: dict) -> str:
    labels = {
        "has_diabetes": "Diabetes",
        "has_chf": "CHF",
        "has_copd": "COPD",
        "has_ckd": "CKD",
        "has_hypertension": "Hypertension",
        "has_immunosuppression": "Immunosuppression",
    }
    return "".join(
        f'<span class="comorbidity-tag {"active" if bool(comorbidities_dict.get(key)) else ""}">{escape(label)}</span>'
        for key, label in labels.items()
    )


def patient_header_card(patient_dict: dict) -> str:
    return f"""
    <div class="patient-header-card">
      <div class="patient-header-title">Patient {escape(str(patient_dict.get("patientunitstayid", "Unknown")))}</div>
      <div style="display:grid;grid-template-columns:2fr 1fr;gap:18px;margin-top:12px;">
        <div>
          <div><strong>Age:</strong> {escape(str(patient_dict.get("age", patient_dict.get("age_years", "Unknown"))))}</div>
          <div><strong>Gender:</strong> {escape(str(patient_dict.get("gender", "Unknown")))}</div>
          <div><strong>Ethnicity:</strong> {escape(str(patient_dict.get("ethnicity", "Unknown")))}</div>
          <div><strong>Unit:</strong> {escape(str(patient_dict.get("unittype", "Unknown")))}</div>
        </div>
        <div>
          <div><strong>APACHE:</strong> {escape(str(patient_dict.get("apachescore", "—")))}</div>
          <div><strong>ICU LOS:</strong> {escape(str(patient_dict.get("icu_los_hours", "—")))}</div>
          <div><strong>Outcome:</strong> {escape("Expired" if patient_dict.get("hospital_mortality") == 1 else "Survived")}</div>
        </div>
      </div>
      <div style="margin-top:14px;">{comorbidity_tags(patient_dict)}</div>
    </div>
    """


def alert_banner(message: str, level: str = "warning") -> str:
    css_class = {"critical": "alert-critical", "warning": "alert-warning", "info": "alert-info"}.get(level, "alert-info")
    icon = {"critical": "ALERT", "warning": "NOTICE", "info": "INFO"}.get(level, "INFO")
    return f'<div class="alert-banner {css_class}"><strong>{icon}:</strong> {escape(message)}</div>'
