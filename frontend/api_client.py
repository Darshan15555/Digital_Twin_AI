from __future__ import annotations

from typing import Any

import requests
import streamlit as st

from config.config import API_HOST, API_PORT

BASE_URL = f"http://{API_HOST}:{API_PORT}"
TIMEOUT_SECONDS = 8


def _parse_json_response(response: requests.Response) -> Any:
    try:
        return response.json()
    except ValueError:
        return None


def _sanitize_error_text(text: str | None) -> str:
    if not text:
        return ""
    cleaned = str(text).strip()
    if "Traceback" in cleaned or "File \"" in cleaned:
        return ""
    if len(cleaned) > 280:
        cleaned = cleaned[:277].rstrip() + "..."
    return cleaned


def _extract_backend_detail(payload: Any) -> str:
    if isinstance(payload, dict):
        detail = payload.get("detail")
        if isinstance(detail, list):
            messages: list[str] = []
            for item in detail[:3]:
                if isinstance(item, dict):
                    msg = str(item.get("msg", "")).strip()
                    loc = item.get("loc", [])
                    field = str(loc[-1]) if isinstance(loc, list) and loc else ""
                    if msg and field:
                        messages.append(f"{field}: {msg}")
                    elif msg:
                        messages.append(msg)
            return "; ".join(messages)
        if isinstance(detail, str):
            return detail
        for key in ("error", "message"):
            value = payload.get(key)
            if isinstance(value, str):
                return value
    elif isinstance(payload, str):
        return payload
    return ""


def _build_error_payload(status_code: int | None, detail: str, fallback_message: str) -> dict[str, Any]:
    safe_detail = _sanitize_error_text(detail)
    if safe_detail and "Model unavailable" in safe_detail:
        message = "Prediction service is temporarily unavailable. Please try again shortly."
    elif status_code in (400, 422):
        message = safe_detail or "Some input values are invalid. Please review and try again."
    elif status_code == 404:
        message = "Requested data was not found."
    elif status_code is not None and status_code >= 500:
        message = "Server error occurred. Please try again."
    else:
        message = safe_detail or fallback_message

    return {
        "ok": False,
        "status_code": status_code,
        "error": "request_failed",
        "message": message,
    }


def _request(path: str, params: dict | None = None, *, include_error: bool = False) -> Any:
    try:
        response = requests.get(f"{BASE_URL}{path}", params=params, timeout=TIMEOUT_SECONDS)
        payload = _parse_json_response(response)
        if response.ok:
            return payload
        if not include_error:
            return None
        detail = _extract_backend_detail(payload)
        return _build_error_payload(response.status_code, detail, "Request failed. Please try again.")
    except requests.exceptions.ConnectionError:
        if include_error:
            return _build_error_payload(None, "", "Cannot connect to server. Please check backend status.")
        return None
    except requests.exceptions.Timeout:
        if include_error:
            return _build_error_payload(None, "", "Server is taking too long to respond. Please try again.")
        return None
    except requests.exceptions.RequestException as exc:
        if include_error:
            return _build_error_payload(None, _sanitize_error_text(str(exc)), "Request failed. Please try again.")
        return None


def _post(path: str, payload: dict | list, *, include_error: bool = False) -> Any:
    try:
        response = requests.post(f"{BASE_URL}{path}", json=payload, timeout=TIMEOUT_SECONDS)
        body = _parse_json_response(response)
        if response.ok:
            return body
        if not include_error:
            return None
        detail = _extract_backend_detail(body)
        return _build_error_payload(response.status_code, detail, "Request failed. Please try again.")
    except requests.exceptions.ConnectionError:
        if include_error:
            return _build_error_payload(None, "", "Cannot connect to server. Please check backend status.")
        return None
    except requests.exceptions.Timeout:
        if include_error:
            return _build_error_payload(None, "", "Server is taking too long to respond. Please try again.")
        return None
    except requests.exceptions.RequestException as exc:
        if include_error:
            return _build_error_payload(None, _sanitize_error_text(str(exc)), "Request failed. Please try again.")
        return None


@st.cache_data(ttl=120)
def fetch_stats() -> dict | None:
    return _request("/stats")


@st.cache_data(ttl=15)
def fetch_system_health() -> dict | None:
    return _request("/health")


@st.cache_data(ttl=30)
def fetch_patients(
    limit: int = 100,
    offset: int = 0,
    search: str | None = None,
    unit_type: str | None = None,
    high_risk: bool = False,
    on_vasopressor: bool | None = None,
    on_ventilator: bool | None = None,
    age_min: int | None = None,
    age_max: int | None = None,
) -> dict | None:
    return _request(
        "/patients",
        {
            "limit": limit,
            "offset": offset,
            "search": search,
            "unit": unit_type,
            "high_risk_only": high_risk,
            "on_vasopressor": on_vasopressor,
            "on_ventilator": on_ventilator,
            "age_min": age_min,
            "age_max": age_max,
        },
    )


@st.cache_data(ttl=60)
def fetch_patient(pid: int) -> dict | None:
    return _request(f"/patients/{pid}")


@st.cache_data(ttl=60)
def fetch_vitals(pid: int, max_points: int = 300) -> dict | None:
    payload = _request(f"/patients/{pid}/vitals")
    if not payload:
        return None
    periodic = payload.get("periodic", [])
    aperiodic = payload.get("aperiodic", [])
    if len(periodic) > max_points:
        step = max(1, len(periodic) // max_points)
        periodic = periodic[::step]
    return {"periodic": periodic, "aperiodic": aperiodic}


@st.cache_data(ttl=60)
def fetch_labs(pid: int) -> dict | None:
    return _request(f"/patients/{pid}/labs")


@st.cache_data(ttl=60)
def fetch_diagnosis(pid: int) -> dict | None:
    return _request(f"/patients/{pid}/diagnosis")


@st.cache_data(ttl=60)
def fetch_treatments(pid: int) -> dict | None:
    return _request(f"/patients/{pid}/treatments")


@st.cache_data(ttl=60)
def fetch_medications(pid: int) -> dict | None:
    return _request(f"/patients/{pid}/medications")


@st.cache_data(ttl=60)
def fetch_fluid_balance(pid: int) -> dict | None:
    return _request(f"/patients/{pid}/fluid-balance")


@st.cache_data(ttl=60)
def fetch_ventilation(pid: int) -> dict | None:
    return _request(f"/patients/{pid}/ventilation")


@st.cache_data(ttl=60)
def fetch_comorbidities(pid: int) -> dict | None:
    return _request(f"/patients/{pid}/comorbidities")


@st.cache_data(ttl=120)
def fetch_analytics_units() -> list | None:
    return _request("/analytics/unit-breakdown")


@st.cache_data(ttl=120)
def fetch_analytics_age() -> list | None:
    return _request("/analytics/mortality-by-age")


@st.cache_data(ttl=120)
def fetch_analytics_diagnoses() -> list | None:
    return _request("/analytics/top-diagnoses")


@st.cache_data(ttl=120)
def fetch_analytics_vasopressor() -> list | None:
    return _request("/analytics/vasopressor-usage")


@st.cache_data(ttl=120)
def fetch_analytics_ventilator() -> list | None:
    return _request("/analytics/ventilator-usage")


@st.cache_data(ttl=120)
def fetch_analytics_fluid() -> list | None:
    return _request("/analytics/fluid-balance")


def predict_vitals(
    *,
    hr: float,
    spo2: float,
    bp_sys: float,
    bp_dia: float,
    threshold: float | None = None,
) -> dict | None:
    payload: dict[str, Any] = {
        "hr": float(hr),
        "spo2": float(spo2),
        "bp_sys": float(bp_sys),
        "bp_dia": float(bp_dia),
    }
    if threshold is not None:
        payload["threshold"] = float(threshold)
    return _post("/predict", payload, include_error=True)


def predict_vitals_batch(items: list[dict[str, float]], threshold: float | None = None) -> dict | None:
    payload: dict[str, Any] = {"items": items}
    if threshold is not None:
        payload["threshold"] = float(threshold)
    return _post("/predict/batch", payload, include_error=True)
