from __future__ import annotations

from typing import Any

import requests
import streamlit as st

from config.config import API_HOST, API_PORT

BASE_URL = f"http://{API_HOST}:{API_PORT}"
TIMEOUT_SECONDS = 8


def _warn(message: str) -> None:
    st.sidebar.warning(message)


def _request(path: str, params: dict | None = None) -> Any:
    try:
        response = requests.get(f"{BASE_URL}{path}", params=params, timeout=TIMEOUT_SECONDS)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.ConnectionError:
        _warn("API offline: unable to connect to backend")
        return None
    except requests.exceptions.Timeout:
        _warn("API timeout: backend took too long to respond")
        return None
    except requests.exceptions.RequestException as exc:
        _warn(f"API request failed: {exc}")
        return None


def _post(path: str, payload: dict | list) -> Any:
    try:
        response = requests.post(f"{BASE_URL}{path}", json=payload, timeout=TIMEOUT_SECONDS)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.ConnectionError:
        _warn("API offline: unable to connect to backend")
        return None
    except requests.exceptions.Timeout:
        _warn("API timeout: backend took too long to respond")
        return None
    except requests.exceptions.RequestException as exc:
        _warn(f"API request failed: {exc}")
        return None


@st.cache_data(ttl=120)
def fetch_stats() -> dict | None:
    return _request("/stats")


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


@st.cache_data(ttl=120)
def fetch_model_info() -> dict | None:
    return _request("/predictions/model-info")


def predict_mortality(patient_data: dict, threshold_type: str = "max_f1") -> dict | None:
    return _post("/predictions/mortality", {"patient_data": patient_data, "threshold_type": threshold_type})


def predict_deterioration(patient_data: dict, window_id: int, threshold_type: str = "max_f1") -> dict | None:
    return _post(
        "/predictions/deterioration",
        {"patient_data": patient_data, "window_id": window_id, "threshold_type": threshold_type},
    )


def predict_batch(patient_windows: list[dict]) -> dict | None:
    return _post("/predictions/batch", patient_windows)


def predict_recent_patient(patient_id: int, hours_back: int = 8, threshold_type: str = "max_f1") -> dict | None:
    return _request(
        f"/predictions/patients/{patient_id}/recent",
        {"hours_back": hours_back, "threshold_type": threshold_type},
    )
