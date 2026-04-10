from __future__ import annotations

import logging
from typing import Any

from models.inference.predictor import ICUPredictor

log = logging.getLogger(__name__)

_predictor: ICUPredictor | None = None


def get_predictor() -> ICUPredictor:
    global _predictor
    if _predictor is None:
        _predictor = ICUPredictor()
    return _predictor


def get_model_info() -> dict[str, Any]:
    try:
        return get_predictor().get_model_info()
    except Exception as exc:
        log.error("get_model_info error: %s", exc, exc_info=True)
        return {"error": str(exc)}


def predict_mortality(patient_data: dict[str, Any], threshold_type: str = "max_f1") -> dict[str, Any]:
    try:
        return get_predictor().predict_mortality(patient_data, threshold_type=threshold_type)
    except Exception as exc:
        log.error("predict_mortality service error: %s", exc, exc_info=True)
        return {"error": str(exc), "model_ready": False}


def predict_deterioration(patient_data: dict[str, Any], window_id: int, threshold_type: str = "max_f1") -> dict[str, Any]:
    try:
        return get_predictor().predict_deterioration(patient_data, window_id=window_id, threshold_type=threshold_type)
    except Exception as exc:
        log.error("predict_deterioration service error: %s", exc, exc_info=True)
        return {"error": str(exc), "model_ready": False}


def predict_batch(patient_windows: list[dict[str, Any]]) -> dict[str, Any]:
    try:
        return get_predictor().predict_batch(patient_windows)
    except Exception as exc:
        log.error("predict_batch service error: %s", exc, exc_info=True)
        return {"error": str(exc), "model_ready": False}
