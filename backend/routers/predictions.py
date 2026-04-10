from __future__ import annotations

from fastapi import APIRouter

from backend.schemas import DeteriorationPredictionRequest, PredictionRequest
from backend.services.recent_prediction_service import predict_from_recent_history
from backend.services.prediction_service import get_model_info, predict_batch, predict_deterioration, predict_mortality

router = APIRouter(prefix="/predictions", tags=["Predictions"])


@router.get("/model-info")
def model_info():
    return get_model_info()


@router.post("/mortality")
def mortality_prediction(payload: PredictionRequest):
    return predict_mortality(payload.patient_data, threshold_type=payload.threshold_type)


@router.post("/deterioration")
def deterioration_prediction(payload: DeteriorationPredictionRequest):
    return predict_deterioration(payload.patient_data, window_id=payload.window_id, threshold_type=payload.threshold_type)


@router.post("/batch")
def batch_prediction(payload: list[dict]):
    return predict_batch(payload)


@router.get("/patients/{patient_id}/recent")
def recent_patient_prediction(patient_id: int, hours_back: int = 8, threshold_type: str = "max_f1"):
    return predict_from_recent_history(patient_id, hours_back=hours_back, threshold_type=threshold_type)
