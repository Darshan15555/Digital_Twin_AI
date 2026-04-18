from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException

from backend.inference.predictor import VitalPredictor
from backend.inference.schema import (
    BatchPredictRequest,
    BatchPredictResponse,
    PredictRequest,
    PredictResponse,
)


log = logging.getLogger(__name__)
router = APIRouter(tags=["Inference"])
predictor = VitalPredictor()


@router.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest) -> PredictResponse:
    try:
        result = predictor.predict(payload.model_dump(exclude_none=True), threshold=payload.threshold)
        return PredictResponse(**result)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=f"Model unavailable: {exc}") from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        log.error("Prediction failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc


@router.post("/predict/batch", response_model=BatchPredictResponse)
def predict_batch(payload: BatchPredictRequest) -> BatchPredictResponse:
    try:
        items = [item.model_dump(exclude_none=True) for item in payload.items]
        result = predictor.predict_batch(items, threshold=payload.threshold)
        return BatchPredictResponse(count=len(result), predictions=result)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=f"Model unavailable: {exc}") from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        log.error("Batch prediction failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {exc}") from exc
