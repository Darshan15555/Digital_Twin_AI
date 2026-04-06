from __future__ import annotations

from fastapi import APIRouter, Query

from backend.services.prediction_service import get_digital_twin, predict_early_mortality, predict_los, predict_mortality, predict_sepsis

router = APIRouter(prefix="/patients", tags=["Predictions"])


@router.get("/{pid}/predict/mortality")
def mortality(pid: int):
    return predict_mortality(pid)


@router.get("/{pid}/predict/early-mortality")
def early_mortality(pid: int):
    return predict_early_mortality(pid)


@router.get("/{pid}/predict/los")
def los(pid: int):
    return predict_los(pid)


@router.get("/{pid}/predict/sepsis")
def sepsis(pid: int):
    return predict_sepsis(pid)


@router.get("/{pid}/digital-twin")
def digital_twin(pid: int, steps: int = Query(12, ge=3, le=24)):
    return get_digital_twin(pid, steps)

