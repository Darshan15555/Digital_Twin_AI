from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, HTTPException, Query

from backend.services.patient_service import get_patient_by_id, get_patient_list

router = APIRouter(prefix="/patients", tags=["Patients"])


@router.get("")
def list_patients(
    limit: int = Query(100, le=500, ge=1),
    offset: int = Query(0, ge=0),
    search: Optional[str] = None,
    unit_type: Optional[str] = None,
    high_risk_only: bool = False,
    on_vasopressor: bool = False,
    on_ventilator: bool = False,
    age_min: Optional[float] = None,
    age_max: Optional[float] = None,
):
    result = get_patient_list(limit, offset, search, unit_type, high_risk_only, on_vasopressor, on_ventilator, age_min, age_max)
    if "count" in result and "total" not in result:
        result["total"] = result["count"]
    if "total" in result and "count" not in result:
        result["count"] = result["total"]
    return result


@router.get("/{patient_id}")
def get_patient(patient_id: int):
    result = get_patient_by_id(patient_id)
    if result is None:
        raise HTTPException(status_code=404, detail=f"Patient {patient_id} not found")
    return result

