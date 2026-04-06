from __future__ import annotations

from fastapi import APIRouter, Query

from backend.services.clinical_service import get_comorbidities, get_diagnosis, get_fluid_balance, get_labs, get_medications, get_treatments, get_ventilation, get_vitals

router = APIRouter(prefix="/patients", tags=["Clinical"])


@router.get("/{pid}/vitals")
def vitals(pid: int, max_points: int = Query(400, le=1000)):
    return get_vitals(pid, max_points)


@router.get("/{pid}/labs")
def labs(pid: int):
    return get_labs(pid)


@router.get("/{pid}/diagnosis")
def diagnosis(pid: int):
    return get_diagnosis(pid)


@router.get("/{pid}/treatments")
def treatments(pid: int):
    return get_treatments(pid)


@router.get("/{pid}/medications")
def medications(pid: int):
    return get_medications(pid)


@router.get("/{pid}/fluid-balance")
def fluid_balance(pid: int):
    return get_fluid_balance(pid)


@router.get("/{pid}/ventilation")
def ventilation(pid: int):
    return get_ventilation(pid)


@router.get("/{pid}/comorbidities")
def comorbidities(pid: int):
    return get_comorbidities(pid)

