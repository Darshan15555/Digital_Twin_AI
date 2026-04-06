from __future__ import annotations

from fastapi import APIRouter, Query

from backend.services.analytics_service import get_fluid_balance_by_outcome, get_mortality_by_age, get_top_diagnoses, get_unit_breakdown, get_vasopressor_usage, get_ventilator_usage

router = APIRouter(prefix="/analytics", tags=["Analytics"])


@router.get("/unit-breakdown")
def unit_breakdown():
    return get_unit_breakdown()


@router.get("/mortality-by-age")
def mortality_age():
    return get_mortality_by_age()


@router.get("/top-diagnoses")
def top_diagnoses(limit: int = Query(20, le=50)):
    return get_top_diagnoses(limit)


@router.get("/vasopressor-usage")
def vasopressor():
    return get_vasopressor_usage()


@router.get("/ventilator-usage")
def ventilator():
    return get_ventilator_usage()


@router.get("/fluid-balance")
def fluid_balance():
    return get_fluid_balance_by_outcome()

