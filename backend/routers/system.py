from __future__ import annotations

from fastapi import APIRouter

from backend.services.patient_service import get_system_stats

router = APIRouter(tags=["System"])


@router.get("/stats")
def get_stats():
    return get_system_stats()

