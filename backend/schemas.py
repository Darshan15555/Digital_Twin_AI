from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class StatsResponse(BaseModel):
    total_patients: int = 0
    mortality_rate: float = 0.0
    avg_age: float = 0.0
    avg_apache_score: float = 0.0
    avg_icu_los_hours: float = 0.0
    total_on_vasopressor: int = 0
    total_on_ventilator: int = 0
    total_sepsis_risk: int = 0
    model_ready: bool = False
    error: str | None = None


class PatientsResponse(BaseModel):
    patients: list[dict[str, Any]] = Field(default_factory=list)
    total: int = 0
    error: str | None = None


class GenericListResponse(BaseModel):
    items: list[dict[str, Any]] = Field(default_factory=list)
