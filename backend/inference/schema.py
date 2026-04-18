from __future__ import annotations

from pydantic import BaseModel, Field, model_validator


class PredictRequest(BaseModel):
    hr: float = Field(..., ge=30, le=220)
    spo2: float = Field(..., ge=70, le=100)
    bp_sys: float = Field(..., ge=40, le=300)
    bp_dia: float = Field(..., ge=20, le=220)
    bp_mean: float | None = Field(default=None, ge=20, le=250)
    threshold: float | None = Field(default=None, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_payload(self) -> "PredictRequest":
        if self.bp_sys <= self.bp_dia:
            raise ValueError("bp_sys must be greater than bp_dia")
        return self


class PredictResponse(BaseModel):
    prediction: int
    probability: float
    risk_level: str
    threshold_used: float
    model_name: str


class BatchPredictRequest(BaseModel):
    items: list[PredictRequest] = Field(default_factory=list, min_length=1, max_length=1000)
    threshold: float | None = Field(default=None, ge=0.0, le=1.0)


class BatchPredictResponse(BaseModel):
    count: int
    predictions: list[PredictResponse]
