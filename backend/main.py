from __future__ import annotations

import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.inference import api as inference_api
from backend.inference.schema import PredictRequest
from backend.routers import analytics, clinical, patients, system
from config.settings import settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("backend.log", encoding="utf-8")],
)
log = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("=" * 50)
    log.info("  ICU Analytics Backend Starting")
    log.info("  Database URL: %s", settings.DATABASE_URL)
    log.info("  Backend ready.")
    log.info("=" * 50)
    yield
    log.info("Backend shutting down.")


app = FastAPI(title="ICU Analytics API", description="Local ICU monitoring and analytics system", version="2.0.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    log.error("Unhandled exception on %s: %s", request.url, exc, exc_info=True)
    return JSONResponse(status_code=500, content={"error": str(exc), "endpoint": str(request.url), "status": "error"})


app.include_router(system.router)
app.include_router(patients.router)
app.include_router(clinical.router)
app.include_router(analytics.router)
app.include_router(inference_api.router)


@app.get("/")
def root():
    return {"status": "ok", "system": "ICU Analytics API", "version": "2.0.0", "docs": "/docs"}


@app.get("/health")
def health():
    model_ready = False
    prediction_ready = False
    model_name = "unknown"
    reason = "API reachable"
    try:
        model_info = inference_api.predictor.model_info()
        model_ready = bool(model_info.get("model_ready"))
        model_name = str(model_info.get("model_name", "unknown"))
        if model_ready:
            probe_payload = PredictRequest(hr=90.0, spo2=97.0, bp_sys=120.0, bp_dia=80.0)
            probe_result = inference_api.predict(probe_payload)
            prediction_ready = isinstance(getattr(probe_result, "prediction", None), int)
            if prediction_ready:
                reason = "All systems operational"
            else:
                reason = "Prediction service is not ready"
        else:
            reason = "Model artifact is not ready"
    except Exception:
        reason = "Prediction service is not ready"

    status = "online" if prediction_ready else "partial"
    return {
        "status": status,
        "api_ready": True,
        "model_ready": model_ready,
        "prediction_ready": prediction_ready,
        "model_name": model_name,
        "reason": reason,
        "database": settings.DATABASE_URL,
    }


if __name__ == "__main__":
    uvicorn.run("backend.main:app", host=settings.API_HOST, port=settings.API_PORT, reload=settings.API_RELOAD, log_level="info")
