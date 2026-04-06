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

from backend.dependencies import load_model
from backend.routers import analytics, clinical, patients, predictions, system
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
    log.info("  Models dir: %s", settings.MODELS_DIR)
    for model_path in [settings.MORTALITY_MODEL_PATH, settings.LOS_MODEL_PATH, settings.MORTALITY_FEATURES_PATH, settings.EARLY_MORTALITY_MODEL_PATH]:
        resolved = settings.resolve_path(model_path)
        if resolved.exists():
            load_model(model_path)
            log.info("  ✓ Model loaded: %s", model_path)
        else:
            log.info("  ℹ Model not found (OK): %s", model_path)
    log.info("  Backend ready.")
    log.info("=" * 50)
    yield
    log.info("Backend shutting down.")


app = FastAPI(title="ICU Analytics API", description="Local ICU monitoring and analytics system", version="2.0.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    log.error("Unhandled exception on %s: %s", request.url, exc, exc_info=True)
    return JSONResponse(status_code=200, content={"error": str(exc), "endpoint": str(request.url), "status": "error"})


app.include_router(system.router)
app.include_router(patients.router)
app.include_router(clinical.router)
app.include_router(predictions.router)
app.include_router(analytics.router)


@app.get("/")
def root():
    return {"status": "ok", "system": "ICU Analytics API", "version": "2.0.0", "docs": "/docs"}


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "database": settings.DATABASE_URL,
        "models": {"mortality": settings.resolve_path(settings.MORTALITY_MODEL_PATH).exists(), "los": settings.resolve_path(settings.LOS_MODEL_PATH).exists(), "early_mortality": settings.resolve_path(settings.EARLY_MORTALITY_MODEL_PATH).exists()},
    }


if __name__ == "__main__":
    uvicorn.run("backend.main:app", host=settings.API_HOST, port=settings.API_PORT, reload=settings.API_RELOAD, log_level="info")
