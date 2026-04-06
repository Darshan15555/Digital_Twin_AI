from __future__ import annotations

import sys
from pathlib import Path

import uvicorn

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from backend.main import app
from config.settings import settings


if __name__ == "__main__":
    uvicorn.run("backend.main:app", host=settings.API_HOST, port=settings.API_PORT, reload=settings.API_RELOAD, log_level="info")
