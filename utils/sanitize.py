from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd


def sanitize(val: Any) -> Any:
    if val is None:
        return None
    if isinstance(val, (np.integer,)):
        return int(val)
    if isinstance(val, (np.floating,)):
        val = float(val)
    if isinstance(val, float):
        if math.isnan(val) or math.isinf(val):
            return None
        return val
    if isinstance(val, (np.bool_,)):
        return bool(val)
    if isinstance(val, (np.ndarray,)):
        return val.tolist()
    if isinstance(val, pd.Timestamp):
        return str(val)
    if hasattr(val, "item"):
        try:
            return sanitize(val.item())
        except Exception:
            return val
    return val


def sanitize_row(row: dict) -> dict:
    return {k: sanitize(v) for k, v in row.items()}


def sanitize_df(df: pd.DataFrame) -> list:
    if df is None or df.empty:
        return []
    cleaned = df.copy()
    cleaned = cleaned.replace([np.inf, -np.inf], np.nan)
    cleaned = cleaned.where(cleaned.notna(), None)
    return [sanitize_row(record) for record in cleaned.to_dict(orient="records")]


def safe_float(val: Any, default=None):
    try:
        result = float(val)
        if math.isnan(result) or math.isinf(result):
            return default
        return round(result, 4)
    except (TypeError, ValueError):
        return default


def safe_int(val: Any, default=None):
    try:
        return int(val)
    except (TypeError, ValueError):
        return default

