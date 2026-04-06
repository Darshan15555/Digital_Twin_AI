from __future__ import annotations

import logging
import pickle
import sqlite3
from functools import lru_cache
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine, text

from config.settings import settings

log = logging.getLogger(__name__)

IS_POSTGRES = settings.DATABASE_URL.startswith("postgresql")
pg_engine = create_engine(settings.DATABASE_URL, future=True, pool_pre_ping=True) if IS_POSTGRES else None


def get_db() -> sqlite3.Connection:
    db_path = settings.db_path
    if not db_path.exists():
        raise RuntimeError(f"Database not found at {db_path}. Run setup.py first.")
    conn = sqlite3.connect(str(db_path), check_same_thread=False, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA cache_size=10000")
    return conn


def query_df(sql: str, params=()) -> pd.DataFrame:
    if IS_POSTGRES and pg_engine is not None:
        try:
            with pg_engine.connect() as conn:
                result = conn.execute(text(sql), params if isinstance(params, dict) else {})
                return pd.DataFrame(result.fetchall(), columns=result.keys())
        except Exception as exc:
            log.error("PostgreSQL query failed: %s\nSQL: %s", exc, sql, exc_info=True)
            return pd.DataFrame()
    conn = get_db()
    try:
        return pd.read_sql_query(sql, conn, params=params)
    except Exception as exc:
        log.error("SQLite query failed: %s\nSQL: %s", exc, sql, exc_info=True)
        return pd.DataFrame()
    finally:
        conn.close()


def execute_sql(sql: str, params=()) -> bool:
    if IS_POSTGRES and pg_engine is not None:
        try:
            with pg_engine.begin() as conn:
                conn.execute(text(sql), params if isinstance(params, dict) else {})
            return True
        except Exception as exc:
            log.error("PostgreSQL execute failed: %s\nSQL: %s", exc, sql, exc_info=True)
            return False
    conn = get_db()
    try:
        conn.execute(sql, params)
        conn.commit()
        return True
    except Exception as exc:
        log.error("SQLite execute failed: %s\nSQL: %s", exc, sql, exc_info=True)
        return False
    finally:
        conn.close()


@lru_cache(maxsize=8)
def load_model(model_path: str):
    path = settings.resolve_path(model_path)
    if not path.exists():
        log.warning("Model not found: %s", path)
        return None
    try:
        with open(path, "rb") as handle:
            return pickle.load(handle)
    except Exception as exc:
        log.error("Failed to load model %s: %s", path, exc, exc_info=True)
        return None


def get_mortality_model():
    return load_model(settings.MORTALITY_MODEL_PATH)


def get_los_model():
    return load_model(settings.LOS_MODEL_PATH)


def get_mortality_features():
    return load_model(settings.MORTALITY_FEATURES_PATH)


def get_mortality_scaler():
    return load_model(settings.MORTALITY_SCALER_PATH)


def get_early_mortality_model():
    return load_model(settings.EARLY_MORTALITY_MODEL_PATH)


def get_early_mortality_features():
    return load_model(settings.EARLY_MORTALITY_FEATURES_PATH)


def get_early_mortality_imputer():
    return load_model(settings.EARLY_MORTALITY_IMPUTER_PATH)


def table_exists(table_name: str, schema: str | None = None) -> bool:
    try:
        if IS_POSTGRES:
            result = query_df(
                """
                SELECT table_name
                FROM information_schema.tables
                WHERE table_name = :table_name
                  AND (:schema IS NULL OR table_schema = :schema)
                """,
                {"table_name": table_name, "schema": schema},
            )
            return not result.empty
        result = query_df("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table_name,))
        return not result.empty
    except Exception:
        return False


def get_table_columns(table_name: str, schema: str | None = None) -> list[str]:
    try:
        if IS_POSTGRES:
            df = query_df(
                """
                SELECT column_name
                FROM information_schema.columns
                WHERE table_name = :table_name
                  AND (:schema IS NULL OR table_schema = :schema)
                ORDER BY ordinal_position
                """,
                {"table_name": table_name, "schema": schema},
            )
            return df["column_name"].tolist() if not df.empty else []
        df = query_df(f"PRAGMA table_info({table_name})")
        return df["name"].tolist() if not df.empty else []
    except Exception:
        return []
