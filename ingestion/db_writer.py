from __future__ import annotations

import logging
from contextlib import contextmanager

import pandas as pd
import psycopg
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.engine import Engine, make_url

from ingestion.config import PipelineConfig


log = logging.getLogger(__name__)


class PostgresCopyWriter:
    """Writes transformed chunks to PostgreSQL using schema management and COPY bulk load."""

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self.conninfo = self._normalize_conninfo(config.database_url)
        self.engine: Engine = create_engine(config.database_url, future=True, pool_pre_ping=True)

    def ensure_schema(self) -> None:
        with self.engine.begin() as connection:
            connection.execute(text(f'CREATE SCHEMA IF NOT EXISTS "{self.config.schema_name}"'))
        log.info(
            "Database target ready | database_url=%s schema=%s",
            self._safe_database_url(),
            self.config.schema_name,
        )

    def ensure_table(self, table_name: str, frame: pd.DataFrame) -> None:
        if frame.empty:
            return

        inspector = inspect(self.engine)
        table_exists = inspector.has_table(table_name, schema=self.config.schema_name)
        existing_columns = (
            {
                column["name"]
                for column in inspector.get_columns(table_name, schema=self.config.schema_name)
            }
            if table_exists
            else set()
        )

        with self.engine.begin() as connection:
            if not existing_columns:
                ddl = ", ".join(
                    f'"{column}" {self._postgres_type(frame[column])}'
                    for column in frame.columns
                )
                connection.execute(
                    text(f'CREATE TABLE IF NOT EXISTS "{self.config.schema_name}"."{table_name}" ({ddl})')
                )
                self._create_indexes(connection, table_name, frame.columns.tolist())
                log.info(
                    "Created table | schema=%s table=%s columns=%s",
                    self.config.schema_name,
                    table_name,
                    ", ".join(frame.columns),
                )
                return

            for column in frame.columns:
                if column not in existing_columns:
                    connection.execute(
                        text(
                            f'ALTER TABLE "{self.config.schema_name}"."{table_name}" '
                            f'ADD COLUMN "{column}" {self._postgres_type(frame[column])}'
                        )
                    )
                    log.info(
                        "Added column | schema=%s table=%s column=%s",
                        self.config.schema_name,
                        table_name,
                        column,
                    )

    def copy_rows(self, table_name: str, frame: pd.DataFrame) -> int:
        if frame.empty:
            return 0

        prepared = self.prepare_for_copy(frame)
        column_list = ", ".join(f'"{column}"' for column in frame.columns)
        copy_sql = (
            f'COPY "{self.config.schema_name}"."{table_name}" ({column_list}) '
            "FROM STDIN"
        )

        batch_size = max(1, self.config.insert_batch_size)
        rows_written = 0
        with self.connection() as connection:
            with connection.cursor() as cursor:
                values = prepared.itertuples(index=False, name=None)
                while True:
                    batch: list[tuple[object, ...]] = []
                    for _ in range(batch_size):
                        try:
                            batch.append(next(values))
                        except StopIteration:
                            break

                    if not batch:
                        break

                    with cursor.copy(copy_sql) as copy:
                        for row in batch:
                            copy.write_row(self._normalize_row(row))
                    rows_written += len(batch)
            connection.commit()
        log.info(
            "COPY success | schema=%s table=%s rows=%s",
            self.config.schema_name,
            table_name,
            f"{rows_written:,}",
        )
        return rows_written

    def prepare_for_copy(self, frame: pd.DataFrame) -> pd.DataFrame:
        prepared = frame.copy()
        object_columns = prepared.select_dtypes(include=["object", "string"]).columns
        for column in object_columns:
            prepared.loc[:, column] = prepared[column].astype("string").str.strip()
            prepared.loc[:, column] = prepared[column].replace(
                {"": pd.NA, "nan": pd.NA, "None": pd.NA, "<NA>": pd.NA}
            )
        return prepared

    @contextmanager
    def connection(self):
        connection = psycopg.connect(self.conninfo)
        try:
            yield connection
        finally:
            connection.close()

    def count_rows(self, table_name: str) -> int:
        with self.connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    f'SELECT COUNT(*) FROM "{self.config.schema_name}"."{table_name}"'
                )
                row = cursor.fetchone()
        return int(row[0]) if row else 0

    def _create_indexes(self, connection, table_name: str, columns: list[str]) -> None:
        for column in ("patientunitstayid", "patienthealthsystemstayid", "hospitalid"):
            if column in columns:
                connection.execute(
                    text(
                        f'CREATE INDEX IF NOT EXISTS "idx_{table_name}_{column}" '
                        f'ON "{self.config.schema_name}"."{table_name}" ("{column}")'
                    )
                )

    @staticmethod
    def _normalize_row(row: tuple[object, ...]) -> tuple[object, ...]:
        normalized: list[object] = []
        for value in row:
            if pd.isna(value):
                normalized.append(None)
            elif hasattr(value, "item"):
                normalized.append(value.item())
            else:
                normalized.append(value)
        return tuple(normalized)

    @staticmethod
    def _postgres_type(series: pd.Series) -> str:
        if pd.api.types.is_integer_dtype(series.dtype):
            return "BIGINT"
        if pd.api.types.is_float_dtype(series.dtype):
            return "DOUBLE PRECISION"
        if pd.api.types.is_bool_dtype(series.dtype):
            return "BOOLEAN"
        if pd.api.types.is_datetime64_any_dtype(series.dtype):
            return "TIMESTAMP"
        return "TEXT"

    @staticmethod
    def _normalize_conninfo(database_url: str) -> str:
        url = make_url(database_url)
        if not url.drivername.startswith("postgresql"):
            raise ValueError("DATABASE_URL must point to PostgreSQL")

        parts = []
        if url.host:
            parts.append(f"host={url.host}")
        if url.port:
            parts.append(f"port={url.port}")
        if url.database:
            parts.append(f"dbname={url.database}")
        if url.username:
            parts.append(f"user={url.username}")
        if url.password:
            parts.append(f"password={url.password}")
        return " ".join(parts)

    def _safe_database_url(self) -> str:
        url = make_url(self.config.database_url)
        return url.render_as_string(hide_password=True)
