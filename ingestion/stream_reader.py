from __future__ import annotations

import csv
import gc
import gzip
import logging
from pathlib import Path
from typing import Iterator

import pandas as pd

from ingestion.config import PipelineConfig


log = logging.getLogger(__name__)


class StreamingReader:
    """Generator-based reader with pandas chunking and csv fallback for difficult files."""

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config

    def read_header(self, file_path: Path) -> list[str]:
        try:
            header = pd.read_csv(
                file_path,
                compression="gzip",
                nrows=0,
                engine="python",
                on_bad_lines="skip",
            )
            return [str(column) for column in header.columns]
        except Exception:
            with gzip.open(file_path, mode="rt", newline="", encoding="utf-8", errors="ignore") as handle:
                reader = csv.reader(handle)
                return next(reader)

    def iter_chunks(
        self,
        file_path: Path,
        usecols: list[str] | None = None,
        dtype: dict[str, str] | None = None,
        chunk_size: int | None = None,
        prefer_csv_fallback: bool = False,
    ) -> Iterator[tuple[int, pd.DataFrame]]:
        chunk_size = chunk_size or self.config.chunk_size
        if prefer_csv_fallback:
            yield from self._iter_csv_fallback(file_path, usecols, chunk_size)
            return

        try:
            reader = pd.read_csv(
                file_path,
                compression="gzip",
                chunksize=chunk_size,
                usecols=usecols,
                dtype=dtype,
                engine="python",
                on_bad_lines="skip",
            )
            for chunk_number, chunk in enumerate(reader, start=1):
                yield chunk_number, chunk
                del chunk
                gc.collect()
            return
        except (pd.errors.ParserError, MemoryError) as exc:
            log.warning("Falling back to csv streaming for %s after pandas failure: %s", file_path.name, exc)

        yield from self._iter_csv_fallback(file_path, usecols, chunk_size)

    def _iter_csv_fallback(
        self,
        file_path: Path,
        usecols: list[str] | None,
        chunk_size: int,
    ) -> Iterator[tuple[int, pd.DataFrame]]:
        with gzip.open(file_path, mode="rt", newline="", encoding="utf-8", errors="ignore") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                return

            available_columns = [column for column in (usecols or reader.fieldnames) if column in reader.fieldnames]
            buffer: list[dict[str, str | None]] = []
            chunk_number = 0

            for row in reader:
                buffer.append({column: row.get(column) for column in available_columns})
                if len(buffer) >= chunk_size:
                    chunk_number += 1
                    yield chunk_number, pd.DataFrame.from_records(buffer, columns=available_columns)
                    buffer.clear()
                    gc.collect()

            if buffer:
                chunk_number += 1
                yield chunk_number, pd.DataFrame.from_records(buffer, columns=available_columns)
                buffer.clear()
                gc.collect()

