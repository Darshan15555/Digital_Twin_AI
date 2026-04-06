from __future__ import annotations

import logging
from pathlib import Path

from ingestion.config import PipelineConfig


log = logging.getLogger(__name__)


class FileScanner:
    """Detects available gzip-compressed CSV files and resolves selected subsets."""

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config

    def scan(self) -> list[Path]:
        files = sorted(self.config.data_path.glob("*.csv.gz"))
        if not files:
            raise FileNotFoundError(f"No .csv.gz files found in {self.config.data_path}")
        return files

    def select(self, selected_files: list[str] | None = None, important_only: bool = False) -> list[Path]:
        available = self.scan()
        if important_only and not selected_files:
            selected_files = list(self.config.important_files)

        if not selected_files:
            return available

        requested = {self._normalize_selector(name) for name in selected_files}
        matched = [
            path
            for path in available
            if self._normalize_selector(path.name) in requested
        ]
        missing = sorted(requested - {self._normalize_selector(path.name) for path in matched})
        if missing:
            log.warning("Requested files were not found and will be skipped: %s", ", ".join(missing))
        return matched

    @staticmethod
    def _normalize_selector(name: str) -> str:
        normalized = name.strip().lower()
        if normalized.endswith(".csv.gz"):
            return normalized[: -len(".csv.gz")]
        if normalized.endswith(".csv"):
            return normalized[: -len(".csv")]
        return normalized
