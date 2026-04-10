"""
Minimal setup checks for the ingestion-only ICU analytics system.
"""

from __future__ import annotations

import logging

from ingestion.config import PipelineConfig
from utils.data_loader import validate_source_data

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("setup.log")],
)
log = logging.getLogger(__name__)


def main() -> None:
    missing = validate_source_data()
    if missing:
        raise FileNotFoundError(
            "Missing source files under configured EICU_RAW_PATH:\n" + "\n".join(f"- {name}" for name in missing)
        )

    config = PipelineConfig()
    config.validate()

    log.info("Setup validation complete")
    log.info("Raw data path: %s", config.data_path)
    log.info("Target schema: %s", config.schema_name)
    log.info("Run ingestion with: python ingest.py --files patient")


if __name__ == "__main__":
    main()
