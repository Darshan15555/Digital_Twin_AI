"""
Local setup pipeline for the ICU analytics system.
"""

from __future__ import annotations

import logging

from models.ml_model import model_is_trained, train_models
from utils.data_loader import load_all_processed_tables, validate_source_data
from utils.db_manager import init_db, write_processed_tables
from utils.feature_engineer import build_feature_dataset

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

    log.info("Initializing SQLite database")
    init_db()

    log.info("Loading and preprocessing eICU tables")
    tables = load_all_processed_tables(force=False)

    log.info("Building feature dataset")
    ml_df = build_feature_dataset(force=False)
    tables["ml_dataset"] = ml_df

    log.info("Writing processed tables to SQLite")
    write_processed_tables(tables)

    if model_is_trained():
        log.info("Model artifacts already exist, skipping retraining")
    else:
        metrics = train_models(ml_df)
        log.info("Training complete: %s", metrics)

    log.info("Setup complete")
    log.info("Start API with: python backend/api.py")
    log.info("Start dashboard with: streamlit run frontend/dashboard.py")


if __name__ == "__main__":
    main()
