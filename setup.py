"""
setup.py
Run this ONCE to:
1. Load and cache all eICU data as Parquet
2. Build SQLite database
3. Train the ML model

After this, the app runs fast using cached data.
"""

import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("setup.log")
    ]
)
log = logging.getLogger(__name__)


def main():
    log.info("=" * 60)
    log.info("  eICU ICU Analytics System - Setup")
    log.info("=" * 60)

    # Step 1: Init DB
    log.info("\n[Step 1] Initializing SQLite database...")
    from utils.db_manager import init_db
    init_db()

    # Step 2: Load all data
    log.info("\n[Step 2] Loading eICU data files (this may take a few minutes)...")
    from utils.data_loader import (
        load_patients, load_vitals, load_labs,
        load_diagnosis, load_apache, load_treatment,
        build_ml_dataset
    )

    patients = load_patients()
    log.info(f"  ✓ Patients: {len(patients):,}")

    vitals = load_vitals()
    log.info(f"  ✓ Vitals: {len(vitals):,}")

    labs = load_labs()
    log.info(f"  ✓ Labs: {len(labs):,}")

    diagnosis = load_diagnosis()
    log.info(f"  ✓ Diagnosis: {len(diagnosis):,}")

    apache = load_apache()
    log.info(f"  ✓ Apache: {len(apache):,}")

    treatment = load_treatment()
    log.info(f"  ✓ Treatments: {len(treatment):,}")

    # Step 3: Build ML dataset
    log.info("\n[Step 3] Building ML dataset...")
    ml_df = build_ml_dataset()
    log.info(f"  ✓ ML dataset: {ml_df.shape}")

    # Step 4: Save to SQLite
    log.info("\n[Step 4] Saving to SQLite database...")
    from utils.db_manager import (
        save_patients_to_db, save_vitals_to_db,
        save_labs_to_db, save_diagnosis_to_db,
        save_treatments_to_db
    )

    # Save patient summary (ml_df has all features)
    save_patients_to_db(ml_df)
    log.info("  ✓ Patients saved")

    # Save vitals (limit per patient for DB size)
    log.info("  Saving vitals (this may take a moment)...")
    save_vitals_to_db(vitals)
    log.info(f"  ✓ Vitals saved: {len(vitals):,} rows")

    # Save labs
    save_labs_to_db(labs)
    log.info(f"  ✓ Labs saved: {len(labs):,} rows")

    save_diagnosis_to_db(diagnosis)
    save_treatments_to_db(treatment)
    log.info("  ✓ Diagnosis + Treatments saved")

    # Step 5: Train ML model
    log.info("\n[Step 5] Training mortality prediction model...")
    from models.ml_model import train_model, model_is_trained

    if model_is_trained():
        log.info("  ℹ Model already trained. Skipping. (Delete models/ to retrain)")
    else:
        rf, scaler, features, auc = train_model(ml_df)
        log.info(f"  ✓ Model trained! AUC = {auc:.4f}")

    log.info("\n" + "=" * 60)
    log.info("  ✓ Setup complete! Run the app:")
    log.info("    Backend:  python backend/api.py")
    log.info("    Frontend: streamlit run frontend/dashboard.py")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
