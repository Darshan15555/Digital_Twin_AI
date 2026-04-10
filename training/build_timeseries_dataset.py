from __future__ import annotations

import gc
import logging
import os
import sys
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from training.steps.step1_load_patients import load_and_filter_patients
from training.steps.step2_load_vitals import load_vitals_for_patients
from training.steps.step3_load_labs import load_labs_for_patients
from training.steps.step4_load_interventions import load_all_interventions
from training.steps.step5_build_windows import build_window_skeleton
from training.steps.step6_aggregate import aggregate_all_features
from training.steps.step7_compute_labels import compute_all_labels
from training.steps.step8_compute_scores import compute_all_scores
from training.steps.step9_save import save_dataset
from training.utils.range_filters import resolve_eicu_path

load_dotenv(BASE_DIR / ".env")

EICU_PATH = resolve_eicu_path(os.getenv("EICU_RAW_PATH", r"D:\physionet-data\eicu"))
OUTPUT_PATH = BASE_DIR / "data" / "timeseries_training_dataset.parquet"
CHECKPOINT_DIR = BASE_DIR / "data" / "ts_checkpoints"
MAX_PATIENTS = 200_000
MAX_ROWS = 1_200_000
WINDOW_HOURS = [0, 4, 8, 12, 16, 20]
WINDOW_SIZE_MIN = 240
CHUNK_SIZE = 50_000

CHECKPOINT_DIR.mkdir(exist_ok=True, parents=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(BASE_DIR / "training" / "ts_build.log", encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)


def _checkpoint_path(name: str) -> Path:
    return CHECKPOINT_DIR / f"{name}.parquet"


def _load_or_build_df(name: str, builder) -> pd.DataFrame:
    checkpoint = _checkpoint_path(name)
    if checkpoint.exists():
        log.info("%s: loading checkpoint %s", name, checkpoint.name)
        return pd.read_parquet(checkpoint)
    df = builder()
    df.to_parquet(checkpoint, index=False)
    return df


def _load_or_build_object(name: str, builder):
    checkpoint = CHECKPOINT_DIR / name
    if checkpoint.exists():
        log.info("%s: loading checkpoint directory %s", name, checkpoint.name)
        loaded: dict[str, object] = {}
        for item in checkpoint.iterdir():
            if item.is_file() and item.suffix == ".parquet":
                loaded[item.stem] = pd.read_parquet(item)
            elif item.is_dir():
                nested: dict[str, pd.DataFrame] = {}
                for nested_item in item.iterdir():
                    if nested_item.suffix == ".parquet":
                        nested[nested_item.stem] = pd.read_parquet(nested_item)
                loaded[item.name] = nested
        return loaded
    obj = builder()
    checkpoint.mkdir(parents=True, exist_ok=True)
    for key, value in obj.items():
        if isinstance(value, dict):
            nested_dir = checkpoint / key
            nested_dir.mkdir(exist_ok=True)
            for nested_key, nested_value in value.items():
                nested_value.to_parquet(nested_dir / f"{nested_key}.parquet", index=False)
        else:
            value.to_parquet(checkpoint / f"{key}.parquet", index=False)
    return obj


def run() -> None:
    start = time.time()
    log.info("=" * 60)
    log.info("  eICU Time-Series Dataset Builder")
    log.info("  Data path: %s", EICU_PATH)
    log.info("=" * 60)

    log.info("[STEP 1] Loading patients...")
    patients_df = _load_or_build_df("step1_patients", lambda: load_and_filter_patients(EICU_PATH, MAX_PATIENTS))
    patient_ids = set(patients_df["patientunitstayid"].tolist())
    log.info("  -> %s patients selected", f"{len(patient_ids):,}")

    log.info("[STEP 2] Loading vitals...")
    vitals_df = _load_or_build_object("step2_vitals", lambda: load_vitals_for_patients(EICU_PATH, patient_ids, CHUNK_SIZE))
    log.info("  -> periodic=%s aperiodic=%s", f"{len(vitals_df['periodic']):,}", f"{len(vitals_df['aperiodic']):,}")

    log.info("[STEP 3] Loading labs...")
    labs_df = _load_or_build_df("step3_labs", lambda: load_labs_for_patients(EICU_PATH, patient_ids, CHUNK_SIZE))
    log.info("  -> %s lab rows", f"{len(labs_df):,}")

    log.info("[STEP 4] Loading interventions...")
    interventions = _load_or_build_object("step4_interventions", lambda: load_all_interventions(EICU_PATH, patient_ids, CHUNK_SIZE))
    log.info("  -> interventions loaded")

    log.info("[STEP 5] Building windows...")
    windows_df = _load_or_build_df("step5_windows", lambda: build_window_skeleton(patients_df, WINDOW_HOURS, WINDOW_SIZE_MIN))
    log.info("  -> %s window rows", f"{len(windows_df):,}")

    log.info("[STEP 6] Aggregating features...")
    features_df = _load_or_build_df("step6_features", lambda: aggregate_all_features(windows_df, vitals_df, labs_df, interventions, patients_df))
    log.info("  -> feature matrix %s", features_df.shape)

    log.info("[STEP 7] Computing labels...")
    labeled_df = _load_or_build_df("step7_labels", lambda: compute_all_labels(features_df, patients_df, vitals_df, labs_df))
    log.info("  -> labels added")

    log.info("[STEP 8] Computing scores...")
    final_df = _load_or_build_df("step8_scores", lambda: compute_all_scores(labeled_df))
    log.info("  -> scores computed")

    log.info("[STEP 9] Saving dataset...")
    stats = save_dataset(final_df, OUTPUT_PATH, MAX_ROWS)

    del patients_df, vitals_df, labs_df, interventions, windows_df, features_df, labeled_df, final_df
    gc.collect()

    elapsed = (time.time() - start) / 60.0
    log.info("=" * 60)
    log.info("  COMPLETE in %.1f minutes", elapsed)
    log.info("  Final shape: %s", stats["shape"])
    log.info("  Output: %s", OUTPUT_PATH)
    log.info("  Mortality rate: %.2f%%", stats["mortality_rate"] * 100.0)
    log.info("  Patients: %s", f"{stats['patients']:,}")
    log.info("  Windows per patient (avg): %.2f", stats["avg_windows"])
    log.info("=" * 60)


if __name__ == "__main__":
    run()
