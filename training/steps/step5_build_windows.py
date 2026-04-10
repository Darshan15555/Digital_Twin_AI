from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_WINDOW_HOURS = [0, 4, 8, 12, 16, 20]
DEFAULT_WINDOW_SIZE_MIN = 4 * 60
PATIENT_ID_COL = "patientunitstayid"


def build_window_skeleton(
    patients_df: pd.DataFrame,
    window_hours: list[int],
    window_size_min: int,
) -> pd.DataFrame:
    patient_ids = (
        patients_df[[PATIENT_ID_COL]]
        .dropna(subset=[PATIENT_ID_COL])
        .drop_duplicates(subset=[PATIENT_ID_COL])
        .sort_values(PATIENT_ID_COL)
        .reset_index(drop=True)
    )

    hours = window_hours or DEFAULT_WINDOW_HOURS
    window_df = pd.DataFrame(
        {
            "window_id": range(len(hours)),
            "window_start_hour": hours,
        }
    )
    window_df["window_start_min"] = window_df["window_start_hour"] * 60
    window_df["window_end_min"] = window_df["window_start_min"] + int(window_size_min)
    window_df["window_end_hour"] = window_df["window_end_min"] // 60

    skeleton = patient_ids.merge(window_df, how="cross")
    skeleton["window_id"] = skeleton["window_id"].astype("int8")
    skeleton["window_start_hour"] = skeleton["window_start_hour"].astype("int16")
    skeleton["window_end_hour"] = skeleton["window_end_hour"].astype("int16")
    skeleton["window_start_min"] = skeleton["window_start_min"].astype("int16")
    skeleton["window_end_min"] = skeleton["window_end_min"].astype("int16")

    logger.info("Built window skeleton with %d rows for %d patients", len(skeleton), len(patient_ids))
    return skeleton[
        [
            PATIENT_ID_COL,
            "window_id",
            "window_start_min",
            "window_end_min",
            "window_start_hour",
            "window_end_hour",
        ]
    ]


def build_feature_windows(patients_df: pd.DataFrame) -> pd.DataFrame:
    return build_window_skeleton(
        patients_df=patients_df,
        window_hours=DEFAULT_WINDOW_HOURS,
        window_size_min=DEFAULT_WINDOW_SIZE_MIN,
    )