from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from training.utils.range_filters import filter_numeric_range, normalize_temperature, read_csv_kwargs, resolve_source_file

log = logging.getLogger(__name__)


def _load_periodic(eicu_path: Path, patient_ids: set[int], chunk_size: int) -> pd.DataFrame:
    path = resolve_source_file(eicu_path, "vitalPeriodic")
    usecols = [
        "patientunitstayid",
        "observationoffset",
        "heartrate",
        "respiration",
        "sao2",
        "systemicsystolic",
        "systemicdiastolic",
        "temperature",
        "cvp",
    ]
    frames: list[pd.DataFrame] = []
    for chunk in pd.read_csv(path, usecols=usecols, chunksize=chunk_size, **read_csv_kwargs(path)):
        chunk = chunk[chunk["patientunitstayid"].isin(patient_ids)].copy()
        chunk["observationoffset"] = pd.to_numeric(chunk["observationoffset"], errors="coerce")
        chunk = chunk[chunk["observationoffset"].between(0, 1439, inclusive="both")]
        if chunk.empty:
            continue
        chunk["heartrate"] = filter_numeric_range(chunk["heartrate"], 0, 300)
        chunk["respiration"] = filter_numeric_range(chunk["respiration"], 0, 80)
        chunk["sao2"] = filter_numeric_range(chunk["sao2"], 50, 100)
        chunk["systemicsystolic"] = filter_numeric_range(chunk["systemicsystolic"], 40, 300)
        chunk["systemicdiastolic"] = filter_numeric_range(chunk["systemicdiastolic"], 20, 200)
        invalid_bp = chunk["systemicsystolic"] <= chunk["systemicdiastolic"]
        chunk.loc[invalid_bp, ["systemicsystolic", "systemicdiastolic"]] = pd.NA
        chunk["temperature"] = normalize_temperature(chunk["temperature"])
        chunk["cvp"] = filter_numeric_range(chunk["cvp"], -5, 40)
        chunk["map_estimate"] = (chunk["systemicsystolic"] + 2 * chunk["systemicdiastolic"]) / 3.0
        frames.append(chunk)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=usecols + ["map_estimate"])


def _load_aperiodic(eicu_path: Path, patient_ids: set[int], chunk_size: int) -> pd.DataFrame:
    path = resolve_source_file(eicu_path, "vitalAperiodic")
    usecols = [
        "patientunitstayid",
        "observationoffset",
        "noninvasivesystolic",
        "noninvasivemean",
        "noninvasivediastolic",
        "paop",
    ]
    frames: list[pd.DataFrame] = []
    for chunk in pd.read_csv(path, usecols=usecols, chunksize=chunk_size, **read_csv_kwargs(path)):
        chunk = chunk[chunk["patientunitstayid"].isin(patient_ids)].copy()
        chunk["observationoffset"] = pd.to_numeric(chunk["observationoffset"], errors="coerce")
        chunk = chunk[chunk["observationoffset"].between(0, 1439, inclusive="both")]
        if chunk.empty:
            continue
        chunk["noninvasivesystolic"] = filter_numeric_range(chunk["noninvasivesystolic"], 40, 300)
        chunk["noninvasivediastolic"] = filter_numeric_range(chunk["noninvasivediastolic"], 20, 200)
        chunk["noninvasivemean"] = filter_numeric_range(chunk["noninvasivemean"], 20, 200)
        invalid_bp = chunk["noninvasivesystolic"] <= chunk["noninvasivediastolic"]
        chunk.loc[invalid_bp, ["noninvasivesystolic", "noninvasivediastolic", "noninvasivemean"]] = pd.NA
        chunk["paop"] = filter_numeric_range(chunk["paop"], 0, 40)
        frames.append(chunk)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=usecols)


def load_vitals_for_patients(eicu_path: Path, patient_ids: set[int], chunk_size: int) -> dict[str, pd.DataFrame]:
    periodic = _load_periodic(eicu_path, patient_ids, chunk_size)
    aperiodic = _load_aperiodic(eicu_path, patient_ids, chunk_size)
    coverage = periodic["patientunitstayid"].nunique() / max(len(patient_ids), 1)
    log.info("Loaded %s periodic rows and %s aperiodic rows", f"{len(periodic):,}", f"{len(aperiodic):,}")
    log.info("Patients with periodic vitals in first 24h: %.2f%%", coverage * 100.0)
    return {"periodic": periodic, "aperiodic": aperiodic}
