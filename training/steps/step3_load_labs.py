from __future__ import annotations

import logging
import re
from pathlib import Path

import pandas as pd

from training.utils.range_filters import filter_numeric_range, read_csv_kwargs, resolve_source_file

log = logging.getLogger(__name__)

DEFAULT_CHUNK_SIZE = 50_000

LAB_ALIASES = {
    "creatinine": ["creatinine", "creat", "serum creatinine"],
    "lactate": ["lactate", "lactic acid", "lactate, whole blood", "lactic acid, whole blood"],
    "glucose": ["glucose", "glucose (whole blood)", "glucose (whole bl)", "bedside glucose"],
    "wbc": ["wbc", "wbc count", "white blood cells", "white blood cell count"],
    "hemoglobin": ["hgb", "hemoglobin", "hgb/hct", "haemoglobin"],
    "platelets": ["plt", "platelets", "platelet count"],
    "sodium": ["sodium", "na", "sodium (whole blood)"],
    "potassium": ["potassium", "k", "potassium (whole blood)"],
    "bicarbonate": ["bicarb", "bicarbonate", "co2, total", "hco3", "hco3-"],
    "bun": ["bun", "urea nitrogen", "blood urea nitrogen"],
    "inr": ["inr", "pt/inr", "pt - inr"],
    "ph": ["ph", "arterial ph", "ph (whole blood)"],
    "pco2": ["pco2", "pco2 (whole blood)", "arterial co2", "arterial pco2"],
    "po2": ["po2", "po2 (whole blood)", "arterial po2", "arterial pO2"],
    "alt": ["alt", "alanine transaminase", "alanine aminotransferase"],
    "ast": ["ast", "aspartate transaminase", "aspartate aminotransferase"],
}

LAB_RANGES = {
    "creatinine": (0, 50),
    "lactate": (0, 30),
    "glucose": (10, 2000),
    "hemoglobin": (1, 25),
    "wbc": (0, 200),
    "platelets": (0, 2000),
    "potassium": (1, 10),
    "sodium": (100, 200),
    "bicarbonate": (5, 50),
    "bun": (1, 300),
    "inr": (0, 20),
    "ph": (6.5, 8.0),
    "pco2": (5, 150),
    "po2": (10, 700),
    "alt": (0, 5000),
    "ast": (0, 5000),
}

LAB_NAME_MAP = {
    alias: canonical
    for canonical, aliases in LAB_ALIASES.items()
    for alias in aliases
}


def _clean_text(series: pd.Series) -> pd.Series:
    return (
        series.fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
        .str.replace(r"[_/]+", " ", regex=True)
        .str.replace(r"\s+", " ", regex=True)
        .str.replace(r"[^\w\s(),.-]+", "", regex=True)
        .str.strip()
    )


def _normalize_lab_name(series: pd.Series) -> pd.Series:
    normalized = _clean_text(series)
    normalized = normalized.replace(LAB_NAME_MAP)

    direct_mask = normalized.isin(LAB_RANGES)
    if direct_mask.all():
        return normalized

    for canonical, aliases in LAB_ALIASES.items():
        alias_pattern = "|".join(re.escape(alias) for alias in aliases)
        match_mask = ~direct_mask & normalized.str.contains(rf"\b(?:{alias_pattern})\b", regex=True, na=False)
        normalized = normalized.where(~match_mask, canonical)
        direct_mask = normalized.isin(LAB_RANGES)

    return normalized


def load_labs_for_patients(eicu_path: Path, patient_ids: set[int], chunk_size: int) -> pd.DataFrame:
    source_path = resolve_source_file(eicu_path, "lab")
    effective_chunk_size = chunk_size or DEFAULT_CHUNK_SIZE
    usecols = ["patientunitstayid", "labresultoffset", "labname", "labresult"]
    frames: list[pd.DataFrame] = []

    for chunk in pd.read_csv(
        source_path,
        usecols=usecols,
        chunksize=effective_chunk_size,
        low_memory=False,
        **read_csv_kwargs(source_path),
    ):
        chunk = chunk[chunk["patientunitstayid"].isin(patient_ids)].copy()
        if chunk.empty:
            continue

        chunk["labresultoffset"] = pd.to_numeric(chunk["labresultoffset"], errors="coerce")
        chunk = chunk[chunk["labresultoffset"].between(-60, 1439, inclusive="both")]
        if chunk.empty:
            continue

        chunk["labname"] = _normalize_lab_name(chunk["labname"])
        chunk = chunk[chunk["labname"].isin(LAB_RANGES)].copy()
        if chunk.empty:
            continue

        chunk["labresult"] = pd.to_numeric(chunk["labresult"], errors="coerce")
        for lab_name, (lower, upper) in LAB_RANGES.items():
            lab_mask = chunk["labname"].eq(lab_name)
            if lab_mask.any():
                chunk.loc[lab_mask, "labresult"] = filter_numeric_range(chunk.loc[lab_mask, "labresult"], lower, upper)

        chunk = chunk.dropna(subset=["labresult"])
        if chunk.empty:
            continue

        frames.append(chunk[usecols])

    labs = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=usecols)
    log.info("Loaded %s lab rows from %s", f"{len(labs):,}", source_path.name)
    return labs