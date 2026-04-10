from __future__ import annotations

import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd

from training.utils.range_filters import filter_numeric_range, normalize_fio2, read_csv_kwargs, resolve_source_file

log = logging.getLogger(__name__)

DEFAULT_CHUNK_SIZE = 50_000

VASOPRESSOR_KEYWORDS = {
    "norepinephrine": ["norepinephrine", "levophed", "noradrenaline"],
    "epinephrine": ["epinephrine", "adrenaline"],
    "dopamine": ["dopamine"],
    "vasopressin": ["vasopressin"],
    "phenylephrine": ["phenylephrine", "neosynephrine", "neo-synephrine"],
    "dobutamine": ["dobutamine"],
}

NURSE_MAP = {
    "glasgow coma score": "gcs_total",
    "gcs total": "gcs_total",
    "gcs score": "gcs_total",
    "gcs - motor": "gcs_motor",
    "gcs motor": "gcs_motor",
    "gcs - verbal": "gcs_verbal",
    "gcs verbal": "gcs_verbal",
    "gcs - eyes": "gcs_eye",
    "gcs eye": "gcs_eye",
    "pain score/goal": "pain_score",
    "pain score": "pain_score",
    "rass score": "rass_score",
    "rass": "rass_score",
    "delirium score": "cam_icu",
    "cam-icu": "cam_icu",
    "cam icu": "cam_icu",
}

RESP_FIO2_LABEL_PATTERN = re.compile(r"\bfi(?:o2|02)\b|fraction of inspired oxygen", re.IGNORECASE)
RESP_PEEP_LABEL_PATTERN = re.compile(r"\bpeep\b", re.IGNORECASE)


def _clean_text(series: pd.Series) -> pd.Series:
    return (
        series.fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
        .str.replace(r"[_/]+", " ", regex=True)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )


def _load_infusion(eicu_path: Path, patient_ids: set[int], chunk_size: int) -> pd.DataFrame:
    source_path = resolve_source_file(eicu_path, "infusionDrug")
    effective_chunk_size = chunk_size or DEFAULT_CHUNK_SIZE
    usecols = ["patientunitstayid", "infusionoffset", "drugname", "drugrate", "drugamount"]
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

        chunk["infusionoffset"] = pd.to_numeric(chunk["infusionoffset"], errors="coerce")
        chunk = chunk[chunk["infusionoffset"].between(0, 1439, inclusive="both")]
        if chunk.empty:
            continue

        chunk["drugname"] = _clean_text(chunk["drugname"])
        chunk["drugrate"] = pd.to_numeric(chunk["drugrate"], errors="coerce")
        chunk["drugamount"] = pd.to_numeric(chunk["drugamount"], errors="coerce")
        chunk = chunk[(chunk["drugrate"].isna()) | (chunk["drugrate"] > 0) | (chunk["drugamount"].fillna(0) > 0)].copy()
        if chunk.empty:
            continue

        activity_columns: list[str] = []
        for canonical, keywords in VASOPRESSOR_KEYWORDS.items():
            pattern = "|".join(re.escape(keyword) for keyword in keywords)
            column_name = f"{canonical}_active"
            chunk[column_name] = chunk["drugname"].str.contains(pattern, regex=True, na=False).astype("int8")
            activity_columns.append(column_name)

        chunk["is_vasopressor"] = chunk[activity_columns].max(axis=1).astype("int8")
        chunk = chunk[chunk["is_vasopressor"].eq(1)].copy()
        if chunk.empty:
            continue

        frames.append(chunk[usecols + activity_columns + ["is_vasopressor"]])

    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=usecols + [f"{name}_active" for name in VASOPRESSOR_KEYWORDS] + ["is_vasopressor"])


def _load_respiratory(eicu_path: Path, patient_ids: set[int], chunk_size: int) -> dict[str, pd.DataFrame]:
    effective_chunk_size = chunk_size or DEFAULT_CHUNK_SIZE

    care_path = resolve_source_file(eicu_path, "respiratoryCare")
    care_usecols = ["patientunitstayid", "respcarestatusoffset", "airwaytype", "airwaysize", "peeplimit", "setapneafio2"]
    care_frames: list[pd.DataFrame] = []
    for chunk in pd.read_csv(
        care_path,
        usecols=care_usecols,
        chunksize=effective_chunk_size,
        low_memory=False,
        **read_csv_kwargs(care_path),
    ):
        chunk = chunk[chunk["patientunitstayid"].isin(patient_ids)].copy()
        if chunk.empty:
            continue

        chunk["respcarestatusoffset"] = pd.to_numeric(chunk["respcarestatusoffset"], errors="coerce")
        chunk = chunk[chunk["respcarestatusoffset"].between(0, 1439, inclusive="both")]
        if chunk.empty:
            continue

        chunk["airwaytype"] = _clean_text(chunk["airwaytype"])
        chunk["airwaysize"] = pd.to_numeric(chunk["airwaysize"], errors="coerce")
        chunk["peep"] = filter_numeric_range(chunk["peeplimit"], 0, 30)
        chunk["fio2_inline"] = normalize_fio2(chunk["setapneafio2"])
        chunk["on_ventilator"] = chunk["airwaytype"].ne("").astype("int8")

        care_frames.append(
            chunk[["patientunitstayid", "respcarestatusoffset", "airwaytype", "airwaysize", "peep", "fio2_inline", "on_ventilator"]]
        )

    care = (
        pd.concat(care_frames, ignore_index=True)
        if care_frames
        else pd.DataFrame(columns=["patientunitstayid", "respcarestatusoffset", "airwaytype", "airwaysize", "peep", "fio2_inline", "on_ventilator"])
    )

    chart_path = resolve_source_file(eicu_path, "respiratoryCharting")
    chart_usecols = ["patientunitstayid", "respchartoffset", "respchartvaluelabel", "respchartvalue"]
    chart_frames: list[pd.DataFrame] = []
    for chunk in pd.read_csv(
        chart_path,
        usecols=chart_usecols,
        chunksize=effective_chunk_size,
        low_memory=False,
        **read_csv_kwargs(chart_path),
    ):
        chunk = chunk[chunk["patientunitstayid"].isin(patient_ids)].copy()
        if chunk.empty:
            continue

        chunk["respchartoffset"] = pd.to_numeric(chunk["respchartoffset"], errors="coerce")
        chunk = chunk[chunk["respchartoffset"].between(0, 1439, inclusive="both")]
        if chunk.empty:
            continue

        labels = _clean_text(chunk["respchartvaluelabel"])

        fio2_chunk = chunk[labels.str.contains(RESP_FIO2_LABEL_PATTERN, na=False)].copy()
        if not fio2_chunk.empty:
            fio2_chunk["fio2"] = normalize_fio2(fio2_chunk["respchartvalue"])
            fio2_chunk = fio2_chunk.dropna(subset=["fio2"])
            if not fio2_chunk.empty:
                chart_frames.append(fio2_chunk[["patientunitstayid", "respchartoffset", "fio2"]])

    chart = pd.concat(chart_frames, ignore_index=True) if chart_frames else pd.DataFrame(columns=["patientunitstayid", "respchartoffset", "fio2"])
    return {"care": care, "charting": chart}


def _load_intake_output(eicu_path: Path, patient_ids: set[int], chunk_size: int) -> pd.DataFrame:
    source_path = resolve_source_file(eicu_path, "intakeOutput")
    effective_chunk_size = chunk_size or DEFAULT_CHUNK_SIZE
    usecols = ["patientunitstayid", "intakeoutputoffset", "celllabel", "cellvaluenumeric"]
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

        chunk["intakeoutputoffset"] = pd.to_numeric(chunk["intakeoutputoffset"], errors="coerce")
        chunk = chunk[chunk["intakeoutputoffset"].between(0, 1439, inclusive="both")]
        if chunk.empty:
            continue

        chunk["celllabel"] = _clean_text(chunk["celllabel"])
        chunk["cellvaluenumeric"] = pd.to_numeric(chunk["cellvaluenumeric"], errors="coerce")
        chunk = chunk[chunk["cellvaluenumeric"] > 0].copy()
        if chunk.empty:
            continue

        intake_mask = chunk["celllabel"].str.contains(r"\b(intake|volume in|infusion|fluids?|iv|tube feed)\b", regex=True, na=False)
        urine_mask = chunk["celllabel"].str.contains(r"\b(urine|void|foley|catheter|uop)\b", regex=True, na=False)
        output_mask = chunk["celllabel"].str.contains(r"\b(output|drain|ng|tube|emesis|stool)\b", regex=True, na=False)

        chunk["io_type"] = np.select([urine_mask, intake_mask, output_mask], ["urine", "intake", "output"], default="")
        chunk = chunk[chunk["io_type"] != ""].copy()
        if chunk.empty:
            continue

        frames.append(chunk[usecols + ["io_type"]])

    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=usecols + ["io_type"])


def _normalize_cam_icu(series: pd.Series) -> pd.Series:
    cleaned = _clean_text(series)
    numeric = pd.to_numeric(cleaned, errors="coerce")
    mapped = numeric.where(numeric.isin([0, 1]))

    positive_mask = cleaned.str.contains(r"\b(positive|pos|yes|present)\b", regex=True, na=False)
    negative_mask = cleaned.str.contains(r"\b(negative|neg|no|absent)\b", regex=True, na=False)

    mapped = mapped.where(~positive_mask, 1.0)
    mapped = mapped.where(~negative_mask, 0.0)
    return mapped


def _load_nurse_charting(eicu_path: Path, patient_ids: set[int], chunk_size: int) -> pd.DataFrame:
    source_path = resolve_source_file(eicu_path, "nurseCharting")
    effective_chunk_size = chunk_size or DEFAULT_CHUNK_SIZE
    usecols = ["patientunitstayid", "nursingchartoffset", "nursingchartcelltypevallabel", "nursingchartcelltypevalname", "nursingchartvalue"]
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

        chunk["nursingchartoffset"] = pd.to_numeric(chunk["nursingchartoffset"], errors="coerce")
        chunk = chunk[chunk["nursingchartoffset"].between(0, 1439, inclusive="both")]
        if chunk.empty:
            continue

        names = _clean_text(chunk["nursingchartcelltypevalname"])
        labels = _clean_text(chunk["nursingchartcelltypevallabel"])

        chunk["charttype"] = names.map(NURSE_MAP)
        chunk.loc[chunk["charttype"].isna(), "charttype"] = labels.map(NURSE_MAP)
        chunk = chunk[chunk["charttype"].notna()].copy()
        if chunk.empty:
            continue

        chunk["nursingchartvalue"] = pd.to_numeric(chunk["nursingchartvalue"], errors="coerce")

        range_map = {
            "gcs_total": (3, 15),
            "gcs_motor": (1, 6),
            "gcs_verbal": (1, 5),
            "gcs_eye": (1, 4),
            "pain_score": (0, 10),
            "rass_score": (-5, 4),
        }
        for charttype, (lower, upper) in range_map.items():
            mask = chunk["charttype"].eq(charttype)
            if mask.any():
                chunk.loc[mask, "nursingchartvalue"] = filter_numeric_range(chunk.loc[mask, "nursingchartvalue"], lower, upper)

        cam_mask = chunk["charttype"].eq("cam_icu")
        if cam_mask.any():
            chunk.loc[cam_mask, "nursingchartvalue"] = _normalize_cam_icu(chunk.loc[cam_mask, "nursingchartvalue"])

        chunk = chunk.dropna(subset=["nursingchartvalue"])
        if chunk.empty:
            continue

        frames.append(chunk[["patientunitstayid", "nursingchartoffset", "charttype", "nursingchartvalue"]])

    return (
        pd.concat(frames, ignore_index=True)
        if frames
        else pd.DataFrame(columns=["patientunitstayid", "nursingchartoffset", "charttype", "nursingchartvalue"])
    )


def load_all_interventions(eicu_path: Path, patient_ids: set[int], chunk_size: int) -> dict[str, object]:
    interventions = {
        "infusion": _load_infusion(eicu_path, patient_ids, chunk_size),
        "respiratory": _load_respiratory(eicu_path, patient_ids, chunk_size),
        "intake_output": _load_intake_output(eicu_path, patient_ids, chunk_size),
        "nurse_charting": _load_nurse_charting(eicu_path, patient_ids, chunk_size),
    }
    log.info(
        "Interventions loaded | infusion=%s respiratory_care=%s respiratory_chart=%s io=%s nurse=%s",
        f"{len(interventions['infusion']):,}",
        f"{len(interventions['respiratory']['care']):,}",
        f"{len(interventions['respiratory']['charting']):,}",
        f"{len(interventions['intake_output']):,}",
        f"{len(interventions['nurse_charting']):,}",
    )
    return interventions