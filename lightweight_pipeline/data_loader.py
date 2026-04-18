from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import pandas as pd


@dataclass(slots=True)
class LoaderConfig:
    data_path: Path
    chunk_size: int = 100_000
    include_aperiodic_bp: bool = False

    @classmethod
    def from_env(
        cls,
        *,
        chunk_size: int = 100_000,
        include_aperiodic_bp: bool = False,
    ) -> "LoaderConfig":
        configured = os.getenv("DATA_PATH") or os.getenv("EICU_RAW_PATH") or "data/parquet"
        return cls(
            data_path=Path(configured),
            chunk_size=chunk_size,
            include_aperiodic_bp=include_aperiodic_bp,
        )


def resolve_source_file(data_path: Path, stem: str) -> Path:
    candidates = [data_path / f"{stem}.csv.gz", data_path / f"{stem}.csv"]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Missing source file for {stem} under {data_path}")


def _read_csv_kwargs(path: Path) -> dict[str, object]:
    return {"compression": "gzip"} if path.suffix == ".gz" else {}


def _available_columns(path: Path) -> set[str]:
    header = pd.read_csv(path, nrows=0, low_memory=False, **_read_csv_kwargs(path))
    return set(header.columns.tolist())


def iter_vital_periodic_chunks(config: LoaderConfig) -> Iterator[pd.DataFrame]:
    source = resolve_source_file(config.data_path, "vitalPeriodic")
    desired = [
        "patientunitstayid",
        "observationoffset",
        "heartrate",
        "sao2",
        "systemicsystolic",
        "systemicdiastolic",
        "systemicmean",
    ]
    available = _available_columns(source)
    usecols = [col for col in desired if col in available]

    required = {"patientunitstayid", "observationoffset", "heartrate", "sao2", "systemicsystolic", "systemicdiastolic"}
    missing = sorted(required - set(usecols))
    if missing:
        raise ValueError(f"{source.name} is missing required columns: {missing}")

    for chunk in pd.read_csv(
        source,
        usecols=usecols,
        chunksize=config.chunk_size,
        low_memory=False,
        **_read_csv_kwargs(source),
    ):
        yield chunk


def iter_vital_aperiodic_chunks(config: LoaderConfig) -> Iterator[pd.DataFrame]:
    source = resolve_source_file(config.data_path, "vitalAperiodic")
    desired = [
        "patientunitstayid",
        "observationoffset",
        "noninvasivesystolic",
        "noninvasivediastolic",
        "noninvasivemean",
    ]
    available = _available_columns(source)
    usecols = [col for col in desired if col in available]

    required = {"patientunitstayid", "observationoffset"}
    missing = sorted(required - set(usecols))
    if missing:
        raise ValueError(f"{source.name} is missing required columns: {missing}")

    if not any(col in usecols for col in ("noninvasivesystolic", "noninvasivediastolic", "noninvasivemean")):
        raise ValueError(f"{source.name} does not contain BP columns")

    for chunk in pd.read_csv(
        source,
        usecols=usecols,
        chunksize=config.chunk_size,
        low_memory=False,
        **_read_csv_kwargs(source),
    ):
        yield chunk


def load_patient_label_map(config: LoaderConfig) -> pd.Series:
    source = resolve_source_file(config.data_path, "patient")
    usecols = ["patientunitstayid", "hospitaldischargestatus"]
    frames: list[pd.DataFrame] = []

    for chunk in pd.read_csv(
        source,
        usecols=usecols,
        chunksize=config.chunk_size,
        low_memory=False,
        **_read_csv_kwargs(source),
    ):
        chunk = chunk.copy()
        chunk["patientunitstayid"] = pd.to_numeric(chunk["patientunitstayid"], errors="coerce")
        chunk = chunk.dropna(subset=["patientunitstayid"])
        if chunk.empty:
            continue
        chunk["patientunitstayid"] = chunk["patientunitstayid"].astype("int64")
        chunk["target_label"] = (
            chunk["hospitaldischargestatus"].fillna("").astype(str).str.lower().eq("expired").astype("int8")
        )
        frames.append(chunk[["patientunitstayid", "target_label"]])

    if not frames:
        return pd.Series(dtype="int8")

    labels = (
        pd.concat(frames, ignore_index=True)
        .drop_duplicates(subset=["patientunitstayid"], keep="last")
        .set_index("patientunitstayid")["target_label"]
    )
    labels.index = labels.index.astype("int64")
    return labels.astype("int8")
