from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd


log = logging.getLogger(__name__)


INT_COLUMNS_MAP: dict[str, list[str]] = {
    "lab": ["patientunitstayid", "labresultoffset"],
    "patient": ["patientunitstayid"],
    "vital_periodic": ["patientunitstayid"],
}


@dataclass(frozen=True, slots=True)
class TableSpec:
    table_name: str
    usecols: list[str] | None = None
    rename_map: dict[str, str] | None = None
    numeric_columns: tuple[str, ...] = ()
    datetime_columns: tuple[str, ...] = ()
    dtype_map: dict[str, str] | None = None


TABLE_SPECS: dict[str, TableSpec] = {
    "patient.csv.gz": TableSpec(
        table_name="patient",
        usecols=[
            "patientunitstayid",
            "patienthealthsystemstayid",
            "hospitalid",
            "gender",
            "age",
            "ethnicity",
            "admissionheight",
            "admissionweight",
            "hospitaladmitsource",
            "hospitaldischargestatus",
            "unitadmitsource",
            "unitdischargestatus",
            "unitdischargeoffset",
            "hospitaldischargeoffset",
            "apacheadmissiondx",
        ],
        numeric_columns=(
            "patientunitstayid",
            "patienthealthsystemstayid",
            "hospitalid",
            "admissionheight",
            "admissionweight",
            "unitdischargeoffset",
            "hospitaldischargeoffset",
        ),
        dtype_map={
            "patientunitstayid": "Int64",
            "patienthealthsystemstayid": "Int64",
            "hospitalid": "Int64",
            "admissionheight": "float32",
            "admissionweight": "float32",
            "unitdischargeoffset": "float32",
            "hospitaldischargeoffset": "float32",
        },
    ),
    "admissiondx.csv.gz": TableSpec(
        table_name="admission_dx",
        usecols=[
            "patientunitstayid",
            "admitdxpath",
            "admitdxenteredoffset",
            "admitdxname",
        ],
        numeric_columns=("patientunitstayid", "admitdxenteredoffset"),
        dtype_map={
            "patientunitstayid": "Int64",
            "admitdxenteredoffset": "float32",
        },
    ),
    "lab.csv.gz": TableSpec(
        table_name="lab",
        usecols=[
            "patientunitstayid",
            "labresultoffset",
            "labname",
            "labresult",
            "labmeasurenameinterface",
        ],
        numeric_columns=("patientunitstayid", "labresultoffset", "labresult"),
        dtype_map={
            "patientunitstayid": "Int64",
            "labresultoffset": "float32",
            "labresult": "float32",
        },
    ),
    "vitalperiodic.csv.gz": TableSpec(
        table_name="vital_periodic",
        usecols=[
            "patientunitstayid",
            "observationoffset",
            "temperature",
            "sao2",
            "heartrate",
            "respiration",
            "systemicsystolic",
            "systemicdiastolic",
            "systemicmean",
        ],
        numeric_columns=(
            "patientunitstayid",
            "observationoffset",
            "temperature",
            "sao2",
            "heartrate",
            "respiration",
            "systemicsystolic",
            "systemicdiastolic",
            "systemicmean",
        ),
        dtype_map={
            "patientunitstayid": "Int64",
            "observationoffset": "float32",
            "temperature": "float32",
            "sao2": "float32",
            "heartrate": "float32",
            "respiration": "float32",
            "systemicsystolic": "float32",
            "systemicdiastolic": "float32",
            "systemicmean": "float32",
        },
    ),
}


def spec_for(file_name: str, available_columns: list[str]) -> TableSpec:
    key = file_name.lower()
    if key in TABLE_SPECS:
        spec = TABLE_SPECS[key]
        usecols = [column for column in (spec.usecols or available_columns) if column in available_columns]
        return TableSpec(
            table_name=spec.table_name,
            usecols=usecols,
            rename_map=spec.rename_map,
            numeric_columns=tuple(column for column in spec.numeric_columns if column in usecols),
            datetime_columns=tuple(column for column in spec.datetime_columns if column in usecols),
            dtype_map={column: dtype for column, dtype in (spec.dtype_map or {}).items() if column in usecols},
        )

    table_name = key.removesuffix(".csv.gz").replace("-", "_")
    return TableSpec(table_name=table_name, usecols=available_columns)


class DataTransformer:
    """Applies light standardization without forcing full in-memory transforms."""

    def transform_chunk(self, file_name: str, chunk: pd.DataFrame, spec: TableSpec) -> pd.DataFrame:
        transformed = chunk
        transformed.columns = [self._normalize_column_name(column) for column in transformed.columns]

        if spec.rename_map:
            transformed = transformed.rename(columns=spec.rename_map)

        for column in spec.numeric_columns:
            normalized = self._normalize_column_name(column)
            if normalized in transformed.columns:
                transformed.loc[:, normalized] = pd.to_numeric(transformed[normalized], errors="coerce")
                target_dtype = (spec.dtype_map or {}).get(column)
                if target_dtype:
                    try:
                        transformed.loc[:, normalized] = transformed[normalized].astype(target_dtype)
                    except (TypeError, ValueError):
                        log.debug("Unable to coerce %s.%s to %s", file_name, normalized, target_dtype)

        for column in spec.datetime_columns:
            normalized = self._normalize_column_name(column)
            if normalized in transformed.columns:
                transformed.loc[:, normalized] = pd.to_datetime(
                    transformed[normalized],
                    errors="coerce",
                    utc=False,
                )

        object_columns = transformed.select_dtypes(include=["object"]).columns
        for column in object_columns:
            transformed.loc[:, column] = transformed[column].astype("string").str.strip()
            transformed.loc[:, column] = transformed[column].replace(
                {"": pd.NA, "nan": pd.NA, "None": pd.NA, "<NA>": pd.NA}
            )

        transformed = self._enforce_integer_columns(transformed, spec.table_name)

        transformed = transformed.dropna(axis=1, how="all")
        transformed = transformed.drop_duplicates()

        if "patientunitstayid" in transformed.columns:
            transformed = transformed[transformed["patientunitstayid"].notna()]

        log.debug("Transformed %s chunk to %s rows", file_name, f"{len(transformed):,}")
        return transformed

    def _enforce_integer_columns(self, frame: pd.DataFrame, table_name: str) -> pd.DataFrame:
        int_columns = [
            self._normalize_column_name(column)
            for column in INT_COLUMNS_MAP.get(table_name, [])
            if self._normalize_column_name(column) in frame.columns
        ]
        if not int_columns:
            return frame

        transformed = frame.copy()
        for column in int_columns:
            transformed[column] = self.clean_int_column(transformed[column], preserve_index=True)

        transformed = transformed.dropna(subset=int_columns)

        for column in int_columns:
            transformed[column] = (
                pd.to_numeric(transformed[column], errors="coerce")
                .round(0)
                .astype("int64")
            )

        return transformed

    @staticmethod
    def clean_int_column(series: pd.Series, preserve_index: bool = False) -> pd.Series:
        cleaned = series.astype("string").str.strip()
        cleaned = pd.to_numeric(cleaned, errors="coerce")
        if preserve_index:
            return cleaned
        cleaned = cleaned.dropna()
        return cleaned.astype("int64")

    @staticmethod
    def _normalize_column_name(column: str) -> str:
        return (
            str(column)
            .strip()
            .lower()
            .replace(" ", "_")
            .replace("/", "_")
            .replace("-", "_")
        )
