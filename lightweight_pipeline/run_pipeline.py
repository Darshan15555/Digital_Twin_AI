from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd

from lightweight_pipeline.data_loader import (
    LoaderConfig,
    iter_vital_aperiodic_chunks,
    iter_vital_periodic_chunks,
    load_patient_label_map,
)
from lightweight_pipeline.preprocess import (
    REQUIRED_OUTPUT_COLUMNS,
    aggregate_mean_per_patient,
    attach_target_label,
    clean_vital_aperiodic_chunk,
    clean_vital_periodic_chunk,
    handle_missing,
)
from lightweight_pipeline.sampler import RandomKeySampler, StratifiedKeySampler, compute_stratified_quotas


log = logging.getLogger("lightweight_pipeline")


def _iter_feature_chunks(
    config: LoaderConfig,
    *,
    label_map: pd.Series | None,
    first_24h_only: bool,
) -> Iterator[pd.DataFrame]:
    for raw in iter_vital_periodic_chunks(config):
        cleaned = clean_vital_periodic_chunk(raw, first_24h_only=first_24h_only)
        if cleaned.empty:
            continue
        yield attach_target_label(cleaned, label_map)

    if config.include_aperiodic_bp:
        for raw in iter_vital_aperiodic_chunks(config):
            cleaned = clean_vital_aperiodic_chunk(raw, first_24h_only=first_24h_only)
            if cleaned.empty:
                continue
            yield attach_target_label(cleaned, label_map)


def _count_labels(
    config: LoaderConfig,
    *,
    label_map: pd.Series | None,
    first_24h_only: bool,
) -> dict[int, int]:
    if label_map is None or label_map.empty:
        return {}

    counts: dict[int, int] = {}
    for chunk in _iter_feature_chunks(config, label_map=label_map, first_24h_only=first_24h_only):
        if "target_label" not in chunk.columns:
            continue
        observed = chunk["target_label"].dropna().astype(int).value_counts()
        for label, count in observed.items():
            counts[int(label)] = counts.get(int(label), 0) + int(count)
    return counts


def _finalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    cols = REQUIRED_OUTPUT_COLUMNS.copy()
    if "target_label" in df.columns:
        cols.append("target_label")
    return df[cols].copy()


def _pick_numeric(df: pd.DataFrame, candidates: list[str]) -> pd.Series:
    for name in candidates:
        if name in df.columns:
            return pd.to_numeric(df[name], errors="coerce")
    return pd.Series(np.nan, index=df.index, dtype="float64")


def _raw_sources_exist(data_path: Path) -> bool:
    has_patient = any((data_path / name).exists() for name in ("patient.csv.gz", "patient.csv"))
    has_vitals = any((data_path / name).exists() for name in ("vitalPeriodic.csv.gz", "vitalPeriodic.csv"))
    return has_patient and has_vitals


def _build_from_ml_dataset_parquet(
    parquet_path: Path,
    *,
    max_rows: int,
    random_state: int,
    include_label: bool,
) -> tuple[pd.DataFrame, int]:
    source = pd.read_parquet(parquet_path)
    if source.empty:
        raise ValueError(f"Parquet source is empty: {parquet_path}")

    out = pd.DataFrame(
        {
            "patient_id": _pick_numeric(source, ["patient_id", "patientunitstayid"]),
            "hr": _pick_numeric(source, ["hr", "heartrate_mean", "heartrate"]),
            "spo2": _pick_numeric(source, ["spo2", "sao2_mean", "sao2"]),
            "bp_sys": _pick_numeric(source, ["bp_sys", "nibp_systolic_mean", "sbp_mean", "systemicsystolic_mean"]),
            "bp_mean": _pick_numeric(source, ["bp_mean", "nibp_mean_mean", "map_mean", "systemicmean_mean"]),
        }
    )

    out["bp_dia"] = (3.0 * out["bp_mean"] - out["bp_sys"]) / 2.0
    out["bp_mean"] = out["bp_mean"].fillna((out["bp_sys"] + 2.0 * out["bp_dia"]) / 3.0)

    out["patient_id"] = out["patient_id"].fillna(0).astype("int64")
    out["hr"] = out["hr"].where(out["hr"].between(30, 220, inclusive="both"))
    out["spo2"] = out["spo2"].where(out["spo2"].between(50, 100, inclusive="both"))
    out["bp_sys"] = out["bp_sys"].where(out["bp_sys"].between(40, 300, inclusive="both"))
    out["bp_dia"] = out["bp_dia"].where(out["bp_dia"].between(20, 220, inclusive="both"))
    out["bp_mean"] = out["bp_mean"].where(out["bp_mean"].between(20, 250, inclusive="both"))

    invalid_pair = out["bp_sys"].notna() & out["bp_dia"].notna() & (out["bp_sys"] <= out["bp_dia"])
    out.loc[invalid_pair, ["bp_sys", "bp_dia", "bp_mean"]] = np.nan

    if include_label:
        raw_target = _pick_numeric(source, ["target_label", "hospital_mortality", "mortality"]).fillna(0)
        out["target_label"] = (raw_target > 0).astype("int8")

    out = out.dropna(subset=["hr", "spo2", "bp_sys", "bp_dia", "bp_mean"], how="all")
    total_seen = len(out)
    if total_seen > max_rows:
        out = out.sample(n=max_rows, random_state=random_state)
    return out.reset_index(drop=True), int(total_seen)


def run(
    *,
    data_path: Path,
    output_path: Path,
    chunk_size: int,
    max_rows: int,
    sampling: str,
    missing: str,
    include_label: bool,
    first_24h_only: bool,
    aggregate_patient_mean: bool,
    include_aperiodic_bp: bool,
    random_state: int,
) -> dict[str, object]:
    config = LoaderConfig(
        data_path=data_path,
        chunk_size=chunk_size,
        include_aperiodic_bp=include_aperiodic_bp,
    )
    if not config.data_path.exists():
        raise FileNotFoundError(f"Data path does not exist: {config.data_path}")

    if not _raw_sources_exist(config.data_path):
        parquet_fallback = config.data_path / "ml_dataset.parquet"
        if parquet_fallback.exists():
            sampled, total_seen = _build_from_ml_dataset_parquet(
                parquet_fallback,
                max_rows=max_rows,
                random_state=random_state,
                include_label=include_label,
            )
            sampled = handle_missing(sampled, strategy=missing)
            if aggregate_patient_mean:
                sampled = aggregate_mean_per_patient(sampled)
            sampled = _finalize_columns(sampled)

            output_path.parent.mkdir(parents=True, exist_ok=True)
            if output_path.suffix.lower() == ".csv":
                sampled.to_csv(output_path, index=False)
            else:
                sampled.to_parquet(output_path, index=False, compression="snappy")

            summary = {
                "rows_seen": int(total_seen),
                "rows_saved": int(len(sampled)),
                "patients_saved": int(sampled["patient_id"].nunique()) if not sampled.empty else 0,
                "output_path": str(output_path),
                "sampling": "random",
                "missing_strategy": missing,
                "first_24h_only": False,
                "aggregated_per_patient": aggregate_patient_mean,
                "source_mode": "ml_dataset_parquet",
            }
            if "target_label" in sampled.columns:
                summary["target_positive_rate"] = float(sampled["target_label"].mean())
            return summary

        raise FileNotFoundError(
            f"No raw vital sources found in {config.data_path} and fallback parquet is missing: {parquet_fallback}"
        )

    label_map = load_patient_label_map(config) if include_label else None
    if label_map is not None and not label_map.empty:
        log.info("Loaded patient labels: %s", f"{len(label_map):,}")
    else:
        log.info("No labels loaded; target_label will be omitted")

    sampling = sampling.lower().strip()
    if sampling not in {"random", "stratified"}:
        raise ValueError("sampling must be one of: random, stratified")

    if sampling == "stratified":
        label_counts = _count_labels(config, label_map=label_map, first_24h_only=first_24h_only)
        quotas = compute_stratified_quotas(label_counts, max_rows)
        if quotas:
            log.info("Using stratified sampling quotas: %s", quotas)
            sampler: RandomKeySampler | StratifiedKeySampler = StratifiedKeySampler(
                quotas=quotas,
                random_state=random_state,
                label_col="target_label",
            )
        else:
            log.warning("Stratified sampling requested but label counts unavailable. Falling back to random sampling.")
            sampler = RandomKeySampler(max_rows=max_rows, random_state=random_state)
    else:
        sampler = RandomKeySampler(max_rows=max_rows, random_state=random_state)

    total_seen = 0
    for chunk in _iter_feature_chunks(config, label_map=label_map, first_24h_only=first_24h_only):
        total_seen += len(chunk)
        sampler.update(chunk)

    sampled = sampler.result()
    sampled = handle_missing(sampled, strategy=missing)
    if aggregate_patient_mean:
        sampled = aggregate_mean_per_patient(sampled)

    sampled = _finalize_columns(sampled)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix.lower() == ".csv":
        sampled.to_csv(output_path, index=False)
    else:
        sampled.to_parquet(output_path, index=False, compression="snappy")

    summary = {
        "rows_seen": int(total_seen),
        "rows_saved": int(len(sampled)),
        "patients_saved": int(sampled["patient_id"].nunique()) if not sampled.empty else 0,
        "output_path": str(output_path),
        "sampling": sampling,
        "missing_strategy": missing,
        "first_24h_only": first_24h_only,
        "aggregated_per_patient": aggregate_patient_mean,
    }
    if "target_label" in sampled.columns:
        summary["target_positive_rate"] = float(sampled["target_label"].mean())
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build lightweight HR/SpO2/BP dataset from raw eICU files.")
    parser.add_argument("--data-path", type=Path, default=None, help="Path to raw eICU csv/csv.gz files")
    parser.add_argument("--output-path", type=Path, default=Path("data/parquet/minimal_hr_spo2_bp.parquet"))
    parser.add_argument("--chunk-size", type=int, default=100_000)
    parser.add_argument("--max-rows", type=int, default=1_000_000)
    parser.add_argument("--sampling", choices=["random", "stratified"], default="random")
    parser.add_argument("--missing", choices=["drop", "median", "mean"], default="drop")
    parser.add_argument("--no-label", action="store_true", help="Do not attach target label")
    parser.add_argument("--no-first-24h", action="store_true", help="Disable first-24h filter")
    parser.add_argument("--aggregate-patient-mean", action="store_true", help="Aggregate to one row per patient")
    parser.add_argument("--include-aperiodic-bp", action="store_true", help="Also stream vitalAperiodic for BP-only rows")
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    args = parse_args()

    if args.data_path is None:
        config = LoaderConfig.from_env(
            chunk_size=args.chunk_size,
            include_aperiodic_bp=args.include_aperiodic_bp,
        )
        data_path = config.data_path
    else:
        data_path = args.data_path

    result = run(
        data_path=data_path,
        output_path=args.output_path,
        chunk_size=args.chunk_size,
        max_rows=args.max_rows,
        sampling=args.sampling,
        missing=args.missing,
        include_label=not args.no_label,
        first_24h_only=not args.no_first_24h,
        aggregate_patient_mean=args.aggregate_patient_mean,
        include_aperiodic_bp=args.include_aperiodic_bp,
        random_state=args.random_state,
    )

    for key, value in result.items():
        log.info("%s: %s", key, value)


if __name__ == "__main__":
    main()
