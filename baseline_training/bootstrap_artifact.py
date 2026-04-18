from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import train_test_split


PATIENT_COL = "patient_id"
TARGET_COL = "target_label"
FEATURE_COLS = ["hr", "spo2", "bp_sys", "bp_dia", "bp_mean"]


def _pick_column(df: pd.DataFrame, candidates: list[str]) -> pd.Series:
    for name in candidates:
        if name in df.columns:
            return pd.to_numeric(df[name], errors="coerce")
    return pd.Series(np.nan, index=df.index, dtype="float64")


def _normalize_target(series: pd.Series) -> pd.Series:
    if series.dtype == object:
        normalized = (
            series.fillna("")
            .astype(str)
            .str.strip()
            .str.lower()
            .map({"expired": 1, "dead": 1, "1": 1, "alive": 0, "0": 0})
        )
        return normalized.fillna(0).astype("int8")
    numeric = pd.to_numeric(series, errors="coerce").fillna(0)
    return (numeric > 0).astype("int8")


def _enforce_ranges(frame: pd.DataFrame) -> pd.DataFrame:
    frame["hr"] = frame["hr"].where(frame["hr"].between(30, 220, inclusive="both"))
    frame["spo2"] = frame["spo2"].where(frame["spo2"].between(50, 100, inclusive="both"))
    frame["bp_sys"] = frame["bp_sys"].where(frame["bp_sys"].between(40, 300, inclusive="both"))
    frame["bp_dia"] = frame["bp_dia"].where(frame["bp_dia"].between(20, 220, inclusive="both"))
    frame["bp_mean"] = frame["bp_mean"].where(frame["bp_mean"].between(20, 250, inclusive="both"))

    invalid_pair = frame["bp_sys"].notna() & frame["bp_dia"].notna() & (frame["bp_sys"] <= frame["bp_dia"])
    frame.loc[invalid_pair, ["bp_sys", "bp_dia"]] = np.nan

    computed_mean = (frame["bp_sys"] + 2.0 * frame["bp_dia"]) / 3.0
    frame["bp_mean"] = frame["bp_mean"].fillna(computed_mean)
    frame["bp_mean"] = frame["bp_mean"].where(frame["bp_mean"].between(20, 250, inclusive="both"))

    computed_dia = (3.0 * frame["bp_mean"] - frame["bp_sys"]) / 2.0
    frame["bp_dia"] = frame["bp_dia"].fillna(computed_dia)
    frame["bp_dia"] = frame["bp_dia"].where(frame["bp_dia"].between(20, 220, inclusive="both"))
    return frame


def _build_from_ml_dataset(path: Path, max_rows: int, seed: int) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if df.empty:
        raise ValueError(f"Parquet source is empty: {path}")

    out = pd.DataFrame(
        {
            PATIENT_COL: _pick_column(df, ["patient_id", "patientunitstayid"]).fillna(0).astype("int64"),
            "hr": _pick_column(df, ["hr", "heartrate_mean", "heartrate"]),
            "spo2": _pick_column(df, ["spo2", "sao2_mean", "sao2"]),
            "bp_sys": _pick_column(df, ["bp_sys", "nibp_systolic_mean", "sbp_mean", "systemicsystolic_mean"]),
            "bp_mean": _pick_column(df, ["bp_mean", "nibp_mean_mean", "systemicmean_mean", "map_mean"]),
            TARGET_COL: _normalize_target(
                _pick_column(df, [TARGET_COL, "hospital_mortality", "mortality"]).fillna(0)
            ),
        }
    )
    out["bp_dia"] = np.nan

    out = _enforce_ranges(out)
    out = out.dropna(subset=["hr", "spo2", "bp_sys", "bp_dia", "bp_mean"], how="all")
    if out.empty:
        raise ValueError("No usable rows after cleaning ml_dataset parquet")

    if len(out) > max_rows:
        out = out.sample(n=max_rows, random_state=seed)
    return out.reset_index(drop=True)


def _build_synthetic(max_rows: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = max(max_rows, 5000)
    hr = rng.normal(88, 16, size=rows).clip(30, 220)
    spo2 = rng.normal(95, 2.5, size=rows).clip(50, 100)
    bp_sys = rng.normal(118, 18, size=rows).clip(40, 260)
    bp_dia = (bp_sys * rng.uniform(0.52, 0.72, size=rows)).clip(20, 180)
    bp_mean = (bp_sys + 2.0 * bp_dia) / 3.0
    risk = (0.02 * (hr - 90)) - (0.08 * (spo2 - 95)) + (0.012 * (bp_sys - 120))
    prob = 1.0 / (1.0 + np.exp(-risk / 3.0))
    target = (rng.random(rows) < prob).astype("int8")

    return pd.DataFrame(
        {
            PATIENT_COL: np.arange(1, rows + 1, dtype=np.int64),
            "hr": hr,
            "spo2": spo2,
            "bp_sys": bp_sys,
            "bp_dia": bp_dia,
            "bp_mean": bp_mean,
            TARGET_COL: target,
        }
    )


def ensure_dataset(dataset_path: Path, *, max_rows: int = 120_000, seed: int = 42) -> Path:
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    if dataset_path.exists():
        df = pd.read_parquet(dataset_path) if dataset_path.suffix.lower() == ".parquet" else pd.read_csv(dataset_path)
        required = {PATIENT_COL, TARGET_COL, *FEATURE_COLS}
        if required.issubset(set(df.columns)):
            return dataset_path

    ml_dataset_path = Path("data/parquet/ml_dataset.parquet")
    if ml_dataset_path.exists():
        built = _build_from_ml_dataset(ml_dataset_path, max_rows=max_rows, seed=seed)
    else:
        built = _build_synthetic(max_rows=max_rows, seed=seed)

    if dataset_path.suffix.lower() == ".csv":
        built.to_csv(dataset_path, index=False)
    else:
        built.to_parquet(dataset_path, index=False, compression="snappy")
    return dataset_path


def _best_f1_threshold(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    thresholds = np.arange(0.10, 0.91, 0.01)
    best_thr = 0.5
    best_f1 = -1.0
    for thr in thresholds:
        pred = (y_proba >= thr).astype(int)
        score = f1_score(y_true, pred, zero_division=0)
        if score > best_f1:
            best_f1 = float(score)
            best_thr = float(thr)
    return best_thr


def ensure_artifact(
    artifact_path: Path = Path("artifacts/baseline_training/best_model.joblib"),
    dataset_path: Path = Path("data/parquet/minimal_hr_spo2_bp.parquet"),
    *,
    seed: int = 42,
) -> Path:
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    if artifact_path.exists():
        return artifact_path

    dataset_path = ensure_dataset(dataset_path, seed=seed)
    df = pd.read_parquet(dataset_path) if dataset_path.suffix.lower() == ".parquet" else pd.read_csv(dataset_path)

    required = {PATIENT_COL, TARGET_COL, *FEATURE_COLS}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Cannot train fallback model, dataset missing required columns: {missing}")

    X = df[FEATURE_COLS].copy()
    y = pd.to_numeric(df[TARGET_COL], errors="coerce").fillna(0).astype("int8")

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        train_size=0.8,
        random_state=seed,
        stratify=y if y.nunique() > 1 else None,
    )

    imputer = SimpleImputer(strategy="median")
    X_train_imp = pd.DataFrame(
        imputer.fit_transform(X_train),
        columns=FEATURE_COLS,
        index=X_train.index,
    )
    X_val_imp = pd.DataFrame(
        imputer.transform(X_val),
        columns=FEATURE_COLS,
        index=X_val.index,
    )

    model = LogisticRegression(
        max_iter=400,
        class_weight="balanced",
        random_state=seed,
    )
    model.fit(X_train_imp, y_train)

    val_proba = model.predict_proba(X_val_imp)[:, 1]
    threshold = _best_f1_threshold(y_val.to_numpy(), val_proba)
    auroc = float(roc_auc_score(y_val, val_proba)) if y_val.nunique() > 1 else float("nan")

    bundle = {
        "model_name": "logistic_regression_fallback",
        "features": FEATURE_COLS,
        "threshold": float(threshold),
        "imputer_medians": {
            col: float(val) for col, val in zip(FEATURE_COLS, imputer.statistics_)
        },
        "model": model,
    }
    joblib.dump(bundle, artifact_path)

    feature_path = artifact_path.parent / "feature_list.json"
    metrics_path = artifact_path.parent / "metrics.json"
    feature_path.write_text(json.dumps({"features": FEATURE_COLS}, indent=2), encoding="utf-8")
    metrics_path.write_text(
        json.dumps(
            {
                "best_model": "logistic_regression_fallback",
                "selected_threshold": float(threshold),
                "validation_auroc": auroc,
                "dataset_path": str(dataset_path),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return artifact_path
