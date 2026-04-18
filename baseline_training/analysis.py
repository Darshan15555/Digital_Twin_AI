from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd


def extract_feature_importance(model, feature_cols: list[str]) -> pd.DataFrame:
    if hasattr(model, "feature_importances_"):
        raw = np.asarray(model.feature_importances_, dtype=float)
    elif hasattr(model, "coef_"):
        raw = np.abs(np.asarray(model.coef_, dtype=float).reshape(-1))
    else:
        raw = np.zeros(len(feature_cols), dtype=float)

    if raw.shape[0] != len(feature_cols):
        raw = np.resize(raw, len(feature_cols))

    df = pd.DataFrame(
        {
            "feature": feature_cols,
            "importance": raw.astype(float),
        }
    )
    total = float(df["importance"].sum())
    if total > 0.0:
        df["importance_pct"] = (df["importance"] / total) * 100.0
    else:
        df["importance_pct"] = 0.0
    return df.sort_values("importance", ascending=False).reset_index(drop=True)


def summarize_feature_importance(feature_importance_df: pd.DataFrame) -> dict[str, object]:
    if feature_importance_df.empty:
        return {"top_feature": None, "ranking": []}

    top_row = feature_importance_df.iloc[0]
    ranking = feature_importance_df[["feature", "importance", "importance_pct"]].to_dict(orient="records")
    return {
        "top_feature": str(top_row["feature"]),
        "top_feature_pct": float(top_row["importance_pct"]),
        "ranking": [
            {
                "feature": str(item["feature"]),
                "importance": float(item["importance"]),
                "importance_pct": float(item["importance_pct"]),
            }
            for item in ranking
        ],
    }


def measure_inference_latency(model, X_reference: pd.DataFrame, *, runs: int = 200) -> dict[str, float]:
    if X_reference.empty:
        return {"runs": 0.0, "avg_ms": 0.0, "p50_ms": 0.0, "p95_ms": 0.0, "max_ms": 0.0}

    sample = X_reference.iloc[[0]]
    _ = model.predict_proba(sample)

    durations_ms: list[float] = []
    for _ in range(int(max(runs, 1))):
        start = time.perf_counter()
        _ = model.predict_proba(sample)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        durations_ms.append(float(elapsed_ms))

    arr = np.asarray(durations_ms, dtype=float)
    return {
        "runs": float(len(durations_ms)),
        "avg_ms": float(arr.mean()),
        "p50_ms": float(np.percentile(arr, 50)),
        "p95_ms": float(np.percentile(arr, 95)),
        "max_ms": float(arr.max()),
    }


def model_size_mb(artifact_path: Path) -> float:
    if not artifact_path.exists():
        return 0.0
    return float(artifact_path.stat().st_size / (1024 * 1024))


def optimization_suggestions(model, latency: dict[str, float], artifact_size_mb: float) -> list[str]:
    suggestions: list[str] = []
    n_estimators = getattr(model, "n_estimators", None)

    if latency.get("p95_ms", 0.0) > 100.0:
        suggestions.append("P95 inference latency >100ms: reduce n_estimators and/or max_depth, then retrain.")
    else:
        suggestions.append("Latency target met (<100ms P95) on single-row inference.")

    if n_estimators is not None and int(n_estimators) > 400:
        suggestions.append("n_estimators is high; try 200-400 with early stopping to reduce latency/size.")

    if artifact_size_mb > 25.0:
        suggestions.append("Model artifact is large; consider joblib compression or fewer trees.")
    else:
        suggestions.append("Model artifact size is lightweight for API deployment.")

    suggestions.append("Benchmark batch prediction latency with realistic API payload sizes before production rollout.")
    return suggestions


def run_robustness_checks(
    model,
    *,
    feature_cols: list[str],
    medians: dict[str, float],
    transform_fn,
    threshold: float,
) -> list[dict[str, object]]:
    baseline = {col: float(medians.get(col, 0.0)) for col in feature_cols}

    cases: dict[str, dict[str, float | None]] = {
        "baseline_medians": baseline.copy(),
        "missing_spo2": {**baseline, "spo2": None},
        "missing_all": {col: None for col in feature_cols},
        "extreme_low": {**baseline, "hr": 20.0, "spo2": 45.0, "bp_sys": 70.0, "bp_dia": 68.0, "bp_mean": 60.0},
        "extreme_high": {**baseline, "hr": 240.0, "spo2": 101.0, "bp_sys": 250.0, "bp_dia": 140.0, "bp_mean": 180.0},
    }

    rows = []
    for case_name, payload in cases.items():
        raw_df = pd.DataFrame([payload], columns=feature_cols)
        X_ready = transform_fn(raw_df)
        proba = float(model.predict_proba(X_ready)[:, 1][0])
        rows.append(
            {
                "case": case_name,
                "input": {k: (None if v is None else float(v)) for k, v in payload.items()},
                "probability": proba,
                "prediction": int(proba >= float(threshold)),
                "is_finite_probability": bool(np.isfinite(proba)),
            }
        )
    return rows


def conceptual_comparison() -> dict[str, object]:
    return {
        "old_pipeline_status": "Removed",
        "lightweight_model_profile": {
            "features": ["HR", "SpO2", "BP (sys/dia/mean)"],
            "pipeline_complexity": "Low",
            "expected_latency": "Lower than complex multi-feature pipeline",
            "expected_interpretability": "Higher",
            "expected_ceiling_accuracy": "Potentially lower due to reduced feature space",
        },
        "recommended_use_cases": [
            "Real-time bedside risk scoring",
            "High-throughput API inference with strict latency budgets",
            "Fallback model when full feature pipeline is unavailable",
        ],
    }
