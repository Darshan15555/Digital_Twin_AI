from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.common import DATA_DIR, ensure_directories, now_iso, setup_logger, write_json

log = setup_logger("models.audit", "models/audit.log")


def _window_stats(df: pd.DataFrame) -> dict[str, float]:
    windows_per_patient = df.groupby("patientunitstayid").size()
    return {
        "rows": int(len(df)),
        "patients": int(df["patientunitstayid"].nunique()),
        "avg_windows_per_patient": float(windows_per_patient.mean()),
    }


def _plot_missingness(train: pd.DataFrame, output_dir: Path) -> list[tuple[str, float]]:
    missingness = train.isnull().mean().sort_values(ascending=False)
    top_missing = missingness.head(30)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns

        plt.figure(figsize=(10, 8))
        sns.barplot(x=top_missing.values * 100.0, y=top_missing.index, orient="h")
        plt.xlabel("Missingness (%)")
        plt.ylabel("Feature")
        plt.title("Top 30 Missing Features")
        plt.tight_layout()
        plt.savefig(output_dir / "audit_missingness.png", dpi=150, bbox_inches="tight")
        plt.close()
    except ImportError:
        log.warning("matplotlib/seaborn not installed; skipping missingness plot")
    return [(column, float(value)) for column, value in top_missing.items()]


def _numeric_distribution_summary(df: pd.DataFrame) -> tuple[dict[str, dict[str, float]], list[str], list[dict[str, float]]]:
    numeric = df.select_dtypes(include=[np.number])
    distributions = {}
    warnings = []
    constants = []
    for column in numeric.columns:
        series = numeric[column].dropna()
        if series.empty:
            continue
        distributions[column] = {
            "mean": float(series.mean()),
            "std": float(series.std()),
            "min": float(series.min()),
            "p5": float(series.quantile(0.05)),
            "p25": float(series.quantile(0.25)),
            "p50": float(series.quantile(0.50)),
            "p75": float(series.quantile(0.75)),
            "p95": float(series.quantile(0.95)),
            "max": float(series.max()),
            "skewness": float(series.skew()),
            "kurtosis": float(series.kurtosis()),
            "unique_values": int(series.nunique()),
        }
        if distributions[column]["std"] == 0:
            constants.append(column)
        if distributions[column]["unique_values"] == 2 and not column.endswith(("_active", "_measured", "_positive", "_high", "_low", "_male")):
            warnings.append(f"{column} has only 2 unique values but is not clearly binary")
    corr_candidates = numeric.drop(columns=[column for column in numeric.columns if numeric[column].nunique(dropna=True) <= 1], errors="ignore")
    corr_matrix = corr_candidates.corr(numeric_only=True).abs()
    high_corr_pairs = []
    if not corr_matrix.empty:
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        for row, col in upper.stack().sort_values(ascending=False).items():
            if col > 0.95:
                high_corr_pairs.append({"feature_a": row[0], "feature_b": row[1], "correlation": float(col)})
    return distributions, warnings, high_corr_pairs[:20] + [{"feature_a": column, "feature_b": column, "correlation": 1.0} for column in constants[:0]]


def run_audit(train_path: str, val_path: str, test_path: str, output_dir: str | Path) -> dict:
    ensure_directories()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train = pd.read_parquet(train_path)
    val = pd.read_parquet(val_path)
    test = pd.read_parquet(test_path)

    critical_issues: list[str] = []
    warnings: list[str] = []
    info: list[str] = []
    recommendations: list[str] = []

    train_pids = set(train["patientunitstayid"])
    val_pids = set(val["patientunitstayid"])
    test_pids = set(test["patientunitstayid"])
    overlaps = {
        "train_val": len(train_pids & val_pids),
        "train_test": len(train_pids & test_pids),
        "val_test": len(val_pids & test_pids),
    }
    if any(overlaps.values()):
        critical_issues.append(f"Patient leakage across splits detected: {overlaps}")
    if not train["window_id"].between(0, 5).all() or not val["window_id"].between(0, 5).all() or not test["window_id"].between(0, 5).all():
        critical_issues.append("window_id values outside 0-5 detected")
    if not pd.api.types.is_integer_dtype(train["patientunitstayid"]):
        critical_issues.append("patientunitstayid is not integer typed in train split")

    positive_only = ["hr_mean", "sao2_mean", "resp_mean", "sbp_mean", "dbp_mean", "temp_mean", "creatinine_value", "glucose_value", "potassium_value", "sodium_value", "lactate_value", "fio2_mean", "peep_mean"]
    for column in [name for name in positive_only if name in train.columns]:
        if (train[column].dropna() < 0).any():
            critical_issues.append(f"Negative values found in positive-only feature: {column}")

    label_cols = [column for column in train.columns if column.startswith("label_")]
    label_stats = {}
    for label in label_cols:
        combined = pd.concat([train[[label]], val[[label]], test[[label]]], axis=0)
        label_stats[label] = {
            "train_positive_rate": float(train[label].mean(skipna=True)),
            "val_positive_rate": float(val[label].mean(skipna=True)),
            "test_positive_rate": float(test[label].mean(skipna=True)),
            "train_null_rate": float(train[label].isnull().mean()),
            "unique_values_train": int(train[label].dropna().nunique()),
        }
        if label != "label_deterioration_next":
            if label_stats[label]["train_positive_rate"] > 0.5:
                warnings.append(f"{label} positive rate > 0.5 in train split")
            if 0 < label_stats[label]["train_positive_rate"] < 0.01:
                warnings.append(f"{label} positive rate < 1% in train split")
            if label_stats[label]["train_null_rate"] > 0.2:
                warnings.append(f"{label} null rate > 20% in train split")
            if train[label].dropna().nunique() > 2 and label.startswith("label_") and "los" not in label:
                warnings.append(f"{label} looks non-binary in train split")
    mortality_spread = max(label_stats["label_hospital_mortality"]["train_positive_rate"], label_stats["label_hospital_mortality"]["val_positive_rate"], label_stats["label_hospital_mortality"]["test_positive_rate"]) - min(label_stats["label_hospital_mortality"]["train_positive_rate"], label_stats["label_hospital_mortality"]["val_positive_rate"], label_stats["label_hospital_mortality"]["test_positive_rate"])
    if mortality_spread > 0.02:
        warnings.append("Hospital mortality rate differs by >2% across splits")
    if not train.loc[train["window_id"] == 5, "label_deterioration_next"].isnull().all():
        critical_issues.append("label_deterioration_next is not null for all window_id == 5 rows in train split")

    missingness = train.isnull().mean().sort_values(ascending=False)
    top_missing = _plot_missingness(train, output_dir)
    for column, value in missingness.items():
        if column.startswith(("hr_", "sao2_", "resp_", "sbp_", "temp_", "cvp_")) and value > 0.5:
            warnings.append(f"Vital feature {column} has >50% missingness")
        if column.endswith(("_value", "_measured")) and any(lab in column for lab in ["lactate", "creatinine", "glucose", "hemoglobin", "wbc", "potassium", "sodium", "bicarbonate", "bun", "platelets", "inr", "ph", "pco2", "po2"]) and value > 0.9:
            warnings.append(f"Lab feature {column} has >90% missingness")
        if column in {"age", "gender_male", "apache_score", "apache_predicted_mort"} and value > 0.2:
            warnings.append(f"Static feature {column} has >20% missingness")
        if column.startswith(("vasopressor_", "ventilator_")) and value > 0:
            warnings.append(f"Intervention feature {column} has nulls")

    distributions, dist_warnings, high_corr_pairs = _numeric_distribution_summary(train)
    warnings.extend(dist_warnings)

    leakage_risks = []
    det_df = train[train["label_deterioration_next"].notna()].copy()
    numeric_cols = det_df.select_dtypes(include=[np.number]).columns.tolist()
    feature_numeric_cols = [column for column in numeric_cols if not column.startswith("label_")]
    leak_scan = []
    for column in feature_numeric_cols[:]:
        series = det_df[column]
        if series.nunique(dropna=True) <= 1:
            continue
        corr = det_df["label_deterioration_next"].corr(series)
        if pd.notna(corr):
            leak_scan.append((column, abs(float(corr))))
    leak_scan.sort(key=lambda item: item[1], reverse=True)
    for column, corr in leak_scan[:20]:
        if corr > 0.8:
            leakage_risks.append(f"{column} has unusually high absolute correlation ({corr:.3f}) with label_deterioration_next")
    mortality_24_corr = train["label_mortality_24h"].corr(train["label_hospital_mortality"])
    if pd.notna(mortality_24_corr) and abs(float(mortality_24_corr) - 1.0) < 1e-12:
        leakage_risks.append("label_mortality_24h is perfectly correlated with label_hospital_mortality")
    if "apache_score" in train.columns and train.loc[train["window_id"] == 0, "apache_score"].notna().any():
        leakage_risks.append("APACHE features are populated in window_0 and may be future-aware for real-time use")
        recommendations.append("For real-time deployment, consider excluding APACHE-derived features or masking them for window_0.")

    mortality_by_window = train.groupby("window_id")["label_hospital_mortality"].mean().to_dict()
    if max(mortality_by_window.values()) - min(mortality_by_window.values()) > 0.01:
        warnings.append("Hospital mortality rate varies by >1% across windows")
    deterioration_by_window = train.groupby("window_id")["label_deterioration_next"].mean().dropna().to_dict()

    sample_patients = train["patientunitstayid"].drop_duplicates().head(1000)
    sample_df = train[train["patientunitstayid"].isin(sample_patients)]
    if "hr_mean" in sample_df.columns:
        hr_std = sample_df.groupby("patientunitstayid")["hr_mean"].std().fillna(0)
        if (hr_std == 0).mean() > 0.10:
            warnings.append("More than 10% of sampled patients have no hr_mean variation across windows")
    sequential_ok = train.groupby("patientunitstayid")["window_id"].apply(lambda x: list(sorted(x.tolist())) == list(range(int(x.min()), int(x.min()) + len(x))))
    if not sequential_ok.all():
        warnings.append("Some patients have non-sequential window_id sequences")

    dataset_stats = {
        "train": _window_stats(train),
        "val": _window_stats(val),
        "test": _window_stats(test),
        "total_patients_all_splits": int(len(train_pids | val_pids | test_pids)),
        "column_count": int(len(train.columns)),
        "columns": train.columns.tolist(),
        "split_overlap": overlaps,
        "mortality_by_window": {str(int(key)): float(value) for key, value in mortality_by_window.items()},
        "deterioration_by_window": {str(int(key)): float(value) for key, value in deterioration_by_window.items()},
    }

    if critical_issues:
        overall_status = "FAIL"
    elif warnings:
        overall_status = "WARN"
    else:
        overall_status = "PASS"

    report = {
        "audit_timestamp": now_iso(),
        "overall_status": overall_status,
        "critical_issues": critical_issues,
        "warnings": warnings,
        "info": info,
        "dataset_stats": dataset_stats,
        "label_stats": label_stats,
        "missingness_summary": {
            "top_30_missing_features": [{"feature": column, "missing_rate": value} for column, value in top_missing],
        },
        "distribution_summary": distributions,
        "high_correlation_pairs": high_corr_pairs,
        "leakage_risks": leakage_risks,
        "recommendations": recommendations,
    }
    write_json(output_dir / "audit_report.json", report)
    html = [
        "<html><head><title>ICU Dataset Audit</title></head><body>",
        f"<h1>ICU Dataset Audit: {overall_status}</h1>",
        f"<p>Generated at {report['audit_timestamp']}</p>",
        "<h2>Critical Issues</h2><ul>",
    ]
    html.extend([f"<li>{issue}</li>" for issue in critical_issues] or ["<li>None</li>"])
    html.append("</ul><h2>Warnings</h2><ul>")
    html.extend([f"<li>{warning}</li>" for warning in warnings] or ["<li>None</li>"])
    html.append("</ul><h2>Recommendations</h2><ul>")
    html.extend([f"<li>{item}</li>" for item in recommendations] or ["<li>None</li>"])
    html.append("</ul></body></html>")
    (output_dir / "audit_report.html").write_text("".join(html), encoding="utf-8")

    if overall_status == "PASS":
        log.info("PASS — Dataset ready for training")
    elif overall_status == "WARN":
        log.warning("WARN — %s warnings found, review before training", len(warnings))
    else:
        log.error("FAIL — %s critical issues found", len(critical_issues))
    return report


if __name__ == "__main__":
    run_audit("data/ts_train.parquet", "data/ts_val.parquet", "data/ts_test.parquet", DATA_DIR)
