from __future__ import annotations

import argparse
import json
import logging
import pickle
from dataclasses import dataclass

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine

from config.config import ARTIFACTS_DIR, DATABASE_URL, MODELS_DIR


log = logging.getLogger(__name__)


POSTGRES_MODEL_FEATURES = [
    "age_years",
    "icu_los_hours",
    "hospital_los_hours",
    "admissionheight",
    "admissionweight",
    "creatinine_max",
    "glucose_max",
    "glucose_min",
    "lactate_max",
    "wbc_max",
    "heartrate_mean",
    "heartrate_max",
    "heartrate_min",
    "sao2_mean",
    "sao2_min",
    "resp_mean",
    "resp_max",
    "sbp_mean",
    "sbp_min",
    "map_mean",
    "temp_mean",
    "temp_max",
    "lab_row_count",
    "vital_row_count",
    "diagnosis_row_count",
    "dx_sepsis_flag",
    "dx_respiratory_flag",
    "dx_cardiac_flag",
    "dx_renal_flag",
    "gender_male",
    "gender_female",
    "gender_unknown",
    "ethnicity_asian",
    "ethnicity_white",
    "ethnicity_black",
    "ethnicity_hispanic",
    "ethnicity_other",
    "age_group_young",
    "age_group_adult",
    "age_group_senior",
    "icu_stay_short",
    "icu_stay_medium",
    "icu_stay_long",
    "qsofa_score",
    "admissionheight_missing_flag",
    "admissionweight_missing_flag",
    "creatinine_missing_flag",
    "glucose_missing_flag",
    "lactate_missing_flag",
    "wbc_missing_flag",
    "heartrate_missing_flag",
    "sao2_missing_flag",
    "respiration_missing_flag",
    "sbp_missing_flag",
    "map_missing_flag",
    "temperature_missing_flag",
    "lab_missing_flag",
    "vital_missing_flag",
]

LEAKAGE_FEATURES = [
    "icu_los_hours",
    "hospital_los_hours",
    "icu_stay_short",
    "icu_stay_medium",
    "icu_stay_long",
]


POSTGRES_MORTALITY_MODEL_PATH = MODELS_DIR / "mortality_postgres_rf.pkl"
POSTGRES_BASELINE_MODEL_PATH = MODELS_DIR / "mortality_postgres_logreg.pkl"
POSTGRES_IMPUTER_PATH = MODELS_DIR / "mortality_postgres_imputer.pkl"
POSTGRES_FEATURE_NAMES_PATH = MODELS_DIR / "feature_names_postgres.pkl"
POSTGRES_EARLY_RF_MODEL_PATH = MODELS_DIR / "mortality_postgres_rf_early.pkl"
POSTGRES_EARLY_BASELINE_MODEL_PATH = MODELS_DIR / "mortality_postgres_logreg_early.pkl"
POSTGRES_EARLY_IMPUTER_PATH = MODELS_DIR / "mortality_postgres_imputer_early.pkl"
POSTGRES_EARLY_FEATURE_NAMES_PATH = MODELS_DIR / "feature_names_postgres_early.pkl"
POSTGRES_METRICS_PATH = ARTIFACTS_DIR / "postgres_model_metrics.json"
POSTGRES_CLASSIFICATION_REPORT_PATH = ARTIFACTS_DIR / "postgres_classification_report.json"
POSTGRES_EARLY_METRICS_PATH = ARTIFACTS_DIR / "postgres_early_model_metrics.json"
POSTGRES_EARLY_CLASSIFICATION_REPORT_PATH = ARTIFACTS_DIR / "postgres_early_classification_report.json"


@dataclass(slots=True)
class TrainingConfig:
    database_url: str
    source_schema: str
    source_table: str
    test_size: float = 0.2
    random_state: int = 42


def configure_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("training.log"),
        ],
    )


class PostgresModelTrainer:
    def __init__(self, config: TrainingConfig) -> None:
        self.config = config
        self.engine: Engine = create_engine(config.database_url, future=True, pool_pre_ping=True)

    def load_dataset(self) -> pd.DataFrame:
        query = (
            f'SELECT * FROM "{self.config.source_schema}"."{self.config.source_table}" '
            "WHERE hospital_mortality IS NOT NULL"
        )
        log.info(
            "Loading training dataset | schema=%s table=%s",
            self.config.source_schema,
            self.config.source_table,
        )
        return pd.read_sql(query, self.engine)

    def train(self) -> dict:
        df = self.load_dataset()
        y = df["hospital_mortality"].astype(int)

        full_result = self._train_variant(
            df=df,
            y=y,
            feature_names=[column for column in POSTGRES_MODEL_FEATURES if column in df.columns],
            variant_name="full_model",
        )
        early_result = self._train_variant(
            df=df,
            y=y,
            feature_names=[
                column
                for column in POSTGRES_MODEL_FEATURES
                if column in df.columns and column not in LEAKAGE_FEATURES
            ],
            variant_name="early_model",
        )

        metrics = {
            "dataset": {
                "rows": int(len(df)),
                "positive_rate": float(y.mean()),
            },
            "full_model": full_result["metrics"],
            "early_model": early_result["metrics"],
        }
        self._save_variant_artifacts(
            rf_model=full_result["rf_model"],
            baseline_model=full_result["baseline_model"],
            imputer=full_result["imputer"],
            feature_names=full_result["feature_names"],
            metrics=full_result["metrics"],
            classification=full_result["classification"],
            model_path=POSTGRES_MORTALITY_MODEL_PATH,
            baseline_path=POSTGRES_BASELINE_MODEL_PATH,
            imputer_path=POSTGRES_IMPUTER_PATH,
            feature_path=POSTGRES_FEATURE_NAMES_PATH,
            metrics_path=POSTGRES_METRICS_PATH,
            classification_path=POSTGRES_CLASSIFICATION_REPORT_PATH,
        )
        self._save_variant_artifacts(
            rf_model=early_result["rf_model"],
            baseline_model=early_result["baseline_model"],
            imputer=early_result["imputer"],
            feature_names=early_result["feature_names"],
            metrics=early_result["metrics"],
            classification=early_result["classification"],
            model_path=POSTGRES_EARLY_RF_MODEL_PATH,
            baseline_path=POSTGRES_EARLY_BASELINE_MODEL_PATH,
            imputer_path=POSTGRES_EARLY_IMPUTER_PATH,
            feature_path=POSTGRES_EARLY_FEATURE_NAMES_PATH,
            metrics_path=POSTGRES_EARLY_METRICS_PATH,
            classification_path=POSTGRES_EARLY_CLASSIFICATION_REPORT_PATH,
        )
        return metrics

    def _train_variant(
        self,
        df: pd.DataFrame,
        y: pd.Series,
        feature_names: list[str],
        variant_name: str,
    ) -> dict:
        X = df[feature_names].copy()
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=self.config.test_size,
            random_state=self.config.random_state,
            stratify=y,
        )

        imputer = SimpleImputer(strategy="median")
        X_train_imputed = pd.DataFrame(imputer.fit_transform(X_train), columns=feature_names, index=X_train.index)
        X_test_imputed = pd.DataFrame(imputer.transform(X_test), columns=feature_names, index=X_test.index)

        rf_model = RandomForestClassifier(
            n_estimators=300,
            max_depth=12,
            min_samples_leaf=5,
            class_weight="balanced_subsample",
            random_state=self.config.random_state,
            n_jobs=-1,
        )
        rf_model.fit(X_train_imputed, y_train)
        rf_proba = rf_model.predict_proba(X_test_imputed)[:, 1]
        rf_pred = (rf_proba >= 0.5).astype(int)

        baseline_model = LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=self.config.random_state,
        )
        baseline_model.fit(X_train_imputed, y_train)
        baseline_proba = baseline_model.predict_proba(X_test_imputed)[:, 1]
        baseline_pred = (baseline_proba >= 0.5).astype(int)

        metrics = {
            "variant": variant_name,
            "features_used": len(feature_names),
            "random_forest": {
                "auc_roc": float(roc_auc_score(y_test, rf_proba)),
                "auprc": float(average_precision_score(y_test, rf_proba)),
                "precision": float(precision_score(y_test, rf_pred, zero_division=0)),
                "recall": float(recall_score(y_test, rf_pred, zero_division=0)),
                "f1": float(f1_score(y_test, rf_pred, zero_division=0)),
                "confusion_matrix": confusion_matrix(y_test, rf_pred).tolist(),
            },
            "logistic_regression": {
                "auc_roc": float(roc_auc_score(y_test, baseline_proba)),
                "auprc": float(average_precision_score(y_test, baseline_proba)),
                "precision": float(precision_score(y_test, baseline_pred, zero_division=0)),
                "recall": float(recall_score(y_test, baseline_pred, zero_division=0)),
                "f1": float(f1_score(y_test, baseline_pred, zero_division=0)),
                "confusion_matrix": confusion_matrix(y_test, baseline_pred).tolist(),
            },
            "top_random_forest_features": [
                {"feature": name, "importance": float(value)}
                for name, value in sorted(
                    zip(feature_names, rf_model.feature_importances_),
                    key=lambda item: item[1],
                    reverse=True,
                )[:15]
            ],
        }
        classification = classification_report(y_test, rf_pred, output_dict=True, zero_division=0)
        return {
            "rf_model": rf_model,
            "baseline_model": baseline_model,
            "imputer": imputer,
            "feature_names": feature_names,
            "metrics": metrics,
            "classification": classification,
        }

    def _save_variant_artifacts(
        self,
        rf_model,
        baseline_model,
        imputer: SimpleImputer,
        feature_names: list[str],
        metrics: dict,
        classification: dict,
        model_path,
        baseline_path,
        imputer_path,
        feature_path,
        metrics_path,
        classification_path,
    ) -> None:
        with open(feature_path, "wb") as handle:
            pickle.dump(feature_names, handle)
        with open(model_path, "wb") as handle:
            pickle.dump(rf_model, handle)
        with open(baseline_path, "wb") as handle:
            pickle.dump(baseline_model, handle)
        with open(imputer_path, "wb") as handle:
            pickle.dump(imputer, handle)

        metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        classification_path.write_text(json.dumps(classification, indent=2), encoding="utf-8")
        log.info("Saved training artifacts to %s", ARTIFACTS_DIR)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train baseline mortality models from ml_prep.ml_dataset.")
    parser.add_argument("--schema", default="ml_prep", help="Source schema for the ML dataset.")
    parser.add_argument("--table", default="ml_dataset", help="Source table for the ML dataset.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_logging()
    trainer = PostgresModelTrainer(
        TrainingConfig(
            database_url=DATABASE_URL,
            source_schema=args.schema,
            source_table=args.table,
        )
    )
    metrics = trainer.train()
    log.info(
        "Training complete | full_rf_auc=%.4f early_rf_auc=%.4f full_baseline_auc=%.4f early_baseline_auc=%.4f",
        metrics["full_model"]["random_forest"]["auc_roc"],
        metrics["early_model"]["random_forest"]["auc_roc"],
        metrics["full_model"]["logistic_regression"]["auc_roc"],
        metrics["early_model"]["logistic_regression"]["auc_roc"],
    )


if __name__ == "__main__":
    main()
