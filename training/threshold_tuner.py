from __future__ import annotations

import argparse
import json
import logging
import pickle
from dataclasses import dataclass

import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine

from config.config import ARTIFACTS_DIR, DATABASE_URL
from training.postgres_trainer import (
    POSTGRES_EARLY_FEATURE_NAMES_PATH,
    POSTGRES_EARLY_IMPUTER_PATH,
    POSTGRES_EARLY_RF_MODEL_PATH,
)


log = logging.getLogger(__name__)

THRESHOLD_REPORT_PATH = ARTIFACTS_DIR / "postgres_early_thresholds.json"


@dataclass(slots=True)
class ThresholdConfig:
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


class ThresholdTuner:
    def __init__(self, config: ThresholdConfig) -> None:
        self.config = config
        self.engine: Engine = create_engine(config.database_url, future=True, pool_pre_ping=True)

    def load_dataset(self) -> pd.DataFrame:
        query = (
            f'SELECT * FROM "{self.config.source_schema}"."{self.config.source_table}" '
            "WHERE hospital_mortality IS NOT NULL"
        )
        return pd.read_sql(query, self.engine)

    def evaluate(self) -> dict:
        df = self.load_dataset()
        feature_names = self._load_pickle(POSTGRES_EARLY_FEATURE_NAMES_PATH)
        imputer = self._load_pickle(POSTGRES_EARLY_IMPUTER_PATH)
        model = self._load_pickle(POSTGRES_EARLY_RF_MODEL_PATH)

        X = df[feature_names].copy()
        y = df["hospital_mortality"].astype(int)
        _, X_test, _, y_test = train_test_split(
            X,
            y,
            test_size=self.config.test_size,
            random_state=self.config.random_state,
            stratify=y,
        )

        X_test_imputed = pd.DataFrame(imputer.transform(X_test), columns=feature_names, index=X_test.index)
        proba = model.predict_proba(X_test_imputed)[:, 1]

        thresholds = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]
        rows = []
        for threshold in thresholds:
            pred = (proba >= threshold).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
            rows.append(
                {
                    "threshold": threshold,
                    "precision": float(precision_score(y_test, pred, zero_division=0)),
                    "recall": float(recall_score(y_test, pred, zero_division=0)),
                    "f1": float(f1_score(y_test, pred, zero_division=0)),
                    "true_negative": int(tn),
                    "false_positive": int(fp),
                    "false_negative": int(fn),
                    "true_positive": int(tp),
                }
            )

        best_f1 = max(rows, key=lambda item: item["f1"])
        high_recall = max(
            (row for row in rows if row["recall"] >= 0.75),
            key=lambda item: item["precision"],
            default=max(rows, key=lambda item: item["recall"]),
        )
        higher_precision = max(
            (row for row in rows if row["precision"] >= 0.35),
            key=lambda item: item["recall"],
            default=max(rows, key=lambda item: item["precision"]),
        )

        report = {
            "variant": "early_random_forest_threshold_tuning",
            "threshold_metrics": rows,
            "recommended": {
                "balanced_f1": best_f1,
                "high_recall": high_recall,
                "higher_precision": higher_precision,
            },
        }
        THRESHOLD_REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
        return report

    @staticmethod
    def _load_pickle(path):
        with open(path, "rb") as handle:
            return pickle.load(handle)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate thresholds for the early RF mortality model.")
    parser.add_argument("--schema", default="ml_prep", help="Source schema for the ML dataset.")
    parser.add_argument("--table", default="ml_dataset", help="Source table for the ML dataset.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_logging()
    tuner = ThresholdTuner(
        ThresholdConfig(
            database_url=DATABASE_URL,
            source_schema=args.schema,
            source_table=args.table,
        )
    )
    report = tuner.evaluate()
    log.info(
        "Threshold tuning complete | best_f1_threshold=%.2f high_recall_threshold=%.2f higher_precision_threshold=%.2f",
        report["recommended"]["balanced_f1"]["threshold"],
        report["recommended"]["high_recall"]["threshold"],
        report["recommended"]["higher_precision"]["threshold"],
    )


if __name__ == "__main__":
    main()
