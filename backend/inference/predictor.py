from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd


log = logging.getLogger(__name__)

DEFAULT_ARTIFACT_PATH = Path("artifacts/baseline_training/best_model.joblib")
DEFAULT_FEATURES = ["hr", "spo2", "bp_sys", "bp_dia", "bp_mean"]


class VitalPredictor:
    def __init__(self, artifact_path: str | Path = DEFAULT_ARTIFACT_PATH) -> None:
        self.artifact_path = Path(artifact_path)
        self._loaded = False
        self._model = None
        self._model_name = "unknown"
        self._features = DEFAULT_FEATURES.copy()
        self._default_threshold = 0.5
        self._imputer_medians: dict[str, float] = {}

    @property
    def model_name(self) -> str:
        self._ensure_loaded()
        return self._model_name

    @property
    def features(self) -> list[str]:
        self._ensure_loaded()
        return self._features.copy()

    @property
    def default_threshold(self) -> float:
        self._ensure_loaded()
        return float(self._default_threshold)

    def model_info(self) -> dict[str, Any]:
        try:
            self._ensure_loaded()
            return {
                "model_ready": True,
                "model_name": self._model_name,
                "features": self._features.copy(),
                "default_threshold": float(self._default_threshold),
                "artifact_path": str(self.artifact_path),
            }
        except Exception as exc:
            return {
                "model_ready": False,
                "artifact_path": str(self.artifact_path),
                "error": str(exc),
            }

    def predict(self, payload: dict[str, Any], threshold: float | None = None) -> dict[str, Any]:
        self._ensure_loaded()
        row = self._build_feature_row(payload)
        X = pd.DataFrame([row], columns=self._features)
        probability = float(self._model.predict_proba(X)[:, 1][0])
        threshold_used = float(self._default_threshold if threshold is None else threshold)
        prediction = int(probability >= threshold_used)
        risk_level = self._risk_level(probability, threshold_used)
        return {
            "prediction": prediction,
            "probability": round(probability, 6),
            "risk_level": risk_level,
            "threshold_used": threshold_used,
            "model_name": self._model_name,
        }

    def predict_batch(self, payloads: list[dict[str, Any]], threshold: float | None = None) -> list[dict[str, Any]]:
        self._ensure_loaded()
        rows = [self._build_feature_row(item) for item in payloads]
        X = pd.DataFrame(rows, columns=self._features)
        probabilities = self._model.predict_proba(X)[:, 1]
        threshold_used = float(self._default_threshold if threshold is None else threshold)
        outputs: list[dict[str, Any]] = []
        for probability in probabilities:
            score = float(probability)
            outputs.append(
                {
                    "prediction": int(score >= threshold_used),
                    "probability": round(score, 6),
                    "risk_level": self._risk_level(score, threshold_used),
                    "threshold_used": threshold_used,
                    "model_name": self._model_name,
                }
            )
        return outputs

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        self._load_artifact()
        self._loaded = True

    def _load_artifact(self) -> None:
        if not self.artifact_path.exists():
            try:
                from baseline_training.bootstrap_artifact import ensure_artifact

                ensure_artifact(self.artifact_path)
            except Exception as exc:
                raise RuntimeError(f"Model initialization failed: {exc}") from exc

        if not self.artifact_path.exists():
            raise FileNotFoundError(f"Model artifact not found: {self.artifact_path}")

        bundle = joblib.load(self.artifact_path)
        if isinstance(bundle, dict) and "model" in bundle:
            self._model = bundle["model"]
            self._model_name = str(bundle.get("model_name", type(self._model).__name__))
            features = bundle.get("features") or DEFAULT_FEATURES
            self._features = [str(feature) for feature in features]
            self._default_threshold = float(bundle.get("threshold", 0.5))
            self._imputer_medians = {
                str(key): float(value)
                for key, value in (bundle.get("imputer_medians") or {}).items()
            }
        else:
            self._model = bundle
            self._model_name = type(self._model).__name__
            self._features = DEFAULT_FEATURES.copy()
            self._default_threshold = 0.5
            self._imputer_medians = {}

        if not hasattr(self._model, "predict_proba"):
            raise ValueError("Loaded model does not support predict_proba")

        log.info(
            "Inference model loaded | name=%s features=%s threshold=%.3f path=%s",
            self._model_name,
            self._features,
            self._default_threshold,
            self.artifact_path,
        )

    def _build_feature_row(self, payload: dict[str, Any]) -> dict[str, float]:
        source = dict(payload)
        if source.get("bp_mean") is None and source.get("bp_sys") is not None and source.get("bp_dia") is not None:
            source["bp_mean"] = (float(source["bp_sys"]) + 2.0 * float(source["bp_dia"])) / 3.0
        if source.get("bp_sys") is not None and source.get("bp_dia") is not None:
            if float(source["bp_sys"]) <= float(source["bp_dia"]):
                raise ValueError("bp_sys must be greater than bp_dia")

        row: dict[str, float] = {}
        for feature in self._features:
            value = source.get(feature)
            if value is None or (isinstance(value, float) and np.isnan(value)):
                if feature in self._imputer_medians:
                    row[feature] = float(self._imputer_medians[feature])
                else:
                    raise ValueError(f"Missing value for required feature '{feature}' and no median is available")
            else:
                row[feature] = float(value)
        return row

    @staticmethod
    def _risk_level(probability: float, threshold: float) -> str:
        if probability >= threshold:
            return "HIGH"
        medium_cutoff = max(0.0, min(1.0, threshold * 0.60))
        if probability >= medium_cutoff:
            return "MEDIUM"
        return "LOW"
