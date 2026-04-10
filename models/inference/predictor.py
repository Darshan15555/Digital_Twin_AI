from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.common import setup_logger

log = setup_logger("models.predictor", "models/predictor.log")


class ICUPredictor:
    def __init__(self, artifacts_dir: str = "models/artifacts") -> None:
        self.artifacts_dir = Path(artifacts_dir)
        self._load_all_artifacts()

    def _load_all_artifacts(self) -> None:
        raw_prep_config = json.loads((self.artifacts_dir / "preprocessing_config.json").read_text(encoding="utf-8"))
        if "preprocessors" in raw_prep_config:
            self.preprocessors = raw_prep_config["preprocessors"]
            self.default_target = raw_prep_config.get("default_target", "label_hospital_mortality")
        else:
            target = raw_prep_config.get("target", "label_hospital_mortality")
            self.preprocessors = {target: raw_prep_config}
            self.default_target = target
        self.thresholds = json.loads((self.artifacts_dir / "thresholds.json").read_text(encoding="utf-8")) if (self.artifacts_dir / "thresholds.json").exists() else {}
        self.metadata = json.loads((self.artifacts_dir / "model_metadata.json").read_text(encoding="utf-8")) if (self.artifacts_dir / "model_metadata.json").exists() else {}
        models_loaded = {}
        for model_file in self.artifacts_dir.glob("*.pkl"):
            with model_file.open("rb") as handle:
                models_loaded[model_file.stem] = pickle.load(handle)
        self.mortality_model = models_loaded.get("mortality_lgbm") or models_loaded.get("mortality_xgb")
        self.deterioration_model = models_loaded.get("deterioration_lgbm")
        default_config = self._get_prep_config(self.default_target)
        self.feature_names = default_config["feature_names"]
        self.imputer_medians = default_config["imputer_medians"]
        self.categorical_encodings = default_config["categorical_encodings"]
        log.info("Loaded %s models and %s preprocessors", len(models_loaded), len(self.preprocessors))

    def _get_prep_config(self, target: str) -> dict:
        if target in self.preprocessors:
            return self.preprocessors[target]
        mortality_fallbacks = ("label_hospital_mortality", "label_icu_mortality")
        if target in mortality_fallbacks:
            for candidate in mortality_fallbacks:
                if candidate in self.preprocessors:
                    return self.preprocessors[candidate]
        raise KeyError(f"No preprocessing config available for target '{target}'")

    def preprocess_row(self, patient_data: dict, target: str = "label_hospital_mortality") -> pd.DataFrame:
        prep_config = self._get_prep_config(target)
        feature_names = prep_config["feature_names"]
        imputer_medians = prep_config.get("imputer_medians", {})
        categorical_encodings = prep_config.get("categorical_encodings", {})
        row = {}
        freq_map = categorical_encodings.get("apache_diagnosis", {})
        fallback = min(freq_map.values()) if freq_map else 0.0
        for feature in feature_names:
            if feature == "apache_diagnosis_enc":
                raw_label = str(patient_data.get("apache_diagnosis", "UNKNOWN"))
                row[feature] = float(freq_map.get(raw_label, fallback))
                continue
            raw_val = patient_data.get(feature)
            if raw_val is None or (isinstance(raw_val, float) and np.isnan(raw_val)):
                row[feature] = float(imputer_medians.get(feature, 0.0))
            else:
                try:
                    row[feature] = float(raw_val)
                except (TypeError, ValueError):
                    row[feature] = float(imputer_medians.get(feature, 0.0))
        return pd.DataFrame([row], columns=feature_names)

    def predict_mortality(self, patient_data: dict, threshold_type: str = "max_f1") -> dict:
        if self.mortality_model is None:
            return {"error": "Mortality model not loaded", "model_ready": False}
        try:
            X = self.preprocess_row(patient_data, target="label_hospital_mortality")
            threshold = float(self.thresholds.get("mortality", {}).get(threshold_type, 0.5))
            proba = float(self.mortality_model.predict_proba(X)[0][1])
            prediction = int(proba >= threshold)
            if proba >= 0.75:
                risk_label = "CRITICAL"
            elif proba >= 0.50:
                risk_label = "HIGH"
            elif proba >= 0.30:
                risk_label = "MODERATE"
            else:
                risk_label = "LOW"
            top_features = []
            if hasattr(self.mortality_model, "feature_importances_"):
                importance_pairs = list(zip(X.columns.tolist(), self.mortality_model.feature_importances_))
                top_features = sorted(importance_pairs, key=lambda item: -item[1])[:8]
            return {
                "risk_score": round(proba, 4),
                "risk_label": risk_label,
                "prediction": prediction,
                "threshold_used": threshold,
                "threshold_type": threshold_type,
                "top_features": top_features,
                "model_ready": True,
                "interpretation": {
                    "CRITICAL": "Critical risk. Immediate review required.",
                    "HIGH": "High risk. Intensify monitoring.",
                    "MODERATE": "Moderate risk. Monitor closely.",
                    "LOW": "Low risk. Standard protocol.",
                }[risk_label],
            }
        except Exception as exc:
            log.error("predict_mortality error: %s", exc, exc_info=True)
            return {"risk_score": 0.0, "risk_label": "UNKNOWN", "prediction": 0, "model_ready": False, "error": str(exc)}

    def predict_deterioration(self, patient_data: dict, window_id: int, threshold_type: str = "max_f1") -> dict:
        if window_id == 5:
            return {"error": "Last window, no next window", "model_ready": False}
        if self.deterioration_model is None:
            return {"error": "Deterioration model not loaded", "model_ready": False}
        payload = dict(patient_data)
        payload["window_id"] = window_id
        try:
            X = self.preprocess_row(payload, target="label_deterioration_next")
            threshold = float(self.thresholds.get("deterioration", {}).get(threshold_type, 0.5))
            proba = float(self.deterioration_model.predict_proba(X)[0][1])
            return {
                "deterioration_risk": round(proba, 4),
                "prediction": int(proba >= threshold),
                "threshold_used": threshold,
                "threshold_type": threshold_type,
                "model_ready": True,
            }
        except Exception as exc:
            log.error("predict_deterioration error: %s", exc, exc_info=True)
            return {"deterioration_risk": 0.0, "prediction": 0, "model_ready": False, "error": str(exc)}

    def predict_batch(self, patient_windows: list[dict]) -> dict:
        per_window = []
        for item in patient_windows:
            per_window.append(self.predict_mortality(item))
        scores = [result.get("risk_score", 0.0) for result in per_window if result.get("model_ready")]
        if not scores:
            return {"error": "No valid predictions", "per_window": per_window}
        max_score = max(scores)
        max_index = scores.index(max_score)
        if max_score >= 0.75:
            overall = "CRITICAL"
        elif max_score >= 0.50:
            overall = "HIGH"
        elif max_score >= 0.30:
            overall = "MODERATE"
        else:
            overall = "LOW"
        return {
            "max_risk_score": max_score,
            "risk_trajectory": scores,
            "overall_risk_label": overall,
            "most_alarming_window": int(patient_windows[max_index].get("window_id", max_index)),
            "per_window": per_window,
        }

    def estimate_prediction_confidence(self, mortality_risk: float, n_similar_patients: int) -> dict:
        risk = float(mortality_risk)
        supports = int(n_similar_patients)
        if supports < 10:
            level = "LOW"
            note = "limited similar historical support"
        elif supports < 50:
            level = "MEDIUM"
            note = "moderate historical support"
        else:
            level = "HIGH"
            note = "strong historical support"
        if risk >= 0.85 and level != "HIGH":
            level = "MEDIUM"
            note = f"{note}; high-risk predictions should be clinically verified"
        return {
            "confidence_level": level,
            "n_similar_patients": supports,
            "mortality_risk": round(risk, 4),
            "confidence_text": f"Confidence: {level} | Based on {supports} similar patients ({note})",
        }

    def get_model_info(self) -> dict:
        info = {}
        mortality_meta = self.metadata.get("mortality")
        if mortality_meta:
            info["mortality_model"] = {
                "type": mortality_meta.get("model_type"),
                "auroc": mortality_meta.get("performance", {}).get("test", {}).get("auroc"),
                "auprc": mortality_meta.get("performance", {}).get("test", {}).get("auprc"),
                "threshold": mortality_meta.get("recommended_threshold"),
                "trained_at": mortality_meta.get("trained_at"),
                "features": mortality_meta.get("n_features"),
                "target": mortality_meta.get("preprocessor_target"),
            }
        deterioration_meta = self.metadata.get("deterioration")
        info["deterioration_model"] = None
        if deterioration_meta:
            info["deterioration_model"] = {
                "type": deterioration_meta.get("model_type"),
                "auroc": deterioration_meta.get("performance", {}).get("test", {}).get("auroc"),
                "auprc": deterioration_meta.get("performance", {}).get("test", {}).get("auprc"),
                "threshold": deterioration_meta.get("recommended_threshold"),
                "trained_at": deterioration_meta.get("trained_at"),
                "features": deterioration_meta.get("n_features"),
                "target": deterioration_meta.get("preprocessor_target"),
            }
        return info
