from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.common import ARTIFACTS_DIR, ensure_directories, now_iso, setup_logger, write_json

log = setup_logger("models.preprocessing", "models/preprocessing.log")


def _suffix_match(columns: list[str], suffixes: tuple[str, ...]) -> list[str]:
    return [column for column in columns if column.endswith(suffixes)]


@dataclass
class ICUPreprocessor:
    target: str
    exclude_leaky: bool = True
    feature_names_: list[str] | None = None
    imputer_medians_: dict[str, float] = field(default_factory=dict)
    categorical_encodings_: dict[str, dict[str, float]] = field(default_factory=dict)
    features_to_drop_: list[str] = field(default_factory=list)
    is_fitted_: bool = False
    feature_groups_: dict[str, list[str]] = field(default_factory=dict)

    ID_COLS = ["patientunitstayid", "window_id", "window_start_hour", "window_end_hour", "window_start_min", "window_end_min"]
    EXCLUDE_RAW_STRING_COLS = [
        "gender",
        "ethnicity",
        "hospitaldischargestatus",
        "unittype",
        "unitadmitsource",
        "unitdischargestatus",
    ]
    STATIC_CANDIDATES = [
        "age",
        "admissionweight_kg",
        "admissionheight_cm",
        "bmi",
        "apache_score",
        "apache_predicted_mort",
        "apache_predicted_icu_mort",
        "hospitaladmit_offset_h",
        "vital_count_24h",
        "comorbid_count",
        "diag_count",
    ]
    VITAL_CANDIDATES = [
        "hr_mean", "hr_min", "hr_max", "hr_std", "hr_obs_count",
        "sao2_mean", "sao2_min", "sao2_std",
        "resp_mean", "resp_max", "resp_std",
        "sbp_mean", "sbp_min", "sbp_std",
        "dbp_mean", "map_mean", "temp_mean", "temp_max", "cvp_mean",
        "nibp_systolic_mean", "nibp_mean_mean",
        "fluid_intake_window", "fluid_output_window", "urine_output_window",
        "fluid_balance_window", "urine_rate_mlhr", "peep_mean", "fio2_mean",
        "norepinephrine_dose", "gcs_total", "pain_score", "gcs_motor",
        "gcs_verbal", "gcs_eye", "rass_score", "cam_icu",
    ]
    LAB_CANDIDATES = [
        "lactate_value", "creatinine_value", "glucose_value", "hemoglobin_value",
        "wbc_value", "potassium_value", "sodium_value", "bicarbonate_value",
        "bun_value", "platelets_value", "inr_value", "ph_value", "pco2_value",
        "po2_value", "pf_ratio", "alt_value", "ast_value",
    ]
    SCORE_CANDIDATES = ["qsofa_score", "sofa_partial", "sirs_score", "shock_index", "news_score"]
    DELTA_CANDIDATES = ["hr_delta", "sao2_delta", "sbp_delta", "resp_delta", "temp_delta"]
    CUMULATIVE_CANDIDATES = [
        "cumulative_fluid_balance", "cumulative_vasopressor_h", "cumulative_ventilator_h",
        "max_lactate_so_far", "min_sao2_so_far", "min_sbp_so_far", "min_gcs_so_far",
    ]
    CATEGORICAL_CANDIDATES = ["apache_diagnosis"]

    def export_config(self) -> dict:
        if not self.is_fitted_:
            raise RuntimeError("Preprocessor must be fitted before exporting config")
        return {
            "target": self.target,
            "exclude_leaky": self.exclude_leaky,
            "feature_names": self.feature_names_,
            "imputer_medians": self.imputer_medians_,
            "categorical_encodings": self.categorical_encodings_,
            "features_to_drop": self.features_to_drop_,
            "fit_timestamp": now_iso(),
            "feature_groups": self.feature_groups_,
        }

    def _identify_feature_groups(self, df: pd.DataFrame) -> dict[str, list[str]]:
        columns = df.columns.tolist()
        label_cols = [column for column in columns if column.startswith("label_")]
        binary_suffixes = (
            "_active", "_measured", "_below_90", "_above_25", "_fever", "_hypothermia",
            "_positive", "_high", "_low", "_hypoglycemia", "_hyperglycemia", "_acidosis",
            "_alkalosis", "_abnormal", "_male", "_white", "_black", "_hispanic",
            "_micu", "_sicu", "_ccu", "_csicu", "_neuro", "_cardiac", "_ed", "_floor", "_or",
        )
        binary_candidates = set(_suffix_match(columns, binary_suffixes)) | {
            "vasopressor_active", "ventilator_active", "oliguria", "gcs_below_8",
            "gcs_measured", "cam_icu_positive", "sao2_below_90", "resp_above_25",
            "sbp_below_90", "map_below_65", "temp_fever", "temp_hypothermia",
            "nibp_below_90", "peep_above_10", "fio2_above_60",
        }
        static_cols = [column for column in self.STATIC_CANDIDATES if column in columns]
        vital_cols = [column for column in self.VITAL_CANDIDATES if column in columns]
        lab_cols = [column for column in self.LAB_CANDIDATES if column in columns]
        score_cols = [column for column in self.SCORE_CANDIDATES if column in columns]
        delta_cols = [column for column in self.DELTA_CANDIDATES if column in columns]
        cumulative_cols = [column for column in self.CUMULATIVE_CANDIDATES if column in columns]
        categorical_cols = [column for column in self.CATEGORICAL_CANDIDATES if column in columns]
        binary_cols = [column for column in columns if column in binary_candidates]
        leaky_features = []
        if self.exclude_leaky:
            leaky_features.extend([column for column in ["apache_score", "apache_predicted_mort", "apache_predicted_icu_mort"] if column in columns])
        feature_groups = {
            "id_cols": [column for column in self.ID_COLS if column in columns],
            "label_cols": label_cols,
            "binary_cols": sorted(set(binary_cols)),
            "static_cols": static_cols,
            "vital_cols": vital_cols,
            "lab_cols": lab_cols,
            "score_cols": score_cols,
            "delta_cols": delta_cols,
            "cumulative_cols": cumulative_cols,
            "categorical_cols": categorical_cols,
            "drop_raw_cols": [column for column in self.EXCLUDE_RAW_STRING_COLS if column in columns] + [column for column in ["patienthealthsystemstayid", "hospitalid", "hospitaladmitoffset", "hospital_mortality", "icu_mortality", "icu_los_hours", "unitdischargeoffset"] if column in columns],
            "leaky_cols": leaky_features,
        }
        return feature_groups

    def fit(self, train_df: pd.DataFrame) -> "ICUPreprocessor":
        ensure_directories()
        self.feature_groups_ = self._identify_feature_groups(train_df)
        median_cols = (
            self.feature_groups_["vital_cols"]
            + self.feature_groups_["lab_cols"]
            + self.feature_groups_["cumulative_cols"]
            + self.feature_groups_["static_cols"]
        )
        for column in median_cols:
            self.imputer_medians_[column] = float(train_df[column].median()) if column in train_df.columns else 0.0
        for column in self.feature_groups_["score_cols"]:
            self.imputer_medians_[column] = 0.0
        for column in self.feature_groups_["delta_cols"]:
            self.imputer_medians_[column] = 0.0

        if "apache_diagnosis" in train_df.columns:
            freq = train_df["apache_diagnosis"].fillna("UNKNOWN").astype(str).value_counts(normalize=True)
            self.categorical_encodings_["apache_diagnosis"] = {key: float(value) for key, value in freq.to_dict().items()}

        base_feature_cols = []
        for group_name in ["static_cols", "binary_cols", "vital_cols", "lab_cols", "cumulative_cols", "delta_cols", "score_cols"]:
            base_feature_cols.extend(self.feature_groups_[group_name])
        if self.target == "label_deterioration_next" and "window_id" in train_df.columns:
            base_feature_cols.append("window_id")
        if "apache_diagnosis" in self.feature_groups_["categorical_cols"]:
            base_feature_cols.append("apache_diagnosis_enc")
        excluded = set(self.feature_groups_["id_cols"] + self.feature_groups_["label_cols"] + self.feature_groups_["drop_raw_cols"] + self.feature_groups_["leaky_cols"])
        base_feature_cols = [column for column in base_feature_cols if column not in excluded]
        numeric_feature_cols = [
            column
            for column in base_feature_cols
            if column in train_df.columns and pd.api.types.is_numeric_dtype(train_df[column])
        ]
        for column in numeric_feature_cols:
            if column not in self.imputer_medians_:
                self.imputer_medians_[column] = float(train_df[column].median()) if train_df[column].notna().any() else 0.0

        transformed = self._transform_internal(train_df, fit_mode=True)
        for column in transformed.columns:
            if pd.api.types.is_numeric_dtype(transformed[column]) and column not in self.imputer_medians_:
                self.imputer_medians_[column] = float(train_df[column].median()) if column in train_df.columns and train_df[column].notna().any() else 0.0
        transformed = self._transform_internal(train_df, fit_mode=True)
        zero_variance = [column for column in transformed.columns if transformed[column].nunique(dropna=False) <= 1]
        self.features_to_drop_ = sorted(set(zero_variance))
        self.feature_names_ = [column for column in transformed.columns if column not in self.features_to_drop_]

        self.is_fitted_ = True
        config = self.export_config()
        config["train_rows"] = int(len(train_df))
        config["train_patients"] = int(train_df["patientunitstayid"].nunique())
        config_path = ARTIFACTS_DIR / "preprocessing_config.json"
        existing_config = {}
        if config_path.exists():
            with config_path.open("r", encoding="utf-8") as handle:
                existing_config = json.load(handle)
        if "preprocessors" not in existing_config:
            if existing_config.get("feature_names"):
                existing_config = {
                    "default_target": existing_config.get("target", self.target),
                    "preprocessors": {
                        existing_config.get("target", self.target): existing_config,
                    },
                }
            else:
                existing_config = {"default_target": self.target, "preprocessors": {}}
        existing_config["default_target"] = existing_config.get("default_target", self.target)
        existing_config["preprocessors"][self.target] = config
        write_json(config_path, existing_config)
        log.info("Preprocessor fitted for %s with %s features", self.target, len(self.feature_names_))
        return self

    def _transform_internal(self, df: pd.DataFrame, fit_mode: bool = False) -> pd.DataFrame:
        transformed = df.copy()
        if not fit_mode and not self.is_fitted_:
            raise RuntimeError("Preprocessor must be fitted before transform")

        for column in self.feature_groups_["binary_cols"]:
            if column in transformed.columns:
                transformed[column] = transformed[column].fillna(0)
        for column, median in self.imputer_medians_.items():
            if column in transformed.columns:
                transformed[column] = transformed[column].fillna(median)
        if "apache_diagnosis" in transformed.columns:
            freq_map = self.categorical_encodings_.get("apache_diagnosis", {})
            fallback = min(freq_map.values()) if freq_map else 0.0
            transformed["apache_diagnosis_enc"] = (
                transformed["apache_diagnosis"].fillna("UNKNOWN").astype(str).map(freq_map).fillna(fallback)
            )

        excluded = set(self.feature_groups_["id_cols"] + self.feature_groups_["label_cols"] + self.feature_groups_["drop_raw_cols"] + self.feature_groups_["leaky_cols"])
        candidate_cols = [column for column in transformed.columns if column not in excluded]
        if "apache_diagnosis" in candidate_cols:
            candidate_cols.remove("apache_diagnosis")
        if fit_mode:
            output = transformed[candidate_cols].copy()
        else:
            for column in self.feature_names_:
                if column not in transformed.columns:
                    transformed[column] = 0.0
            output = transformed[self.feature_names_].copy()
        return output

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        transformed = self._transform_internal(df, fit_mode=False)
        transformed = transformed.drop(columns=[column for column in self.features_to_drop_ if column in transformed.columns], errors="ignore")
        transformed = transformed[self.feature_names_].copy()
        null_count = int(transformed.isnull().sum().sum())
        if null_count != 0:
            raise ValueError(f"NaN remaining after preprocessing: {null_count}")
        return transformed

    def fit_transform(self, train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame):
        self.fit(train_df)
        X_train = self.transform(train_df)
        X_val = self.transform(val_df)
        X_test = self.transform(test_df)
        y_train = train_df[self.target]
        y_val = val_df[self.target]
        y_test = test_df[self.target]
        train_mask = y_train.notna()
        val_mask = y_val.notna()
        test_mask = y_test.notna()
        return (
            X_train.loc[train_mask].reset_index(drop=True),
            y_train.loc[train_mask].reset_index(drop=True),
            X_val.loc[val_mask].reset_index(drop=True),
            y_val.loc[val_mask].reset_index(drop=True),
            X_test.loc[test_mask].reset_index(drop=True),
            y_test.loc[test_mask].reset_index(drop=True),
            train_df.loc[train_mask].reset_index(drop=True),
            val_df.loc[val_mask].reset_index(drop=True),
            test_df.loc[test_mask].reset_index(drop=True),
        )


def main() -> None:
    train = pd.read_parquet("data/ts_train.parquet")
    val = pd.read_parquet("data/ts_val.parquet")
    test = pd.read_parquet("data/ts_test.parquet")
    preprocessor = ICUPreprocessor(target="label_hospital_mortality", exclude_leaky=True)
    preprocessor.fit_transform(train, val, test)
    log.info("Saved preprocessing config to %s", ARTIFACTS_DIR / "preprocessing_config.json")


if __name__ == "__main__":
    main()
