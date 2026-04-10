from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.common import setup_logger

log = setup_logger("models.pipeline", "models/run_pipeline.log")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-audit", action="store_true")
    parser.add_argument("--skip-mortality", action="store_true")
    parser.add_argument("--skip-deterioration", action="store_true")
    parser.add_argument(
        "--target",
        default="label_hospital_mortality",
        choices=["label_hospital_mortality", "label_icu_mortality", "label_deterioration_next"],
    )
    args = parser.parse_args()

    log.info("=" * 60)
    log.info("ICU MODEL DEVELOPMENT PIPELINE")
    log.info("=" * 60)

    if not args.skip_audit:
        from models.audit.dataset_audit import run_audit

        log.info("[PHASE 1] Running dataset audit")
        audit_result = run_audit("data/ts_train.parquet", "data/ts_val.parquet", "data/ts_test.parquet", "data")
        if audit_result["overall_status"] == "FAIL":
            log.error("Audit failed: %s", audit_result["critical_issues"])
            sys.exit(1)

    if not args.skip_mortality and args.target != "label_deterioration_next":
        from models.training.train_mortality import train_mortality_models

        log.info("[PHASE 2+3] Training mortality model")
        train_mortality_models(target=args.target)

    if not args.skip_deterioration:
        from models.training.train_deterioration import train_deterioration_models

        log.info("[PHASE 4] Training deterioration model")
        train_deterioration_models()

    from models.inference.predictor import ICUPredictor

    log.info("[PHASE 5] Verifying inference pipeline")
    predictor = ICUPredictor()
    test_patient = {
        "age": 65,
        "apache_score": 55,
        "hr_mean": 95,
        "sao2_mean": 93,
        "sbp_mean": 105,
        "resp_mean": 22,
        "lactate_value": 3.2,
        "creatinine_value": 1.8,
        "gcs_total": 13,
        "vasopressor_active": 1,
        "ventilator_active": 0,
        "window_id": 2,
    }
    result = predictor.predict_mortality(test_patient)
    log.info("Test prediction: %s (%.3f)", result.get("risk_label"), result.get("risk_score", 0.0))
    info = predictor.get_model_info()
    if info.get("mortality_model"):
        log.info("Mortality AUROC: %.4f", info["mortality_model"]["auroc"])
    log.info("Pipeline complete")


if __name__ == "__main__":
    main()
