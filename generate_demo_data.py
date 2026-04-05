"""
generate_demo_data.py
Creates synthetic but realistic ICU data for testing when eICU dataset is unavailable.
Generates .csv.gz files in the expected format so setup.py works without changes.

Run: python generate_demo_data.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
import sys
import gzip
import os

sys.path.insert(0, str(Path(__file__).parent))
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
log = logging.getLogger(__name__)

# Output to a demo directory
DEMO_DIR = Path("demo_data")
DEMO_DIR.mkdir(exist_ok=True)

np.random.seed(42)
N_PATIENTS = 2000
N_VITALS = 80000
N_LABS = 40000

UNIT_TYPES = ["MICU", "SICU", "CCU-CTICU", "Cardiac ICU", "CSICU", "Neuro ICU", "Med-Surg ICU"]
GENDERS = ["Male", "Female"]
ETHNICITIES = ["Caucasian", "African American", "Hispanic", "Asian", "Other/Unknown"]
ADMIT_SOURCES = ["Floor", "Direct Admit", "Emergency Department", "Operating Room", "Step-Down Unit"]
DIAGNOSES = [
    "sepsis|bacteremia|septicemia",
    "respiratory failure|pulmonary edema",
    "cardiac arrest|ventricular fibrillation",
    "acute kidney injury",
    "pneumonia|aspiration pneumonia",
    "COPD|emphysema",
    "congestive heart failure",
    "stroke|cerebral infarction",
    "GI bleed|upper GI hemorrhage",
    "diabetic ketoacidosis",
    "acute liver failure",
    "pancreatitis|pancreatic abscess",
    "myocardial infarction",
    "pulmonary embolism",
    "trauma|polytrauma",
]
TREATMENTS = [
    "mechanical ventilation|intubation",
    "vasopressors|norepinephrine",
    "IV antibiotics|vancomycin",
    "renal replacement therapy|hemodialysis",
    "blood transfusion",
    "insulin drip",
    "sedation|propofol",
    "parenteral nutrition",
    "central line placement",
    "foley catheter",
]
LAB_TESTS = {
    "creatinine": (1.0, 2.0, 0, 20),
    "glucose": (120, 60, 50, 500),
    "potassium": (4.0, 0.8, 2.0, 7.0),
    "sodium": (138, 6, 120, 160),
    "lactate": (2.0, 2.5, 0, 15),
    "hemoglobin": (10.0, 2.5, 5, 18),
    "wbc": (12.0, 8.0, 1, 50),
    "platelets": (180, 80, 20, 500),
    "bicarbonate": (22, 5, 10, 35),
    "bun": (25, 20, 5, 100),
    "alt": (50, 80, 5, 500),
    "ast": (50, 80, 5, 500),
    "inr": (1.5, 1.0, 0.8, 8),
    "pco2": (40, 10, 20, 80),
    "po2": (90, 30, 40, 200),
}


def gen_patients():
    log.info("Generating patients...")
    ages = np.random.normal(62, 17, N_PATIENTS).clip(18, 95).astype(int)
    apache_scores = np.random.normal(50, 20, N_PATIENTS).clip(5, 120).astype(int)
    predicted_mort = 1 / (1 + np.exp(-(apache_scores - 60) / 15))
    died = (np.random.random(N_PATIENTS) < predicted_mort).astype(int)

    df = pd.DataFrame({
        "patientunitstayid": np.arange(100000, 100000 + N_PATIENTS),
        "patienthealthsystemstayid": np.arange(200000, 200000 + N_PATIENTS),
        "age": ages.astype(str),
        "gender": np.random.choice(GENDERS, N_PATIENTS),
        "ethnicity": np.random.choice(ETHNICITIES, N_PATIENTS, p=[0.55, 0.2, 0.12, 0.08, 0.05]),
        "admissionheight": np.random.normal(170, 12, N_PATIENTS).clip(140, 210).round(1),
        "admissionweight": np.random.normal(80, 20, N_PATIENTS).clip(40, 200).round(1),
        "unittype": np.random.choice(UNIT_TYPES, N_PATIENTS),
        "unitadmitsource": np.random.choice(ADMIT_SOURCES, N_PATIENTS),
        "unitdischargestatus": np.where(died, "Expired", "Alive"),
        "hospitaldischargestatus": np.where(died, "Expired", "Alive"),
        "apacheadmissiondx": np.random.choice([d.split("|")[0] for d in DIAGNOSES], N_PATIENTS),
        "unitdischargeoffset": np.random.exponential(3000, N_PATIENTS).clip(60, 20000).astype(int),
        "hospitaladmitoffset": -np.random.randint(0, 2880, N_PATIENTS),
    })
    # Fix some ages to "> 89"
    mask = df["age"].astype(int) > 89
    df.loc[mask, "age"] = "> 89"
    return df


def gen_vitals(patient_ids):
    log.info("Generating vitals...")
    rows = []
    selected = np.random.choice(patient_ids, min(400, len(patient_ids)), replace=False)

    for pid in selected:
        n = np.random.randint(50, 300)
        offsets = np.sort(np.random.uniform(0, 5000, n))
        hr = np.random.normal(85, 18, n).clip(30, 200)
        sao2 = np.random.normal(96, 3, n).clip(70, 100)
        resp = np.random.normal(18, 5, n).clip(6, 45)
        sbp = np.random.normal(115, 25, n).clip(50, 220)
        dbp = sbp * np.random.uniform(0.55, 0.65, n)
        temp = np.random.normal(37.1, 0.8, n).clip(34, 41)
        cvp = np.random.normal(8, 4, n).clip(0, 25)

        for i in range(n):
            rows.append({
                "patientunitstayid": pid,
                "observationoffset": round(offsets[i], 1),
                "heartrate": round(hr[i], 1) if np.random.random() > 0.05 else None,
                "respiration": round(resp[i], 1) if np.random.random() > 0.08 else None,
                "sao2": round(sao2[i], 1) if np.random.random() > 0.06 else None,
                "systemicsystolic": round(sbp[i], 1) if np.random.random() > 0.1 else None,
                "systemicdiastolic": round(dbp[i], 1) if np.random.random() > 0.1 else None,
                "temperature": round(temp[i], 2) if np.random.random() > 0.15 else None,
                "cvp": round(cvp[i], 1) if np.random.random() > 0.4 else None,
            })
    return pd.DataFrame(rows)


def gen_labs(patient_ids):
    log.info("Generating labs...")
    rows = []
    selected = np.random.choice(patient_ids, min(500, len(patient_ids)), replace=False)

    for pid in selected:
        n_draws = np.random.randint(3, 20)
        offsets = np.sort(np.random.uniform(0, 4000, n_draws))
        for offset in offsets:
            n_tests = np.random.randint(3, len(LAB_TESTS))
            tests = np.random.choice(list(LAB_TESTS.keys()), n_tests, replace=False)
            for test in tests:
                mean, std, lo, hi = LAB_TESTS[test]
                value = np.random.normal(mean, std)
                value = max(lo, min(hi, value))
                rows.append({
                    "patientunitstayid": pid,
                    "labresultoffset": round(offset, 1),
                    "labname": test,
                    "labresult": round(value, 3),
                    "labresulttext": str(round(value, 2))
                })
    return pd.DataFrame(rows)


def gen_diagnosis(patient_ids):
    log.info("Generating diagnoses...")
    rows = []
    for pid in patient_ids:
        n = np.random.randint(1, 5)
        diags = np.random.choice(DIAGNOSES, n, replace=False)
        for i, d in enumerate(diags):
            rows.append({
                "patientunitstayid": pid,
                "diagnosisoffset": np.random.randint(0, 500),
                "diagnosisstring": d.split("|")[0],
                "icd9code": f"{np.random.randint(100, 999)}.{np.random.randint(0, 9)}",
                "diagnosispriority": "Primary" if i == 0 else "Secondary"
            })
    return pd.DataFrame(rows)


def gen_treatment(patient_ids):
    log.info("Generating treatments...")
    rows = []
    for pid in patient_ids:
        n = np.random.randint(1, 6)
        treats = np.random.choice(TREATMENTS, n, replace=False)
        for t in treats:
            rows.append({
                "patientunitstayid": pid,
                "treatmentoffset": np.random.randint(0, 1000),
                "treatmentstring": t.split("|")[0]
            })
    return pd.DataFrame(rows)


def gen_apache(patient_ids, mortality_labels):
    log.info("Generating APACHE scores...")
    mort_map = dict(zip(patient_ids, mortality_labels))
    rows = []
    for pid in patient_ids:
        died = mort_map[pid]
        score = float(np.clip(np.random.normal(70 if died else 40, 15), 5, 120))
        pred_mort = 1 / (1 + np.exp(-(score - 60) / 15))
        rows.append({
            "patientunitstayid": pid,
            "apachescore": round(score, 1),
            "predictedhospitalmortality": round(pred_mort, 4),
            "actualhospitalmortality": "EXPIRED" if died else "ALIVE",
            "preoperation": np.random.choice(["Yes", "No"])
        })
    return pd.DataFrame(rows)


def save_gz(df, filename):
    path = DEMO_DIR / filename
    df.to_csv(path, index=False, compression="gzip")
    log.info(f"  Saved {filename} ({len(df):,} rows, {path.stat().st_size // 1024} KB)")


def main():
    log.info("=== Generating Demo eICU Data ===")

    patients_df = gen_patients()
    patient_ids = patients_df["patientunitstayid"].values
    died_labels = (patients_df["hospitaldischargestatus"] == "Expired").astype(int).values

    save_gz(patients_df, "patient.csv.gz")
    save_gz(gen_vitals(patient_ids), "vitalPeriodic.csv.gz")
    save_gz(gen_labs(patient_ids), "lab.csv.gz")
    save_gz(gen_diagnosis(patient_ids), "diagnosis.csv.gz")
    save_gz(gen_treatment(patient_ids), "treatment.csv.gz")
    save_gz(gen_apache(patient_ids, died_labels), "apachePatientResult.csv.gz")

    log.info(f"\n✅ Demo data generated in: {DEMO_DIR.absolute()}")
    log.info("Update config/config.py → EICU_RAW_PATH = Path('demo_data') to use this data.")


if __name__ == "__main__":
    main()
