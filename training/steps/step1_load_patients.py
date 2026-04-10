from __future__ import annotations

import logging
import re
from pathlib import Path

import pandas as pd

from training.utils.range_filters import filter_numeric_range, read_csv_kwargs, resolve_source_file

log = logging.getLogger(__name__)

PATIENT_COLUMNS = [
    "patientunitstayid",
    "patienthealthsystemstayid",
    "hospitalid",
    "age",
    "gender",
    "ethnicity",
    "admissionheight",
    "admissionweight",
    "unittype",
    "unitadmitsource",
    "unitdischargestatus",
    "hospitaldischargestatus",
    "apacheadmissiondx",
    "unitdischargeoffset",
    "hospitaladmitoffset",
]

COMORBIDITY_PATTERNS = {
    "comorbid_diabetes": ["diabetes", "dm type", "dmii", "type 2 diabetes", "type ii diabetes"],
    "comorbid_chf": ["heart failure", "chf", "cardiomyopathy"],
    "comorbid_copd": ["copd", "emphysema", "chronic obstructive"],
    "comorbid_ckd": ["renal failure", "ckd", "dialysis", "esrd", "chronic kidney"],
    "comorbid_hypertension": ["hypertension", "htn"],
    "comorbid_immunosup": ["immunosupp", "transplant", "hiv", "aids", "steroid dependent"],
    "comorbid_liver": ["cirrhosis", "liver disease", "hepatic", "hepatitis"],
    "comorbid_cancer": ["cancer", "malignan", "carcinoma", "leukemia", "lymphoma"],
}

DIAGNOSIS_PATTERNS = {
    "diag_sepsis": ["sepsis", "septic", "bacteremia", "septic shock"],
    "diag_respiratory": ["respiratory", "pneumonia", "copd", "asthma", "ards", "hypoxic"],
    "diag_cardiac": ["cardio", "heart", "mi", "coronary", "arrhythm", "chest pain"],
    "diag_neuro": ["stroke", "neuro", "seizure", "intracranial", "brain", "coma"],
    "diag_renal": ["renal", "kidney", "aki", "ckd", "dialysis"],
    "diag_gi": ["gi", "gastro", "liver", "pancrea", "bleed", "bowel"],
    "diag_trauma": ["trauma", "fracture", "injury", "wound", "burn"],
}


def _pattern_union(patterns: list[str]) -> str:
    return "|".join(re.escape(item) for item in patterns)


def _load_patient_table(patient_path: Path) -> pd.DataFrame:
    patients = pd.read_csv(patient_path, usecols=PATIENT_COLUMNS, **read_csv_kwargs(patient_path))
    patients["age"] = patients["age"].replace({"> 89": 90, ">89": 90})
    patients["age"] = pd.to_numeric(patients["age"], errors="coerce")
    patients["admissionweight"] = filter_numeric_range(patients["admissionweight"], 20, 400)
    patients["admissionheight"] = filter_numeric_range(patients["admissionheight"], 100, 250)
    patients["bmi"] = patients["admissionweight"] / ((patients["admissionheight"] / 100.0) ** 2)
    patients["bmi"] = patients["bmi"].clip(10, 70)
    patients["hospital_mortality"] = patients["hospitaldischargestatus"].fillna("").astype(str).str.lower().eq("expired").astype("int8")
    patients["icu_mortality"] = patients["unitdischargestatus"].fillna("").astype(str).str.lower().eq("expired").astype("int8")
    patients["icu_los_hours"] = pd.to_numeric(patients["unitdischargeoffset"], errors="coerce") / 60.0
    patients["hospitaladmit_offset_h"] = pd.to_numeric(patients["hospitaladmitoffset"], errors="coerce") / 60.0

    total_before = len(patients)
    mask_los = pd.to_numeric(patients["unitdischargeoffset"], errors="coerce") >= 1440
    log.info("Patients removed for LOS < 24h: %s", f"{(~mask_los).sum():,}")
    patients = patients.loc[mask_los].copy()

    mask_age = patients["age"].between(18, 110, inclusive="both")
    log.info("Patients removed for age outside 18-110: %s", f"{(~mask_age).sum():,}")
    patients = patients.loc[mask_age].copy()

    before_dedup = len(patients)
    patients = patients.drop_duplicates(subset=["patientunitstayid"], keep="first").copy()
    log.info("Duplicate patientunitstayid rows removed: %s", f"{before_dedup - len(patients):,}")
    log.info("Patients retained after base filters: %s / %s", f"{len(patients):,}", f"{total_before:,}")
    return patients


def _load_apache(eicu_path: Path) -> pd.DataFrame:
    apache_path = resolve_source_file(eicu_path, "apachePatientResult")
    usecols = ["patientunitstayid", "apachescore", "predictedhospitalmortality", "predictedicumortality"]
    apache = pd.read_csv(apache_path, usecols=usecols, **read_csv_kwargs(apache_path))
    apache["apachescore"] = filter_numeric_range(apache["apachescore"], 0, 300)
    apache["predictedhospitalmortality"] = filter_numeric_range(apache["predictedhospitalmortality"], 0, 1)
    apache["predictedicumortality"] = filter_numeric_range(apache["predictedicumortality"], 0, 1)
    return apache.groupby("patientunitstayid", as_index=False).agg(
        apache_score=("apachescore", "max"),
        apache_predicted_mort=("predictedhospitalmortality", "max"),
        apache_predicted_icu_mort=("predictedicumortality", "max"),
    )


def _load_past_history(eicu_path: Path) -> pd.DataFrame:
    path = resolve_source_file(eicu_path, "pastHistory")
    history = pd.read_csv(path, usecols=["patientunitstayid", "pasthistoryvalue"], **read_csv_kwargs(path))
    history["pasthistoryvalue"] = history["pasthistoryvalue"].fillna("").astype(str).str.lower()
    merged = history[["patientunitstayid"]].drop_duplicates().copy()
    for column, patterns in COMORBIDITY_PATTERNS.items():
        mask = history["pasthistoryvalue"].str.contains(_pattern_union(patterns), regex=True, na=False)
        flagged = history.loc[mask, ["patientunitstayid"]].drop_duplicates().copy()
        flagged[column] = 1
        merged = merged.merge(flagged, on="patientunitstayid", how="left")
    feature_cols = list(COMORBIDITY_PATTERNS)
    merged[feature_cols] = merged[feature_cols].fillna(0).astype("int8")
    merged["comorbid_count"] = merged[feature_cols].sum(axis=1).astype("int16")
    return merged


def _load_diagnosis(eicu_path: Path) -> pd.DataFrame:
    path = resolve_source_file(eicu_path, "diagnosis")
    diagnosis = pd.read_csv(path, usecols=["patientunitstayid", "diagnosisstring"], **read_csv_kwargs(path))
    diagnosis["diagnosisstring"] = diagnosis["diagnosisstring"].fillna("").astype(str).str.lower()
    diagnosis = diagnosis.drop_duplicates()
    merged = diagnosis[["patientunitstayid"]].drop_duplicates().copy()
    for column, patterns in DIAGNOSIS_PATTERNS.items():
        mask = diagnosis["diagnosisstring"].str.contains(_pattern_union(patterns), regex=True, na=False)
        flagged = diagnosis.loc[mask, ["patientunitstayid"]].drop_duplicates().copy()
        flagged[column] = 1
        merged = merged.merge(flagged, on="patientunitstayid", how="left")
    feature_cols = list(DIAGNOSIS_PATTERNS)
    merged[feature_cols] = merged[feature_cols].fillna(0).astype("int8")
    diag_counts = diagnosis.groupby("patientunitstayid").size().rename("diag_count").astype("int16").reset_index()
    merged = merged.merge(diag_counts, on="patientunitstayid", how="left")
    merged["diag_count"] = merged["diag_count"].fillna(0).astype("int16")
    return merged


def _load_vital_counts(eicu_path: Path, patient_ids: pd.Series) -> pd.DataFrame:
    path = resolve_source_file(eicu_path, "vitalPeriodic")
    counts: list[pd.DataFrame] = []
    usecols = ["patientunitstayid", "observationoffset"]
    allowed_ids = set(patient_ids.tolist())
    for chunk in pd.read_csv(path, usecols=usecols, chunksize=50_000, **read_csv_kwargs(path)):
        chunk = chunk[chunk["patientunitstayid"].isin(allowed_ids)].copy()
        chunk["observationoffset"] = pd.to_numeric(chunk["observationoffset"], errors="coerce")
        chunk = chunk[chunk["observationoffset"].between(0, 1439, inclusive="both")]
        if chunk.empty:
            continue
        counts.append(chunk.groupby("patientunitstayid").size().rename("vital_count_24h").reset_index())
    if not counts:
        return pd.DataFrame(columns=["patientunitstayid", "vital_count_24h"])
    merged = pd.concat(counts, ignore_index=True).groupby("patientunitstayid", as_index=False)["vital_count_24h"].sum()
    return merged


def _encode_static_features(patients: pd.DataFrame) -> pd.DataFrame:
    patients = patients.copy()
    gender = patients["gender"].fillna("").astype(str).str.lower()
    ethnicity = patients["ethnicity"].fillna("").astype(str).str.lower()
    unit_type = patients["unittype"].fillna("").astype(str).str.lower()
    admit_source = patients["unitadmitsource"].fillna("").astype(str).str.lower()
    patients["gender_male"] = gender.eq("male").astype("int8")
    patients["is_female"] = gender.eq("female").astype("int8")
    patients["ethnicity_white"] = ethnicity.str.contains("caucasian|white", regex=True, na=False).astype("int8")
    patients["ethnicity_black"] = ethnicity.str.contains("african|black", regex=True, na=False).astype("int8")
    patients["ethnicity_hispanic"] = ethnicity.str.contains("hispanic|latino", regex=True, na=False).astype("int8")
    patients["unittype_micu"] = unit_type.str.contains("med-surg icu|medical icu|micu", regex=True, na=False).astype("int8")
    patients["unittype_sicu"] = unit_type.str.contains(r"\bsicu\b|surgical icu", regex=True, na=False).astype("int8")
    patients["unittype_ccu"] = unit_type.str.contains(r"\bccu\b|coronary care", regex=True, na=False).astype("int8")
    patients["unittype_csicu"] = unit_type.str.contains("cardiac surgery icu|csicu", regex=True, na=False).astype("int8")
    patients["unittype_neuro"] = unit_type.str.contains("neuro", regex=True, na=False).astype("int8")
    patients["unittype_cardiac"] = unit_type.str.contains("cardiac", regex=True, na=False).astype("int8")
    patients["unitsource_ed"] = admit_source.str.contains("emergency|ed", regex=True, na=False).astype("int8")
    patients["unitsource_floor"] = admit_source.str.contains("floor", regex=True, na=False).astype("int8")
    patients["unitsource_or"] = admit_source.str.contains(r"\bor\b|operating room", regex=True, na=False).astype("int8")
    return patients.rename(
        columns={
            "admissionweight": "admissionweight_kg",
            "admissionheight": "admissionheight_cm",
            "apacheadmissiondx": "apache_diagnosis",
        }
    )


def load_and_filter_patients(eicu_path: Path, max_patients: int) -> pd.DataFrame:
    patients = _encode_static_features(_load_patient_table(resolve_source_file(eicu_path, "patient")))
    vital_counts = _load_vital_counts(eicu_path, patients["patientunitstayid"])
    patients = patients.merge(vital_counts, on="patientunitstayid", how="left")
    patients["vital_count_24h"] = patients["vital_count_24h"].fillna(0).astype("int32")
    before_vital_filter = len(patients)
    patients = patients[patients["vital_count_24h"] >= 10].copy()
    log.info("Patients removed for <10 vitalPeriodic readings in first 24h: %s", f"{before_vital_filter - len(patients):,}")
    patients = patients.merge(_load_apache(eicu_path), on="patientunitstayid", how="left")
    patients = patients.merge(_load_past_history(eicu_path), on="patientunitstayid", how="left")
    patients = patients.merge(_load_diagnosis(eicu_path), on="patientunitstayid", how="left")

    fill_zero_cols = [column for column in patients.columns if column.startswith("comorbid_") or column.startswith("diag_")]
    patients[fill_zero_cols] = patients[fill_zero_cols].fillna(0)

    if len(patients) > max_patients:
        mortality_counts = patients["hospital_mortality"].value_counts().sort_index()
        target_counts = (mortality_counts / mortality_counts.sum() * max_patients).round().astype(int)
        diff = max_patients - int(target_counts.sum())
        if diff != 0:
            target_counts.loc[mortality_counts.idxmax()] += diff
        sampled_frames = []
        for label, frame in patients.groupby("hospital_mortality", observed=True):
            target_n = min(int(target_counts.get(label, 0)), len(frame))
            sampled_frames.append(frame.sample(n=target_n, random_state=42))
        patients = pd.concat(sampled_frames, ignore_index=True)
        log.info("Applied stratified sampling to %s patients", f"{len(patients):,}")

    return patients.sort_values("patientunitstayid").reset_index(drop=True)