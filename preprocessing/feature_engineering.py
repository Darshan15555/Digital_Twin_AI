from __future__ import annotations

from preprocessing.cleaning import clean_text, iqr_clip, safe_bigint, safe_numeric


def patient_base_sql(raw_schema: str) -> str:
    return f"""
    CREATE TABLE "{{target_schema}}"."patient_base_clean" AS
    WITH patient_dedup AS (
        SELECT DISTINCT ON ({safe_bigint("patientunitstayid")})
            {safe_bigint("patientunitstayid")} AS patientunitstayid,
            {safe_bigint("patienthealthsystemstayid")} AS patienthealthsystemstayid,
            {safe_bigint("hospitalid")} AS hospitalid,
            {clean_text("gender")} AS gender,
            CASE
                WHEN age IS NULL THEN NULL
                WHEN trim(age::text) IN ('', 'nan', 'None', '<NA>') THEN NULL
                WHEN trim(age::text) IN ('> 89', '>89') THEN 90
                WHEN trim(age::text) ~ '^[0-9]+(\\.[0-9]+)?$' THEN round(trim(age::text)::numeric)::int
                ELSE NULL
            END AS age_years,
            {clean_text("ethnicity")} AS ethnicity,
            {safe_numeric("admissionheight")} AS admissionheight,
            {safe_numeric("admissionweight")} AS admissionweight,
            {clean_text("hospitaladmitsource")} AS hospitaladmitsource,
            {clean_text("hospitaldischargestatus")} AS hospitaldischargestatus,
            {clean_text("unitadmitsource")} AS unitadmitsource,
            {clean_text("unitdischargestatus")} AS unitdischargestatus,
            {safe_numeric("unitdischargeoffset")} AS unitdischargeoffset,
            {safe_numeric("hospitaldischargeoffset")} AS hospitaldischargeoffset,
            {clean_text("apacheadmissiondx")} AS apacheadmissiondx
        FROM "{raw_schema}"."patient"
        WHERE {safe_bigint("patientunitstayid")} IS NOT NULL
        ORDER BY {safe_bigint("patientunitstayid")}
    ),
    stats AS (
        SELECT
            percentile_cont(0.5) WITHIN GROUP (ORDER BY age_years) AS age_median,
            percentile_cont(0.5) WITHIN GROUP (ORDER BY admissionheight) AS height_median,
            percentile_cont(0.5) WITHIN GROUP (ORDER BY admissionweight) AS weight_median,
            percentile_cont(0.5) WITHIN GROUP (ORDER BY unitdischargeoffset) AS unit_offset_median,
            percentile_cont(0.5) WITHIN GROUP (ORDER BY hospitaldischargeoffset) AS hospital_offset_median,
            percentile_cont(0.25) WITHIN GROUP (ORDER BY admissionheight) AS height_q1,
            percentile_cont(0.75) WITHIN GROUP (ORDER BY admissionheight) AS height_q3,
            percentile_cont(0.25) WITHIN GROUP (ORDER BY admissionweight) AS weight_q1,
            percentile_cont(0.75) WITHIN GROUP (ORDER BY admissionweight) AS weight_q3
        FROM patient_dedup
    )
    SELECT
        p.patientunitstayid,
        p.patienthealthsystemstayid,
        p.hospitalid,
        p.gender,
        coalesce(p.age_years, round(s.age_median)::int, 0) AS age_years,
        CASE
            WHEN coalesce(p.age_years, round(s.age_median)::int, 0) < 18 THEN 'young'
            WHEN coalesce(p.age_years, round(s.age_median)::int, 0) < 60 THEN 'adult'
            ELSE 'senior'
        END AS age_group,
        p.ethnicity,
        {iqr_clip("coalesce(p.admissionheight, s.height_median)", "s.height_q1", "s.height_q3")} AS admissionheight,
        {iqr_clip("coalesce(p.admissionweight, s.weight_median)", "s.weight_q1", "s.weight_q3")} AS admissionweight,
        p.hospitaladmitsource,
        p.hospitaldischargestatus,
        p.unitadmitsource,
        p.unitdischargestatus,
        coalesce(p.unitdischargeoffset, s.unit_offset_median, 0) AS unitdischargeoffset,
        coalesce(p.hospitaldischargeoffset, s.hospital_offset_median, 0) AS hospitaldischargeoffset,
        round((coalesce(p.unitdischargeoffset, s.unit_offset_median, 0) / 60.0)::numeric, 2) AS icu_los_hours,
        round((coalesce(p.hospitaldischargeoffset, s.hospital_offset_median, 0) / 60.0)::numeric, 2) AS hospital_los_hours,
        CASE
            WHEN coalesce(p.unitdischargeoffset, s.unit_offset_median, 0) / 60.0 < 24 THEN 'short'
            WHEN coalesce(p.unitdischargeoffset, s.unit_offset_median, 0) / 60.0 < 72 THEN 'medium'
            ELSE 'long'
        END AS icu_stay_category,
        p.apacheadmissiondx,
        CASE
            WHEN lower(coalesce(p.hospitaldischargestatus, '')) LIKE '%expired%' THEN 1
            ELSE 0
        END AS hospital_mortality,
        CASE
            WHEN lower(coalesce(p.unitdischargestatus, '')) LIKE '%expired%' THEN 1
            ELSE 0
        END AS unit_mortality
    FROM patient_dedup p
    CROSS JOIN stats s;
    """


def lab_features_sql(raw_schema: str, lab_source_table: str) -> str:
    return f"""
    CREATE TABLE "{{target_schema}}"."lab_features" AS
    WITH lab_clean AS (
        SELECT
            {safe_bigint("patientunitstayid")} AS patientunitstayid,
            {safe_numeric("labresultoffset")} AS labresultoffset,
            lower({clean_text("labname")}) AS labname,
            {safe_numeric("labresult")} AS labresult
        FROM "{raw_schema}"."{lab_source_table}"
        WHERE {safe_bigint("patientunitstayid")} IS NOT NULL
    )
    SELECT
        patientunitstayid,
        COUNT(*) AS lab_row_count,
        COUNT(DISTINCT labname) AS lab_name_count,
        MIN(labresultoffset) AS first_lab_offset,
        MAX(labresultoffset) AS last_lab_offset,
        AVG(labresult) AS lab_result_mean,
        MIN(labresult) AS lab_result_min,
        MAX(labresult) AS lab_result_max,
        AVG(labresult) FILTER (WHERE labname LIKE 'creatinine%') AS creatinine_mean,
        MAX(labresult) FILTER (WHERE labname LIKE 'creatinine%') AS creatinine_max,
        AVG(labresult) FILTER (WHERE labname LIKE 'glucose%') AS glucose_mean,
        MIN(labresult) FILTER (WHERE labname LIKE 'glucose%') AS glucose_min,
        MAX(labresult) FILTER (WHERE labname LIKE 'glucose%') AS glucose_max,
        AVG(labresult) FILTER (WHERE labname LIKE 'lactate%') AS lactate_mean,
        MAX(labresult) FILTER (WHERE labname LIKE 'lactate%') AS lactate_max,
        AVG(labresult) FILTER (WHERE labname LIKE 'wbc%') AS wbc_mean,
        MAX(labresult) FILTER (WHERE labname LIKE 'wbc%') AS wbc_max,
        AVG(labresult) FILTER (WHERE labname LIKE 'potassium%') AS potassium_mean,
        MIN(labresult) FILTER (WHERE labname LIKE 'potassium%') AS potassium_min,
        MAX(labresult) FILTER (WHERE labname LIKE 'potassium%') AS potassium_max,
        AVG(labresult) FILTER (WHERE labname LIKE 'sodium%') AS sodium_mean,
        AVG(labresult) FILTER (WHERE labname LIKE 'hemoglobin%') AS hemoglobin_mean,
        MIN(labresult) FILTER (WHERE labname LIKE 'hemoglobin%') AS hemoglobin_min,
        AVG(labresult) FILTER (WHERE labname LIKE 'platelet%') AS platelet_mean,
        MAX(labresult) FILTER (WHERE labname LIKE 'platelet%') AS platelet_max
    FROM lab_clean
    GROUP BY patientunitstayid;
    """


def vital_features_sql(raw_schema: str) -> str:
    return f"""
    CREATE TABLE "{{target_schema}}"."vital_features" AS
    WITH vital_clean AS (
        SELECT
            {safe_bigint("patientunitstayid")} AS patientunitstayid,
            {safe_numeric("observationoffset")} AS observationoffset,
            {safe_numeric("temperature")} AS temperature,
            {safe_numeric("sao2")} AS sao2,
            {safe_numeric("heartrate")} AS heartrate,
            {safe_numeric("respiration")} AS respiration,
            {safe_numeric("systemicsystolic")} AS systemicsystolic,
            {safe_numeric("systemicdiastolic")} AS systemicdiastolic,
            {safe_numeric("systemicmean")} AS systemicmean
        FROM "{raw_schema}"."vital_periodic"
        WHERE {safe_bigint("patientunitstayid")} IS NOT NULL
    ),
    vital_agg AS (
        SELECT
            patientunitstayid,
            COUNT(*) AS vital_row_count,
            MIN(observationoffset) AS first_vital_offset,
            MAX(observationoffset) AS last_vital_offset,
            AVG(temperature) AS temperature_mean,
            MIN(temperature) AS temperature_min,
            MAX(temperature) AS temperature_max,
            AVG(sao2) AS sao2_mean,
            MIN(sao2) AS sao2_min,
            AVG(heartrate) AS heartrate_mean,
            MIN(heartrate) AS heartrate_min,
            MAX(heartrate) AS heartrate_max,
            AVG(respiration) AS respiration_mean,
            MAX(respiration) AS respiration_max,
            AVG(systemicsystolic) AS sbp_mean,
            MIN(systemicsystolic) AS sbp_min,
            MAX(systemicsystolic) AS sbp_max,
            AVG(systemicdiastolic) AS dbp_mean,
            AVG(systemicmean) AS map_mean
        FROM vital_clean
        GROUP BY patientunitstayid
    ),
    vital_trend AS (
        SELECT
            patientunitstayid,
            regr_slope(heartrate, observationoffset) AS heartrate_trend,
            regr_slope(temperature, observationoffset) AS temperature_trend,
            regr_slope(sao2, observationoffset) AS sao2_trend
        FROM vital_clean
        WHERE observationoffset IS NOT NULL
        GROUP BY patientunitstayid
    )
    SELECT
        a.*,
        t.heartrate_trend,
        t.temperature_trend,
        t.sao2_trend
    FROM vital_agg a
    LEFT JOIN vital_trend t
        ON a.patientunitstayid = t.patientunitstayid;
    """


def diagnosis_features_sql(raw_schema: str) -> str:
    return f"""
    CREATE TABLE "{{target_schema}}"."diagnosis_features" AS
    WITH dx_clean AS (
        SELECT
            {safe_bigint("patientunitstayid")} AS patientunitstayid,
            lower({clean_text("admitdxname")}) AS admitdxname
        FROM "{raw_schema}"."admission_dx"
        WHERE {safe_bigint("patientunitstayid")} IS NOT NULL
    )
    SELECT
        patientunitstayid,
        COUNT(*) AS diagnosis_row_count,
        COUNT(DISTINCT admitdxname) AS diagnosis_distinct_count,
        MAX(CASE WHEN admitdxname LIKE '%sepsis%' THEN 1 ELSE 0 END) AS dx_sepsis_flag,
        MAX(CASE WHEN admitdxname LIKE '%respiratory%' THEN 1 ELSE 0 END) AS dx_respiratory_flag,
        MAX(CASE WHEN admitdxname LIKE '%cardiac%' OR admitdxname LIKE '%heart%' THEN 1 ELSE 0 END) AS dx_cardiac_flag,
        MAX(CASE WHEN admitdxname LIKE '%renal%' OR admitdxname LIKE '%kidney%' THEN 1 ELSE 0 END) AS dx_renal_flag
    FROM dx_clean
    GROUP BY patientunitstayid;
    """
