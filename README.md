# ICU Analytics System

ICU analytics, ML, and dashboarding project built on the eICU Collaborative Research Database v2.0.

## Project Structure

```text
icu_analytics_system/
├── backend/
│   └── api.py
├── config/
│   └── config.py
├── frontend/
│   └── dashboard.py
├── ingestion/
│   ├── config.py
│   ├── stream_reader.py
│   ├── transformer.py
│   ├── db_writer.py
│   ├── pipeline.py
│   └── README.md
├── models/
│   ├── digital_twin.py
│   └── ml_model.py
├── utils/
│   ├── data_loader.py
│   ├── db_manager.py
│   └── feature_engineer.py
├── data/
│   ├── icu_data.db
│   └── parquet/
├── ingest.py
├── requirements.txt
└── setup.py
```

## Configuration

Runtime settings are loaded from [.env](/d:/sem_6/MiniProject/Project/icu_analytics_system/.env).

Example:

```env
EICU_RAW_PATH=D:\physionet-data\eicu
DATA_PATH=D:\physionet-data\eicu\eicu-collaborative-research-database-2.0
API_HOST=127.0.0.1
API_PORT=8000
DATABASE_URL=postgresql+psycopg://postgres:yourpassword@localhost:5432/icu_db
DB_SCHEMA=raw_eicu
CHUNK_SIZE=50000
INSERT_BATCH_SIZE=10000
MAX_WORKERS=2
IMPORTANT_FILES=patient.csv.gz,admissionDx.csv.gz,lab.csv.gz,vitalPeriodic.csv.gz
```

## Available Pipelines

`setup.py` remains the existing local workflow:

- validates expected source files
- preprocesses selected tables with chunked reads
- writes parquet cache to `data/parquet/`
- loads SQLite tables into `data/icu_data.db`
- builds the ML dataset and model artifacts

`ingest.py` is the new production-style raw ingestion workflow:

- auto-discovers `.csv.gz` files under `DATA_PATH`
- reads directly from gzip without manual decompression
- processes data in chunks only
- creates PostgreSQL schema and tables automatically
- inserts rows with PostgreSQL `COPY` in configurable batches
- logs per-file and per-chunk progress to `ingestion.log`
- skips malformed rows and failed chunks without stopping the whole pipeline
- supports selective loading of important files first
- accepts file selectors like `lab` as well as `lab.csv.gz`

## Setup

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Confirm the dataset path and database URL in [.env](/d:/sem_6/MiniProject/Project/icu_analytics_system/.env).

## Existing Local Workflow

Run:

```bash
python setup.py
```

This prepares the existing SQLite-backed app used by the FastAPI and Streamlit components.

## Production PostgreSQL Ingestion

Load every discovered `.csv.gz` file:

```bash
python ingest.py
```

Load only the important files:

```bash
python ingest.py --important-only
```

Load a custom subset:

```bash
python ingest.py --files patient.csv.gz admissionDx.csv.gz lab.csv.gz vitalPeriodic.csv.gz
```

Short selectors work too:

```bash
python ingest.py --files patient lab vitalPeriodic
```

Enable file-level multiprocessing:

```bash
python ingest.py --important-only --parallel
```

## Analytics Layer

Build a curated one-row-per-patient table from the raw PostgreSQL landing tables:

```bash
python build_patient_summary.py
```

Use a custom analytics schema if needed:

```bash
python build_patient_summary.py --analytics-schema analytics
```

This creates `analytics.patient_summary` by joining:

- `raw_eicu.patient`
- `raw_eicu.lab`
- `raw_eicu.vital_periodic`
- `raw_eicu.admission_dx`

The curated table keeps raw ingestion untouched and gives you a clean base for dashboards, feature engineering, and ML training.

## Preprocessing Layer

Build a production-style ML-ready dataset directly in PostgreSQL:

```bash
python build_ml_dataset.py
```

Use a custom target schema if needed:

```bash
python build_ml_dataset.py --target-schema ml_prep
```

This preprocessing pipeline:

- cleans invalid numeric and text values
- removes duplicate patient records safely
- imputes missing values for core patient-level fields
- aggregates `lab` and `vital_periodic` into one-row-per-patient features
- adds categorical encodings and engineered features
- writes the final ML-ready dataset to `ml_prep.ml_dataset`
- saves a JSON summary to `preprocessing_report.json`

By default the preprocessing pipeline reads labs from `raw_eicu.lab_subset`. Override with `LAB_SOURCE_TABLE=lab` in `.env` if you want to use the full raw lab table later.

## Baseline Training

Train baseline mortality models from the PostgreSQL preprocessing output:

```bash
python train_postgres_model.py
```

This trains:

- a full class-balanced Random Forest baseline
- a full class-balanced Logistic Regression baseline
- an early-prediction Random Forest without LOS-derived leakage features
- an early-prediction Logistic Regression without LOS-derived leakage features

Artifacts are written to `models/` and `data/artifacts/`, including metrics and top feature importances.

## Threshold Tuning

Evaluate operating thresholds for the early Random Forest mortality model:

```bash
python tune_early_thresholds.py
```

This writes a threshold comparison report to `data/artifacts/postgres_early_thresholds.json` with suggested balanced, high-recall, and higher-precision operating points.

## Early Prediction Serving

The tuned early mortality model uses `EARLY_MORTALITY_THRESHOLD` from `.env`.

Recommended default:

```env
EARLY_MORTALITY_THRESHOLD=0.60
```

## Module Responsibilities

- `ingestion/stream_reader.py`: file discovery and chunked gzip streaming with csv fallback
- `ingestion/transformer.py`: chunk-level cleaning, type normalization, null handling, and table mapping
- `ingestion/db_writer.py`: PostgreSQL schema creation, table alignment, indexing, and streaming `COPY` batch inserts
- `ingestion/pipeline.py`: orchestration, logging, selective loading, and optional multiprocessing

## Notes

- The production ingestion path does not load entire source files into memory.
- The PostgreSQL loader is append-oriented and suitable for raw landing-zone ingestion.
- The transformer currently applies light standardization so future downstream marts, feature stores, or cloud jobs can build on a stable raw schema.
