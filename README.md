# ICU Analytics System

Ingestion-only ICU data pipeline built on the eICU Collaborative Research Database v2.0.

## Scope

This `feature1` branch keeps the raw data ingestion system and removes training and preprocessing components.

Kept:

- `ingestion/`
- `ingest.py`
- `config/`
- `utils/`
- `.env`
- `setup.py`
- `requirements.txt`
- `launch.bat`

Optional directories left as-is:

- `frontend/`
- `backend/`
- `data/`
- `demo_data/`

## Setup

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Confirm the dataset path and database URL in [.env](/d:/sem_6/MiniProject/Project/icu_analytics_system/.env).

4. Validate ingestion configuration:

```bash
python setup.py
```

## Ingestion

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

## Module Responsibilities

- `ingestion/stream_reader.py`: file discovery and chunked gzip streaming with csv fallback
- `ingestion/transformer.py`: chunk-level cleaning, type normalization, null handling, and table mapping
- `ingestion/db_writer.py`: PostgreSQL schema creation, table alignment, indexing, and streaming `COPY` batch inserts
- `ingestion/pipeline.py`: orchestration, logging, selective loading, and optional multiprocessing

## Notes

- The ingestion path supports raw `.csv.gz` files and csv fallback where configured.
- The production ingestion path does not load entire source files into memory.
- The PostgreSQL loader is append-oriented and suitable for raw landing-zone ingestion.
