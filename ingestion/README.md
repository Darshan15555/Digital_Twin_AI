# Ingestion Pipeline

This package provides a production-oriented raw-data ingestion path for the eICU Collaborative Research Database.

Modules:

- `file_scanner.py`: detects all `.csv.gz` files and resolves selective-load subsets
- `stream_reader.py`: generator-based gzip streaming with pandas chunking and csv fallback
- `transformer.py`: applies chunk-level cleaning, filtering, normalization, and dtype optimization
- `db_writer.py`: uses PostgreSQL `COPY` for bulk inserts and auto-creates schema/tables
- `pipeline.py`: orchestrates stream -> transform -> insert -> cleanup and optional multiprocessing

Design notes:

- no full-file loads into memory
- no DataFrame concatenation across chunks
- no manual decompression to disk
- explicit chunk cleanup with `gc.collect()`
- chunk-level and file-level fault isolation
- schema creation is automatic
- file selection can be all files or a curated subset
