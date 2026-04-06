from __future__ import annotations

import argparse
import gc
import logging
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from ingestion.config import DEFAULT_IMPORTANT_FILES, PipelineConfig
from ingestion.db_writer import PostgresCopyWriter
from ingestion.file_scanner import FileScanner
from ingestion.stream_reader import StreamingReader
from ingestion.transformer import DataTransformer, spec_for


@dataclass(slots=True)
class FileIngestionResult:
    file_name: str
    table_name: str
    rows_inserted: int
    chunks_processed: int
    failed_chunks: int


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("ingestion.log"),
        ],
    )


log = logging.getLogger(__name__)


class IngestionPipeline:
    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self.scanner = FileScanner(config)
        self.reader = StreamingReader(config)
        self.transformer = DataTransformer()
        self.db_writer = PostgresCopyWriter(config)

    def run(self, selected_files: list[str] | None = None) -> list[FileIngestionResult]:
        self.config.validate()
        self.db_writer.ensure_schema()
        log.info(
            "Pipeline starting | data_path=%s schema=%s database_url=%s",
            self.config.data_path,
            self.config.schema_name,
            self.db_writer._safe_database_url(),
        )
        targets = self.scanner.select(selected_files=selected_files)
        results: list[FileIngestionResult] = []

        for file_path in targets:
            try:
                results.append(self._ingest_file(file_path))
            except Exception as exc:
                log.exception("Skipping failed file %s: %s", file_path.name, exc)
        return results

    def _ingest_file(self, file_path: Path) -> FileIngestionResult:
        log.info("Starting file ingestion: %s", file_path.name)
        available_columns = self.reader.read_header(file_path)
        spec = spec_for(file_path.name, available_columns)
        rows_inserted = 0
        chunks_processed = 0
        failed_chunks = 0
        verification_logged = False
        prefer_csv_fallback = file_path.name.lower() in self.config.csv_fallback_tables

        for chunk_number, chunk in self.reader.iter_chunks(
            file_path,
            usecols=spec.usecols,
            dtype=spec.dtype_map,
            chunk_size=self.config.chunk_size,
            prefer_csv_fallback=prefer_csv_fallback,
        ):
            started_at = time.perf_counter()
            transformed = None
            try:
                transformed = self.transformer.transform_chunk(file_path.name, chunk, spec)
                if transformed.empty:
                    chunks_processed += 1
                    gc.collect()
                    continue

                self.db_writer.ensure_table(spec.table_name, transformed)

                inserted = self.db_writer.copy_rows(spec.table_name, transformed)
                rows_inserted += inserted
                chunks_processed += 1
                if not verification_logged and inserted > 0:
                    total_rows = self.db_writer.count_rows(spec.table_name)
                    verification_logged = True
                    log.info(
                        "Verification success | schema=%s table=%s row_count=%s",
                        self.config.schema_name,
                        spec.table_name,
                        f"{total_rows:,}",
                    )
                elapsed = time.perf_counter() - started_at
                log.info(
                    "file=%s chunk=%s rows=%s total_rows=%s elapsed=%.2fs",
                    file_path.name,
                    chunk_number,
                    f"{inserted:,}",
                    f"{rows_inserted:,}",
                    elapsed,
                )
            except Exception as exc:
                failed_chunks += 1
                log.exception("Skipping failed chunk %s from %s: %s", chunk_number, file_path.name, exc)
            finally:
                del chunk
                if transformed is not None:
                    del transformed
                gc.collect()

        log.info(
            "Finished %s -> %s | rows=%s chunks=%s failed_chunks=%s",
            file_path.name,
            spec.table_name,
            f"{rows_inserted:,}",
            chunks_processed,
            failed_chunks,
        )
        return FileIngestionResult(
            file_name=file_path.name,
            table_name=spec.table_name,
            rows_inserted=rows_inserted,
            chunks_processed=chunks_processed,
            failed_chunks=failed_chunks,
        )


def _discover_for_parallel(config: PipelineConfig, selected_files: list[str] | None) -> list[Path]:
    scanner = FileScanner(config)
    return scanner.select(selected_files=selected_files)


def _run_single_file(config: PipelineConfig, file_name: str) -> FileIngestionResult:
    pipeline = IngestionPipeline(config)
    resolved_name = file_name if file_name.lower().endswith(".csv.gz") else f"{file_name}.csv.gz"
    return pipeline._ingest_file(config.data_path / resolved_name)


def run_parallel(config: PipelineConfig, selected_files: list[str] | None = None) -> list[FileIngestionResult]:
    targets = _discover_for_parallel(config, selected_files)
    if len(targets) <= 1 or config.max_workers == 1:
        pipeline = IngestionPipeline(config)
        return pipeline.run(selected_files=[path.name for path in targets])

    bootstrap = PostgresCopyWriter(config)
    bootstrap.ensure_schema()

    results: list[FileIngestionResult] = []
    with ProcessPoolExecutor(max_workers=config.max_workers) as executor:
        futures = {
            executor.submit(_run_single_file, config, path.name): path.name
            for path in targets
        }
        for future in as_completed(futures):
            file_name = futures[future]
            try:
                results.append(future.result())
            except Exception as exc:
                log.exception("Parallel worker failed for %s: %s", file_name, exc)
    return sorted(results, key=lambda item: item.file_name)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stream eICU .csv.gz files into PostgreSQL.")
    parser.add_argument(
        "--files",
        nargs="*",
        help="Specific files to ingest. Accepts either lab or lab.csv.gz style selectors.",
    )
    parser.add_argument(
        "--important-only",
        action="store_true",
        help="Load only the important file set.",
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Process files in parallel with one process per file up to MAX_WORKERS.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        help="Override the configured stream chunk size for this run.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = PipelineConfig()
    if args.chunk_size:
        config.chunk_size = args.chunk_size
    configure_logging(config.log_level)

    selected_files = args.files
    if args.important_only:
        selected_files = list(DEFAULT_IMPORTANT_FILES if not config.important_files else config.important_files)

    if args.parallel:
        results = run_parallel(config, selected_files=selected_files)
    else:
        pipeline = IngestionPipeline(config)
        results = pipeline.run(selected_files=selected_files)

    for result in results:
        log.info(
            "Summary | file=%s table=%s rows=%s chunks=%s failed_chunks=%s",
            result.file_name,
            result.table_name,
            f"{result.rows_inserted:,}",
            result.chunks_processed,
            result.failed_chunks,
        )


if __name__ == "__main__":
    main()
