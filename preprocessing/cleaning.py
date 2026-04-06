from __future__ import annotations


def safe_numeric(column_ref: str) -> str:
    return (
        "CASE "
        f"WHEN {column_ref} IS NULL THEN NULL "
        f"WHEN trim({column_ref}::text) IN ('', 'nan', 'None', '<NA>') THEN NULL "
        f"WHEN trim({column_ref}::text) ~ '^-?[0-9]+(\\.[0-9]+)?$' THEN trim({column_ref}::text)::double precision "
        "ELSE NULL "
        "END"
    )


def safe_bigint(column_ref: str) -> str:
    return (
        "CASE "
        f"WHEN {column_ref} IS NULL THEN NULL "
        f"WHEN trim({column_ref}::text) IN ('', 'nan', 'None', '<NA>') THEN NULL "
        f"WHEN trim({column_ref}::text) ~ '^-?[0-9]+(\\.[0-9]+)?$' THEN round(trim({column_ref}::text)::numeric)::bigint "
        "ELSE NULL "
        "END"
    )


def clean_text(column_ref: str, fallback: str = "Unknown") -> str:
    escaped = fallback.replace("'", "''")
    return (
        "CASE "
        f"WHEN {column_ref} IS NULL THEN '{escaped}' "
        f"WHEN btrim({column_ref}::text) = '' THEN '{escaped}' "
        f"WHEN lower(btrim({column_ref}::text)) IN ('nan', 'none', '<na>') THEN '{escaped}' "
        f"ELSE btrim({column_ref}::text) "
        "END"
    )


def iqr_clip(column_ref: str, q1_ref: str, q3_ref: str) -> str:
    return (
        "CASE "
        f"WHEN {column_ref} IS NULL THEN NULL "
        f"WHEN {column_ref} < ({q1_ref} - 1.5 * ({q3_ref} - {q1_ref})) THEN ({q1_ref} - 1.5 * ({q3_ref} - {q1_ref})) "
        f"WHEN {column_ref} > ({q3_ref} + 1.5 * ({q3_ref} - {q1_ref})) THEN ({q3_ref} + 1.5 * ({q3_ref} - {q1_ref})) "
        f"ELSE {column_ref} "
        "END"
    )
