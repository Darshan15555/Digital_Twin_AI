from __future__ import annotations

import pandas as pd


def carry_forward_with_limit(
    df: pd.DataFrame,
    group_col: str,
    sort_col: str,
    value_columns: list[str],
    limit: int,
) -> pd.DataFrame:
    if df.empty or not value_columns:
        return df.copy()

    ordered = df.sort_values([group_col, sort_col]).copy()
    valid_columns = [column for column in value_columns if column in ordered.columns]
    if not valid_columns:
        return ordered

    ordered[valid_columns] = ordered.groupby(group_col, observed=True, sort=False)[valid_columns].ffill(limit=limit)
    return ordered