"""Generic forward-return label helpers."""

from __future__ import annotations

import pandas as pd


def add_forward_return_labels(
    frame: pd.DataFrame,
    *,
    horizons: tuple[int, ...] = (1, 5, 10, 20, 60, 120),
    asset_column: str = "asset_id",
    date_column: str = "timestamp",
    execution_price_column: str = "adj_open",
) -> pd.DataFrame:
    """Add next-open to horizon-open forward return labels."""

    result = frame.sort_values([asset_column, date_column]).copy()
    grouped = result.groupby(asset_column, sort=False)

    entry_price = grouped[execution_price_column].shift(-1)
    for horizon in horizons:
        exit_price = grouped[execution_price_column].shift(-(horizon + 1))
        result[f"forward_return_{horizon}d"] = exit_price / entry_price - 1.0

    return result


__all__ = ["add_forward_return_labels"]
