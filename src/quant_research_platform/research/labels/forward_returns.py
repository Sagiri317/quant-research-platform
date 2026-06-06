"""Generic forward-return label helpers."""

from __future__ import annotations

import numpy as np
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
    price = pd.to_numeric(result[execution_price_column], errors="coerce")
    grouped_price = price.groupby(result[asset_column], sort=False)

    entry_price = grouped_price.shift(-1)
    entry_price = entry_price.where(np.isfinite(entry_price) & entry_price.ne(0.0))
    for horizon in horizons:
        exit_price = grouped_price.shift(-(horizon + 1))
        exit_price = exit_price.where(np.isfinite(exit_price))
        label = exit_price / entry_price - 1.0
        result[f"forward_return_{horizon}d"] = label.replace(
            [np.inf, -np.inf],
            np.nan,
        )

    return result


__all__ = ["add_forward_return_labels"]
