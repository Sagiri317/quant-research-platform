"""Partition helpers."""

from __future__ import annotations

import pandas as pd


def add_year_partition(
    frame: pd.DataFrame,
    timestamp_column: str = "timestamp",
) -> pd.DataFrame:
    result = frame.copy()
    result["year"] = pd.to_datetime(result[timestamp_column]).dt.year
    return result


__all__ = ["add_year_partition"]
