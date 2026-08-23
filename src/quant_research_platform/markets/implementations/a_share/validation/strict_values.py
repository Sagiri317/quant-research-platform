"""Owner-local strict value semantics for A-share data."""

from __future__ import annotations

import pandas as pd

from quant_research_platform.contracts.validation import (
    strict_bool_series,
    strict_finite_numeric_series,
    strict_integer_series,
    strict_timestamp_series,
)


def strict_trading_bool_series(values: pd.Series, *, error: str) -> pd.Series:
    return strict_bool_series(values, error=error)


def strict_market_type_series(values: pd.Series, *, error: str) -> pd.Series:
    return strict_integer_series(values, error=error)


def strict_timestamp_value_series(values: pd.Series, *, error: str) -> pd.Series:
    return strict_timestamp_series(values, error=error)


def strict_numeric_value_series(
    values: pd.Series,
    *,
    error: str,
    allow_null: bool = False,
) -> pd.Series:
    return strict_finite_numeric_series(values, error=error, allow_null=allow_null)
