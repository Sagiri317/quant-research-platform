"""Owner-local strict value semantics for A-share data."""

from __future__ import annotations

import pandas as pd

from quant_research_platform.contracts.validation import (
    strict_bool_series,
    strict_finite_numeric_series,
    strict_integer_series,
    strict_string_series,
    strict_timestamp_series,
)


def strict_stock_code_series(values: pd.Series, *, error: str) -> pd.Series:
    source = pd.Series(values)
    result: list[str] = []
    for value in source.tolist():
        result.append(_strict_stock_code_value(value, error=error))
    return pd.Series(result, index=source.index, dtype="object")


def strict_source_asset_id_series(
    values: pd.Series,
    *,
    asset_namespace: str,
    error: str,
) -> pd.Series:
    if asset_namespace == "stock":
        return strict_stock_code_series(values, error=error)
    return strict_string_series(values, error=error)


def strict_index_code_series(values: pd.Series, *, error: str) -> pd.Series:
    return strict_string_series(values, error=error)


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


def _strict_stock_code_value(value: object, *, error: str) -> str:
    if isinstance(value, bool):
        raise ValueError(error)
    if isinstance(value, int):
        return _format_stock_code(value, error=error)
    if isinstance(value, float):
        if pd.isna(value) or not value.is_integer():
            raise ValueError(error)
        return _format_stock_code(int(value), error=error)
    if not isinstance(value, str):
        raise ValueError(error)
    text = value.strip()
    if text.endswith(".0"):
        text = text[:-2]
    if not text.isdecimal():
        raise ValueError(error)
    return _format_stock_code(int(text), error=error)


def _format_stock_code(value: int, *, error: str) -> str:
    if value < 0:
        raise ValueError(error)
    text = str(value)
    if len(text) > 6:
        raise ValueError(error)
    return text.zfill(6)


__all__ = [
    "strict_index_code_series",
    "strict_market_type_series",
    "strict_numeric_value_series",
    "strict_source_asset_id_series",
    "strict_stock_code_series",
    "strict_timestamp_value_series",
    "strict_trading_bool_series",
]
