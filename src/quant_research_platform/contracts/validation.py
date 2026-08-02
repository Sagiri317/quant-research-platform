"""Shared contract validation helpers."""

from __future__ import annotations

import warnings
from collections.abc import Iterable
from typing import Any

import numpy as np
import pandas as pd


def _validate_column_tuple(
    columns: tuple[str, ...],
    field_name: str,
    *,
    required: bool = True,
) -> None:
    if required and not columns:
        raise ValueError(f"{field_name}_required")
    if any(not isinstance(column, str) or not column for column in columns):
        raise ValueError(f"{field_name}_must_be_non_empty_strings")
    if len(set(columns)) != len(columns):
        raise ValueError(f"{field_name}_must_not_contain_duplicates")


def strict_string_series(
    values: pd.Series,
    *,
    error: str,
    strip: bool = True,
    allow_empty: bool = False,
    allow_null: bool = False,
) -> pd.Series:
    result = pd.Series(values, copy=True)
    output: list[object] = []
    for value in result.tolist():
        if _is_null_scalar(value):
            if allow_null:
                output.append(pd.NA)
                continue
            raise ValueError(error)
        if not isinstance(value, str):
            raise ValueError(error)
        text = value.strip() if strip else value
        if not allow_empty and text == "":
            raise ValueError(error)
        output.append(text)
    return pd.Series(output, index=result.index, dtype="object")


def strict_membership_series(
    values: pd.Series,
    *,
    allowed: Iterable[Any],
    error: str,
    allow_null: bool = False,
) -> pd.Series:
    allowed_set = set(allowed)
    result = strict_hashable_key_series(values, error=error, allow_null=allow_null)
    non_null = result.dropna()
    if not bool(non_null.isin(allowed_set).all()):
        raise ValueError(error)
    return result


def strict_timestamp_series(
    values: pd.Series,
    *,
    error: str,
    allow_null: bool = False,
) -> pd.Series:
    series = pd.Series(values, copy=True)
    null_mask = _null_mask(series)
    if bool(null_mask.any()) and not allow_null:
        raise ValueError(error)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            parsed = pd.to_datetime(series, errors="coerce", format="mixed")
    except (TypeError, ValueError, OverflowError, Warning):
        return _strict_timestamp_series_fallback(
            series,
            null_mask=null_mask,
            error=error,
        )
    invalid = parsed.isna() & ~(null_mask & allow_null)
    if bool(invalid.any()):
        return _strict_timestamp_series_fallback(
            series,
            null_mask=null_mask,
            error=error,
        )
    if isinstance(parsed.dtype, pd.DatetimeTZDtype):
        local_naive = parsed.dt.tz_localize(None)
        utc_naive = parsed.dt.tz_convert("UTC").dt.tz_localize(None)
        offsets = local_naive - utc_naive
        if offsets.dropna().nunique() > 1:
            raise ValueError(error)
        parsed = utc_naive
    try:
        return pd.Series(parsed, index=series.index, dtype="datetime64[ns]")
    except (TypeError, ValueError, OverflowError):
        return _strict_timestamp_series_fallback(
            series,
            null_mask=null_mask,
            error=error,
        )


def _strict_timestamp_series_fallback(
    series: pd.Series,
    *,
    null_mask: pd.Series,
    error: str,
) -> pd.Series:
    parsed_values: list[pd.Timestamp | pd.NaTType] = []
    aware: bool | None = None
    offset: object = None
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            for value, is_null in zip(
                series.tolist(),
                null_mask.tolist(),
                strict=True,
            ):
                if is_null:
                    parsed_values.append(pd.NaT)
                    continue
                timestamp = pd.Timestamp(
                    pd.to_datetime(value, errors="coerce", format="mixed")
                )
                if pd.isna(timestamp):
                    raise ValueError(error)
                timestamp_offset = timestamp.utcoffset()
                timestamp_aware = timestamp_offset is not None
                if aware is None:
                    aware = timestamp_aware
                    offset = timestamp_offset
                elif timestamp_aware != aware or (
                    timestamp_aware and timestamp_offset != offset
                ):
                    raise ValueError(error)
                if timestamp_aware:
                    timestamp = timestamp.tz_convert("UTC").tz_localize(None)
                parsed_values.append(timestamp)
    except (TypeError, ValueError, OverflowError, Warning) as exc:
        raise ValueError(error) from exc
    try:
        return pd.Series(
            parsed_values,
            index=series.index,
            dtype="datetime64[ns]",
        )
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(error) from exc


def strict_bool_series(
    values: pd.Series,
    *,
    error: str,
    allow_null: bool = False,
) -> pd.Series:
    series = pd.Series(values, copy=True)
    output: list[object] = []
    mapping = {
        "true": True,
        "false": False,
        "1": True,
        "0": False,
    }
    for value in series.tolist():
        if _is_null_scalar(value):
            if allow_null:
                output.append(pd.NA)
                continue
            raise ValueError(error)
        if isinstance(value, bool):
            output.append(value)
            continue
        if isinstance(value, str):
            text = value.strip().lower()
            if text in mapping:
                output.append(mapping[text])
                continue
        raise ValueError(error)
    return pd.Series(output, index=series.index, dtype="object").astype(
        "boolean" if allow_null else "bool"
    )


def strict_integer_series(
    values: pd.Series,
    *,
    error: str,
    allow_null: bool = False,
) -> pd.Series:
    numeric = _strict_numeric_series(values, error=error, allow_null=allow_null)
    non_null = numeric.dropna()
    if not bool(np.equal(non_null, np.floor(non_null)).all()):
        raise ValueError(error)
    if allow_null:
        return numeric.astype("Int64")
    return numeric.astype("int64")


def strict_finite_numeric_series(
    values: pd.Series,
    *,
    error: str,
    allow_null: bool = False,
) -> pd.Series:
    return _strict_numeric_series(values, error=error, allow_null=allow_null)


def strict_hashable_key_series(
    values: pd.Series,
    *,
    error: str,
    allow_null: bool = False,
) -> pd.Series:
    series = pd.Series(values, copy=True)
    output: list[object] = []
    for value in series.tolist():
        if _is_null_scalar(value):
            if allow_null:
                output.append(pd.NA)
                continue
            raise ValueError(error)
        if not _is_hashable(value):
            raise ValueError(error)
        output.append(value)
    return pd.Series(output, index=series.index, dtype="object")


def _strict_numeric_series(
    values: pd.Series,
    *,
    error: str,
    allow_null: bool,
) -> pd.Series:
    series = pd.Series(values, copy=True)
    for value in series.tolist():
        if _is_null_scalar(value):
            if allow_null:
                continue
            raise ValueError(error)
        if isinstance(value, bool) or not _is_hashable(value):
            raise ValueError(error)
    numeric = pd.to_numeric(series, errors="coerce")
    null_mask = _null_mask(series)
    invalid = numeric.isna() & ~(null_mask & allow_null)
    if bool(invalid.any()):
        raise ValueError(error)
    non_null = numeric.dropna()
    if not bool(np.isfinite(non_null).all()):
        raise ValueError(error)
    return numeric.astype("Float64" if allow_null else "float64")


def _is_hashable(value: object) -> bool:
    try:
        hash(value)
    except TypeError:
        return False
    return True


def _is_null_scalar(value: object) -> bool:
    if not _is_hashable(value) and not isinstance(value, type(pd.NA)):
        return False
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _null_mask(series: pd.Series) -> pd.Series:
    return pd.Series(
        [_is_null_scalar(value) for value in series.tolist()],
        index=series.index,
        dtype="bool",
    )


__all__ = [
    "strict_bool_series",
    "strict_finite_numeric_series",
    "strict_hashable_key_series",
    "strict_integer_series",
    "strict_membership_series",
    "strict_string_series",
    "strict_timestamp_series",
]
