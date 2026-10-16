"""Shared contract validation helpers."""

from __future__ import annotations

import math
import warnings
from collections.abc import Iterable
from datetime import date, datetime
from decimal import Decimal, InvalidOperation
from fractions import Fraction
from typing import Any

import numpy as np
import pandas as pd


def validate_financial_disclosure_chronology(
    report_period: pd.Series, first_disclosure_date: pd.Series
) -> None:
    if first_disclosure_date.lt(report_period).any():
        raise ValueError("financial_disclosure_before_report_period")


def _validate_column_tuple(
    columns: tuple[str, ...],
    field_name: str,
    *,
    required: bool = True,
) -> None:
    if type(columns) is not tuple:
        raise ValueError(f"{field_name}_must_be_tuple")
    if required and not columns:
        raise ValueError(f"{field_name}_required")
    if any(type(column) is not str or not column for column in columns):
        raise ValueError(f"{field_name}_must_be_non_empty_strings")
    if len(set(columns)) != len(columns):
        raise ValueError(f"{field_name}_must_not_contain_duplicates")


def _validate_non_empty_string(value: object, error: str) -> str:
    if type(value) is not str or not value:
        raise ValueError(error)
    return value


def strict_finite_float(value: object, *, error: str) -> float:
    if isinstance(value, (bool, np.bool_, complex, np.complexfloating)):
        raise ValueError(error)
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(error) from exc
    if not math.isfinite(number):
        raise ValueError(error)
    return number


def strict_string_series(
    values: pd.Series,
    *,
    error: str,
    strip: bool = True,
    allow_empty: bool = False,
    allow_null: bool = False,
) -> pd.Series:
    result = pd.Series(values, copy=True)
    null_mask = _null_mask(result)
    if bool(null_mask.any()) and not allow_null:
        raise ValueError(error)
    non_null = result.loc[~null_mask]
    is_fast_string = isinstance(result.dtype, pd.StringDtype)
    if not is_fast_string:
        inferred = pd.api.types.infer_dtype(non_null, skipna=False)
        is_fast_string = inferred in {"empty", "string", "unicode"} and bool(
            non_null.map(type).eq(str).all()
        )
    if is_fast_string:
        output = result.astype("string")
        if strip:
            output = output.str.strip()
        if not allow_empty and bool((output.eq("") & ~null_mask).any()):
            raise ValueError(error)
        return output.astype("object")
    output: list[object] = []
    for value, is_null in zip(result.tolist(), null_mask.tolist(), strict=True):
        if is_null:
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
    if not pd.api.types.is_datetime64_any_dtype(series.dtype) and any(
        not is_null and not _is_supported_timestamp_scalar(value)
        for value, is_null in zip(
            series.tolist(),
            null_mask.tolist(),
            strict=True,
        )
    ):
        raise ValueError(error)
    if pd.api.types.is_datetime64_dtype(series.dtype):
        try:
            return pd.Series(
                series,
                index=series.index,
                dtype="datetime64[ns]",
            )
        except (TypeError, ValueError, OverflowError):
            return _strict_timestamp_series_fallback(
                series,
                null_mask=null_mask,
                error=error,
            )
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            parsed = pd.to_datetime(series, errors="raise", format="mixed")
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
        parsed = parsed.dt.tz_convert("UTC").dt.tz_localize(None)
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
                    pd.to_datetime(value, errors="raise", format="mixed")
                )
                timestamp_offset = timestamp.utcoffset()
                timestamp_aware = timestamp_offset is not None
                if aware is None:
                    aware = timestamp_aware
                elif timestamp_aware != aware:
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
    null_mask = _null_mask(series)
    if bool(null_mask.any()) and not allow_null:
        raise ValueError(error)
    if series.dtype == np.dtype("bool") or isinstance(series.dtype, pd.BooleanDtype):
        return series.astype("boolean" if allow_null else "bool").rename(None)
    output: list[object] = []
    mapping = {
        "true": True,
        "false": False,
        "1": True,
        "0": False,
    }
    for value, is_null in zip(series.tolist(), null_mask.tolist(), strict=True):
        if is_null:
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
    series = pd.Series(values, copy=True)
    null_mask = _null_mask(series)
    if bool(null_mask.any()) and not allow_null:
        raise ValueError(error)
    dtype = "Int64" if allow_null else "int64"
    if pd.api.types.is_integer_dtype(series.dtype):
        non_null = series.loc[~null_mask]
        if bool(non_null.lt(-(2**63)).any()) or bool(non_null.gt(2**63 - 1).any()):
            raise ValueError(error)
        return series.astype(dtype)
    if pd.api.types.is_float_dtype(series.dtype):
        non_null = series.loc[~null_mask]
        if (
            not np.isfinite(non_null).all()
            or non_null.lt(-(2**63)).any()
            or non_null.ge(2**63).any()
            or not np.equal(non_null, np.floor(non_null)).all()
        ):
            raise ValueError(error)
        return series.astype(dtype)
    output = []
    for value, is_null in zip(series, null_mask, strict=True):
        if is_null:
            output.append(pd.NA)
            continue
        if isinstance(value, (bool, np.bool_)) or not isinstance(
            value, (int, np.integer, float, np.floating, Decimal, str)
        ):
            raise ValueError(error)
        try:
            numeric = Decimal(value.strip()) if isinstance(value, str) else value
            if not -(2**63) <= numeric <= 2**63 - 1:
                raise ValueError(error)
            integer = int(numeric)
        except (ValueError, TypeError, OverflowError, InvalidOperation) as exc:
            raise ValueError(error) from exc
        if numeric != integer or not -(2**63) <= integer <= 2**63 - 1:
            raise ValueError(error)
        output.append(integer)
    return pd.Series(output, index=series.index, name=series.name, dtype=dtype)


def strict_real_numeric_series(values: pd.Series, *, error: str) -> pd.Series:
    """Parse real observations, retaining missing and infinite values for the caller."""
    series = pd.Series(values, copy=True)
    if (
        pd.api.types.is_bool_dtype(series.dtype)
        or pd.api.types.is_complex_dtype(series.dtype)
        or series.dtype.kind in "mM"
    ):
        raise ValueError(error)
    null_mask = _null_mask(series)
    if not pd.api.types.is_numeric_dtype(series.dtype):
        for value, is_null in zip(series, null_mask, strict=True):
            if isinstance(value, (bool, np.bool_, complex, np.complexfloating)) or (
                not is_null and not _is_hashable(value)
            ):
                raise ValueError(error)
    try:
        numeric = pd.to_numeric(series, errors="coerce")
    except OverflowError as exc:
        raise ValueError(error) from exc
    if (numeric.isna() & ~null_mask).any():
        raise ValueError(error)
    return numeric.astype("float64")


def strict_finite_numeric_series(
    values: pd.Series,
    *,
    error: str,
    allow_null: bool = False,
) -> pd.Series:
    numeric = strict_real_numeric_series(values, error=error)
    if (numeric.isna().any() and not allow_null) or not np.isfinite(
        numeric.dropna()
    ).all():
        raise ValueError(error)
    return numeric.astype("Float64" if allow_null else "float64")


def strict_hashable_key_series(
    values: pd.Series,
    *,
    error: str,
    allow_null: bool = False,
) -> pd.Series:
    series = pd.Series(values, copy=True)
    null_mask = _null_mask(series)
    if bool(null_mask.any()) and not allow_null:
        raise ValueError(error)
    output: list[object] = []
    for value, is_null in zip(series.tolist(), null_mask.tolist(), strict=True):
        if is_null:
            if allow_null:
                output.append(pd.NA)
                continue
            raise ValueError(error)
        if not _is_hashable(value):
            raise ValueError(error)
        output.append(value)
    return pd.Series(output, index=series.index, dtype="object")


def _is_hashable(value: object) -> bool:
    try:
        hash(value)
    except TypeError:
        return False
    return True


def _is_supported_timestamp_scalar(value: object) -> bool:
    return isinstance(value, (str, date, datetime, np.datetime64)) and not isinstance(
        value,
        bool,
    )


def _is_null_scalar(value: object) -> bool:
    if not _is_hashable(value) and not isinstance(value, type(pd.NA)):
        return False
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _null_mask(series: pd.Series) -> pd.Series:
    try:
        return series.isna()
    except (TypeError, ValueError):
        return pd.Series(
            [_is_null_scalar(value) for value in series.tolist()],
            index=series.index,
            dtype="bool",
        )


def encode_exact_numeric(value):
    """Persist exact rationals and binary floats without losing their type."""
    if type(value) is int:
        value = Fraction(value)
    if isinstance(value, Fraction):
        return f"q:{hex(value.numerator)}/{hex(value.denominator)}"
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError("exact_numeric_encoding_invalid")
    try:
        numeric = float(value)
    except OverflowError:
        raise ValueError("exact_numeric_encoding_invalid") from None
    if not math.isfinite(numeric):
        raise ValueError("exact_numeric_encoding_invalid")
    return f"f:{numeric.hex()}"


def decode_exact_numeric(value):
    """Reject alternate spellings so persisted exact values have one identity."""
    try:
        if value.startswith("q:"):
            numerator, denominator = value[2:].split("/")
            result = Fraction(int(numerator, 16), int(denominator, 16))
        elif value.startswith("f:"):
            result = float.fromhex(value[2:])
        else:
            raise ValueError
        if encode_exact_numeric(result) != value:
            raise ValueError
    except (AttributeError, TypeError, ValueError, ZeroDivisionError, OverflowError):
        raise ValueError("exact_numeric_encoding_invalid") from None
    return result


def finite_exact_float(value):
    """Return a representable float, or None for overflow/nonzero underflow."""
    try:
        result = float(value)
    except OverflowError:
        return None
    return result if math.isfinite(result) and (result != 0 or value == 0) else None
