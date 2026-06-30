"""Feature selection helpers for price-volume factor outputs."""

from __future__ import annotations

import warnings
from collections.abc import Iterable

import pandas as pd

from .columns import (
    PRICE_VOLUME_FACTOR_PREFIXES,
    PRICE_VOLUME_FEATURE_EXCLUDE_COLUMNS,
    PRICE_VOLUME_FEATURE_EXCLUDE_PREFIXES,
    price_volume_factor_columns,
)


def select_price_volume_features(
    frame: pd.DataFrame,
    *,
    expected_columns: tuple[str, ...] | None = None,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Select preset 0.1 feature columns and exclude labels/filters/leakage."""

    return select_price_volume_feature_columns_from_schema(
        tuple(frame.columns),
        expected_columns=expected_columns,
        include_actual_factor_columns=expected_columns is None,
    )


def select_price_volume_feature_columns_from_schema(
    columns: Iterable[str],
    *,
    expected_columns: tuple[str, ...] | None = None,
    include_actual_factor_columns: bool = False,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Resolve preset 0.1 feature columns from schema names only."""

    schema_columns = tuple(columns)
    expected = expected_columns or price_volume_factor_columns()
    actual_factor_columns = (
        ()
        if not include_actual_factor_columns
        else tuple(
            column
            for column in schema_columns
            if column.casefold().startswith(PRICE_VOLUME_FACTOR_PREFIXES)
        )
    )
    candidates = tuple(
        dict.fromkeys(
            (
                *actual_factor_columns,
                *(column for column in expected if column in schema_columns),
            )
        )
    )
    warnings_out: list[str] = [
        f"missing expected feature column: {column}"
        for column in expected
        if column not in schema_columns
    ]
    selected = tuple(
        column for column in candidates if not _is_excluded_feature(column)
    )
    if not selected:
        raise ValueError("invalid_feature_preset: no usable feature columns remain")
    dropped = [column for column in candidates if column not in selected]
    warnings_out.extend(
        f"excluded leakage/control column: {column}" for column in dropped
    )
    return selected, tuple(warnings_out)
def warn_for_missing_price_volume_features(
    frame: pd.DataFrame,
    *,
    expected_columns: tuple[str, ...] | None = None,
) -> tuple[str, ...]:
    """Emit Python warnings for missing preset columns and return messages."""

    _, messages = select_price_volume_features(
        frame,
        expected_columns=expected_columns,
    )
    for message in messages:
        warnings.warn(message, RuntimeWarning, stacklevel=2)
    return messages
def _is_excluded_feature(column: str) -> bool:
    normalized = column.casefold()
    if column in PRICE_VOLUME_FEATURE_EXCLUDE_COLUMNS:
        return True
    return any(
        normalized.startswith(prefix)
        for prefix in PRICE_VOLUME_FEATURE_EXCLUDE_PREFIXES
    )

__all__ = [
    "select_price_volume_feature_columns_from_schema",
    "select_price_volume_features",
    "warn_for_missing_price_volume_features",
]
