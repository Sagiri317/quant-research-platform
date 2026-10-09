"""Common forward-return labels on an explicit ordered time axis."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from quant_research_platform.contracts.validation import (
    strict_bool_series,
    strict_finite_numeric_series,
    strict_hashable_key_series,
    strict_timestamp_series,
)


def add_forward_return_labels(
    frame: pd.DataFrame,
    *,
    horizons: tuple[int, ...],
    asset_column: str = "asset_id",
    date_column: str = "timestamp",
    execution_price_column: str = "adj_open",
    sessions: Sequence[object],
    can_enter_column: str = "can_enter",
) -> pd.DataFrame:
    """Add entry-open to exit-open returns on the supplied trading-session axis."""

    horizons = _validate_horizons(horizons)
    validated = _validate_label_input_frame(
        frame,
        asset_column=asset_column,
        date_column=date_column,
        execution_price_column=execution_price_column,
    )
    if can_enter_column not in validated.columns:
        raise ValueError(f"missing_forward_return_columns: {can_enter_column}")
    axis = _session_axis(sessions)
    result = validated.sort_values([asset_column, date_column], kind="mergesort")
    price = strict_finite_numeric_series(
        result[execution_price_column],
        error="invalid_forward_return_price",
        allow_null=True,
    ).astype("float64")
    keys = pd.MultiIndex.from_arrays([result[asset_column], result[date_column]])
    positions = axis.get_indexer(result[date_column])
    prices = price.to_numpy(dtype="float64", copy=False)
    can_enter = (
        strict_bool_series(
            result[can_enter_column],
            error="invalid_forward_return_can_enter",
            allow_null=True,
        )
        .fillna(False)
        .to_numpy(dtype=bool)
    )
    assets = result[asset_column].to_numpy(copy=False)
    for horizon in horizons:
        labels = np.full(len(result), np.nan, dtype="float64")
        available = np.full(len(result), np.datetime64("NaT"), dtype="datetime64[ns]")
        valid_positions = np.flatnonzero(
            (positions >= 0) & (positions + 1 + horizon < len(axis)) & can_enter
        )
        entry_dates = axis.take(positions[valid_positions] + 1)
        exit_dates = axis.take(positions[valid_positions] + 1 + horizon)
        entry_rows = keys.get_indexer(
            pd.MultiIndex.from_arrays([assets[valid_positions], entry_dates])
        )
        exit_rows = keys.get_indexer(
            pd.MultiIndex.from_arrays([assets[valid_positions], exit_dates])
        )
        present = (entry_rows >= 0) & (exit_rows >= 0)
        valid_positions = valid_positions[present]
        exit_dates = exit_dates[present]
        entry_prices, exit_prices = (
            prices[entry_rows[present]],
            prices[exit_rows[present]],
        )
        positive = (
            np.isfinite(entry_prices)
            & (entry_prices > 0)
            & np.isfinite(exit_prices)
            & (exit_prices > 0)
        )
        valid_positions, exit_dates = valid_positions[positive], exit_dates[positive]
        with np.errstate(over="ignore"):
            values = exit_prices[positive] / entry_prices[positive] - 1.0
        finite = np.isfinite(values)
        labels[valid_positions[finite]] = values[finite]
        available[valid_positions[finite]] = exit_dates[finite].to_numpy(
            dtype="datetime64[ns]"
        )
        label_column = forward_return_label_column(horizon)
        result[label_column] = labels
        result[f"{label_column}_available_at"] = available
    return result.reset_index(drop=True)


def forward_return_label_column(horizon: int) -> str:
    _validate_horizon(horizon, error="invalid_forward_return_horizon")
    return f"forward_return_{horizon}d"


def forward_return_available_at_column(horizon: int) -> str:
    return f"{forward_return_label_column(horizon)}_available_at"


def _session_axis(sessions: Sequence[object]) -> pd.DatetimeIndex:
    values = strict_timestamp_series(
        pd.Series(tuple(sessions)), error="invalid_forward_return_sessions"
    ).dt.normalize()
    axis = pd.DatetimeIndex(values).drop_duplicates()
    if axis.empty or not axis.is_monotonic_increasing:
        raise ValueError("invalid_forward_return_sessions")
    return axis


def _validate_horizons(horizons: object) -> tuple[int, ...]:
    if not isinstance(horizons, tuple) or not horizons:
        raise ValueError("invalid_forward_return_horizons")
    values = tuple(
        _validate_horizon(value, error="invalid_forward_return_horizons")
        for value in horizons
    )
    if len(set(values)) != len(values):
        raise ValueError("invalid_forward_return_horizons")
    return values


def _validate_horizon(value: object, *, error: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(error)
    return value


def _validate_label_input_frame(
    frame: pd.DataFrame,
    *,
    asset_column: str,
    date_column: str,
    execution_price_column: str,
) -> pd.DataFrame:
    required_columns = (asset_column, date_column, execution_price_column)
    missing = [column for column in required_columns if column not in frame.columns]
    if missing:
        raise ValueError(f"missing_forward_return_columns: {', '.join(missing)}")
    result = frame.copy()
    if any(isinstance(value, (list, dict, set)) for value in result[asset_column]):
        raise ValueError("invalid_forward_return_asset_key")
    strict_hashable_key_series(
        result[asset_column], error="invalid_forward_return_asset_key"
    )
    result[date_column] = strict_timestamp_series(
        result[date_column], error="invalid_forward_return_timestamp"
    )
    if result.duplicated([asset_column, date_column]).any():
        raise ValueError("duplicate_forward_return_asset_timestamp")
    return result
