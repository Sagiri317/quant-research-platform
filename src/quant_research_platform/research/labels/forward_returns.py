"""Common forward-return labels on an explicit ordered time axis."""

from __future__ import annotations

from collections.abc import Hashable, Sequence

import numpy as np
import pandas as pd

from quant_research_platform.contracts.data import DataFrequency, horizon_unit_suffix
from quant_research_platform.contracts.validation import (
    strict_bool_series,
    strict_finite_numeric_series,
    strict_timestamp_series,
)


def add_forward_return_labels(
    frame: pd.DataFrame,
    *,
    horizons: tuple[int, ...],
    frequency: DataFrequency | str = DataFrequency.DAILY,
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
    result = validated.sort_values([asset_column, date_column], kind="mergesort").copy()
    result[date_column] = strict_timestamp_series(
        result[date_column], error="invalid_forward_return_timestamp"
    )
    price = strict_finite_numeric_series(
        result[execution_price_column],
        error="invalid_forward_return_price",
        allow_null=True,
    ).astype("float64")
    row_lookup = {
        (asset, _timestamp): position
        for position, (asset, _timestamp) in enumerate(
            zip(result[asset_column], result[date_column], strict=True)
        )
    }
    position_by_session = {timestamp: index for index, timestamp in enumerate(axis)}
    positions = result[date_column].map(position_by_session).to_numpy()
    prices = price.to_numpy(dtype="float64", copy=False)
    can_enter = strict_bool_series(
        result[can_enter_column],
        error="invalid_forward_return_can_enter",
        allow_null=True,
    ).to_numpy(copy=False)
    for horizon in horizons:
        labels = np.full(len(result), np.nan, dtype="float64")
        available = np.full(len(result), np.datetime64("NaT"), dtype="datetime64[ns]")
        for position, (asset, _timestamp) in enumerate(
            zip(result[asset_column], result[date_column], strict=True)
        ):
            session_position = positions[position]
            if pd.isna(session_position):
                continue
            entry_position = int(session_position) + 1
            exit_position = entry_position + horizon
            if exit_position >= len(axis):
                continue
            entry_date = axis[entry_position]
            exit_date = axis[exit_position]
            entry_row = row_lookup.get((asset, entry_date))
            exit_row = row_lookup.get((asset, exit_date))
            if entry_row is None or exit_row is None:
                continue
            if pd.isna(can_enter[position]) or not bool(can_enter[position]):
                continue
            entry_price = prices[entry_row]
            exit_price = prices[exit_row]
            if (
                not np.isfinite(entry_price)
                or entry_price <= 0.0
                or not np.isfinite(exit_price)
                or exit_price <= 0.0
            ):
                continue
            with np.errstate(over="ignore"):
                value = exit_price / entry_price - 1.0
            if not np.isfinite(value):
                continue
            labels[position] = value
            available[position] = exit_date.to_datetime64()
        label_column = forward_return_label_column(horizon, frequency)
        result[label_column] = labels
        result[f"{label_column}_available_at"] = available
    return result.reset_index(drop=True)


def forward_return_label_column(
    horizon: int,
    frequency: DataFrequency | str = DataFrequency.DAILY,
) -> str:
    _validate_horizon(horizon, error="invalid_forward_return_horizon")
    return f"forward_return_{horizon}{horizon_unit_suffix(frequency)}"


def forward_return_available_at_column(
    horizon: int,
    frequency: DataFrequency | str = DataFrequency.DAILY,
) -> str:
    return f"{forward_return_label_column(horizon, frequency)}_available_at"


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
    for value in result[asset_column].tolist():
        _strict_asset_key_value(value)
    result[date_column] = strict_timestamp_series(
        result[date_column], error="invalid_forward_return_timestamp"
    )
    if result.duplicated([asset_column, date_column]).any():
        raise ValueError("duplicate_forward_return_asset_timestamp")
    return result


def _strict_asset_key_value(value: object) -> None:
    if (
        _is_null_asset_key(value)
        or isinstance(value, (list, dict, set))
        or not isinstance(value, Hashable)
    ):
        raise ValueError("invalid_forward_return_asset_key")


def _is_null_asset_key(value: object) -> bool:
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False
