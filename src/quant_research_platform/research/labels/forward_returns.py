"""Common forward-return labels on an explicit ordered time axis."""

from __future__ import annotations

from collections.abc import Hashable, Mapping, Sequence

import numpy as np
import pandas as pd

from quant_research_platform.contracts.data import DataFrequency, horizon_unit_suffix
from quant_research_platform.contracts.time import TimeWindow
from quant_research_platform.contracts.validation import (
    strict_bool_series,
    strict_finite_numeric_series,
    strict_timestamp_series,
)
from quant_research_platform.research.time_series.contiguity import (
    contiguous_segment_ids,
    iter_grouped_timestamps,
)


def add_forward_return_labels(
    frame: pd.DataFrame,
    *,
    horizons: tuple[int, ...],
    frequency: DataFrequency | str = DataFrequency.DAILY,
    asset_column: str = "asset_id",
    date_column: str = "timestamp",
    execution_price_column: str = "adj_open",
    sessions: Sequence[object] | None = None,
    can_enter_column: str | None = None,
    expected_interval: pd.Timedelta | None = None,
    require_contiguous: bool = False,
) -> pd.DataFrame:
    """Add entry-open to exit-open returns, using calendar positions when given."""

    horizons = _validate_horizons(horizons)
    validated = _validate_label_input_frame(
        frame,
        asset_column=asset_column,
        date_column=date_column,
        execution_price_column=execution_price_column,
    )
    if can_enter_column is not None and can_enter_column not in validated.columns:
        raise ValueError(f"missing_forward_return_columns: {can_enter_column}")
    if sessions is None:
        return _add_valid_row_labels(
            validated,
            horizons=horizons,
            frequency=frequency,
            asset_column=asset_column,
            date_column=date_column,
            execution_price_column=execution_price_column,
            can_enter_column=can_enter_column,
            expected_interval=expected_interval,
            require_contiguous=require_contiguous,
        )
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
    can_enter = (
        strict_bool_series(
            result[can_enter_column], error="invalid_forward_return_can_enter"
        ).to_numpy(copy=False)
        if can_enter_column is not None
        else None
    )
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
            if can_enter is not None and not bool(can_enter[position]):
                continue
            entry_price = prices[entry_row]
            exit_price = prices[exit_row]
            if (
                not np.isfinite(entry_price)
                or entry_price == 0.0
                or not np.isfinite(exit_price)
            ):
                continue
            labels[position] = exit_price / entry_price - 1.0
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


def forward_return_valid_bar_read_window(
    panel_keys: pd.DataFrame | Mapping[object, Sequence[pd.Timestamp]],
    *,
    read_window: TimeWindow,
    core_sessions: tuple[pd.Timestamp, ...],
    lookahead_bars: int,
    asset_column: str = "asset_id",
    date_column: str = "timestamp",
) -> TimeWindow:
    """Generic read-window helper for sparse non-calendar streams."""

    if (
        not isinstance(lookahead_bars, int)
        or isinstance(lookahead_bars, bool)
        or lookahead_bars < 0
    ):
        raise ValueError("invalid_forward_return_lookahead_bars")
    if lookahead_bars == 0:
        return read_window
    key_groups = tuple(
        iter_grouped_timestamps(
            panel_keys,
            group_columns=(asset_column,),
            timestamp_column=date_column,
        )
    )
    if not key_groups:
        return read_window
    read_start = pd.Timestamp(read_window.start)
    core_dates = {session.date() for session in core_sessions}
    ends = [pd.Timestamp(read_window.end)]
    for asset_dates in key_groups:
        asset_rows = asset_dates.loc[asset_dates.dt.normalize().ge(read_start)]
        core_positions = [
            position
            for position, timestamp in enumerate(asset_rows)
            if timestamp.date() in core_dates
        ]
        if not core_positions:
            continue
        ends.append(
            asset_rows.iloc[
                min(len(asset_rows) - 1, core_positions[-1] + lookahead_bars)
            ]
        )
    end = max(ends).date()
    return read_window if end <= read_window.end else TimeWindow(read_window.start, end)


def session_read_window(
    *,
    read_window: TimeWindow,
    sessions: Sequence[object],
    lookahead_sessions: int,
) -> TimeWindow:
    """Extend a date window by ordered market-session positions."""

    axis = _session_axis(sessions)
    if len(axis) == 0 or lookahead_sessions <= 0:
        return read_window
    start = int(axis.searchsorted(pd.Timestamp(read_window.start), side="left"))
    stop = int(axis.searchsorted(pd.Timestamp(read_window.end), side="right"))
    if start >= stop:
        return read_window
    end = axis[min(len(axis) - 1, stop - 1 + lookahead_sessions)].date()
    return TimeWindow(read_window.start, end)


def _add_valid_row_labels(
    frame: pd.DataFrame,
    *,
    horizons: tuple[int, ...],
    frequency: DataFrequency | str,
    asset_column: str,
    date_column: str,
    execution_price_column: str,
    can_enter_column: str | None,
    expected_interval: pd.Timedelta | None,
    require_contiguous: bool,
) -> pd.DataFrame:
    result = frame.sort_values([asset_column, date_column], kind="mergesort").copy()
    price = strict_finite_numeric_series(
        result[execution_price_column],
        error="invalid_forward_return_price",
        allow_null=True,
    ).astype("float64")
    grouped = price.groupby(result[asset_column], sort=False)
    timestamp_grouped = result[date_column].groupby(result[asset_column], sort=False)
    if require_contiguous:
        if expected_interval is None:
            raise ValueError("expected_interval_required")
        segments = contiguous_segment_ids(
            result,
            group_columns=(asset_column,),
            timestamp_column=date_column,
            expected_interval=expected_interval,
        )
        grouped = price.groupby([result[asset_column], segments], sort=False)
        timestamp_grouped = result[date_column].groupby(
            [result[asset_column], segments], sort=False
        )
    entry = grouped.shift(-1)
    entry = entry.where(np.isfinite(entry) & entry.ne(0.0))
    can_enter = (
        strict_bool_series(
            result[can_enter_column], error="invalid_forward_return_can_enter"
        )
        if can_enter_column is not None
        else None
    )
    for horizon in horizons:
        exit_price = grouped.shift(-(horizon + 1)).where(np.isfinite)
        column = forward_return_label_column(horizon, frequency)
        result[column] = (exit_price / entry - 1.0).replace([np.inf, -np.inf], np.nan)
        if can_enter is not None:
            result[column] = result[column].where(can_enter)
        result[f"{column}_available_at"] = timestamp_grouped.shift(
            -(horizon + 1)
        ).where(result[column].notna())
    return result


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
