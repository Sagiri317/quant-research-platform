"""Generic forward-return label helpers.

Forward-return horizons use per-asset valid bars. They do not imply
calendar/session windows and do not fill missing or suspended rows.
"""

from __future__ import annotations

from collections.abc import Hashable, Mapping, Sequence

import numpy as np
import pandas as pd

from quant_research_platform.contracts.data import (
    DataFrequency,
    horizon_unit_suffix,
)
from quant_research_platform.contracts.time import TimeWindow
from quant_research_platform.contracts.validation import (
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
    expected_interval: pd.Timedelta | None = None,
    require_contiguous: bool = False,
) -> pd.DataFrame:
    """Add per-asset valid-bar next-open to horizon-open labels."""

    horizons = _validate_horizons(horizons)
    frame = _validate_label_input_frame(
        frame,
        asset_column=asset_column,
        date_column=date_column,
        execution_price_column=execution_price_column,
    )
    result = frame.sort_values([asset_column, date_column]).copy()
    price = strict_finite_numeric_series(
        result[execution_price_column],
        error="invalid_forward_return_price",
        allow_null=True,
    ).astype("float64")
    grouped_price = price.groupby(result[asset_column], sort=False)
    if require_contiguous:
        if expected_interval is None:
            raise ValueError("expected_interval_required")
        segments = contiguous_segment_ids(
            result,
            group_columns=(asset_column,),
            timestamp_column=date_column,
            expected_interval=expected_interval,
        )
        grouped_price = price.groupby([result[asset_column], segments], sort=False)

    entry_price = grouped_price.shift(-1)
    entry_price = entry_price.where(np.isfinite(entry_price) & entry_price.ne(0.0))
    for horizon in horizons:
        exit_price = grouped_price.shift(-(horizon + 1))
        exit_price = exit_price.where(np.isfinite(exit_price))
        label = exit_price / entry_price - 1.0
        result[forward_return_label_column(horizon, frequency)] = label.replace(
            [np.inf, -np.inf],
            np.nan,
        )

    return result


def forward_return_label_column(
    horizon: int,
    frequency: DataFrequency | str = DataFrequency.DAILY,
) -> str:
    _validate_horizon(horizon, error="invalid_forward_return_horizon")
    return f"forward_return_{horizon}{horizon_unit_suffix(frequency)}"


def forward_return_valid_bar_read_window(
    panel_keys: pd.DataFrame | Mapping[object, Sequence[pd.Timestamp]],
    *,
    read_window: TimeWindow,
    core_sessions: tuple[pd.Timestamp, ...],
    lookahead_bars: int,
    asset_column: str = "asset_id",
    date_column: str = "timestamp",
) -> TimeWindow:
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
    saw_core = False
    for asset_dates in key_groups:
        asset_rows = asset_dates.loc[asset_dates.dt.normalize().ge(read_start)]
        if asset_rows.empty:
            continue
        asset_core_positions = [
            position
            for position, timestamp in enumerate(asset_rows)
            if timestamp.date() in core_dates
        ]
        if not asset_core_positions:
            continue
        saw_core = True
        read_position = min(
            len(asset_rows) - 1,
            asset_core_positions[-1] + lookahead_bars,
        )
        ends.append(asset_rows.iloc[read_position].normalize())
    if not saw_core:
        return read_window
    end = max(ends).date()
    if end <= read_window.end:
        return read_window
    return TimeWindow(read_window.start, end)


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
        result[date_column],
        error="invalid_forward_return_timestamp",
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
