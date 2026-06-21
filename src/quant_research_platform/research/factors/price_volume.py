"""Reusable valid daily bar price-volume factor definitions and calculations."""

from __future__ import annotations

import re
import warnings
from collections.abc import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from quant_research_platform.contracts.data import (
    DataFrequency,
    DataGrain,
    horizon_unit_suffix,
)
from quant_research_platform.contracts.factors import FactorDefinition, FactorSet
from quant_research_platform.contracts.ids import MarketId
from quant_research_platform.contracts.time import TimeWindow

PRICE_VOLUME_FACTOR_FAMILY = "price_volume"

PRICE_VOLUME_STANDARD_PRESET = "standard"
PRICE_VOLUME_CORE_PRESET = "core"
PRICE_VOLUME_SMOKE_PRESET = "smoke"
# QRP semantic policy: skew/kurt are undefined when window moment_2
# (mean squared centered return) is near zero. std remains sample std.
_ROLLING_MOMENT_NEAR_ZERO_VARIANCE = 1e-12

_PRICE_VOLUME_STANDARD_WINDOWS: tuple[int, ...] = (1, 5, 10, 20, 60, 120, 252)


def price_volume_window_column(
    prefix: str,
    window: int,
    frequency: DataFrequency | str = DataFrequency.DAILY,
) -> str:
    return f"{prefix}_{window}{horizon_unit_suffix(frequency)}"


CORE_PRICE_VOLUME_FACTORS: tuple[str, ...] = (
    price_volume_window_column("rev", 1),
    price_volume_window_column("rev", 5),
    price_volume_window_column("rev", 10),
    price_volume_window_column("rev", 20),
    price_volume_window_column("mom", 20),
    price_volume_window_column("mom", 60),
    price_volume_window_column("mom", 120),
    price_volume_window_column("vol", 20),
    price_volume_window_column("vol", 60),
    price_volume_window_column("downside_vol", 20),
    price_volume_window_column("downside_vol", 60),
    price_volume_window_column("avg_amount", 20),
    price_volume_window_column("avg_amount", 60),
    price_volume_window_column("turnover", 20),
    price_volume_window_column("turnover", 60),
    price_volume_window_column("amihud", 20),
    price_volume_window_column("amihud", 60),
    price_volume_window_column("amplitude", 20),
    price_volume_window_column("amplitude", 60),
    "log_float_market_cap",
    "log_total_market_cap",
)
SMOKE_PRICE_VOLUME_FACTORS: tuple[str, ...] = (
    price_volume_window_column("rev", 1),
    price_volume_window_column("mom", 20),
    price_volume_window_column("vol", 20),
    price_volume_window_column("avg_amount", 20),
    price_volume_window_column("turnover", 20),
    price_volume_window_column("amihud", 20),
    price_volume_window_column("amplitude", 20),
    "log_float_market_cap",
    "log_total_market_cap",
)

PRICE_VOLUME_FEATURE_EXCLUDE_PREFIXES: tuple[str, ...] = (
    "forward_return",
    "target",
    "label",
)
PRICE_VOLUME_FACTOR_PREFIXES: tuple[str, ...] = (
    "mom_",
    "rev_",
    "vol_",
    "downside_vol_",
    "upside_vol_",
    "max_return_",
    "min_return_",
    "drawdown_",
    "skew_",
    "kurt_",
    "avg_amount_",
    "turnover_",
    "turnover_chg_",
    "turnover_vol_",
    "amihud_",
    "zero_volume_ratio_",
    "amplitude_",
    "intraday_range_",
    "log_float_market_cap",
    "log_total_market_cap",
)
PRICE_VOLUME_FEATURE_EXCLUDE_COLUMNS: frozenset[str] = frozenset(
    {
        "market_id",
        "asset_id",
        "timestamp",
    }
)
BASE_PRICE_VOLUME_FACTOR_REQUIREMENTS: dict[str, tuple[str, ...]] = {
    price_volume_window_column("rev", 1): ("period_return",),
    "mom": ("adj_close",),
    "rev": ("adj_close",),
    "vol": ("period_return",),
    "downside_vol": ("period_return",),
    "upside_vol": ("period_return",),
    "max_return": ("period_return",),
    "min_return": ("period_return",),
    "drawdown": ("adj_close",),
    "skew": ("period_return",),
    "kurt": ("period_return",),
    "avg_amount": ("amount",),
    "turnover": ("turnover",),
    "turnover_chg": ("turnover",),
    "turnover_vol": ("turnover",),
    "amihud": ("period_return", "amount"),
    "zero_volume_ratio": ("amount",),
    "amplitude": ("high", "low", "prev_close"),
    "intraday_range": ("high", "low", "adj_close"),
    "log_float_market_cap": ("float_market_cap",),
    "log_total_market_cap": ("total_market_cap",),
}


def price_volume_factor_columns_for_windows(
    windows: Iterable[int],
) -> tuple[str, ...]:
    columns: list[str] = [price_volume_window_column("rev", 1)]
    for window in windows:
        if window != 1:
            columns.extend(
                (
                    price_volume_window_column("rev", window),
                    price_volume_window_column("mom", window),
                )
            )
    for window in (5, 20, 60, 120):
        columns.append(price_volume_window_column("vol", window))
    for window in (20, 60):
        columns.extend(
            (
                price_volume_window_column("downside_vol", window),
                price_volume_window_column("upside_vol", window),
                price_volume_window_column("max_return", window),
                price_volume_window_column("min_return", window),
                price_volume_window_column("drawdown", window),
                price_volume_window_column("skew", window),
                price_volume_window_column("kurt", window),
            )
        )
    for window in (5, 20, 60, 120):
        columns.extend(
            (
                price_volume_window_column("avg_amount", window),
                price_volume_window_column("turnover", window),
            )
        )
    for window in (20, 60):
        columns.extend(
            (
                price_volume_window_column("turnover_chg", window),
                price_volume_window_column("turnover_vol", window),
            )
        )
    for window in (20, 60, 120):
        columns.append(price_volume_window_column("amihud", window))
    for window in (20, 60):
        columns.append(price_volume_window_column("zero_volume_ratio", window))
    for window in (5, 20, 60, 120):
        columns.append(price_volume_window_column("amplitude", window))
    for window in (20, 60):
        columns.append(price_volume_window_column("intraday_range", window))
    columns.extend(("log_float_market_cap", "log_total_market_cap"))
    return tuple(dict.fromkeys(columns))


STANDARD_PRICE_VOLUME_FACTORS: tuple[str, ...] = (
    price_volume_factor_columns_for_windows(_PRICE_VOLUME_STANDARD_WINDOWS)
)

PRICE_VOLUME_FACTOR_PRESETS: dict[str, tuple[str, ...]] = {
    PRICE_VOLUME_STANDARD_PRESET: STANDARD_PRICE_VOLUME_FACTORS,
    PRICE_VOLUME_CORE_PRESET: CORE_PRICE_VOLUME_FACTORS,
    PRICE_VOLUME_SMOKE_PRESET: SMOKE_PRICE_VOLUME_FACTORS,
}


def price_volume_factor_columns(*, preset: str = "standard") -> tuple[str, ...]:
    """Return the official price-volume factor columns for a named preset."""

    try:
        return PRICE_VOLUME_FACTOR_PRESETS[preset]
    except KeyError as exc:
        raise ValueError("invalid_price_volume_preset") from exc


def price_volume_factor_definitions(
    *,
    market_id: MarketId,
    required_input_data_id: str,
    preset: str = "standard",
) -> tuple[FactorDefinition, ...]:
    """Return factor definitions for one market binding."""

    definitions: list[FactorDefinition] = []
    for factor_id in price_volume_factor_columns(preset=preset):
        base_inputs = price_volume_factor_input_columns(factor_id)
        lookback_window = _lookback_window(factor_id)
        definitions.append(
            FactorDefinition(
                factor_id=factor_id,
                market_id=market_id,
                factor_set=_factor_set_for_column(factor_id),
                input_columns=base_inputs,
                base_input_columns=base_inputs,
                derived_input_columns=(),
                output_column=factor_id,
                display_name=factor_id,
                description=(
                    "Daily price-volume factor registered by the QRP 0.1 "
                    "factor catalog."
                ),
                family=PRICE_VOLUME_FACTOR_FAMILY,
                required_windows=_required_windows(factor_id),
                lookback_window=lookback_window,
                grain=DataGrain.ASSET_TIMESTAMP,
                timing_semantics="same_session_close_available_at_next_session",
                null_policy="preserve_nulls_after_min_periods",
                source_layer="processed",
                required_input_data_id=required_input_data_id,
                window=lookback_window,
            )
        )
    return tuple(definitions)


def price_volume_factor_input_columns(factor_id: str) -> tuple[str, ...]:
    """Return exact base input columns required by one price-volume factor."""

    required = _required_fields_for_factor(factor_id)
    if not required:
        raise ValueError(f"unknown_price_volume_factor: {factor_id}")
    return required


def unsupported_price_volume_factors(
    factor_columns: Iterable[str],
) -> tuple[str, ...]:
    """Return requested factor names that the price-volume engine cannot compute."""

    unsupported: list[str] = []
    for factor in tuple(dict.fromkeys(factor_columns)):
        if not _required_fields_for_factor(factor):
            unsupported.append(factor)
    return tuple(unsupported)


def calculate_price_volume_factors(
    panel: pd.DataFrame,
    *,
    factor_columns: Iterable[str] | None = None,
    asset_column: str = "asset_id",
    date_column: str = "timestamp",
    price_column: str = "adj_close",
    return_column: str = "period_return",
    amount_column: str = "amount",
    turnover_column: str = "turnover",
    high_column: str = "high",
    low_column: str = "low",
    prev_close_column: str = "prev_close",
    float_market_cap_column: str = "float_market_cap",
    total_market_cap_column: str = "total_market_cap",
) -> pd.DataFrame:
    """Add requested price-volume factors without future data."""

    requested = tuple(
        dict.fromkeys(factor_columns or price_volume_factor_columns())
    )
    unsupported = unsupported_price_volume_factors(requested)
    if unsupported:
        raise ValueError(
            "unsupported_price_volume_factors: " + ", ".join(unsupported)
        )

    required_inputs = tuple(
        dict.fromkeys(
            (
                asset_column,
                date_column,
                *(
                    column
                    for factor in requested
                    for column in price_volume_factor_input_columns(factor)
                ),
            )
        )
    )
    missing_inputs = [
        column for column in required_inputs if column not in panel.columns
    ]
    if missing_inputs:
        raise ValueError(
            "missing_price_volume_input_columns: " + ", ".join(missing_inputs)
        )

    requested_set = set(requested)
    required_windows = _windows_from_requested_factors(requested_set)
    result = panel.sort_values([asset_column, date_column]).copy()
    group_keys = result[asset_column]
    grouped = result.groupby(asset_column, sort=False)
    group_positions = tuple(grouped.indices.values())
    needs_trailing_price = any(
        price_volume_window_column("mom", window) in requested_set
        or price_volume_window_column("rev", window) in requested_set
        for window in required_windows
        if window != 1
    )
    needs_price = (
        needs_trailing_price
        or any(
            price_volume_window_column("drawdown", window) in requested_set
            for window in required_windows
        )
        or any(
            price_volume_window_column("intraday_range", window) in requested_set
            for window in required_windows
        )
    )
    needs_returns = (
        price_volume_window_column("rev", 1) in requested_set
        or any(
            price_volume_window_column(prefix, window) in requested_set
            for prefix in (
                "vol",
                "downside_vol",
                "upside_vol",
                "max_return",
                "min_return",
                "skew",
                "kurt",
                "amihud",
            )
            for window in required_windows
        )
    )
    needs_amount = any(
        price_volume_window_column(prefix, window) in requested_set
        for prefix in ("avg_amount", "amihud", "zero_volume_ratio")
        for window in required_windows
    )
    needs_turnover = any(
        price_volume_window_column(prefix, window) in requested_set
        for prefix in ("turnover", "turnover_chg", "turnover_vol")
        for window in required_windows
    )
    price = (
        pd.to_numeric(result[price_column], errors="coerce")
        if needs_price
        else None
    )
    returns = (
        pd.to_numeric(result[return_column], errors="coerce")
        if needs_returns
        else None
    )
    amount = (
        pd.to_numeric(result[amount_column], errors="coerce")
        if needs_amount
        else None
    )
    turnover = (
        pd.to_numeric(result[turnover_column], errors="coerce")
        if needs_turnover
        else None
    )
    price_grouped = price.groupby(group_keys, sort=False) if price is not None else None
    amount_grouped = (
        amount.groupby(group_keys, sort=False) if amount is not None else None
    )
    turnover_grouped = (
        turnover.groupby(group_keys, sort=False) if turnover is not None else None
    )

    one_bar_reversal = price_volume_window_column("rev", 1)
    if one_bar_reversal in requested_set:
        result[one_bar_reversal] = -returns
    for window in required_windows:
        if window == 1:
            continue
        momentum_column = price_volume_window_column("mom", window)
        reversal_column = price_volume_window_column("rev", window)
        needs_momentum = momentum_column in requested_set
        needs_reversal = reversal_column in requested_set
        if needs_momentum or needs_reversal:
            shifted_price = price_grouped.shift(window)
            denominator = _valid_denominator(shifted_price)
            trailing_return = _finite_or_nan(price / denominator - 1.0)
        if needs_momentum:
            result[momentum_column] = trailing_return
        if needs_reversal:
            result[reversal_column] = -trailing_return

    for window in required_windows:
        column = price_volume_window_column("vol", window)
        if column in requested_set:
            result[column] = _rolling_std(
                returns,
                group_keys,
                group_positions,
                window,
            )
        column = price_volume_window_column("downside_vol", window)
        if column in requested_set:
            result[column] = _rolling_rms(
                returns.clip(upper=0.0),
                group_keys,
                group_positions,
                window,
            )
        column = price_volume_window_column("upside_vol", window)
        if column in requested_set:
            result[column] = _rolling_rms(
                returns.clip(lower=0.0),
                group_keys,
                group_positions,
                window,
            )
        column = price_volume_window_column("max_return", window)
        if column in requested_set:
            result[column] = _rolling_aggregate(
                result[return_column],
                grouped[return_column],
                group_positions,
                window,
                "max",
            )
        column = price_volume_window_column("min_return", window)
        if column in requested_set:
            result[column] = _rolling_aggregate(
                result[return_column],
                grouped[return_column],
                group_positions,
                window,
                "min",
            )
        column = price_volume_window_column("drawdown", window)
        if column in requested_set:
            result[column] = _rolling_drawdown(
                price,
                group_keys,
                group_positions,
                window,
            )
        column = price_volume_window_column("skew", window)
        if column in requested_set:
            result[column] = _rolling_skew(
                returns,
                group_keys,
                group_positions,
                window,
            )
        column = price_volume_window_column("kurt", window)
        if column in requested_set:
            result[column] = _rolling_kurt(
                returns,
                group_keys,
                group_positions,
                window,
            )

    for window in required_windows:
        column = price_volume_window_column("avg_amount", window)
        if column in requested_set:
            result[column] = _rolling_aggregate(
                amount,
                amount_grouped,
                group_positions,
                window,
                "mean",
            )
        column = price_volume_window_column("turnover", window)
        if column in requested_set:
            result[column] = _rolling_aggregate(
                turnover,
                turnover_grouped,
                group_positions,
                window,
                "mean",
            )
        column = price_volume_window_column("turnover_chg", window)
        if column in requested_set:
            shifted_turnover = turnover_grouped.shift(window)
            denominator = _valid_denominator(shifted_turnover)
            result[column] = turnover / denominator - 1.0
        column = price_volume_window_column("turnover_vol", window)
        if column in requested_set:
            result[column] = _rolling_std(
                turnover,
                group_keys,
                group_positions,
                window,
            )
        column = price_volume_window_column("zero_volume_ratio", window)
        if column in requested_set:
            zero_volume = (amount <= 0.0).astype(float)
            result[column] = _rolling_aggregate(
                zero_volume,
                zero_volume.groupby(group_keys, sort=False),
                group_positions,
                window,
                "mean",
            )

    illiquidity = None
    if any(
        price_volume_window_column("amihud", window) in requested_set
        for window in required_windows
    ):
        denominator = _valid_denominator(amount)
        illiquidity = (returns.abs() / denominator).replace([np.inf, -np.inf], np.nan)
    amplitude = None
    if any(
        price_volume_window_column("amplitude", window) in requested_set
        for window in required_windows
    ):
        amplitude = (
            pd.to_numeric(result[high_column], errors="coerce")
            - pd.to_numeric(result[low_column], errors="coerce")
        ) / _valid_denominator(
            pd.to_numeric(result[prev_close_column], errors="coerce")
        )
        amplitude = _finite_or_nan(amplitude)
    for window in required_windows:
        column = price_volume_window_column("amihud", window)
        if column in requested_set:
            result[column] = _rolling_aggregate(
                illiquidity,
                illiquidity.groupby(group_keys, sort=False),
                group_positions,
                window,
                "mean",
            )
        column = price_volume_window_column("amplitude", window)
        if column in requested_set:
            result[column] = _rolling_aggregate(
                amplitude,
                amplitude.groupby(group_keys, sort=False),
                group_positions,
                window,
                "mean",
            )

    intraday_range = None
    if any(
        price_volume_window_column("intraday_range", window) in requested_set
        for window in required_windows
    ):
        intraday_range = (
            pd.to_numeric(result[high_column], errors="coerce")
            - pd.to_numeric(result[low_column], errors="coerce")
        ) / _valid_denominator(price)
        intraday_range = _finite_or_nan(intraday_range)
    for window in required_windows:
        column = price_volume_window_column("intraday_range", window)
        if column in requested_set:
            result[column] = _rolling_aggregate(
                intraday_range,
                intraday_range.groupby(group_keys, sort=False),
                group_positions,
                window,
                "mean",
            )

    if "log_float_market_cap" in requested_set:
        result["log_float_market_cap"] = _safe_log(result[float_market_cap_column])
    if "log_total_market_cap" in requested_set:
        result["log_total_market_cap"] = _safe_log(result[total_market_cap_column])
    output_columns = [column for column in requested if column in result.columns]
    for column in output_columns:
        result[column] = _finite_or_nan(result[column])
    return result


def price_volume_valid_bar_read_window(
    panel_keys: pd.DataFrame | Mapping[object, Sequence[pd.Timestamp]],
    *,
    read_window: TimeWindow,
    core_sessions: tuple[pd.Timestamp, ...],
    lookback_bars: int,
    asset_column: str = "asset_id",
    date_column: str = "timestamp",
) -> TimeWindow:
    if lookback_bars <= 0:
        return read_window
    key_groups = tuple(
        _iter_panel_key_groups(
            panel_keys,
            asset_column=asset_column,
            date_column=date_column,
        )
    )
    if not key_groups:
        return read_window
    read_end = pd.Timestamp(read_window.end)
    core_dates = {session.date() for session in core_sessions}
    starts = [pd.Timestamp(read_window.start)]
    saw_core = False
    for asset_dates in key_groups:
        asset_rows = asset_dates.loc[asset_dates.dt.normalize().le(read_end)]
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
        read_position = max(0, asset_core_positions[0] - lookback_bars)
        starts.append(asset_rows.iloc[read_position].normalize())
    if not saw_core:
        return read_window
    start = min(starts).date()
    if start >= read_window.start:
        return read_window
    return TimeWindow(start, read_window.end)


def _iter_panel_key_groups(
    panel_keys: pd.DataFrame | Mapping[object, Sequence[pd.Timestamp]],
    *,
    asset_column: str,
    date_column: str,
) -> Iterable[pd.Series]:
    if isinstance(panel_keys, pd.DataFrame):
        if panel_keys.empty:
            return
        keys = panel_keys.loc[:, [asset_column, date_column]].copy()
        keys[date_column] = pd.to_datetime(keys[date_column], errors="coerce")
        keys = keys.dropna(subset=[asset_column, date_column])
        if keys.empty:
            return
        keys = keys.sort_values([asset_column, date_column])
        for _, asset_rows in keys.groupby(asset_column, sort=False):
            yield asset_rows[date_column].reset_index(drop=True)
        return
    for timestamps in panel_keys.values():
        dates = pd.Series(pd.to_datetime(tuple(timestamps), errors="coerce"))
        dates = dates.dropna().sort_values(kind="mergesort").reset_index(drop=True)
        if not dates.empty:
            yield dates


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


def largest_price_volume_lookback_days(
    factor_columns: Iterable[str],
) -> int:
    """Return the largest rolling/trailing lookback encoded in factor names."""

    windows = []
    for column in factor_columns:
        if column == price_volume_window_column("rev", 1):
            windows.append(1)
        windows.extend(int(value) for value in re.findall(r"_(\d+)d(?:_|$)", column))
    return max(windows, default=0)


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


def _factor_set_for_column(factor_id: str) -> FactorSet:
    if factor_id in SMOKE_PRICE_VOLUME_FACTORS:
        return FactorSet.SMOKE
    if factor_id in CORE_PRICE_VOLUME_FACTORS:
        return FactorSet.CORE
    return FactorSet.STANDARD


def _required_windows(factor_id: str) -> tuple[int, ...]:
    windows = tuple(int(match) for match in re.findall(r"_(\d+)d\b", factor_id))
    return windows


def _lookback_window(factor_id: str) -> int | None:
    windows = _required_windows(factor_id)
    if not windows:
        return None
    return max(windows)


def _rolling_aggregate(
    values: pd.Series,
    grouped_values: pd.core.groupby.SeriesGroupBy,
    group_positions: tuple[np.ndarray, ...],
    window: int,
    aggregation: str,
) -> pd.Series:
    del grouped_values
    numeric = pd.to_numeric(values, errors="coerce")
    source = numeric.to_numpy(dtype="float64", copy=False)
    output = np.full(len(values), np.nan, dtype="float64")
    for positions in group_positions:
        output[positions] = _rolling_aggregate_array(
            source[positions],
            window,
            aggregation,
        )
    return pd.Series(output, index=values.index, name=values.name)


def _rolling_aggregate_array(
    values: np.ndarray,
    window: int,
    aggregation: str,
) -> np.ndarray:
    output = np.full(len(values), np.nan, dtype="float64")
    if window <= 0 or len(values) < window:
        return output
    windows = np.lib.stride_tricks.sliding_window_view(values, window_shape=window)
    valid_windows = np.isfinite(windows).all(axis=1)
    if not valid_windows.any():
        return output
    valid_values = windows[valid_windows]
    if aggregation == "mean":
        window_values = np.mean(valid_values, axis=1)
    elif aggregation == "max":
        window_values = np.max(valid_values, axis=1)
    elif aggregation == "min":
        window_values = np.min(valid_values, axis=1)
    elif aggregation == "std":
        window_values = np.std(valid_values, axis=1, ddof=1)
    else:
        raise ValueError(f"unsupported_rolling_aggregation: {aggregation}")
    values_out = np.full(len(windows), np.nan, dtype="float64")
    values_out[valid_windows] = window_values
    output[window - 1 :] = values_out
    return output


def _rolling_rms(
    values: pd.Series,
    group_keys: pd.Series,
    group_positions: tuple[np.ndarray, ...],
    window: int,
) -> pd.Series:
    squared = values.pow(2)
    mean_squared = _rolling_aggregate(
        squared,
        squared.groupby(group_keys, sort=False),
        group_positions,
        window,
        "mean",
    )
    return np.sqrt(mean_squared)


def _rolling_std(
    values: pd.Series,
    group_keys: pd.Series,
    group_positions: tuple[np.ndarray, ...],
    window: int,
) -> pd.Series:
    return _rolling_window_statistic(values, group_keys, group_positions, window, "std")


def _rolling_skew(
    values: pd.Series,
    group_keys: pd.Series,
    group_positions: tuple[np.ndarray, ...],
    window: int,
) -> pd.Series:
    return _rolling_window_statistic(
        values,
        group_keys,
        group_positions,
        window,
        "skew",
    )


def _rolling_kurt(
    values: pd.Series,
    group_keys: pd.Series,
    group_positions: tuple[np.ndarray, ...],
    window: int,
) -> pd.Series:
    return _rolling_window_statistic(
        values,
        group_keys,
        group_positions,
        window,
        "kurt",
    )


def _rolling_window_statistic(
    values: pd.Series,
    group_keys: pd.Series,
    group_positions: tuple[np.ndarray, ...],
    window: int,
    statistic: str,
) -> pd.Series:
    del group_keys
    numeric = pd.to_numeric(values, errors="coerce")
    source = numeric.to_numpy(dtype="float64", copy=False)
    output = np.full(len(numeric), np.nan, dtype="float64")
    for positions in group_positions:
        output[positions] = _rolling_window_statistic_array(
            source[positions],
            window,
            statistic,
        )
    return pd.Series(output, index=values.index, name=values.name)


def _rolling_window_statistic_array(
    values: np.ndarray,
    window: int,
    statistic: str,
) -> np.ndarray:
    output = np.full(len(values), np.nan, dtype="float64")
    if window <= 0 or len(values) < window:
        return output
    windows = np.lib.stride_tricks.sliding_window_view(values, window_shape=window)
    valid_windows = np.isfinite(windows).all(axis=1)
    if not valid_windows.any():
        return output
    valid_values = windows[valid_windows]
    if statistic == "std":
        window_values = _window_std_values(valid_values)
    elif statistic == "skew":
        window_values = _window_skew_values(valid_values)
    elif statistic == "kurt":
        window_values = _window_kurt_values(valid_values)
    else:
        raise ValueError(f"unsupported_rolling_statistic: {statistic}")
    values_out = np.full(len(windows), np.nan, dtype="float64")
    values_out[valid_windows] = window_values
    output[window - 1 :] = values_out
    return output


def _window_std_values(windows: np.ndarray) -> np.ndarray:
    counts = windows.shape[1]
    if counts < 2:
        return np.full(len(windows), np.nan, dtype="float64")
    centered = windows - windows.mean(axis=1, keepdims=True)
    variance = np.sum(centered * centered, axis=1) / (counts - 1)
    variance = np.maximum(variance, 0.0)
    values = np.sqrt(variance)
    return values


def _window_skew_values(windows: np.ndarray) -> np.ndarray:
    counts = windows.shape[1]
    if counts < 3:
        return np.full(len(windows), np.nan, dtype="float64")
    centered = windows - windows.mean(axis=1, keepdims=True)
    moment_2 = np.mean(centered * centered, axis=1)
    moment_3 = np.mean(centered * centered * centered, axis=1)
    valid = moment_2 > _ROLLING_MOMENT_NEAR_ZERO_VARIANCE
    values = np.full(len(windows), np.nan, dtype="float64")
    if not valid.any():
        return values
    denominator = np.power(moment_2[valid], 1.5)
    with np.errstate(divide="ignore", invalid="ignore"):
        values[valid] = (
            np.sqrt(counts * (counts - 1))
            / (counts - 2)
            * (moment_3[valid] / denominator)
        )
    values[~np.isfinite(values)] = np.nan
    return values


def _window_kurt_values(windows: np.ndarray) -> np.ndarray:
    counts = windows.shape[1]
    if counts < 4:
        return np.full(len(windows), np.nan, dtype="float64")
    centered = windows - windows.mean(axis=1, keepdims=True)
    moment_2 = np.mean(centered * centered, axis=1)
    moment_4 = np.mean(centered**4, axis=1)
    valid = moment_2 > _ROLLING_MOMENT_NEAR_ZERO_VARIANCE
    values = np.full(len(windows), np.nan, dtype="float64")
    if not valid.any():
        return values
    with np.errstate(divide="ignore", invalid="ignore"):
        values[valid] = (
            (counts - 1)
            / ((counts - 2) * (counts - 3))
            * (
                (counts + 1)
                * (moment_4[valid] / (moment_2[valid] * moment_2[valid]) - 3.0)
                + 6.0
            )
        )
    values[~np.isfinite(values)] = np.nan
    return values


def _rolling_drawdown(
    values: pd.Series,
    group_keys: pd.Series,
    group_positions: tuple[np.ndarray, ...],
    window: int,
) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    source = numeric.to_numpy(dtype="float64", copy=False)
    output = np.full(len(numeric), np.nan, dtype="float64")
    for positions in group_positions:
        output[positions] = _rolling_drawdown_array(source[positions], window)
    return pd.Series(output, index=values.index, name=values.name)


def _rolling_drawdown_array(values: np.ndarray, window: int) -> np.ndarray:
    output = np.full(len(values), np.nan, dtype="float64")
    if window <= 0:
        return output
    if len(values) < window:
        return output
    windows = np.lib.stride_tricks.sliding_window_view(values, window_shape=window)
    valid_windows = np.isfinite(windows).all(axis=1)
    if not valid_windows.any():
        return output
    valid_values = windows[valid_windows]
    peaks = np.maximum.accumulate(valid_values, axis=1)
    denominators = np.where(peaks == 0.0, np.nan, peaks)
    with np.errstate(divide="ignore", invalid="ignore"):
        drawdowns = valid_values / denominators - 1.0
    drawdowns[~np.isfinite(drawdowns)] = np.nan
    all_nan = np.isnan(drawdowns).all(axis=1)
    minima = np.full(len(drawdowns), np.nan, dtype="float64")
    if (~all_nan).any():
        minima[~all_nan] = np.nanmin(drawdowns[~all_nan], axis=1)
    window_values = np.full(len(windows), np.nan, dtype="float64")
    window_values[valid_windows] = minima
    output[window - 1 :] = window_values
    return output


def _safe_log(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    numeric = numeric.where(np.isfinite(numeric) & numeric.gt(0.0))
    return _finite_or_nan(np.log(numeric))


def _finite_or_nan(values: pd.Series) -> pd.Series:
    return values.replace([np.inf, -np.inf], np.nan)


def _valid_denominator(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    return numeric.where(np.isfinite(numeric) & numeric.ne(0.0))


def _is_excluded_feature(column: str) -> bool:
    normalized = column.casefold()
    if column in PRICE_VOLUME_FEATURE_EXCLUDE_COLUMNS:
        return True
    return any(
        normalized.startswith(prefix)
        for prefix in PRICE_VOLUME_FEATURE_EXCLUDE_PREFIXES
    )


def _required_fields_for_factor(factor: str) -> tuple[str, ...]:
    if factor in BASE_PRICE_VOLUME_FACTOR_REQUIREMENTS:
        return BASE_PRICE_VOLUME_FACTOR_REQUIREMENTS[factor]
    for prefix, required in BASE_PRICE_VOLUME_FACTOR_REQUIREMENTS.items():
        if factor.startswith(f"{prefix}_"):
            return required
    return ()


def _windows_from_requested_factors(requested: set[str]) -> tuple[int, ...]:
    values = {1} if price_volume_window_column("rev", 1) in requested else set()
    for factor in requested:
        for token in factor.split("_"):
            if token.endswith("d") and token[:-1].isdigit():
                values.add(int(token[:-1]))
    return tuple(sorted(values))


__all__ = [
    "BASE_PRICE_VOLUME_FACTOR_REQUIREMENTS",
    "CORE_PRICE_VOLUME_FACTORS",
    "PRICE_VOLUME_CORE_PRESET",
    "PRICE_VOLUME_FACTOR_FAMILY",
    "PRICE_VOLUME_FACTOR_PRESETS",
    "PRICE_VOLUME_FEATURE_EXCLUDE_COLUMNS",
    "PRICE_VOLUME_FEATURE_EXCLUDE_PREFIXES",
    "PRICE_VOLUME_FACTOR_PREFIXES",
    "PRICE_VOLUME_SMOKE_PRESET",
    "PRICE_VOLUME_STANDARD_PRESET",
    "SMOKE_PRICE_VOLUME_FACTORS",
    "STANDARD_PRICE_VOLUME_FACTORS",
    "calculate_price_volume_factors",
    "largest_price_volume_lookback_days",
    "price_volume_factor_columns",
    "price_volume_factor_columns_for_windows",
    "price_volume_factor_definitions",
    "price_volume_factor_input_columns",
    "price_volume_window_column",
    "price_volume_valid_bar_read_window",
    "select_price_volume_feature_columns_from_schema",
    "select_price_volume_features",
    "unsupported_price_volume_factors",
    "warn_for_missing_price_volume_features",
]
