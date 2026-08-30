"""Generic microstructure factor math."""

from __future__ import annotations

import pandas as pd
from pandas.api.types import is_bool_dtype

from quant_research_platform.contracts.validation import strict_finite_numeric_series
from quant_research_platform.research.time_series.contiguity import (
    contiguous_segment_ids,
)

DEFAULT_EXPECTED_INTERVAL = pd.Timedelta(minutes=1)


def calculate_microstructure_factors(
    frame: pd.DataFrame,
    *,
    rolling_window: int = 5,
    expected_interval: pd.Timedelta = DEFAULT_EXPECTED_INTERVAL,
    min_periods: int = 5,
    trade_flow_observed_column: str | None = None,
) -> pd.DataFrame:
    """Add the approved 1m microstructure factors using contiguous clock windows."""

    if rolling_window != 5:
        raise ValueError("unsupported_microstructure_rolling_window")
    if min_periods != rolling_window:
        raise ValueError("microstructure_rolling_requires_full_window")
    if not isinstance(expected_interval, pd.Timedelta):
        raise ValueError("invalid_expected_interval")
    if expected_interval <= pd.Timedelta(0):
        raise ValueError("invalid_expected_interval")

    result = frame.sort_values(
        ["market_id", "asset_id", "timestamp"],
        kind="mergesort",
    ).reset_index(drop=True)
    best_bid = _numeric(result, "best_bid")
    best_ask = _numeric(result, "best_ask")
    mid_price = _numeric(result, "mid_price")
    bid_quantity = _numeric(result, "bid_quantity")
    ask_quantity = _numeric(result, "ask_quantity")
    bid_depth = _numeric(result, "bid_depth")
    ask_depth = _numeric(result, "ask_depth")

    spread = best_ask - best_bid
    result["spread"] = spread
    result["relative_spread"] = (spread / mid_price).mask(mid_price.le(0))

    depth_denominator = bid_depth + ask_depth
    result["depth_imbalance"] = ((bid_depth - ask_depth) / depth_denominator).mask(
        depth_denominator.le(0)
    )

    quote_denominator = bid_quantity + ask_quantity
    microprice = (
        (best_ask * bid_quantity + best_bid * ask_quantity) / quote_denominator
    ).mask(quote_denominator.le(0))
    result["microprice"] = microprice
    result["microprice_deviation"] = (microprice / mid_price - 1.0).mask(
        mid_price.le(0)
    )

    if trade_flow_observed_column is None:
        observed = pd.Series(True, index=result.index, dtype="bool")
    elif trade_flow_observed_column in result:
        observed = _strict_bool_series(result[trade_flow_observed_column])
    else:
        raise ValueError("missing_trade_flow_observed_column")
    trade_count_sum = _rolling_group_sum(
        result,
        _numeric(result, "trade_count"),
        rolling_window=rolling_window,
        expected_interval=expected_interval,
        min_periods=min_periods,
        observed=observed,
    )
    signed_volume_sum = _rolling_group_sum(
        result,
        _numeric(result, "signed_volume"),
        rolling_window=rolling_window,
        expected_interval=expected_interval,
        min_periods=min_periods,
        observed=observed,
    )
    volume_sum = _rolling_group_sum(
        result,
        _numeric(result, "volume"),
        rolling_window=rolling_window,
        expected_interval=expected_interval,
        min_periods=min_periods,
        observed=observed,
    )
    result["trade_intensity_5m"] = trade_count_sum
    result["signed_volume_5m"] = signed_volume_sum
    result["flow_imbalance_5m"] = (signed_volume_sum / volume_sum).mask(
        volume_sum.le(0)
    )
    return result


def _numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    return strict_finite_numeric_series(
        frame[column],
        error=f"invalid_microstructure_numeric: {column}",
        allow_null=True,
    ).astype("float64")


def _strict_bool_series(series: pd.Series) -> pd.Series:
    if series.isna().any():
        raise ValueError("invalid_trade_flow_observed")
    if is_bool_dtype(series):
        return series.astype("bool")
    if not series.map(lambda value: isinstance(value, bool)).all():
        raise ValueError("invalid_trade_flow_observed")
    return pd.Series(series.to_numpy(dtype=bool), index=series.index)


def _rolling_group_sum(
    frame: pd.DataFrame,
    series: pd.Series,
    *,
    rolling_window: int,
    expected_interval: pd.Timedelta,
    min_periods: int,
    observed: pd.Series,
) -> pd.Series:
    segments = contiguous_segment_ids(
        frame,
        group_columns=("market_id", "asset_id"),
        timestamp_column="timestamp",
        expected_interval=expected_interval,
    )
    valid_mask = observed.eq(True)
    valid_series = series.where(valid_mask)
    valid_count = valid_mask.astype("int64")
    grouped = [frame["market_id"], frame["asset_id"], segments]
    rolling_sum = (
        valid_series.groupby(grouped, sort=False)
        .rolling(window=rolling_window, min_periods=min_periods)
        .sum()
        .reset_index(level=[0, 1, 2], drop=True)
        .sort_index()
    )
    rolling_observed = (
        valid_count.groupby(grouped, sort=False)
        .rolling(window=rolling_window, min_periods=min_periods)
        .sum()
        .reset_index(level=[0, 1, 2], drop=True)
        .sort_index()
    )
    return rolling_sum.where(rolling_observed.eq(rolling_window))
