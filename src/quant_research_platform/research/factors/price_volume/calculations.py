"""Price-volume factor calculations."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

from .columns import price_volume_window_column
from .presets import price_volume_factor_columns
from .requirements import (
    price_volume_factor_input_columns,
    unsupported_price_volume_factors,
)
from .rolling import (
    rolling_aggregate,
    rolling_correlation,
    rolling_drawdown,
    rolling_kurt,
    rolling_rms,
    rolling_skew,
    rolling_std,
)
from .safety import finite_or_nan, safe_log, valid_denominator
from .windows import windows_from_requested_factors


def calculate_price_volume_factors(
    panel: pd.DataFrame,
    *,
    factor_columns: Iterable[str] | None = None,
    asset_column: str = "asset_id",
    date_column: str = "timestamp",
    price_column: str = "adj_close",
    return_column: str = "period_return",
    amount_column: str = "amount",
    volume_column: str = "volume",
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
    required_windows = windows_from_requested_factors(requested_set)
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
                "ret_amount_corr",
                "ret_volume_corr",
                "ret_turnover_corr",
                "trend_efficiency",
                "signed_amount_flow",
                "ret_amount_imbalance",
            )
            for window in required_windows
        )
    )
    needs_amount = any(
        price_volume_window_column(prefix, window) in requested_set
        for prefix in (
            "avg_amount",
            "amount_chg",
            "amount_shock",
            "amount_vol",
            "amihud",
            "zero_volume_ratio",
            "ret_amount_corr",
            "signed_amount_flow",
            "ret_amount_imbalance",
        )
        for window in required_windows
    )
    needs_volume = any(
        price_volume_window_column(prefix, window) in requested_set
        for prefix in ("volume_chg", "volume_shock", "ret_volume_corr")
        for window in required_windows
    )
    needs_turnover = any(
        price_volume_window_column(prefix, window) in requested_set
        for prefix in (
            "turnover",
            "turnover_chg",
            "turnover_vol",
            "ret_turnover_corr",
        )
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
    volume = (
        pd.to_numeric(result[volume_column], errors="coerce")
        if needs_volume
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
    volume_grouped = (
        volume.groupby(group_keys, sort=False) if volume is not None else None
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
            denominator = valid_denominator(shifted_price)
            trailing_return = finite_or_nan(price / denominator - 1.0)
        if needs_momentum:
            result[momentum_column] = trailing_return
        if needs_reversal:
            result[reversal_column] = -trailing_return

    for window in required_windows:
        column = price_volume_window_column("vol", window)
        if column in requested_set:
            result[column] = rolling_std(
                returns,
                group_keys,
                group_positions,
                window,
            )
        column = price_volume_window_column("downside_vol", window)
        if column in requested_set:
            result[column] = rolling_rms(
                returns.clip(upper=0.0),
                group_keys,
                group_positions,
                window,
            )
        column = price_volume_window_column("upside_vol", window)
        if column in requested_set:
            result[column] = rolling_rms(
                returns.clip(lower=0.0),
                group_keys,
                group_positions,
                window,
            )
        column = price_volume_window_column("max_return", window)
        if column in requested_set:
            result[column] = rolling_aggregate(
                result[return_column],
                grouped[return_column],
                group_positions,
                window,
                "max",
            )
        column = price_volume_window_column("min_return", window)
        if column in requested_set:
            result[column] = rolling_aggregate(
                result[return_column],
                grouped[return_column],
                group_positions,
                window,
                "min",
            )
        column = price_volume_window_column("drawdown", window)
        if column in requested_set:
            result[column] = rolling_drawdown(
                price,
                group_keys,
                group_positions,
                window,
            )
        column = price_volume_window_column("skew", window)
        if column in requested_set:
            result[column] = rolling_skew(
                returns,
                group_keys,
                group_positions,
                window,
            )
        column = price_volume_window_column("kurt", window)
        if column in requested_set:
            result[column] = rolling_kurt(
                returns,
                group_keys,
                group_positions,
                window,
            )

    for window in required_windows:
        column = price_volume_window_column("avg_amount", window)
        if column in requested_set:
            result[column] = rolling_aggregate(
                amount,
                amount_grouped,
                group_positions,
                window,
                "mean",
            )
        column = price_volume_window_column("turnover", window)
        if column in requested_set:
            result[column] = rolling_aggregate(
                turnover,
                turnover_grouped,
                group_positions,
                window,
                "mean",
            )
        column = price_volume_window_column("turnover_chg", window)
        if column in requested_set:
            shifted_turnover = turnover_grouped.shift(window)
            denominator = valid_denominator(shifted_turnover)
            result[column] = turnover / denominator - 1.0
        column = price_volume_window_column("turnover_vol", window)
        if column in requested_set:
            result[column] = rolling_std(
                turnover,
                group_keys,
                group_positions,
                window,
            )
        column = price_volume_window_column("zero_volume_ratio", window)
        if column in requested_set:
            zero_volume = (amount <= 0.0).astype(float)
            result[column] = rolling_aggregate(
                zero_volume,
                zero_volume.groupby(group_keys, sort=False),
                group_positions,
                window,
                "mean",
            )
        column = price_volume_window_column("amount_chg", window)
        if column in requested_set:
            shifted_amount = amount_grouped.shift(window)
            denominator = valid_denominator(shifted_amount)
            result[column] = amount / denominator - 1.0
        column = price_volume_window_column("amount_shock", window)
        if column in requested_set:
            mean_amount = rolling_aggregate(
                amount,
                amount_grouped,
                group_positions,
                window,
                "mean",
            )
            result[column] = amount / valid_denominator(mean_amount) - 1.0
        column = price_volume_window_column("amount_vol", window)
        if column in requested_set:
            result[column] = rolling_std(
                amount,
                group_keys,
                group_positions,
                window,
            )
        column = price_volume_window_column("volume_chg", window)
        if column in requested_set:
            shifted_volume = volume_grouped.shift(window)
            denominator = valid_denominator(shifted_volume)
            result[column] = volume / denominator - 1.0
        column = price_volume_window_column("volume_shock", window)
        if column in requested_set:
            mean_volume = rolling_aggregate(
                volume,
                volume_grouped,
                group_positions,
                window,
                "mean",
            )
            result[column] = volume / valid_denominator(mean_volume) - 1.0
        column = price_volume_window_column("ret_amount_corr", window)
        if column in requested_set:
            result[column] = rolling_correlation(
                returns,
                amount,
                group_positions,
                window,
            )
        column = price_volume_window_column("ret_volume_corr", window)
        if column in requested_set:
            result[column] = rolling_correlation(
                returns,
                volume,
                group_positions,
                window,
            )
        column = price_volume_window_column("ret_turnover_corr", window)
        if column in requested_set:
            result[column] = rolling_correlation(
                returns,
                turnover,
                group_positions,
                window,
            )
        column = price_volume_window_column("trend_efficiency", window)
        if column in requested_set:
            return_mean = rolling_aggregate(
                returns,
                returns.groupby(group_keys, sort=False),
                group_positions,
                window,
                "mean",
            )
            abs_return_mean = rolling_aggregate(
                returns.abs(),
                returns.abs().groupby(group_keys, sort=False),
                group_positions,
                window,
                "mean",
            )
            result[column] = return_mean.abs() / valid_denominator(abs_return_mean)
        column = price_volume_window_column("signed_amount_flow", window)
        if column in requested_set:
            signed_amount = np.sign(returns) * amount
            numerator = rolling_aggregate(
                signed_amount,
                signed_amount.groupby(group_keys, sort=False),
                group_positions,
                window,
                "mean",
            )
            denominator = rolling_aggregate(
                amount,
                amount_grouped,
                group_positions,
                window,
                "mean",
            )
            result[column] = numerator / valid_denominator(denominator)
        column = price_volume_window_column("ret_amount_imbalance", window)
        if column in requested_set:
            weighted_return = returns * amount
            weighted_abs_return = returns.abs() * amount
            numerator = rolling_aggregate(
                weighted_return,
                weighted_return.groupby(group_keys, sort=False),
                group_positions,
                window,
                "mean",
            )
            denominator = rolling_aggregate(
                weighted_abs_return,
                weighted_abs_return.groupby(group_keys, sort=False),
                group_positions,
                window,
                "mean",
            )
            result[column] = numerator / valid_denominator(denominator)

    illiquidity = None
    if any(
        price_volume_window_column("amihud", window) in requested_set
        for window in required_windows
    ):
        denominator = valid_denominator(amount)
        illiquidity = (returns.abs() / denominator).replace([np.inf, -np.inf], np.nan)
    amplitude = None
    if any(
        price_volume_window_column("amplitude", window) in requested_set
        for window in required_windows
    ):
        amplitude = (
            pd.to_numeric(result[high_column], errors="coerce")
            - pd.to_numeric(result[low_column], errors="coerce")
        ) / valid_denominator(
            pd.to_numeric(result[prev_close_column], errors="coerce")
        )
        amplitude = finite_or_nan(amplitude)
    for window in required_windows:
        column = price_volume_window_column("amihud", window)
        if column in requested_set:
            result[column] = rolling_aggregate(
                illiquidity,
                illiquidity.groupby(group_keys, sort=False),
                group_positions,
                window,
                "mean",
            )
        column = price_volume_window_column("amplitude", window)
        if column in requested_set:
            result[column] = rolling_aggregate(
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
        ) / valid_denominator(price)
        intraday_range = finite_or_nan(intraday_range)
    for window in required_windows:
        column = price_volume_window_column("intraday_range", window)
        if column in requested_set:
            result[column] = rolling_aggregate(
                intraday_range,
                intraday_range.groupby(group_keys, sort=False),
                group_positions,
                window,
                "mean",
            )

    if "log_float_market_cap" in requested_set:
        result["log_float_market_cap"] = safe_log(result[float_market_cap_column])
    if "log_total_market_cap" in requested_set:
        result["log_total_market_cap"] = safe_log(result[total_market_cap_column])
    output_columns = [column for column in requested if column in result.columns]
    for column in output_columns:
        result[column] = finite_or_nan(result[column])
    return result


__all__ = ["calculate_price_volume_factors"]
