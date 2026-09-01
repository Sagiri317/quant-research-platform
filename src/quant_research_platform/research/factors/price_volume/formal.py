"""Frozen A-share PV25 factor math."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

from quant_research_platform.contracts.validation import (
    strict_finite_numeric_series,
    strict_timestamp_series,
)

from .columns import FORMAL_PRICE_VOLUME_FACTORS


def calculate_formal_price_volume_factors(
    panel: pd.DataFrame,
    *,
    factor_columns: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Calculate PV25 using complete market-session windows only."""

    requested = tuple(dict.fromkeys(factor_columns or FORMAL_PRICE_VOLUME_FACTORS))
    unknown = sorted(set(requested).difference(FORMAL_PRICE_VOLUME_FACTORS))
    if unknown:
        raise ValueError("unknown_formal_price_volume_factor: " + ", ".join(unknown))
    required = {"asset_id", "timestamp"}
    if not required.issubset(panel.columns):
        raise ValueError("formal_price_volume_keys_invalid")
    result = panel.loc[:, ("asset_id", "timestamp")].copy()
    result["timestamp"] = strict_timestamp_series(
        result["timestamp"], error="formal_price_volume_timestamp_invalid"
    )
    if result.duplicated(["asset_id", "timestamp"]).any():
        raise ValueError("formal_price_volume_key_duplicate")
    output: dict[str, list[float]] = {factor: [] for factor in requested}
    for _asset_id, group in (
        panel.assign(timestamp=result["timestamp"])
        .sort_values(["asset_id", "timestamp"], kind="mergesort")
        .groupby("asset_id", sort=False)
    ):
        values = _group_values(group)
        calculated = _calculate_group(values)
        for factor in requested:
            output[factor].extend(calculated[factor])
    ordered = (
        panel.assign(timestamp=result["timestamp"])
        .sort_values(["asset_id", "timestamp"], kind="mergesort")
        .loc[:, ("asset_id", "timestamp")]
        .reset_index(drop=True)
    )
    for factor in requested:
        ordered[factor] = output[factor]
    return ordered


def _group_values(group: pd.DataFrame) -> dict[str, np.ndarray]:
    names = (
        "adj_close",
        "period_return",
        "high",
        "low",
        "close",
        "amount",
        "volume",
        "float_market_cap",
    )
    missing = [name for name in names if name not in group]
    if missing:
        raise ValueError("formal_price_volume_inputs_missing: " + ", ".join(missing))
    return {
        name: strict_finite_numeric_series(
            group[name],
            error=f"formal_price_volume_value_invalid: {name}",
            allow_null=True,
        ).to_numpy(dtype="float64", copy=False)
        for name in names
    }


def _calculate_group(values: dict[str, np.ndarray]) -> dict[str, list[float]]:
    n = len(values["adj_close"])
    price = values["adj_close"]
    returns = values["period_return"]
    high = values["high"]
    low = values["low"]
    close = values["close"]
    amount = values["amount"]
    volume = values["volume"]
    float_cap = values["float_market_cap"]
    free_shares = np.divide(
        float_cap,
        close,
        out=np.full(n, np.nan),
        where=np.isfinite(float_cap) & np.isfinite(close) & (close != 0.0),
    )
    turnover = np.divide(
        volume,
        free_shares,
        out=np.full(n, np.nan),
        where=np.isfinite(volume) & np.isfinite(free_shares) & (free_shares > 0.0),
    )
    result = {
        factor: np.full(n, np.nan, dtype="float64")
        for factor in FORMAL_PRICE_VOLUME_FACTORS
    }
    for index in range(n):
        if index >= 1:
            result["reversal_1d"][index] = _return(
                price[index], price[index - 1], negative=True
            )
        if index >= 5:
            result["reversal_5d"][index] = _return(
                price[index], price[index - 5], negative=True
            )
        if index >= 20:
            result["reversal_20d"][index] = _return(
                price[index], price[index - 20], negative=True
            )
        result["momentum_60_20"][index] = _lagged_momentum(price, index, 60, 20)
        result["momentum_120_20"][index] = _lagged_momentum(price, index, 120, 20)
        result["momentum_252_20"][index] = _lagged_momentum(price, index, 252, 20)
        result["volatility_20d"][index] = _window_stat(returns, index, 20, "std")
        result["volatility_60d"][index] = _window_stat(returns, index, 60, "std")
        result["downside_volatility_20d"][index] = _window_rms(returns, index, 20)
        result["downside_volatility_60d"][index] = _window_rms(returns, index, 60)
        result["max_return_20d"][index] = _window_stat(returns, index, 20, "max")
        result["skewness_60d"][index] = _window_stat(returns, index, 60, "skew")
        result["drawdown_60d"][index] = _window_drawdown(price, index, 60)
        result["amplitude_20d"][index] = _window_ratio_mean(high, low, close, index, 20)
        result["log_avg_amount_20d"][index] = _log(
            _window_stat(amount, index, 20, "mean")
        )
        result["turnover_20d"][index] = _window_stat(turnover, index, 20, "mean")
        result["turnover_60d"][index] = _window_stat(turnover, index, 60, "mean")
        result["turnover_volatility_20d"][index] = _window_stat(
            turnover, index, 20, "std"
        )
        result["amihud_20d"][index] = _window_amihud(returns, amount, index, 20)
        result["amihud_60d"][index] = _window_amihud(returns, amount, index, 60)
        result["amount_shock_20d"][index] = _amount_shock(amount, index, 20)
        result["return_amount_corr_20d"][index] = _window_corr(
            returns, amount, index, 20
        )
        result["trend_efficiency_60d"][index] = _trend_efficiency(returns, index, 60)
        result["signed_amount_flow_20d"][index] = _signed_amount_flow(
            returns, amount, index, 20
        )
        result["log_float_market_cap"][index] = _log(float_cap[index])
    return {factor: values.tolist() for factor, values in result.items()}


def _window(values: np.ndarray, index: int, size: int) -> np.ndarray | None:
    start = index - size + 1
    if start < 0:
        return None
    current = values[start : index + 1]
    return current if np.isfinite(current).all() else None


def _window_stat(values: np.ndarray, index: int, size: int, statistic: str) -> float:
    current = _window(values, index, size)
    if current is None:
        return float("nan")
    if statistic == "std":
        return float(np.std(current, ddof=1)) if size >= 2 else float("nan")
    if statistic == "max":
        return float(np.max(current))
    if statistic == "mean":
        return float(np.mean(current))
    if statistic == "skew":
        return float(pd.Series(current).skew()) if size >= 3 else float("nan")
    raise ValueError("formal_price_volume_statistic_invalid")


def _window_rms(values: np.ndarray, index: int, size: int) -> float:
    current = _window(values, index, size)
    if current is None:
        return float("nan")
    return float(np.sqrt(np.mean(np.minimum(current, 0.0) ** 2)))


def _window_drawdown(values: np.ndarray, index: int, size: int) -> float:
    current = _window(values, index, size)
    if current is None or (current <= 0.0).any():
        return float("nan")
    drawdowns = current / np.maximum.accumulate(current) - 1.0
    return float(np.min(drawdowns))


def _window_ratio_mean(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    index: int,
    size: int,
) -> float:
    values = _window(high, index, size)
    lows = _window(low, index, size)
    closes = _window(close, index, size)
    if values is None or lows is None or closes is None or (closes == 0.0).any():
        return float("nan")
    return float(np.mean((values - lows) / closes))


def _window_amihud(
    returns: np.ndarray, amount: np.ndarray, index: int, size: int
) -> float:
    ret = _window(returns, index, size)
    flow = _window(amount, index, size)
    if ret is None or flow is None or (flow <= 0.0).any():
        return float("nan")
    return float(np.mean(np.abs(ret) / flow))


def _amount_shock(amount: np.ndarray, index: int, size: int) -> float:
    if index < size:
        return float("nan")
    current = amount[index]
    baseline = amount[index - size : index]
    if not np.isfinite(current) or not np.isfinite(baseline).all():
        return float("nan")
    mean = float(np.mean(baseline))
    return float(current / mean - 1.0) if mean != 0.0 else float("nan")


def _window_corr(left: np.ndarray, right: np.ndarray, index: int, size: int) -> float:
    left_window = _window(left, index, size)
    right_window = _window(right, index, size)
    if left_window is None or right_window is None:
        return float("nan")
    if np.ptp(left_window) == 0.0 or np.ptp(right_window) == 0.0:
        return float("nan")
    return float(np.corrcoef(left_window, right_window)[0, 1])


def _trend_efficiency(values: np.ndarray, index: int, size: int) -> float:
    current = _window(values, index, size)
    if current is None:
        return float("nan")
    denominator = float(np.sum(np.abs(current)))
    return (
        float(abs(np.sum(current)) / denominator)
        if denominator != 0.0
        else float("nan")
    )


def _signed_amount_flow(
    returns: np.ndarray, amount: np.ndarray, index: int, size: int
) -> float:
    ret = _window(returns, index, size)
    flow = _window(amount, index, size)
    if ret is None or flow is None:
        return float("nan")
    denominator = float(np.sum(flow))
    return (
        float(np.sum(np.sign(ret) * flow) / denominator)
        if denominator != 0.0
        else float("nan")
    )


def _lagged_momentum(
    price: np.ndarray, index: int, long_window: int, skip: int
) -> float:
    if index < long_window:
        return float("nan")
    numerator = price[index - skip]
    denominator = price[index - long_window]
    return _return(numerator, denominator, negative=False)


def _return(numerator: float, denominator: float, *, negative: bool) -> float:
    if not np.isfinite(numerator) or not np.isfinite(denominator) or denominator == 0.0:
        return float("nan")
    value = numerator / denominator - 1.0
    return float(-value if negative else value)


def _log(value: float) -> float:
    return float(np.log(value)) if np.isfinite(value) and value > 0.0 else float("nan")
