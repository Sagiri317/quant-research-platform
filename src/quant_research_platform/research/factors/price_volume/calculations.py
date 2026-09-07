"""Price-volume calculations using complete session windows and exact endpoints."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

from quant_research_platform.contracts.time import (
    TimeWindow,
    time_window_timestamp_interval,
)
from quant_research_platform.contracts.validation import (
    strict_finite_numeric_series,
    strict_timestamp_series,
)

from .definitions import _PRICE_VOLUME_INPUT_WINDOWS, FORMAL_PRICE_VOLUME_FACTORS


def calculate_price_volume_factors(
    panel: pd.DataFrame,
    *,
    factor_columns: Iterable[str] | None = None,
    output_window: TimeWindow | None = None,
) -> pd.DataFrame:
    """Calculate only requested columns and output rows, preserving warmup inputs."""
    requested = tuple(
        dict.fromkeys(
            FORMAL_PRICE_VOLUME_FACTORS if factor_columns is None else factor_columns
        )
    )
    unknown = sorted(set(requested).difference(_PRICE_VOLUME_INPUT_WINDOWS))
    if unknown:
        raise ValueError("unknown_price_volume_factor: " + ", ".join(unknown))
    names = tuple(
        dict.fromkeys(
            name
            for factor in requested
            for name in _PRICE_VOLUME_INPUT_WINDOWS[factor][0]
        )
    )
    missing = [name for name in ("asset_id", "timestamp", *names) if name not in panel]
    if missing:
        raise ValueError("price_volume_inputs_missing: " + ", ".join(missing))
    ordered = panel.loc[:, ("asset_id", "timestamp", *names)].copy()
    ordered["timestamp"] = strict_timestamp_series(
        ordered["timestamp"], error="price_volume_timestamp_invalid"
    )
    if ordered.duplicated(["asset_id", "timestamp"]).any():
        raise ValueError("price_volume_key_duplicate")
    ordered = ordered.sort_values(["asset_id", "timestamp"], kind="mergesort")
    output: dict[str, list[float]] = {factor: [] for factor in requested}
    bounds = (
        None if output_window is None else time_window_timestamp_interval(output_window)
    )
    for _asset_id, group in ordered.groupby("asset_id", sort=False):
        values = {
            name: strict_finite_numeric_series(
                group[name],
                error=f"price_volume_value_invalid: {name}",
                allow_null=True,
            ).to_numpy(dtype="float64", copy=False)
            for name in names
        }
        if any("turnover" in factor for factor in requested):
            float_cap, close, volume = (
                values["float_market_cap"],
                values["close"],
                values["volume"],
            )
            free_shares = np.divide(
                float_cap,
                close,
                out=np.full(len(group), np.nan),
                where=np.isfinite(float_cap) & np.isfinite(close) & (close != 0.0),
            )
            values["turnover"] = np.divide(
                volume,
                free_shares,
                out=np.full(len(group), np.nan),
                where=np.isfinite(volume)
                & np.isfinite(free_shares)
                & (free_shares > 0.0),
            )
        start, stop = 0, len(group)
        if bounds is not None:
            lower, upper, closed = bounds
            start = int(group["timestamp"].searchsorted(lower, side="left"))
            stop = int(
                group["timestamp"].searchsorted(
                    upper, side="right" if closed else "left"
                )
            )
        for factor in requested:
            windows = _PRICE_VOLUME_INPUT_WINDOWS[factor][1]
            size = windows[0] if windows else 0
            output[factor].extend(
                _calculate_value(values, factor, index, size)
                for index in range(start, stop)
            )
    ordered = ordered.loc[:, ("asset_id", "timestamp")]
    if bounds is not None:
        lower, upper, closed = bounds
        timestamps = ordered["timestamp"]
        ordered = ordered.loc[
            timestamps.ge(lower)
            & (timestamps.le(upper) if closed else timestamps.lt(upper))
        ]
    ordered = ordered.reset_index(drop=True)
    for factor in requested:
        ordered[factor] = np.asarray(output[factor], dtype="float64")
    return ordered


def _calculate_value(
    values: dict[str, np.ndarray], factor: str, index: int, size: int
) -> float:
    source = values[_PRICE_VOLUME_INPUT_WINDOWS[factor][0][0]]
    if factor.startswith("reversal_") or factor in {"momentum_20d", "momentum_60d"}:
        return _lagged_momentum(source, index, size, 0) * (
            -1.0 if factor.startswith("reversal_") else 1.0
        )
    if factor.startswith("momentum_"):
        return _lagged_momentum(
            source, index, size, _PRICE_VOLUME_INPUT_WINDOWS[factor][1][1]
        )
    if factor.startswith("volatility_") or factor == "amount_volatility_20d":
        return _window_stat(source, index, size, "std")
    if factor.startswith("downside_volatility_"):
        return _window_rms(source, index, size)
    if factor == "upside_volatility_20d":
        current = _window(source, index, size)
        return (
            float("nan")
            if current is None
            else float(np.sqrt(np.mean(np.maximum(current, 0.0) ** 2)))
        )
    if factor in {"max_return_20d", "min_return_20d", "skewness_60d", "kurtosis_60d"}:
        statistic = {
            "max_return_20d": "max",
            "min_return_20d": "min",
            "skewness_60d": "skew",
            "kurtosis_60d": "kurt",
        }[factor]
        return _window_stat(source, index, size, statistic)
    if factor == "drawdown_60d":
        return _window_drawdown(source, index, size)
    if factor == "amplitude_20d":
        return _window_ratio_mean(
            values["high"], values["low"], values["close"], index, size
        )
    if factor in {"log_avg_amount_20d", "avg_amount_20d"}:
        mean = _window_stat(source, index, size, "mean")
        return _log(mean) if factor == "log_avg_amount_20d" else mean
    if factor in {"turnover_20d", "turnover_60d", "turnover_volatility_20d"}:
        return _window_stat(
            values["turnover"],
            index,
            size,
            "std" if factor == "turnover_volatility_20d" else "mean",
        )
    if factor.endswith("_change_20d"):
        current = values["turnover"] if factor == "turnover_change_20d" else source
        return _lagged_momentum(current, index, size, 0)
    if factor.startswith("amihud_"):
        return _window_amihud(source, values["amount"], index, size)
    if factor in {"amount_shock_20d", "volume_shock_20d"}:
        return _amount_shock(source, index, size)
    if factor.endswith("_corr_20d"):
        right = values[factor.removeprefix("return_").removesuffix("_corr_20d")]
        return _window_corr(source, right, index, size)
    if factor == "trend_efficiency_60d":
        return _trend_efficiency(source, index, size)
    if factor == "signed_amount_flow_20d":
        return _signed_amount_flow(source, values["amount"], index, size)
    if factor == "return_amount_imbalance_20d":
        ret, flow = _window(source, index, size), _window(values["amount"], index, size)
        if ret is None or flow is None:
            return float("nan")
        denominator = float(np.sum(np.abs(ret) * flow))
        return (
            float(np.sum(ret * flow) / denominator)
            if denominator != 0.0
            else float("nan")
        )
    if factor == "zero_amount_ratio_20d":
        current = _window(source, index, size)
        return float("nan") if current is None else float(np.mean(current <= 0.0))
    if factor in {"log_float_market_cap", "log_total_market_cap"}:
        return _log(source[index])
    raise ValueError("price_volume_calculation_missing")


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
    if statistic == "min":
        return float(np.min(current))
    if statistic == "kurt":
        if size < 4 or np.min(current) == np.max(current):
            return float("nan")
        normalized = current / np.max(np.abs(current))
        centered = normalized - np.mean(normalized)
        centered /= np.max(np.abs(centered))
        moment_2 = np.mean(centered * centered)
        moment_4 = np.mean(centered**4)
        return float(
            (size - 1)
            / ((size - 2) * (size - 3))
            * ((size + 1) * (moment_4 / (moment_2 * moment_2) - 3.0) + 6.0)
        )
    if statistic == "mean":
        return float(np.mean(current))
    if statistic == "skew":
        return float(pd.Series(current).skew()) if size >= 3 else float("nan")
    raise ValueError("price_volume_statistic_invalid")


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
    return _return(numerator, denominator)


def _return(numerator: float, denominator: float) -> float:
    if not np.isfinite(numerator) or not np.isfinite(denominator) or denominator == 0.0:
        return float("nan")
    value = numerator / denominator - 1.0
    return float(value)


def _log(value: float) -> float:
    return float(np.log(value)) if np.isfinite(value) and value > 0.0 else float("nan")
