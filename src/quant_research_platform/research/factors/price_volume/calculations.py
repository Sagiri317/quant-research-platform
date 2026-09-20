"""Price-volume calculations using complete session windows and exact endpoints."""

from __future__ import annotations

from collections.abc import Iterable
from math import hypot, sqrt
from statistics import stdev

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from quant_research_platform.compute.dataframe.cross_section import (
    _pearson_correlation_result,
)
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
    output: dict[str, list[np.ndarray]] = {factor: [] for factor in requested}
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
            output[factor].append(_calculate_values(values, factor, start, stop))
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
        ordered[factor] = (
            np.concatenate(output[factor])
            if output[factor]
            else np.empty(0, dtype="float64")
        )
    return ordered


def _calculate_values(
    values: dict[str, np.ndarray], factor: str, start: int, stop: int
) -> np.ndarray:
    names, windows = _PRICE_VOLUME_INPUT_WINDOWS[factor]
    size = windows[0] if windows else 0
    source = values["turnover" if factor.startswith("turnover_") else names[0]]
    output = np.full(stop - start, np.nan)
    if factor.startswith(("reversal_", "momentum_")) or factor.endswith("_change_20d"):
        first = max(start, size)
        if first >= stop:
            return output
        skip = windows[1] if len(windows) > 1 else 0
        numerator = source[first - skip : stop - skip]
        denominator = source[first - size : stop - size]
        valid = np.isfinite(numerator) & np.isfinite(denominator) & (denominator != 0)
        current = output[first - start :]
        current[valid] = numerator[valid] / denominator[valid] - 1.0
        if factor.startswith("reversal_"):
            current *= -1.0
        return output
    if factor in {"log_float_market_cap", "log_total_market_cap"}:
        return _log(source[start:stop])
    shift = int(factor in {"amount_shock_20d", "volume_shock_20d"})
    first = max(start, size - 1 + shift)
    if first >= stop:
        return output

    def window(column: np.ndarray) -> np.ndarray:
        return sliding_window_view(column, size)[
            first - size + 1 - shift : stop - size + 1 - shift
        ]

    matrix = window(source)
    valid = np.isfinite(matrix).all(axis=1)
    extra_names = (
        ("low", "close")
        if factor == "amplitude_20d"
        else ("amount",)
        if factor.startswith("amihud_")
        or factor in {"signed_amount_flow_20d", "return_amount_imbalance_20d"}
        else (factor.removeprefix("return_").removesuffix("_corr_20d"),)
        if factor.endswith("_corr_20d")
        else ()
    )
    extra = {name: window(values[name]) for name in extra_names}
    for matrix_extra in extra.values():
        valid &= np.isfinite(matrix_extra).all(axis=1)
    if factor == "drawdown_60d":
        valid &= (matrix > 0).all(axis=1)
    if factor == "amplitude_20d":
        valid &= (extra["close"] != 0).all(axis=1)
    if factor.startswith("amihud_"):
        valid &= (extra["amount"] > 0).all(axis=1)
    if factor.endswith("_corr_20d"):
        other = extra[extra_names[0]]
        valid &= (np.min(matrix, axis=1) != np.max(matrix, axis=1)) & (
            np.min(other, axis=1) != np.max(other, axis=1)
        )
    if factor in {"skewness_60d", "kurtosis_60d"}:
        valid &= np.min(matrix, axis=1) != np.max(matrix, axis=1)
    if shift:
        valid &= np.isfinite(source[first:stop])
    current = matrix[valid]
    extra = {name: data[valid] for name, data in extra.items()}
    if factor.startswith("volatility_") or factor in {
        "amount_volatility_20d",
        "turnover_volatility_20d",
    }:
        result = np.fromiter((stdev(row) for row in current), dtype=float)
    elif factor.startswith("downside_volatility_") or factor == "upside_volatility_20d":
        clipped = (
            np.minimum(current, 0.0)
            if factor.startswith("downside")
            else np.maximum(current, 0.0)
        )
        result = np.fromiter(
            (
                scale * (hypot(*(row / scale)) / sqrt(size))
                if (scale := float(np.max(np.abs(row))))
                else 0.0
                for row in clipped
            ),
            dtype=float,
        )
    elif factor == "max_return_20d":
        result = np.max(current, axis=1)
    elif factor == "min_return_20d":
        result = np.min(current, axis=1)
    elif factor == "skewness_60d":
        # Batched pandas skew has measured ULP differences from the scalar kernel.
        result = np.fromiter((pd.Series(row).skew() for row in current), dtype=float)
    elif factor == "kurtosis_60d":
        normalized = current / np.max(np.abs(current), axis=1, keepdims=True)
        centered = normalized - np.mean(normalized, axis=1, keepdims=True)
        centered /= np.max(np.abs(centered), axis=1, keepdims=True)
        moment_2 = np.mean(centered * centered, axis=1)
        moment_4 = np.mean(centered**4, axis=1)
        result = (
            (size - 1)
            / ((size - 2) * (size - 3))
            * ((size + 1) * (moment_4 / (moment_2 * moment_2) - 3.0) + 6.0)
        )
    elif factor == "drawdown_60d":
        result = np.min(current / np.maximum.accumulate(current, axis=1) - 1.0, axis=1)
    elif factor == "amplitude_20d":
        result = np.mean((current - extra["low"]) / extra["close"], axis=1)
    elif factor in {
        "log_avg_amount_20d",
        "avg_amount_20d",
        "turnover_20d",
        "turnover_60d",
    }:
        result = np.mean(current, axis=1)
        if factor == "log_avg_amount_20d":
            result = _log(result)
    elif factor.startswith("amihud_"):
        result = np.mean(np.abs(current) / extra["amount"], axis=1)
    elif shift:
        mean = np.mean(current, axis=1)
        result = (
            np.divide(
                source[first:stop][valid],
                mean,
                out=np.full(len(current), np.nan),
                where=mean != 0,
            )
            - 1.0
        )
    elif factor.endswith("_corr_20d"):
        result = np.fromiter(
            (
                _pearson_correlation_result(a, b)[0]
                for a, b in zip(current, extra[extra_names[0]], strict=True)
            ),
            dtype=float,
        )
    elif factor == "trend_efficiency_60d":
        denominator = np.sum(np.abs(current), axis=1)
        result = np.divide(
            np.abs(np.sum(current, axis=1)),
            denominator,
            out=np.full(len(current), np.nan),
            where=denominator != 0,
        )
    elif factor in {"signed_amount_flow_20d", "return_amount_imbalance_20d"}:
        flow = extra["amount"]
        numerator = np.sum(
            (np.sign(current) if factor == "signed_amount_flow_20d" else current)
            * flow,
            axis=1,
        )
        denominator = np.sum(
            flow if factor == "signed_amount_flow_20d" else np.abs(current) * flow,
            axis=1,
        )
        result = np.divide(
            numerator,
            denominator,
            out=np.full(len(current), np.nan),
            where=denominator != 0,
        )
    elif factor == "zero_amount_ratio_20d":
        result = np.mean(current <= 0.0, axis=1)
    else:
        raise ValueError("price_volume_calculation_missing")
    output[first - start :][valid] = result
    return output


def _log(values: np.ndarray) -> np.ndarray:
    result = np.full(len(values), np.nan)
    valid = np.isfinite(values) & (values > 0)
    result[valid] = np.log(values[valid])
    return result
