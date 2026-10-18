"""Price-volume statistics on nominal sessions, complete paths and exact endpoints."""

from __future__ import annotations

from collections.abc import Iterable
from math import ceil

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from quant_research_platform.compute.dataframe.cross_section import (
    _centered_values,
    _pearson_correlations,
)
from quant_research_platform.contracts.time import (
    TimeWindow,
    time_window_timestamp_interval,
)
from quant_research_platform.contracts.validation import (
    strict_finite_numeric_series,
    strict_timestamp_series,
)

from .definitions import (
    _PRICE_VOLUME_INPUT_WINDOWS,
    PRICE_VOLUME_LIBRARY,
    PRICE_VOLUME_PARTIAL_STATISTICS,
)


def calculate_price_volume_factors(
    panel: pd.DataFrame,
    *,
    factor_columns: Iterable[str] | None = None,
    output_window: TimeWindow | None = None,
    history_keys: tuple[str, ...] = ("asset_id",),
    min_fresh_fraction: float = 0.75,
) -> pd.DataFrame:
    """Consume a complete own-market session grid, preserving lifecycle warmup.

    Native PANEL callers supply filling_flag. For generic numeric-only callers,
    supplied finite numeric observations are treated as fresh observations.
    """
    if (
        isinstance(min_fresh_fraction, (bool, np.bool_))
        or not np.isfinite(min_fresh_fraction)
        or not 0 < min_fresh_fraction <= 1
    ):
        raise ValueError("price_volume_min_fresh_fraction_invalid")
    requested = tuple(
        dict.fromkeys(
            PRICE_VOLUME_LIBRARY if factor_columns is None else factor_columns
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
    keys = tuple(dict.fromkeys(("asset_id", *history_keys, "timestamp")))
    missing = [name for name in (*keys, *names) if name not in panel]
    if missing:
        raise ValueError("price_volume_inputs_missing: " + ", ".join(missing))
    status = ("filling_flag",) if "filling_flag" in panel else ()
    ordered = panel.loc[:, (*keys, *names, *status)].copy()
    ordered["timestamp"] = strict_timestamp_series(
        ordered["timestamp"], error="price_volume_timestamp_invalid"
    )
    if ordered.duplicated(["asset_id", "timestamp"]).any():
        raise ValueError("price_volume_key_duplicate")
    ordered = ordered.sort_values([*history_keys, "timestamp"], kind="mergesort")
    output: dict[str, list[np.ndarray]] = {factor: [] for factor in requested}
    bounds = (
        None if output_window is None else time_window_timestamp_interval(output_window)
    )
    for _identity, group in ordered.groupby(
        list(history_keys), sort=False, dropna=False
    ):
        values = {
            name: strict_finite_numeric_series(
                group[name],
                error=f"price_volume_value_invalid: {name}",
                allow_null=True,
            ).to_numpy(dtype="float64", copy=False)
            for name in names
        }
        statistical_values = values
        if status:
            known = group["filling_flag"].isin((0, 1)).to_numpy(dtype=bool)
            statistical_values = {
                name: np.where(known, value, np.nan) for name, value in values.items()
            }
        fresh = (
            group["filling_flag"].eq(0).fillna(False).to_numpy(dtype=bool)
            if status
            else np.ones(len(group), dtype=bool)
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
        sums, statistical_sums = {}, {}
        for factor in requested:
            partial = factor in PRICE_VOLUME_PARTIAL_STATISTICS
            output[factor].append(
                _calculate_values(
                    statistical_values if partial else values,
                    statistical_sums if partial else sums,
                    factor,
                    start,
                    stop,
                    fresh=fresh,
                    min_fresh_fraction=min_fresh_fraction,
                )
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
        ordered[factor] = (
            np.concatenate(output[factor])
            if output[factor]
            else np.empty(0, dtype="float64")
        )
        ordered[factor] = ordered[factor].where(np.isfinite(ordered[factor]))
    return ordered


def _calculate_values(
    values: dict[str, np.ndarray],
    sums: dict,
    factor: str,
    start: int,
    stop: int,
    *,
    fresh: np.ndarray | None = None,
    min_fresh_fraction: float = 0.75,
) -> np.ndarray:
    names, windows = _PRICE_VOLUME_INPUT_WINDOWS[factor]
    if fresh is None:
        fresh = np.ones(len(values[names[0]]), dtype=bool)
    minimum = ceil(min_fresh_fraction * windows[0]) if windows else 0
    size = windows[0] if windows else 0
    source_name = (
        "turnover"
        if "turnover" in factor and not factor.startswith("return_")
        else names[0]
    )
    source = values[source_name]
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
        current[valid] = (numerator[valid] - denominator[valid]) / denominator[valid]
        if factor.startswith("reversal_"):
            current *= -1.0
        return output
    if factor == "share_price":
        return source[start:stop].copy()
    if factor in {"log_float_market_cap", "log_total_market_cap"}:
        return _log(source[start:stop])
    shift = int(factor in {"amount_shock_20d", "volume_shock_20d"})
    first = max(start, size - 1 + shift)
    if first >= stop:
        return output

    if factor.endswith("_corr_20d"):
        other = values[factor.removeprefix("return_").removesuffix("_corr_20d")]
        begin = first - size + 1
        left = sliding_window_view(source[begin:stop], size)
        right = sliding_window_view(other[begin:stop], size)
        paired = np.isfinite(left) & np.isfinite(right)
        counts = paired.sum(axis=1)
        fresh_pairs = paired & sliding_window_view(fresh[begin:stop], size)
        valid = (fresh_pairs.sum(axis=1) >= minimum) & (counts >= 2)
        left, right, paired = left[valid], right[valid], paired[valid]
        output[first - start :][valid] = _pearson_correlations(
            left, right, paired=paired
        )
        return output

    def window(column: np.ndarray) -> np.ndarray:
        return sliding_window_view(column, size)[
            first - size + 1 - shift : stop - size + 1 - shift
        ]

    extra_names = (
        ("low", "close")
        if factor == "amplitude_20d"
        else ("amount",)
        if factor.startswith("amihud_")
        or factor in {"signed_amount_flow_20d", "return_amount_imbalance_20d"}
        else ()
    )
    partial = factor in PRICE_VOLUME_PARTIAL_STATISTICS
    numeric = np.isfinite(source)
    if factor == "drawdown_60d":
        numeric &= source > 0
    for name in extra_names:
        numeric &= np.isfinite(values[name])
    if factor == "amplitude_20d":
        numeric &= values["close"] != 0
    if factor.startswith("amihud_"):
        numeric &= values["amount"] > 0

    def window_count(mask: np.ndarray, width: int = size) -> np.ndarray:
        prefix = np.empty(len(mask) + 1, dtype=np.int64)
        prefix[0] = 0
        np.cumsum(mask, dtype=np.int64, out=prefix[1:])
        return (
            prefix[first + 1 - shift : stop + 1 - shift]
            - prefix[first + 1 - shift - width : stop + 1 - shift - width]
        )

    counts = window_count(numeric)
    valid = counts == size
    if partial:
        valid = window_count(numeric & fresh) >= minimum
        math_minimum = (
            4
            if factor.startswith("kurtosis_")
            else 3
            if factor.startswith("skewness_")
            else 2
            if "volatility" in factor
            and not factor.startswith(("downside", "upside"))
            or "_cv_" in factor
            else 1
        )
        valid &= counts >= math_minimum
    if shift:
        valid &= np.isfinite(source[first:stop]) & fresh[first:stop]

    def rolling_mean(width: int) -> np.ndarray:
        key = ("mean", source_name, width)
        if key not in sums:
            means = pd.Series(source).rolling(width, min_periods=1).mean().to_numpy()
            if np.any(source < 0):
                magnitude = (
                    pd.Series(np.abs(source))
                    .rolling(width, min_periods=1)
                    .mean()
                    .to_numpy()
                )
                cancelled = (magnitude > 0) & (
                    np.abs(means) <= width * np.finfo(float).eps * magnitude
                )
                # Signed means within their summation error scale are numerical
                # zero, not valid logarithms or divisors for huge ratios.
                means = np.where(cancelled, 0.0, means)
            sums[key] = means
        return sums[key][first - shift : stop - shift]

    if shift or factor in {
        "log_avg_amount_20d",
        "avg_amount_20d",
        "avg_amount_126d",
        "turnover_250d",
        "turnover_20d",
        "turnover_60d",
    }:
        means = rolling_mean(size)
        if shift:
            valid &= means != 0
            with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
                result = (source[first:stop][valid] - means[valid]) / means[valid]
        elif factor == "log_avg_amount_20d":
            result = _log(means[valid])
        else:
            result = means[valid]
        output[first - start :][valid] = result
        return output
    if factor in {"amount_cv_126d", "turnover_cv_126d"}:
        current = window(source)[valid]
        center = rolling_mean(size)[valid]
        deviation = _sample_standard_deviation(current)
        result = np.divide(
            deviation, center, out=np.full(len(current), np.nan), where=center > 0
        )
        output[first - start :][valid] = result
        return output
    if factor == "abnormal_turnover_20_250":
        short = windows[1]
        short_fresh = window_count(numeric & fresh, short)
        valid &= short_fresh >= ceil(min_fresh_fraction * short)
        means = rolling_mean(size)
        valid &= means > 0
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            result = rolling_mean(short)[valid] / means[valid]
        output[first - start :][valid] = result
        return output
    matrix = window(np.where(numeric, source, np.nan)) if partial else window(source)
    if factor in {"skewness_20d", "skewness_60d", "kurtosis_60d"}:
        valid &= np.min(
            np.where(np.isfinite(matrix), matrix, np.inf), axis=1
        ) != np.max(np.where(np.isfinite(matrix), matrix, -np.inf), axis=1)
    current = matrix[valid]
    sample_counts = counts[valid]
    extra = {name: window(values[name])[valid] for name in extra_names}
    if factor.startswith("volatility_") or factor in {
        "amount_volatility_20d",
        "turnover_volatility_20d",
    }:
        result = _sample_standard_deviation(current)
    elif factor.startswith("downside_volatility_") or factor == "upside_volatility_20d":
        clipped = (
            np.minimum(current, 0.0)
            if factor.startswith("downside")
            else np.maximum(current, 0.0)
        )
        result = np.sqrt(np.nansum(clipped * clipped, axis=1) / sample_counts)
    elif factor == "max_return_20d":
        result = np.max(current, axis=1)
    elif factor == "top5_return_mean_20d":
        top = np.partition(current, -5, axis=1)[:, -5:]
        result = np.mean(top, axis=1)
    elif factor == "min_return_20d":
        result = np.min(current, axis=1)
    elif factor in {"skewness_20d", "skewness_60d", "kurtosis_60d"}:
        centered = _centered_values(current, np.isfinite(current))
        centered /= np.max(np.abs(centered), axis=1, keepdims=True)
        moment_2 = np.sum(centered * centered, axis=1) / sample_counts
        size = sample_counts
        if factor == "kurtosis_60d":
            moment_4 = np.sum(centered**4, axis=1) / sample_counts
            result = (
                (size - 1)
                / ((size - 2) * (size - 3))
                * ((size + 1) * (moment_4 / (moment_2 * moment_2) - 3.0) + 6.0)
            )
        else:
            moment_3 = np.sum(centered**3, axis=1) / sample_counts
            result = (
                np.sqrt(size * (size - 1))
                / (size - 2)
                * (moment_3 / (moment_2 * np.sqrt(moment_2)))
            )
    elif factor == "drawdown_60d":
        result = np.min(current / np.maximum.accumulate(current, axis=1) - 1.0, axis=1)
    elif factor == "amplitude_20d" or factor.startswith("amihud_"):
        numerator = (
            current - extra["low"] if factor == "amplitude_20d" else np.abs(current)
        )
        denominator = extra["close"] if factor == "amplitude_20d" else extra["amount"]
        result = np.nanmean(numerator / denominator, axis=1)
    elif factor == "trend_efficiency_60d":
        denominator = np.sum(np.abs(current), axis=1)
        numerator = np.sum(current, axis=1)
        result = np.divide(
            np.abs(numerator),
            denominator,
            out=np.full(len(current), np.nan),
            where=denominator != 0,
        )
    elif factor in {"signed_amount_flow_20d", "return_amount_imbalance_20d"}:
        flow = extra["amount"]
        if factor == "signed_amount_flow_20d":
            numerator = np.sum(np.sign(current) * flow, axis=1)
            denominator = np.sum(flow, axis=1)
        else:
            numerator = np.sum(current * flow, axis=1)
            denominator = np.sum(np.abs(current) * flow, axis=1)
        result = np.divide(
            numerator,
            denominator,
            out=np.full(len(current), np.nan),
            where=denominator != 0,
        )
    elif factor == "positive_return_fraction_20d":
        result = np.mean(current > 0.0, axis=1)
    elif factor in {"largest_daily_contribution_20d", "path_roughness_20d"}:
        denominator = np.sum(np.abs(current), axis=1)
        numerator = (
            np.max(np.abs(current), axis=1)
            if factor == "largest_daily_contribution_20d"
            else np.abs(np.sum(current, axis=1))
        )
        result = np.divide(
            numerator,
            denominator,
            out=np.full(len(current), np.nan),
            where=denominator > 0,
        )
        if factor == "path_roughness_20d":
            result = 1.0 - result
    elif factor == "zero_amount_ratio_20d":
        result = np.mean(current <= 0.0, axis=1)
    else:
        raise ValueError("price_volume_calculation_missing")
    output[first - start :][valid] = result
    return output


def _sample_standard_deviation(windows: np.ndarray) -> np.ndarray:
    finite = np.isfinite(windows)
    centered = _centered_values(windows, finite)
    return np.sqrt(np.sum(centered * centered, axis=1) / (finite.sum(axis=1) - 1))


def _log(values: np.ndarray) -> np.ndarray:
    result = np.full(len(values), np.nan)
    valid = np.isfinite(values) & (values > 0)
    result[valid] = np.log(values[valid])
    return result
