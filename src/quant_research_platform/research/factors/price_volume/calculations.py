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
    PRICE_VOLUME_LEVEL_DESCRIPTIONS,
    PRICE_VOLUME_LIBRARY,
    PRICE_VOLUME_PARTIAL_STATISTICS,
    PRICE_VOLUME_PROFILE_DESCRIPTIONS,
    PRICE_VOLUME_THEME_DESCRIPTIONS,
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
    if factor in PRICE_VOLUME_PROFILE_DESCRIPTIONS:
        if "amount_profile_60d" not in sums:
            sums["amount_profile_60d"] = _calculate_amount_profile(values)
        return sums["amount_profile_60d"][factor][start:stop].copy()
    if factor in PRICE_VOLUME_LEVEL_DESCRIPTIONS:
        return _calculate_level_values(values, factor, start, stop)
    if factor in PRICE_VOLUME_THEME_DESCRIPTIONS:
        return _calculate_theme_values(values, factor, start, stop)
    if factor.startswith(("alpha101_", "alpha191_", "qlib_")):
        return _calculate_published_values(values, factor, start, stop)
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


def _calculate_theme_values(values, factor, start, stop):
    """Compute the declared candidates on complete causal own-market windows."""
    names, windows = _PRICE_VOLUME_INPUT_WINDOWS[factor]
    size = windows[0]
    output = np.full(stop - start, np.nan)
    first = max(start, size - 1)
    if first >= stop:
        return output
    widths = {name: (size, 0) for name in names}
    position_interaction = factor.startswith(
        ("low_position_turnover_", "high_position_turnover_")
    )
    if position_interaction:
        widths = {
            "adj_close": (size - 1, 1),
            "turnover": (21, 0),
            "period_return": (21, 0),
        }
    elif "adj_open" in names:
        widths["adj_open"] = (1, 0)
        if factor == "opening_gap":
            widths["adj_close"] = (1, 1)
        elif factor != "intraday_return":
            widths["adj_close"] = (2, 0)
    elif factor == "price_impact_efficiency_20d":
        widths["turnover"] = (20, 0)
    matrices = {
        name: sliding_window_view(
            values[name][first - width + 1 - lag : stop - lag], width
        )
        for name, (width, lag) in widths.items()
    }
    valid = np.ones(stop - first, dtype=bool)
    for name, matrix in matrices.items():
        required = (
            matrix[:, -1:]
            if name == "period_return" and factor == "drawdown_rebound_20d"
            else matrix[:, (0, -1)]
            if name == "adj_close" and factor == "price_impact_efficiency_20d"
            else matrix
        )
        valid &= np.isfinite(required).all(axis=1)
        if name in {"adj_close", "adj_open"}:
            valid &= (required > 0).all(axis=1)
        elif name == "turnover":
            valid &= (required >= 0).all(axis=1)
    selected = {name: matrix[valid] for name, matrix in matrices.items()}
    if not valid.any():
        return output

    if "adj_open" in names:
        opening = selected["adj_open"][:, -1]
        if factor == "opening_gap":
            result = opening / selected["adj_close"][:, -1] - 1
        else:
            intraday = selected["adj_close"][:, -1] / opening - 1
            if factor == "intraday_return":
                result = intraday
            else:
                gap = opening / selected["adj_close"][:, -2] - 1
                result = (
                    np.maximum(-gap, 0) * intraday
                    if factor == "down_gap_recovery"
                    else np.maximum(gap, 0) * -intraday
                )
    elif factor.startswith("price_position_") or position_interaction:
        prices = selected["adj_close"]
        low, high = prices.min(axis=1), prices.max(axis=1)
        result = np.divide(
            prices[:, -1] - low,
            high - low,
            out=np.full(len(low), np.nan),
            where=high > low,
        )
        if position_interaction:
            turnover, returns = selected["turnover"], selected["period_return"]
            baseline = turnover[:, :-1].mean(axis=1)
            activity = np.divide(
                turnover[:, -1],
                baseline,
                out=np.full(len(low), np.nan),
                where=baseline > 0,
            )
            scale = _sample_standard_deviation(returns[:, :-1])
            shock = np.divide(
                returns[:, -1] - returns[:, :-1].mean(axis=1),
                scale,
                out=np.full(len(low), np.nan),
                where=scale > 0,
            )
            if factor.startswith("low_position_"):
                result = (
                    (1 - result) * np.maximum(activity - 1, 0) * np.maximum(shock, 0)
                )
            else:
                result = result * np.maximum(activity - 1, 0) * np.maximum(-shock, 0)
    elif factor == "relative_turnover_20d":
        turnover = selected["turnover"]
        baseline = turnover[:, :-1].mean(axis=1)
        result = np.divide(
            turnover[:, -1],
            baseline,
            out=np.full(len(baseline), np.nan),
            where=baseline > 0,
        )
    elif factor == "return_shock_1_20":
        returns = selected["period_return"]
        scale = _sample_standard_deviation(returns[:, :-1])
        result = np.divide(
            returns[:, -1] - returns[:, :-1].mean(axis=1),
            scale,
            out=np.full(len(scale), np.nan),
            where=scale > 0,
        )
    elif "compression" in factor:
        returns = selected["period_return"]
        long = _sample_standard_deviation(returns)
        short = _sample_standard_deviation(returns[:, -20:])
        ratio = np.divide(short, long, out=np.full(len(long), np.nan), where=long > 0)
        result = ratio
        if factor == "compression_trend_20_60":
            path = np.abs(returns[:, -20:]).sum(axis=1)
            trend = np.divide(
                returns[:, -20:].sum(axis=1),
                path,
                out=np.full(len(path), np.nan),
                where=path > 0,
            )
            result = np.maximum(1 - ratio, 0) * trend
    elif factor == "breakout_efficiency_20d":
        prices = selected["adj_close"]
        prior, current = prices[:, :-1], prices[:, -1]
        distance = np.maximum(current - prior.max(axis=1), 0) - np.maximum(
            prior.min(axis=1) - current, 0
        )
        path = np.abs(np.diff(prices, axis=1)).sum(axis=1)
        result = np.divide(
            distance, path, out=np.full(len(path), np.nan), where=path > 0
        )
    elif factor == "price_impact_efficiency_20d":
        prices = selected["adj_close"]
        turnover = selected["turnover"][:, -20:].sum(axis=1)
        result = np.divide(
            prices[:, -1] / prices[:, 0] - 1,
            turnover,
            out=np.full(len(turnover), np.nan),
            where=turnover > 0,
        )
    else:
        turnover = selected["turnover"]
        baseline = turnover[:, :-1].mean(axis=1)
        shock = np.divide(
            turnover[:, -1],
            baseline,
            out=np.full(len(baseline), np.nan),
            where=baseline > 0,
        )
        if factor == "turnover_absorption_20d":
            returns = selected["period_return"]
            average = np.abs(returns[:, :-1]).mean(axis=1)
            impact = np.divide(
                np.abs(returns[:, -1]),
                average,
                out=np.full(len(average), np.nan),
                where=average > 0,
            )
            result = shock * (1 - impact)
        else:
            prices = selected["adj_close"]
            result = (
                np.maximum(1 - prices[:, -1] / prices[:, :-1].max(axis=1), 0) * shock
            )
            if factor == "drawdown_rebound_20d":
                result *= selected["period_return"][:, -1]
    output[first - start :][valid] = result
    return output


def _calculate_amount_profile(values):
    """Allocate daily amount once, then share rolling histograms across outputs."""
    price, close = values["adj_close"], values["close"]
    high, low, amount = values["high"], values["low"], values["amount"]
    output = {
        name: np.full(len(price), np.nan) for name in PRICE_VOLUME_PROFILE_DESCRIPTIONS
    }
    valid = (
        np.isfinite(price)
        & (price > 0)
        & np.isfinite(close)
        & (close > 0)
        & np.isfinite(high)
        & np.isfinite(low)
        & (low > 0)
        & (high >= low)
        & np.isfinite(amount)
        & (amount >= 0)
    )
    if len(price) < 61 or not valid.any():
        return output
    adjustment = price[valid] / close[valid]
    lower, upper = low[valid] * adjustment, high[valid] * adjustment
    step = np.log(1.02)
    bins = np.arange(
        np.floor(np.log(lower.min()) / step), np.floor(np.log(upper.max()) / step) + 1
    )
    edges = np.exp(np.r_[bins, bins[-1] + 1] * step)
    overlap = np.maximum(
        np.minimum(upper[:, None], edges[1:]) - np.maximum(lower[:, None], edges[:-1]),
        0,
    )
    width = upper - lower
    allocation = np.divide(
        overlap, width[:, None], out=np.zeros_like(overlap), where=width[:, None] > 0
    )
    flat = width == 0
    indices = np.floor(np.log(lower[flat]) / step).astype(int) - int(bins[0])
    allocation[np.flatnonzero(flat), indices] = 1
    contributions = np.zeros((len(price), len(bins)))
    contributions[valid] = allocation * amount[valid, None]
    profiles = (
        pd.DataFrame(contributions)
        .rolling(60, min_periods=60)
        .sum()
        .shift(1)
        .to_numpy()
    )
    complete = pd.Series(valid.astype(int)).rolling(60).sum().shift(1).eq(60).to_numpy()
    totals = profiles.sum(axis=1)
    profile_known = complete & (totals > 0)
    output["traded_amount_concentration_60d"][profile_known] = (
        profiles[profile_known].max(axis=1) / totals[profile_known]
    )
    usable = profile_known & np.isfinite(price) & (price > 0)
    if not usable.any():
        return output
    profile = profiles[usable]
    total = totals[usable]
    node = np.exp((bins[np.argmax(profile, axis=1)] + 0.5) * step)
    output["traded_amount_node_distance_60d"][usable] = np.log(price[usable] / node)
    above = np.clip((edges[1:] - price[usable, None]) / np.diff(edges), 0, 1)
    output["traded_amount_above_price_60d"][usable] = (profile * above).sum(
        axis=1
    ) / total
    return output


def _calculate_level_values(values, factor, start, stop):
    if "_break_retest_" in factor:
        return _calculate_retest_values(values, factor, start, stop)
    names, windows = _PRICE_VOLUME_INPUT_WINDOWS[factor]
    width = windows[0]
    output = np.full(stop - start, np.nan)
    first = max(start, width - 1)
    if first >= stop:
        return output
    matrices = {
        name: sliding_window_view(values[name][first - width + 1 : stop], width)
        for name in names
    }
    reference_only = factor.startswith(("support_distance_", "resistance_distance_"))
    valid = np.logical_and.reduce(
        [
            (np.isfinite(required) & (required > 0)).all(axis=1)
            for name, matrix in matrices.items()
            for required in (
                matrix[:, :-1]
                if factor.startswith("peak_age_")
                or reference_only
                and name != "adj_close"
                else matrix,
            )
        ]
    )
    if not valid.any():
        return output
    prices = matrices["adj_close"][valid]
    prior, current = prices[:, :-1], prices[:, -1]
    peak = prior.max(axis=1)
    if factor.startswith("peak_drawdown_"):
        result = np.maximum(1 - current / peak, 0)
    elif factor.startswith(("peak_age_", "up_leg_retracement_")):
        peak_index = prior.shape[1] - 1 - prior[:, ::-1].argmax(axis=1)
        if factor.startswith("peak_age_"):
            result = prior.shape[1] - peak_index
        else:
            before_peak = np.arange(prior.shape[1])[None, :] < peak_index[:, None]
            low = np.where(before_peak, prior, np.inf).min(axis=1)
            result = np.divide(
                peak - current,
                peak - low,
                out=np.full(len(peak), np.nan),
                where=np.isfinite(low) & (peak > low),
            )
    else:
        side = (
            "low"
            if factor.startswith("support_") or factor == "failed_down_breakout_20d"
            else "high"
        )
        extremes, close = matrices[side][valid], matrices["close"][valid]
        reference = extremes[:, :-1] * prior / close[:, :-1]
        level = reference.min(axis=1) if side == "low" else reference.max(axis=1)
        if factor.startswith("support_distance_"):
            result = current / level - 1
        elif factor.startswith("resistance_distance_"):
            result = current / level - 1
        elif factor.startswith(("support_rejection_", "resistance_rejection_")):
            extreme = extremes[:, -1] * current / close[:, -1]
            if side == "low":
                touch = (extreme >= level) & (extreme <= 1.01 * level)
                result = np.where(touch, (current - extreme) / level, 0)
                result[extreme > current] = np.nan
            else:
                touch = (extreme >= 0.99 * level) & (extreme <= level)
                result = np.where(touch, (extreme - current) / level, 0)
                result[extreme < current] = np.nan
        elif factor == "failed_up_breakout_20d":
            result = np.where(
                current <= level,
                np.maximum(extremes[:, -1] * current / close[:, -1] / level - 1, 0),
                0,
            )
        else:
            result = np.where(
                current >= level,
                np.maximum(1 - extremes[:, -1] * current / close[:, -1] / level, 0),
                0,
            )
    output[first - start :][valid] = result
    return output


def _calculate_retest_values(values, factor, start, stop):
    """Freeze each observed breakout's level instead of moving it with the range."""
    names, windows = _PRICE_VOLUME_INPUT_WINDOWS[factor]
    width = windows[0]
    output = np.full(stop - start, np.nan)
    first = max(start, width - 1)
    if first >= stop:
        return output
    matrices = {
        name: sliding_window_view(values[name][first - width + 1 : stop], width)
        for name in names
    }
    upward = factor.startswith("up_")
    reference_side, touch_side = ("high", "low") if upward else ("low", "high")
    prices, close = matrices["adj_close"], matrices["close"]
    prices = np.where(np.isfinite(prices) & (prices > 0), prices, np.nan)
    close = np.where(np.isfinite(close) & (close > 0), close, np.nan)
    reference = matrices[reference_side][:, :-2] * prices[:, :-2] / close[:, :-2]
    reference = np.where(np.isfinite(reference) & (reference > 0), reference, np.nan)
    touch = matrices[touch_side][:, -1] * prices[:, -1] / close[:, -1]
    touch_known = (
        np.isfinite(touch)
        & (touch > 0)
        & ((touch <= prices[:, -1]) if upward else (touch >= prices[:, -1]))
    )
    result = np.zeros(len(prices))
    window = width - 6
    for lag in range(5, 0, -1):
        event = width - 1 - lag
        past = reference[:, event - window : event]
        known = np.isfinite(past)
        level = np.where(known, past, -np.inf if upward else np.inf)
        level = level.max(axis=1) if upward else level.min(axis=1)
        event_known = np.isfinite(prices[:, event])
        did_not_break = event_known & (
            (prices[:, event] <= level) if upward else (prices[:, event] >= level)
        )
        complete = known.all(axis=1) & event_known
        path = prices[:, event:]
        crossed_back = (
            (path < level[:, None]) if upward else (path > level[:, None])
        ).any(axis=1)
        result[~complete & ~did_not_break & ~crossed_back] = np.nan
        path_known = np.isfinite(path).all(axis=1)
        held = (
            (path >= level[:, None]).all(axis=1)
            if upward
            else (path <= level[:, None]).all(axis=1)
        )
        broke = complete & ~did_not_break
        result[broke & ~path_known & ~crossed_back] = np.nan
        touched = (touch >= 0.99 * level) & (touch <= 1.01 * level)
        move = (prices[:, -1] - touch) if upward else (touch - prices[:, -1])
        score = np.where(touch_known, np.where(touched, move / level, 0), np.nan)
        result = np.where(broke & held, score, result)
    output[first - start :] = result
    return output


def _calculate_published_values(
    values: dict[str, np.ndarray], factor: str, start: int, stop: int
) -> np.ndarray:
    """Evaluate the catalog's published formulas directly."""
    names, windows = _PRICE_VOLUME_INPUT_WINDOWS[factor]
    output = np.full(stop - start, np.nan)
    if factor == "alpha101_054":
        opening, high, low, close = (values[name][start:stop] for name in names)
        valid = (
            np.isfinite(opening)
            & np.isfinite(high)
            & np.isfinite(low)
            & np.isfinite(close)
            & (low != high)
            & (close != 0)
        )
        # Cancel the common fifth power before evaluation to avoid price-scale overflow.
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            output[valid] = (
                -(low[valid] - close[valid])
                / (low[valid] - high[valid])
                * (opening[valid] / close[valid]) ** 5
            )
        return output

    size = windows[0]
    first = max(start, size - 1)
    if first >= stop:
        return output
    matrix = sliding_window_view(values["adj_close"][first - size + 1 : stop], size)
    valid = np.isfinite(matrix).all(axis=1)
    current = matrix[valid]
    if not len(current):
        return output
    if factor == "alpha191_046":
        normalized = np.divide(
            current,
            current[:, -1, None],
            out=np.full_like(current, np.nan),
            where=current[:, -1, None] != 0,
        )
        result = sum(np.mean(normalized[:, -width:], axis=1) for width in windows) / 4
    elif factor == "qlib_sump_20d":
        differences = np.diff(current, axis=1)
        result = np.maximum(differences, 0).sum(axis=1) / (
            np.abs(differences).sum(axis=1) + 1e-12
        )
    elif factor in {"qlib_rsqr_20d", "qlib_resi_20d"}:
        positions = np.arange(1, size + 1, dtype=float)
        positions -= positions.mean()
        paired = np.ones(current.shape, dtype=bool)
        if factor == "qlib_rsqr_20d":
            result = (
                _pearson_correlations(
                    current, np.broadcast_to(positions, current.shape), paired=paired
                )
                ** 2
            )
            result[_sample_standard_deviation(current) <= 2e-5] = np.nan
        else:
            centered = _centered_values(current, paired)
            slope = np.sum(centered * positions, axis=1) / np.sum(positions**2)
            residual = centered[:, -1] - slope * positions[-1]
            result = np.divide(
                residual,
                current[:, -1],
                out=np.full(len(current), np.nan),
                where=current[:, -1] != 0,
            )
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
