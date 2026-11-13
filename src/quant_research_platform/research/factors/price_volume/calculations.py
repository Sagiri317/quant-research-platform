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
from quant_research_platform.compute.dataframe.history import (
    completed_month_return_acceleration,
    path_efficiency,
    quadratic_log_trend,
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
    for name in names:
        ordered[name] = strict_finite_numeric_series(
            ordered[name],
            error=f"price_volume_value_invalid: {name}",
            allow_null=True,
        ).to_numpy(dtype="float64", copy=False)
    ordered = ordered.sort_values([*history_keys, "timestamp"], kind="mergesort")
    bounds = (
        None if output_window is None else time_window_timestamp_interval(output_window)
    )
    output_keys = ordered.loc[:, ("asset_id", "timestamp")]
    if bounds is not None:
        lower, upper, closed = bounds
        timestamps = output_keys["timestamp"]
        output_keys = output_keys.loc[
            timestamps.ge(lower)
            & (timestamps.le(upper) if closed else timestamps.lt(upper))
        ]
    output_keys = output_keys.reset_index(drop=True)
    output = np.empty((len(output_keys), len(requested)), dtype="float64", order="F")
    offset = 0
    for _identity, group in ordered.groupby(
        list(history_keys), sort=False, dropna=False
    ):
        values = {
            name: group[name].to_numpy(dtype="float64", copy=False) for name in names
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
        if "return_acceleration_6m6m" in requested:
            sums["return_acceleration_6m6m"] = completed_month_return_acceleration(
                values["adj_close"], group.timestamp
            )
        for column, factor in enumerate(requested):
            partial = factor in PRICE_VOLUME_PARTIAL_STATISTICS
            output[offset : offset + stop - start, column] = _calculate_values(
                statistical_values if partial else values,
                statistical_sums if partial else sums,
                factor,
                start,
                stop,
                fresh=fresh,
                min_fresh_fraction=min_fresh_fraction,
            )
        offset += stop - start
    output[~np.isfinite(output)] = np.nan
    computed = pd.DataFrame(
        output, columns=requested, index=output_keys.index, copy=False
    )
    return pd.concat([output_keys, computed], axis=1)


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
    if factor == "return_acceleration_6m6m":
        return sums[factor][start:stop].copy()
    if factor.startswith("breakout_activity_divergence_"):
        width = windows[0] - 1
        price = pd.Series(values["adj_close"]).where(lambda x: x > 0)
        prior = price.shift().rolling(width, min_periods=width)
        excess = np.log(price / prior.max()).clip(lower=0) + np.log(
            price / prior.min()
        ).clip(upper=0)
        amount = pd.Series(values["amount"]).where(lambda x: x >= 0)
        mean = amount.shift().rolling(width, min_periods=width).mean()
        shortfall = np.log(mean.where(mean > 0) / amount.where(amount > 0)).clip(
            lower=0
        )
        result = (excess * shortfall).where(excess.ne(0), 0)
        return result.to_numpy()[start:stop].copy()
    if factor.startswith(("price_up_acceleration_", "price_down_acceleration_")):
        width = windows[0]
        key = ("quadratic_log_trend", width)
        if key not in sums:
            prices = values["adj_close"]
            trend = np.full((len(prices), 2), np.nan)
            if len(prices) >= width:
                matrix = sliding_window_view(prices, width)
                valid = np.isfinite(matrix).all(axis=1) & (matrix > 0).all(axis=1)
                slope, curvature = quadratic_log_trend(matrix[valid])
                trend[width - 1 :][valid] = np.column_stack((slope, curvature))
            sums[key] = trend
        slope, curvature = sums[key].T
        direction = 1 if factor.startswith("price_up_") else -1
        known = np.isfinite(slope) & np.isfinite(curvature)
        result = np.where(
            known,
            np.where(direction * slope > 0, np.maximum(direction * curvature, 0), 0),
            np.nan,
        )
        return result[start:stop].copy()
    if factor.startswith(("signed_breakout_", "breakout_activity_confirmation_")):
        width = windows[0] - 1
        # The episode is shared by price-only selections and the activity view.
        key = ("breakout_episode", width)
        if key not in sums:
            sums[key] = _breakout_episode(values["adj_close"], width)
        direction, duration, followthrough = sums[key]
        if factor.startswith("signed_breakout_duration_"):
            result = duration
        elif factor.startswith("signed_breakout_followthrough_"):
            result = followthrough
        else:
            amount = pd.Series(values["amount"]).where(lambda x: x >= 0)
            mean = amount.shift().rolling(width, min_periods=width).mean()
            activity = np.log(amount.where(amount > 0) / mean.where(mean > 0))
            result = np.where(direction == 0, 0, direction * activity)
        return np.asarray(result)[start:stop].copy()
    if factor in PRICE_VOLUME_PROFILE_DESCRIPTIONS:
        if "amount_profile_60d" not in sums:
            sums["amount_profile_60d"] = _calculate_amount_profile(values)
        return sums["amount_profile_60d"][factor][start:stop].copy()
    if factor in PRICE_VOLUME_LEVEL_DESCRIPTIONS:
        return _calculate_level_values(values, sums, factor, start, stop)
    if factor in {"volatility_compression_20_60", "compression_trend_20_60"}:
        size = windows[0]
        first = max(start, size - 1)
        if first >= stop:
            return np.full(stop - start, np.nan)
        key = "volatility_compression_20_60"
        if key not in sums:
            ratio = np.full(stop - start, np.nan)
            returns = sliding_window_view(
                values["period_return"][first - size + 1 : stop], size
            )
            valid = np.isfinite(returns).all(axis=1)
            current = returns[valid]
            long = _sample_standard_deviation(current)
            short = _sample_standard_deviation(current[:, -20:])
            ratio[first - start :][valid] = np.divide(
                short, long, out=np.full(len(long), np.nan), where=long > 0
            )
            sums[key] = ratio
        ratio = sums[key]
        if factor == key:
            return ratio.copy()
        recent = sliding_window_view(values["period_return"][first - 19 : stop], 20)
        path = np.abs(recent).sum(axis=1)
        trend = np.divide(
            recent.sum(axis=1),
            path,
            out=np.full(len(path), np.nan),
            where=path > 0,
        )
        output = np.full(stop - start, np.nan)
        output[first - start :] = np.maximum(1 - ratio[first - start :], 0) * trend
        return output
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
        "avg_amount_20d",
        "avg_volume_20d",
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
    if factor in {"max_return_20d", "min_return_20d"}:
        reducer = np.max if factor == "max_return_20d" else np.min
        output[first - start :][valid] = reducer(window(source), axis=1)[valid]
        return output
    if factor in {"positive_return_fraction_20d", "zero_amount_ratio_20d"}:
        mask = source > 0 if factor == "positive_return_fraction_20d" else source <= 0
        output[first - start :][valid] = window_count(mask)[valid] / size
        return output
    semivolatility = factor.startswith("downside_volatility_") or factor == (
        "upside_volatility_20d"
    )
    if semivolatility or factor == "amplitude_20d" or factor.startswith("amihud_"):
        required = slice(first - size + 1, stop)
        sample, known = source[required], numeric[required]
        if semivolatility:
            clipped = (
                np.minimum(sample, 0.0)
                if factor.startswith("downside")
                else np.maximum(sample, 0.0)
            )
            contributions = clipped * clipped
        else:
            numerator = (
                sample - values["low"][required]
                if factor == "amplitude_20d"
                else np.abs(sample)
            )
            denominator = values["close" if factor == "amplitude_20d" else "amount"][
                required
            ]
            contributions = np.divide(
                numerator, denominator, out=np.full(len(sample), np.nan), where=known
            )
        means = (
            pd.Series(np.where(known, contributions, np.nan))
            .rolling(size, min_periods=1)
            .mean()
            .to_numpy()[size - 1 :]
        )
        output[first - start :][valid] = (
            np.sqrt(means[valid]) if semivolatility else means[valid]
        )
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
    elif factor == "top5_return_mean_20d":
        top = np.partition(current, -5, axis=1)[:, -5:]
        result = np.mean(top, axis=1)
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
    elif factor == "trend_efficiency_60d":
        result = np.abs(
            path_efficiency(current.sum(axis=1), np.abs(current).sum(axis=1))
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
    elif factor in {"largest_daily_contribution_20d", "path_roughness_20d"}:
        denominator = np.sum(np.abs(current), axis=1)
        result = (
            path_efficiency(np.max(np.abs(current), axis=1), denominator)
            if factor == "largest_daily_contribution_20d"
            else np.abs(path_efficiency(current.sum(axis=1), denominator))
        )
        if factor == "path_roughness_20d":
            result = 1.0 - result
    else:
        raise ValueError("price_volume_calculation_missing")
    output[first - start :][valid] = result
    return output


def _breakout_episode(price, width):
    """Track a frozen range boundary in linear time without look-ahead."""
    price = np.asarray(price, dtype=float)
    history = pd.Series(price).where(lambda x: x > 0).shift().rolling(width)
    high, low = history.max().to_numpy(), history.min().to_numpy()
    known = np.isfinite(price) & (price > 0) & np.isfinite(high) & np.isfinite(low)
    direction, duration, followthrough = np.full((3, len(price)), np.nan)
    side, level, origin, started = 0, np.nan, np.nan, 0
    for i, current in enumerate(price):
        if not known[i]:
            side = 0
            continue
        if side and side * (current - level) <= 0:
            side = 0
        if side == 0:
            side = int(current > high[i]) - int(current < low[i])
            if side:
                level, origin, started = high[i] if side > 0 else low[i], current, i
        direction[i] = side
        duration[i] = side * (i - started + 1) if side else 0
        followthrough[i] = current / origin - 1 if side else 0
    return direction, duration, followthrough


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


def _price_extreme(values, sums, side):
    """Own adjusted references once per asset, without window-sized copies."""
    key = ("price_extreme", side)
    if key not in sums:
        price = values["adj_close"]
        valid = np.isfinite(price) & (price > 0)
        if side == "adj_close":
            extreme = price
        else:
            raw, close = values[side], values["close"]
            valid &= np.isfinite(raw) & (raw > 0) & np.isfinite(close) & (close > 0)
            extreme = np.divide(
                raw * price, close, out=np.full(len(price), np.nan), where=valid
            )
        sums[key] = np.where(valid, extreme, np.nan)
    return sums[key]


def _price_reference(values, sums, side, width):
    key = ("price_reference", side, width)
    if key not in sums:
        rolling = pd.Series(_price_extreme(values, sums, side)).rolling(
            width, min_periods=1
        )
        extreme = rolling.min() if side == "low" else rolling.max()
        sums[key] = extreme.to_numpy(), rolling.count().eq(width).to_numpy()
    return sums[key]


def _calculate_level_values(values, sums, factor, start, stop):
    if factor.startswith("failed_close_"):
        side = "high" if factor.startswith("failed_close_up_") else "low"
        horizon, width = map(int, factor.removesuffix("d").split("_")[-2:])
        key = ("failed_close_episode", side, width)
        if key not in sums:
            sums[key] = _closing_breakout_failure(values, sums, side, width)
        age, distance = sums[key]
        return np.where(age > horizon, 0, distance)[start:stop].copy()
    if "_break_retest_" in factor:
        return _calculate_retest_values(values, sums, factor, start, stop)
    width = _PRICE_VOLUME_INPUT_WINDOWS[factor][1][0] - 1
    output = np.full(stop - start, np.nan)
    first = max(start, width)
    if first >= stop:
        return output
    if not factor.startswith(("peak_age_", "up_leg_retracement_")):
        side = (
            "adj_close"
            if factor.startswith("peak_drawdown_")
            else "low"
            if factor.startswith(("support_", "failed_down_breakout_"))
            else "high"
        )
        references, complete = _price_reference(values, sums, side, width)
        level = references[first - 1 : stop - 1]
        current = values["adj_close"][first:stop]
        valid = complete[first - 1 : stop - 1] & np.isfinite(current) & (current > 0)
        if side == "adj_close":
            result = np.maximum(1 - current / level, 0)
        elif factor.startswith(("support_distance_", "resistance_distance_")):
            result = current / level - 1
        else:
            extreme = _price_extreme(values, sums, side)[first:stop]
            valid &= np.isfinite(extreme)
            if factor.startswith("support_rejection_"):
                touch = (extreme >= level) & (extreme <= 1.01 * level)
                result = np.where(touch, (current - extreme) / level, 0)
                valid &= extreme <= current
            elif factor.startswith("resistance_rejection_"):
                touch = (extreme >= 0.99 * level) & (extreme <= level)
                result = np.where(touch, (extreme - current) / level, 0)
                valid &= extreme >= current
            elif factor.startswith("failed_up_breakout_"):
                result = np.where(
                    current <= level, np.maximum(extreme / level - 1, 0), 0
                )
            else:
                result = np.where(
                    current >= level, np.maximum(1 - extreme / level, 0), 0
                )
        output[first - start :][valid] = result[valid]
        return output
    prices = sliding_window_view(
        _price_extreme(values, sums, "adj_close")[first - width : stop], width + 1
    )
    valid = np.isfinite(prices)
    if factor.startswith("peak_age_"):
        valid = valid[:, :-1]
    valid = valid.all(axis=1)
    if not valid.any():
        return output
    prices = prices[valid]
    prior, current = prices[:, :-1], prices[:, -1]
    peak_index = prior.shape[1] - 1 - prior[:, ::-1].argmax(axis=1)
    if factor.startswith("peak_age_"):
        result = prior.shape[1] - peak_index
    else:
        peak = prior.max(axis=1)
        before_peak = np.arange(prior.shape[1])[None, :] < peak_index[:, None]
        low = np.where(before_peak, prior, np.inf).min(axis=1)
        result = np.divide(
            peak - current,
            peak - low,
            out=np.full(len(peak), np.nan),
            where=np.isfinite(low) & (peak > low),
        )
    output[first - start :][valid] = result
    return output


def _closing_breakout_failure(values, sums, side, width):
    """Keep an episode's original boundary, including beyond its test horizon."""
    reference, complete = _price_reference(values, sums, side, width)
    price = _price_extreme(values, sums, "adj_close")
    age, distance = np.full((2, len(price)), np.nan)
    sign = 1 if side == "high" else -1
    active, level, began = False, np.nan, 0
    known = np.isfinite(price)
    for i in range(1, len(price)):
        if not known[i]:
            active = False
            continue
        if active:
            age[i], distance[i] = 0, 0
            if sign * (price[i] - level) < 0:
                age[i] = i - began
                distance[i] = sign * (1 - price[i] / level)
                active = False
        elif i >= width and (
            complete[i - 1] or sign * (price[i] - reference[i - 1]) <= 0
        ):
            age[i], distance[i] = 0, 0
            if complete[i - 1] and sign * (price[i] - reference[i - 1]) > 0:
                active, level, began = True, reference[i - 1], i
    return age, distance


def _calculate_retest_values(values, sums, factor, start, stop):
    """Freeze each observed breakout's level instead of moving it with the range."""
    width = _PRICE_VOLUME_INPUT_WINDOWS[factor][1][0]
    output = np.full(stop - start, np.nan)
    first = max(start, width - 1)
    if first >= stop:
        return output
    upward = factor.startswith("up_")
    reference_side, touch_side = ("high", "low") if upward else ("low", "high")
    price = _price_extreme(values, sums, "adj_close")
    prices = sliding_window_view(price[first - 5 : stop], 6)
    references, complete_references = _price_reference(
        values, sums, reference_side, width - 6
    )
    touch = _price_extreme(values, sums, touch_side)[first:stop]
    touch_known = np.isfinite(touch) & (
        (touch <= prices[:, -1]) if upward else (touch >= prices[:, -1])
    )
    result = np.zeros(len(prices))
    for lag in range(5, 0, -1):
        event = 5 - lag
        level = references[first - lag - 1 : stop - lag - 1]
        level = np.where(np.isfinite(level), level, -np.inf if upward else np.inf)
        event_known = np.isfinite(prices[:, event])
        did_not_break = event_known & (
            (prices[:, event] <= level) if upward else (prices[:, event] >= level)
        )
        complete = complete_references[first - lag - 1 : stop - lag - 1] & event_known
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
    return np.sqrt(np.vecdot(centered, centered) / (finite.sum(axis=1) - 1))


def _log(values: np.ndarray) -> np.ndarray:
    result = np.full(len(values), np.nan)
    valid = np.isfinite(values) & (values > 0)
    result[valid] = np.log(values[valid])
    return result
