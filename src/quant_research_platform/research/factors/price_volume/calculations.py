"""Price-volume statistics on nominal sessions, complete paths and exact endpoints."""

from __future__ import annotations

from collections.abc import Iterable
from decimal import Decimal, localcontext
from fractions import Fraction
from itertools import accumulate
from math import ceil, fsum
from statistics import mean as arithmetic_mean

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from quant_research_platform.compute.dataframe.cross_section import (
    _CorrelationMoments,
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
        with np.errstate(over="ignore"):
            difference = numerator[valid] - denominator[valid]
            current[valid] = np.where(
                np.isfinite(difference),
                difference / denominator[valid],
                numerator[valid] / denominator[valid] - 1.0,
            )
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
        counts = counts[valid]
        exceptional = np.zeros(len(left), dtype=bool)
        centered = []
        for current in (left, right):
            anchor = current[np.arange(len(current)), np.argmax(paired, axis=1)]
            with np.errstate(over="ignore", invalid="ignore"):
                differences = np.where(paired, current - anchor[:, None], 0.0)
                scale = np.max(np.abs(differences), axis=1, keepdims=True)
                normalized = np.divide(
                    differences, scale, out=np.zeros_like(differences), where=scale > 0
                )
            exceptional |= ~np.isfinite(normalized).all(axis=1)
            exceptional |= ((differences != 0) & (normalized == 0)).any(axis=1)
            mean = normalized.sum(axis=1) / counts
            centered.append(np.where(paired, normalized - mean[:, None], 0.0))
        x, y = centered
        products = x * y
        covariance = products.sum(axis=1)
        denominator = np.sqrt((x * x).sum(axis=1)) * np.sqrt((y * y).sum(axis=1))
        result = np.divide(
            covariance,
            denominator,
            out=np.full(len(left), np.nan),
            where=denominator > 0,
        )
        # Cancellation-sensitive and extreme-range windows retain the exact owner.
        exceptional |= (denominator > 0) & (
            np.abs(covariance)
            <= size * np.finfo(float).eps * np.abs(products).sum(axis=1)
        )
        for index in np.flatnonzero(exceptional):
            moments = _CorrelationMoments()
            for x_value, y_value in zip(
                left[index][paired[index]], right[index][paired[index]], strict=True
            ):
                moments.update(float(x_value), float(y_value))
            value, _reason = moments.result()
            result[index] = np.nan if value is None else value
        output[first - start :][valid] = np.clip(result, -1.0, 1.0)
        return output

    def window(column: np.ndarray) -> np.ndarray:
        return sliding_window_view(column, size)[
            first - size + 1 - shift : stop - size + 1 - shift
        ]

    def complete_window(name: str, condition: str = "finite") -> np.ndarray:
        key = ("invalid_count", name, condition)
        if key not in sums:
            column = values[name]
            if len(column) >= np.iinfo(np.int64).max:
                raise ValueError("price_volume_history_too_long")
            bad = (
                ~np.isfinite(column)
                if condition == "finite"
                else column <= 0
                if condition == "positive"
                else column == 0
            )
            prefix = np.empty(len(column) + 1, dtype=np.int64)
            prefix[0] = 0
            np.cumsum(bad, dtype=np.int64, out=prefix[1:])
            sums[key] = prefix
        prefix = sums[key]
        return (
            prefix[first + 1 - shift : stop + 1 - shift]
            == prefix[first + 1 - shift - size : stop + 1 - shift - size]
        )

    valid = complete_window(source_name)
    extra_names = (
        ("low", "close")
        if factor == "amplitude_20d"
        else ("amount",)
        if factor.startswith("amihud_")
        or factor in {"signed_amount_flow_20d", "return_amount_imbalance_20d"}
        else ()
    )
    for name in extra_names:
        valid &= complete_window(name)
    if factor == "drawdown_60d":
        valid &= complete_window(source_name, "positive")
    if factor == "amplitude_20d":
        valid &= complete_window("close", "nonzero")
    if factor.startswith("amihud_"):
        valid &= complete_window("amount", "positive")
    partial = factor in PRICE_VOLUME_PARTIAL_STATISTICS
    numeric = np.isfinite(source)
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
    if shift or factor in {
        "log_avg_amount_20d",
        "avg_amount_20d",
        "avg_amount_126d",
        "turnover_250d",
        "turnover_20d",
        "turnover_60d",
    }:
        if source_name not in sums:
            sums[source_name] = _binary_prefix(source)
        totals, unit = sums[source_name]
        result = np.full(np.count_nonzero(valid), np.nan)
        for index, position in enumerate(np.flatnonzero(valid) + first):
            count = int(counts[position - first])
            total = totals[position + 1 - shift] - totals[position + 1 - shift - size]
            if shift:
                if total:
                    numerator, denominator = float(source[position]).as_integer_ratio()
                    numerator = numerator * count * unit - denominator * total
                    denominator *= total
                    try:
                        result[index] = numerator / denominator
                    except OverflowError:
                        result[index] = (
                            np.inf if (numerator > 0) == (denominator > 0) else -np.inf
                        )
            elif factor == "log_avg_amount_20d":
                if total > 0:
                    mean = total / (count * unit)
                    if mean > 5e-324 or total * (1 << 1074) >= count * unit:
                        result[index] = np.log(mean)
                    else:
                        with localcontext() as context:
                            context.prec = 80
                            result[index] = float(
                                (Decimal(total) / Decimal(count * unit)).ln()
                            )
            else:
                result[index] = total / (count * unit)
        output[first - start :][valid] = result
        return output
    if factor in {"amount_cv_126d", "turnover_cv_126d"}:
        current = window(source)[valid]
        _, exponent = np.frexp(np.nanmax(np.abs(current), axis=1, keepdims=True))
        scaled = np.ldexp(current, -exponent)
        center = np.nanmean(scaled, axis=1)
        deviation = _sample_standard_deviation(scaled)
        result = np.divide(
            deviation, center, out=np.full(len(current), np.nan), where=center > 0
        )
        # Preserve meaningful extreme cancellation without Decimal on ordinary
        # positive amount/turnover windows.
        exceptional = ((current > 0).any(axis=1) & (current < 0).any(axis=1)) | (
            (current != 0) & (scaled == 0)
        ).any(axis=1)
        for index in np.flatnonzero(exceptional):
            terms = tuple(map(Fraction, current[index][np.isfinite(current[index])]))
            count = len(terms)
            mean = sum(terms, Fraction()) / count
            if mean <= 0:
                result[index] = np.nan
                continue
            squared = sum((value - mean) ** 2 for value in terms) / (
                (count - 1) * mean * mean
            )
            with localcontext() as context:
                context.prec = 80
                result[index] = float(
                    (Decimal(squared.numerator) / Decimal(squared.denominator)).sqrt()
                )
        output[first - start :][valid] = result
        return output
    if factor == "abnormal_turnover_20_250":
        if source_name not in sums:
            sums[source_name] = _binary_prefix(source)
        totals, _unit = sums[source_name]
        short = windows[1]
        short_numeric = window_count(numeric, short)
        short_fresh = window_count(numeric & fresh, short)
        valid &= short_fresh >= ceil(min_fresh_fraction * short)
        result = np.full(np.count_nonzero(valid), np.nan)
        for index, position in enumerate(np.flatnonzero(valid) + first + 1):
            denominator = totals[position] - totals[position - size]
            if denominator > 0:
                numerator = int(counts[position - first - 1]) * (
                    totals[position] - totals[position - short]
                )
                try:
                    result[index] = numerator / (
                        int(short_numeric[position - first - 1]) * denominator
                    )
                except OverflowError:
                    # Preserve the nonfinite result of an unrepresentable signed ratio.
                    result[index] = np.inf if numerator > 0 else -np.inf
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
        scale = np.nanmax(np.abs(clipped), axis=1, keepdims=True)
        normalized = np.divide(
            clipped, scale, out=np.zeros_like(clipped), where=scale > 0
        )
        result = scale[:, 0] * np.sqrt(
            np.nansum(normalized * normalized, axis=1) / sample_counts
        )
    elif factor == "max_return_20d":
        result = np.max(current, axis=1)
    elif factor == "top5_return_mean_20d":
        top = np.partition(current, -5, axis=1)[:, -5:]
        scale = np.max(np.abs(top), axis=1, keepdims=True)
        normalized = np.divide(top, scale, out=np.zeros_like(top), where=scale > 0)
        total = normalized.sum(axis=1)
        result = (total / 5) * scale[:, 0]
        exceptional = ((top != 0) & (normalized == 0)).any(axis=1) | (
            np.abs(total) <= 5 * np.finfo(float).eps * np.abs(normalized).sum(axis=1)
        )
        for index in np.flatnonzero(exceptional):
            result[index] = arithmetic_mean(top[index])
    elif factor == "min_return_20d":
        result = np.min(current, axis=1)
    elif factor in {"skewness_20d", "skewness_60d", "kurtosis_60d"}:
        # Binary scaling preserves adjacent values and bounds anchor subtraction.
        # Center the differences so the original offset cannot round away the mean.
        _, exponent = np.frexp(np.nanmax(np.abs(current), axis=1, keepdims=True))
        normalized = np.ldexp(current, -exponent)
        anchor = normalized[
            np.arange(len(normalized)), np.argmax(np.isfinite(normalized), axis=1)
        ]
        differences = normalized - anchor[:, None]
        centered = differences - np.nanmean(differences, axis=1, keepdims=True)
        centered /= np.nanmax(np.abs(centered), axis=1, keepdims=True)
        moment_2 = np.nanmean(centered * centered, axis=1)
        size = sample_counts
        if factor == "kurtosis_60d":
            moment_4 = np.nanmean(centered**4, axis=1)
            result = (
                (size - 1)
                / ((size - 2) * (size - 3))
                * ((size + 1) * (moment_4 / (moment_2 * moment_2) - 3.0) + 6.0)
            )
        else:
            moment_3 = np.nanmean(centered**3, axis=1)
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
        return_mantissa, return_exponent = np.frexp(numerator)
        amount_mantissa, amount_exponent = np.frexp(denominator)
        quotients = np.divide(
            return_mantissa,
            amount_mantissa,
            out=np.full_like(return_mantissa, np.nan),
            where=amount_mantissa != 0,
        )
        exponents = return_exponent - amount_exponent
        scale = np.max(
            np.where(
                quotients != 0, exponents, np.min(exponents, axis=1, keepdims=True)
            ),
            axis=1,
            keepdims=True,
        )
        normalized = np.ldexp(quotients, exponents - scale)
        result = np.ldexp(np.nanmean(normalized, axis=1), scale[:, 0])
    elif factor == "trend_efficiency_60d":
        _, exponent = np.frexp(np.max(np.abs(current), axis=1, keepdims=True))
        scaled = np.ldexp(current, -exponent)
        denominator = np.sum(np.abs(scaled), axis=1)
        numerator = np.sum(scaled, axis=1)
        # Only cancellation at the reduction's error scale needs stable summation.
        near_cancellation = np.abs(numerator) <= (
            np.finfo("float64").eps * current.shape[1] * denominator
        )
        for row in np.flatnonzero(near_cancellation & (denominator != 0)):
            numerator[row] = fsum(scaled[row])
        result = np.divide(
            np.abs(numerator),
            denominator,
            out=np.full(len(current), np.nan),
            where=denominator != 0,
        )
    elif factor in {"signed_amount_flow_20d", "return_amount_imbalance_20d"}:
        flow = extra["amount"]
        if factor == "signed_amount_flow_20d":
            scale = np.max(np.abs(flow), axis=1, keepdims=True)
            normalized = np.divide(
                flow, scale, out=np.zeros_like(flow), where=scale > 0
            )
            terms = np.sign(current) * normalized
            numerator = np.sum(terms, axis=1)
            denominator = np.sum(normalized, axis=1)
        else:
            return_mantissa, return_exponent = np.frexp(current)
            flow_mantissa, flow_exponent = np.frexp(flow)
            products = return_mantissa * flow_mantissa
            exponents = return_exponent + flow_exponent
            scale = np.max(
                np.where(
                    products != 0,
                    exponents,
                    np.min(exponents, axis=1, keepdims=True),
                ),
                axis=1,
                keepdims=True,
            )
            normalized = np.ldexp(products, exponents - scale)
            terms = normalized
            numerator = np.sum(terms, axis=1)
            denominator = np.sum(np.abs(normalized) * np.sign(flow), axis=1)
        result = np.divide(
            numerator,
            denominator,
            out=np.full(len(current), np.nan),
            where=denominator != 0,
        )
        # Only ill-conditioned signed sums need exact products and cancellation.
        denominator_terms = (
            normalized
            if factor == "signed_amount_flow_20d"
            else np.abs(normalized) * np.sign(flow)
        )
        cancelled = np.zeros(len(current), dtype=bool)
        for terms_, totals in ((terms, numerator), (denominator_terms, denominator)):
            cancelled |= (
                (terms_ > 0).any(axis=1)
                & (terms_ < 0).any(axis=1)
                & (
                    np.abs(totals)
                    <= size * np.finfo(float).eps * np.abs(terms_).sum(axis=1)
                )
            )
        for row in np.flatnonzero(cancelled):
            returns = tuple(map(Fraction, current[row]))
            amounts = tuple(map(Fraction, flow[row]))
            if factor == "signed_amount_flow_20d":
                exact_numerator = sum(
                    (1 if value > 0 else -1 if value < 0 else 0) * amount
                    for value, amount in zip(returns, amounts, strict=True)
                )
                exact_denominator = sum(amounts)
            else:
                exact_numerator = sum(
                    value * amount
                    for value, amount in zip(returns, amounts, strict=True)
                )
                exact_denominator = sum(
                    abs(value) * amount
                    for value, amount in zip(returns, amounts, strict=True)
                )
            if exact_denominator:
                ratio = exact_numerator / exact_denominator
                try:
                    result[row] = float(ratio)
                except OverflowError:
                    result[row] = np.inf if ratio > 0 else -np.inf
            else:
                result[row] = np.nan
    elif factor == "positive_return_fraction_20d":
        result = np.mean(current > 0.0, axis=1)
    elif factor in {"largest_daily_contribution_20d", "path_roughness_20d"}:
        scale = np.max(np.abs(current), axis=1, keepdims=True)
        normalized = np.divide(
            current, scale, out=np.zeros_like(current), where=scale > 0
        )
        denominator = np.sum(np.abs(normalized), axis=1)
        numerator = (
            np.max(np.abs(normalized), axis=1)
            if factor == "largest_daily_contribution_20d"
            else np.abs(np.sum(normalized, axis=1))
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


def _binary_prefix(values: np.ndarray) -> tuple[list[int], int]:
    # Exact binary sums make subtraction safe across windows and chunk boundaries.
    # Missing values are zero only inside this accumulator; callers use numeric
    # sample counts and separate fresh-observation validity gates.
    ratios = [
        float(value).as_integer_ratio() if np.isfinite(value) else (0, 1)
        for value in values
    ]
    exponent = max(denominator.bit_length() - 1 for _, denominator in ratios)
    unit = 1 << exponent
    totals = list(
        accumulate(
            (
                numerator << (exponent - (denominator.bit_length() - 1))
                for numerator, denominator in ratios
            ),
            initial=0,
        )
    )
    return totals, unit


def _sample_standard_deviation(windows: np.ndarray) -> np.ndarray:
    finite = np.isfinite(windows)
    if not finite.all():
        counts = finite.sum(axis=1)
        result = np.full(len(windows), np.nan)
        for count in np.unique(counts[counts >= 2]):
            selected = counts == count
            compact = windows[selected][finite[selected]].reshape(-1, count)
            result[selected] = _sample_standard_deviation(compact)
        return result
    # Subtract first to preserve adjacent prices; scale before squaring to keep
    # representable deviations finite even at extreme economic units.
    with np.errstate(over="ignore"):
        differences = windows - windows[:, :1]
    overflow = ~np.isfinite(differences).all(axis=1)
    unit = np.max(np.abs(windows[overflow]), axis=1, keepdims=True)
    differences[overflow] = windows[overflow] / unit
    spread = np.max(np.abs(differences), axis=1, keepdims=True)
    normalized = np.divide(
        differences, spread, out=np.zeros_like(differences), where=spread > 0
    )
    result = np.std(normalized, axis=1, ddof=1) * spread[:, 0]
    result[overflow] *= unit[:, 0]
    return result


def _log(values: np.ndarray) -> np.ndarray:
    result = np.full(len(values), np.nan)
    valid = np.isfinite(values) & (values > 0)
    result[valid] = np.log(values[valid])
    return result
