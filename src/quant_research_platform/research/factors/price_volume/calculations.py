"""Price-volume calculations using complete session windows and exact endpoints."""

from __future__ import annotations

from collections.abc import Iterable
from decimal import Decimal, localcontext
from fractions import Fraction
from itertools import accumulate
from math import fsum, hypot, sqrt
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

from .definitions import _PRICE_VOLUME_INPUT_WINDOWS, PRICE_VOLUME_LIBRARY


def calculate_price_volume_factors(
    panel: pd.DataFrame,
    *,
    factor_columns: Iterable[str] | None = None,
    output_window: TimeWindow | None = None,
    history_keys: tuple[str, ...] = ("asset_id",),
) -> pd.DataFrame:
    """Calculate only requested columns and output rows, preserving warmup inputs."""
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
    ordered = panel.loc[:, (*keys, *names)].copy()
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
        start, stop = 0, len(group)
        if bounds is not None:
            lower, upper, closed = bounds
            start = int(group["timestamp"].searchsorted(lower, side="left"))
            stop = int(
                group["timestamp"].searchsorted(
                    upper, side="right" if closed else "left"
                )
            )
        sums = {}
        for factor in requested:
            output[factor].append(_calculate_values(values, sums, factor, start, stop))
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
    values: dict[str, np.ndarray], sums: dict, factor: str, start: int, stop: int
) -> np.ndarray:
    names, windows = _PRICE_VOLUME_INPUT_WINDOWS[factor]
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
        finite = np.isfinite(source[begin:stop]) & np.isfinite(other[begin:stop])
        moments = _CorrelationMoments()
        for offset, position in enumerate(range(begin, stop)):
            if offset >= size and finite[offset - size]:
                moments.remove(
                    float(source[position - size]), float(other[position - size])
                )
            if finite[offset]:
                moments.update(float(source[position]), float(other[position]))
            if position >= first and moments.count == size:
                value, _reason = moments.result()
                output[position - start] = np.nan if value is None else value
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
    if shift:
        valid &= np.isfinite(source[first:stop])
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
            total = totals[position + 1 - shift] - totals[position + 1 - shift - size]
            if shift:
                if total:
                    numerator, denominator = float(source[position]).as_integer_ratio()
                    numerator = numerator * size * unit - denominator * total
                    denominator *= total
                    try:
                        result[index] = numerator / denominator
                    except OverflowError:
                        result[index] = (
                            np.inf if (numerator > 0) == (denominator > 0) else -np.inf
                        )
            elif factor == "log_avg_amount_20d":
                if total > 0:
                    mean = total / (size * unit)
                    if mean > 5e-324 or total * (1 << 1074) >= size * unit:
                        result[index] = np.log(mean)
                    else:
                        with localcontext() as context:
                            context.prec = 80
                            result[index] = float(
                                (Decimal(total) / Decimal(size * unit)).ln()
                            )
            else:
                result[index] = total / (size * unit)
        output[first - start :][valid] = result
        return output
    if factor in {"amount_cv_126d", "turnover_cv_126d"}:
        current = window(source)[valid]
        _, exponent = np.frexp(np.max(np.abs(current), axis=1, keepdims=True))
        scaled = np.ldexp(current, -exponent)
        center = scaled.mean(axis=1)
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
            terms = tuple(map(Fraction, current[index]))
            mean = sum(terms, Fraction()) / size
            if mean <= 0:
                result[index] = np.nan
                continue
            squared = sum((value - mean) ** 2 for value in terms) / (
                (size - 1) * mean * mean
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
        result = np.full(np.count_nonzero(valid), np.nan)
        for index, position in enumerate(np.flatnonzero(valid) + first + 1):
            denominator = totals[position] - totals[position - size]
            if denominator > 0:
                numerator = size * (totals[position] - totals[position - short])
                try:
                    result[index] = numerator / (short * denominator)
                except OverflowError:
                    # Preserve the nonfinite result of an unrepresentable signed ratio.
                    result[index] = np.inf if numerator > 0 else -np.inf
        output[first - start :][valid] = result
        return output
    matrix = window(source)
    if factor in {"skewness_20d", "skewness_60d", "kurtosis_60d"}:
        valid &= np.min(matrix, axis=1) != np.max(matrix, axis=1)
    current = matrix[valid]
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
    elif factor == "top5_return_mean_20d":
        result = np.fromiter(
            (arithmetic_mean(row) for row in np.partition(current, -5, axis=1)[:, -5:]),
            dtype=float,
        )
    elif factor == "min_return_20d":
        result = np.min(current, axis=1)
    elif factor in {"skewness_20d", "skewness_60d", "kurtosis_60d"}:
        # Binary scaling preserves adjacent values and bounds anchor subtraction.
        # Center the differences so the original offset cannot round away the mean.
        _, exponent = np.frexp(np.max(np.abs(current), axis=1, keepdims=True))
        normalized = np.ldexp(current, -exponent)
        differences = normalized - normalized[:, :1]
        centered = differences - np.mean(differences, axis=1, keepdims=True)
        centered /= np.max(np.abs(centered), axis=1, keepdims=True)
        moment_2 = np.mean(centered * centered, axis=1)
        if factor == "kurtosis_60d":
            moment_4 = np.mean(centered**4, axis=1)
            result = (
                (size - 1)
                / ((size - 2) * (size - 3))
                * ((size + 1) * (moment_4 / (moment_2 * moment_2) - 3.0) + 6.0)
            )
        else:
            moment_3 = np.mean(centered**3, axis=1)
            result = (
                sqrt(size * (size - 1))
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
        quotients = return_mantissa / amount_mantissa
        exponents = return_exponent - amount_exponent
        scale = np.max(
            np.where(
                quotients != 0, exponents, np.min(exponents, axis=1, keepdims=True)
            ),
            axis=1,
            keepdims=True,
        )
        normalized = np.ldexp(quotients, exponents - scale)
        result = np.ldexp(np.mean(normalized, axis=1), scale[:, 0])
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
    # Missing values are zero only inside this accumulator; the caller's complete
    # window mask excludes every result containing them.
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
