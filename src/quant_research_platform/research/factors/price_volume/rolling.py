"""Rolling-window kernels for price-volume factors."""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant_research_platform.contracts.validation import strict_finite_numeric_series


def rolling_aggregate(
    values: pd.Series,
    grouped_values: pd.core.groupby.SeriesGroupBy,
    group_positions: tuple[np.ndarray, ...],
    window: int,
    aggregation: str,
) -> pd.Series:
    del grouped_values
    numeric = _numeric(values)
    source = numeric.to_numpy(dtype="float64", copy=False)
    output = np.full(len(values), np.nan, dtype="float64")
    for positions in group_positions:
        output[positions] = rolling_aggregate_array(
            source[positions],
            window,
            aggregation,
        )
    return pd.Series(output, index=values.index, name=values.name)


def rolling_aggregate_array(
    values: np.ndarray,
    window: int,
    aggregation: str,
) -> np.ndarray:
    output = np.full(len(values), np.nan, dtype="float64")
    if window <= 0 or len(values) < window:
        return output
    windows = np.lib.stride_tricks.sliding_window_view(values, window_shape=window)
    valid_windows = np.isfinite(windows).all(axis=1)
    if not valid_windows.any():
        return output
    valid_values = windows[valid_windows]
    if aggregation == "mean":
        window_values = np.mean(valid_values, axis=1)
    elif aggregation == "max":
        window_values = np.max(valid_values, axis=1)
    elif aggregation == "min":
        window_values = np.min(valid_values, axis=1)
    elif aggregation == "std":
        window_values = _window_std_values(valid_values)
    else:
        raise ValueError(f"unsupported_rolling_aggregation: {aggregation}")
    values_out = np.full(len(windows), np.nan, dtype="float64")
    values_out[valid_windows] = window_values
    output[window - 1 :] = values_out
    return output


def rolling_rms(
    values: pd.Series,
    group_keys: pd.Series,
    group_positions: tuple[np.ndarray, ...],
    window: int,
) -> pd.Series:
    del group_keys
    numeric = _numeric(values)
    source = numeric.to_numpy(dtype="float64", copy=False)
    output = np.full(len(values), np.nan, dtype="float64")
    for positions in group_positions:
        output[positions] = _rolling_rms_array(source[positions], window)
    return pd.Series(output, index=values.index, name=values.name)


def rolling_std(
    values: pd.Series,
    group_keys: pd.Series,
    group_positions: tuple[np.ndarray, ...],
    window: int,
) -> pd.Series:
    return _rolling_window_statistic(values, group_keys, group_positions, window, "std")


def rolling_correlation(
    left: pd.Series,
    right: pd.Series,
    group_positions: tuple[np.ndarray, ...],
    window: int,
) -> pd.Series:
    left_values = _numeric(left).to_numpy(dtype="float64", copy=False)
    right_values = _numeric(right).to_numpy(dtype="float64", copy=False)
    output = np.full(len(left), np.nan, dtype="float64")
    for positions in group_positions:
        output[positions] = rolling_correlation_array(
            left_values[positions],
            right_values[positions],
            window,
        )
    return pd.Series(output, index=left.index, name=left.name)


def rolling_correlation_array(
    left: np.ndarray,
    right: np.ndarray,
    window: int,
) -> np.ndarray:
    output = np.full(len(left), np.nan, dtype="float64")
    if window <= 1 or len(left) < window:
        return output
    left_windows = np.lib.stride_tricks.sliding_window_view(
        left,
        window_shape=window,
    )
    right_windows = np.lib.stride_tricks.sliding_window_view(
        right,
        window_shape=window,
    )
    valid_windows = np.isfinite(left_windows).all(axis=1) & np.isfinite(
        right_windows
    ).all(axis=1)
    if not valid_windows.any():
        return output
    valid_left = left_windows[valid_windows]
    valid_right = right_windows[valid_windows]
    left_centered, _, left_varies = _scaled_centered_windows(valid_left)
    right_centered, _, right_varies = _scaled_centered_windows(valid_right)
    numerator = np.sum(left_centered * right_centered, axis=1)
    denominator = np.sqrt(
        np.sum(left_centered * left_centered, axis=1)
        * np.sum(right_centered * right_centered, axis=1)
    )
    values = np.full(len(valid_left), np.nan, dtype="float64")
    valid_denominator = left_varies & right_varies & (denominator != 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        values[valid_denominator] = (
            numerator[valid_denominator] / denominator[valid_denominator]
        )
    values[~np.isfinite(values)] = np.nan
    window_values = np.full(len(left_windows), np.nan, dtype="float64")
    window_values[valid_windows] = values
    output[window - 1 :] = window_values
    return output


def rolling_skew(
    values: pd.Series,
    group_keys: pd.Series,
    group_positions: tuple[np.ndarray, ...],
    window: int,
) -> pd.Series:
    return _rolling_window_statistic(
        values,
        group_keys,
        group_positions,
        window,
        "skew",
    )


def rolling_kurt(
    values: pd.Series,
    group_keys: pd.Series,
    group_positions: tuple[np.ndarray, ...],
    window: int,
) -> pd.Series:
    return _rolling_window_statistic(
        values,
        group_keys,
        group_positions,
        window,
        "kurt",
    )


def _rolling_window_statistic(
    values: pd.Series,
    group_keys: pd.Series,
    group_positions: tuple[np.ndarray, ...],
    window: int,
    statistic: str,
) -> pd.Series:
    del group_keys
    numeric = _numeric(values)
    source = numeric.to_numpy(dtype="float64", copy=False)
    output = np.full(len(numeric), np.nan, dtype="float64")
    for positions in group_positions:
        output[positions] = _rolling_window_statistic_array(
            source[positions],
            window,
            statistic,
        )
    return pd.Series(output, index=values.index, name=values.name)


def _rolling_window_statistic_array(
    values: np.ndarray,
    window: int,
    statistic: str,
) -> np.ndarray:
    output = np.full(len(values), np.nan, dtype="float64")
    if window <= 0 or len(values) < window:
        return output
    windows = np.lib.stride_tricks.sliding_window_view(values, window_shape=window)
    valid_windows = np.isfinite(windows).all(axis=1)
    if not valid_windows.any():
        return output
    valid_values = windows[valid_windows]
    if statistic == "std":
        window_values = _window_std_values(valid_values)
    elif statistic == "skew":
        window_values = _window_skew_values(valid_values)
    elif statistic == "kurt":
        window_values = _window_kurt_values(valid_values)
    else:
        raise ValueError(f"unsupported_rolling_statistic: {statistic}")
    values_out = np.full(len(windows), np.nan, dtype="float64")
    values_out[valid_windows] = window_values
    output[window - 1 :] = values_out
    return output


def _window_std_values(windows: np.ndarray) -> np.ndarray:
    counts = windows.shape[1]
    if counts < 2:
        return np.full(len(windows), np.nan, dtype="float64")
    centered, scale, varies = _scaled_centered_windows(windows)
    variance = np.sum(centered * centered, axis=1) / (counts - 1)
    values = np.zeros(len(windows), dtype="float64")
    values[varies] = np.sqrt(variance[varies]) * scale[varies]
    return values


def _rolling_rms_array(values: np.ndarray, window: int) -> np.ndarray:
    output = np.full(len(values), np.nan, dtype="float64")
    if window <= 0 or len(values) < window:
        return output
    windows = np.lib.stride_tricks.sliding_window_view(values, window_shape=window)
    valid_windows = np.isfinite(windows).all(axis=1)
    if not valid_windows.any():
        return output
    valid_values = windows[valid_windows]
    scale = np.max(np.abs(valid_values), axis=1)
    scale = np.where(scale == 0, 1.0, scale)
    window_values = np.sqrt(np.mean((valid_values / scale[:, None]) ** 2, axis=1))
    window_values *= scale
    values_out = np.full(len(windows), np.nan, dtype="float64")
    values_out[valid_windows] = window_values
    output[window - 1 :] = values_out
    return output


def _window_skew_values(windows: np.ndarray) -> np.ndarray:
    counts = windows.shape[1]
    if counts < 3:
        return np.full(len(windows), np.nan, dtype="float64")
    centered, _, varies = _scaled_centered_windows(windows)
    moment_2 = np.mean(centered * centered, axis=1)
    moment_3 = np.mean(centered * centered * centered, axis=1)
    valid = varies
    values = np.full(len(windows), np.nan, dtype="float64")
    if not valid.any():
        return values
    denominator = np.power(moment_2[valid], 1.5)
    with np.errstate(divide="ignore", invalid="ignore"):
        values[valid] = (
            np.sqrt(counts * (counts - 1))
            / (counts - 2)
            * (moment_3[valid] / denominator)
        )
    values[~np.isfinite(values)] = np.nan
    return values


def _window_kurt_values(windows: np.ndarray) -> np.ndarray:
    counts = windows.shape[1]
    if counts < 4:
        return np.full(len(windows), np.nan, dtype="float64")
    centered, _, varies = _scaled_centered_windows(windows)
    moment_2 = np.mean(centered * centered, axis=1)
    moment_4 = np.mean(centered**4, axis=1)
    valid = varies
    values = np.full(len(windows), np.nan, dtype="float64")
    if not valid.any():
        return values
    with np.errstate(divide="ignore", invalid="ignore"):
        values[valid] = (
            (counts - 1)
            / ((counts - 2) * (counts - 3))
            * (
                (counts + 1)
                * (moment_4[valid] / (moment_2[valid] * moment_2[valid]) - 3.0)
                + 6.0
            )
        )
    values[~np.isfinite(values)] = np.nan
    return values


def rolling_drawdown(
    values: pd.Series,
    group_positions: tuple[np.ndarray, ...],
    window: int,
) -> pd.Series:
    numeric = _numeric(values)
    source = numeric.to_numpy(dtype="float64", copy=False)
    output = np.full(len(numeric), np.nan, dtype="float64")
    for positions in group_positions:
        output[positions] = rolling_drawdown_array(source[positions], window)
    return pd.Series(output, index=values.index, name=values.name)


def rolling_drawdown_array(values: np.ndarray, window: int) -> np.ndarray:
    output = np.full(len(values), np.nan, dtype="float64")
    if window <= 0:
        return output
    if len(values) < window:
        return output
    windows = np.lib.stride_tricks.sliding_window_view(values, window_shape=window)
    valid_windows = np.isfinite(windows).all(axis=1)
    if not valid_windows.any():
        return output
    valid_values = windows[valid_windows]
    peaks = np.maximum.accumulate(valid_values, axis=1)
    denominators = np.where(peaks == 0.0, np.nan, peaks)
    with np.errstate(divide="ignore", invalid="ignore"):
        drawdowns = valid_values / denominators - 1.0
    drawdowns[~np.isfinite(drawdowns)] = np.nan
    all_nan = np.isnan(drawdowns).all(axis=1)
    minima = np.full(len(drawdowns), np.nan, dtype="float64")
    if (~all_nan).any():
        minima[~all_nan] = np.nanmin(drawdowns[~all_nan], axis=1)
    window_values = np.full(len(windows), np.nan, dtype="float64")
    window_values[valid_windows] = minima
    output[window - 1 :] = window_values
    return output


def _numeric(values: pd.Series) -> pd.Series:
    return strict_finite_numeric_series(
        values,
        error="invalid_price_volume_numeric_values",
        allow_null=True,
    ).astype("float64")


def _scaled_centered_windows(
    windows: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    scale = np.max(np.abs(windows), axis=1, keepdims=True)
    scale = np.where(scale == 0, 1.0, scale)
    normalized = windows / scale
    centered = normalized - normalized.mean(axis=1, keepdims=True)
    centered_scale = np.max(np.abs(centered), axis=1, keepdims=True)
    varies = np.min(windows, axis=1) != np.max(windows, axis=1)
    unit_scale = (scale * np.where(centered_scale == 0, 1.0, centered_scale))[:, 0]
    centered_scale = np.where(centered_scale == 0, 1.0, centered_scale)
    return centered / centered_scale, unit_scale, varies
