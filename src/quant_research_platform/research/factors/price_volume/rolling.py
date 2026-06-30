"""Rolling-window kernels for price-volume factors."""

from __future__ import annotations

import numpy as np
import pandas as pd

# QRP semantic policy: skew/kurt are undefined when window moment_2
# (mean squared centered return) is near zero. std remains sample std.
_ROLLING_MOMENT_NEAR_ZERO_VARIANCE = 1e-12
def _rolling_aggregate(
    values: pd.Series,
    grouped_values: pd.core.groupby.SeriesGroupBy,
    group_positions: tuple[np.ndarray, ...],
    window: int,
    aggregation: str,
) -> pd.Series:
    del grouped_values
    numeric = pd.to_numeric(values, errors="coerce")
    source = numeric.to_numpy(dtype="float64", copy=False)
    output = np.full(len(values), np.nan, dtype="float64")
    for positions in group_positions:
        output[positions] = _rolling_aggregate_array(
            source[positions],
            window,
            aggregation,
        )
    return pd.Series(output, index=values.index, name=values.name)


def _rolling_aggregate_array(
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
        window_values = np.std(valid_values, axis=1, ddof=1)
    else:
        raise ValueError(f"unsupported_rolling_aggregation: {aggregation}")
    values_out = np.full(len(windows), np.nan, dtype="float64")
    values_out[valid_windows] = window_values
    output[window - 1 :] = values_out
    return output


def _rolling_rms(
    values: pd.Series,
    group_keys: pd.Series,
    group_positions: tuple[np.ndarray, ...],
    window: int,
) -> pd.Series:
    squared = values.pow(2)
    mean_squared = _rolling_aggregate(
        squared,
        squared.groupby(group_keys, sort=False),
        group_positions,
        window,
        "mean",
    )
    return np.sqrt(mean_squared)


def _rolling_std(
    values: pd.Series,
    group_keys: pd.Series,
    group_positions: tuple[np.ndarray, ...],
    window: int,
) -> pd.Series:
    return _rolling_window_statistic(values, group_keys, group_positions, window, "std")


def _rolling_correlation(
    left: pd.Series,
    right: pd.Series,
    group_positions: tuple[np.ndarray, ...],
    window: int,
) -> pd.Series:
    left_values = pd.to_numeric(left, errors="coerce").to_numpy(
        dtype="float64",
        copy=False,
    )
    right_values = pd.to_numeric(right, errors="coerce").to_numpy(
        dtype="float64",
        copy=False,
    )
    output = np.full(len(left), np.nan, dtype="float64")
    for positions in group_positions:
        output[positions] = _rolling_correlation_array(
            left_values[positions],
            right_values[positions],
            window,
        )
    return pd.Series(output, index=left.index, name=left.name)


def _rolling_correlation_array(
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
    left_centered = valid_left - valid_left.mean(axis=1, keepdims=True)
    right_centered = valid_right - valid_right.mean(axis=1, keepdims=True)
    numerator = np.sum(left_centered * right_centered, axis=1)
    denominator = np.sqrt(
        np.sum(left_centered * left_centered, axis=1)
        * np.sum(right_centered * right_centered, axis=1)
    )
    values = np.full(len(valid_left), np.nan, dtype="float64")
    valid_denominator = denominator > _ROLLING_MOMENT_NEAR_ZERO_VARIANCE
    with np.errstate(divide="ignore", invalid="ignore"):
        values[valid_denominator] = (
            numerator[valid_denominator] / denominator[valid_denominator]
        )
    values[~np.isfinite(values)] = np.nan
    window_values = np.full(len(left_windows), np.nan, dtype="float64")
    window_values[valid_windows] = values
    output[window - 1 :] = window_values
    return output


def _rolling_skew(
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


def _rolling_kurt(
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
    numeric = pd.to_numeric(values, errors="coerce")
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
    centered = windows - windows.mean(axis=1, keepdims=True)
    variance = np.sum(centered * centered, axis=1) / (counts - 1)
    variance = np.maximum(variance, 0.0)
    values = np.sqrt(variance)
    return values


def _window_skew_values(windows: np.ndarray) -> np.ndarray:
    counts = windows.shape[1]
    if counts < 3:
        return np.full(len(windows), np.nan, dtype="float64")
    centered = windows - windows.mean(axis=1, keepdims=True)
    moment_2 = np.mean(centered * centered, axis=1)
    moment_3 = np.mean(centered * centered * centered, axis=1)
    valid = moment_2 > _ROLLING_MOMENT_NEAR_ZERO_VARIANCE
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
    centered = windows - windows.mean(axis=1, keepdims=True)
    moment_2 = np.mean(centered * centered, axis=1)
    moment_4 = np.mean(centered**4, axis=1)
    valid = moment_2 > _ROLLING_MOMENT_NEAR_ZERO_VARIANCE
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


def _rolling_drawdown(
    values: pd.Series,
    group_keys: pd.Series,
    group_positions: tuple[np.ndarray, ...],
    window: int,
) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    source = numeric.to_numpy(dtype="float64", copy=False)
    output = np.full(len(numeric), np.nan, dtype="float64")
    for positions in group_positions:
        output[positions] = _rolling_drawdown_array(source[positions], window)
    return pd.Series(output, index=values.index, name=values.name)


def _rolling_drawdown_array(values: np.ndarray, window: int) -> np.ndarray:
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
