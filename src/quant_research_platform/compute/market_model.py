"""Batch rolling market regressions and residual volatility on paired histories."""

from math import ceil

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from quant_research_platform.compute.dataframe.cross_section import _centered_values


def rolling_market_regression(returns, markets, *, min_fresh_fraction):
    """Return OLS slopes and latest residuals on each asset's finite paired domain."""
    paired = np.isfinite(markets) & np.isfinite(returns)
    counts = paired.sum(axis=1)
    enough = counts >= max(2, ceil(min_fresh_fraction * returns.shape[1]))
    x = _centered_values(markets, paired)
    y = _centered_values(returns, paired)
    variance = np.vecdot(x, x)
    slopes = np.divide(
        np.vecdot(x, y),
        variance,
        out=np.full(len(returns), np.nan),
        where=enough & (variance > 0),
    )
    slopes[~np.isfinite(slopes)] = np.nan
    residuals = np.where(paired[:, -1], y[:, -1] - slopes * x[:, -1], np.nan)
    return slopes, residuals


def rolling_residual_volatility(values, *, min_fresh_fraction):
    """Return population residual deviations when enough history is observed."""
    finite = np.isfinite(values)
    counts = finite.sum(axis=1)
    valid = counts >= ceil(min_fresh_fraction * values.shape[1])
    output = np.full(len(values), np.nan)
    values, finite, counts = values[valid], finite[valid], counts[valid]
    centered = _centered_values(values, finite)
    output[valid] = np.sqrt(np.vecdot(centered, centered) / counts)
    return output


def relative_downside_beta(
    returns, markets, *, window=252, min_observations=126, min_downside=50
):
    """Subtract the all-days slope from the below-window-mean market slope."""
    output = np.full(len(returns), np.nan)
    if len(returns) < window:
        return output
    y = sliding_window_view(np.asarray(returns, float), window)
    x = sliding_window_view(np.asarray(markets, float), window)
    known_market = np.isfinite(x)
    count = known_market.sum(axis=1)
    mean = np.divide(
        np.where(known_market, x, 0).sum(axis=1),
        count,
        out=np.full(len(x), np.nan),
        where=count > 0,
    )
    ordinary, _ = rolling_market_regression(
        y, x, min_fresh_fraction=min_observations / window
    )
    downside, _ = rolling_market_regression(
        np.where(x < mean[:, None], y, np.nan),
        x,
        min_fresh_fraction=min_downside / window,
    )
    output[window - 1 :] = downside - ordinary
    return output


def rolling_partial_slope(
    returns, signal, controls, *, window=252, min_observations=126
):
    """First OLS slope from shared-domain rolling covariances, with an intercept."""
    values = np.column_stack((returns, signal, *controls)).astype(float)
    values[~np.isfinite(values).all(axis=1)] = np.nan
    columns = [pd.Series(values[:, i]) for i in range(values.shape[1])]
    predictors = columns[1:]
    count = len(predictors)
    covariance = np.empty((len(values), count, count))
    response = np.empty((len(values), count))
    for i, left in enumerate(predictors):
        rolling = left.rolling(window, min_periods=min_observations)
        response[:, i] = rolling.cov(columns[0])
        for j in range(i + 1):
            covariance[:, i, j] = covariance[:, j, i] = rolling.cov(predictors[j])
    scale = np.sqrt(np.maximum(np.diagonal(covariance, axis1=1, axis2=2), 0))
    valid = np.isfinite(covariance).all(axis=(1, 2)) & (scale > 0).all(axis=1)
    indices = np.flatnonzero(valid)
    correlations = covariance[valid] / (scale[valid, :, None] * scale[valid, None, :])
    eigenvalues = np.linalg.eigvalsh(correlations)
    nonsingular = eigenvalues[:, 0] > 16 * np.finfo(float).eps * eigenvalues[:, -1]
    indices = indices[nonsingular]
    result = np.full(len(values), np.nan)
    coefficients = (
        np.linalg.solve(
            correlations[nonsingular],
            (response[indices] / scale[indices])[..., None],
        )[..., 0]
        / scale[indices]
    )
    result[indices] = coefficients[:, 0]
    return result
