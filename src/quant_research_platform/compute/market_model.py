"""Batch rolling market regressions and residual volatility on paired histories."""

from math import ceil

import numpy as np

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
