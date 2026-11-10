"""Numerical windows on sorted, contiguous histories."""

import numpy as np
import pandas as pd
from pandas.api.indexers import BaseIndexer


class HistoryWindow(BaseIndexer):
    """Bound native rolling kernels to each history's first position."""

    def __init__(self, history_start, window_size):
        super().__init__(window_size=window_size)
        self.history_start = history_start

    def get_window_bounds(
        self, num_values=0, min_periods=None, center=None, closed=None, step=None
    ):
        end = np.arange(1, num_values + 1, dtype=np.int64)
        return np.maximum(end - self.window_size, self.history_start), end


def path_efficiency(net_change, total_variation):
    """Signed displacement per total path variation; a flat path is undefined."""
    net_change, total_variation = np.broadcast_arrays(net_change, total_variation)
    return np.divide(
        net_change,
        total_variation,
        out=np.full(net_change.shape, np.nan),
        where=total_variation > 0,
    )


def quadratic_log_trend(prices):
    """Return endpoint log-price slope and curvature per RMS log return."""
    size = prices.shape[-1]
    x = np.arange(size, dtype=float)
    x -= x.mean()
    basis = np.column_stack((np.ones(size), x, x * x))
    logged = np.log(prices)
    coefficients = logged @ np.linalg.pinv(basis).T
    slope = coefficients[..., 1] + 2 * x[-1] * coefficients[..., 2]
    rms = np.sqrt(np.mean(np.diff(logged, axis=-1) ** 2, axis=-1))
    curvature = np.divide(
        2 * coefficients[..., 2],
        rms,
        out=np.full(rms.shape, np.nan),
        where=rms > 0,
    )
    return slope, curvature


def completed_month_return_acceleration(prices, timestamps):
    """Mean returns in the last six completed months minus the preceding six."""
    months = pd.DatetimeIndex(timestamps).to_period("M")
    observed = pd.Series(prices, index=months)
    closes = observed.groupby(level=0, sort=False).tail(1)
    axis = pd.period_range(months.min(), months.max(), freq="M")
    closes = closes.reindex(axis)
    returns = closes / closes.shift().where(closes.shift() > 0) - 1
    acceleration = (
        returns.rolling(6, min_periods=6).mean()
        - returns.shift(6).rolling(6, min_periods=6).mean()
    )
    return acceleration.reindex(months - 1).to_numpy()


def liquidity_beta(
    returns,
    shocks,
    *,
    market_returns=None,
    window=252,
    min_observations=126,
    min_changes=20,
):
    """Estimate historical sensitivity with explicit effective shock counts."""
    data = pd.DataFrame(
        {"x": np.asarray(shocks, float), "y": np.asarray(returns, float)}
    )
    if market_returns is not None:
        data["m"] = np.asarray(market_returns, float)
    data.loc[~np.isfinite(data).all(axis=1)] = np.nan
    x, y = data.x, data.y
    rolling = x.rolling(window, min_periods=min_observations)
    variance = rolling.var()
    covariance = rolling.cov(y)
    valid = data.x.notna() & data.x.ne(0)
    enough = valid.rolling(window, min_periods=1).sum().ge(min_changes)
    if market_returns is None:
        result = covariance / variance.where(variance.gt(0))
    else:
        market_variance = data.m.rolling(window, min_periods=min_observations).var()
        xm = rolling.cov(data.m)
        ym = y.rolling(window, min_periods=min_observations).cov(data.m)
        product = variance * market_variance
        determinant = product - xm * xm
        invertible = determinant.gt(16 * np.finfo(float).eps * product) & product.gt(0)
        result = (covariance * market_variance - ym * xm) / determinant.where(
            invertible
        )
    return result.where(enough).to_numpy()
