"""Paired rolling regressions checked against hand-derived moments."""

import numpy as np

from quant_research_platform.compute.market_model import (
    rolling_market_regression,
    rolling_residual_volatility,
)


def test_rolling_market_moments_keep_pairing_and_incomplete_domains():
    markets = np.array([[0.01, 0.02, 0.03, 0.04]] * 4)
    returns = 2 * markets
    # This zero-mean disturbance is orthogonal to centered market returns.
    returns[0] += [0.01, -0.01, -0.01, 0.01]
    returns[1, 0] = np.nan
    returns[2, :2] = np.nan
    markets[3] = 0.01
    beta, latest = rolling_market_regression(returns, markets, min_fresh_fraction=0.75)
    np.testing.assert_allclose(beta, [2, 2, np.nan, np.nan], rtol=1e-14)
    np.testing.assert_allclose(latest, [0.01, 0, np.nan, np.nan], atol=1e-16)

    values = np.array(
        [
            [0.01, 0.02, 0.03, 0.04],
            [0.01, np.nan, 0.03, 0.05],
            [0, 0.01, np.nan, np.nan],
        ]
    )
    actual = rolling_residual_volatility(values, min_fresh_fraction=0.75)
    np.testing.assert_allclose(
        actual, [np.sqrt(1.25) / 100, np.sqrt(8 / 3) / 100, np.nan], rtol=1e-14
    )
