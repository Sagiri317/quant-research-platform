from decimal import Decimal, localcontext

import numpy as np
import pandas as pd
import pytest

from quant_research_platform.research.portfolios.risk import (
    FactorRiskConfig,
    FactorRiskState,
)


@pytest.mark.parametrize("half_life", [1.0, 63.0])
@pytest.mark.parametrize("magnitude", [0.01, 0.05])
def test_actual_risk_estimate_retains_tail_after_latest_missing_session(
    half_life, magnitude
):
    config = FactorRiskConfig(
        factor_window=3,
        specific_window=3,
        weight_column="float_market_cap",
        minimum_cross_section=10,
        minimum_factor_dates=2,
        minimum_specific_dates=2,
        factor_half_life=half_life,
        specific_half_life=half_life,
        annualization=1.0,
        diagonal_shrinkage=0.0,
    )
    state = FactorRiskState(style_columns=(), config=config)
    frame = pd.DataFrame(
        {
            "listing_date": pd.Timestamp("2000-01-01"),
            "timestamp": pd.Timestamp("2020-01-01"),
            "asset_id": [f"a{i}" for i in range(10)],
            "industry_standard_id": "CSRC2012",
            "industry_level1_code": "A",
            "float_market_cap": 1.0,
            "asset_return": np.linspace(0.5 * magnitude, 1.5 * magnitude, 10),
        }
    )
    fitted = state.fit_return_session(frame)
    state.observe_return_session(frame)
    state.observe_return_session(
        frame.assign(timestamp=pd.Timestamp("2020-01-02"), asset_return=0.0)
    )
    state.observe_return_session(
        frame.assign(timestamp=pd.Timestamp("2020-01-03"), asset_return=np.nan)
    )
    result = state.estimate()
    with localcontext() as ctx:
        ctx.prec = 100
        tail = Decimal(2) ** (-Decimal(1) / Decimal.from_float(half_life))
        w = tail / (1 + tail)
        expected_cov = float(
            w * (1 - w) * Decimal.from_float(float(fitted.factor_returns.iloc[0])) ** 2
        )
        expected_specific = float(
            w * Decimal.from_float(float(fitted.residuals.iloc[0])) ** 2
        )
    assert expected_cov > 0 and expected_specific > 0
    assert result.factor_covariance.iloc[0, 0] == pytest.approx(
        expected_cov, rel=1e-12, abs=0
    )
    assert result.specific_variance.iloc[0] == pytest.approx(
        expected_specific, rel=1e-12, abs=0
    )
    assert np.isfinite(result.factor_covariance).all().all()


@pytest.mark.parametrize("half_life", [1.0, 10.0, 63.0])
def test_ewma_covariance_is_invariant_to_common_missing_age(half_life):
    from quant_research_platform.research.portfolios.risk import _ewma_covariance

    arguments = dict(
        values=np.array([[0.01, -0.005], [0.0, 0.0]]),
        half_life=half_life,
        diagonal_shrinkage=0.2,
        annualization_factor=252.0,
        columns=("a", "b"),
    )
    first = _ewma_covariance(ages=np.array([2, 1]), **arguments)
    shifted = _ewma_covariance(ages=np.array([2002, 2001]), **arguments)
    pd.testing.assert_frame_equal(first.isna(), shifted.isna())
    pd.testing.assert_frame_equal(first, shifted, check_exact=False, rtol=1e-12, atol=0)


def test_nonfinite_risk_fails_explicitly():
    from quant_research_platform.research.portfolios.risk import _ewma_covariance

    with pytest.raises(ValueError, match="factor_risk_covariance_non_finite"):
        _ewma_covariance(
            np.array([[np.inf], [-np.inf]]),
            ages=np.array([1, 0]),
            half_life=63.0,
            diagonal_shrinkage=0.2,
            annualization_factor=252.0,
            columns=("a",),
        )
