"""Historical return residuals using the common industry and style regression."""

from math import ceil

import numpy as np
import pandas as pd

from quant_research_platform.research.filtering import a_share_eligibility
from quant_research_platform.research.risk import FactorExposureSession, FactorRiskState

RESIDUAL_FACTOR_WINDOWS = {
    "risk_residual_reversal_5d": 5,
    "risk_residual_momentum_20d": 20,
    "risk_residual_volatility_60d": 60,
}


def calculate_residual_factors(
    panel,
    *,
    style_columns,
    factor_columns,
    history_keys,
    min_fresh_fraction,
):
    """Fit today's return on prior-session facts, then aggregate past residuals."""
    frame = panel.sort_values([*history_keys, "timestamp"], kind="stable").copy()
    frame["eligible"] = a_share_eligibility(frame, "a_share_all")
    state = FactorRiskState(style_columns=style_columns)
    for _, cross in frame.groupby("timestamp", sort=True):
        prepared = state.prepare_exposure_session(cross)
        frame.loc[cross.index, list(style_columns)] = prepared.frame.reindex(
            cross.index
        )[list(style_columns)]
    groups = frame.groupby(list(history_keys), sort=False, dropna=False)
    prior_columns = [
        "industry_standard_id",
        "industry_level1_code",
        "total_market_cap",
        "eligible",
        *style_columns,
    ]
    prior = groups[prior_columns].shift()
    prior["eligible"] = prior.eligible.eq(True)
    prior["asset_id"] = frame.asset_id
    prior["timestamp"] = frame.timestamp
    prior["asset_return"] = frame.period_return.where(frame.is_trading.eq(True))
    residuals = pd.Series(np.nan, index=frame.index, dtype="float64")
    for _, cross in prior.groupby("timestamp", sort=True):
        prepared = FactorExposureSession.from_preprocessed(
            state.prepare_exposures(cross)
        )
        fitted = state.fit_return_session(cross, prepared=prepared)
        if fitted is not None:
            residuals.loc[cross.index] = cross.asset_id.map(fitted.residuals)
    frame["_residual"] = residuals
    groups = frame.groupby(list(history_keys), sort=False, dropna=False)
    age = groups.cumcount()
    output = frame.loc[:, ["asset_id", "timestamp"]].copy()
    for column in factor_columns:
        window = RESIDUAL_FACTOR_WINDOWS[column]
        rolling = groups._residual.rolling(
            window, min_periods=ceil(window * min_fresh_fraction)
        )
        values = rolling.std(ddof=0) if "volatility" in column else rolling.sum()
        values = values.droplevel(list(range(len(history_keys)))).reindex(frame.index)
        output[column] = (-values if "reversal" in column else values).where(
            age >= window
        )
    return output
