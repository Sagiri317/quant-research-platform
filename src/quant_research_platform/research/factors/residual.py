"""Historical return residuals using the common industry and style regression."""

from math import ceil

import numpy as np
import pandas as pd

from quant_research_platform.compute.dataframe.cross_section import leave_one_out_mean
from quant_research_platform.compute.dataframe.history import HistoryWindow
from quant_research_platform.research.filtering import a_share_eligibility
from quant_research_platform.research.risk import (
    FactorExposureSession,
    FactorRiskState,
    preprocess_risk_styles,
)

RESIDUAL_FACTOR_WINDOWS = {
    "risk_residual_reversal_5d": 5,
    "risk_residual_momentum_20d": 20,
    "risk_residual_volatility_60d": 60,
    "risk_residual_drawdown_20d": 20,
    "risk_residual_drawdown_60d": 60,
    "industry_residual_comovement_20d": 20,
    "industry_residual_comovement_60d": 60,
}


def _residual_drawdown(values, window):
    """Current drawdown of the additive residual path, including its zero origin."""
    values = np.asarray(values, dtype=float)
    known = np.isfinite(values)
    cumulative = np.r_[0.0, np.cumsum(np.where(known, values, 0.0))]
    # The origin of each window and every earlier cumulative point are peaks.
    peak = pd.Series(cumulative[:-1]).rolling(window).max().to_numpy()
    complete = pd.Series(known).rolling(window).sum().eq(window).to_numpy()
    return np.where(complete, np.maximum(peak - cumulative[1:], 0), np.nan)


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
    frame.loc[:, list(style_columns)] = preprocess_risk_styles(
        state.prepare_exposures(frame), style_columns=style_columns
    ).reindex(frame.index)[list(style_columns)]
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
    age = groups.cumcount()
    output = frame.loc[:, ["asset_id", "timestamp"]].copy()
    if any(
        column.startswith("industry_residual_comovement_") for column in factor_columns
    ):
        industry = frame.groupby(
            ["timestamp", "industry_standard_id", "industry_level1_code"], sort=False
        ).ngroup()
        peers = leave_one_out_mean(frame._residual, industry)
        history_start = (
            np.arange(len(frame), dtype=np.int64) - groups.cumcount().to_numpy()
        )
    for column in factor_columns:
        window = RESIDUAL_FACTOR_WINDOWS[column]
        if column.startswith("industry_residual_comovement_"):
            output[column] = (
                frame._residual.rolling(
                    HistoryWindow(history_start, window),
                    min_periods=ceil(window * min_fresh_fraction),
                )
                .corr(peers)
                .where(age >= window)
            )
            continue
        if "drawdown" in column:
            values = groups._residual.transform(
                lambda x, window=window: _residual_drawdown(x.to_numpy(), window)
            )
            output[column] = values.where(age >= window)
            continue
        rolling = groups._residual.rolling(
            window, min_periods=ceil(window * min_fresh_fraction)
        )
        values = rolling.std(ddof=0) if "volatility" in column else rolling.sum()
        values = values.droplevel(list(range(len(history_keys)))).reindex(frame.index)
        output[column] = (-values if "reversal" in column else values).where(
            age >= window
        )
    return output
