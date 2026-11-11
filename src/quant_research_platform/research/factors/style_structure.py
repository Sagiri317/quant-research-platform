"""Historical high-minus-low style returns on lagged, observed exposures."""

from math import ceil

import numpy as np
import pandas as pd

from quant_research_platform.compute.dataframe.cross_section import (
    _select_quantile_boundary,
)
from quant_research_platform.contracts.factors import FactorDefinition, FactorSet

STYLES = {
    "size": "log_total_market_cap",
    "market_beta": "beta_252d",
    "momentum": "momentum_252_20",
    "residual_volatility": "residual_volatility_60d",
    "illiquidity": "amihud_60d",
}
RECIPES = {
    f"style_{style}_return_trend_{window}d": (style, None, window)
    for style in STYLES
    for window in (20, 60)
}
RECIPES.update(
    {
        "style_size_momentum_correlation_60d": ("size", "momentum", 60),
        "style_volatility_illiquidity_correlation_60d": (
            "residual_volatility",
            "illiquidity",
            60,
        ),
    }
)
FACTOR_UNITS = {
    name: "dimensionless" if recipe[1] else "decimal_return"
    for name, recipe in RECIPES.items()
}


def factor_definitions():
    return tuple(
        FactorDefinition(
            factor_id=name,
            market_id="a_share",
            family="style_structure",
            factor_set=FactorSet.STANDARD,
            input_columns=(
                "period_return",
                "is_a_share_scope",
                *(STYLES[s] for s in recipe[:2] if s),
            ),
            output_column=name,
            required_input_data_id="style_structure_factors",
            required_windows=(recipe[2] + 1,),
            lookback_window=recipe[2],
            description=(
                "Market state: trailing sum of equal-weight high-minus-low "
                "quintile returns on prior-session style exposures; paired "
                "style-return correlation for correlation columns. Ties retained "
                "at selection boundaries; each style has its own finite domain."
            ),
            timing_semantics="same_session_close_available_at_next_session",
            null_policy="full_session_span_with_75_percent_valid_returns",
        )
        for name, recipe in RECIPES.items()
    )


def calculate(panel, *, factor_columns, history_keys=("asset_id",)):
    frame = panel.sort_values([*history_keys, "timestamp"], kind="stable").reset_index(
        drop=True
    )
    requested = {
        style for name in factor_columns for style in RECIPES[name][:2] if style
    }
    own = frame.groupby(list(history_keys), sort=False, dropna=False)
    prior = own[[STYLES[s] for s in requested]].shift()
    prior_scope = own.is_a_share_scope.shift().eq(True)
    returns = frame.period_return
    daily = pd.DataFrame(index=pd.DatetimeIndex(sorted(frame.timestamp.unique())))
    time_groups = frame.groupby("timestamp", sort=True).indices
    for style in sorted(requested):
        values = prior[STYLES[style]].where(prior_scope)
        records = {}
        for date, indices in time_groups.items():
            exposure = values.iloc[indices].dropna()
            if len(exposure) < 2 or exposure.nunique() < 2:
                records[date] = np.nan
                continue
            top = exposure.index[
                _select_quantile_boundary(exposure, 0.2, ascending=False)
            ]
            bottom = exposure.index[
                _select_quantile_boundary(exposure, 0.2, ascending=True)
            ]
            # Missing returns do not replace exposure-selected holdings.
            records[date] = returns.loc[top].mean(skipna=False) - returns.loc[
                bottom
            ].mean(skipna=False)
        daily[style] = pd.Series(records, dtype="Float64")
    out = frame.loc[:, ["asset_id", "timestamp"]].copy()
    for name in factor_columns:
        left, right, window = RECIPES[name]
        rolling = daily[left].rolling(window, min_periods=ceil(0.75 * window))
        value = rolling.corr(daily[right]) if right else rolling.sum()
        value.iloc[: window - 1] = np.nan
        out[name] = frame.timestamp.map(value)
    return out
