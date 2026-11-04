"""Cross-sectional training targets from future returns and known exposures."""

import numpy as np
import pandas as pd

from quant_research_platform.compute.dataframe.cross_section import _stable_zscore
from quant_research_platform.contracts.validation import (
    strict_bool_series,
    strict_finite_numeric_series,
)
from quant_research_platform.research.features.transforms import (
    _industry_group_codes,
    _mad_winsorize,
    _neutralize_cross_section,
    _winsorize,
)
from quant_research_platform.research.protocols.a_share import RISK_STYLE_COLUMNS
from quant_research_platform.research.risk import FactorRiskState


def a_share_model_target(
    frame: pd.DataFrame, *, transform: str = "date_zscore", clip: str = "none"
) -> pd.Series:
    values = strict_finite_numeric_series(
        frame["forward_return_20d"],
        error="invalid_a_share_model_target_label",
        allow_null=True,
    ).astype("float64")
    eligible = strict_bool_series(
        frame["eligible"], error="invalid_a_share_eligibility"
    )
    scoped = frame.loc[eligible].copy()
    scoped["forward_return_20d"] = values.loc[eligible]
    if transform == "industry_size_residual":
        sizes = strict_finite_numeric_series(
            scoped["log_total_market_cap"],
            error="invalid_neutralization_size",
            allow_null=True,
        )
        scoped = scoped.loc[
            _industry_group_codes(scoped).notna() & sizes.notna()
        ].copy()
    groups = scoped.groupby("timestamp", sort=False).indices
    if clip == "mad":
        scoped["forward_return_20d"] = _mad_winsorize(scoped.forward_return_20d, groups)
    elif clip == "quantile_1_99":
        scoped["forward_return_20d"] = _winsorize(scoped.forward_return_20d, groups)
    if transform == "risk_residual":
        state = FactorRiskState(style_columns=RISK_STYLE_COLUMNS)
        target = pd.Series(np.nan, index=frame.index, dtype="float64")
        for positions in groups.values():
            cross = scoped.iloc[positions]
            fit = state.fit_return_session(
                cross.rename(columns={"forward_return_20d": "asset_return"})
            )
            if fit is not None:
                target.loc[cross.index] = cross.asset_id.map(fit.residuals).to_numpy()
        return target
    if transform == "industry_size_residual":
        scoped["industry_level1_code"] = _industry_group_codes(scoped)
        scoped = _neutralize_cross_section(
            scoped,
            factor_columns=("forward_return_20d",),
            industry_column="industry_level1_code",
            size_column="log_total_market_cap",
            rescale=False,
            time_groups=groups,
        )
    target = pd.Series(np.nan, index=frame.index, dtype="float64")
    for positions in groups.values():
        group = scoped.forward_return_20d.iloc[positions]
        valid = group.dropna()
        if transform in {"raw", "industry_size_residual"}:
            target.loc[group.index] = group
        elif len(valid) >= 2:
            target.loc[group.index] = (
                _stable_zscore(group)
                if transform == "date_zscore"
                else group.rank(method="average", pct=True)
            )
    return target
