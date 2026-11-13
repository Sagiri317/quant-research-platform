"""Cross-sectional targets and maturity from their actual return dependencies."""

import numpy as np
import pandas as pd

from quant_research_platform.compute.dataframe.cross_section import (
    _industry_group_codes,
    _mad_winsorize,
    _neutralize_cross_section,
    _stable_zscore,
    _winsorize,
)
from quant_research_platform.contracts.risk import FactorRiskModel
from quant_research_platform.contracts.validation import (
    strict_bool_series,
    strict_finite_numeric_series,
    strict_timestamp_series,
)
from quant_research_platform.research.risk import FactorRiskState


def a_share_model_target(
    frame: pd.DataFrame,
    *,
    transform: str = "date_zscore",
    clip: str = "none",
    label_column: str = "forward_return_20d",
    availability_column: str | None = None,
    risk_model: FactorRiskModel | None = None,
    risk_sessions: dict | None = None,
) -> pd.DataFrame:
    """Transform an explicit return label; no predictive model is trained."""
    if availability_column is None:
        availability_column = f"{label_column}_available_at"
    values = strict_finite_numeric_series(
        frame[label_column],
        error="invalid_a_share_model_target_label",
        allow_null=True,
    ).astype("float64")
    eligible = strict_bool_series(
        frame["eligible"], error="invalid_a_share_eligibility", allow_null=True
    ).fillna(False)
    scoped = frame.loc[eligible].copy()
    scoped[label_column] = values.loc[eligible]
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
    # Clipping can use returns whose rows are later excluded from risk regression.
    dependencies = scoped[label_column].notna().reindex(frame.index, fill_value=False)
    if clip == "mad":
        scoped[label_column] = _mad_winsorize(scoped[label_column], groups)
    elif clip == "quantile_1_99":
        scoped[label_column] = _winsorize(scoped[label_column], groups)
    if transform == "industry_size_residual":
        scoped["industry_level1_code"] = _industry_group_codes(scoped)
        scoped = _neutralize_cross_section(
            scoped,
            factor_columns=(label_column,),
            industry_column="industry_level1_code",
            size_column="log_total_market_cap",
            rescale=False,
            time_groups=groups,
        )
    risk_model = FactorRiskModel() if risk_model is None else risk_model
    state = (
        FactorRiskState(
            style_columns=risk_model.style_columns, config=risk_model.config
        )
        if transform == "risk_residual"
        else None
    )
    target = pd.Series(np.nan, index=frame.index, dtype="float64")
    for positions in groups.values():
        group = scoped[label_column].iloc[positions]
        if state is not None:
            cross = scoped.iloc[positions]
            fit = state.fit_return_session(
                cross.assign(asset_return=cross[label_column]),
                prepared=(risk_sessions or {}).get(
                    pd.Timestamp(cross.timestamp.iloc[0])
                ),
            )
            if fit is not None:
                target.loc[cross.index] = cross.asset_id.map(fit.residuals).to_numpy()
        elif transform in {"raw", "industry_size_residual"}:
            target.loc[group.index] = group
        elif group.notna().sum() >= 2:
            target.loc[group.index] = (
                _stable_zscore(group)
                if transform == "date_zscore"
                else group.rank(method="average", pct=True)
            )
    if clip == "none":
        dependencies = target.notna()
    clocks = strict_timestamp_series(
        frame[availability_column].where(dependencies),
        error="invalid_model_target_label_availability",
        allow_null=True,
    )
    if transform != "raw" or clip != "none":
        unknown = (
            (dependencies & clocks.isna())
            .groupby(frame.timestamp, sort=False)
            .transform("any")
        )
        clocks = (
            clocks.groupby(frame.timestamp, sort=False)
            .transform("max")
            .where(target.notna() & ~unknown)
        )
    return pd.DataFrame({"model_target": target, "model_target_available_at": clocks})
