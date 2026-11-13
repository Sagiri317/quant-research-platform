"""Transform eligible cross sections while preserving missing values."""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant_research_platform.compute.dataframe.cross_section import (
    _center_scale,
    _industry_group_codes,
    _mad_winsorize,
    _neutralize_cross_section,
    _winsorize,
)
from quant_research_platform.contracts.features import (
    FeatureTransformSpec,
)
from quant_research_platform.contracts.risk import FactorRiskModel
from quant_research_platform.contracts.validation import strict_finite_numeric_series


def apply_feature_transforms(
    frame: pd.DataFrame,
    transforms: tuple[FeatureTransformSpec, ...],
    *,
    industry_column: str = "industry_level1_code",
    diagnostics: dict | None = None,
    risk_sessions: dict | None = None,
) -> pd.DataFrame:
    """Apply the already validated recipe to each eligible cross section."""
    if len(transforms) != 1:
        raise ValueError("feature_recipe_must_be_exclusive")
    recipe = transforms[0]
    factor_columns = recipe.source_columns
    size_column = recipe.size_column
    output_groups = recipe.output_groups
    risk_model = recipe.risk_model
    center, scale = recipe.center, recipe.scale
    industry_neutralize = recipe.industry_neutralize and "processed" in output_groups
    size_neutralize = recipe.size_neutralize and "processed" in output_groups
    if "eligible" not in frame:
        raise ValueError("feature_eligibility_required")
    universe = frame["eligible"]
    if not pd.api.types.is_bool_dtype(universe):
        raise ValueError("feature_eligibility_invalid")
    scoped = frame.loc[universe.fillna(False)].copy()
    if industry_neutralize and (
        industry_column not in scoped or "industry_standard_id" not in scoped
    ):
        raise ValueError("feature_industry_context_missing")
    needs_size = size_neutralize and any(
        factor != size_column for factor in factor_columns
    )
    if needs_size and size_column not in scoped:
        raise ValueError("feature_size_context_missing")
    inputs = tuple(
        dict.fromkeys((*factor_columns, *((size_column,) if needs_size else ())))
    )
    time_groups = scoped.groupby("timestamp", sort=False).indices
    winsorized = scoped.copy()
    mad_counts = variance_counts = None
    if diagnostics is not None:
        mad_counts = diagnostics.setdefault("mad_zero_by_input", {})
        variance_counts = diagnostics.setdefault("zero_variance_by_factor", {})
    for factor in inputs:
        if mad_counts is not None:
            mad_counts.setdefault(factor, 0)
        values = strict_finite_numeric_series(
            winsorized[factor],
            error=f"invalid_feature_factor: {factor}",
            allow_null=True,
        )
        if recipe.clip == "mad":
            values = _mad_winsorize(
                values, time_groups, zero_counts=mad_counts, column=factor
            )
        elif recipe.clip == "quantile_1_99":
            values = _winsorize(values, time_groups)
        winsorized[factor] = values
    if industry_neutralize:
        winsorized[industry_column] = _industry_group_codes(winsorized, industry_column)
    if risk_model is not None and "processed" in output_groups:
        from quant_research_platform.research.risk import FactorRiskState

        state = FactorRiskState(
            style_columns=risk_model.style_columns, config=risk_model.config
        )
        risk_weights = pd.Series(np.nan, index=scoped.index, dtype="float64")
        residualized = winsorized.copy()
        residualized.loc[:, list(factor_columns)] = np.nan
        for positions in time_groups.values():
            # Keep raw risk descriptors separate from clipped alpha responses.
            cross = scoped.iloc[positions]
            prepared = (risk_sessions or {}).get(pd.Timestamp(cross.timestamp.iloc[0]))
            if prepared is None:
                prepared = state.prepare_exposure_session(cross)
            risk_weights.loc[cross.index] = cross.asset_id.map(
                np.sqrt(
                    prepared.frame.set_index("asset_id")[
                        risk_model.config.weight_column
                    ]
                )
            )
            for factor in factor_columns:
                response = cross[["timestamp", "asset_id"]].assign(
                    asset_return=winsorized[factor].iloc[positions]
                )
                fitted = state.fit_return_session(response, prepared=prepared)
                if fitted is not None:
                    residualized.loc[cross.index, factor] = cross.asset_id.map(
                        fitted.residuals
                    )
    else:
        residualized = _neutralize_cross_section(
            winsorized,
            factor_columns=factor_columns,
            industry_column=industry_column,
            size_column=size_column,
            rescale=center and scale,
            industry_neutralize=industry_neutralize,
            size_neutralize=size_neutralize,
            time_groups=time_groups,
        )
    feature_values: dict[str, pd.Series] = {}
    result = frame.copy()
    for group_name in output_groups:
        group_frame = winsorized if group_name == "pre_neutral" else residualized
        for factor in factor_columns:
            values = group_frame[factor]
            suffix = ("pre_neutral_" if group_name == "pre_neutral" else "") + factor
            if variance_counts is not None:
                variance_counts.setdefault(suffix, 0)
            scoped_feature = pd.Series(np.nan, index=scoped.index, dtype="float64")
            for positions in time_groups.values():
                group = values.iloc[positions]
                observed = group.dropna()
                if (
                    scale
                    and variance_counts is not None
                    and len(observed) >= 2
                    and observed.eq(observed.iloc[0]).all()
                ):
                    variance_counts[suffix] += 1
                weights = (
                    risk_weights.loc[group.index]
                    if risk_model is not None and group_name == "processed" and center
                    else None
                )
                transformed = _center_scale(
                    group, center=center, scale=scale, weights=weights
                )
                scoped_feature.iloc[positions] = transformed
            feature_values["feature_" + suffix] = scoped_feature.reindex(result.index)
    return pd.concat((result, pd.DataFrame(feature_values, index=result.index)), axis=1)


def a_share_feature_pipeline(
    frame: pd.DataFrame,
    *,
    factor_columns: tuple[str, ...],
    recipe_id: str = "a_share_canonical",
    size_column: str = "log_total_market_cap",
    industry_column: str = "industry_level1_code",
    diagnostics: dict | None = None,
    clip: str | None = None,
    industry_neutralize: bool = True,
    size_neutralize: bool = True,
    center: bool = True,
    scale: bool = True,
    output_groups: tuple[str, ...] = ("processed",),
    risk_model: FactorRiskModel | None = None,
    risk_sessions: dict | None = None,
) -> pd.DataFrame:
    """Clip, neutralize, center and scale in that fixed order on eligible rows."""
    recipe = FeatureTransformSpec(
        recipe_id,
        factor_columns,
        clip,
        industry_neutralize,
        size_neutralize,
        center,
        scale,
        size_column=size_column,
        output_groups=output_groups,
        risk_model=risk_model,
    )
    return apply_feature_transforms(
        frame,
        (recipe,),
        industry_column=industry_column,
        diagnostics=diagnostics,
        risk_sessions=risk_sessions,
    )
