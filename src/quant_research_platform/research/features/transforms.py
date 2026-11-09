"""Transform eligible cross sections while preserving missing values."""

from __future__ import annotations

from math import fsum

import numpy as np
import pandas as pd

from quant_research_platform.compute.dataframe.cross_section import (
    _industry_group_codes,
    _mad_winsorize,
    _stable_zscore,
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
    diagnostics: dict | None = None,
    risk_sessions: dict | None = None,
) -> pd.DataFrame:
    if len(transforms) != 1:
        raise ValueError("feature_recipe_must_be_exclusive")
    recipe = transforms[0]
    return a_share_feature_pipeline(
        frame,
        factor_columns=recipe.source_columns,
        recipe_id=recipe.recipe_id,
        clip=recipe.clip,
        industry_neutralize=recipe.industry_neutralize,
        size_neutralize=recipe.size_neutralize,
        size_column=recipe.size_column,
        center=recipe.center,
        scale=recipe.scale,
        diagnostics=diagnostics,
        output_groups=recipe.output_groups,
        risk_model=recipe.risk_model,
        risk_sessions=risk_sessions,
    )


def _neutralization_design(industries, sizes):
    """Cache grouped centered size coordinates for a shared valid domain."""
    codes, _ = pd.factorize(industries, sort=False)
    first = np.unique(codes, return_index=True)[1]
    centered = None
    xx = None
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        if sizes is not None:
            sizes = np.asarray(sizes, dtype="float64")
            centered = sizes - sizes[first[codes]]
            centered -= (
                pd.Series(centered)
                .groupby(codes, sort=False)
                .transform("mean")
                .to_numpy()
            )
            xx = fsum(pd.Series(centered * centered).groupby(codes, sort=False).sum())
    return codes, first, centered, xx


def _industry_size_residuals(target, industries, sizes, *, rescale, design=None):
    """Project onto industry intercepts and one within-industry size slope."""
    target = np.asarray(target, dtype="float64")
    codes, first, x, xx = (
        _neutralization_design(industries, sizes) if design is None else design
    )
    y = target - target[first[codes]]
    y -= pd.Series(y).groupby(codes, sort=False).transform("mean").to_numpy()
    residual = y
    if x is not None:
        if xx == 0:
            return np.full(len(target), np.nan)
        xy = fsum(pd.Series(x * y).groupby(codes, sort=False).sum())
        residual = y - x * (xy / xx)
    magnitude = np.max(np.abs(residual))
    # Compensated group sums keep roundoff independent of group replication.
    # Judge an exact fit at each industry's own scale, not the largest industry.
    scale = pd.Series(np.abs(y)).groupby(codes, sort=False).transform("max").to_numpy()
    if np.all(np.abs(residual) <= 8 * np.finfo(float).eps * scale):
        return np.zeros(len(target))
    return residual / magnitude if rescale else residual


def _neutralize_cross_section(
    frame,
    *,
    factor_columns,
    industry_column,
    size_column,
    minimum_observations=100,
    rescale,
    industry_neutralize=True,
    size_neutralize=True,
    time_groups=None,
):
    required = {"timestamp", *factor_columns}
    if industry_neutralize:
        required.add(industry_column)
    if not required.issubset(frame.columns):
        raise ValueError("neutralization_columns_invalid")
    if type(minimum_observations) is not int or minimum_observations < 1:
        raise ValueError("neutralization_minimum_observations_invalid")
    if not industry_neutralize and not size_neutralize:
        return frame
    result = frame.copy()
    for column in factor_columns:
        result[column] = np.nan
    if time_groups is None:
        time_groups = frame.groupby("timestamp", sort=False).indices
    for positions in time_groups.values():
        cross = frame.iloc[positions]
        industry = (
            cross[industry_column]
            if industry_neutralize
            else pd.Series(0, index=cross.index)
        )
        valid_industry = industry.notna()
        if not valid_industry.any():
            continue
        size = None
        if size_neutralize and any(factor != size_column for factor in factor_columns):
            if size_column not in cross:
                raise ValueError("neutralization_size_column_required")
            size = strict_finite_numeric_series(
                cross.loc[valid_industry, size_column],
                error="invalid_neutralization_size",
                allow_null=True,
            )
        designs = {}
        for factor in factor_columns:
            # A size source is never regressed on itself. With no industry step,
            # its value passes through to the separately selected centering step.
            use_size = size_neutralize and factor != size_column
            if not industry_neutralize and not use_size:
                result.loc[cross.index, factor] = cross[factor].to_numpy(
                    dtype="float64", na_value=np.nan
                )
                continue
            values = strict_finite_numeric_series(
                cross.loc[valid_industry, factor],
                error=f"invalid_neutralization_factor: {factor}",
                allow_null=True,
            )
            valid = values.notna()
            if use_size:
                valid &= size.notna()
            indices = values.index[valid]
            groups = industry.loc[indices]
            parameters = groups.nunique() + use_size
            if len(indices) < max(minimum_observations, 5 * parameters):
                continue
            target = values.loc[indices].to_numpy(dtype="float64", copy=False)
            sizes = size.loc[indices].to_numpy() if use_size else None
            key = (use_size, valid.to_numpy(dtype=bool).tobytes())
            if key not in designs:
                designs[key] = _neutralization_design(groups, sizes)
            result.loc[indices, factor] = _industry_size_residuals(
                target, groups, sizes, rescale=rescale, design=designs[key]
            )
    return result


def _center_scale(values, *, center, scale, weights=None):
    if center and scale and weights is None:
        return _stable_zscore(values)
    if not center and not scale:
        return values.astype("float64")
    observed = values.dropna()
    output = pd.Series(np.nan, index=values.index, dtype="float64")
    if not len(observed) or (scale and len(observed) < 2):
        return output
    shifted = observed - observed.iloc[0]
    centered = shifted - shifted.mean()
    targets = centered if center else observed
    if center and weights is not None:
        targets = shifted - np.average(shifted, weights=weights.loc[observed.index])
    if scale:
        std = centered.std(ddof=0)
        if std == 0:
            return output
        targets = targets / std
    output.loc[observed.index] = targets
    return output


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
    industry_neutralize = industry_neutralize and "processed" in output_groups
    size_neutralize = size_neutralize and "processed" in output_groups
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
