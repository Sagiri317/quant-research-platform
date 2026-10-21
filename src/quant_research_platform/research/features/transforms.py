"""Feature transform materialization for research datasets."""

from __future__ import annotations

from math import fsum

import numpy as np
import pandas as pd

from quant_research_platform.compute.dataframe.cross_section import _stable_zscore
from quant_research_platform.contracts.features import (
    FEATURE_PREFIXES,
    FeatureTransformSpec,
)
from quant_research_platform.contracts.models import _model_algorithm_facts
from quant_research_platform.contracts.validation import strict_finite_numeric_series


def canonical_feature_validity_columns(input_columns, declared_columns):
    """Resolve declared canonical families, with each source value before its flags."""
    declared = set(declared_columns)
    dependencies = []
    for column in input_columns:
        for prefix in reversed(FEATURE_PREFIXES):
            if column.startswith(prefix):
                suffix = column[len(prefix) :]
                value = FEATURE_PREFIXES[0] + suffix
                if value in declared and any(
                    companion + suffix in declared for companion in FEATURE_PREFIXES[1:]
                ):
                    dependencies.extend(
                        companion + suffix
                        for companion in FEATURE_PREFIXES
                        if companion + suffix in declared
                    )
                    break
    return tuple(dict.fromkeys(dependencies))


def feature_observation_columns(columns, declared_columns):
    """Return declared observation companions, without interpreting ordinary names."""
    declared = set(declared_columns)
    return tuple(
        dict.fromkeys(
            "feature_observed_" + column[len("feature_") :]
            for column in columns
            if column.startswith("feature_")
            and "feature_observed_" + column[len("feature_") :] in declared
        )
    )


def _feature_observation_masks(frame, flags):
    for flag in flags:
        if flag not in frame:
            raise ValueError(f"feature_observation_column_missing: {flag}")
        observed = strict_finite_numeric_series(
            frame[flag], error=f"invalid_feature_observation_flag: {flag}"
        )
        if not observed.isin((0, 1)).all():
            raise ValueError(f"invalid_feature_observation_flag: {flag}")
        column = "feature_" + flag[len("feature_observed_") :]
        yield column, observed.eq(1)


def mask_unobserved_features(
    frame: pd.DataFrame, columns, declared_columns
) -> pd.DataFrame:
    """Exclude technical fill values using the full declared input schema."""
    flags = feature_observation_columns(columns, declared_columns)
    if not flags:
        return frame
    result = frame.copy()
    for column, observed in _feature_observation_masks(frame, flags):
        values = frame[column]
        if (
            isinstance(values.dtype, np.dtype)
            and values.dtype.kind in "iu"
            and (~observed).any()
        ):
            prefix = "UInt" if values.dtype.kind == "u" else "Int"
            nullable = f"{prefix}{values.dtype.itemsize * 8}"
            values = values.astype(nullable)
        result[column] = values.where(observed)
    return result


def model_input_validity_columns(input_columns, model_algorithm, declared_columns):
    """Own structural and direct-input validity for every model consumer."""
    direct = (
        ()
        if _model_algorithm_facts(model_algorithm).input_kind == "rank_ensemble_inputs"
        else input_columns
    )
    return tuple(
        dict.fromkeys(
            (
                *canonical_feature_validity_columns(input_columns, declared_columns),
                *feature_observation_columns(input_columns, declared_columns),
                *direct,
            )
        )
    )


def model_observed_input_mask(frame, input_columns, declared_columns):
    """Require usable consumed families and at least one original observation."""
    originals = tuple(
        dict.fromkeys(
            (
                canonical_feature_validity_columns((column,), declared_columns)
                or (column,)
            )[0]
            for column in input_columns
        )
    )
    flags = feature_observation_columns(originals, declared_columns)
    observed = {
        column: frame[column].notna() & flag
        for column, flag in _feature_observation_masks(frame, flags)
    }
    observed = pd.DataFrame(
        {
            column: observed[column] if column in observed else frame[column].notna()
            for column in originals
        },
        index=frame.index,
    )
    keep = observed.any(axis=1)
    canonical = canonical_feature_validity_columns(input_columns, declared_columns)
    for column in originals:
        if column not in canonical:
            continue
        usable = observed[column].copy()
        missing = "feature_missing_" + column[len("feature_") :]
        if missing in canonical:
            ordinary_missing = strict_finite_numeric_series(
                frame[missing], error=f"invalid_feature_missing_flag: {missing}"
            )
            observation = "feature_observed_" + column[len("feature_") :]
            if not ordinary_missing.isin((0, 1)).all() or (
                observation in canonical
                and (ordinary_missing.eq(1) & frame[observation].eq(1)).any()
            ):
                raise ValueError(f"invalid_feature_missing_flag: {missing}")
            usable |= ordinary_missing.eq(1)
        keep &= usable
    return keep


def apply_feature_transforms(
    frame: pd.DataFrame,
    transforms: tuple[FeatureTransformSpec, ...],
    *,
    diagnostics: dict | None = None,
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
        impute_missing=recipe.impute_missing,
        diagnostics=diagnostics,
        output_groups=recipe.output_groups,
    )


def _winsorize(
    values: pd.Series,
    grouped: pd.core.groupby.SeriesGroupBy,
) -> pd.Series:
    source = grouped.obj if hasattr(grouped, "obj") else values
    lower = np.full(len(values), np.nan)
    upper = np.full(len(values), np.nan)
    for positions in (
        grouped.indices if hasattr(grouped, "indices") else grouped
    ).values():
        lower[positions], upper[positions] = _quantile(
            source.iloc[positions], 0.01, 0.99
        )
    return values.clip(
        lower=pd.Series(lower, index=values.index),
        upper=pd.Series(upper, index=values.index),
    )


def _quantile(values: pd.Series, *quantiles: float) -> tuple[float, ...]:
    observed = values.dropna().to_numpy(dtype="float64")
    return (
        tuple(np.quantile(observed, quantiles))
        if len(observed)
        else (np.nan,) * len(quantiles)
    )


def _mad_winsorize(
    values: pd.Series,
    grouped: pd.core.groupby.SeriesGroupBy,
    *,
    zero_counts: dict[str, int] | None = None,
    column: str | None = None,
) -> pd.Series:
    source = grouped.obj if hasattr(grouped, "obj") else values
    lower = np.full(len(values), np.nan)
    upper = np.full(len(values), np.nan)
    for positions in (
        grouped.indices if hasattr(grouped, "indices") else grouped
    ).values():
        observed = source.iloc[positions].dropna().to_numpy(dtype="float64")
        if not len(observed):
            continue
        center = np.median(observed)
        mad = np.median(np.abs(observed - center))
        if mad == 0:
            if zero_counts is not None:
                zero_counts[column] += 1
            continue
        bound = 3 * 1.4826 * mad
        lower[positions], upper[positions] = center - bound, center + bound
    return values.clip(
        lower=pd.Series(lower, index=values.index),
        upper=pd.Series(upper, index=values.index),
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


def _industry_group_codes(
    frame: pd.DataFrame, industry_column: str = "industry_level1_code"
) -> pd.Series:
    """Encode the standard/code pair without conflating literal separators."""
    standard, code = (
        frame[column]
        .astype("string")
        .str.replace("\\", "\\\\", regex=False)
        .str.replace(":", "\\:", regex=False)
        for column in ("industry_standard_id", industry_column)
    )
    return standard.str.cat(code, sep=":")


def _center_scale(values, *, center, scale):
    if center and scale:
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
    impute_missing: bool = True,
    output_groups: tuple[str, ...] = ("processed",),
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
        impute_missing,
        size_column=size_column,
        output_groups=output_groups,
    )
    industry_neutralize = industry_neutralize and "processed" in output_groups
    size_neutralize = size_neutralize and "processed" in output_groups
    if "eligible" not in frame:
        raise ValueError("feature_eligibility_required")
    universe = frame["eligible"]
    if not pd.api.types.is_bool_dtype(universe) or universe.isna().any():
        raise ValueError("feature_eligibility_invalid")
    scoped = frame.loc[universe].copy()
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
        context_present = (
            winsorized[industry_column].notna()
            if industry_neutralize and group_name == "processed"
            else pd.Series(True, index=scoped.index)
        )
        for factor in factor_columns:
            values = group_frame[factor]
            raw_missing = scoped[factor].isna()
            needs_fit = (
                recipe.clip != "none"
                or center
                or scale
                or (
                    group_name == "processed"
                    and (
                        industry_neutralize
                        or (size_neutralize and factor != size_column)
                    )
                )
            )
            suffix = ("pre_neutral_" if group_name == "pre_neutral" else "") + factor
            if variance_counts is not None:
                variance_counts.setdefault(suffix, 0)
            scoped_feature = pd.Series(np.nan, index=scoped.index, dtype="float64")
            identifiable = pd.Series(False, index=scoped.index)
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
                transformed = _center_scale(group, center=center, scale=scale)
                scoped_feature.iloc[positions] = transformed
                identifiable.iloc[positions] = (
                    not needs_fit or transformed.notna().any()
                )
            final_missing = raw_missing | scoped_feature.isna()
            context_valid = context_present
            if group_name == "processed" and needs_size and factor != size_column:
                context_valid = context_valid & scoped[size_column].notna()
            ordinary_missing = raw_missing & identifiable & context_valid
            feature = pd.Series(np.nan, index=result.index, dtype="float64")
            feature.loc[scoped.index] = scoped_feature.mask(
                ordinary_missing & impute_missing, 0.0
            )
            missing = pd.Series(0, index=result.index, dtype="int8")
            observed = pd.Series(0, index=result.index, dtype="int8")
            missing.loc[scoped.index] = ordinary_missing.astype("int8")
            observed.loc[scoped.index] = final_missing.eq(False).astype("int8")
            for prefix, values in zip(
                FEATURE_PREFIXES, (feature, missing, observed), strict=True
            ):
                feature_values[prefix + suffix] = values
    return pd.concat((result, pd.DataFrame(feature_values, index=result.index)), axis=1)
