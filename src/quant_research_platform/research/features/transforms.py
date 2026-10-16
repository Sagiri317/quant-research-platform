"""Feature transform materialization for research datasets."""

from __future__ import annotations

from decimal import Decimal, localcontext
from fractions import Fraction

import numpy as np
import pandas as pd

from quant_research_platform.compute.dataframe.cross_section import _stable_zscore
from quant_research_platform.contracts.features import (
    FEATURE_PREFIXES,
    FeatureTransformSpec,
)
from quant_research_platform.contracts.models import _model_algorithm_facts
from quant_research_platform.contracts.validation import strict_finite_numeric_series


def canonical_feature_validity_columns(input_columns):
    """Project consumed encodings to their canonical structural-validity values."""
    values = []
    for column in input_columns:
        for prefix in reversed(FEATURE_PREFIXES):
            if column.startswith(prefix):
                values.append(FEATURE_PREFIXES[0] + column[len(prefix) :])
                break
    return tuple(dict.fromkeys(values))


def feature_observation_columns(columns):
    """Return observation flags only for canonical values, not indicator signals."""
    return tuple(
        dict.fromkeys(
            "feature_observed_" + column[len("feature_") :]
            for column in columns
            if column.startswith("feature_")
            and not column.startswith(("feature_missing_", "feature_observed_"))
        )
    )


def mask_unobserved_features(frame: pd.DataFrame, columns) -> pd.DataFrame:
    """Exclude technical fill values from statistical samples using observed flags."""
    flags = feature_observation_columns(columns)
    if not flags:
        return frame
    result = frame.copy()
    for flag in flags:
        if flag not in frame:
            raise ValueError(f"feature_observation_column_missing: {flag}")
        observed = strict_finite_numeric_series(
            frame[flag], error=f"invalid_feature_observation_flag: {flag}"
        )
        if not observed.isin((0, 1)).all():
            raise ValueError(f"invalid_feature_observation_flag: {flag}")
        column = "feature_" + flag[len("feature_observed_") :]
        result[column] = frame[column].where(observed.eq(1))
    return result


def model_input_validity_columns(input_columns, model_algorithm):
    """Own structural and direct-input validity for every model consumer."""
    direct = (
        ()
        if _model_algorithm_facts(model_algorithm).input_kind == "rank_ensemble_inputs"
        else input_columns
    )
    return tuple(
        dict.fromkeys((*canonical_feature_validity_columns(input_columns), *direct))
    )


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
        center=recipe.center,
        scale=recipe.scale,
        impute_missing=recipe.impute_missing,
        diagnostics=diagnostics,
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
    """Interpolate shared finite order statistics before rounding to binary64."""
    ordered = values.dropna().sort_values().to_numpy()
    if not len(ordered):
        return (np.nan,) * len(quantiles)
    result = []
    for quantile in quantiles:
        position = (len(ordered) - 1) * quantile
        lower = int(position)
        weight = Fraction(position - lower)
        result.append(
            float(
                (1 - weight) * Fraction(float(ordered[lower]))
                + weight * Fraction(float(ordered[min(lower + 1, len(ordered) - 1)]))
            )
        )
    return tuple(result)


def _integer_coordinates(values):
    """Represent finite binary64 values on one exact dyadic integer scale."""
    ratios = [float(value).as_integer_ratio() for value in values]
    scale = max((denominator for _, denominator in ratios), default=1)
    return [
        numerator * (scale // denominator) for numerator, denominator in ratios
    ], scale


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
        exact, scale = _integer_coordinates(source.iloc[positions].dropna())
        if not exact:
            continue
        exact.sort()
        middle = len(exact) // 2
        twice_center = exact[middle] + exact[-middle - 1]
        distances = sorted(abs(2 * value - twice_center) for value in exact)
        mad = Fraction(distances[middle] + distances[-middle - 1], 4 * scale)
        if mad == 0:
            if zero_counts is not None:
                zero_counts[column] += 1
            continue
        center = Fraction(twice_center, 2 * scale)
        bound = 3 * Fraction(1.4826) * mad
        for output, threshold in ((lower, center - bound), (upper, center + bound)):
            try:
                output[positions] = float(threshold)
            except OverflowError:
                output[positions] = -np.inf if threshold < 0 else np.inf
    return values.clip(
        lower=pd.Series(lower, index=values.index),
        upper=pd.Series(upper, index=values.index),
    )


def _neutralization_design(industries, sizes):
    """Prepare exact design moments once for targets with the same valid domain."""
    counts = industries.value_counts(sort=False)
    x_values = (
        (0,) * len(industries) if sizes is None else _integer_coordinates(sizes)[0]
    )
    sums = {group: [0, 0] for group in counts.index}
    for group, x in zip(industries, x_values, strict=True):
        sums[group][0] += x
        sums[group][1] += x * x
    xx = sum(
        (
            Fraction(int(counts[group]) * v[1] - v[0] ** 2, int(counts[group]))
            for group, v in sums.items()
        ),
        Fraction(),
    )
    return counts, x_values, sums, xx


def _industry_size_residuals(target, industries, sizes, *, rescale, design=None):
    """Project binary64 inputs exactly onto industry intercepts and one size slope."""
    counts, x_values, x_sums, xx = (
        _neutralization_design(industries, sizes) if design is None else design
    )
    y_values, y_scale = _integer_coordinates(target)
    sums = {group: [0, 0] for group in counts.index}
    for group, y, x in zip(industries, y_values, x_values, strict=True):
        sums[group][0] += y
        sums[group][1] += x * y
    slope = Fraction()
    if sizes is not None:
        xy = sum(
            (
                Fraction(
                    int(counts[group]) * v[1] - v[0] * x_sums[group][0],
                    int(counts[group]),
                )
                for group, v in sums.items()
            ),
            Fraction(),
        )
        if xx == 0:
            return np.full(len(target), np.nan)
        slope = xy / xx

    def residuals():
        for group, y, x in zip(industries, y_values, x_values, strict=True):
            count = int(counts[group])
            y = y * count - sums[group][0]
            x = x * count - x_sums[group][0]
            yield (y - slope * x) / (count * y_scale)

    if rescale:
        exact = list(residuals())
        scale = max(map(abs, exact))
        return np.array(
            [float(value / scale) for value in exact] if scale else [0.0] * len(target),
            dtype="float64",
        )
    return np.fromiter(map(float, residuals()), dtype="float64", count=len(target))


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
    """Keep industry categories from different standards distinct."""
    return (
        frame["industry_standard_id"]
        .astype("string")
        .str.cat(frame[industry_column].astype("string"), sep=":")
    )


def _center_scale(values, *, center, scale):
    if center and scale:
        return _stable_zscore(values)
    if not center and not scale:
        return values.astype("float64")
    observed = values.dropna()
    output = pd.Series(np.nan, index=values.index, dtype="float64")
    if not len(observed) or (scale and len(observed) < 2):
        return output
    integers, unit = _integer_coordinates(observed)
    count = len(integers)
    mean = Fraction(sum(integers), count)
    targets = [
        Fraction(value) - mean if center else Fraction(value) for value in integers
    ]
    if not scale:
        output.loc[observed.index] = [float(value / unit) for value in targets]
        return output
    variance = sum((Fraction(value) - mean) ** 2 for value in integers) / count
    if not variance:
        return output
    with localcontext() as context:
        context.prec = 80
        std = (Decimal(variance.numerator) / Decimal(variance.denominator)).sqrt()
        output.loc[observed.index] = [
            float(Decimal(value.numerator) / Decimal(value.denominator) / std)
            for value in targets
        ]
    return output


def a_share_feature_pipeline(
    frame: pd.DataFrame,
    *,
    factor_columns: tuple[str, ...],
    recipe_id: str = "a_share_canonical",
    size_column: str = "log_float_market_cap",
    industry_column: str = "industry_level1_code",
    diagnostics: dict | None = None,
    clip: str | None = None,
    industry_neutralize: bool = True,
    size_neutralize: bool = True,
    center: bool = True,
    scale: bool = True,
    impute_missing: bool = True,
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
    )
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
    context_present = (
        winsorized[industry_column].notna()
        if industry_neutralize
        else pd.Series(True, index=scoped.index)
    )
    for factor in factor_columns:
        values = residualized[factor]
        raw_missing = scoped[factor].isna()
        if variance_counts is not None:
            variance_counts.setdefault(factor, 0)
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
                variance_counts[factor] += 1
            transformed = _center_scale(group, center=center, scale=scale)
            scoped_feature.iloc[positions] = transformed
            identifiable.iloc[positions] = transformed.notna().any()
        final_missing = raw_missing | scoped_feature.isna()
        context_valid = context_present
        if needs_size and factor != size_column:
            context_valid = context_valid & scoped[size_column].notna()
        ordinary_missing = raw_missing & identifiable & context_valid & impute_missing
        feature = pd.Series(np.nan, index=result.index, dtype="float64")
        feature.loc[scoped.index] = scoped_feature.mask(ordinary_missing, 0.0)
        missing = pd.Series(0, index=result.index, dtype="int8")
        observed = pd.Series(0, index=result.index, dtype="int8")
        missing.loc[scoped.index] = final_missing.astype("int8")
        observed.loc[scoped.index] = final_missing.eq(False).astype("int8")
        for prefix, values in zip(
            FEATURE_PREFIXES, (feature, missing, observed), strict=True
        ):
            feature_values[prefix + factor] = values
    return pd.concat((result, pd.DataFrame(feature_values, index=result.index)), axis=1)
