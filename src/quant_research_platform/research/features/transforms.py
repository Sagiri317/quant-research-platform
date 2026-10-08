"""Feature transform materialization for research datasets."""

from __future__ import annotations

from fractions import Fraction

import numpy as np
import pandas as pd

from quant_research_platform.compute.dataframe.cross_section import _stable_zscore
from quant_research_platform.contracts.features import (
    FEATURE_PREFIXES,
    FEATURE_RECIPES,
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
        diagnostics=diagnostics,
    )


def _winsorize(
    values: pd.Series,
    grouped: pd.core.groupby.SeriesGroupBy,
) -> pd.Series:
    lower = np.full(len(values), np.nan)
    upper = np.full(len(values), np.nan)
    for positions in grouped.indices.values():
        lower[positions], upper[positions] = _quantile(
            grouped.obj.iloc[positions], 0.01, 0.99
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
    lower = np.full(len(values), np.nan)
    upper = np.full(len(values), np.nan)
    for positions in grouped.indices.values():
        exact, scale = _integer_coordinates(grouped.obj.iloc[positions].dropna())
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


def _industry_size_residuals(target, industries, sizes):
    """Project binary64 inputs exactly onto industry intercepts and one size slope."""
    counts = industries.value_counts(sort=False)
    y_values, y_scale = _integer_coordinates(target)
    x_values = (0,) * len(target) if sizes is None else _integer_coordinates(sizes)[0]
    sums = {group: [0, 0, 0, 0] for group in counts.index}
    for group, y, x in zip(industries, y_values, x_values, strict=True):
        totals = sums[group]
        totals[0] += y
        totals[1] += x
        totals[2] += x * x
        totals[3] += x * y
    slope = Fraction()
    if sizes is not None:
        xx = sum(
            (
                Fraction(int(counts[group]) * v[2] - v[1] ** 2, int(counts[group]))
                for group, v in sums.items()
            ),
            Fraction(),
        )
        xy = sum(
            (
                Fraction(int(counts[group]) * v[3] - v[0] * v[1], int(counts[group]))
                for group, v in sums.items()
            ),
            Fraction(),
        )
        if xx == 0:
            return np.full(len(target), np.nan)
        slope = xy / xx
    residuals = np.empty(len(target), dtype="float64")
    for index, (group, y, x) in enumerate(
        zip(industries, y_values, x_values, strict=True)
    ):
        count = int(counts[group])
        y = y * count - sums[group][0]
        x = x * count - sums[group][1]
        residuals[index] = float((y - slope * x) / (count * y_scale))
    return residuals


def neutralize_cross_section(
    frame: pd.DataFrame,
    *,
    factor_columns: tuple[str, ...],
    industry_column: str = "industry_level1_code",
    size_column: str = "log_float_market_cap",
    minimum_observations: int = 100,
) -> pd.DataFrame:
    """Equal-weight, no-intercept industry/size residualization by date."""
    required = {"timestamp", industry_column, *factor_columns}
    if not required.issubset(frame.columns):
        raise ValueError("neutralization_columns_invalid")
    if type(minimum_observations) is not int or minimum_observations < 1:
        raise ValueError("neutralization_minimum_observations_invalid")
    result = frame.copy()
    for column in factor_columns:
        result[column] = np.nan
    for _timestamp, positions in result.groupby("timestamp", sort=False).groups.items():
        cross = frame.loc[positions]
        industry = cross[industry_column]
        valid_industry = industry.notna()
        if not valid_industry.any():
            continue
        for factor in factor_columns:
            values = strict_finite_numeric_series(
                cross.loc[valid_industry, factor],
                error=f"invalid_neutralization_factor: {factor}",
                allow_null=True,
            )
            valid = values.notna()
            if factor != size_column and size_column not in cross:
                raise ValueError("neutralization_size_column_required")
            if factor != size_column:
                size = strict_finite_numeric_series(
                    cross.loc[valid_industry, size_column],
                    error="invalid_neutralization_size",
                    allow_null=True,
                )
                valid &= size.notna()
            indices = values.index[valid]
            groups = industry.loc[indices]
            parameters = groups.nunique() + (factor != size_column)
            if len(indices) < max(minimum_observations, 5 * parameters):
                continue
            target = values.loc[indices].to_numpy(dtype="float64", copy=False)
            result.loc[indices, factor] = _industry_size_residuals(
                target,
                groups,
                None if factor == size_column else size.loc[indices].to_numpy(),
            )
    return result


def canonical_feature_input_factors(
    factor_columns: tuple[str, ...], *, size_column: str = "log_float_market_cap"
) -> tuple[str, ...]:
    """Own the economic inputs of canonical outputs, including auxiliary size."""
    return (
        tuple(dict.fromkeys((*factor_columns, size_column))) if factor_columns else ()
    )


def _industry_group_codes(
    frame: pd.DataFrame, industry_column: str = "industry_level1_code"
) -> pd.Series:
    """Keep industry categories from different standards distinct."""
    return (
        frame["industry_standard_id"]
        .astype("string")
        .str.cat(frame[industry_column].astype("string"), sep=":")
    )


def a_share_feature_pipeline(
    frame: pd.DataFrame,
    *,
    factor_columns: tuple[str, ...],
    recipe_id: str = "a_share_canonical",
    size_column: str = "log_float_market_cap",
    industry_column: str = "industry_level1_code",
    diagnostics: dict | None = None,
) -> pd.DataFrame:
    """Apply the named clipping recipe, neutralization and population zscore."""
    if type(recipe_id) is not str or recipe_id not in FEATURE_RECIPES:
        raise ValueError("unsupported_feature_recipe")
    if "eligible" not in frame:
        raise ValueError("feature_eligibility_required")
    universe = frame["eligible"]
    if not pd.api.types.is_bool_dtype(universe) or universe.isna().any():
        raise ValueError("feature_eligibility_invalid")
    scoped = frame.loc[universe].copy()
    if industry_column not in scoped:
        raise ValueError("feature_industry_context_missing")
    if factor_columns and size_column not in scoped:
        raise ValueError("feature_size_context_missing")
    winsorized = scoped.copy()
    mad_counts = variance_counts = None
    if diagnostics is not None:
        mad_counts = diagnostics.setdefault("mad_zero_by_input", {})
        variance_counts = diagnostics.setdefault("zero_variance_by_factor", {})
    for factor in canonical_feature_input_factors(
        factor_columns, size_column=size_column
    ):
        if mad_counts is not None:
            mad_counts.setdefault(factor, 0)
        values = strict_finite_numeric_series(
            winsorized[factor],
            error=f"invalid_feature_factor: {factor}",
            allow_null=True,
        )
        grouped = values.groupby(winsorized["timestamp"], sort=False)
        winsorized[factor] = (
            _mad_winsorize(values, grouped, zero_counts=mad_counts, column=factor)
            if recipe_id == "a_share_canonical"
            else _winsorize(values, grouped)
        )
    winsorized[industry_column] = _industry_group_codes(winsorized, industry_column)
    residualized = neutralize_cross_section(
        winsorized,
        factor_columns=factor_columns,
        industry_column=industry_column,
        size_column=size_column,
    )
    feature_values: dict[str, pd.Series] = {}
    result = frame.copy()
    industry_present = scoped[industry_column].notna()
    industry_size_present = (
        industry_present & scoped[size_column].notna() if factor_columns else None
    )
    for factor in factor_columns:
        values = residualized[factor]
        raw_missing = scoped[factor].isna()
        if variance_counts is not None:
            variance_counts.setdefault(factor, 0)

        def zscore(group, factor=factor):
            observed = group.dropna()
            if (
                variance_counts is not None
                and len(observed) >= 2
                and observed.eq(observed.iloc[0]).all()
            ):
                variance_counts[factor] += 1
            return _stable_zscore(group)

        scoped_feature = values.groupby(scoped["timestamp"], sort=False).transform(
            zscore
        )
        final_missing = raw_missing | scoped_feature.isna()
        feature = pd.Series(np.nan, index=result.index, dtype="float64")
        identifiable = (
            values.notna().groupby(scoped["timestamp"], sort=False).transform("any")
        )
        context_valid = (
            industry_present if factor == size_column else industry_size_present
        )
        ordinary_missing = raw_missing & identifiable & context_valid
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


def feature_transforms(factor_columns, *, recipe_id="a_share_canonical"):
    """Declare one executable recipe with derived output columns."""
    return (FeatureTransformSpec(recipe_id, tuple(factor_columns)),)
