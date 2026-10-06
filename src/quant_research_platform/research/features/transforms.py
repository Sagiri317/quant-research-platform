"""Feature transform materialization for research datasets."""

from __future__ import annotations

from fractions import Fraction

import numpy as np
import pandas as pd

from quant_research_platform.compute.dataframe.cross_section import _stable_zscore
from quant_research_platform.contracts.features import FeatureTransformSpec
from quant_research_platform.contracts.models import _model_algorithm_facts
from quant_research_platform.contracts.validation import strict_finite_numeric_series

_CANONICAL_FEATURE_PREFIXES = ("feature_", "feature_missing_", "feature_observed_")


def canonical_feature_validity_columns(input_columns):
    """Project consumed encodings to their canonical structural-validity values."""
    values = []
    for column in input_columns:
        for prefix in reversed(_CANONICAL_FEATURE_PREFIXES):
            if column.startswith(prefix):
                values.append(_CANONICAL_FEATURE_PREFIXES[0] + column[len(prefix) :])
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
    timestamp_column: str = "timestamp",
) -> pd.DataFrame:
    canonical = tuple(t for t in transforms if t.transform == "a_share_canonical")
    if canonical:
        if len(transforms) != 1:
            raise ValueError("canonical_feature_pipeline_must_be_exclusive")
    if canonical:
        return canonical_feature_pipeline(
            frame, factor_columns=canonical[0].source_columns
        )
    result = frame.copy()
    for transform in transforms:
        _apply_feature_transform(result, transform, timestamp_column=timestamp_column)
    return result


def _apply_feature_transform(
    result: pd.DataFrame, transform: FeatureTransformSpec, *, timestamp_column: str
) -> None:
    for source_column, output_column in zip(
        transform.source_columns, transform.output_columns, strict=True
    ):
        if source_column not in result:
            raise ValueError(f"missing_feature_transform_source: {source_column}")
        values = strict_finite_numeric_series(
            result[source_column],
            error=f"invalid_feature_transform_source: {source_column}",
            allow_null=True,
        ).astype("float64")
        transformed = _transform_values(
            values, timestamps=result[timestamp_column], transform=transform
        )
        result[output_column] = _apply_fillna(
            transformed,
            timestamps=result[timestamp_column],
            fillna_policy=transform.fillna_policy,
        )


def _transform_values(
    values: pd.Series, *, timestamps: pd.Series, transform: FeatureTransformSpec
) -> pd.Series:
    grouped = values.groupby(timestamps, sort=False)
    if transform.transform == "cross_sectional_zscore":
        return grouped.transform(_stable_zscore)
    if transform.transform == "cross_sectional_rank":
        return grouped.rank(method="average", pct=True)
    if transform.transform == "winsorize":
        return _winsorize(values, grouped, transform.winsorize_quantiles)
    if transform.transform == "winsorized_zscore":
        winsorized = _winsorize(values, grouped, transform.winsorize_quantiles)
        return winsorized.groupby(timestamps, sort=False).transform(_stable_zscore)
    if transform.transform == "winsorized_rank":
        winsorized = _winsorize(values, grouped, transform.winsorize_quantiles)
        return winsorized.groupby(timestamps, sort=False).rank(
            method="average", pct=True
        )
    if transform.transform == "mad_winsorize":
        return _mad_winsorize(values, grouped)
    if transform.transform == "mad_winsorized_zscore":
        clipped = _mad_winsorize(values, grouped)
        return clipped.groupby(timestamps, sort=False).transform(_stable_zscore)
    if transform.transform == "mad_winsorized_rank":
        clipped = _mad_winsorize(values, grouped)
        return clipped.groupby(timestamps, sort=False).rank(method="average", pct=True)
    if transform.transform == "missing_indicator":
        return values.isna().astype("float64")
    if transform.transform == "observed_indicator":
        return values.notna().astype("float64")
    if transform.transform == "fillna_zero":
        return values.fillna(0.0)
    if transform.transform == "fillna_median":
        medians = values.groupby(timestamps, sort=False).transform("median")
        return values.fillna(medians)
    raise ValueError(f"unsupported_feature_transform: {transform.transform}")


def _winsorize(
    values: pd.Series,
    grouped: pd.core.groupby.SeriesGroupBy,
    quantiles: tuple[float, float] | None,
) -> pd.Series:
    lower_q, upper_q = quantiles or (0.01, 0.99)
    lower = grouped.transform(lambda item: item.quantile(lower_q))
    upper = grouped.transform(lambda item: item.quantile(upper_q))
    return values.clip(lower=lower, upper=upper)


def _mad_winsorize(
    values: pd.Series, grouped: pd.core.groupby.SeriesGroupBy
) -> pd.Series:
    median = grouped.transform("median")
    mad = grouped.transform(lambda group: (group - group.median()).abs().median())
    bound = (3.0 * 1.4826 * mad).where(mad.ne(0.0))
    return values.clip(lower=median - bound, upper=median + bound)


def _industry_size_residuals(target, industries, sizes):
    """Project binary64 inputs exactly onto industry intercepts and one size slope."""
    counts = industries.value_counts(sort=False)
    y_scale = max(float(value).as_integer_ratio()[1] for value in target)
    x_scale = (
        1
        if sizes is None
        else max(float(value).as_integer_ratio()[1] for value in sizes)
    )

    def integer(value, scale):
        numerator, denominator = float(value).as_integer_ratio()
        return numerator * (scale // denominator)

    sums = {group: [0, 0, 0, 0] for group in counts.index}
    for index, (group, value) in enumerate(zip(industries, target, strict=True)):
        y = integer(value, y_scale)
        x = 0 if sizes is None else integer(sizes[index], x_scale)
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
    for index, (group, value) in enumerate(zip(industries, target, strict=True)):
        count = int(counts[group])
        y = integer(value, y_scale) * count - sums[group][0]
        x = (
            0
            if sizes is None
            else integer(sizes[index], x_scale) * count - sums[group][1]
        )
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


def canonical_feature_pipeline(
    frame: pd.DataFrame,
    *,
    factor_columns: tuple[str, ...],
    size_column: str = "log_float_market_cap",
    industry_column: str = "industry_level1_code",
) -> pd.DataFrame:
    """Apply eligible -> MAD winsor -> neutralize -> population zscore."""
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
    for factor in canonical_feature_input_factors(
        factor_columns, size_column=size_column
    ):
        values = strict_finite_numeric_series(
            winsorized[factor],
            error=f"invalid_feature_factor: {factor}",
            allow_null=True,
        )
        winsorized[factor] = _mad_winsorize(
            values, values.groupby(winsorized["timestamp"], sort=False)
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
    for factor in factor_columns:
        values = residualized[factor]
        raw_missing = scoped[factor].isna()
        scoped_feature = values.groupby(scoped["timestamp"], sort=False).transform(
            _stable_zscore
        )
        final_missing = raw_missing | scoped_feature.isna()
        feature = pd.Series(np.nan, index=result.index, dtype="float64")
        identifiable = (
            values.notna().groupby(scoped["timestamp"], sort=False).transform("any")
        )
        context_valid = scoped[industry_column].notna()
        if factor != size_column:
            context_valid &= scoped[size_column].notna()
        ordinary_missing = raw_missing & identifiable & context_valid
        feature.loc[scoped.index] = scoped_feature.mask(ordinary_missing, 0.0)
        missing = pd.Series(0, index=result.index, dtype="int8")
        observed = pd.Series(0, index=result.index, dtype="int8")
        missing.loc[scoped.index] = final_missing.astype("int8")
        observed.loc[scoped.index] = final_missing.eq(False).astype("int8")
        for prefix, values in zip(
            _CANONICAL_FEATURE_PREFIXES, (feature, missing, observed), strict=True
        ):
            feature_values[prefix + factor] = values
    return pd.concat((result, pd.DataFrame(feature_values, index=result.index)), axis=1)


def _apply_fillna(
    values: pd.Series, *, timestamps: pd.Series, fillna_policy: str
) -> pd.Series:
    if fillna_policy == "none":
        return values
    if fillna_policy == "zero":
        return values.fillna(0.0)
    if fillna_policy == "median":
        medians = values.groupby(timestamps, sort=False).transform("median")
        return values.fillna(medians)
    raise ValueError("unsupported_feature_transform_fillna_policy")


def canonical_feature_transforms(factor_columns):
    """Persist the executable eligible/MAD/neutralize/zscore/encoding pipeline."""
    return (
        FeatureTransformSpec(
            transform_id="a_share_canonical",
            source_columns=tuple(factor_columns),
            output_columns=tuple(
                prefix + column
                for column in factor_columns
                for prefix in _CANONICAL_FEATURE_PREFIXES
            ),
            transform="a_share_canonical",
        ),
    )
