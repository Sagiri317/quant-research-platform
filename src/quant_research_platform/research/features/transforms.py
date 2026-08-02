"""Feature transform materialization for research datasets."""

from __future__ import annotations

import pandas as pd

from quant_research_platform.contracts.features import FeatureTransformSpec


def apply_feature_transforms(
    frame: pd.DataFrame,
    transforms: tuple[FeatureTransformSpec, ...],
    *,
    timestamp_column: str = "timestamp",
) -> pd.DataFrame:
    if not transforms:
        return frame.copy()
    result = frame
    for transform in transforms:
        result = apply_feature_transform(
            result,
            transform,
            timestamp_column=timestamp_column,
        )
    return result


def apply_feature_transform(
    frame: pd.DataFrame,
    transform: FeatureTransformSpec,
    *,
    timestamp_column: str = "timestamp",
) -> pd.DataFrame:
    result = frame.copy()
    for source_column, output_column in zip(
        transform.source_columns,
        transform.output_columns,
        strict=True,
    ):
        if source_column not in result:
            raise ValueError(f"missing_feature_transform_source: {source_column}")
        values = pd.to_numeric(result[source_column], errors="coerce")
        transformed = _transform_values(
            values,
            timestamps=result[timestamp_column],
            transform=transform,
        )
        result[output_column] = _apply_fillna(
            transformed,
            timestamps=result[timestamp_column],
            fillna_policy=transform.fillna_policy,
        )
    return result


def _transform_values(
    values: pd.Series,
    *,
    timestamps: pd.Series,
    transform: FeatureTransformSpec,
) -> pd.Series:
    grouped = values.groupby(timestamps, sort=False)
    if transform.transform == "cross_sectional_zscore":
        return _zscore(values, grouped)
    if transform.transform == "cross_sectional_rank":
        return grouped.rank(method="average", pct=True)
    if transform.transform == "winsorize":
        return _winsorize(values, grouped, transform.winsorize_quantiles)
    if transform.transform == "winsorized_zscore":
        winsorized = _winsorize(values, grouped, transform.winsorize_quantiles)
        return _zscore(winsorized, winsorized.groupby(timestamps, sort=False))
    if transform.transform == "winsorized_rank":
        winsorized = _winsorize(values, grouped, transform.winsorize_quantiles)
        return winsorized.groupby(timestamps, sort=False).rank(
            method="average",
            pct=True,
        )
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


def _zscore(
    values: pd.Series,
    grouped: pd.core.groupby.SeriesGroupBy,
) -> pd.Series:
    mean = grouped.transform("mean")
    std = grouped.transform(lambda item: item.std(ddof=0))
    return (values - mean) / std.mask(std.eq(0))


def _apply_fillna(
    values: pd.Series,
    *,
    timestamps: pd.Series,
    fillna_policy: str,
) -> pd.Series:
    if fillna_policy == "none":
        return values
    if fillna_policy == "zero":
        return values.fillna(0.0)
    if fillna_policy == "median":
        medians = values.groupby(timestamps, sort=False).transform("median")
        return values.fillna(medians)
    raise ValueError("unsupported_feature_transform_fillna_policy")


__all__ = ["apply_feature_transform", "apply_feature_transforms"]
