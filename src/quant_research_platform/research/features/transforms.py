"""Feature transform materialization for research datasets."""

from __future__ import annotations

from contextlib import nullcontext

import pandas as pd

from quant_research_platform.contracts.features import FeatureTransformSpec
from quant_research_platform.runtime.resources import (
    current_runtime_resource_policy,
    pandas_frame_payload_bytes,
)


def apply_feature_transforms(
    frame: pd.DataFrame,
    transforms: tuple[FeatureTransformSpec, ...],
    *,
    timestamp_column: str = "timestamp",
) -> pd.DataFrame:
    return _apply_feature_transforms_to_copy(
        frame,
        transforms,
        timestamp_column=timestamp_column,
    )


def _apply_feature_transforms_to_copy(
    frame: pd.DataFrame,
    transforms: tuple[FeatureTransformSpec, ...],
    *,
    timestamp_column: str,
) -> pd.DataFrame:
    row_count = len(frame)
    index_bytes = int(frame.index.memory_usage(deep=True))
    vector_bytes = index_bytes + row_count * 8
    output_bound = pandas_frame_payload_bytes(frame) + row_count * 8 * sum(
        len(transform.output_columns) for transform in transforms
    )
    workspace_bound = max(
        (
            _feature_transform_workspace_bytes(
                transform,
                row_count=row_count,
                vector_bytes=vector_bytes,
            )
            for transform in transforms
        ),
        default=0,
    )
    policy = current_runtime_resource_policy()
    reservation = (
        policy.reserve(
            "research.feature.transform",
            output_bound + workspace_bound,
        )
        if policy is not None
        else nullcontext()
    )
    with reservation:
        result = frame.copy()
        for transform in transforms:
            _apply_feature_transform_to_owned_frame(
                result,
                transform,
                timestamp_column=timestamp_column,
            )
        if pandas_frame_payload_bytes(result) > output_bound:
            raise ValueError("research_feature_working_set_exceeds_task_memory_budget")
        return result


def _feature_transform_workspace_bytes(
    transform: FeatureTransformSpec,
    *,
    row_count: int,
    vector_bytes: int,
) -> int:
    group_codes = vector_bytes
    numeric_values = vector_bytes
    assignment_values = row_count * 8
    if transform.transform == "cross_sectional_zscore":
        transform_state = sum(
            (
                vector_bytes,  # group means
                vector_bytes,  # group standard deviations
                vector_bytes,  # nonzero standard deviations
                vector_bytes,  # centered values
                vector_bytes,  # normalized values
                row_count,  # zero-standard-deviation mask
            )
        )
    elif transform.transform == "cross_sectional_rank":
        transform_state = vector_bytes + vector_bytes  # rank workspace + output
    elif transform.transform == "winsorize":
        transform_state = sum(
            (vector_bytes, vector_bytes, vector_bytes)
        )  # lower, upper, clipped
    elif transform.transform == "winsorized_zscore":
        transform_state = sum(
            (
                vector_bytes,  # lower quantile
                vector_bytes,  # upper quantile
                vector_bytes,  # clipped values
                vector_bytes,  # group means
                vector_bytes,  # group standard deviations
                vector_bytes,  # nonzero standard deviations
                vector_bytes,  # centered values
                vector_bytes,  # normalized values
                row_count,
            )
        )
    elif transform.transform == "winsorized_rank":
        transform_state = sum(
            (
                vector_bytes,  # lower quantile
                vector_bytes,  # upper quantile
                vector_bytes,  # clipped values
                vector_bytes,  # rank workspace
                vector_bytes,  # ranked values
            )
        )
    elif transform.transform == "fillna_median":
        transform_state = vector_bytes + vector_bytes  # medians + filled values
    else:
        transform_state = vector_bytes
    if transform.fillna_policy == "median":
        transform_state += vector_bytes + vector_bytes  # medians + filled values
    elif transform.fillna_policy == "zero":
        transform_state += vector_bytes
    return numeric_values + group_codes + transform_state + assignment_values


def _apply_feature_transform_to_owned_frame(
    result: pd.DataFrame,
    transform: FeatureTransformSpec,
    *,
    timestamp_column: str,
) -> None:
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


__all__ = ["apply_feature_transforms"]
