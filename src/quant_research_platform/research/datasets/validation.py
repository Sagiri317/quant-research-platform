"""ResearchDataset validation helpers."""

from __future__ import annotations

import pandas as pd

from quant_research_platform.contracts.columns import ColumnRole
from quant_research_platform.contracts.datasets import ResearchDatasetSpec
from quant_research_platform.research.datasets._contracts import (
    RESEARCH_DATASET_PRIMARY_KEY_COLUMNS,
)


def validate_research_dataset(
    frame: pd.DataFrame,
    spec: ResearchDatasetSpec,
    *,
    require_all_columns: bool = True,
) -> None:
    if spec.primary_key != RESEARCH_DATASET_PRIMARY_KEY_COLUMNS:
        raise ValueError("invalid_research_dataset_primary_key")
    required = set(spec.required_columns if require_all_columns else spec.primary_key)
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"missing_research_required_fields: {', '.join(missing)}")

    null_key_columns = [
        column for column in spec.primary_key if bool(frame[column].isna().any())
    ]
    if null_key_columns:
        joined = ", ".join(null_key_columns)
        raise ValueError(f"null_research_dataset_key_columns: {joined}")

    duplicates = frame.duplicated(list(spec.primary_key))
    if bool(duplicates.any()):
        raise ValueError("duplicate_research_dataset_keys")

    invalid_factors = [
        column
        for column in spec.factor_columns
        if spec.column_roles.get(column) is not ColumnRole.FACTOR
    ]
    if invalid_factors:
        joined = ", ".join(invalid_factors)
        raise ValueError(f"invalid_research_dataset_factors: {joined}")

    invalid_features = [
        column
        for column in spec.feature_columns
        if spec.column_roles.get(column) is not ColumnRole.FEATURE
    ]
    if invalid_features:
        joined = ", ".join(invalid_features)
        raise ValueError(f"invalid_research_dataset_features: {joined}")

    invalid_labels = [
        column
        for column in spec.label_columns
        if spec.column_roles.get(column) is not ColumnRole.LABEL
    ]
    if invalid_labels:
        joined = ", ".join(invalid_labels)
        raise ValueError(f"invalid_research_dataset_labels: {joined}")

    mask_inputs = [
        column
        for column in (*spec.factor_columns, *spec.feature_columns)
        if spec.column_roles.get(column) is ColumnRole.MASK
    ]
    if mask_inputs:
        raise ValueError(f"mask_column_in_research_inputs: {', '.join(mask_inputs)}")

    invalid_filters = [
        column
        for column in spec.filter_columns
        if spec.column_roles.get(column) not in {ColumnRole.MASK, ColumnRole.UNIVERSE}
    ]
    if invalid_filters:
        joined = ", ".join(invalid_filters)
        raise ValueError(f"invalid_research_dataset_filters: {joined}")

    invalid_context = [
        column
        for column in spec.context_columns
        if spec.column_roles.get(column) is not ColumnRole.CONTEXT
    ]
    if invalid_context:
        joined = ", ".join(invalid_context)
        raise ValueError(f"invalid_research_dataset_context: {joined}")


def validate_evaluation_input(
    frame: pd.DataFrame,
    spec: ResearchDatasetSpec,
    *,
    signal_columns: tuple[str, ...],
    label_columns: tuple[str, ...],
) -> None:
    validate_research_dataset(frame, spec, require_all_columns=False)
    _require_declared_columns(
        signal_columns,
        declared=(*spec.factor_columns, *spec.feature_columns),
        frame=frame,
        error_name="invalid_evaluation_signal_columns",
    )
    _require_declared_columns(
        label_columns,
        declared=spec.label_columns,
        frame=frame,
        error_name="invalid_evaluation_label_columns",
    )


def validate_prediction_input(
    frame: pd.DataFrame,
    spec: ResearchDatasetSpec,
    *,
    input_columns: tuple[str, ...],
) -> None:
    validate_research_dataset(frame, spec, require_all_columns=False)
    _require_declared_columns(
        input_columns,
        declared=(*spec.factor_columns, *spec.feature_columns),
        frame=frame,
        error_name="invalid_prediction_input_columns",
    )


def _require_declared_columns(
    columns: tuple[str, ...],
    *,
    declared: tuple[str, ...],
    frame: pd.DataFrame,
    error_name: str,
) -> None:
    declared_set = set(declared)
    missing_or_undeclared = [
        column
        for column in columns
        if column not in declared_set or column not in frame.columns
    ]
    if missing_or_undeclared:
        raise ValueError(f"{error_name}: {', '.join(missing_or_undeclared)}")


__all__ = [
    "validate_evaluation_input",
    "validate_prediction_input",
    "validate_research_dataset",
]
