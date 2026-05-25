"""Evaluation panel schema and column validation helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

from quant_research_platform.kernels.kernel_protocol import PanelSpec, validate_panel

if TYPE_CHECKING:
    from quant_research_platform.evaluations.evaluation_spec import EvaluationSpec


def validate_evaluation_spec_columns(spec: EvaluationSpec) -> None:
    """Validate all columns referenced by an evaluation spec."""

    panel = spec.panel
    requested_columns = [
        spec.timestamp_column,
        spec.asset_column,
        *spec.feature_columns,
        *spec.label_columns,
        *spec.segment_columns,
        spec.return_source_column,
        spec.tradability_config.eligibility_column,
        spec.tradability_config.can_enter_column,
        spec.tradability_config.can_exit_column,
        spec.execution_config.execution_price_column,
        *spec.neutralization_config.group_columns,
        *spec.neutralization_config.exposure_columns,
        *(config.risk_column for config in spec.weight_configs),
    ]
    missing = sorted(
        {
            column
            for column in requested_columns
            if column is not None and column not in panel.columns
        }
    )
    if missing:
        raise ValueError(f"evaluation panel is missing columns: {missing}.")


def validate_evaluation_set_return_source_column(spec: EvaluationSpec) -> str:
    """Resolve the realized-return column for portfolio cumulative returns."""

    return_source_column = spec.resolve_return_source_column(spec.panel)
    if return_source_column not in spec.panel.columns:
        raise ValueError(
            "Portfolio cumulative returns require return_source_column, but column "
            f"{return_source_column!r} was not found."
        )
    if not pd.api.types.is_numeric_dtype(spec.panel[return_source_column]):
        raise TypeError(
            f"return_source_column {return_source_column!r} must be numeric."
        )
    return return_source_column


__all__ = [
    "PanelSpec",
    "validate_evaluation_set_return_source_column",
    "validate_evaluation_spec_columns",
    "validate_panel",
]
