"""提供 evaluation panel schema 和 column validation helper。"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

from quant_research_platform.kernels.kernel_protocol import PanelSpec, validate_panel

if TYPE_CHECKING:
    from quant_research_platform.evaluations.evaluation_spec import EvaluationSpec


def validate_evaluation_spec_columns(spec: EvaluationSpec) -> None:
    """校验 evaluation spec 引用的所有必需和可选 columns。"""

    panel = spec.panel
    requested_columns = [
        spec.timestamp_column,
        spec.asset_column,
        *spec.feature_columns,
        *spec.label_columns,
        *spec.segment_columns,
        spec.pnl_return_column,
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


def validate_evaluation_set_pnl_return_column(spec: EvaluationSpec) -> str:
    """解析并校验 portfolio PnL 使用的 realized-return column。"""

    pnl_return_column = spec.resolve_pnl_return_column(spec.panel)
    if pnl_return_column not in spec.panel.columns:
        raise ValueError(
            "PnL curve requires pnl_return_column, but column "
            f"{pnl_return_column!r} was not found."
        )
    if not pd.api.types.is_numeric_dtype(spec.panel[pnl_return_column]):
        raise TypeError(f"pnl_return_column {pnl_return_column!r} must be numeric.")
    return pnl_return_column


__all__ = [
    "PanelSpec",
    "validate_evaluation_set_pnl_return_column",
    "validate_evaluation_spec_columns",
    "validate_panel",
]
