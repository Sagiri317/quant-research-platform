"""Backend-neutral private portfolio math helpers."""

from __future__ import annotations

from math import floor

import pandas as pd

from quant_research_platform.compute.panels import ComputePanelSpec
from quant_research_platform.contracts.portfolios import (
    CostConfig,
    TradabilityConfig,
    WeightConfig,
)


def select_signals(
    spec: ComputePanelSpec,
    portfolio_signals: tuple[str, ...] | None,
) -> tuple[str, ...]:
    if portfolio_signals is None:
        return spec.signal_columns
    missing_signals = [
        signal for signal in portfolio_signals if signal not in spec.signal_columns
    ]
    if missing_signals:
        raise ValueError(f"portfolio signals are not in spec: {missing_signals}.")
    return portfolio_signals


def selection_count(n_obs: int, quantile: float) -> int:
    return max(1, floor(n_obs * quantile))


def validate_segment_columns(
    panel: pd.DataFrame,
    segment_columns: tuple[str, ...],
) -> None:
    missing_columns = [column for column in segment_columns if column not in panel]
    if missing_columns:
        raise ValueError(f"segment columns are missing: {missing_columns}.")


def validate_tradability_columns(
    panel: pd.DataFrame,
    tradability_config: TradabilityConfig,
) -> None:
    requested_columns = [
        tradability_config.eligibility_column,
        tradability_config.can_enter_column,
        tradability_config.can_exit_column,
    ]
    missing_columns = sorted(
        {
            column
            for column in requested_columns
            if column is not None and column not in panel
        }
    )
    if missing_columns:
        raise ValueError(f"tradability columns are missing: {missing_columns}.")
    for column in requested_columns:
        if column is not None and not pd.api.types.is_bool_dtype(panel[column]):
            raise TypeError(f"tradability column {column!r} must be boolean.")


def validate_weight_columns(panel: pd.DataFrame, weight_config: WeightConfig) -> None:
    if weight_config.risk_column is None:
        return
    if weight_config.risk_column not in panel:
        raise ValueError(f"risk column {weight_config.risk_column!r} is missing.")
    if not pd.api.types.is_numeric_dtype(panel[weight_config.risk_column]):
        raise TypeError(f"risk column {weight_config.risk_column!r} must be numeric.")
    if (panel[weight_config.risk_column].dropna() <= 0).any():
        raise ValueError(f"risk column {weight_config.risk_column!r} must be positive.")


def cost_components(
    *,
    buy_turnover: float,
    sell_turnover: float,
    turnover: float,
    cost_config: CostConfig,
) -> dict[str, object]:
    if not _uses_detailed_cost_config(cost_config):
        return {
            "buy_cost": 0.0,
            "sell_cost": 0.0,
            "slippage_cost": 0.0,
            "spread_cost": 0.0,
            "impact_cost": 0.0,
            "total_cost": 0.0,
            "cost_model": "zero_cost",
        }

    buy_cost = buy_turnover * float(cost_config.buy_cost_bps) / 10_000.0
    sell_cost = sell_turnover * float(cost_config.sell_cost_bps) / 10_000.0
    slippage_cost = turnover * float(cost_config.slippage_bps) / 10_000.0
    spread_bps = 0.0 if cost_config.spread_bps is None else cost_config.spread_bps
    spread_cost = turnover * float(spread_bps) / 10_000.0
    impact_cost = 0.0
    return {
        "buy_cost": buy_cost,
        "sell_cost": sell_cost,
        "slippage_cost": slippage_cost,
        "spread_cost": spread_cost,
        "impact_cost": impact_cost,
        "total_cost": buy_cost + sell_cost + slippage_cost + spread_cost + impact_cost,
        "cost_model": cost_config.impact_model or "detailed_bps",
    }


def require_return_source_column(return_source_column: str | None) -> str:
    if not isinstance(return_source_column, str) or not return_source_column:
        raise ValueError("portfolio_return_source_column_required")
    return return_source_column


def _uses_detailed_cost_config(cost_config: CostConfig) -> bool:
    return (
        cost_config.buy_cost_bps != 0
        or cost_config.sell_cost_bps != 0
        or cost_config.slippage_bps != 0
        or cost_config.spread_bps not in (None, 0)
        or cost_config.impact_model is not None
    )
