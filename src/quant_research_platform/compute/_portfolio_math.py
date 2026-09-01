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
from quant_research_platform.contracts.validation import strict_finite_numeric_series


def select_signals(
    spec: ComputePanelSpec,
    portfolio_signals: tuple[str, ...] | None,
) -> tuple[str, ...]:
    if portfolio_signals is None:
        return spec.signal_columns
    if len(set(portfolio_signals)) != len(portfolio_signals):
        raise ValueError("portfolio signals must not contain duplicates.")
    missing_signals = [
        signal for signal in portfolio_signals if signal not in spec.signal_columns
    ]
    if missing_signals:
        raise ValueError(f"portfolio signals are not in spec: {missing_signals}.")
    return portfolio_signals


def selection_count(n_obs: int, quantile: float) -> int:
    return max(1, floor(n_obs * quantile))


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
        if column is None:
            continue
        if not pd.api.types.is_bool_dtype(panel[column]):
            raise TypeError(f"tradability column {column!r} must be boolean.")
        if bool(panel[column].isna().any()):
            raise ValueError(f"tradability column {column!r} contains null values.")


def validate_weight_columns(panel: pd.DataFrame, weight_config: WeightConfig) -> None:
    if weight_config.risk_column is None:
        return
    if weight_config.risk_column not in panel:
        raise ValueError(f"risk column {weight_config.risk_column!r} is missing.")
    if not pd.api.types.is_numeric_dtype(panel[weight_config.risk_column]):
        raise TypeError(f"risk column {weight_config.risk_column!r} must be numeric.")
    risk = strict_finite_numeric_series(
        panel[weight_config.risk_column],
        error=f"risk column {weight_config.risk_column!r} must be finite numeric.",
    )
    if (risk <= 0).any():
        raise ValueError(f"risk column {weight_config.risk_column!r} must be positive.")


def cost_components(
    *,
    buy_turnover: float,
    sell_turnover: float,
    turnover: float,
    cost_config: CostConfig,
    trade_notional: float | None = None,
    average_daily_volume: float | None = None,
) -> dict[str, object]:
    buy_cost = buy_turnover * float(cost_config.commission_bps) / 10_000.0
    sell_cost = sell_turnover * float(cost_config.commission_bps) / 10_000.0
    slippage_cost = turnover * float(cost_config.slippage_bps) / 10_000.0
    impact_cost = 0.0
    if trade_notional is not None and average_daily_volume is not None:
        if average_daily_volume <= 0.0:
            raise ValueError("average_daily_volume_must_be_positive")
        impact_rate = (
            float(cost_config.impact_bps)
            / 10_000.0
            * (trade_notional / average_daily_volume) ** 0.5
        )
        impact_cost = trade_notional * impact_rate
    return {
        "buy_cost": buy_cost,
        "sell_cost": sell_cost,
        "slippage_cost": slippage_cost,
        "spread_cost": 0.0,
        "impact_cost": impact_cost,
        "total_cost": buy_cost + sell_cost + slippage_cost + impact_cost,
        "cost_model": "commission_slippage_sqrt_impact",
    }
