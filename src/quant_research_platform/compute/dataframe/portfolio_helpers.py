"""    DataFrame portfolio weighting  ?tradability helpers ?"""

from __future__ import annotations

from math import floor

import numpy as np
import pandas as pd

from quant_research_platform.compute.panels import ComputePanelSpec
from quant_research_platform.contracts.portfolios import (
    CostConfig,
    WeightConfig,
)


def _cost_components(
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


def _uses_detailed_cost_config(cost_config: CostConfig) -> bool:
    return (
        cost_config.buy_cost_bps != 0
        or cost_config.sell_cost_bps != 0
        or cost_config.slippage_bps != 0
        or cost_config.spread_bps not in (None, 0)
        or cost_config.impact_model is not None
    )


def _validate_segment_columns(
    panel: pd.DataFrame,
    segment_columns: tuple[str, ...],
) -> None:
    missing_columns = [column for column in segment_columns if column not in panel]
    if missing_columns:
        raise ValueError(f"segment columns are missing: {missing_columns}.")


def _validate_weight_columns(panel: pd.DataFrame, weight_config: WeightConfig) -> None:
    if weight_config.risk_column is None:
        return
    if weight_config.risk_column not in panel:
        raise ValueError(f"risk column {weight_config.risk_column!r} is missing.")
    if not pd.api.types.is_numeric_dtype(panel[weight_config.risk_column]):
        raise TypeError(f"risk column {weight_config.risk_column!r} must be numeric.")
    if (panel[weight_config.risk_column].dropna() <= 0).any():
        raise ValueError(f"risk column {weight_config.risk_column!r} must be positive.")



def _risk_values(
    timestamp_frame: pd.DataFrame,
    weight_config: WeightConfig,
) -> np.ndarray | None:
    if weight_config.method != "inverse_vol_scaled_quantile":
        return None
    return timestamp_frame[str(weight_config.risk_column)].to_numpy(
        dtype="float64",
        copy=False,
    )



def _select_signals(
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


def _select_signal(spec: ComputePanelSpec, signal_column: str | None) -> str:
    signal = spec.signal_columns[0] if signal_column is None else signal_column
    if signal not in spec.signal_columns:
        raise ValueError(f"signal_column {signal!r} is not in the panel spec.")
    return signal


def _selection_count(n_obs: int, quantile: float) -> int:
    return max(1, floor(n_obs * quantile))


def _normalize_group_key(
    group_key: object,
    group_columns: list[str],
) -> dict[str, object]:
    if len(group_columns) == 1:
        values = group_key if isinstance(group_key, tuple) else (group_key,)
    else:
        values = tuple(group_key)
    return dict(zip(group_columns, values, strict=True))
