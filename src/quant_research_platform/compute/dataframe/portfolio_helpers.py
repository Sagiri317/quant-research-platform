"""提供 DataFrame portfolio weighting 和 tradability helpers。"""

from __future__ import annotations

from math import floor

import numpy as np
import pandas as pd

from quant_research_platform.compute.protocols import (
    CostConfig,
    PanelSpec,
    WeightConfig,
)


def _cost_components(
    *,
    buy_turnover: float,
    sell_turnover: float,
    turnover: float,
    weight_config: WeightConfig,
    cost_config: CostConfig,
) -> dict[str, object]:
    if not _uses_detailed_cost_config(cost_config):
        return {
            "buy_cost": 0.0,
            "sell_cost": 0.0,
            "slippage_cost": 0.0,
            "spread_cost": 0.0,
            "impact_cost": 0.0,
            "total_cost": turnover * float(weight_config.cost_bps) / 10_000.0,
            "cost_model": "scalar_cost_bps",
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



def _select_features(
    spec: PanelSpec,
    portfolio_features: tuple[str, ...] | None,
) -> tuple[str, ...]:
    if portfolio_features is None:
        return spec.feature_columns
    missing_features = [
        feature for feature in portfolio_features if feature not in spec.feature_columns
    ]
    if missing_features:
        raise ValueError(f"portfolio features are not in spec: {missing_features}.")
    return portfolio_features


def _select_feature(spec: PanelSpec, feature_column: str | None) -> str:
    feature = spec.feature_columns[0] if feature_column is None else feature_column
    if feature not in spec.feature_columns:
        raise ValueError(f"feature_column {feature!r} is not in the panel spec.")
    return feature


def _direct_signal(values: pd.Series, direction: str) -> pd.Series:
    if direction == "long_high":
        return values
    if direction == "long_low":
        return -values
    raise ValueError("direction must be 'long_high' or 'long_low'.")


def _direct_signal_array(values: np.ndarray, direction: str) -> np.ndarray:
    if direction == "long_high":
        return values
    if direction == "long_low":
        return -values
    raise ValueError("direction must be 'long_high' or 'long_low'.")


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
