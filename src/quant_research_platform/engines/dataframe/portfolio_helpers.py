"""DataFrame portfolio weighting and tradability helpers."""

from __future__ import annotations

from math import floor

import numpy as np
import pandas as pd

from quant_research_platform.core.contracts import (
    CostConfig,
    TradabilityConfig,
    WeightConfig,
)
from quant_research_platform.core.panel import PanelSpec


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
            "cost_model": "legacy_scalar_cost_bps",
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
        "total_cost": buy_cost
        + sell_cost
        + slippage_cost
        + spread_cost
        + impact_cost,
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


def _validate_tradability_columns(
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


def _filter_eligible_assets(
    ranked: pd.DataFrame,
    tradability_config: TradabilityConfig,
) -> pd.DataFrame:
    if tradability_config.eligibility_column is None:
        return ranked
    return ranked[ranked[tradability_config.eligibility_column]].copy()


def _filter_new_entry_blocked_assets(
    ranked: pd.DataFrame,
    *,
    spec: PanelSpec,
    tradability_config: TradabilityConfig,
    previous_positions: dict[str, float],
) -> pd.DataFrame:
    if tradability_config.can_enter_column is None:
        return ranked
    can_enter = ranked[tradability_config.can_enter_column]
    has_previous = ranked[spec.asset_column].astype(str).map(
        lambda asset: previous_positions.get(asset, 0.0) != 0.0
    )
    return ranked[can_enter | has_previous].copy()


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


def _apply_tradability_constraints(
    *,
    positions: dict[str, float],
    timestamp_frame: pd.DataFrame,
    spec: PanelSpec,
    tradability_config: TradabilityConfig,
    previous_positions: dict[str, float],
    tradability_context: dict[str, object] | None = None,
) -> dict[str, float]:
    if (
        tradability_config.can_enter_column is None
        and tradability_config.can_exit_column is None
    ):
        return positions

    context = tradability_context or _build_tradability_context(
        timestamp_frame=timestamp_frame,
        spec=spec,
        tradability_config=tradability_config,
    )
    entry_checks_required = bool(context["entry_checks_required"])
    exit_checks_required = bool(context["exit_checks_required"])
    if not entry_checks_required and not exit_checks_required:
        return positions

    can_enter_flags = context["can_enter_flags"]
    can_exit_flags = context["can_exit_flags"]
    target_assets = positions.keys()
    if (
        exit_checks_required
        and isinstance(can_exit_flags, dict)
        and can_exit_flags is not None
    ):
        target_assets = target_assets | (
            previous_positions.keys() & can_exit_flags.keys()
        )
    for asset in target_assets:
        target_position = positions.get(asset, 0.0)
        previous_position = previous_positions.get(asset, 0.0)
        adjusted_position = target_position
        if exit_checks_required and _is_exit_blocked(
            asset=asset,
            target_position=target_position,
            previous_position=previous_position,
            can_exit_flags=can_exit_flags,
            tradability_config=tradability_config,
        ):
            if tradability_config.blocked_exit_policy == "error":
                raise ValueError(f"exit is blocked for asset {asset!r}.")
            if tradability_config.blocked_exit_policy == "carry_forward":
                adjusted_position = previous_position
        entry_check_required = (
            entry_checks_required
            and adjusted_position != 0.0
            and not (
                previous_position != 0.0
                and np.sign(previous_position) == np.sign(adjusted_position)
            )
        )
        if entry_check_required and _is_entry_blocked(
            asset=asset,
            target_position=adjusted_position,
            previous_position=previous_position,
            can_enter_flags=can_enter_flags,
            tradability_config=tradability_config,
        ):
            adjusted_position = previous_position if previous_position != 0.0 else 0.0
        positions[asset] = float(adjusted_position)
    return positions


def _build_tradability_context(
    *,
    timestamp_frame: pd.DataFrame,
    spec: PanelSpec,
    tradability_config: TradabilityConfig,
) -> dict[str, object]:
    asset_values = timestamp_frame[spec.asset_column].astype(str).to_numpy()
    can_enter_values = _tradability_flag_values(
        timestamp_frame,
        tradability_config.can_enter_column,
    )
    can_exit_values = _tradability_flag_values(
        timestamp_frame,
        tradability_config.can_exit_column,
    )
    entry_checks_required = (
        can_enter_values is not None and not bool(can_enter_values.all())
    )
    exit_checks_required = (
        can_exit_values is not None
        and tradability_config.blocked_exit_policy != "force_exit"
        and not bool(can_exit_values.all())
    )
    return {
        "entry_checks_required": entry_checks_required,
        "exit_checks_required": exit_checks_required,
        "can_enter_flags": (
            dict(zip(asset_values, can_enter_values, strict=True))
            if entry_checks_required
            else None
        ),
        "can_exit_flags": (
            dict(zip(asset_values, can_exit_values, strict=True))
            if exit_checks_required
            else None
        ),
    }


def _tradability_flag_values(
    timestamp_frame: pd.DataFrame,
    column: str | None,
) -> np.ndarray | None:
    if column is None:
        return None
    return timestamp_frame[column].to_numpy(dtype=bool, copy=False)


def _is_exit_blocked(
    *,
    asset: str,
    target_position: float,
    previous_position: float,
    can_exit_flags: dict[str, bool] | None,
    tradability_config: TradabilityConfig,
) -> bool:
    if (
        can_exit_flags is None
        or previous_position == 0.0
        or target_position == previous_position
    ):
        return False
    if tradability_config.blocked_exit_policy == "force_exit":
        return False
    return not can_exit_flags[asset]


def _is_entry_blocked(
    *,
    asset: str,
    target_position: float,
    previous_position: float,
    can_enter_flags: dict[str, bool] | None,
    tradability_config: TradabilityConfig,
) -> bool:
    if can_enter_flags is None or target_position == 0.0:
        return False
    if previous_position != 0.0 and np.sign(previous_position) == np.sign(
        target_position
    ):
        return False
    return not can_enter_flags[asset]


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
