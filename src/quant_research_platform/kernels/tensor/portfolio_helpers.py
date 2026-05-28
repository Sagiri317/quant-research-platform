"""提供 Tensor portfolio weighting 和 tradability helpers。"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from quant_research_platform.kernels.kernel_protocol import (
    TradabilityConfig,
    WeightConfig,
)


def _optional_bool_array(frame: pd.DataFrame, column: str | None) -> np.ndarray | None:
    if column is None:
        return None
    return frame[column].to_numpy(dtype="bool", copy=False)


def _ranked_asset_indices(
    *,
    assets: tuple[str, ...],
    feature_values: np.ndarray,
    weight_config: WeightConfig,
    tradability_config: TradabilityConfig,
    timestamp_input: Any,
    previous_positions: dict[str, float],
    previous_values: np.ndarray | None = None,
) -> np.ndarray:
    valid = np.isfinite(feature_values)
    if timestamp_input.eligible is not None:
        valid &= timestamp_input.eligible
    if (
        tradability_config.blocked_entry_policy == "redistribute"
        and timestamp_input.can_enter is not None
    ):
        previous_vector = (
            previous_values
            if previous_values is not None
            else _previous_position_vector(assets, previous_positions)
        )
        valid &= timestamp_input.can_enter | (previous_vector != 0.0)
    candidate_indices = np.flatnonzero(valid)
    directed = _directed_signal_values(
        feature_values[candidate_indices],
        weight_config.direction,
    )
    order = np.argsort(directed, kind="mergesort")
    return candidate_indices[order]


def _directed_signal_values(values: np.ndarray, direction: str) -> np.ndarray:
    if direction == "long_high":
        return values
    if direction == "long_low":
        return -values
    raise ValueError("direction must be 'long_high' or 'long_low'.")


def _leg_weight_values(
    *,
    asset_indices: np.ndarray,
    feature_values: np.ndarray,
    risk_values: np.ndarray | None,
    feature: str,
    weight_config: WeightConfig,
    sign: float,
) -> tuple[np.ndarray, str]:
    if len(asset_indices) == 0:
        return np.array([], dtype="float64"), ""
    fallback_reason = ""
    leg_values = feature_values[asset_indices]
    if weight_config.method == "equal_weight_quantile":
        return np.full(
            len(asset_indices),
            sign / len(asset_indices),
            dtype="float64",
        ), fallback_reason
    elif weight_config.method == "rank_weighted_quantile":
        raw = np.arange(1, len(asset_indices) + 1, dtype="float64")
    elif weight_config.method == "zscore_weighted_quantile":
        std = leg_values.std(ddof=0)
        raw = (
            np.abs((leg_values - leg_values.mean()) / std)
            if std > 0
            else leg_values * 0
        )
    elif weight_config.method == "signal_abs_weighted_quantile":
        raw = np.abs(leg_values)
    elif weight_config.method == "inverse_vol_scaled_quantile":
        if risk_values is None:
            raise ValueError(f"risk_column is required for feature {feature!r}.")
        risk = risk_values[asset_indices]
        if (risk <= 0).any():
            raise ValueError("risk_column values must be positive.")
        raw = np.abs(leg_values) / risk
    else:
        raise ValueError(f"unsupported weight method {weight_config.method!r}.")
    if not np.isfinite(raw).all() or float(raw.sum()) <= 0:
        raw = np.ones(len(asset_indices), dtype="float64")
        fallback_reason = "equal_weight_fallback"
    return sign * raw / raw.sum(), fallback_reason


def _apply_array_tradability_constraints(
    *,
    positions: np.ndarray,
    assets: tuple[str, ...],
    timestamp_input: Any,
    tradability_config: TradabilityConfig,
    previous_positions: dict[str, float],
    previous_values: np.ndarray | None = None,
) -> np.ndarray:
    if timestamp_input.can_enter is None and timestamp_input.can_exit is None:
        return positions
    if (
        timestamp_input.can_enter is None or bool(timestamp_input.can_enter.all())
    ) and (timestamp_input.can_exit is None or bool(timestamp_input.can_exit.all())):
        return positions
    constrained = positions.copy()
    previous = (
        previous_values
        if previous_values is not None
        else _previous_position_vector(assets, previous_positions)
    )

    if (
        timestamp_input.can_exit is not None
        and tradability_config.blocked_exit_policy != "force_exit"
    ):
        exit_blocked = (
            (previous != 0.0) & (constrained != previous) & ~timestamp_input.can_exit
        )
        if exit_blocked.any():
            if tradability_config.blocked_exit_policy == "error":
                offset = int(np.flatnonzero(exit_blocked)[0])
                raise ValueError(f"exit is blocked for asset {assets[offset]!r}.")
            if tradability_config.blocked_exit_policy == "carry_forward":
                constrained[exit_blocked] = previous[exit_blocked]

    if timestamp_input.can_enter is not None:
        same_sign_existing = (previous != 0.0) & (
            np.sign(previous) == np.sign(constrained)
        )
        entry_blocked = (
            (constrained != 0.0) & ~same_sign_existing & ~timestamp_input.can_enter
        )
        constrained[entry_blocked] = np.where(
            previous[entry_blocked] != 0.0,
            previous[entry_blocked],
            0.0,
        )
    return constrained


def _previous_position_vector(
    assets: tuple[str, ...],
    previous_positions: dict[str, float],
    asset_to_offset: dict[str, int] | None = None,
) -> np.ndarray:
    if not previous_positions:
        return np.zeros(len(assets), dtype="float64")
    if asset_to_offset is not None:
        values = np.zeros(len(assets), dtype="float64")
        for asset, position in previous_positions.items():
            offset = asset_to_offset.get(asset)
            if offset is not None:
                values[offset] = float(position)
        return values
    return np.array(
        [float(previous_positions.get(asset, 0.0)) for asset in assets],
        dtype="float64",
    )


def _filter_panel_group(
    *,
    panel: pd.DataFrame,
    segment_columns: tuple[str, ...],
    group_values: dict[str, object],
) -> pd.DataFrame:
    if not segment_columns:
        return panel
    mask = pd.Series(True, index=panel.index)
    for column in segment_columns:
        mask &= panel[column] == group_values[column]
    return panel.loc[mask]
