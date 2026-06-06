"""Tensor portfolio tradability constraint algorithms."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from quant_research_platform.compute.config import (
    TradabilityConfig,
    WeightConfig,
)
from quant_research_platform.compute.tensor.portfolio_helpers import (
    _directed_signal_values,
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


__all__ = [
    "_apply_array_tradability_constraints",
    "_optional_bool_array",
    "_previous_position_vector",
    "_ranked_asset_indices",
]
