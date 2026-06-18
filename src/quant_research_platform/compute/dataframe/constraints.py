"""DataFrame portfolio tradability constraint algorithms."""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant_research_platform.compute.panels import ComputePanelSpec
from quant_research_platform.contracts.portfolios import TradabilityConfig


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
    spec: ComputePanelSpec,
    tradability_config: TradabilityConfig,
    previous_positions: dict[str, float],
) -> pd.DataFrame:
    if tradability_config.can_enter_column is None:
        return ranked
    can_enter = ranked[tradability_config.can_enter_column]
    has_previous = (
        ranked[spec.asset_column]
        .astype(str)
        .map(lambda asset: previous_positions.get(asset, 0.0) != 0.0)
    )
    return ranked[can_enter | has_previous].copy()


def _tradability_diagnostic_counts(
    *,
    timestamp_frame: pd.DataFrame,
    signal: str,
    spec: ComputePanelSpec,
    tradability_config: TradabilityConfig,
) -> dict[str, int]:
    signal_available = timestamp_frame[signal].notna()
    universe_count = int(len(timestamp_frame))
    eligible_mask = signal_available.copy()
    if tradability_config.eligibility_column is not None:
        eligible_mask &= timestamp_frame[tradability_config.eligibility_column]
    entry_mask = eligible_mask.copy()
    if tradability_config.can_enter_column is not None:
        entry_mask &= timestamp_frame[tradability_config.can_enter_column]
    exit_mask = pd.Series(True, index=timestamp_frame.index)
    if tradability_config.can_exit_column is not None:
        exit_mask = timestamp_frame[tradability_config.can_exit_column].astype(bool)
    eligible_count = int(eligible_mask.sum())
    can_enter_count = int(entry_mask.sum())
    can_exit_count = int(exit_mask.sum())
    return {
        "universe_count": universe_count,
        "eligible_count": eligible_count,
        "can_enter_count": can_enter_count,
        "can_exit_count": can_exit_count,
        "entry_blocked_count": max(0, eligible_count - can_enter_count),
        "exit_blocked_count": max(0, universe_count - can_exit_count),
    }


def _apply_tradability_constraints(
    *,
    positions: dict[str, float],
    timestamp_frame: pd.DataFrame,
    spec: ComputePanelSpec,
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
    spec: ComputePanelSpec,
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
    entry_checks_required = can_enter_values is not None and not bool(
        can_enter_values.all()
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


__all__: list[str] = []
