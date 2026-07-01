"""Private DataFrame position construction and validation math."""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant_research_platform.compute.dataframe.constraints import (
    apply_tradability_constraints,
    tradability_diagnostic_counts,
)
from quant_research_platform.compute.dataframe.portfolio_constants import (
    POSITION_COLUMNS,
)
from quant_research_platform.compute.dataframe.portfolio_helpers import (
    risk_values,
    selection_count,
)
from quant_research_platform.compute.panels import ComputePanelSpec
from quant_research_platform.contracts.portfolios import TradabilityConfig, WeightConfig


def build_signal_positions(
    *,
    timestamp_frame: pd.DataFrame,
    spec: ComputePanelSpec,
    signal: str,
    weight_config: WeightConfig,
    tradability_config: TradabilityConfig,
    tradability_context: dict[str, object] | None = None,
    group_values: dict[str, object],
    previous_positions: dict[str, float],
) -> tuple[list[dict[str, object]], dict[str, object]]:
    asset_values = timestamp_frame[spec.asset_column].astype(str).to_numpy()
    signal_values = timestamp_frame[signal].to_numpy(dtype="float64", copy=False)
    valid_mask = ~pd.isna(signal_values)
    if tradability_config.eligibility_column is not None:
        valid_mask &= timestamp_frame[tradability_config.eligibility_column].to_numpy(
            dtype=bool,
            copy=False,
        )
    if (
        tradability_config.blocked_entry_policy == "redistribute"
        and tradability_config.can_enter_column is not None
    ):
        can_enter = timestamp_frame[tradability_config.can_enter_column].to_numpy(
            dtype=bool,
            copy=False,
        )
        has_previous = np.fromiter(
            (previous_positions.get(asset, 0.0) != 0.0 for asset in asset_values),
            dtype=bool,
            count=len(asset_values),
        )
        valid_mask &= can_enter | has_previous

    eligible_indices = np.flatnonzero(valid_mask)
    sorted_indices = eligible_indices[
        np.argsort(signal_values[eligible_indices], kind="mergesort")
    ]
    positions: dict[str, float] = {}
    fallback_reason = ""
    long_count = 0
    short_count = 0
    minimum_count = 1 if weight_config.portfolio_mode == "long_only" else 2
    if len(sorted_indices) >= minimum_count:
        long_count = selection_count(len(sorted_indices), weight_config.top_quantile)
        short_count = (
            0
            if weight_config.portfolio_mode == "long_only"
            else selection_count(len(sorted_indices), weight_config.bottom_quantile)
        )
        if long_count + short_count <= len(sorted_indices):
            long_leg = sorted_indices[-long_count:]
            risk_array = risk_values(timestamp_frame, weight_config)
            long_weights, long_fallback = _leg_weights(
                asset_values=asset_values[long_leg],
                signal_values=signal_values[long_leg],
                risk_values=None if risk_array is None else risk_array[long_leg],
                signal=signal,
                weight_config=weight_config,
                sign=1.0,
            )
            short_weights, short_fallback = (
                ({}, "")
                if weight_config.portfolio_mode == "long_only"
                else _leg_weights(
                    asset_values=asset_values[sorted_indices[:short_count]],
                    signal_values=signal_values[sorted_indices[:short_count]],
                    risk_values=(
                        None
                        if risk_array is None
                        else risk_array[sorted_indices[:short_count]]
                    ),
                    signal=signal,
                    weight_config=weight_config,
                    sign=-1.0,
                )
            )
            fallback_reason = ";".join(
                reason for reason in (long_fallback, short_fallback) if reason
            )
            positions.update(long_weights)
            positions.update(short_weights)
    positions = apply_tradability_constraints(
        positions=positions,
        timestamp_frame=timestamp_frame,
        spec=spec,
        tradability_config=tradability_config,
        previous_positions=previous_positions,
        tradability_context=tradability_context,
    )

    sorted_positions = sorted(positions.items())
    nonzero_positions = [
        (asset, position)
        for asset, position in sorted_positions
        if float(position) != 0.0
    ]
    if not nonzero_positions and len(asset_values):
        marker_asset = str(asset_values[0])
        nonzero_positions = [(marker_asset, float(positions.get(marker_asset, 0.0)))]
    records = [
        {
            **{
                column: value
                for column, value in group_values.items()
                if column != spec.timestamp_column
            },
            "timestamp": group_values[spec.timestamp_column],
            "asset": asset,
            "signal": signal,
            "position": float(position),
        }
        for asset, position in nonzero_positions
    ]
    diagnostic = {
        "signal": signal,
        "method": weight_config.method,
        **tradability_diagnostic_counts(
            timestamp_frame=timestamp_frame,
            signal=signal,
            spec=spec,
            tradability_config=tradability_config,
        ),
        "long_count": int(long_count),
        "short_count": int(short_count),
        "fallback_reason": fallback_reason,
    }
    return records, diagnostic


def _leg_weights(
    *,
    asset_values: np.ndarray,
    signal_values: np.ndarray,
    risk_values: np.ndarray | None,
    signal: str,
    weight_config: WeightConfig,
    sign: float,
) -> tuple[dict[str, float], str]:
    if len(asset_values) == 0:
        return {}, ""
    fallback_reason = ""
    if weight_config.method == "equal_weight_quantile":
        raw = np.ones(len(asset_values), dtype="float64")
    elif weight_config.method == "rank_weighted_quantile":
        raw = (
            np.arange(len(asset_values), 0, -1, dtype="float64")
            if sign < 0
            else np.arange(1, len(asset_values) + 1, dtype="float64")
        )
    elif weight_config.method == "zscore_weighted_quantile":
        std = float(np.std(signal_values, ddof=0))
        raw = (
            np.abs(signal_values - float(np.mean(signal_values))) / std
            if std > 0
            else signal_values * 0
        )
    elif weight_config.method == "signal_abs_weighted_quantile":
        raw = np.abs(signal_values)
    elif weight_config.method == "inverse_vol_scaled_quantile":
        if risk_values is None:
            raise ValueError("risk_column is required for inverse_vol_scaled_quantile.")
        if (risk_values <= 0).any():
            raise ValueError("risk_column values must be positive.")
        raw = np.abs(signal_values) / risk_values
    else:
        raise ValueError(f"unsupported weight method {weight_config.method!r}.")

    raw = np.asarray(raw, dtype="float64")
    raw_sum = float(raw.sum())
    if not np.isfinite(raw).all() or raw_sum <= 0:
        raw = np.ones(len(asset_values), dtype="float64")
        raw_sum = float(raw.sum())
        fallback_reason = "equal_weight_fallback"
    normalized = raw / raw_sum
    return (
        {
            str(asset): float(sign * weight)
            for asset, weight in zip(asset_values, normalized, strict=True)
        },
        fallback_reason,
    )


def validate_positions(
    positions: pd.DataFrame,
    *,
    segment_columns: tuple[str, ...] = (),
) -> pd.DataFrame:
    if not isinstance(positions, pd.DataFrame):
        raise TypeError("positions must be a pandas DataFrame.")
    required_columns = [*segment_columns, *POSITION_COLUMNS]
    missing_columns = [column for column in required_columns if column not in positions]
    if missing_columns:
        raise ValueError(f"positions are missing columns: {missing_columns}.")
    result = positions.copy()
    result["timestamp"] = pd.to_datetime(result["timestamp"])
    result["asset"] = result["asset"].astype(str)
    result["signal"] = result["signal"].astype(str)
    if not pd.api.types.is_numeric_dtype(result["position"]):
        raise TypeError("position must be numeric.")
    key_columns = [*segment_columns, "timestamp", "asset", "signal"]
    if result.duplicated(key_columns).any():
        raise ValueError(
            "positions must have unique segment-timestamp-asset-signal rows."
        )
    return result.sort_values(key_columns, kind="mergesort").reset_index(drop=True)
