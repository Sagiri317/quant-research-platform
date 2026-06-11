"""DataFrame portfolio construction algorithms."""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant_research_platform.compute.config import (
    TradabilityConfig,
    WeightConfig,
)
from quant_research_platform.compute.dataframe.constraints import (
    _apply_tradability_constraints,
    _build_tradability_context,
    _tradability_diagnostic_counts,
    _validate_tradability_columns,
)
from quant_research_platform.compute.dataframe.portfolio_constants import (
    POSITION_COLUMNS,
    WEIGHT_DIAGNOSTIC_COLUMNS,
)
from quant_research_platform.compute.dataframe.portfolio_helpers import (
    _direct_signal_array,
    _normalize_group_key,
    _risk_values,
    _select_feature,
    _select_features,
    _selection_count,
    _validate_segment_columns,
    _validate_weight_columns,
)
from quant_research_platform.compute.panels import (
    ComputePanelSpec,
    validate_compute_panel,
)


def build_weighted_positions(
    frame: pd.DataFrame,
    spec: ComputePanelSpec,
    weight_config: WeightConfig,
    *,
    segment_columns: tuple[str, ...] = (),
    portfolio_features: tuple[str, ...] | None = None,
    tradability_config: TradabilityConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """为每个请求的 signal feature 构造 timestamp-level positions。"""

    panel = validate_compute_panel(frame, spec)
    return _build_weighted_positions_from_validated_panel(
        panel,
        spec,
        weight_config,
        segment_columns=segment_columns,
        portfolio_features=portfolio_features,
        tradability_config=tradability_config,
    )

def _build_weighted_positions_from_validated_panel(
    panel: pd.DataFrame,
    spec: ComputePanelSpec,
    weight_config: WeightConfig,
    *,
    segment_columns: tuple[str, ...] = (),
    portfolio_features: tuple[str, ...] | None = None,
    tradability_config: TradabilityConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """基于调用方已校验且按 timestamp/asset 排序的 panel 构造 positions。"""

    _validate_segment_columns(panel, segment_columns)
    _validate_weight_columns(panel, weight_config)
    if tradability_config is None:
        tradability_config = TradabilityConfig()
    _validate_tradability_columns(panel, tradability_config)
    features = _select_features(spec, portfolio_features)
    group_columns = [*segment_columns, spec.timestamp_column]

    position_records: list[dict[str, object]] = []
    diagnostic_records: list[dict[str, object]] = []
    previous_positions: dict[tuple[object, ...], dict[str, dict[str, float]]] = {}
    for group_key, timestamp_frame in panel.groupby(group_columns, sort=True):
        group_values = _normalize_group_key(group_key, group_columns)
        timestamp = group_values[spec.timestamp_column]
        segment_key = tuple(group_values[column] for column in segment_columns)
        feature_previous = previous_positions.setdefault(segment_key, {})
        tradability_context = _build_tradability_context(
            timestamp_frame=timestamp_frame,
            spec=spec,
            tradability_config=tradability_config,
        )
        for feature in features:
            records, diagnostic = _build_feature_positions(
                timestamp_frame=timestamp_frame,
                spec=spec,
                feature=feature,
                weight_config=weight_config,
                tradability_config=tradability_config,
                tradability_context=tradability_context,
                group_values=group_values,
                previous_positions=feature_previous.get(feature, {}),
            )
            feature_previous[feature] = {
                str(record["asset"]): float(record["position"]) for record in records
            }
            position_records.extend(records)
            diagnostic_records.append(
                {
                    **{column: group_values[column] for column in segment_columns},
                    "timestamp": timestamp,
                    **diagnostic,
                }
            )

    position_columns = [*segment_columns, *POSITION_COLUMNS]
    diagnostic_columns = [*segment_columns, *WEIGHT_DIAGNOSTIC_COLUMNS]
    return (
        pd.DataFrame.from_records(position_records, columns=position_columns),
        pd.DataFrame.from_records(diagnostic_records, columns=diagnostic_columns),
    )

def build_long_short_positions(
    frame: pd.DataFrame,
    spec: ComputePanelSpec,
    top_quantile: float = 0.2,
    bottom_quantile: float | None = None,
    feature_column: str | None = None,
) -> pd.DataFrame:
    """基于 timestamp-level ranks 构造 equal-weight long-short positions。"""

    feature = _select_feature(spec, feature_column)
    config = WeightConfig(
        method="equal_weight_quantile",
        top_quantile=top_quantile,
        bottom_quantile=top_quantile if bottom_quantile is None else bottom_quantile,
    )
    positions, _diagnostics = build_weighted_positions(
        frame,
        spec,
        config,
        portfolio_features=(feature,),
    )
    return positions[POSITION_COLUMNS]

def _build_feature_positions(
    *,
    timestamp_frame: pd.DataFrame,
    spec: ComputePanelSpec,
    feature: str,
    weight_config: WeightConfig,
    tradability_config: TradabilityConfig,
    tradability_context: dict[str, object] | None = None,
    group_values: dict[str, object],
    previous_positions: dict[str, float],
) -> tuple[list[dict[str, object]], dict[str, object]]:
    asset_values = timestamp_frame[spec.asset_column].astype(str).to_numpy()
    feature_values = timestamp_frame[feature].to_numpy(dtype="float64", copy=False)
    valid_mask = ~pd.isna(feature_values)
    if tradability_config.eligibility_column is not None:
        valid_mask &= timestamp_frame[tradability_config.eligibility_column].to_numpy(
            dtype=bool, copy=False
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

    directed_signal = _direct_signal_array(feature_values, weight_config.direction)
    eligible_indices = np.flatnonzero(valid_mask)
    sorted_indices = eligible_indices[
        np.argsort(directed_signal[eligible_indices], kind="mergesort")
    ]
    positions: dict[str, float] = {}
    fallback_reason = ""
    long_count = 0
    short_count = 0
    minimum_count = 1 if weight_config.portfolio_mode == "long_only" else 2
    if len(sorted_indices) >= minimum_count:
        long_count = _selection_count(len(sorted_indices), weight_config.top_quantile)
        short_count = (
            0
            if weight_config.portfolio_mode == "long_only"
            else _selection_count(len(sorted_indices), weight_config.bottom_quantile)
        )
        if long_count + short_count <= len(sorted_indices):
            long_leg = sorted_indices[-long_count:]
            risk_values = _risk_values(timestamp_frame, weight_config)
            long_weights, long_fallback = _leg_weights(
                asset_values=asset_values[long_leg],
                feature_values=feature_values[long_leg],
                risk_values=None if risk_values is None else risk_values[long_leg],
                feature=feature,
                weight_config=weight_config,
                sign=1.0,
            )
            short_weights, short_fallback = (
                ({}, "")
                if weight_config.portfolio_mode == "long_only"
                else _leg_weights(
                    asset_values=asset_values[sorted_indices[:short_count]],
                    feature_values=feature_values[sorted_indices[:short_count]],
                    risk_values=(
                        None
                        if risk_values is None
                        else risk_values[sorted_indices[:short_count]]
                    ),
                    feature=feature,
                    weight_config=weight_config,
                    sign=-1.0,
                )
            )
            fallback_reason = ";".join(
                reason for reason in (long_fallback, short_fallback) if reason
            )
            positions.update(long_weights)
            positions.update(short_weights)
    positions = _apply_tradability_constraints(
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
            "feature": feature,
            "position": float(position),
        }
        for asset, position in nonzero_positions
    ]
    diagnostic = {
        "feature": feature,
        "method": weight_config.method,
        "direction": weight_config.direction,
        **_tradability_diagnostic_counts(
            timestamp_frame=timestamp_frame,
            feature=feature,
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
    feature_values: np.ndarray,
    risk_values: np.ndarray | None,
    feature: str,
    weight_config: WeightConfig,
    sign: float,
) -> tuple[dict[str, float], str]:
    if len(asset_values) == 0:
        return {}, ""
    fallback_reason = ""
    if weight_config.method == "equal_weight_quantile":
        raw = np.ones(len(asset_values), dtype="float64")
    elif weight_config.method == "rank_weighted_quantile":
        raw = np.arange(1, len(asset_values) + 1, dtype="float64")
    elif weight_config.method == "zscore_weighted_quantile":
        std = float(np.std(feature_values, ddof=0))
        raw = (
            np.abs(feature_values - float(np.mean(feature_values))) / std
            if std > 0
            else feature_values * 0
        )
    elif weight_config.method == "signal_abs_weighted_quantile":
        raw = np.abs(feature_values)
    elif weight_config.method == "inverse_vol_scaled_quantile":
        if risk_values is None:
            raise ValueError("risk_column is required for inverse_vol_scaled_quantile.")
        if (risk_values <= 0).any():
            raise ValueError("risk_column values must be positive.")
        raw = np.abs(feature_values) / risk_values
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

def _validate_positions(
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
    result["feature"] = result["feature"].astype(str)
    if not pd.api.types.is_numeric_dtype(result["position"]):
        raise TypeError("position must be numeric.")
    key_columns = [*segment_columns, "timestamp", "asset", "feature"]
    if result.duplicated(key_columns).any():
        raise ValueError(
            "positions must have unique segment-timestamp-asset-feature rows."
        )
    return result.sort_values(key_columns, kind="mergesort").reset_index(drop=True)

__all__ = [
    "build_long_short_positions",
    "build_weighted_positions",
]
