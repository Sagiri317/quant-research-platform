"""Tensor portfolio construction algorithms."""

from __future__ import annotations

from operator import itemgetter

import numpy as np
import pandas as pd

from quant_research_platform.compute.dataframe.constraints import (
    _validate_tradability_columns,
)
from quant_research_platform.compute.dataframe.portfolio_constants import (
    COST_BREAKDOWN_COLUMNS,
    POSITION_COLUMNS,
    RETURNS_COLUMNS,
    TURNOVER_COLUMNS,
    WEIGHT_DIAGNOSTIC_COLUMNS,
)
from quant_research_platform.compute.dataframe.portfolio_helpers import (
    _normalize_group_key,
    _select_features,
    _selection_count,
    _validate_segment_columns,
    _validate_weight_columns,
)
from quant_research_platform.compute.protocols import (
    CostConfig,
    PanelSpec,
    TradabilityConfig,
    WeightConfig,
)
from quant_research_platform.compute.tensor.constraints import (
    _apply_array_tradability_constraints,
    _optional_bool_array,
    _previous_position_vector,
    _ranked_asset_indices,
)
from quant_research_platform.compute.tensor.portfolio_constants import (
    EXPOSURE_PATH_COLUMNS,
)
from quant_research_platform.compute.tensor.portfolio_helpers import (
    _leg_weight_values,
)
from quant_research_platform.compute.tensor.reporting import (
    _build_path_records_from_arrays,
)
from quant_research_platform.compute.tensor.turnover import _position_map_from_arrays
from quant_research_platform.runtime.profiling import measure_stage


class _TimestampPositionInput:
    assets: tuple[str, ...]
    asset_to_offset: dict[str, int]
    feature_values: dict[str, np.ndarray]
    label_values: np.ndarray
    risk_values: np.ndarray | None
    eligible: np.ndarray | None
    can_enter: np.ndarray | None
    can_exit: np.ndarray | None

class TensorPortfolioArtifacts:
    """一次 streaming pass 构造的内部 tensor portfolio artifacts。"""

    positions: pd.DataFrame
    diagnostics: pd.DataFrame
    turnover: pd.DataFrame
    cost_breakdown: pd.DataFrame
    returns: pd.DataFrame
    exposure_path: pd.DataFrame

def build_tensor_weighted_positions(
    panel: pd.DataFrame,
    spec: PanelSpec,
    weight_config: WeightConfig,
    *,
    segment_columns: tuple[str, ...] = (),
    portfolio_features: tuple[str, ...] | None = None,
    tradability_config: TradabilityConfig | None = None,
    return_source_column: str | None = None,
    value_dtype: str = "float64",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """从 tensor signal arrays 构建 timestamp-level positions。

    当前 tensor compute backend 保持 public artifact contract
    与 DataFrame backend 一致，因此返回相同结构的 positions 和 diagnostics frames。
    """

    artifacts = build_tensor_portfolio_artifacts(
        panel,
        spec,
        weight_config,
        segment_columns=segment_columns,
        portfolio_features=portfolio_features,
        tradability_config=tradability_config,
        return_source_column=return_source_column,
        value_dtype=value_dtype,
    )
    return artifacts.positions, artifacts.diagnostics

def build_tensor_portfolio_artifacts(
    panel: pd.DataFrame,
    spec: PanelSpec,
    weight_config: WeightConfig,
    cost_config: CostConfig | None = None,
    *,
    segment_columns: tuple[str, ...] = (),
    portfolio_features: tuple[str, ...] | None = None,
    tradability_config: TradabilityConfig | None = None,
    return_source_column: str | None = None,
    value_dtype: str = "float64",
) -> TensorPortfolioArtifacts:
    """构造 tensor portfolio artifacts，同时不保留 dense path matrices。"""

    dtype = _validate_value_dtype(value_dtype)
    _validate_segment_columns(panel, segment_columns)
    _validate_weight_columns(panel, weight_config)
    if cost_config is None:
        cost_config = CostConfig()
    if tradability_config is None:
        tradability_config = TradabilityConfig()
    _validate_tradability_columns(panel, tradability_config)
    return_source_column = return_source_column or spec.label_column
    features = _select_features(spec, portfolio_features)
    panel = panel.sort_values(
        [*segment_columns, spec.timestamp_column, spec.asset_column],
        kind="mergesort",
    )
    position_columns = _empty_position_columns(segment_columns)
    diagnostic_records: list[dict[str, object]] = []
    turnover_records: list[dict[str, object]] = []
    cost_records: list[dict[str, object]] = []
    return_records: list[dict[str, object]] = []
    exposure_records: list[dict[str, object]] = []
    previous_position_arrays: dict[
        tuple[object, ...],
        dict[str, tuple[tuple[str, ...], np.ndarray]],
    ] = {}

    group_columns = [*segment_columns, spec.timestamp_column]
    for group_key, timestamp_frame in panel.groupby(group_columns, sort=False):
        group_values = _normalize_group_key(group_key, group_columns)
        timestamp = group_values[spec.timestamp_column]
        segment_context = {column: group_values[column] for column in segment_columns}
        segment_key = tuple(segment_context[column] for column in segment_columns)
        feature_previous_arrays = previous_position_arrays.setdefault(segment_key, {})
        timestamp_input = measure_stage(
            "metric",
            "positions.prepare_timestamp",
            lambda timestamp_frame=timestamp_frame: _prepare_timestamp_position_input(
                timestamp_frame=timestamp_frame,
                spec=spec,
                features=features,
                weight_config=weight_config,
                tradability_config=tradability_config,
                return_source_column=return_source_column,
                value_dtype=dtype,
            ),
            sample_memory=False,
        )
        reference_previous_assets = (
            next(iter(feature_previous_arrays.values()))[0]
            if feature_previous_arrays
            else None
        )
        reference_assets_match = (
            reference_previous_assets == timestamp_input.assets
            if reference_previous_assets is not None
            else False
        )
        for feature in features:
            previous_array_state = feature_previous_arrays.get(feature)
            previous_assets_match = (
                previous_array_state is not None
                and reference_assets_match
                and previous_array_state[0] is reference_previous_assets
            )
            cached_previous_values = (
                previous_array_state[1] if previous_assets_match else None
            )
            previous_map = (
                _position_map_from_arrays(
                    assets=previous_array_state[0],
                    positions=previous_array_state[1],
                )
                if previous_array_state is not None and not previous_assets_match
                else {}
            )
            diagnostic, position_values, previous_values = measure_stage(
                "metric",
                "positions.rank_select",
                lambda feature=feature,
                cached_previous_values=cached_previous_values,
                previous_map=previous_map,
                timestamp_input=timestamp_input: (
                    _build_tensor_feature_positions(
                        timestamp_input=timestamp_input,
                        feature=feature,
                        weight_config=weight_config,
                        tradability_config=tradability_config,
                        previous_positions=previous_map,
                        cached_previous_values=cached_previous_values,
                    )
                ),
                sample_memory=False,
            )
            (
                nonzero_offsets,
                turnover_record,
                cost_record,
                return_record,
                exposure_record,
            ) = measure_stage(
                "metric",
                "positions.path_records",
                lambda feature=feature,
                previous_map=previous_map,
                previous_assets_match=previous_assets_match,
                segment_context=segment_context,
                timestamp=timestamp,
                timestamp_input=timestamp_input,
                position_values=position_values,
                previous_values=previous_values: (
                    _build_path_records_from_arrays(
                        context=segment_context,
                        timestamp=timestamp,
                        feature=feature,
                        label=return_source_column,
                        assets=timestamp_input.assets,
                        position_values=position_values,
                        label_values=timestamp_input.label_values,
                        previous_values=previous_values,
                        previous_positions=previous_map,
                        asset_to_offset=timestamp_input.asset_to_offset,
                        weight_config=weight_config,
                        cost_config=cost_config,
                        segment_columns=segment_columns,
                        previous_assets_match=previous_assets_match,
                    )
                ),
                sample_memory=False,
            )
            feature_previous_arrays[feature] = (timestamp_input.assets, position_values)
            exposure_records.append(exposure_record)
            _append_position_columns(
                columns=position_columns,
                context=segment_context,
                segment_columns=segment_columns,
                timestamp=timestamp,
                assets=timestamp_input.assets,
                feature=feature,
                positions=position_values,
                nonzero_offsets=nonzero_offsets,
            )
            turnover_records.append(turnover_record)
            cost_records.append(cost_record)
            if return_record is not None:
                return_records.append(return_record)
            diagnostic_records.append(
                {
                    **segment_context,
                    "timestamp": timestamp,
                    **diagnostic,
                }
            )

    positions = pd.DataFrame(
        position_columns,
        columns=[*segment_columns, *POSITION_COLUMNS],
    )
    diagnostics = pd.DataFrame.from_records(
        diagnostic_records,
        columns=[*segment_columns, *WEIGHT_DIAGNOSTIC_COLUMNS],
    )
    turnover = pd.DataFrame.from_records(
        turnover_records,
        columns=[*segment_columns, *TURNOVER_COLUMNS],
    )
    cost_breakdown = pd.DataFrame.from_records(
        cost_records,
        columns=[*segment_columns, *COST_BREAKDOWN_COLUMNS],
    )
    returns = pd.DataFrame.from_records(
        return_records,
        columns=[*segment_columns, *RETURNS_COLUMNS],
    )
    exposure_path = pd.DataFrame.from_records(
        exposure_records,
        columns=[*segment_columns, *EXPOSURE_PATH_COLUMNS],
    )
    return TensorPortfolioArtifacts(
        positions=positions,
        diagnostics=diagnostics,
        turnover=turnover,
        cost_breakdown=cost_breakdown,
        returns=returns,
        exposure_path=exposure_path,
    )

def _build_tensor_feature_positions(
    *,
    timestamp_input: _TimestampPositionInput,
    feature: str,
    weight_config: WeightConfig,
    tradability_config: TradabilityConfig,
    previous_positions: dict[str, float],
    cached_previous_values: np.ndarray | None = None,
) -> tuple[dict[str, object], np.ndarray, np.ndarray]:
    assets = timestamp_input.assets
    previous_values = measure_stage(
        "metric",
        "positions.rank_previous",
        lambda: (
            cached_previous_values
            if cached_previous_values is not None
            else _previous_position_vector(
                assets,
                previous_positions,
                timestamp_input.asset_to_offset,
            )
        ),
        sample_memory=False,
    )
    feature_values = timestamp_input.feature_values[feature]
    ranked_indices = measure_stage(
        "metric",
        "positions.rank_indices",
        lambda: _ranked_asset_indices(
            assets=assets,
            feature_values=feature_values,
            weight_config=weight_config,
            tradability_config=tradability_config,
            timestamp_input=timestamp_input,
            previous_positions=previous_positions,
            previous_values=previous_values,
        ),
        sample_memory=False,
    )
    positions, long_count, short_count, fallback_reason = measure_stage(
        "metric",
        "positions.rank_weight_assign",
        lambda: _assign_ranked_position_weights(
            assets=assets,
            feature=feature,
            feature_values=feature_values,
            risk_values=timestamp_input.risk_values,
            ranked_indices=ranked_indices,
            weight_config=weight_config,
        ),
        sample_memory=False,
    )
    positions = measure_stage(
        "metric",
        "positions.rank_tradability",
        lambda: _apply_array_tradability_constraints(
            positions=positions,
            assets=assets,
            timestamp_input=timestamp_input,
            tradability_config=tradability_config,
            previous_positions=previous_positions,
            previous_values=previous_values,
        ),
        sample_memory=False,
    )
    diagnostic = {
        "feature": feature,
        "method": weight_config.method,
        "direction": weight_config.direction,
        **_tensor_tradability_diagnostic_counts(timestamp_input, feature),
        "long_count": int(long_count),
        "short_count": int(short_count),
        "fallback_reason": fallback_reason,
    }
    return diagnostic, positions, previous_values

def _tensor_tradability_diagnostic_counts(
    timestamp_input: _TimestampPositionInput,
    feature: str,
) -> dict[str, int]:
    signal_available = np.isfinite(timestamp_input.feature_values[feature])
    universe_count = len(timestamp_input.assets)
    eligible_mask = signal_available.copy()
    if timestamp_input.eligible is not None:
        eligible_mask &= timestamp_input.eligible
    entry_mask = eligible_mask.copy()
    if timestamp_input.can_enter is not None:
        entry_mask &= timestamp_input.can_enter
    can_exit_count = (
        int(timestamp_input.can_exit.sum())
        if timestamp_input.can_exit is not None
        else universe_count
    )
    eligible_count = int(eligible_mask.sum())
    can_enter_count = int(entry_mask.sum())
    return {
        "universe_count": universe_count,
        "eligible_count": eligible_count,
        "can_enter_count": can_enter_count,
        "can_exit_count": can_exit_count,
        "entry_blocked_count": max(0, eligible_count - can_enter_count),
        "exit_blocked_count": max(0, universe_count - can_exit_count),
    }

def _assign_ranked_position_weights(
    *,
    assets: tuple[str, ...],
    feature: str,
    feature_values: np.ndarray,
    risk_values: np.ndarray | None,
    ranked_indices: np.ndarray,
    weight_config: WeightConfig,
) -> tuple[np.ndarray, int, int, str]:
    positions = np.zeros(len(assets), dtype=feature_values.dtype)
    fallback_reason = ""
    long_count = 0
    short_count = 0
    minimum_count = 1 if weight_config.portfolio_mode == "long_only" else 2
    if len(ranked_indices) < minimum_count:
        return positions, long_count, short_count, fallback_reason

    long_count = _selection_count(len(ranked_indices), weight_config.top_quantile)
    short_count = (
        0
        if weight_config.portfolio_mode == "long_only"
        else _selection_count(len(ranked_indices), weight_config.bottom_quantile)
    )
    if long_count + short_count > len(ranked_indices):
        return positions, long_count, short_count, fallback_reason

    long_indices = ranked_indices[-long_count:]
    long_positions, long_fallback = _leg_weight_values(
        asset_indices=long_indices,
        feature_values=feature_values,
        risk_values=risk_values,
        feature=feature,
        weight_config=weight_config,
        sign=1.0,
    )
    short_positions, short_fallback = (
        (np.array([], dtype="float64"), "")
        if weight_config.portfolio_mode == "long_only"
        else _leg_weight_values(
            asset_indices=ranked_indices[:short_count],
            feature_values=feature_values,
            risk_values=risk_values,
            feature=feature,
            weight_config=weight_config,
            sign=-1.0,
        )
    )
    fallback_reason = ";".join(
        reason for reason in (long_fallback, short_fallback) if reason
    )
    positions[long_indices] = long_positions
    if weight_config.portfolio_mode == "long_short":
        positions[ranked_indices[:short_count]] = short_positions
    return positions, long_count, short_count, fallback_reason

def _empty_position_columns(
    segment_columns: tuple[str, ...],
) -> dict[str, list[object]]:
    return {column: [] for column in [*segment_columns, *POSITION_COLUMNS]}

def _append_position_columns(
    *,
    columns: dict[str, list[object]],
    context: dict[str, object],
    segment_columns: tuple[str, ...],
    timestamp: object,
    assets: tuple[str, ...],
    feature: str,
    positions: np.ndarray,
    nonzero_offsets: np.ndarray | None = None,
) -> None:
    measure_stage(
        "metric",
        "positions.artifact_columns",
        lambda: _append_position_columns_unprofiled(
            columns=columns,
            context=context,
            segment_columns=segment_columns,
            timestamp=timestamp,
            assets=assets,
            feature=feature,
            positions=positions,
            nonzero_offsets=nonzero_offsets,
        ),
        sample_memory=False,
    )

def _append_position_columns_unprofiled(
    *,
    columns: dict[str, list[object]],
    context: dict[str, object],
    segment_columns: tuple[str, ...],
    timestamp: object,
    assets: tuple[str, ...],
    feature: str,
    positions: np.ndarray,
    nonzero_offsets: np.ndarray | None = None,
) -> None:
    if nonzero_offsets is None:
        nonzero_offsets = np.flatnonzero(positions != 0.0)
    if len(nonzero_offsets) == 0 and assets:
        nonzero_offsets = np.array([0], dtype="int64")
    n_assets = len(nonzero_offsets)
    for column in segment_columns:
        columns[column].extend([context[column]] * n_assets)
    columns["timestamp"].extend([timestamp] * n_assets)
    columns["asset"].extend(_selected_assets(assets, nonzero_offsets))
    columns["feature"].extend([feature] * n_assets)
    columns["position"].extend(positions[nonzero_offsets].tolist())

def _selected_assets(assets: tuple[str, ...], offsets: np.ndarray) -> list[str]:
    if len(offsets) == 0:
        return []
    if len(offsets) == 1:
        return [assets[int(offsets[0])]]
    selected = itemgetter(*offsets.tolist())(assets)
    return list(selected)

def _prepare_timestamp_position_input(
    *,
    timestamp_frame: pd.DataFrame,
    spec: PanelSpec,
    features: tuple[str, ...],
    weight_config: WeightConfig,
    tradability_config: TradabilityConfig,
    return_source_column: str | None = None,
    value_dtype: np.dtype | str = "float64",
) -> _TimestampPositionInput:
    return_source_column = return_source_column or spec.label_column
    dtype = _validate_value_dtype(str(value_dtype))
    asset_values = timestamp_frame[spec.asset_column].to_numpy(
        dtype=str,
        copy=False,
    )
    assets = tuple(asset_values.tolist())
    feature_matrix = timestamp_frame.loc[:, list(features)].to_numpy(
        dtype=dtype,
        copy=False,
    )
    return _TimestampPositionInput(
        assets=assets,
        asset_to_offset={asset: offset for offset, asset in enumerate(assets)},
        feature_values={
            feature: feature_matrix[:, offset]
            for offset, feature in enumerate(features)
        },
        label_values=timestamp_frame[return_source_column].to_numpy(
            dtype=dtype,
            copy=False,
        ),
        risk_values=(
            timestamp_frame[str(weight_config.risk_column)].to_numpy(
                dtype=dtype,
                copy=False,
            )
            if weight_config.method == "inverse_vol_scaled_quantile"
            else None
        ),
        eligible=_optional_bool_array(
            timestamp_frame,
            tradability_config.eligibility_column,
        ),
        can_enter=_optional_bool_array(
            timestamp_frame,
            tradability_config.can_enter_column,
        ),
        can_exit=_optional_bool_array(
            timestamp_frame,
            tradability_config.can_exit_column,
        ),
    )

def _validate_value_dtype(value_dtype: str) -> np.dtype:
    if value_dtype not in {"float32", "float64"}:
        raise ValueError("value_dtype must be one of ['float32', 'float64'].")
    return np.dtype(value_dtype)

__all__ = [
    "TensorPortfolioArtifacts",
    "build_tensor_portfolio_artifacts",
    "build_tensor_weighted_positions",
    "_TimestampPositionInput",
    "_build_tensor_feature_positions",
    "_tensor_tradability_diagnostic_counts",
    "_assign_ranked_position_weights",
    "_empty_position_columns",
    "_append_position_columns",
    "_append_position_columns_unprofiled",
    "_selected_assets",
    "_prepare_timestamp_position_input",
    "_validate_value_dtype",
]
