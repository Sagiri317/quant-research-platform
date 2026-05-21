"""Tensor-backed portfolio path construction."""

from __future__ import annotations

from dataclasses import dataclass
from operator import itemgetter

import numpy as np
import pandas as pd

from quant_research_platform.core.contracts import (
    CostConfig,
    TradabilityConfig,
    WeightConfig,
)
from quant_research_platform.core.panel import PanelSpec
from quant_research_platform.core.profiling import measure_stage
from quant_research_platform.engines.dataframe.portfolio import (
    COST_BREAKDOWN_COLUMNS,
    EXPOSURE_SUMMARY_COLUMNS,
    POSITION_COLUMNS,
    RETURNS_COLUMNS,
    TURNOVER_COLUMNS,
    WEIGHT_DIAGNOSTIC_COLUMNS,
    _cost_components,
    _normalize_group_key,
    _select_features,
    _selection_count,
    _validate_segment_columns,
    _validate_tradability_columns,
    _validate_weight_columns,
)
from quant_research_platform.engines.tensor.portfolio_helpers import (
    _apply_array_tradability_constraints,
    _filter_panel_group,
    _leg_weight_values,
    _optional_bool_array,
    _previous_position_vector,
    _ranked_asset_indices,
)


@dataclass(frozen=True)
class _TimestampPositionInput:
    assets: tuple[str, ...]
    asset_to_offset: dict[str, int]
    feature_values: dict[str, np.ndarray]
    label_values: np.ndarray
    risk_values: np.ndarray | None
    eligible: np.ndarray | None
    can_enter: np.ndarray | None
    can_exit: np.ndarray | None


@dataclass(frozen=True)
class TensorPortfolioArtifacts:
    """Internal tensor portfolio artifacts built in one streaming pass."""

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
    pnl_return_column: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build timestamp-level positions from tensor signal arrays.

    The first tensor engine keeps the public artifact contract identical to the
    DataFrame engine, so it returns the same positions and diagnostics frames.
    """

    artifacts = build_tensor_portfolio_artifacts(
        panel,
        spec,
        weight_config,
        segment_columns=segment_columns,
        portfolio_features=portfolio_features,
        tradability_config=tradability_config,
        pnl_return_column=pnl_return_column,
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
    pnl_return_column: str | None = None,
) -> TensorPortfolioArtifacts:
    """Build tensor portfolio artifacts without retaining dense path matrices."""

    _validate_segment_columns(panel, segment_columns)
    _validate_weight_columns(panel, weight_config)
    if cost_config is None:
        cost_config = CostConfig()
    if tradability_config is None:
        tradability_config = TradabilityConfig()
    _validate_tradability_columns(panel, tradability_config)
    pnl_return_column = pnl_return_column or spec.label_column
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
            lambda timestamp_frame=timestamp_frame: (
                _prepare_timestamp_position_input(
                    timestamp_frame=timestamp_frame,
                    spec=spec,
                    features=features,
                    weight_config=weight_config,
                    tradability_config=tradability_config,
                    pnl_return_column=pnl_return_column,
                )
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
            diagnostic, position_values, previous_values = (
                measure_stage(
                    "metric",
                    "positions.rank_select",
                    lambda feature=feature,
                    cached_previous_values=cached_previous_values,
                    previous_map=previous_map,
                    timestamp_input=timestamp_input: _build_tensor_feature_positions(
                        timestamp_input=timestamp_input,
                        feature=feature,
                        weight_config=weight_config,
                        tradability_config=tradability_config,
                        previous_positions=previous_map,
                        cached_previous_values=cached_previous_values,
                    ),
                    sample_memory=False,
                )
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
                        label=pnl_return_column,
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
        columns=[*segment_columns, *_EXPOSURE_PATH_COLUMNS],
    )
    return TensorPortfolioArtifacts(
        positions=positions,
        diagnostics=diagnostics,
        turnover=turnover,
        cost_breakdown=cost_breakdown,
        returns=returns,
        exposure_path=exposure_path,
    )


_EXPOSURE_PATH_COLUMNS = [
    "timestamp",
    "feature",
    "gross_exposure",
    "net_exposure",
    "position_count",
]


def summarize_tensor_exposure_from_path(
    exposure_path: pd.DataFrame,
    returns: pd.DataFrame,
    *,
    segment_columns: tuple[str, ...] = (),
) -> pd.DataFrame:
    """Summarize Tensor exposure from the streamed timestamp exposure path."""

    output_columns = [*segment_columns, *EXPOSURE_SUMMARY_COLUMNS]
    if exposure_path.empty:
        return pd.DataFrame(columns=output_columns)
    group_columns = [*segment_columns, "feature"]
    exposure_summary = (
        exposure_path.groupby(group_columns, sort=True, dropna=False)
        .agg(
            mean_gross_exposure=("gross_exposure", "mean"),
            mean_net_exposure=("net_exposure", "mean"),
            max_gross_exposure=("gross_exposure", "max"),
            max_abs_net_exposure=("net_exposure", lambda values: values.abs().max()),
            mean_position_count=("position_count", "mean"),
        )
        .reset_index()
    )
    count_summary = (
        returns.groupby(group_columns, sort=True, dropna=False)
        .agg(
            mean_long_count=("long_count", "mean"),
            mean_short_count=("short_count", "mean"),
        )
        .reset_index()
        if not returns.empty
        else pd.DataFrame(
            columns=[*group_columns, "mean_long_count", "mean_short_count"],
        )
    )
    summary = exposure_summary.merge(
        count_summary,
        on=group_columns,
        how="left",
    )
    summary["mean_long_count"] = pd.to_numeric(
        summary["mean_long_count"],
        errors="coerce",
    )
    summary["mean_short_count"] = pd.to_numeric(
        summary["mean_short_count"],
        errors="coerce",
    )
    return summary[output_columns]


def calculate_tensor_turnover_and_cost(
    positions: pd.DataFrame,
    weight_config: WeightConfig,
    cost_config: CostConfig | None = None,
    *,
    segment_columns: tuple[str, ...] = (),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calculate turnover and cost from tensor-shaped position matrices."""

    if cost_config is None:
        cost_config = CostConfig()
    if positions.empty:
        return (
            pd.DataFrame(columns=[*segment_columns, *TURNOVER_COLUMNS]),
            pd.DataFrame(columns=[*segment_columns, *COST_BREAKDOWN_COLUMNS]),
        )
    turnover_records: list[dict[str, object]] = []
    cost_records: list[dict[str, object]] = []
    group_columns = [*segment_columns, "feature"]
    for group_key, group in positions.groupby(group_columns, sort=True, dropna=False):
        group_values = _normalize_group_key(group_key, group_columns)
        matrix = (
            group.pivot(index="timestamp", columns="asset", values="position")
            .fillna(0.0)
            .sort_index()
        )
        values = matrix.to_numpy(dtype="float64", copy=True)
        changes = np.empty_like(values)
        if len(values) == 0:
            continue
        changes[0] = values[0]
        changes[1:] = values[1:] - values[:-1]
        buy_turnover = np.clip(changes, 0.0, None).sum(axis=1)
        sell_turnover = (-np.clip(changes, None, 0.0)).sum(axis=1)
        turnover = buy_turnover + sell_turnover
        for row_offset, timestamp in enumerate(matrix.index):
            base = {
                **{column: group_values[column] for column in segment_columns},
                "timestamp": timestamp,
                "feature": group_values["feature"],
            }
            costs = _cost_components(
                buy_turnover=float(buy_turnover[row_offset]),
                sell_turnover=float(sell_turnover[row_offset]),
                turnover=float(turnover[row_offset]),
                weight_config=weight_config,
                cost_config=cost_config,
            )
            turnover_records.append(
                {
                    **base,
                    "turnover": float(turnover[row_offset]),
                }
            )
            cost_records.append(
                {
                    **base,
                    "buy_turnover": float(buy_turnover[row_offset]),
                    "sell_turnover": float(sell_turnover[row_offset]),
                    "turnover": float(turnover[row_offset]),
                    **costs,
                }
            )
    return (
        pd.DataFrame.from_records(
            turnover_records,
            columns=[*segment_columns, *TURNOVER_COLUMNS],
        ),
        pd.DataFrame.from_records(
            cost_records,
            columns=[*segment_columns, *COST_BREAKDOWN_COLUMNS],
        ),
    )


def calculate_tensor_long_short(
    *,
    panel: pd.DataFrame,
    spec: PanelSpec,
    positions: pd.DataFrame,
    turnover: pd.DataFrame,
    cost_breakdown: pd.DataFrame,
    weight_config: WeightConfig | None = None,
    segment_columns: tuple[str, ...] = (),
) -> pd.DataFrame:
    """Calculate portfolio return path from tensor-shaped positions."""

    if weight_config is None:
        weight_config = WeightConfig()
    if positions.empty:
        return pd.DataFrame(columns=[*segment_columns, *RETURNS_COLUMNS])
    records: list[dict[str, object]] = []
    metric_index = [*segment_columns, "timestamp", "feature"]
    turnover_lookup = turnover.set_index(metric_index)["turnover"].to_dict()
    cost_lookup = cost_breakdown.set_index(metric_index)["total_cost"].to_dict()
    group_columns = [*segment_columns, "feature"]
    for group_key, group in positions.groupby(group_columns, sort=True, dropna=False):
        group_values = _normalize_group_key(group_key, group_columns)
        position_matrix = (
            group.pivot(index="timestamp", columns="asset", values="position")
            .fillna(0.0)
            .sort_index()
        )
        panel_group = _filter_panel_group(
            panel=panel,
            segment_columns=segment_columns,
            group_values=group_values,
        )
        label_matrix = (
            panel_group.pivot(
                index=spec.timestamp_column,
                columns=spec.asset_column,
                values=spec.label_column,
            )
            .reindex(index=position_matrix.index, columns=position_matrix.columns)
            .sort_index()
        )
        position_values = position_matrix.to_numpy(dtype="float64", copy=False)
        label_values = label_matrix.to_numpy(dtype="float64", copy=False)
        valid = np.isfinite(label_values)
        long_mask = (position_values > 0) & valid
        short_mask = (position_values < 0) & valid
        long_counts = long_mask.sum(axis=1)
        short_counts = short_mask.sum(axis=1)
        long_returns = np.where(long_mask, position_values * label_values, 0.0).sum(
            axis=1
        )
        short_returns = np.where(short_mask, -position_values * label_values, 0.0).sum(
            axis=1
        )
        gross_returns = (
            long_returns - short_returns
            if weight_config.portfolio_mode == "long_short"
            else long_returns
        )
        for row_offset, timestamp in enumerate(position_matrix.index):
            if long_counts[row_offset] == 0 or (
                weight_config.portfolio_mode == "long_short"
                and short_counts[row_offset] == 0
            ):
                continue
            metric_key = tuple(
                [
                    *[group_values[column] for column in segment_columns],
                    timestamp,
                    group_values["feature"],
                ]
            )
            cost = float(cost_lookup[metric_key])
            records.append(
                {
                    **{column: group_values[column] for column in segment_columns},
                    "timestamp": timestamp,
                    "feature": group_values["feature"],
                    "label": spec.label_column,
                    "portfolio_mode": weight_config.portfolio_mode,
                    "long_return": float(long_returns[row_offset]),
                    "short_return": float(short_returns[row_offset]),
                    "gross_portfolio_return": float(gross_returns[row_offset]),
                    "turnover": float(turnover_lookup[metric_key]),
                    "cost": cost,
                    "net_portfolio_return": float(gross_returns[row_offset] - cost),
                    "long_count": int(long_counts[row_offset]),
                    "short_count": int(short_counts[row_offset]),
                }
            )
    return pd.DataFrame.from_records(
        records,
        columns=[*segment_columns, *RETURNS_COLUMNS],
    )


def _path_turnover_and_cost_records(
    *,
    context: dict[str, object],
    timestamp: object,
    feature: str,
    previous_positions: dict[str, float],
    current_positions: dict[str, float],
    weight_config: WeightConfig,
    cost_config: CostConfig,
    segment_columns: tuple[str, ...] = (),
) -> tuple[dict[str, object], dict[str, object]]:
    all_assets = set(previous_positions) | set(current_positions)
    if all_assets:
        changes = np.fromiter(
            (
                float(current_positions.get(asset, 0.0))
                - float(previous_positions.get(asset, 0.0))
                for asset in all_assets
            ),
            dtype="float64",
            count=len(all_assets),
        )
        buy_turnover = float(np.clip(changes, 0.0, None).sum())
        sell_turnover = float((-np.clip(changes, None, 0.0)).sum())
    else:
        buy_turnover = 0.0
        sell_turnover = 0.0
    turnover = buy_turnover + sell_turnover
    base = {
        **{column: context[column] for column in segment_columns},
        "timestamp": timestamp,
        "feature": feature,
    }
    costs = _cost_components(
        buy_turnover=buy_turnover,
        sell_turnover=sell_turnover,
        turnover=turnover,
        weight_config=weight_config,
        cost_config=cost_config,
    )
    return (
        {
            **base,
            "turnover": turnover,
        },
        {
            **base,
            "buy_turnover": buy_turnover,
            "sell_turnover": sell_turnover,
            "turnover": turnover,
            **costs,
        },
    )


def _path_turnover_and_cost_records_from_arrays(
    *,
    context: dict[str, object],
    timestamp: object,
    feature: str,
    previous_values: np.ndarray,
    current_values: np.ndarray,
    previous_positions: dict[str, float],
    asset_to_offset: dict[str, int],
    weight_config: WeightConfig,
    cost_config: CostConfig,
    segment_columns: tuple[str, ...] = (),
    previous_assets_match: bool = False,
) -> tuple[dict[str, object], dict[str, object]]:
    changes = current_values - previous_values
    buy_turnover = float(np.clip(changes, 0.0, None).sum())
    sell_turnover = float((-np.clip(changes, None, 0.0)).sum())
    if previous_positions and not previous_assets_match:
        missing_previous = np.fromiter(
            (
                float(position)
                for asset, position in previous_positions.items()
                if asset not in asset_to_offset
            ),
            dtype="float64",
        )
        if len(missing_previous) > 0:
            missing_changes = -missing_previous
            buy_turnover += float(np.clip(missing_changes, 0.0, None).sum())
            sell_turnover += float((-np.clip(missing_changes, None, 0.0)).sum())
    turnover = buy_turnover + sell_turnover
    base = {
        **{column: context[column] for column in segment_columns},
        "timestamp": timestamp,
        "feature": feature,
    }
    costs = _cost_components(
        buy_turnover=buy_turnover,
        sell_turnover=sell_turnover,
        turnover=turnover,
        weight_config=weight_config,
        cost_config=cost_config,
    )
    return (
        {
            **base,
            "turnover": turnover,
        },
        {
            **base,
            "buy_turnover": buy_turnover,
            "sell_turnover": sell_turnover,
            "turnover": turnover,
            **costs,
        },
    )


def _long_short_record_from_arrays(
    *,
    context: dict[str, object],
    timestamp: object,
    feature: str,
    label: str,
    portfolio_mode: str,
    position_values: np.ndarray,
    label_values: np.ndarray,
    turnover: float,
    cost: float,
    segment_columns: tuple[str, ...],
) -> dict[str, object] | None:
    valid = np.isfinite(label_values)
    long_mask = (position_values > 0) & valid
    short_mask = (position_values < 0) & valid
    long_count = int(long_mask.sum())
    short_count = int(short_mask.sum())
    if long_count == 0 or (portfolio_mode == "long_short" and short_count == 0):
        return None
    long_return = float(np.where(long_mask, position_values * label_values, 0.0).sum())
    short_return = float(
        np.where(short_mask, -position_values * label_values, 0.0).sum()
    )
    gross_return = (
        long_return - short_return if portfolio_mode == "long_short" else long_return
    )
    return {
        **{column: context[column] for column in segment_columns},
        "timestamp": timestamp,
        "feature": feature,
        "label": label,
        "portfolio_mode": portfolio_mode,
        "long_return": long_return,
        "short_return": short_return,
        "gross_portfolio_return": gross_return,
        "turnover": turnover,
        "cost": cost,
        "net_portfolio_return": gross_return - cost,
        "long_count": long_count,
        "short_count": short_count,
    }


def _build_path_records_from_arrays(
    *,
    context: dict[str, object],
    timestamp: object,
    feature: str,
    label: str,
    assets: tuple[str, ...],
    position_values: np.ndarray,
    label_values: np.ndarray,
    previous_values: np.ndarray,
    previous_positions: dict[str, float],
    asset_to_offset: dict[str, int],
    weight_config: WeightConfig,
    cost_config: CostConfig,
    segment_columns: tuple[str, ...],
    previous_assets_match: bool = False,
) -> tuple[
    np.ndarray,
    dict[str, object],
    dict[str, object],
    dict[str, object] | None,
    dict[str, object],
]:
    nonzero_offsets, exposure_values = measure_stage(
        "metric",
        "positions.path_current_state",
        lambda: _path_current_state(
            assets=assets,
            position_values=position_values,
        ),
        sample_memory=False,
    )
    turnover_record, cost_record = measure_stage(
        "metric",
        "positions.path_turnover_cost",
        lambda: _path_turnover_and_cost_records_from_arrays(
            context=context,
            timestamp=timestamp,
            feature=feature,
            previous_values=previous_values,
            current_values=position_values,
            previous_positions=previous_positions,
            asset_to_offset=asset_to_offset,
            weight_config=weight_config,
            cost_config=cost_config,
            segment_columns=segment_columns,
            previous_assets_match=previous_assets_match,
        ),
        sample_memory=False,
    )
    long_short_record = measure_stage(
        "metric",
        "positions.path_long_short",
        lambda: _long_short_record_from_arrays(
            context=context,
            timestamp=timestamp,
            feature=feature,
            label=label,
            portfolio_mode=weight_config.portfolio_mode,
            position_values=position_values,
            label_values=label_values,
            turnover=float(turnover_record["turnover"]),
            cost=float(cost_record["total_cost"]),
            segment_columns=segment_columns,
        ),
        sample_memory=False,
    )
    exposure_record = measure_stage(
        "metric",
        "positions.path_exposure",
        lambda: _exposure_record_from_arrays(
            context=context,
            timestamp=timestamp,
            feature=feature,
            exposure_values=exposure_values,
            segment_columns=segment_columns,
        ),
        sample_memory=False,
    )
    return (
        nonzero_offsets,
        turnover_record,
        cost_record,
        long_short_record,
        exposure_record,
    )


def _path_current_state(
    *,
    assets: tuple[str, ...],
    position_values: np.ndarray,
) -> tuple[np.ndarray, dict[str, float | int]]:
    nonzero_offsets = np.flatnonzero(position_values != 0.0)
    nonzero_positions = position_values[nonzero_offsets]
    return nonzero_offsets, {
        "gross_exposure": float(np.abs(nonzero_positions).sum()),
        "net_exposure": float(nonzero_positions.sum()),
        "position_count": len(nonzero_offsets),
    }


def _exposure_record_from_arrays(
    *,
    context: dict[str, object],
    timestamp: object,
    feature: str,
    exposure_values: dict[str, float | int],
    segment_columns: tuple[str, ...],
) -> dict[str, object]:
    return {
        **{column: context[column] for column in segment_columns},
        "timestamp": timestamp,
        "feature": feature,
        **exposure_values,
    }


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
        "long_count": int(long_count),
        "short_count": int(short_count),
        "fallback_reason": fallback_reason,
    }
    return diagnostic, positions, previous_values


def _assign_ranked_position_weights(
    *,
    assets: tuple[str, ...],
    feature: str,
    feature_values: np.ndarray,
    risk_values: np.ndarray | None,
    ranked_indices: np.ndarray,
    weight_config: WeightConfig,
) -> tuple[np.ndarray, int, int, str]:
    positions = np.zeros(len(assets), dtype="float64")
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


def _current_position_state(
    *,
    assets: tuple[str, ...],
    positions: np.ndarray,
    nonzero_offsets: np.ndarray,
) -> tuple[dict[str, float], dict[str, float | int]]:
    nonzero_positions = positions[nonzero_offsets]
    current = {
        assets[int(offset)]: float(position)
        for offset, position in zip(nonzero_offsets, nonzero_positions, strict=True)
    }
    return current, {
        "gross_exposure": float(np.abs(nonzero_positions).sum()),
        "net_exposure": float(nonzero_positions.sum()),
        "position_count": len(nonzero_offsets),
    }


def _position_map_from_arrays(
    *,
    assets: tuple[str, ...],
    positions: np.ndarray,
) -> dict[str, float]:
    nonzero_offsets = np.flatnonzero(positions != 0.0)
    return {
        assets[int(offset)]: float(positions[int(offset)])
        for offset in nonzero_offsets
    }


def _prepare_timestamp_position_input(
    *,
    timestamp_frame: pd.DataFrame,
    spec: PanelSpec,
    features: tuple[str, ...],
    weight_config: WeightConfig,
    tradability_config: TradabilityConfig,
    pnl_return_column: str | None = None,
) -> _TimestampPositionInput:
    pnl_return_column = pnl_return_column or spec.label_column
    asset_values = timestamp_frame[spec.asset_column].to_numpy(
        dtype=str,
        copy=False,
    )
    assets = tuple(asset_values.tolist())
    feature_matrix = timestamp_frame.loc[:, list(features)].to_numpy(
        dtype="float64",
        copy=False,
    )
    return _TimestampPositionInput(
        assets=assets,
        asset_to_offset={asset: offset for offset, asset in enumerate(assets)},
        feature_values={
            feature: feature_matrix[:, offset]
            for offset, feature in enumerate(features)
        },
        label_values=timestamp_frame[pnl_return_column].to_numpy(
            dtype="float64",
            copy=False,
        ),
        risk_values=(
            timestamp_frame[str(weight_config.risk_column)].to_numpy(
                dtype="float64",
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


