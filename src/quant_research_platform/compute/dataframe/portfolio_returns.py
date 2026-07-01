"""DataFrame portfolio return algorithms."""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant_research_platform.compute.dataframe.constraints import (
    build_tradability_context,
    validate_tradability_columns,
)
from quant_research_platform.compute.dataframe.portfolio_constants import (
    COST_BREAKDOWN_COLUMNS,
    POSITION_COLUMNS,
    RETURNS_COLUMNS,
    TURNOVER_COLUMNS,
    WEIGHT_DIAGNOSTIC_COLUMNS,
)
from quant_research_platform.compute.dataframe.portfolio_helpers import (
    cost_components,
    normalize_group_key,
    select_signal,
    select_signals,
    validate_segment_columns,
    validate_weight_columns,
)
from quant_research_platform.compute.dataframe.position_math import (
    build_signal_positions,
)
from quant_research_platform.compute.dataframe.turnover import (
    active_position_map,
    position_change_turnover,
)
from quant_research_platform.compute.panels import (
    ComputePanelSpec,
    validate_compute_panel,
)
from quant_research_platform.contracts.portfolios import (
    CostConfig,
    TradabilityConfig,
    WeightConfig,
)
from quant_research_platform.runtime.profiling import measure_stage


def simulate_weighted_long_short_returns(
    frame: pd.DataFrame,
    spec: ComputePanelSpec,
    weight_config: WeightConfig,
    *,
    segment_columns: tuple[str, ...] = (),
    portfolio_signals: tuple[str, ...] | None = None,
    tradability_config: TradabilityConfig | None = None,
    cost_config: CostConfig | None = None,
    return_source_column: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Simulate portfolio returns from weighted positions."""

    return_source_column = _require_return_source_column(return_source_column)
    panel = validate_compute_panel(frame, spec, segment_columns=segment_columns)
    return _simulate_weighted_long_short_returns_from_validated_panel(
        panel,
        spec,
        weight_config,
        segment_columns=segment_columns,
        portfolio_signals=portfolio_signals,
        tradability_config=tradability_config,
        cost_config=cost_config,
        return_source_column=return_source_column,
    )

def _simulate_weighted_long_short_returns_from_validated_panel(
    panel: pd.DataFrame,
    spec: ComputePanelSpec,
    weight_config: WeightConfig,
    *,
    segment_columns: tuple[str, ...] = (),
    portfolio_signals: tuple[str, ...] | None = None,
    tradability_config: TradabilityConfig | None = None,
    cost_config: CostConfig | None = None,
    return_source_column: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Simulate returns from a validated timestamp-asset panel."""

    positions, turnover, returns, diagnostics, _cost_breakdown = (
        simulate_weighted_long_short_returns_and_cost_from_validated_panel(
            panel,
            spec,
            weight_config,
            segment_columns=segment_columns,
            portfolio_signals=portfolio_signals,
            tradability_config=tradability_config,
            cost_config=cost_config,
            return_source_column=return_source_column,
        )
    )
    return positions, turnover, returns, diagnostics

def simulate_weighted_long_short_returns_and_cost_from_validated_panel(
    panel: pd.DataFrame,
    spec: ComputePanelSpec,
    weight_config: WeightConfig,
    *,
    segment_columns: tuple[str, ...] = (),
    portfolio_signals: tuple[str, ...] | None = None,
    tradability_config: TradabilityConfig | None = None,
    cost_config: CostConfig | None = None,
    return_source_column: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Simulate returns and cost breakdown artifacts."""

    return measure_stage(
        "metric",
        "portfolio_stream",
        lambda: _stream_weighted_portfolio_from_validated_panel(
            panel,
            spec,
            weight_config,
            segment_columns=segment_columns,
            portfolio_signals=portfolio_signals,
            tradability_config=tradability_config,
            cost_config=cost_config,
            return_source_column=return_source_column,
        ),
    )

def _stream_weighted_portfolio_from_validated_panel(
    panel: pd.DataFrame,
    spec: ComputePanelSpec,
    weight_config: WeightConfig,
    *,
    segment_columns: tuple[str, ...] = (),
    portfolio_signals: tuple[str, ...] | None = None,
    tradability_config: TradabilityConfig | None = None,
    cost_config: CostConfig | None = None,
    return_source_column: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Stream positions, turnover, costs, and portfolio returns."""

    validate_segment_columns(panel, segment_columns)
    validate_weight_columns(panel, weight_config)
    if tradability_config is None:
        tradability_config = TradabilityConfig()
    if cost_config is None:
        cost_config = CostConfig()
    validate_tradability_columns(panel, tradability_config)
    return_source_column = _require_return_source_column(return_source_column)
    signals = select_signals(spec, portfolio_signals)
    group_columns = [*segment_columns, spec.timestamp_column]

    position_records: list[dict[str, object]] = []
    diagnostic_records: list[dict[str, object]] = []
    turnover_records: list[dict[str, object]] = []
    cost_records: list[dict[str, object]] = []
    return_records: list[dict[str, object]] = []
    previous_positions: dict[tuple[object, ...], dict[str, dict[str, float]]] = {}
    for group_key, timestamp_frame in panel.groupby(group_columns, sort=True):
        group_values = normalize_group_key(group_key, group_columns)
        timestamp = group_values[spec.timestamp_column]
        segment_key = tuple(group_values[column] for column in segment_columns)
        signal_previous = previous_positions.setdefault(segment_key, {})
        label_lookup = _timestamp_label_lookup(
            timestamp_frame,
            spec,
            return_source_column,
        )
        tradability_context = build_tradability_context(
            timestamp_frame=timestamp_frame,
            spec=spec,
            tradability_config=tradability_config,
        )
        for signal in signals:
            previous = signal_previous.get(signal, {})
            records, diagnostic = build_signal_positions(
                timestamp_frame=timestamp_frame,
                spec=spec,
                signal=signal,
                weight_config=weight_config,
                tradability_config=tradability_config,
                tradability_context=tradability_context,
                group_values=group_values,
                previous_positions=previous,
            )
            current = active_position_map(records)
            buy_turnover, sell_turnover = position_change_turnover(
                previous=previous,
                current=current,
            )
            turnover = buy_turnover + sell_turnover
            costs = cost_components(
                buy_turnover=buy_turnover,
                sell_turnover=sell_turnover,
                turnover=turnover,
                cost_config=cost_config,
            )
            base_values = {
                **{column: group_values[column] for column in segment_columns},
                "timestamp": timestamp,
                "signal": signal,
            }
            position_records.extend(records)
            diagnostic_records.append(
                {
                    **{column: group_values[column] for column in segment_columns},
                    "timestamp": timestamp,
                    **diagnostic,
                }
            )
            turnover_records.append({**base_values, "turnover": turnover})
            cost_records.append(
                {
                    **base_values,
                    "buy_turnover": buy_turnover,
                    "sell_turnover": sell_turnover,
                    "turnover": turnover,
                    **costs,
                }
            )
            return_record = _build_portfolio_return_record_from_positions(
                group_values=base_values,
                portfolio_mode=weight_config.portfolio_mode,
                positions=current,
                label_lookup=label_lookup,
                return_source_column=return_source_column,
                turnover=turnover,
                cost=float(costs["total_cost"]),
                segment_columns=segment_columns,
            )
            if return_record is not None:
                return_records.append(return_record)
            signal_previous[signal] = current

    position_columns = [*segment_columns, *POSITION_COLUMNS]
    turnover_columns = [*segment_columns, *TURNOVER_COLUMNS]
    returns_columns = [*segment_columns, *RETURNS_COLUMNS]
    diagnostic_columns = [*segment_columns, *WEIGHT_DIAGNOSTIC_COLUMNS]
    cost_columns = [*segment_columns, *COST_BREAKDOWN_COLUMNS]
    return (
        pd.DataFrame.from_records(position_records, columns=position_columns),
        pd.DataFrame.from_records(turnover_records, columns=turnover_columns),
        pd.DataFrame.from_records(return_records, columns=returns_columns),
        pd.DataFrame.from_records(diagnostic_records, columns=diagnostic_columns),
        pd.DataFrame.from_records(cost_records, columns=cost_columns),
    )

def simulate_long_short_returns(
    frame: pd.DataFrame,
    spec: ComputePanelSpec,
    top_quantile: float = 0.2,
    bottom_quantile: float = 0.2,
    cost_config: CostConfig | None = None,
    signal_column: str | None = None,
    return_source_column: str | None = None,
) -> pd.DataFrame:
    """Simulate equal-weight long-short returns with turnover-based costs."""

    signal = select_signal(spec, signal_column)
    _positions, _turnover, returns, _diagnostics = simulate_weighted_long_short_returns(
        frame,
        spec,
        WeightConfig(
            method="equal_weight_quantile",
            top_quantile=top_quantile,
            bottom_quantile=bottom_quantile,
        ),
        cost_config=cost_config,
        portfolio_signals=(signal,),
        return_source_column=return_source_column,
    )
    return returns[RETURNS_COLUMNS]

def _calculate_long_short_from_positions(
    *,
    panel: pd.DataFrame,
    spec: ComputePanelSpec,
    positions: pd.DataFrame,
    turnover: pd.DataFrame,
    cost_breakdown: pd.DataFrame,
    weight_config: WeightConfig,
    segment_columns: tuple[str, ...],
    return_source_column: str | None = None,
) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    return_source_column = _require_return_source_column(return_source_column)
    lookup_columns = [*segment_columns, spec.timestamp_column, spec.asset_column]
    lookup_index = [*segment_columns, "timestamp", "asset"]
    label_lookup = (
        panel[[*lookup_columns, return_source_column]]
        .rename(
            columns={
                spec.timestamp_column: "timestamp",
                spec.asset_column: "asset",
            },
        )
        .set_index(lookup_index)
    )
    position_group_columns = [*segment_columns, "timestamp", "signal"]
    positioned_positions = positions.join(
        label_lookup,
        on=lookup_index,
        how="left",
    )
    metric_index = [*segment_columns, "timestamp", "signal"]
    turnover_lookup = turnover.set_index(metric_index)["turnover"].to_dict()
    cost_lookup = cost_breakdown.set_index(metric_index)["total_cost"].to_dict()
    for group_key, timestamp_positions in positioned_positions.groupby(
        position_group_columns,
        sort=True,
    ):
        group_values = normalize_group_key(group_key, position_group_columns)
        positions_array = timestamp_positions["position"].to_numpy(
            dtype="float64",
            copy=False,
        )
        labels_array = timestamp_positions[return_source_column].to_numpy(
            dtype="float64",
            copy=False,
        )
        valid_labels = np.isfinite(labels_array)
        long_mask = (positions_array > 0) & valid_labels
        short_mask = (positions_array < 0) & valid_labels
        long_count = int(long_mask.sum())
        short_count = int(short_mask.sum())
        if long_count == 0 or (
            weight_config.portfolio_mode == "long_short" and short_count == 0
        ):
            continue
        long_return = float(np.dot(positions_array[long_mask], labels_array[long_mask]))
        short_return = float(
            np.dot(-positions_array[short_mask], labels_array[short_mask])
        )
        gross_return = (
            long_return - short_return
            if weight_config.portfolio_mode == "long_short"
            else long_return
        )
        metric_key = tuple(group_values[column] for column in metric_index)
        turnover_value = float(turnover_lookup[metric_key])
        cost = float(cost_lookup[metric_key])
        records.append(
            {
                **{column: group_values[column] for column in segment_columns},
                "timestamp": group_values["timestamp"],
                "signal": group_values["signal"],
                "return_source_column": return_source_column,
                "portfolio_mode": weight_config.portfolio_mode,
                "long_return": long_return,
                "short_return": short_return,
                "gross_portfolio_return": gross_return,
                "turnover": turnover_value,
                "cost": cost,
                "net_portfolio_return": gross_return - cost,
                "long_count": long_count,
                "short_count": short_count,
            }
        )

    return pd.DataFrame.from_records(
        records,
        columns=[*segment_columns, *RETURNS_COLUMNS],
    )

def _timestamp_label_lookup(
    timestamp_frame: pd.DataFrame,
    spec: ComputePanelSpec,
    return_source_column: str,
) -> dict[str, float]:
    assets = timestamp_frame[spec.asset_column].astype(str).to_numpy(copy=False)
    labels = timestamp_frame[return_source_column].to_numpy(
        dtype="float64",
        copy=False,
    )
    return dict(zip(assets, labels, strict=True))


def _require_return_source_column(return_source_column: str | None) -> str:
    if not isinstance(return_source_column, str) or not return_source_column:
        raise ValueError("portfolio_return_source_column_required")
    return return_source_column

def _build_portfolio_return_record_from_positions(
    *,
    group_values: dict[str, object],
    portfolio_mode: str,
    positions: dict[str, float],
    label_lookup: dict[str, float],
    return_source_column: str,
    turnover: float,
    cost: float,
    segment_columns: tuple[str, ...],
) -> dict[str, object] | None:
    long_return = 0.0
    short_return = 0.0
    long_count = 0
    short_count = 0
    for asset, position in positions.items():
        label = label_lookup.get(asset, np.nan)
        if not np.isfinite(label):
            continue
        if position > 0:
            long_count += 1
            long_return += position * label
        elif position < 0:
            short_count += 1
            short_return += -position * label
    if long_count == 0 or (portfolio_mode == "long_short" and short_count == 0):
        return None
    gross_return = (
        long_return - short_return if portfolio_mode == "long_short" else long_return
    )
    return {
        **{column: group_values[column] for column in segment_columns},
        "timestamp": group_values["timestamp"],
        "signal": group_values["signal"],
        "return_source_column": return_source_column,
        "portfolio_mode": portfolio_mode,
        "long_return": float(long_return),
        "short_return": float(short_return),
        "gross_portfolio_return": float(gross_return),
        "turnover": float(turnover),
        "cost": float(cost),
        "net_portfolio_return": float(gross_return - cost),
        "long_count": int(long_count),
        "short_count": int(short_count),
    }

__all__ = [
    "simulate_long_short_returns",
    "simulate_weighted_long_short_returns",
]
