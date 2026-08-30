"""DataFrame portfolio return algorithms."""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant_research_platform.compute._portfolio_math import (
    cost_components,
    require_return_source_column,
    select_signals,
    validate_tradability_columns,
    validate_weight_columns,
)
from quant_research_platform.compute.dataframe.constraints import (
    build_tradability_context,
)
from quant_research_platform.compute.dataframe.portfolio_constants import (
    COST_BREAKDOWN_COLUMNS,
    POSITION_COLUMNS,
    RETURNS_COLUMNS,
    TURNOVER_COLUMNS,
    WEIGHT_DIAGNOSTIC_COLUMNS,
)
from quant_research_platform.compute.dataframe.position_math import (
    build_signal_positions,
)
from quant_research_platform.compute.dataframe.turnover import (
    position_change_turnover,
)
from quant_research_platform.compute.grouping import normalize_group_key
from quant_research_platform.compute.panels import (
    ComputePanelSpec,
    validate_compute_panel,
)
from quant_research_platform.contracts.portfolios import (
    CostConfig,
    TradabilityConfig,
    WeightConfig,
)
from quant_research_platform.contracts.validation import strict_finite_numeric_series


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
    previous_positions: dict[tuple[object, ...], dict[str, dict[str, float]]]
    | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Stream positions, turnover, costs, and portfolio returns."""

    panel = validate_compute_panel(
        panel,
        spec,
        segment_columns=segment_columns,
    )
    validate_weight_columns(panel, weight_config)
    if tradability_config is None:
        tradability_config = TradabilityConfig()
    if cost_config is None:
        cost_config = CostConfig()
    validate_tradability_columns(panel, tradability_config)
    return_source_column = require_return_source_column(return_source_column)
    if return_source_column != spec.label_column:
        if return_source_column not in panel:
            raise ValueError(f"panel is missing columns: ['{return_source_column}'].")
        if not pd.api.types.is_numeric_dtype(panel[return_source_column]):
            raise TypeError(
                f"return source column {return_source_column!r} must be numeric."
            )
        panel[return_source_column] = strict_finite_numeric_series(
            panel[return_source_column],
            error=(
                f"portfolio return source {return_source_column!r} "
                "contains invalid values."
            ),
            allow_null=True,
        ).astype("float64")
    signals = select_signals(spec, portfolio_signals)
    group_columns = [*segment_columns, spec.timestamp_column]

    position_columns = [*segment_columns, *POSITION_COLUMNS]
    turnover_columns = [*segment_columns, *TURNOVER_COLUMNS]
    returns_columns = [*segment_columns, *RETURNS_COLUMNS]
    diagnostic_columns = [*segment_columns, *WEIGHT_DIAGNOSTIC_COLUMNS]
    cost_columns = [*segment_columns, *COST_BREAKDOWN_COLUMNS]
    position_data = {column: [] for column in position_columns}
    turnover_data = {column: [] for column in turnover_columns}
    return_data = {column: [] for column in returns_columns}
    diagnostic_data = {column: [] for column in diagnostic_columns}
    cost_data = {column: [] for column in cost_columns}
    if previous_positions is None:
        previous_positions = {}
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
            if any(
                position != 0.0 and asset not in label_lookup
                for asset, position in previous.items()
            ):
                raise ValueError("held_asset_missing_from_portfolio_frame")
            positions, diagnostic = build_signal_positions(
                timestamp_frame=timestamp_frame,
                spec=spec,
                signal=signal,
                weight_config=weight_config,
                tradability_config=tradability_config,
                tradability_context=tradability_context,
                previous_positions=previous,
            )
            current = {
                asset: float(position)
                for asset, position in positions.items()
                if float(position) != 0.0
            }
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
            position_items = sorted(current.items())
            if not position_items:
                marker_asset = next(iter(positions))
                position_items = [(marker_asset, 0.0)]
            position_count = len(position_items)
            for column in segment_columns:
                position_data[column].extend([group_values[column]] * position_count)
            position_data["timestamp"].extend([timestamp] * position_count)
            position_data["asset"].extend(asset for asset, _ in position_items)
            position_data["signal"].extend([signal] * position_count)
            position_data["position"].extend(
                float(position) for _, position in position_items
            )
            for column in segment_columns:
                value = group_values[column]
                turnover_data[column].append(value)
                return_data[column].append(value)
                diagnostic_data[column].append(value)
                cost_data[column].append(value)
            turnover_data["timestamp"].append(timestamp)
            turnover_data["signal"].append(signal)
            turnover_data["turnover"].append(turnover)
            diagnostic_data["timestamp"].append(timestamp)
            for column in WEIGHT_DIAGNOSTIC_COLUMNS[1:]:
                diagnostic_data[column].append(diagnostic[column])
            cost_data["timestamp"].append(timestamp)
            cost_data["signal"].append(signal)
            cost_data["buy_turnover"].append(buy_turnover)
            cost_data["sell_turnover"].append(sell_turnover)
            cost_data["turnover"].append(turnover)
            for column in COST_BREAKDOWN_COLUMNS[5:]:
                cost_data[column].append(costs[column])
            long_return, short_return, gross_return, long_count, short_count = (
                _portfolio_return_components(
                    portfolio_mode=weight_config.portfolio_mode,
                    positions=current,
                    label_lookup=label_lookup,
                )
            )
            cost = float(costs["total_cost"])
            return_data["timestamp"].append(timestamp)
            return_data["signal"].append(signal)
            return_data["return_source_column"].append(return_source_column)
            return_data["portfolio_mode"].append(weight_config.portfolio_mode)
            return_data["long_return"].append(long_return)
            return_data["short_return"].append(short_return)
            return_data["gross_portfolio_return"].append(gross_return)
            return_data["turnover"].append(float(turnover))
            return_data["cost"].append(cost)
            return_data["net_portfolio_return"].append(gross_return - cost)
            return_data["long_count"].append(long_count)
            return_data["short_count"].append(short_count)
            signal_previous[signal] = current

    return (
        _frame_from_column_buffers(position_data, position_columns),
        _frame_from_column_buffers(turnover_data, turnover_columns),
        _frame_from_column_buffers(return_data, returns_columns),
        _frame_from_column_buffers(diagnostic_data, diagnostic_columns),
        _frame_from_column_buffers(cost_data, cost_columns),
    )


def _frame_from_column_buffers(
    data: dict[str, list[object]],
    columns: list[str],
) -> pd.DataFrame:
    if not columns or not data[columns[0]]:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(data, columns=columns)


def _timestamp_label_lookup(
    timestamp_frame: pd.DataFrame,
    spec: ComputePanelSpec,
    return_source_column: str,
) -> dict[str, float]:
    assets = timestamp_frame[spec.asset_column].to_numpy(copy=False)
    labels = timestamp_frame[return_source_column].to_numpy(
        dtype="float64",
        copy=False,
    )
    return dict(zip(assets, labels, strict=True))


def _portfolio_return_components(
    *,
    portfolio_mode: str,
    positions: dict[str, float],
    label_lookup: dict[str, float],
) -> tuple[float, float, float, int, int]:
    long_return = 0.0
    short_return = 0.0
    long_count = 0
    short_count = 0
    for asset, position in positions.items():
        label = label_lookup.get(asset, np.nan)
        if not np.isfinite(label):
            raise ValueError("nonfinite_active_position_return")
        if position > 0:
            long_count += 1
            long_return += position * label
        elif position < 0:
            short_count += 1
            short_return += -position * label
    gross_return = (
        long_return - short_return if portfolio_mode == "long_short" else long_return
    )
    return (
        float(long_return),
        float(short_return),
        float(gross_return),
        int(long_count),
        int(short_count),
    )
