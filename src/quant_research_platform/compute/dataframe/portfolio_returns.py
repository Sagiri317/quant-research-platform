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
    active_position_map,
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

    return _stream_weighted_portfolio_from_validated_panel(
        panel,
        spec,
        weight_config,
        segment_columns=segment_columns,
        portfolio_signals=portfolio_signals,
        tradability_config=tradability_config,
        cost_config=cost_config,
        return_source_column=return_source_column,
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
            raise ValueError(
                f"panel is missing columns: ['{return_source_column}']."
            )
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
            if any(
                position != 0.0 and asset not in label_lookup
                for asset, position in previous.items()
            ):
                raise ValueError("held_asset_missing_from_portfolio_frame")
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
            return_records.append(
                _build_portfolio_return_record_from_positions(
                    group_values=base_values,
                    portfolio_mode=weight_config.portfolio_mode,
                    positions=current,
                    label_lookup=label_lookup,
                    return_source_column=return_source_column,
                    turnover=turnover,
                    cost=float(costs["total_cost"]),
                    segment_columns=segment_columns,
                )
            )
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
) -> dict[str, object]:
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
    "simulate_weighted_long_short_returns_and_cost_from_validated_panel",
]
