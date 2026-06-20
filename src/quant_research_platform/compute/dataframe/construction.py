"""DataFrame portfolio construction algorithms."""

from __future__ import annotations

import pandas as pd

from quant_research_platform.compute.dataframe.constraints import (
    _build_tradability_context,
    _validate_tradability_columns,
)
from quant_research_platform.compute.dataframe.portfolio_constants import (
    POSITION_COLUMNS,
    WEIGHT_DIAGNOSTIC_COLUMNS,
)
from quant_research_platform.compute.dataframe.portfolio_helpers import (
    _normalize_group_key,
    _select_signal,
    _select_signals,
    _validate_segment_columns,
    _validate_weight_columns,
)
from quant_research_platform.compute.dataframe.position_math import (
    _build_signal_positions,
)
from quant_research_platform.compute.panels import (
    ComputePanelSpec,
    validate_compute_panel,
)
from quant_research_platform.contracts.portfolios import (
    TradabilityConfig,
    WeightConfig,
)


def build_weighted_positions(
    frame: pd.DataFrame,
    spec: ComputePanelSpec,
    weight_config: WeightConfig,
    *,
    segment_columns: tuple[str, ...] = (),
    portfolio_signals: tuple[str, ...] | None = None,
    tradability_config: TradabilityConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build timestamp-level positions for selected signals."""

    panel = validate_compute_panel(frame, spec, segment_columns=segment_columns)
    return _build_weighted_positions_from_validated_panel(
        panel,
        spec,
        weight_config,
        segment_columns=segment_columns,
        portfolio_signals=portfolio_signals,
        tradability_config=tradability_config,
    )

def _build_weighted_positions_from_validated_panel(
    panel: pd.DataFrame,
    spec: ComputePanelSpec,
    weight_config: WeightConfig,
    *,
    segment_columns: tuple[str, ...] = (),
    portfolio_signals: tuple[str, ...] | None = None,
    tradability_config: TradabilityConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build positions from a validated timestamp-asset panel."""

    _validate_segment_columns(panel, segment_columns)
    _validate_weight_columns(panel, weight_config)
    if tradability_config is None:
        tradability_config = TradabilityConfig()
    _validate_tradability_columns(panel, tradability_config)
    signals = _select_signals(spec, portfolio_signals)
    group_columns = [*segment_columns, spec.timestamp_column]

    position_records: list[dict[str, object]] = []
    diagnostic_records: list[dict[str, object]] = []
    previous_positions: dict[tuple[object, ...], dict[str, dict[str, float]]] = {}
    for group_key, timestamp_frame in panel.groupby(group_columns, sort=True):
        group_values = _normalize_group_key(group_key, group_columns)
        timestamp = group_values[spec.timestamp_column]
        segment_key = tuple(group_values[column] for column in segment_columns)
        signal_previous = previous_positions.setdefault(segment_key, {})
        tradability_context = _build_tradability_context(
            timestamp_frame=timestamp_frame,
            spec=spec,
            tradability_config=tradability_config,
        )
        for signal in signals:
            records, diagnostic = _build_signal_positions(
                timestamp_frame=timestamp_frame,
                spec=spec,
                signal=signal,
                weight_config=weight_config,
                tradability_config=tradability_config,
                tradability_context=tradability_context,
                group_values=group_values,
                previous_positions=signal_previous.get(signal, {}),
            )
            signal_previous[signal] = {
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
    signal_column: str | None = None,
) -> pd.DataFrame:
    """Build equal-weight long-short positions from timestamp-level ranks."""

    signal = _select_signal(spec, signal_column)
    config = WeightConfig(
        method="equal_weight_quantile",
        top_quantile=top_quantile,
        bottom_quantile=top_quantile if bottom_quantile is None else bottom_quantile,
    )
    positions, _diagnostics = build_weighted_positions(
        frame,
        spec,
        config,
        portfolio_signals=(signal,),
    )
    return positions[POSITION_COLUMNS]

__all__ = [
    "build_long_short_positions",
    "build_weighted_positions",
]
