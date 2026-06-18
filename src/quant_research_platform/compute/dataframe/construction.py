"""DataFrame portfolio construction algorithms."""

from __future__ import annotations

import pandas as pd

from quant_research_platform.compute.config import (
    TradabilityConfig,
    WeightConfig,
)
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
    _select_feature,
    _select_features,
    _validate_segment_columns,
    _validate_weight_columns,
)
from quant_research_platform.compute.dataframe.position_math import (
    _build_feature_positions,
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

    panel = validate_compute_panel(frame, spec, segment_columns=segment_columns)
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

__all__ = [
    "build_long_short_positions",
    "build_weighted_positions",
]
