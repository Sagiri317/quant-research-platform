"""Signal weighting, positions, turnover, and portfolio return evaluation."""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant_research_platform.core.contracts import (
    CostConfig,
    TradabilityConfig,
    WeightConfig,
)
from quant_research_platform.core.panel import PanelSpec, validate_panel
from quant_research_platform.core.profiling import measure_stage
from quant_research_platform.engines.dataframe.performance import (
    PERFORMANCE_SUMMARY_COLUMNS,
    PNL_COLUMNS,
    calculate_cumulative_pnl,
    summarize_performance,
)
from quant_research_platform.engines.dataframe.portfolio_helpers import (
    _apply_tradability_constraints,
    _build_tradability_context,
    _cost_components,
    _direct_signal_array,
    _normalize_group_key,
    _risk_values,
    _select_feature,
    _select_features,
    _selection_count,
    _validate_segment_columns,
    _validate_tradability_columns,
    _validate_weight_columns,
)

__all__ = [
    "EXPOSURE_SUMMARY_COLUMNS",
    "PERFORMANCE_SUMMARY_COLUMNS",
    "PNL_COLUMNS",
    "POSITION_COLUMNS",
    "RETURNS_COLUMNS",
    "TURNOVER_COLUMNS",
    "WEIGHT_DIAGNOSTIC_COLUMNS",
    "build_long_short_positions",
    "build_weighted_positions",
    "calculate_cost_breakdown",
    "calculate_cumulative_pnl",
    "calculate_turnover",
    "simulate_long_short_returns",
    "simulate_weighted_long_short_returns",
    "summarize_exposure",
    "summarize_performance",
    "_simulate_weighted_long_short_returns_and_cost_from_validated_panel",
    "_simulate_weighted_long_short_returns_from_validated_panel",
]

POSITION_COLUMNS = ["timestamp", "asset", "feature", "position"]
TURNOVER_COLUMNS = ["timestamp", "feature", "turnover"]
RETURNS_COLUMNS = [
    "timestamp",
    "feature",
    "label",
    "portfolio_mode",
    "long_return",
    "short_return",
    "gross_portfolio_return",
    "turnover",
    "cost",
    "net_portfolio_return",
    "long_count",
    "short_count",
]
COST_BREAKDOWN_COLUMNS = [
    "timestamp",
    "feature",
    "buy_turnover",
    "sell_turnover",
    "turnover",
    "buy_cost",
    "sell_cost",
    "slippage_cost",
    "spread_cost",
    "impact_cost",
    "total_cost",
    "cost_model",
]
EXPOSURE_SUMMARY_COLUMNS = [
    "feature",
    "mean_gross_exposure",
    "mean_net_exposure",
    "max_gross_exposure",
    "max_abs_net_exposure",
    "mean_long_count",
    "mean_short_count",
    "mean_position_count",
]
WEIGHT_DIAGNOSTIC_COLUMNS = [
    "timestamp",
    "feature",
    "method",
    "direction",
    "long_count",
    "short_count",
    "fallback_reason",
]


def build_weighted_positions(
    frame: pd.DataFrame,
    spec: PanelSpec,
    weight_config: WeightConfig,
    *,
    segment_columns: tuple[str, ...] = (),
    portfolio_features: tuple[str, ...] | None = None,
    tradability_config: TradabilityConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build timestamp-level positions for each requested signal feature."""

    panel = validate_panel(frame, spec)
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
    spec: PanelSpec,
    weight_config: WeightConfig,
    *,
    segment_columns: tuple[str, ...] = (),
    portfolio_features: tuple[str, ...] | None = None,
    tradability_config: TradabilityConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build positions from a caller-validated timestamp/asset-sorted panel."""

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
    spec: PanelSpec,
    top_quantile: float = 0.2,
    bottom_quantile: float | None = None,
    feature_column: str | None = None,
) -> pd.DataFrame:
    """Build equal-weight long-short positions from timestamp-level ranks."""

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


def calculate_turnover(
    positions: pd.DataFrame,
    *,
    segment_columns: tuple[str, ...] = (),
) -> pd.DataFrame:
    """Calculate one-way turnover from timestamp-by-asset position changes."""

    validated = _validate_positions(positions, segment_columns=segment_columns)
    output_columns = [*segment_columns, *TURNOVER_COLUMNS]
    if validated.empty:
        return pd.DataFrame(columns=output_columns)

    records: list[dict[str, object]] = []
    for group_key, group in validated.groupby([*segment_columns, "feature"], sort=True):
        group_values = _normalize_group_key(group_key, [*segment_columns, "feature"])
        position_matrix = (
            group.pivot(index="timestamp", columns="asset", values="position")
            .fillna(0.0)
            .sort_index()
        )
        position_changes = position_matrix.diff()
        position_changes.iloc[0] = position_matrix.iloc[0]
        turnover = position_changes.abs().sum(axis=1)
        for timestamp, value in turnover.items():
            records.append(
                {
                    **{column: group_values[column] for column in segment_columns},
                    "timestamp": timestamp,
                    "feature": group_values["feature"],
                    "turnover": float(value),
                }
            )
    return pd.DataFrame.from_records(records, columns=output_columns)


def calculate_cost_breakdown(
    positions: pd.DataFrame,
    weight_config: WeightConfig,
    cost_config: CostConfig | None = None,
    *,
    segment_columns: tuple[str, ...] = (),
) -> pd.DataFrame:
    """Calculate buy/sell turnover and cost components from position changes."""

    if cost_config is None:
        cost_config = CostConfig()
    validated = _validate_positions(positions, segment_columns=segment_columns)
    output_columns = [*segment_columns, *COST_BREAKDOWN_COLUMNS]
    if validated.empty:
        return pd.DataFrame(columns=output_columns)

    records: list[dict[str, object]] = []
    for group_key, group in validated.groupby([*segment_columns, "feature"], sort=True):
        group_values = _normalize_group_key(group_key, [*segment_columns, "feature"])
        position_matrix = (
            group.pivot(index="timestamp", columns="asset", values="position")
            .fillna(0.0)
            .sort_index()
        )
        position_changes = position_matrix.diff()
        position_changes.iloc[0] = position_matrix.iloc[0]
        for timestamp, changes in position_changes.iterrows():
            buy_turnover = float(changes.clip(lower=0).sum())
            sell_turnover = float((-changes.clip(upper=0)).sum())
            turnover = buy_turnover + sell_turnover
            costs = _cost_components(
                buy_turnover=buy_turnover,
                sell_turnover=sell_turnover,
                turnover=turnover,
                weight_config=weight_config,
                cost_config=cost_config,
            )
            records.append(
                {
                    **{column: group_values[column] for column in segment_columns},
                    "timestamp": timestamp,
                    "feature": group_values["feature"],
                    "buy_turnover": buy_turnover,
                    "sell_turnover": sell_turnover,
                    "turnover": turnover,
                    **costs,
                }
            )
    return pd.DataFrame.from_records(records, columns=output_columns)


def simulate_weighted_long_short_returns(
    frame: pd.DataFrame,
    spec: PanelSpec,
    weight_config: WeightConfig,
    *,
    segment_columns: tuple[str, ...] = (),
    portfolio_features: tuple[str, ...] | None = None,
    tradability_config: TradabilityConfig | None = None,
    cost_config: CostConfig | None = None,
    pnl_return_column: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Evaluate portfolio returns from weighted positions."""

    panel = validate_panel(frame, spec)
    return _simulate_weighted_long_short_returns_from_validated_panel(
        panel,
        spec,
        weight_config,
        segment_columns=segment_columns,
        portfolio_features=portfolio_features,
        tradability_config=tradability_config,
        cost_config=cost_config,
        pnl_return_column=pnl_return_column,
    )


def _simulate_weighted_long_short_returns_from_validated_panel(
    panel: pd.DataFrame,
    spec: PanelSpec,
    weight_config: WeightConfig,
    *,
    segment_columns: tuple[str, ...] = (),
    portfolio_features: tuple[str, ...] | None = None,
    tradability_config: TradabilityConfig | None = None,
    cost_config: CostConfig | None = None,
    pnl_return_column: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Evaluate returns from a caller-validated timestamp/asset-sorted panel."""

    positions, turnover, returns, diagnostics, _cost_breakdown = (
        _simulate_weighted_long_short_returns_and_cost_from_validated_panel(
            panel,
            spec,
            weight_config,
            segment_columns=segment_columns,
            portfolio_features=portfolio_features,
            tradability_config=tradability_config,
            cost_config=cost_config,
            pnl_return_column=pnl_return_column,
        )
    )
    return positions, turnover, returns, diagnostics


def _simulate_weighted_long_short_returns_and_cost_from_validated_panel(
    panel: pd.DataFrame,
    spec: PanelSpec,
    weight_config: WeightConfig,
    *,
    segment_columns: tuple[str, ...] = (),
    portfolio_features: tuple[str, ...] | None = None,
    tradability_config: TradabilityConfig | None = None,
    cost_config: CostConfig | None = None,
    pnl_return_column: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Evaluate returns and expose cost breakdown for internal artifact reuse."""

    return measure_stage(
        "metric",
        "portfolio_stream",
        lambda: _stream_weighted_portfolio_from_validated_panel(
            panel,
            spec,
            weight_config,
            segment_columns=segment_columns,
            portfolio_features=portfolio_features,
            tradability_config=tradability_config,
            cost_config=cost_config,
            pnl_return_column=pnl_return_column,
        ),
    )


def _stream_weighted_portfolio_from_validated_panel(
    panel: pd.DataFrame,
    spec: PanelSpec,
    weight_config: WeightConfig,
    *,
    segment_columns: tuple[str, ...] = (),
    portfolio_features: tuple[str, ...] | None = None,
    tradability_config: TradabilityConfig | None = None,
    cost_config: CostConfig | None = None,
    pnl_return_column: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build positions, turnover, costs, and portfolio returns in one scan."""

    _validate_segment_columns(panel, segment_columns)
    _validate_weight_columns(panel, weight_config)
    if tradability_config is None:
        tradability_config = TradabilityConfig()
    if cost_config is None:
        cost_config = CostConfig()
    _validate_tradability_columns(panel, tradability_config)
    pnl_return_column = pnl_return_column or spec.label_column
    features = _select_features(spec, portfolio_features)
    group_columns = [*segment_columns, spec.timestamp_column]

    position_records: list[dict[str, object]] = []
    diagnostic_records: list[dict[str, object]] = []
    turnover_records: list[dict[str, object]] = []
    cost_records: list[dict[str, object]] = []
    return_records: list[dict[str, object]] = []
    previous_positions: dict[tuple[object, ...], dict[str, dict[str, float]]] = {}
    for group_key, timestamp_frame in panel.groupby(group_columns, sort=True):
        group_values = _normalize_group_key(group_key, group_columns)
        timestamp = group_values[spec.timestamp_column]
        segment_key = tuple(group_values[column] for column in segment_columns)
        feature_previous = previous_positions.setdefault(segment_key, {})
        label_lookup = _timestamp_label_lookup(
            timestamp_frame,
            spec,
            pnl_return_column,
        )
        tradability_context = _build_tradability_context(
            timestamp_frame=timestamp_frame,
            spec=spec,
            tradability_config=tradability_config,
        )
        for feature in features:
            previous = feature_previous.get(feature, {})
            records, diagnostic = _build_feature_positions(
                timestamp_frame=timestamp_frame,
                spec=spec,
                feature=feature,
                weight_config=weight_config,
                tradability_config=tradability_config,
                tradability_context=tradability_context,
                group_values=group_values,
                previous_positions=previous,
            )
            current = _active_position_map(records)
            buy_turnover, sell_turnover = _position_change_turnover(
                previous=previous,
                current=current,
            )
            turnover = buy_turnover + sell_turnover
            costs = _cost_components(
                buy_turnover=buy_turnover,
                sell_turnover=sell_turnover,
                turnover=turnover,
                weight_config=weight_config,
                cost_config=cost_config,
            )
            base_values = {
                **{column: group_values[column] for column in segment_columns},
                "timestamp": timestamp,
                "feature": feature,
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
                label_column=pnl_return_column,
                turnover=turnover,
                cost=float(costs["total_cost"]),
                segment_columns=segment_columns,
            )
            if return_record is not None:
                return_records.append(return_record)
            feature_previous[feature] = current

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
    spec: PanelSpec,
    top_quantile: float = 0.2,
    bottom_quantile: float = 0.2,
    cost_bps: float = 0.0,
    feature_column: str | None = None,
) -> pd.DataFrame:
    """Evaluate equal-weight long-short returns with turnover-based costs."""

    feature = _select_feature(spec, feature_column)
    _positions, _turnover, returns, _diagnostics = (
        simulate_weighted_long_short_returns(
            frame,
            spec,
            WeightConfig(
                method="equal_weight_quantile",
                top_quantile=top_quantile,
                bottom_quantile=bottom_quantile,
                cost_bps=cost_bps,
            ),
            portfolio_features=(feature,),
        )
    )
    return returns[RETURNS_COLUMNS]


def summarize_exposure(
    positions: pd.DataFrame,
    returns: pd.DataFrame,
    *,
    segment_columns: tuple[str, ...] = (),
) -> pd.DataFrame:
    """Summarize gross/net exposure and active position counts."""

    validated = _validate_positions(positions, segment_columns=segment_columns)
    output_columns = [*segment_columns, *EXPOSURE_SUMMARY_COLUMNS]
    if validated.empty:
        return pd.DataFrame(columns=output_columns)

    group_columns = [*segment_columns, "feature"]
    exposure = (
        validated.assign(abs_position=validated["position"].abs())
        .groupby([*group_columns, "timestamp"], sort=True)
        .agg(
            gross_exposure=("abs_position", "sum"),
            net_exposure=("position", "sum"),
            position_count=("position", lambda values: int((values != 0).sum())),
        )
        .reset_index()
    )
    count_frame = returns.copy()
    records: list[dict[str, object]] = []
    for group_key, group in exposure.groupby(group_columns, sort=True, dropna=False):
        group_values = _normalize_group_key(group_key, group_columns)
        filters = count_frame["feature"] == group_values["feature"]
        for segment in segment_columns:
            filters &= count_frame[segment] == group_values[segment]
        count_group = count_frame.loc[filters]
        records.append(
            {
                **{column: group_values[column] for column in segment_columns},
                "feature": group_values["feature"],
                "mean_gross_exposure": float(group["gross_exposure"].mean()),
                "mean_net_exposure": float(group["net_exposure"].mean()),
                "max_gross_exposure": float(group["gross_exposure"].max()),
                "max_abs_net_exposure": float(group["net_exposure"].abs().max()),
                "mean_long_count": float(
                    pd.to_numeric(count_group["long_count"], errors="coerce").mean()
                ),
                "mean_short_count": float(
                    pd.to_numeric(count_group["short_count"], errors="coerce").mean()
                ),
                "mean_position_count": float(group["position_count"].mean()),
            }
        )
    return pd.DataFrame.from_records(records, columns=output_columns)


def _calculate_long_short_from_positions(
    *,
    panel: pd.DataFrame,
    spec: PanelSpec,
    positions: pd.DataFrame,
    turnover: pd.DataFrame,
    cost_breakdown: pd.DataFrame,
    weight_config: WeightConfig,
    segment_columns: tuple[str, ...],
    pnl_return_column: str | None = None,
) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    pnl_return_column = pnl_return_column or spec.label_column
    lookup_columns = [*segment_columns, spec.timestamp_column, spec.asset_column]
    lookup_index = [*segment_columns, "timestamp", "asset"]
    label_lookup = (
        panel[[*lookup_columns, pnl_return_column]]
        .rename(
            columns={
                spec.timestamp_column: "timestamp",
                spec.asset_column: "asset",
            },
        )
        .set_index(lookup_index)
    )
    position_group_columns = [*segment_columns, "timestamp", "feature"]
    positioned_positions = positions.join(
        label_lookup,
        on=lookup_index,
        how="left",
    )
    metric_index = [*segment_columns, "timestamp", "feature"]
    turnover_lookup = turnover.set_index(metric_index)["turnover"].to_dict()
    cost_lookup = cost_breakdown.set_index(metric_index)["total_cost"].to_dict()
    for group_key, timestamp_positions in positioned_positions.groupby(
        position_group_columns,
        sort=True,
    ):
        group_values = _normalize_group_key(group_key, position_group_columns)
        positions_array = timestamp_positions["position"].to_numpy(
            dtype="float64",
            copy=False,
        )
        labels_array = timestamp_positions[pnl_return_column].to_numpy(
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
                "feature": group_values["feature"],
                "label": pnl_return_column,
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


def _active_position_map(records: list[dict[str, object]]) -> dict[str, float]:
    return {
        str(record["asset"]): float(record["position"])
        for record in records
        if float(record["position"]) != 0.0
    }


def _timestamp_label_lookup(
    timestamp_frame: pd.DataFrame,
    spec: PanelSpec,
    label_column: str,
) -> dict[str, float]:
    assets = timestamp_frame[spec.asset_column].astype(str).to_numpy(copy=False)
    labels = timestamp_frame[label_column].to_numpy(dtype="float64", copy=False)
    return dict(zip(assets, labels, strict=True))


def _position_change_turnover(
    *,
    previous: dict[str, float],
    current: dict[str, float],
) -> tuple[float, float]:
    buy_turnover = 0.0
    sell_turnover = 0.0
    for asset in previous.keys() | current.keys():
        change = current.get(asset, 0.0) - previous.get(asset, 0.0)
        if change > 0:
            buy_turnover += change
        elif change < 0:
            sell_turnover -= change
    return buy_turnover, sell_turnover


def _build_portfolio_return_record_from_positions(
    *,
    group_values: dict[str, object],
    portfolio_mode: str,
    positions: dict[str, float],
    label_lookup: dict[str, float],
    label_column: str,
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
        "feature": group_values["feature"],
        "label": label_column,
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


def _build_feature_positions(
    *,
    timestamp_frame: pd.DataFrame,
    spec: PanelSpec,
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
        valid_mask &= timestamp_frame[
            tradability_config.eligibility_column
        ].to_numpy(dtype=bool, copy=False)
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


def _lookup_turnover(
    turnover: pd.DataFrame,
    *,
    group_values: dict[str, object],
    segment_columns: tuple[str, ...],
) -> float:
    filters = (
        (turnover["timestamp"] == group_values["timestamp"])
        & (turnover["feature"] == group_values["feature"])
    )
    for segment in segment_columns:
        filters &= turnover[segment] == group_values[segment]
    return float(turnover.loc[filters, "turnover"].iloc[0])


def _lookup_cost(
    cost_breakdown: pd.DataFrame,
    *,
    group_values: dict[str, object],
    segment_columns: tuple[str, ...],
) -> float:
    filters = (
        (cost_breakdown["timestamp"] == group_values["timestamp"])
        & (cost_breakdown["feature"] == group_values["feature"])
    )
    for segment in segment_columns:
        filters &= cost_breakdown[segment] == group_values[segment]
    return float(cost_breakdown.loc[filters, "total_cost"].iloc[0])


