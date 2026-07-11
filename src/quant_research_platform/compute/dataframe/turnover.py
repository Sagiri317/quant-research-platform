"""DataFrame portfolio turnover and cost algorithms."""

from __future__ import annotations

import pandas as pd

from quant_research_platform.compute.dataframe.portfolio_constants import (
    COST_BREAKDOWN_COLUMNS,
    TURNOVER_COLUMNS,
)
from quant_research_platform.compute.dataframe.portfolio_helpers import (
    cost_components,
    normalize_group_key,
)
from quant_research_platform.compute.dataframe.position_math import validate_positions
from quant_research_platform.contracts.portfolios import (
    CostConfig,
    WeightConfig,
)


def calculate_turnover(
    positions: pd.DataFrame,
    *,
    segment_columns: tuple[str, ...] = (),
) -> pd.DataFrame:
    """Calculate one-way turnover from timestamp-by-asset position changes."""

    validated = validate_positions(positions, segment_columns=segment_columns)
    output_columns = [*segment_columns, *TURNOVER_COLUMNS]
    if validated.empty:
        return pd.DataFrame(columns=output_columns)

    records: list[dict[str, object]] = []
    for group_key, group in validated.groupby([*segment_columns, "signal"], sort=True):
        group_values = normalize_group_key(group_key, [*segment_columns, "signal"])
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
                    "signal": group_values["signal"],
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
    validated = validate_positions(positions, segment_columns=segment_columns)
    output_columns = [*segment_columns, *COST_BREAKDOWN_COLUMNS]
    if validated.empty:
        return pd.DataFrame(columns=output_columns)

    records: list[dict[str, object]] = []
    for group_key, group in validated.groupby([*segment_columns, "signal"], sort=True):
        group_values = normalize_group_key(group_key, [*segment_columns, "signal"])
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
            costs = cost_components(
                buy_turnover=buy_turnover,
                sell_turnover=sell_turnover,
                turnover=turnover,
                cost_config=cost_config,
            )
            records.append(
                {
                    **{column: group_values[column] for column in segment_columns},
                    "timestamp": timestamp,
                    "signal": group_values["signal"],
                    "buy_turnover": buy_turnover,
                    "sell_turnover": sell_turnover,
                    "turnover": turnover,
                    **costs,
                }
            )
    return pd.DataFrame.from_records(records, columns=output_columns)

def active_position_map(records: list[dict[str, object]]) -> dict[str, float]:
    return {
        str(record["asset"]): float(record["position"])
        for record in records
        if float(record["position"]) != 0.0
    }

def position_change_turnover(
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

__all__ = [
    "calculate_cost_breakdown",
    "calculate_turnover",
]
