"""Tensor portfolio turnover and cost algorithms."""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant_research_platform.compute._portfolio_math import (
    cost_components,
    normalize_group_key,
)
from quant_research_platform.compute.dataframe.portfolio_constants import (
    COST_BREAKDOWN_COLUMNS,
    TURNOVER_COLUMNS,
)
from quant_research_platform.contracts.portfolios import (
    CostConfig,
    WeightConfig,
)


def calculate_tensor_turnover_and_cost(
    positions: pd.DataFrame,
    weight_config: WeightConfig,
    cost_config: CostConfig | None = None,
    *,
    segment_columns: tuple[str, ...] = (),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calculate turnover and cost for tensor-shaped position matrices."""

    if cost_config is None:
        cost_config = CostConfig()
    if positions.empty:
        return (
            pd.DataFrame(columns=[*segment_columns, *TURNOVER_COLUMNS]),
            pd.DataFrame(columns=[*segment_columns, *COST_BREAKDOWN_COLUMNS]),
        )
    turnover_records: list[dict[str, object]] = []
    cost_records: list[dict[str, object]] = []
    group_columns = [*segment_columns, "signal"]
    for group_key, group in positions.groupby(group_columns, sort=True, dropna=False):
        group_values = normalize_group_key(group_key, group_columns)
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
                "signal": group_values["signal"],
            }
            costs = cost_components(
                buy_turnover=float(buy_turnover[row_offset]),
                sell_turnover=float(sell_turnover[row_offset]),
                turnover=float(turnover[row_offset]),
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

def path_turnover_and_cost_records_from_arrays(
    *,
    context: dict[str, object],
    timestamp: object,
    signal: str,
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
        "signal": signal,
    }
    costs = cost_components(
        buy_turnover=buy_turnover,
        sell_turnover=sell_turnover,
        turnover=turnover,
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

def path_current_state(
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

def position_map_from_arrays(
    *,
    assets: tuple[str, ...],
    positions: np.ndarray,
) -> dict[str, float]:
    nonzero_offsets = np.flatnonzero(positions != 0.0)
    return {
        assets[int(offset)]: float(positions[int(offset)]) for offset in nonzero_offsets
    }

__all__ = [
    "calculate_tensor_turnover_and_cost",
]
