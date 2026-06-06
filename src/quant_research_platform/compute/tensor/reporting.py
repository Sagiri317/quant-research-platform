"""Tensor portfolio long-short reporting algorithms."""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant_research_platform.compute.config import (
    CostConfig,
    WeightConfig,
)
from quant_research_platform.compute.dataframe.portfolio_constants import (
    RETURNS_COLUMNS,
)
from quant_research_platform.compute.dataframe.portfolio_helpers import (
    _normalize_group_key,
)
from quant_research_platform.compute.panels import ComputePanelSpec
from quant_research_platform.compute.tensor.exposure import _exposure_record_from_arrays
from quant_research_platform.compute.tensor.portfolio_helpers import (
    _filter_panel_group,
)
from quant_research_platform.compute.tensor.turnover import (
    _path_current_state,
    _path_turnover_and_cost_records_from_arrays,
)
from quant_research_platform.runtime.profiling import measure_stage


def calculate_tensor_long_short(
    *,
    panel: pd.DataFrame,
    spec: ComputePanelSpec,
    positions: pd.DataFrame,
    turnover: pd.DataFrame,
    cost_breakdown: pd.DataFrame,
    weight_config: WeightConfig | None = None,
    segment_columns: tuple[str, ...] = (),
) -> pd.DataFrame:
    """基于 tensor-shaped positions 计算 portfolio return path。"""

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

__all__ = [
    "calculate_tensor_long_short",
    "_long_short_record_from_arrays",
    "_build_path_records_from_arrays",
]
