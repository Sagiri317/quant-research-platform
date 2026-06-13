"""DataFrame portfolio exposure summary algorithms."""

from __future__ import annotations

import pandas as pd

from quant_research_platform.compute.dataframe.portfolio_constants import (
    EXPOSURE_SUMMARY_COLUMNS,
)
from quant_research_platform.compute.dataframe.portfolio_helpers import (
    _normalize_group_key,
)
from quant_research_platform.compute.dataframe.position_math import _validate_positions


def summarize_exposure(
    positions: pd.DataFrame,
    returns: pd.DataFrame,
    *,
    segment_columns: tuple[str, ...] = (),
) -> pd.DataFrame:
    """汇总 gross/net exposure 和 active position counts。"""

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

__all__ = [
    "summarize_exposure",
]
