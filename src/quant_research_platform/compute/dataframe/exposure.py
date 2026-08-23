"""DataFrame portfolio exposure summary algorithms."""

from __future__ import annotations

import pandas as pd

from quant_research_platform.compute.dataframe.performance import (
    _validate_returns_frame,
)
from quant_research_platform.compute.dataframe.portfolio_constants import (
    EXPOSURE_SUMMARY_COLUMNS,
)
from quant_research_platform.compute.dataframe.position_math import (
    validate_positions,
)
from quant_research_platform.compute.grouping import normalize_group_key
from quant_research_platform.contracts.validation import strict_integer_series


def summarize_exposure(
    positions: pd.DataFrame,
    returns: pd.DataFrame,
    *,
    segment_columns: tuple[str, ...] = (),
) -> pd.DataFrame:
    """Summarize gross/net exposure and active position counts."""

    validated = validate_positions(positions, segment_columns=segment_columns)
    count_frame = _validate_returns_frame(
        returns,
        segment_columns=segment_columns,
        include_performance_columns=False,
        required_numeric_columns=(),
    )
    for column in ("long_count", "short_count"):
        if column not in count_frame:
            raise ValueError(f"returns is missing columns: ['{column}'].")
        if not pd.api.types.is_numeric_dtype(count_frame[column]):
            raise ValueError(
                f"returns column {column!r} must contain numeric values."
            )
        count_frame[column] = strict_integer_series(
            count_frame[column],
            error=f"returns column {column!r} must contain finite integer values.",
        )
        if bool((count_frame[column] < 0).any()):
            raise ValueError(f"returns column {column!r} must be nonnegative.")
    output_columns = [*segment_columns, *EXPOSURE_SUMMARY_COLUMNS]
    if validated.empty:
        return pd.DataFrame(columns=output_columns)

    group_columns = [*segment_columns, "signal"]
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
    records: list[dict[str, object]] = []
    for group_key, group in exposure.groupby(group_columns, sort=True, dropna=False):
        group_values = normalize_group_key(group_key, group_columns)
        filters = count_frame["signal"] == group_values["signal"]
        for segment in segment_columns:
            filters &= count_frame[segment] == group_values[segment]
        count_group = count_frame.loc[filters]
        records.append(
            {
                **{column: group_values[column] for column in segment_columns},
                "signal": group_values["signal"],
                "mean_gross_exposure": float(group["gross_exposure"].mean()),
                "mean_net_exposure": float(group["net_exposure"].mean()),
                "max_gross_exposure": float(group["gross_exposure"].max()),
                "max_abs_net_exposure": float(group["net_exposure"].abs().max()),
                "mean_long_count": float(
                    count_group["long_count"].mean()
                ),
                "mean_short_count": float(
                    count_group["short_count"].mean()
                ),
                "mean_position_count": float(group["position_count"].mean()),
            }
        )
    return pd.DataFrame.from_records(records, columns=output_columns)
