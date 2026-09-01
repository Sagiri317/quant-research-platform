"""Common label dispatch for market-session research datasets."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

from quant_research_platform.contracts.labels import LabelDefinition
from quant_research_platform.contracts.time import TimeWindow
from quant_research_platform.research.labels.forward_returns import (
    add_forward_return_labels,
    forward_return_label_column,
    session_read_window,
)


def calculate_labels_for_panel(
    panel_frame: pd.DataFrame,
    *,
    definitions: Sequence[LabelDefinition],
    sessions: Sequence[object] | None = None,
) -> pd.DataFrame:
    requested = tuple(definitions)
    _require_label_input_columns(panel_frame, requested)
    if not requested:
        return panel_frame.loc[:, ["market_id", "asset_id", "timestamp"]].copy()
    families = tuple(dict.fromkeys(definition.family for definition in requested))
    if families != ("forward_return",):
        raise ValueError("unknown_label_family")
    axis = (
        sessions
        if sessions is not None
        else tuple(sorted(pd.to_datetime(panel_frame["timestamp"]).unique()))
    )
    result = add_forward_return_labels(
        panel_frame.loc[:, list(label_panel_input_columns(requested))],
        horizons=tuple(definition.horizon for definition in requested),
        asset_column="asset_id",
        date_column="timestamp",
        execution_price_column="adj_open",
        sessions=axis,
        can_enter_column="can_enter" if "can_enter" in panel_frame else None,
    )
    result["model_target"] = _model_target(result, forward_return_label_column(20))
    return result


def label_panel_input_columns(
    definitions: Sequence[LabelDefinition],
) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(
            (
                "market_id",
                "asset_id",
                "timestamp",
                *(
                    column
                    for definition in definitions
                    for column in definition.input_columns
                ),
            )
        )
    )


def label_lookahead_bars(definitions: Sequence[LabelDefinition]) -> int:
    return max((definition.lookahead for definition in definitions), default=0)


def label_read_window_for_panel(
    panel_keys: pd.DataFrame | Mapping[object, Sequence[pd.Timestamp]],
    *,
    definitions: Sequence[LabelDefinition],
    read_window: TimeWindow,
    core_sessions: tuple[pd.Timestamp, ...],
) -> TimeWindow:
    del panel_keys
    return session_read_window(
        read_window=read_window,
        sessions=core_sessions,
        lookahead_sessions=label_lookahead_bars(definitions),
    )


def _model_target(frame: pd.DataFrame, label_column: str) -> pd.Series:
    values = pd.to_numeric(frame[label_column], errors="raise").astype("float64")
    grouped = values.groupby(frame["timestamp"], sort=False)
    mean = grouped.transform("mean")
    centered = values - mean
    scale = centered.groupby(frame["timestamp"], sort=False).transform(
        lambda item: float(np.sqrt(np.mean(item.to_numpy() ** 2)))
    )
    target = centered.div(scale.where(scale.ne(0.0)))
    target = target.where(values.notna())
    return target.replace([np.inf, -np.inf], np.nan)


def _require_label_input_columns(
    frame: pd.DataFrame,
    definitions: Sequence[LabelDefinition],
) -> None:
    missing = sorted(
        {
            column
            for definition in definitions
            for column in definition.input_columns
            if column not in frame.columns
        }
    )
    if missing:
        raise ValueError("missing_label_input_columns: " + ", ".join(missing))
