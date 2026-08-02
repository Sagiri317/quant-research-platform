"""A-share-specific label calculations."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import pandas as pd

from quant_research_platform.contracts.labels import LabelDefinition
from quant_research_platform.contracts.time import TimeWindow
from quant_research_platform.research.labels.forward_returns import (
    add_forward_return_labels,
    forward_return_valid_bar_read_window,
)


def calculate_labels_for_panel(
    panel_frame: pd.DataFrame,
    *,
    definitions: Sequence[LabelDefinition],
) -> pd.DataFrame:
    requested = tuple(definitions)
    _require_label_input_columns(panel_frame, requested)
    families = tuple(dict.fromkeys(definition.family for definition in requested))
    if not requested:
        return panel_frame.loc[:, ["market_id", "asset_id", "timestamp"]].copy()
    if families != ("forward_return",):
        unsupported = next(family for family in families if family != "forward_return")
        raise ValueError(f"unsupported_label_family: {unsupported}")
    return add_forward_return_labels(
        panel_frame.loc[:, list(label_panel_input_columns(requested))],
        horizons=tuple(definition.horizon for definition in requested),
        asset_column="asset_id",
        date_column="timestamp",
        execution_price_column="adj_open",
    )


def label_panel_input_columns(
    definitions: Sequence[LabelDefinition],
) -> tuple[str, ...]:
    requested = tuple(definitions)
    families = tuple(dict.fromkeys(definition.family for definition in requested))
    if not requested:
        return ("market_id", "asset_id", "timestamp")
    if families != ("forward_return",):
        unsupported = next(family for family in families if family != "forward_return")
        raise ValueError(f"unsupported_label_family: {unsupported}")
    return tuple(
        dict.fromkeys(
            (
                "market_id",
                "asset_id",
                "timestamp",
                *(
                    column
                    for definition in requested
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
    requested = tuple(definitions)
    families = tuple(dict.fromkeys(definition.family for definition in requested))
    if not requested:
        return read_window
    if families != ("forward_return",):
        unsupported = next(family for family in families if family != "forward_return")
        raise ValueError(f"unsupported_label_family: {unsupported}")
    return forward_return_valid_bar_read_window(
        panel_keys,
        read_window=read_window,
        core_sessions=core_sessions,
        lookahead_bars=label_lookahead_bars(requested),
    )


def _require_label_input_columns(
    panel_frame: pd.DataFrame,
    definitions: Sequence[LabelDefinition],
) -> None:
    for definition in definitions:
        missing = [
            column
            for column in definition.input_columns
            if column not in panel_frame.columns
        ]
        if missing:
            raise ValueError(
                "missing_label_input_columns: "
                f"{definition.family}: {', '.join(missing)}"
            )


__all__ = [
    "calculate_labels_for_panel",
    "label_panel_input_columns",
    "label_lookahead_bars",
    "label_read_window_for_panel",
]
