"""Common label dispatch for market-session research datasets."""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

from quant_research_platform.contracts.labels import LabelDefinition
from quant_research_platform.research.labels.forward_returns import (
    add_forward_return_labels,
)


def calculate_labels_for_panel(
    panel_frame: pd.DataFrame,
    *,
    definitions: Sequence[LabelDefinition],
    sessions: Sequence[object],
) -> pd.DataFrame:
    requested = tuple(definitions)
    _require_label_input_columns(panel_frame, requested)
    if not requested:
        return panel_frame.loc[:, ["market_id", "asset_id", "timestamp"]].copy()
    families = tuple(dict.fromkeys(definition.family for definition in requested))
    if families != ("forward_return",):
        raise ValueError("unknown_label_family")
    result = add_forward_return_labels(
        panel_frame.loc[:, list(label_panel_input_columns(requested))],
        horizons=tuple(definition.horizon for definition in requested),
        asset_column="asset_id",
        date_column="timestamp",
        execution_price_column="adj_open",
        sessions=sessions,
        can_enter_column="can_enter" if "can_enter" in panel_frame else None,
    )
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
