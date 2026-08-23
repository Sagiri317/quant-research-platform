"""A-share-specific label definitions."""

from __future__ import annotations

from quant_research_platform.catalog.labels import LabelCatalog
from quant_research_platform.contracts.data import DataFrequency
from quant_research_platform.contracts.labels import LabelDefinition
from quant_research_platform.markets.implementations.a_share.ids import (
    MARKET_ID,
)
from quant_research_platform.research.labels.forward_returns import (
    forward_return_label_column,
)

LABEL_BASE_COLUMNS: tuple[str, ...] = (
    "market_id",
    "asset_id",
    "timestamp",
)
LABEL_PANEL_INPUT_COLUMNS: tuple[str, ...] = (*LABEL_BASE_COLUMNS, "adj_open")
FORWARD_RETURN_HORIZONS: tuple[int, ...] = (1, 5, 10, 20, 60, 120)


def label_definitions() -> tuple[LabelDefinition, ...]:
    return tuple(
        LabelDefinition(
            label_id=forward_return_label_column(horizon, DataFrequency.DAILY),
            market_id=MARKET_ID,
            family="forward_return",
            input_columns=LABEL_PANEL_INPUT_COLUMNS,
            output_column=forward_return_label_column(horizon, DataFrequency.DAILY),
            label_timing="next_valid_bar_open_to_horizon_valid_bar_open",
            horizon=horizon,
            lookahead=horizon + 1,
            entry_rule="next_asset_valid_bar_open",
            exit_rule="horizon_asset_valid_bar_open",
        )
        for horizon in FORWARD_RETURN_HORIZONS
    )


def label_catalog() -> LabelCatalog:
    return LabelCatalog(label_definitions())


__all__ = [
    "LABEL_BASE_COLUMNS",
    "LABEL_PANEL_INPUT_COLUMNS",
    "FORWARD_RETURN_HORIZONS",
    "label_catalog",
    "label_definitions",
]
