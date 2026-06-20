"""A-share-specific label definitions."""

from __future__ import annotations

from quant_research_platform.catalog.labels import LabelCatalog
from quant_research_platform.contracts.data import DataFrequency, horizon_unit_suffix
from quant_research_platform.contracts.labels import LabelDefinition
from quant_research_platform.markets.implementations.a_share.ids import (
    MARKET_ID,
)

LABEL_BASE_COLUMNS: tuple[str, ...] = (
    "market_id",
    "asset_id",
    "timestamp",
)
LABEL_PANEL_INPUT_COLUMNS: tuple[str, ...] = (*LABEL_BASE_COLUMNS, "adj_open")
FORWARD_RETURN_HORIZONS: tuple[int, ...] = (1, 5, 10, 20, 60, 120)


def a_share_label_definitions() -> tuple[LabelDefinition, ...]:
    suffix = horizon_unit_suffix(DataFrequency.DAILY)
    return tuple(
        LabelDefinition(
            label_id=f"forward_return_{horizon}{suffix}",
            market_id=MARKET_ID,
            family="forward_return",
            input_columns=LABEL_PANEL_INPUT_COLUMNS,
            output_column=f"forward_return_{horizon}{suffix}",
            label_timing="next_valid_bar_open_to_horizon_valid_bar_open",
            horizon=horizon,
            lookahead=horizon + 1,
            entry_rule="next_asset_valid_bar_open",
            exit_rule="horizon_asset_valid_bar_open",
        )
        for horizon in FORWARD_RETURN_HORIZONS
    )


def a_share_label_catalog() -> LabelCatalog:
    catalog = LabelCatalog()
    for definition in a_share_label_definitions():
        catalog.register(definition)
    return catalog


__all__ = [
    "LABEL_BASE_COLUMNS",
    "LABEL_PANEL_INPUT_COLUMNS",
    "FORWARD_RETURN_HORIZONS",
    "a_share_label_catalog",
    "a_share_label_definitions",
]
