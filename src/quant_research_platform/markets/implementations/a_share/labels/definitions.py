"""A-share-specific label definitions."""

from __future__ import annotations

from quant_research_platform.contracts.labels import LabelDefinition
from quant_research_platform.markets.implementations.a_share.ids import (
    A_SHARE_MARKET_ID,
)

LABEL_BASE_COLUMNS: tuple[str, ...] = (
    "market_id",
    "asset_id",
    "timestamp",
)
LABEL_PANEL_INPUT_COLUMNS: tuple[str, ...] = (*LABEL_BASE_COLUMNS, "adj_open")


def a_share_forward_return_label_definitions() -> tuple[LabelDefinition, ...]:
    return tuple(
        LabelDefinition(
            label_id=f"forward_return_{horizon}d",
            market_id=A_SHARE_MARKET_ID,
            output_column=f"forward_return_{horizon}d",
            label_timing="next_open_to_horizon_open",
            horizon=horizon,
            lookahead=horizon + 1,
            entry_rule="next_session_open",
            exit_rule="horizon_session_open",
        )
        for horizon in (1, 5, 10, 20, 60, 120)
    )


def a_share_forward_return_horizons() -> tuple[int, ...]:
    return tuple(
        definition.horizon for definition in a_share_forward_return_label_definitions()
    )


__all__ = [
    "LABEL_BASE_COLUMNS",
    "LABEL_PANEL_INPUT_COLUMNS",
    "a_share_forward_return_horizons",
    "a_share_forward_return_label_definitions",
]
