"""Crypto forward-return label definitions."""

from __future__ import annotations

from quant_research_platform.catalog.labels import LabelCatalog
from quant_research_platform.contracts.data import DataFrequency
from quant_research_platform.contracts.labels import LabelDefinition
from quant_research_platform.markets.implementations.crypto.ids import MARKET_ID
from quant_research_platform.research.labels.forward_returns import (
    forward_return_label_column,
)

CRYPTO_FORWARD_RETURN_HORIZONS = (1, 5, 15)
CRYPTO_FORWARD_RETURN_COLUMNS = tuple(
    forward_return_label_column(horizon, DataFrequency.MINUTE)
    for horizon in CRYPTO_FORWARD_RETURN_HORIZONS
)


def label_catalog() -> LabelCatalog:
    catalog = LabelCatalog()
    for horizon, column in zip(
        CRYPTO_FORWARD_RETURN_HORIZONS,
        CRYPTO_FORWARD_RETURN_COLUMNS,
        strict=True,
    ):
        catalog.register(
            LabelDefinition(
                label_id=column,
                market_id=MARKET_ID,
                family="forward_return",
                input_columns=("close",),
                output_column=column,
                label_timing=(
                    "next_contiguous_bar_close_to_horizon_contiguous_bar_close"
                ),
                horizon=horizon,
                lookahead=horizon + 1,
                entry_rule="next_contiguous_1m_bar_close",
                exit_rule="horizon_contiguous_1m_bar_close",
            )
        )
    return catalog


__all__ = [
    "CRYPTO_FORWARD_RETURN_COLUMNS",
    "CRYPTO_FORWARD_RETURN_HORIZONS",
    "label_catalog",
]
