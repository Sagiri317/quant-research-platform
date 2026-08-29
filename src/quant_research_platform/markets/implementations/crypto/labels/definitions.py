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
HFT_LABEL_HORIZONS_MS = (100, 500, 1_000, 5_000)
HFT_PRICE_LABEL_COLUMNS = tuple(
    column
    for horizon in HFT_LABEL_HORIZONS_MS
    for column in (
        f"future_mid_return_{horizon}ms",
        f"future_mid_direction_{horizon}ms",
    )
)
HFT_MAKER_FILL_PROXY_HORIZON_MS = 500
HFT_MAKER_BID_FILL_RATIO_PROXY_COLUMN = (
    f"maker_bid_fill_ratio_proxy_{HFT_MAKER_FILL_PROXY_HORIZON_MS}ms"
)
HFT_LABEL_COLUMNS = (
    *HFT_PRICE_LABEL_COLUMNS,
    HFT_MAKER_BID_FILL_RATIO_PROXY_COLUMN,
)


def label_catalog() -> LabelCatalog:
    return LabelCatalog(
        LabelDefinition(
            label_id=column,
            market_id=MARKET_ID,
            family="forward_return",
            input_columns=("close",),
            output_column=column,
            label_timing=("next_contiguous_bar_close_to_horizon_contiguous_bar_close"),
            horizon=horizon,
            lookahead=horizon + 1,
            entry_rule="next_contiguous_1m_bar_close",
            exit_rule="horizon_contiguous_1m_bar_close",
        )
        for horizon, column in zip(
            CRYPTO_FORWARD_RETURN_HORIZONS,
            CRYPTO_FORWARD_RETURN_COLUMNS,
            strict=True,
        )
    )
