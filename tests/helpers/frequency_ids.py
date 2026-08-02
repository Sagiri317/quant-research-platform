from __future__ import annotations

from pathlib import Path

from quant_research_platform.contracts.data import (
    DataFrequency,
    data_frequency_slug,
)
from quant_research_platform.markets.implementations.a_share.ids import (
    market_context_data_id,
    market_panel_data_id,
)
from quant_research_platform.sources.csmar import ids as csmar_ids

DAILY_FREQUENCY = DataFrequency.DAILY
DAILY_FREQUENCY_SLUG = data_frequency_slug(DAILY_FREQUENCY)
MARKET_BARS_DATA_ID = csmar_ids.market_bars_data_id(DAILY_FREQUENCY)
MARKET_PANEL_DATA_ID = market_panel_data_id(DAILY_FREQUENCY)
MARKET_CONTEXT_DATA_ID = market_context_data_id(DAILY_FREQUENCY)


def csmar_market_bars_interim_root(root: Path) -> Path:
    return root / "data" / "interim" / "csmar" / MARKET_BARS_DATA_ID


def market_panel_processed_root(root: Path) -> Path:
    return root / "data" / "processed" / "a_share" / "panel" / MARKET_PANEL_DATA_ID
