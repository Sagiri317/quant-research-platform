from __future__ import annotations

import pandas as pd

from quant_research_platform.catalog.fields import platform_interim_column_names
from quant_research_platform.markets.implementations.a_share.adapter import (
    AShareMarketAdapter,
)


def test_a_share_adapter_generates_canonical_daily_stock_fields() -> None:
    frame = pd.DataFrame({"stock_code": ["7"], "trade_date": ["2014-01-02"]})
    adapter = AShareMarketAdapter()

    result = adapter.with_daily_bar_canonical_fields(
        frame,
        source_id="csmar",
        source_asset_column="stock_code",
        session_date_column="trade_date",
    )

    assert list(result.columns[:7]) == list(platform_interim_column_names())
    assert result.loc[0, "market_id"] == "a_share"
    assert result.loc[0, "asset_id"] == "000007"
    assert str(result.loc[0, "timestamp"].date()) == "2014-01-02"
    assert str(result.loc[0, "session_date"].date()) == "2014-01-02"
    assert str(result.loc[0, "available_at"].date()) == "2014-01-02"
    assert result.loc[0, "source_id"] == "csmar"
    assert result.loc[0, "source_asset_id"] == "000007"


def test_a_share_adapter_generates_index_asset_namespace() -> None:
    frame = pd.DataFrame({"index_code": ["000985"], "trade_date": ["2014-01-02"]})
    adapter = AShareMarketAdapter()

    result = adapter.with_daily_bar_canonical_fields(
        frame,
        source_id="csmar",
        source_asset_column="index_code",
        session_date_column="trade_date",
        asset_namespace="index",
    )

    assert result.loc[0, "asset_id"] == "index:000985"
