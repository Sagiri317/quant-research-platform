from __future__ import annotations

import pandas as pd
import pytest

from quant_research_platform.catalog.fields import platform_interim_column_names
from quant_research_platform.markets.implementations.a_share.adapter import (
    MarketAdapter,
)


def test_a_share_adapter_generates_canonical_market_bar_fields() -> None:
    frame = pd.DataFrame({"stock_code": ["7"], "trade_date": ["2014-01-02"]})
    adapter = MarketAdapter()

    result = adapter.with_market_bar_canonical_fields(
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
    adapter = MarketAdapter()

    result = adapter.with_market_bar_canonical_fields(
        frame,
        source_id="csmar",
        source_asset_column="index_code",
        session_date_column="trade_date",
        asset_namespace="index",
    )

    assert result.loc[0, "asset_id"] == "index:000985"


@pytest.mark.parametrize(
    "value",
    [
        None,
        pd.NA,
        float("nan"),
        "",
        "nan",
        "None",
        True,
        1,
        1.0,
        -1,
        "-1",
        1.5,
        "1.5",
        "000001.0",
        "1000000",
        [],
        {},
        set(),
    ],
)
def test_a_share_adapter_rejects_invalid_stock_source_asset_id(value: object) -> None:
    adapter = MarketAdapter()

    with pytest.raises(ValueError, match="invalid_stock_source_asset_id"):
        adapter.with_market_bar_canonical_fields(
            pd.DataFrame({"stock_code": [value], "trade_date": ["2014-01-02"]}),
            source_id="csmar",
            source_asset_column="stock_code",
            session_date_column="trade_date",
        )


@pytest.mark.parametrize("value", [None, pd.NA, "", 123, [], {}, set()])
def test_a_share_adapter_rejects_invalid_non_stock_source_asset_id(
    value: object,
) -> None:
    adapter = MarketAdapter()

    with pytest.raises(ValueError, match="invalid_source_asset_id"):
        adapter.with_market_bar_canonical_fields(
            pd.DataFrame({"index_code": [value], "trade_date": ["2014-01-02"]}),
            source_id="csmar",
            source_asset_column="index_code",
            session_date_column="trade_date",
            asset_namespace="index",
        )
