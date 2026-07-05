from __future__ import annotations

import pytest

from quant_research_platform.sources.binance.capture_plan import BinanceCapturePlan


def test_binance_capture_plan_accepts_parameterized_spot_window() -> None:
    plan = BinanceCapturePlan(
        capture_id="capture_20260706_000000",
        market_type="spot",
        data_kind="klines",
        symbols=("BTCUSDT", "ETHUSDT"),
        interval="1m",
        start_time="2026-07-06T00:00:00Z",
        end_time="2026-07-06T02:00:00Z",
        limit=120,
        depth_limit=20,
        chunk_size=50,
    )

    assert plan.time_zone == "UTC"
    assert plan.symbols == ("BTCUSDT", "ETHUSDT")
    assert plan.start_time == "2026-07-06T00:00:00Z"
    assert plan.end_time == "2026-07-06T02:00:00Z"


@pytest.mark.parametrize("market_type", ("futures", "margin", ""))
def test_binance_capture_plan_rejects_non_spot_market_type(market_type: str) -> None:
    with pytest.raises(ValueError, match="unsupported_binance_market_type"):
        BinanceCapturePlan(
            capture_id="capture_20260706_000000",
            market_type=market_type,
            data_kind="klines",
            symbols=("BTCUSDT",),
            interval="1m",
            start_time="2026-07-06T00:00:00Z",
            end_time="2026-07-06T02:00:00Z",
            limit=120,
            depth_limit=20,
            chunk_size=50,
        )


def test_binance_capture_plan_rejects_unsupported_data_kind() -> None:
    with pytest.raises(ValueError, match="unsupported_binance_data_kind"):
        BinanceCapturePlan(
            capture_id="capture_20260706_000000",
            market_type="spot",
            data_kind="funding",
            symbols=("BTCUSDT",),
            interval="1m",
            start_time="2026-07-06T00:00:00Z",
            end_time="2026-07-06T02:00:00Z",
            limit=120,
            depth_limit=20,
            chunk_size=50,
        )
