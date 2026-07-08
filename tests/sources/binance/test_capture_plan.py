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


def test_binance_capture_plan_rejects_invalid_symbols_and_duplicate_symbols() -> None:
    with pytest.raises(ValueError, match="invalid_binance_symbol"):
        BinanceCapturePlan(
            capture_id="capture_20260706_000000",
            market_type="spot",
            data_kind="klines",
            symbols=("btc_usdt",),
            interval="1m",
            start_time="2026-07-06T00:00:00Z",
            end_time="2026-07-06T02:00:00Z",
            limit=120,
            depth_limit=20,
            chunk_size=50,
        )
    with pytest.raises(ValueError, match="duplicate_binance_symbols"):
        BinanceCapturePlan(
            capture_id="capture_20260706_000000",
            market_type="spot",
            data_kind="klines",
            symbols=("BTCUSDT", "BTCUSDT"),
            interval="1m",
            start_time="2026-07-06T00:00:00Z",
            end_time="2026-07-06T02:00:00Z",
            limit=120,
            depth_limit=20,
            chunk_size=50,
        )


def test_binance_capture_plan_rejects_bare_str_symbols() -> None:
    with pytest.raises(ValueError, match="binance_symbols_required"):
        BinanceCapturePlan(
            capture_id="capture_20260706_000000",
            market_type="spot",
            data_kind="klines",
            symbols="BTCUSDT",  # type: ignore[arg-type]
            interval="1m",
            start_time="2026-07-06T00:00:00Z",
            end_time="2026-07-06T02:00:00Z",
            limit=120,
            depth_limit=20,
            chunk_size=50,
        )


@pytest.mark.parametrize("limit", [True, 120.5, "120"])
def test_binance_capture_plan_rejects_non_strict_limits(limit: object) -> None:
    with pytest.raises(ValueError, match="binance_capture_positive_limits_required"):
        BinanceCapturePlan(
            capture_id="capture_20260706_000000",
            market_type="spot",
            data_kind="klines",
            symbols=("BTCUSDT",),
            interval="1m",
            start_time="2026-07-06T00:00:00Z",
            end_time="2026-07-06T02:00:00Z",
            limit=limit,  # type: ignore[arg-type]
            depth_limit=20,
            chunk_size=50,
        )


def test_binance_capture_plan_requires_interval_only_for_klines() -> None:
    with pytest.raises(ValueError, match="unsupported_binance_interval"):
        BinanceCapturePlan(
            capture_id="capture_20260706_000000",
            market_type="spot",
            data_kind="klines",
            symbols=("BTCUSDT",),
            interval=None,  # type: ignore[arg-type]
            start_time="2026-07-06T00:00:00Z",
            end_time="2026-07-06T02:00:00Z",
            limit=120,
            depth_limit=20,
            chunk_size=50,
        )
    with pytest.raises(ValueError, match="unexpected_binance_interval"):
        BinanceCapturePlan(
            capture_id="capture_20260706_000000",
            market_type="spot",
            data_kind="depth",
            symbols=("BTCUSDT",),
            interval="1m",
            start_time="2026-07-06T00:00:00Z",
            end_time="2026-07-06T02:00:00Z",
            limit=120,
            depth_limit=20,
            chunk_size=50,
        )


def test_binance_capture_plan_rejects_invalid_utc_window_and_limits() -> None:
    with pytest.raises(ValueError, match="invalid_binance_capture_time_order"):
        BinanceCapturePlan(
            capture_id="capture_20260706_000000",
            market_type="spot",
            data_kind="klines",
            symbols=("BTCUSDT",),
            interval="1m",
            start_time="2026-07-06T02:00:00Z",
            end_time="2026-07-06T00:00:00Z",
            limit=120,
            depth_limit=20,
            chunk_size=50,
        )
    with pytest.raises(ValueError, match="invalid_binance_start_time"):
        BinanceCapturePlan(
            capture_id="capture_20260706_000000",
            market_type="spot",
            data_kind="klines",
            symbols=("BTCUSDT",),
            interval="1m",
            start_time="2026-07-06T00:00:00",
            end_time="2026-07-06T02:00:00Z",
            limit=120,
            depth_limit=20,
            chunk_size=50,
        )
    with pytest.raises(ValueError, match="binance_capture_limit_too_large"):
        BinanceCapturePlan(
            capture_id="capture_20260706_000000",
            market_type="spot",
            data_kind="klines",
            symbols=("BTCUSDT",),
            interval="1m",
            start_time="2026-07-06T00:00:00Z",
            end_time="2026-07-06T02:00:00Z",
            limit=1001,
            depth_limit=20,
            chunk_size=50,
        )
