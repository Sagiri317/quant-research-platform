from __future__ import annotations

import math

import pytest

from quant_research_platform.contracts.order_book import BookBoundary, BookState
from quant_research_platform.markets.implementations.crypto.hft import (
    GRID_MS,
    iter_factor_rows,
    iter_label_rows,
    iter_panel_rows,
    l1_ofi,
)
from quant_research_platform.sources.binance.ids import FUTURES_MARKET_TYPE


def _state(
    timestamp_ms: int,
    *,
    bid_tick: int = 995,
    ask_tick: int = 1005,
    bid_qty: float = 2.0,
    ask_qty: float = 1.0,
    segment_id: str = "s1",
) -> BookState:
    return BookState(
        capture_seq=timestamp_ms + 1,
        event_time_ms=timestamp_ms + 1,
        transaction_time_ms=timestamp_ms,
        receive_time="2026-08-27T00:00:00.000000Z",
        receive_monotonic_ns=timestamp_ms + 1,
        update_id=timestamp_ms + 1,
        segment_id=segment_id,
        connection_id=f"connection-{segment_id}",
        market_type=FUTURES_MARKET_TYPE,
        tick_size="0.1",
        bids=((bid_tick, bid_qty), (bid_tick - 10, 3.0)),
        asks=((ask_tick, ask_qty), (ask_tick + 10, 2.0)),
    )


def _boundary(timestamp_ms: int, *, segment_id: str = "s1") -> BookBoundary:
    return BookBoundary(
        capture_seq=timestamp_ms + 1,
        event_time_ms=timestamp_ms + 1,
        transaction_time_ms=timestamp_ms,
        receive_time="2026-08-27T00:00:00.000000Z",
        receive_monotonic_ns=timestamp_ms + 1,
        segment_id=segment_id,
        connection_id=f"connection-{segment_id}",
        market_type=FUTURES_MARKET_TYPE,
        reason="sequence_gap",
    )


def _trade(timestamp_ms: int, quantity: float, *, maker_buyer: bool) -> dict:
    return {
        "capture_seq": timestamp_ms + 10,
        "receive_time": "2026-08-27T00:00:00.000000Z",
        "receive_monotonic_ns": timestamp_ms + 10,
        "market_type": FUTURES_MARKET_TYPE,
        "tick_size": "0.1",
        "payload": {
            "e": "aggTrade",
            "E": timestamp_ms + 1,
            "T": timestamp_ms,
            "s": "BTCUSDT",
            "ps": "BTCUSDT",
            "st": 1,
            "a": timestamp_ms + 1,
            "f": timestamp_ms + 1,
            "l": timestamp_ms + 1,
            "p": "100.5",
            "q": str(quantity),
            "m": maker_buyer,
        },
    }


def test_l1_ofi_uses_exact_integer_tick_identity() -> None:
    assert l1_ofi((1000, 2.0), (1010, 3.0), (1020, 1.0), (1020, 2.0)) == 2.0


def test_quiet_valid_book_persists_without_fabricated_gap() -> None:
    rows = list(
        iter_panel_rows(
            [_state(0), _state(500)],
            [],
            start_ms=0,
            end_ms=500,
        )
    )

    assert [row["timestamp_ms"] for row in rows] == [0, 100, 200, 300, 400, 500]
    assert {row["segment_id"] for row in rows} == {"s1"}


def test_panel_excludes_pre_interval_trades_and_keeps_stream_clocks_separate() -> None:
    rows = list(
        iter_panel_rows(
            [_state(0), _state(600)],
            [_trade(100, 9.0, maker_buyer=False), _trade(500, 2.0, maker_buyer=True)],
            start_ms=500,
            end_ms=500,
        )
    )

    assert len(rows) == 1
    row = rows[0]
    assert row["trade_volume_100ms"] == 2.0
    assert row["observed_signed_trade_volume_100ms"] == -2.0
    assert row["book_state_time_ms"] == 0
    assert row["exchange_event_time_ms"] == 1
    assert row["transaction_time_ms"] == 0
    assert row["capture_seq"] == 1


def test_explicit_book_boundary_stops_panel_until_new_segment() -> None:
    rows = list(
        iter_panel_rows(
            [_state(0), _boundary(250), _state(400, segment_id="s2")],
            [],
            start_ms=0,
            end_ms=400,
        )
    )

    assert [(row["timestamp_ms"], row["segment_id"]) for row in rows] == [
        (0, "s1"),
        (100, "s1"),
        (200, "s1"),
        (400, "s2"),
    ]


def test_factor_rows_are_hand_calculated_from_panel_sufficient_statistics() -> None:
    panel = list(
        iter_panel_rows(
            [
                _state(0),
                _state(100, bid_tick=1000, ask_tick=1010, bid_qty=3.0, ask_qty=1.0),
                _state(200, bid_tick=1005, ask_tick=1015, bid_qty=4.0, ask_qty=2.0),
            ],
            [_trade(100, 2.0, maker_buyer=False)],
            start_ms=0,
            end_ms=200,
        )
    )
    factors = list(iter_factor_rows(panel))

    assert factors[1]["spread"] == 1.0
    assert math.isclose(factors[1]["imbalance_1"], 0.5)
    assert factors[1]["ofi_100ms"] == 4.0
    assert factors[1]["signed_trade_flow_100ms"] == 2.0
    assert math.isclose(factors[1]["return_100ms"], 0.5 / 100.0)
    assert math.isnan(factors[0]["return_100ms"])
    assert not {"best_bid_price", "trade_count_100ms"}.intersection(factors[0])


def test_labels_are_exact_horizon_same_segment_and_keep_zero_distinct_from_nan() -> (
    None
):
    rows = [
        {
            "timestamp_ms": index * GRID_MS,
            "timestamp": f"1970-01-01T00:00:00.{index}00",
            "segment_id": "s1" if index < 3 else "s2",
            "mid": 100.0 if index != 1 else 101.0,
            "market_id": "crypto",
            "asset_id": "btc_usdt",
        }
        for index in range(5)
    ]

    labels = list(iter_label_rows(rows, horizons_ms=(100, 200)))

    assert math.isclose(labels[0]["future_mid_return_100ms"], 0.01)
    assert labels[0]["future_mid_direction_100ms"] == 1.0
    assert labels[3]["future_mid_direction_100ms"] == 0.0
    assert math.isnan(labels[1]["future_mid_return_200ms"])


def test_factor_and_label_rows_reject_non_grid_gap_in_same_segment() -> None:
    complete = list(
        iter_panel_rows(
            [_state(0), _state(200)],
            [],
            start_ms=0,
            end_ms=200,
        )
    )
    rows = [complete[0], complete[2]]

    with pytest.raises(ValueError, match="hft_grid_non_contiguous"):
        list(iter_factor_rows(rows))
    with pytest.raises(ValueError, match="hft_grid_non_contiguous"):
        list(iter_label_rows(rows, horizons_ms=(100,)))
