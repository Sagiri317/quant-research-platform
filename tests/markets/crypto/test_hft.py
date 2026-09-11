from __future__ import annotations

import math
from dataclasses import replace

import pandas as pd
import pytest

from quant_research_platform.contracts.order_book import (
    BookBoundary,
    BookState,
    TradeEvent,
)
from quant_research_platform.markets.implementations.crypto.hft import (
    HFT_GRID_MS,
    iter_factor_frames,
    iter_label_frames,
    iter_panel_frames,
    l1_ofi,
)
from quant_research_platform.runtime.resources import (
    OwnedValue,
    runtime_resource_policy_scope,
)
from quant_research_platform.sources.binance.ids import FUTURES_MARKET_TYPE
from tests.helpers.binance_hft import (
    _coverage,
    _panel_rows,
    _row_timestamp_ms,
    coverage_lookup,
)
from tests.helpers.runtime_resources import capture_test_policy


def test_panel_uses_exact_validated_receipt_nanoseconds() -> None:
    observed = pd.Timestamp("2023-01-01T00:00:00.000000123").value
    origin_ms = observed // 1_000_000
    state = replace(
        _state(origin_ms),
        observed_time_ns=observed,
        receive_time=pd.Timestamp(observed, unit="ns").isoformat() + "Z",
    )
    frame = pd.concat(
        tuple(
            iter_panel_frames(
                [state, _state(origin_ms + 300)],
                [],
                start_ms=origin_ms + 100,
                end_ms=origin_ms + 300,
                book_coverage={"s1": _coverage(origin_ms, origin_ms + 400)},
                trade_coverage=coverage_lookup(
                    {"trade-connection-1": _coverage(origin_ms, origin_ms + 400)}
                ),
                market_id="crypto",
                asset_id="btc_usdt",
                batch_rows=4,
            )
        ),
        ignore_index=True,
    )
    assert pd.Timestamp(frame.local_receive_time.iloc[0]).value == observed


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
        observed_time_ns=timestamp_ms * 1_000_000,
        receive_time=pd.Timestamp(timestamp_ms, unit="ms").isoformat() + "Z",
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
        observed_time_ns=timestamp_ms * 1_000_000,
        receive_time=pd.Timestamp(timestamp_ms, unit="ms").isoformat() + "Z",
        receive_monotonic_ns=timestamp_ms + 1,
        segment_id=segment_id,
        connection_id=f"connection-{segment_id}",
        market_type=FUTURES_MARKET_TYPE,
        reason="sequence_gap",
    )


def _trade(timestamp_ms: int, quantity: float, *, maker_buyer: bool) -> TradeEvent:
    return TradeEvent(
        capture_seq=timestamp_ms + 10,
        event_time_ms=timestamp_ms + 1,
        transaction_time_ms=timestamp_ms,
        observed_time_ns=timestamp_ms * 1_000_000,
        receive_time=pd.Timestamp(timestamp_ms, unit="ms").isoformat() + "Z",
        receive_monotonic_ns=timestamp_ms + 10,
        connection_id="trade-connection-1",
        market_type=FUTURES_MARKET_TYPE,
        tick_size="0.1",
        price_tick=1005,
        quantity=quantity,
        visible_quantity=quantity,
        buyer_is_maker=maker_buyer,
    )


def test_l1_ofi_uses_exact_integer_tick_identity() -> None:
    assert l1_ofi((1000, 2.0), (1010, 3.0), (1020, 1.0), (1020, 2.0)) == 2.0


def test_panel_frames_build_from_column_buffers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        pd.DataFrame,
        "from_records",
        lambda *_args, **_kwargs: pytest.fail("hft_panel_row_graph_used"),
    )

    frames = list(
        iter_panel_frames(
            [_state(0), _state(200)],
            [],
            market_id="crypto",
            asset_id="btc_usdt",
            start_ms=0,
            end_ms=200,
            book_coverage={"s1": _coverage(0, 300)},
            trade_coverage=coverage_lookup({"trade-connection-1": _coverage(0, 300)}),
            batch_rows=2,
        )
    )

    assert [len(frame) for frame in frames] == [2, 1]
    assert frames[0]["market_id"].tolist() == ["crypto", "crypto"]
    assert frames[0]["asset_id"].tolist() == ["btc_usdt", "btc_usdt"]


def test_panel_first_trade_failure_releases_first_book_and_closes_children() -> None:
    primary = KeyboardInterrupt("trade_primary")

    class OneBook:
        def __init__(self, value: OwnedValue[BookState]) -> None:
            self.value = value
            self.closed = False

        def __iter__(self):  # noqa: ANN204
            return self

        def __next__(self):  # noqa: ANN204
            if self.value is None:
                raise StopIteration
            value, self.value = self.value, None
            return value

        def close(self) -> None:
            self.closed = True
            raise OSError("book_close_secondary")

    class FailingTrades:
        closed = False

        def __iter__(self):  # noqa: ANN204
            return self

        def __next__(self):  # noqa: ANN204
            raise primary

        def close(self) -> None:
            self.closed = True
            raise OSError("trade_close_secondary")

    policy = capture_test_policy(available_memory_bytes=64 * 1024**2)
    with runtime_resource_policy_scope(policy):
        lease = policy.lease("test.crypto.hft.first_book", 2345)
        owned = OwnedValue(_state(0), lease)
        books = OneBook(owned)
        trades = FailingTrades()
        try:
            with pytest.raises(KeyboardInterrupt) as raised:
                next(
                    iter_panel_frames(
                        books,
                        trades,
                        market_id="crypto",
                        asset_id="btc_usdt",
                        batch_rows=2,
                        start_ms=0,
                        end_ms=100,
                        book_coverage={},
                        trade_coverage=coverage_lookup({}),
                    )
                )
            assert raised.value is primary
            assert not lease.active
            assert books.closed and trades.closed
            assert policy.used_memory_bytes == 0
        finally:
            if lease.active:
                owned.release(owner="test.crypto.hft.first_book")


def test_quiet_valid_book_persists_without_fabricated_gap() -> None:
    rows = list(
        _panel_rows(
            [_state(0), _state(500)],
            [],
            start_ms=0,
            end_ms=500,
            book_coverage={"s1": _coverage(0, 501)},
            trade_coverage={"trade-connection-1": _coverage(0, 501)},
        )
    )

    assert [_row_timestamp_ms(row) for row in rows] == [0, 100, 200, 300, 400, 500]
    assert {row["segment_id"] for row in rows} == {"s1:trade-connection-1"}


def test_panel_excludes_pre_interval_trades_and_keeps_stream_clocks_separate() -> None:
    rows = list(
        _panel_rows(
            [_state(0), _state(600)],
            [_trade(100, 9.0, maker_buyer=False), _trade(500, 2.0, maker_buyer=True)],
            start_ms=500,
            end_ms=500,
            book_coverage={"s1": _coverage(0, 601)},
            trade_coverage={"trade-connection-1": _coverage(0, 601)},
        )
    )

    assert len(rows) == 1
    row = rows[0]
    assert row["trade_volume_100ms"] == 2.0
    assert row["observed_signed_trade_volume_100ms"] == -2.0
    assert row["book_observed_time_ms"] == 0
    assert row["exchange_event_time_ms"] == 1
    assert row["transaction_time_ms"] == 0
    assert row["book_capture_seq"] == 1


def test_panel_ofi_excludes_book_events_before_interval_lower_bound() -> None:
    rows = list(
        _panel_rows(
            [_state(0), _state(50, bid_qty=3.0)],
            [],
            start_ms=200,
            end_ms=200,
            book_coverage={"s1": _coverage(0, 301)},
            trade_coverage={"trade-connection-1": _coverage(0, 301)},
        )
    )

    assert len(rows) == 1
    assert rows[0]["book_ofi_100ms"] == 0.0


def test_explicit_book_boundary_stops_panel_until_new_segment() -> None:
    rows = list(
        _panel_rows(
            [_state(0), _boundary(250), _state(400, segment_id="s2")],
            [],
            start_ms=0,
            end_ms=400,
            book_coverage={"s1": _coverage(0, 250), "s2": _coverage(400, 500)},
            trade_coverage={"trade-connection-1": _coverage(0, 500)},
        )
    )

    assert [(_row_timestamp_ms(row), row["segment_id"]) for row in rows] == [
        (0, "s1:trade-connection-1"),
        (100, "s1:trade-connection-1"),
        (200, "s1:trade-connection-1"),
    ]


def test_factor_rows_are_hand_calculated_from_panel_sufficient_statistics() -> None:
    panel = list(
        _panel_rows(
            [
                _state(0),
                _state(100, bid_tick=1000, ask_tick=1010, bid_qty=3.0, ask_qty=1.0),
                _state(200, bid_tick=1005, ask_tick=1015, bid_qty=4.0, ask_qty=2.0),
            ],
            [_trade(100, 2.0, maker_buyer=False)],
            start_ms=0,
            end_ms=200,
            book_coverage={"s1": _coverage(0, 201)},
            trade_coverage={"trade-connection-1": _coverage(0, 201)},
        )
    )
    factors = pd.concat(
        iter_factor_frames((pd.DataFrame(panel),)),
        ignore_index=True,
    ).to_dict("records")

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
            "timestamp_ms": index * HFT_GRID_MS,
            "timestamp": f"1970-01-01T00:00:00.{index}00",
            "segment_id": "s1" if index < 3 else "s2",
            "mid": 100.0 if index != 1 else 101.0,
            "market_id": "crypto",
            "asset_id": "btc_usdt",
        }
        for index in range(5)
    ]

    labels = pd.concat(
        iter_label_frames(
            (pd.DataFrame(rows),),
            horizons_ms=(HFT_GRID_MS, 200),
        ),
        ignore_index=True,
    ).to_dict("records")

    assert math.isclose(labels[0]["future_mid_return_100ms"], 0.01)
    assert labels[0]["future_mid_direction_100ms"] == 1.0
    assert labels[3]["future_mid_direction_100ms"] == 0.0
    assert math.isnan(labels[1]["future_mid_return_200ms"])


def test_maker_bid_sell_pressure_proxy_uses_future_observed_sell_flow() -> None:
    sell_volume = [0.0, 0.5, 0.5, 0.5, 0.5, 0.0, 0.0]
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range(
                "1970-01-01",
                periods=len(sell_volume),
                freq="100ms",
            ),
            "segment_id": ["s1"] * len(sell_volume),
            "mid": [100.0] * len(sell_volume),
            "best_bid_qty": [2.0] * len(sell_volume),
            "observed_taker_sell_volume_100ms": sell_volume,
            "market_id": ["crypto"] * len(sell_volume),
            "asset_id": ["btc_usdt"] * len(sell_volume),
        }
    )

    labels = pd.concat(
        iter_label_frames((frame,), horizons_ms=(500,)),
        ignore_index=True,
    )

    assert "maker_bid_sell_pressure_proxy_500ms" in labels
    assert set(column for column in labels if column.startswith("maker_bid_")) == {
        "maker_bid_sell_pressure_proxy_500ms",
        "maker_bid_sell_pressure_proxy_500ms_available_at",
    }
    assert labels.loc[0, "maker_bid_sell_pressure_proxy_500ms"] == 1.0
    assert labels.loc[1, "maker_bid_sell_pressure_proxy_500ms"] == 0.75
    assert labels["maker_bid_sell_pressure_proxy_500ms"].iloc[-1:].isna().all()
    available = labels["maker_bid_sell_pressure_proxy_500ms_available_at"]
    assert available.iloc[:2].tolist() == [
        pd.Timestamp("1970-01-01T00:00:00.500000"),
        pd.Timestamp("1970-01-01T00:00:00.600000"),
    ]
    assert available.iloc[2:].isna().all()


def test_factor_and_label_rows_reject_non_grid_gap_in_same_segment() -> None:
    complete = list(
        _panel_rows(
            [_state(0), _state(200)],
            [],
            start_ms=0,
            end_ms=200,
            book_coverage={"s1": _coverage(0, 201)},
            trade_coverage={"trade-connection-1": _coverage(0, 201)},
        )
    )
    rows = [complete[0], complete[2]]

    with pytest.raises(ValueError, match="hft_grid_non_contiguous"):
        list(iter_factor_frames((pd.DataFrame(rows),)))
    with pytest.raises(ValueError, match="hft_grid_non_contiguous"):
        list(
            iter_label_frames(
                (pd.DataFrame(rows),),
                horizons_ms=(HFT_GRID_MS,),
            )
        )


def test_factor_and_label_frames_match_across_chunk_boundaries() -> None:
    panel = pd.DataFrame(
        _panel_rows(
            [_state(index * HFT_GRID_MS) for index in range(15)],
            [],
            start_ms=0,
            end_ms=14 * HFT_GRID_MS,
            book_coverage={"s1": _coverage(0, 14 * HFT_GRID_MS + 1)},
            trade_coverage={"trade-connection-1": _coverage(0, 14 * HFT_GRID_MS + 1)},
        )
    )
    chunks = (panel.iloc[:7].copy(), panel.iloc[7:].copy())

    factor_full = pd.concat(iter_factor_frames((panel,)), ignore_index=True)
    factor_chunked = pd.concat(iter_factor_frames(chunks), ignore_index=True)
    label_full = pd.concat(
        iter_label_frames((panel,), horizons_ms=(HFT_GRID_MS, 500)),
        ignore_index=True,
    )
    label_chunked = pd.concat(
        iter_label_frames(chunks, horizons_ms=(HFT_GRID_MS, 500)),
        ignore_index=True,
    )

    pd.testing.assert_frame_equal(factor_chunked, factor_full, check_exact=True)
    pd.testing.assert_frame_equal(label_chunked, label_full, check_exact=True)
