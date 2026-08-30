from __future__ import annotations

import math

import pytest

import quant_research_platform.markets.implementations.crypto.execution as execution
from quant_research_platform.contracts.order_book import (
    BookBoundary,
    BookState,
    TradeEvent,
    price_to_tick,
)
from quant_research_platform.markets.implementations.crypto.execution import (
    MakerConfig,
    TakerConfig,
    _Metrics,
    _PendingMarkouts,
    simulate_maker,
    simulate_taker,
    taker_latency_sweep,
)
from quant_research_platform.runtime.resources import (
    OwnedValue,
    runtime_resource_policy_scope,
)
from quant_research_platform.sources.binance.ids import FUTURES_MARKET_TYPE
from tests.helpers.binance_hft import _coverage, _quote
from tests.helpers.runtime_resources import capture_test_policy


def _state(
    timestamp_ms: int,
    *,
    bid_tick: int = 995,
    ask_tick: int = 1005,
    bid_levels: tuple[tuple[int, float], ...] | None = None,
    ask_levels: tuple[tuple[int, float], ...] | None = None,
    segment_id: str = "s1",
) -> BookState:
    return BookState(
        capture_seq=timestamp_ms + 1,
        event_time_ms=timestamp_ms + 1,
        transaction_time_ms=timestamp_ms,
        observed_time_ns=timestamp_ms * 1_000_000,
        receive_time="2026-08-27T00:00:00.000000Z",
        receive_monotonic_ns=timestamp_ms + 1,
        update_id=timestamp_ms + 1,
        segment_id=segment_id,
        connection_id=f"connection-{segment_id}",
        market_type=FUTURES_MARKET_TYPE,
        tick_size="0.1",
        bids=bid_levels or ((bid_tick, 2.0), (bid_tick - 5, 3.0)),
        asks=ask_levels or ((ask_tick, 1.0), (ask_tick + 5, 2.0)),
    )


def _boundary(timestamp_ms: int, *, segment_id: str = "s1") -> BookBoundary:
    return BookBoundary(
        capture_seq=timestamp_ms + 1,
        event_time_ms=timestamp_ms + 1,
        transaction_time_ms=timestamp_ms,
        observed_time_ns=timestamp_ms * 1_000_000,
        receive_time="2026-08-27T00:00:00.000000Z",
        receive_monotonic_ns=timestamp_ms + 1,
        segment_id=segment_id,
        connection_id=f"connection-{segment_id}",
        market_type=FUTURES_MARKET_TYPE,
        reason="sequence_gap",
    )


def _trade(
    timestamp_ms: int,
    *,
    price: str,
    quantity: str,
    maker_buyer: bool,
) -> TradeEvent:
    quantity_value = float(quantity)
    return TradeEvent(
        capture_seq=timestamp_ms + 10,
        event_time_ms=timestamp_ms,
        transaction_time_ms=timestamp_ms,
        observed_time_ns=timestamp_ms * 1_000_000,
        receive_time="2026-08-27T00:00:00.000000Z",
        receive_monotonic_ns=timestamp_ms + 10,
        connection_id="trade-connection-1",
        market_type=FUTURES_MARKET_TYPE,
        tick_size="0.1",
        price_tick=price_to_tick(price, "0.1", error="bad"),
        quantity=quantity_value,
        visible_quantity=quantity_value,
        buyer_is_maker=maker_buyer,
    )


def _taker_config(*, latency_ms: int = 0, order_quantity: float = 1.0) -> TakerConfig:
    return TakerConfig(
        order_quantity=order_quantity,
        position_size=2.0,
        latency_ms=latency_ms,
        taker_fee_bps=1.0,
        threshold_bps=0.0,
        markout_horizon_ms=100,
    )


def _maker_config() -> MakerConfig:
    return MakerConfig(
        order_quantity=1.0,
        max_position=2.0,
        maker_fee_bps=-1.0,
        inventory_target=0.0,
        inventory_skew=0.0,
        markout_horizon_ms=100,
    )


def test_market_event_first_trade_failure_releases_first_book_and_children() -> None:
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
        lease = policy.lease("test.crypto.execution.first_book", 1234)
        owned = OwnedValue(_state(0), lease)
        books = OneBook(owned)
        trades = FailingTrades()
        try:
            with pytest.raises(KeyboardInterrupt) as raised:
                next(execution._market_events(books, trades))
            assert raised.value is primary
            assert not lease.active
            assert books.closed and trades.closed
            assert policy.used_memory_bytes == 0
        finally:
            if lease.active:
                owned.release(owner="test.crypto.execution.first_book")


def test_taker_uses_latest_actual_state_at_effective_time() -> None:
    fills = []
    diagnostics = simulate_taker(
        [_state(5), _state(15, bid_tick=1005, ask_tick=1015)],
        [{"timestamp_ms": 0, "prediction": 0.02, "segment_id": "s1"}],
        prediction_column="prediction",
        config=_taker_config(latency_ms=10),
        observed_coverage={"s1": _coverage(5, 15)},
        on_fill=fills.append,
    )

    assert fills[0]["decision_time_ms"] == 0
    assert fills[0]["effective_time_ms"] == 10
    assert fills[0]["state_observed_time_ms"] == 5
    assert fills[0]["timestamp_ms"] == 10
    assert fills[0]["fill_price"] == 100.5
    assert diagnostics["latency_fidelity"] == "full_l2_observed_event"
    assert diagnostics["assumption"] == "historical_book_zero_market_impact"


def test_pending_markout_close_failure_preserves_body_primary() -> None:
    class FailingClose:
        close_calls = 0

        def close(self) -> None:
            self.close_calls += 1
            raise OSError("close_failure")

    handle = FailingClose()
    pending = object.__new__(_PendingMarkouts)
    pending._file = handle
    pending.expire_all = lambda: None

    with pytest.raises(KeyboardInterrupt, match="body_primary") as raised:
        with pending:
            raise KeyboardInterrupt("body_primary")

    assert handle.close_calls == 1
    assert raised.value.__notes__ == [
        "execution pending markout handle close also failed: OSError"
    ]


def test_pending_markout_rejects_disk_before_tempfile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        execution,
        "require_free_disk_headroom",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            OSError("pending_markout_disk_headroom")
        ),
        raising=False,
    )
    monkeypatch.setattr(
        execution.tempfile,
        "TemporaryFile",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("temporary_file_created_before_disk_preflight")
        ),
    )

    with pytest.raises(OSError, match="pending_markout_disk_headroom"):
        _PendingMarkouts(_Metrics(), None)


def test_pending_markout_rechecks_disk_and_closes_after_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[int] = []

    def gate(_path, *, reserve_bytes: int) -> int:
        calls.append(reserve_bytes)
        if len(calls) == 2:
            raise OSError("pending_markout_disk_headroom")
        return reserve_bytes

    monkeypatch.setattr(
        execution,
        "require_free_disk_headroom",
        gate,
        raising=False,
    )
    monkeypatch.setattr(
        execution,
        "_PENDING_MARKOUT_DISK_CHECK_BYTES",
        1,
        raising=False,
    )
    pending = _PendingMarkouts(_Metrics(), None)
    fill = {
        "fill_qty": 1.0,
        "fill_price": 100.0,
        "side": "buy",
        "segment_id": "s1",
        "markout_target_time_ns": 1,
    }

    with pytest.raises(OSError, match="pending_markout_disk_headroom"):
        with pending:
            pending.add(fill)

    assert len(calls) == 2
    assert pending._file.closed is True


def test_taker_walks_known_l2_and_reports_partial_coverage_honestly() -> None:
    fills = []
    diagnostics = simulate_taker(
        [
            _state(
                0,
                ask_levels=((1005, 0.4), (1010, 0.6)),
            )
        ],
        [{"timestamp_ms": 0, "prediction": 0.02}],
        prediction_column="prediction",
        config=_taker_config(order_quantity=1.5),
        observed_coverage={"s1": _coverage(0, 1)},
        on_fill=fills.append,
    )

    assert len(fills) == 1
    assert math.isclose(fills[0]["fill_qty"], 1.0)
    assert fills[0]["levels_consumed"] == 2
    assert math.isclose(fills[0]["fill_price"], 100.8)
    assert math.isclose(diagnostics["unfilled_quantity"], 0.5)
    assert diagnostics["zero_fill_orders"] == 0
    assert "unfilled_orders" not in diagnostics
    assert diagnostics["markout_undefined_fill_events"] == 1
    assert diagnostics["average_markout_bps"] is None


def test_taker_does_not_fill_after_coverage_or_explicit_boundary() -> None:
    fills = []
    diagnostics = simulate_taker(
        [_state(0), _boundary(50)],
        [{"timestamp_ms": 100, "prediction": 0.02}],
        prediction_column="prediction",
        config=_taker_config(),
        observed_coverage={"s1": _coverage(0, 49)},
        on_fill=fills.append,
    )

    assert fills == []
    assert diagnostics["filled_orders"] == 0
    assert diagnostics["gross_pnl_usd"] == 0.0


def test_taker_terminal_pnl_is_unknown_after_book_boundary() -> None:
    fills = []
    diagnostics = simulate_taker(
        [_state(0), _boundary(50)],
        [{"timestamp_ms": 0, "prediction": 0.02}],
        prediction_column="prediction",
        config=_taker_config(),
        observed_coverage={"s1": _coverage(0, 50)},
        on_fill=fills.append,
    )

    assert len(fills) == 1
    assert diagnostics["gross_pnl_usd"] is None
    assert diagnostics["net_pnl_usd"] is None


def test_taker_markout_does_not_use_state_past_observed_coverage() -> None:
    diagnostics = simulate_taker(
        [_state(0), _state(200, bid_tick=1005, ask_tick=1015)],
        [{"timestamp_ms": 0, "prediction": 0.02}],
        prediction_column="prediction",
        config=_taker_config(),
        observed_coverage={"s1": _coverage(0, 100)},
    )

    assert diagnostics["average_markout_bps"] is None
    assert diagnostics["markout_undefined_fill_events"] == 1


def test_taker_markout_uses_latest_state_as_of_target() -> None:
    fills: list[dict[str, object]] = []
    diagnostics = simulate_taker(
        [
            _state(0, bid_tick=1000, ask_tick=1005),
            _state(50, bid_tick=1005, ask_tick=1015),
            _state(200, bid_tick=1095, ask_tick=1105),
        ],
        [{"timestamp_ms": 0, "prediction": 0.1}],
        prediction_column="prediction",
        config=_taker_config(),
        observed_coverage={"s1": _coverage(0, 300)},
        on_fill=fills.append,
    )

    assert diagnostics["markout_undefined_fill_events"] == 0
    assert fills[0]["markout_usd_per_unit"] == pytest.approx(0.5)


def test_taker_target_position_does_not_accumulate_repeated_signal() -> None:
    diagnostics = simulate_taker(
        [_state(0), _state(100), _state(200)],
        [
            {"timestamp_ms": 0, "prediction": 0.02},
            {"timestamp_ms": 100, "prediction": 0.02},
            {"timestamp_ms": 200, "prediction": 0.02},
        ],
        prediction_column="prediction",
        config=TakerConfig(2.0, 2.0, 0, 0.0, 0.0, 100),
        observed_coverage={"s1": _coverage(0, 200)},
    )

    assert diagnostics["created_orders"] == 1
    assert diagnostics["max_abs_position"] == 2.0


def test_maker_processes_same_time_events_in_local_observation_order() -> None:
    fills = []
    diagnostics = simulate_maker(
        [_state(0), _state(100, bid_tick=994, ask_tick=1006)],
        [_trade(100, price="99.5", quantity="3", maker_buyer=True)],
        [_quote(0, book_segment_id="s1"), _quote(100, book_segment_id="s1")],
        config=_maker_config(),
        observed_coverage={"s1": _coverage(0, 100)},
        trade_coverage={"trade-connection-1": _coverage(0, 101)},
        on_fill=fills.append,
    )

    assert fills == []
    assert diagnostics["same_timestamp_policy"] == "local_observation_order"
    assert diagnostics["queue_model"] == "estimated_queue_ahead"


def test_maker_same_price_depletion_partial_fill_and_order_lifetime() -> None:
    fills = []
    diagnostics = simulate_maker(
        [_state(0), _state(200, bid_tick=1000, ask_tick=1010)],
        [
            _trade(50, price="99.5", quantity="2.5", maker_buyer=True),
            _trade(100, price="99.5", quantity="0.5", maker_buyer=True),
        ],
        [_quote(0, book_segment_id="s1")],
        config=_maker_config(),
        observed_coverage={"s1": _coverage(0, 201)},
        trade_coverage={"trade-connection-1": _coverage(0, 201)},
        on_fill=fills.append,
    )

    assert [fill["fill_qty"] for fill in fills] == [0.5, 0.5]
    assert diagnostics["partial_fill_events"] == 1
    assert diagnostics["filled_orders"] == 1
    assert diagnostics["average_filled_order_lifetime_ms"] == 100.0
    assert diagnostics["average_markout_bps"] is not None
    assert diagnostics["fees_usd"] < 0
    assert diagnostics["spread_capture_pnl_usd"] > 0
    assert diagnostics["inventory_revaluation_pnl_usd"] is not None


def test_maker_trade_through_fills_and_boundary_blocks_stale_trade() -> None:
    fills = []
    diagnostics = simulate_maker(
        [_state(0), _boundary(75)],
        [
            _trade(50, price="99.4", quantity="0.1", maker_buyer=True),
            _trade(100, price="99.0", quantity="1", maker_buyer=True),
        ],
        [_quote(0, book_segment_id="s1")],
        config=_maker_config(),
        observed_coverage={"s1": _coverage(0, 74)},
        trade_coverage={"trade-connection-1": _coverage(0, 101)},
        on_fill=fills.append,
    )

    assert len(fills) == 1
    assert fills[0]["fill_reason"] == "trade_through"
    assert diagnostics["trade_through_fill_events"] == 1
    assert diagnostics["gross_pnl_usd"] is None


def test_maker_terminal_pnl_is_unknown_after_uncovered_state() -> None:
    fills = []
    diagnostics = simulate_maker(
        [_state(0), _boundary(75), _state(100, segment_id="s2")],
        [_trade(50, price="99.4", quantity="1", maker_buyer=True)],
        [_quote(0, book_segment_id="s1")],
        config=_maker_config(),
        observed_coverage={"s1": _coverage(0, 75)},
        trade_coverage={"trade-connection-1": _coverage(0, 101)},
        on_fill=fills.append,
    )

    assert len(fills) == 1
    assert diagnostics["gross_pnl_usd"] is None
    assert diagnostics["net_pnl_usd"] is None


def test_maker_terminal_pnl_preserves_flat_realized_cash_after_boundary() -> None:
    fills = []
    diagnostics = simulate_maker(
        [_state(0), _boundary(75)],
        [
            _trade(50, price="99.4", quantity="1", maker_buyer=True),
            _trade(60, price="100.6", quantity="1", maker_buyer=False),
        ],
        [_quote(0, book_segment_id="s1")],
        config=_maker_config(),
        observed_coverage={"s1": _coverage(0, 75)},
        trade_coverage={"trade-connection-1": _coverage(0, 101)},
        on_fill=fills.append,
    )

    assert len(fills) == 2
    assert diagnostics["gross_pnl_usd"] == pytest.approx(1.0)
    assert diagnostics["net_pnl_usd"] == pytest.approx(1.02)


def test_latency_sweep_has_exact_requested_scenarios() -> None:
    result = taker_latency_sweep(
        lambda: [_state(0), _state(100), _state(200)],
        lambda: [{"timestamp_ms": 0, "prediction": 0.02}],
        prediction_column="prediction",
        config=_taker_config(),
        latency_scenarios_ms=(0, 100, 200),
        observed_coverage={"s1": _coverage(0, 200)},
    )

    assert [item["latency_ms"] for item in result] == [0, 100, 200]


def test_taker_long_stream_retains_no_fill_or_equity_dataframe() -> None:
    def decisions():
        for index in range(20_000):
            yield {"timestamp_ms": index, "prediction": 0.0}

    diagnostics = simulate_taker(
        [_state(0), _state(19_999)],
        decisions(),
        prediction_column="prediction",
        config=_taker_config(),
        observed_coverage={"s1": _coverage(0, 19_999)},
    )

    assert diagnostics["fill_events"] == 0
    assert diagnostics["equity_rows"] == 20_000
    assert not any("dataframe" in key.lower() for key in diagnostics)
