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
from quant_research_platform.contracts.portfolios import (
    FundingForecast,
    PortfolioCapital,
)
from quant_research_platform.markets.implementations.crypto.execution import (
    MakerConfig,
    TakerConfig,
    _Metrics,
    _PendingMarkouts,
)
from quant_research_platform.markets.implementations.crypto.execution import (
    simulate_maker as _simulate_maker,
)
from quant_research_platform.markets.implementations.crypto.execution import (
    simulate_taker as _simulate_taker,
)
from quant_research_platform.research.portfolios.diagnostics import (
    fixed_turnover_cost_sensitivity,
)
from quant_research_platform.runtime.resources import (
    OwnedValue,
    runtime_resource_policy_scope,
)
from quant_research_platform.sources.binance.ids import FUTURES_MARKET_TYPE
from tests.helpers.binance_hft import (
    KNOWN_NO_FUNDING,
    _coverage,
    _quote,
    with_test_resource_policy,
)
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
        markout_horizon_ms=100,
    )


def _maker_config() -> MakerConfig:
    return MakerConfig(
        latency_ms=0,
        order_quantity=1.0,
        max_position=2.0,
        maker_fee_bps=-1.0,
        inventory_target=0.0,
        inventory_skew=0.0,
        markout_horizon_ms=100,
    )


simulate_taker = with_test_resource_policy(_simulate_taker)
simulate_maker = with_test_resource_policy(_simulate_maker)


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
        [_state(0), _state(5), _state(15, bid_tick=1005, ask_tick=1015)],
        [{"timestamp_ms": 0, "prediction": 0.02, "segment_id": "s1"}],
        funding_forecasts=KNOWN_NO_FUNDING,
        prediction_horizon_ms=500,
        prediction_column="prediction",
        config=_taker_config(latency_ms=10),
        observed_coverage={"s1": _coverage(0, 15)},
        on_fill=fills.append,
        capital=PortfolioCapital("synthetic", 1000.0, "USDT", "linear_perpetual"),
        asset_id="btc_usdt",
        funding_events=(),
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
    pending._next = None
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

    policy = capture_test_policy(available_memory_bytes=64 * 1024**2)
    with pytest.raises(OSError, match="pending_markout_disk_headroom"):
        _PendingMarkouts(_Metrics(), None, policy)


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
    policy = capture_test_policy(available_memory_bytes=64 * 1024**2)
    pending = _PendingMarkouts(_Metrics(), None, policy)
    fill = {
        "fill_qty": 1.0,
        "fill_price": 100.0,
        "side": "buy",
        "segment_id": "s1",
        "markout_target_time_ns": 1,
    }

    with runtime_resource_policy_scope(policy):
        with pytest.raises(OSError, match="pending_markout_disk_headroom"):
            with pending:
                pending.add(fill)

    assert len(calls) == 2
    assert pending._file.closed is True


def test_pending_markout_encode_and_retained_decode_are_admitted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = capture_test_policy(
        available_memory_bytes=64 * 1024**2,
        explicit_cap_bytes=16 * 1024**2,
    )
    observed: list[tuple[str, int]] = []
    dumps = execution.json.dumps
    parse_json = execution.parse_qrp_json_object

    def observe_encode(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
        observed.append(("encode", policy.used_memory_bytes))
        return dumps(*args, **kwargs)

    def observe_decode(raw: bytes):
        observed.append(("decode", policy.used_memory_bytes))
        return parse_json(raw)

    monkeypatch.setattr(execution.json, "dumps", observe_encode)
    monkeypatch.setattr(execution, "parse_qrp_json_object", observe_decode)
    fill = {
        "fill_qty": 1.0,
        "fill_price": 100.0,
        "side": "buy",
        "segment_id": "s1",
        "markout_target_time_ns": 1,
    }

    with runtime_resource_policy_scope(policy):
        with _PendingMarkouts(_Metrics(), None, policy) as pending:
            pending.add(fill)
            assert policy.used_memory_bytes == 0
            assert pending._peek() == fill
            assert policy.used_memory_bytes > 0
            pending.expire_all()
            assert policy.used_memory_bytes == 0

    assert {name for name, _used in observed} == {"encode", "decode"}
    assert all(used > 0 for _name, used in observed)


def test_execution_rejects_fill_before_explicit_cap_is_crossed() -> None:
    policy = capture_test_policy(
        available_memory_bytes=1024 * 1024,
        explicit_cap_bytes=1,
    )

    with runtime_resource_policy_scope(policy):
        with pytest.raises(MemoryError, match="runtime_memory_budget_exceeded"):
            _simulate_taker(
                [_state(0)],
                [{"timestamp_ms": 0, "prediction": 0.02}],
                funding_forecasts=KNOWN_NO_FUNDING,
                prediction_horizon_ms=500,
                prediction_column="prediction",
                config=_taker_config(),
                observed_coverage={"s1": _coverage(0, 1)},
                resource_policy=policy,
                capital=PortfolioCapital(
                    "synthetic", 1000.0, "USDT", "linear_perpetual"
                ),
                asset_id="btc_usdt",
                funding_events=(),
            )

    assert policy.used_memory_bytes == 0


def test_failing_fill_observer_is_not_replayed_during_cleanup() -> None:
    calls: list[int] = []
    policy = capture_test_policy(available_memory_bytes=64 * 1024**2)

    def fail(fill):  # noqa: ANN001, ANN202
        calls.append(fill["order_id"])
        raise KeyboardInterrupt("observer_primary")

    with runtime_resource_policy_scope(policy):
        with pytest.raises(KeyboardInterrupt, match="observer_primary"):
            _simulate_taker(
                [_state(0), _state(200)],
                [{"timestamp_ms": 0, "prediction": 0.02}],
                funding_forecasts=KNOWN_NO_FUNDING,
                prediction_horizon_ms=500,
                prediction_column="prediction",
                config=_taker_config(),
                observed_coverage={"s1": _coverage(0, 300)},
                resource_policy=policy,
                on_fill=fail,
                capital=PortfolioCapital(
                    "synthetic", 1000.0, "USDT", "linear_perpetual"
                ),
                asset_id="btc_usdt",
                funding_events=(),
            )

    assert calls == [1]
    assert policy.used_memory_bytes == 0


def test_taker_walks_known_l2_and_reports_partial_coverage_honestly() -> None:
    fills = []
    diagnostics = simulate_taker(
        [
            _state(0),
            _state(
                5,
                ask_levels=((1005, 0.4), (1010, 0.6)),
            ),
        ],
        [{"timestamp_ms": 0, "prediction": 0.02}],
        funding_forecasts=KNOWN_NO_FUNDING,
        prediction_horizon_ms=500,
        prediction_column="prediction",
        config=_taker_config(latency_ms=10, order_quantity=1.5),
        observed_coverage={"s1": _coverage(0, 11)},
        on_fill=fills.append,
        capital=PortfolioCapital("synthetic", 1000.0, "USDT", "linear_perpetual"),
        asset_id="btc_usdt",
        funding_events=(),
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
        funding_forecasts=KNOWN_NO_FUNDING,
        prediction_horizon_ms=500,
        prediction_column="prediction",
        config=_taker_config(),
        observed_coverage={"s1": _coverage(0, 49)},
        on_fill=fills.append,
        capital=PortfolioCapital("synthetic", 1000.0, "USDT", "linear_perpetual"),
        asset_id="btc_usdt",
        funding_events=(),
    )

    assert fills == []
    assert diagnostics["filled_orders"] == 0
    assert diagnostics["gross_pnl"] == 0.0


def test_taker_terminal_pnl_is_unknown_after_book_boundary() -> None:
    fills = []
    diagnostics = simulate_taker(
        [_state(0), _boundary(50)],
        [{"timestamp_ms": 0, "prediction": 0.02}],
        funding_forecasts=KNOWN_NO_FUNDING,
        prediction_horizon_ms=500,
        prediction_column="prediction",
        config=_taker_config(),
        observed_coverage={"s1": _coverage(0, 50)},
        on_fill=fills.append,
        capital=PortfolioCapital("synthetic", 1000.0, "USDT", "linear_perpetual"),
        asset_id="btc_usdt",
        funding_events=(),
    )

    assert len(fills) == 1
    assert diagnostics["gross_pnl"] is None
    assert diagnostics["net_pnl"] is None


def test_taker_markout_does_not_use_state_past_observed_coverage() -> None:
    diagnostics = simulate_taker(
        [_state(0), _state(200, bid_tick=1005, ask_tick=1015)],
        [{"timestamp_ms": 0, "prediction": 0.02}],
        funding_forecasts=KNOWN_NO_FUNDING,
        prediction_horizon_ms=500,
        prediction_column="prediction",
        config=_taker_config(),
        observed_coverage={"s1": _coverage(0, 100)},
        capital=PortfolioCapital("synthetic", 1000.0, "USDT", "linear_perpetual"),
        asset_id="btc_usdt",
        funding_events=(),
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
        funding_forecasts=KNOWN_NO_FUNDING,
        prediction_horizon_ms=500,
        prediction_column="prediction",
        config=_taker_config(),
        observed_coverage={"s1": _coverage(0, 300)},
        on_fill=fills.append,
        capital=PortfolioCapital("synthetic", 1000.0, "USDT", "linear_perpetual"),
        asset_id="btc_usdt",
        funding_events=(),
    )

    assert diagnostics["markout_undefined_fill_events"] == 0
    assert fills[0]["markout_per_unit"] == pytest.approx(0.5)


def test_taker_target_position_does_not_accumulate_repeated_signal() -> None:
    diagnostics = simulate_taker(
        [_state(0), _state(100), _state(200)],
        [
            {"timestamp_ms": 0, "prediction": 0.02},
            {"timestamp_ms": 100, "prediction": 0.02},
            {"timestamp_ms": 200, "prediction": 0.02},
        ],
        funding_forecasts=KNOWN_NO_FUNDING,
        prediction_horizon_ms=500,
        prediction_column="prediction",
        config=TakerConfig(2.0, 2.0, 0, 0.0, 100),
        observed_coverage={"s1": _coverage(0, 200)},
        capital=PortfolioCapital("synthetic", 1000.0, "USDT", "linear_perpetual"),
        asset_id="btc_usdt",
        funding_events=(),
    )

    assert diagnostics["created_orders"] == 1
    assert diagnostics["max_abs_position"] == 2.0


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
        capital=PortfolioCapital("synthetic", 1000.0, "USDT", "linear_perpetual"),
        asset_id="btc_usdt",
        funding_events=(),
    )

    assert [fill["fill_qty"] for fill in fills] == [0.5, 0.5]
    assert diagnostics["partial_fill_events"] == 1
    assert diagnostics["filled_orders"] == 1
    assert diagnostics["average_filled_order_lifetime_ms"] == 100.0
    assert diagnostics["average_markout_bps"] is not None
    assert diagnostics["fees"] < 0
    assert diagnostics["spread_capture_pnl"] > 0
    assert diagnostics["inventory_revaluation_pnl"] is not None


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
        capital=PortfolioCapital("synthetic", 1000.0, "USDT", "linear_perpetual"),
        asset_id="btc_usdt",
        funding_events=(),
    )

    assert len(fills) == 1
    assert fills[0]["fill_reason"] == "trade_through"
    assert diagnostics["trade_through_fill_events"] == 1
    assert diagnostics["gross_pnl"] is None


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
        capital=PortfolioCapital("synthetic", 1000.0, "USDT", "linear_perpetual"),
        asset_id="btc_usdt",
        funding_events=(),
    )

    assert len(fills) == 1
    assert diagnostics["gross_pnl"] is None
    assert diagnostics["net_pnl"] is None


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
        capital=PortfolioCapital("synthetic", 1000.0, "USDT", "linear_perpetual"),
        asset_id="btc_usdt",
        funding_events=(),
    )

    assert len(fills) == 2
    assert diagnostics["gross_pnl"] == pytest.approx(1.0)
    assert diagnostics["net_pnl"] == pytest.approx(1.02)


def test_cost_sensitivity_reprices_actual_turnover_and_net_pnl() -> None:
    result = fixed_turnover_cost_sensitivity(
        turnover=10_000.0,
        net_pnl=5.0,
        base_fee_bps=1.0,
        fee_scenarios_bps=(1.0, 2.0),
        additional_spread_scenarios_bps=(0.0, 1.0),
    )

    assert [row["net_pnl"] for row in result] == pytest.approx([5.0, 4.0, 4.0, 3.0])
    assert {row["method"] for row in result} == {
        "constant_actual_turnover_additive_sensitivity"
    }


def test_taker_long_stream_retains_no_fill_or_equity_dataframe() -> None:
    def decisions():
        for index in range(20_000):
            yield {"timestamp_ms": index, "prediction": 0.0}

    diagnostics = simulate_taker(
        [_state(0), _state(19_999)],
        decisions(),
        funding_forecasts=KNOWN_NO_FUNDING,
        prediction_horizon_ms=500,
        prediction_column="prediction",
        config=_taker_config(),
        observed_coverage={"s1": _coverage(0, 19_999)},
        capital=PortfolioCapital("synthetic", 1000.0, "USDT", "linear_perpetual"),
        asset_id="btc_usdt",
        funding_events=(),
    )

    assert diagnostics["fill_events"] == 0
    assert diagnostics["equity_rows"] == 20_000
    assert not any("dataframe" in key.lower() for key in diagnostics)


def test_taker_freezes_orders_across_overlapping_latency_and_future_book_changes():
    from quant_research_platform.contracts.portfolios import FundingEvent

    selections = []
    fills_by_case = []
    for future_quantity in (2.0, 0.25):
        ledger = []
        fills = []
        policy = capture_test_policy(available_memory_bytes=64 * 1024**2)
        with runtime_resource_policy_scope(policy):
            result = _simulate_taker(
                [
                    _state(0),
                    _state(25, ask_levels=((1100, future_quantity),)),
                    _state(80),
                ],
                [{"timestamp_ms": stamp, "prediction": 0.02} for stamp in (0, 10, 20)],
                funding_forecasts=KNOWN_NO_FUNDING,
                prediction_column="prediction",
                prediction_horizon_ms=100,
                config=_taker_config(latency_ms=30),
                observed_coverage={"s1": _coverage(0, 100)},
                resource_policy=policy,
                capital=PortfolioCapital("fixture", 1000.0, "USDT", "linear_perpetual"),
                asset_id="btc_usdt",
                funding_events=(),
                on_fill=fills.append,
                on_ledger=ledger.append,
            )
            assert policy.used_memory_bytes == 0
        quantities = [
            row.number
            for batch in ledger
            for row in batch.diagnostics
            if row.name == "decision_requested_quantity"
        ]
        selections.append(quantities)
        fills_by_case.append([row["fill_qty"] for row in fills])
        assert result["created_orders"] == 2 and result["max_abs_position"] <= 2.0
    assert selections == [[1.0, 1.0, 0.0], [1.0, 1.0, 0.0]]
    assert fills_by_case == [[1.0, 1.0], [0.25, 0.25]]

    # Funding settled while flat, before the frozen order becomes effective,
    # does not enter that order's holding-period expected cost.
    result = simulate_taker(
        [_state(0), _state(100)],
        [{"timestamp_ms": 0, "prediction": 0.02}],
        funding_forecasts=KNOWN_NO_FUNDING,
        prediction_column="prediction",
        prediction_horizon_ms=100,
        config=_taker_config(latency_ms=50),
        observed_coverage={"s1": _coverage(0, 101)},
        capital=PortfolioCapital("fixture", 1000.0, "USDT", "linear_perpetual"),
        asset_id="btc_usdt",
        funding_events=(FundingEvent(25_000_000, None, None),),
    )
    assert result["selection_status"] == "COMPLETE"
    assert result["created_orders"] == 1 and result["funding"] == 0.0

    # A forecast ending before funding does not force liquidation: its existing
    # inventory still settles the later real cash flow. The next forecast has
    # missing expected funding evidence and blocks selection, not actual accounting.
    for rate in (0.001, -0.001):
        ledger = []
        result = simulate_taker(
            [_state(0), _state(50), _state(100)],
            [
                {"timestamp_ms": 0, "prediction": 0.02},
                {"timestamp_ms": 50, "prediction": 0.02},
            ],
            funding_forecasts=(
                FundingForecast(0, 1_000_000_000, ((75_000_000, None),)),
            ),
            prediction_column="prediction",
            prediction_horizon_ms=50,
            config=_taker_config(),
            observed_coverage={"s1": _coverage(0, 101)},
            capital=PortfolioCapital("fixture", 1000.0, "USDT", "linear_perpetual"),
            asset_id="btc_usdt",
            funding_events=(FundingEvent(75_000_000, rate, 100.0),),
            on_ledger=ledger.append,
        )
        assert (
            result["selection_status"] == "BLOCKED" and result["blocked_decisions"] == 1
        )
        assert result["funding"] == pytest.approx(-100 * rate)
        assert result["net_pnl"] == pytest.approx(-0.5 - 0.01005 - 100 * rate)
        assert any(
            row.text == "portfolio_expected_funding_missing"
            for batch in ledger
            for row in batch.diagnostics
        )


def test_taker_depth_breakpoint_optimum_zero_and_unavailable_decision():
    cases = [
        (
            [_state(0, ask_levels=((1005, 0.4), (1030, 10.0)))],
            0,
            0.02,
            0,
            0.4,
            "positive_expected_net_gain",
        ),
        ([_state(0)], 0, 0.0, 0, 0.0, "no_positive_expected_net_gain"),
        ([_state(5)], 0, 0.02, 10, 0.0, "decision_book_unavailable"),
        ([_state(0)], 0, 0.02, 100, 0.0, "forecast_expired_before_execution"),
    ]
    for books, stamp, prediction, latency, quantity, reason in cases:
        equity = []
        fills = []
        result = simulate_taker(
            books,
            [{"timestamp_ms": stamp, "prediction": prediction}],
            funding_forecasts=KNOWN_NO_FUNDING,
            prediction_column="prediction",
            prediction_horizon_ms=100,
            config=_taker_config(latency_ms=latency, order_quantity=2.0),
            observed_coverage={"s1": _coverage(0, 200)},
            capital=PortfolioCapital("fixture", 1000.0, "USDT", "linear_perpetual"),
            asset_id="btc_usdt",
            funding_events=(),
            on_fill=fills.append,
            on_equity=equity.append,
        )
        assert equity[0]["requested_quantity"] == quantity
        assert equity[0]["decision_reason"] == reason
        assert result["selection_status"] == (
            "UNDEFINED"
            if reason
            in {"decision_book_unavailable", "forecast_expired_before_execution"}
            else "COMPLETE"
        )
        assert result["created_orders"] == int(quantity > 0)
        if quantity:
            assert fills[0]["requested_quantity"] == 0.4
            assert fills[0]["expected_net_gain"] == pytest.approx(
                0.4 * (102 - 100.5 - 0.01005)
            )


def test_taker_rejects_nonpositive_forecast_and_releases_pending_orders_on_primary():
    for prediction in (-1.0, -2.0):
        with pytest.raises(ValueError, match="hft_prediction_price_invalid"):
            simulate_taker(
                [_state(0)],
                [{"timestamp_ms": 0, "prediction": prediction}],
                funding_forecasts=KNOWN_NO_FUNDING,
                prediction_column="prediction",
                prediction_horizon_ms=100,
                config=_taker_config(),
                observed_coverage={"s1": _coverage(0, 100)},
                capital=PortfolioCapital("fixture", 1000.0, "USDT", "linear_perpetual"),
                asset_id="btc_usdt",
                funding_events=(),
            )
    primary = KeyboardInterrupt("decision_primary")
    closed = []
    calls = []
    policy = capture_test_policy(available_memory_bytes=64 * 1024**2)
    with runtime_resource_policy_scope(policy):

        def books():
            try:
                for stamp in (0, 100):
                    yield OwnedValue(_state(stamp), policy.lease("fixture.book", 4096))
            finally:
                closed.append(True)
                raise OSError("book_close_secondary")

        def emit(batch):
            calls.append(batch.sequence)
            if len(calls) == 2:
                raise primary

        with pytest.raises(KeyboardInterrupt) as raised:
            _simulate_taker(
                books(),
                [{"timestamp_ms": stamp, "prediction": 0.02} for stamp in (0, 10)],
                funding_forecasts=KNOWN_NO_FUNDING,
                prediction_column="prediction",
                prediction_horizon_ms=100,
                config=_taker_config(latency_ms=30),
                observed_coverage={"s1": _coverage(0, 101)},
                resource_policy=policy,
                capital=PortfolioCapital("fixture", 1000.0, "USDT", "linear_perpetual"),
                asset_id="btc_usdt",
                funding_events=(),
                on_ledger=emit,
            )
        assert raised.value is primary and closed
        assert policy.used_memory_bytes == 0

    policy = capture_test_policy(
        available_memory_bytes=64 * 1024**2,
        explicit_cap_bytes=320 * 1024,
    )
    decisions_seen = []
    with runtime_resource_policy_scope(policy):
        with pytest.raises(MemoryError, match="runtime_memory_budget_exceeded"):
            _simulate_taker(
                [_state(0, ask_levels=((1005, 0.001),))],
                [{"timestamp_ms": stamp, "prediction": 0.02} for stamp in range(400)],
                funding_forecasts=KNOWN_NO_FUNDING,
                prediction_column="prediction",
                prediction_horizon_ms=1000,
                config=_taker_config(latency_ms=500),
                observed_coverage={"s1": _coverage(0, 1001)},
                resource_policy=policy,
                capital=PortfolioCapital("fixture", 1000.0, "USDT", "linear_perpetual"),
                asset_id="btc_usdt",
                funding_events=(),
                on_ledger=lambda batch: decisions_seen.append(batch.sequence),
            )
        assert len(decisions_seen) > 2
        assert policy.used_memory_bytes == 0


def test_taker_terminal_mark_cannot_reuse_price_before_pending_effective_time():
    result = simulate_taker(
        [_state(0), _state(20)],
        [
            {"timestamp_ms": 0, "prediction": 0.02},
            {"timestamp_ms": 20, "prediction": 0.02},
        ],
        funding_forecasts=KNOWN_NO_FUNDING,
        prediction_column="prediction",
        prediction_horizon_ms=100,
        config=_taker_config(latency_ms=10),
        observed_coverage={"s1": _coverage(0, 25)},
        capital=PortfolioCapital("fixture", 1000.0, "USDT", "linear_perpetual"),
        asset_id="btc_usdt",
        funding_events=(),
    )
    assert result["filled_orders"] == 1 and result["unfilled_quantity"] == 1.0
    assert result["net_pnl"] is None and result["nav"] is None
