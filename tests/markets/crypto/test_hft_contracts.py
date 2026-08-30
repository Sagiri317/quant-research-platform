from __future__ import annotations

import inspect
import math

import pandas as pd
import pytest

from quant_research_platform.contracts.order_book import (
    BookBoundary,
    BookState,
    TradeEvent,
)
from quant_research_platform.markets.implementations.crypto.execution import (
    MakerConfig,
    TakerConfig,
    simulate_maker,
    simulate_taker,
)
from quant_research_platform.markets.implementations.crypto.hft import (
    HFT_GRID_MS,
    iter_label_frames,
)
from quant_research_platform.markets.implementations.crypto.research_scenarios import (
    HFT_MODEL_LABEL_HORIZON_MS,
    build_hft_model_catalog,
)
from quant_research_platform.sources.binance.ids import FUTURES_MARKET_TYPE
from tests.helpers.binance_hft import (
    _coverage,
    _panel_rows,
    _quote,
    _row_timestamp_ms,
)


def _state(
    observed_time_ms: int,
    *,
    event_time_ms: int = 0,
    transaction_time_ms: int = 0,
    bid_tick: int = 995,
    ask_tick: int = 1005,
    bid_qty: float = 2.0,
    ask_qty: float = 1.0,
    segment_id: str = "book-1",
    connection_id: str = "book-connection-1",
    capture_seq: int | None = None,
    receive_monotonic_ns: int | None = None,
) -> BookState:
    sequence = observed_time_ms + 1 if capture_seq is None else capture_seq
    monotonic = (
        observed_time_ms + 1 if receive_monotonic_ns is None else receive_monotonic_ns
    )
    return BookState(
        capture_seq=sequence,
        event_time_ms=event_time_ms,
        transaction_time_ms=transaction_time_ms,
        observed_time_ns=observed_time_ms * 1_000_000,
        receive_time="2026-08-27T00:00:00.000000Z",
        receive_monotonic_ns=monotonic,
        update_id=sequence,
        segment_id=segment_id,
        connection_id=connection_id,
        market_type=FUTURES_MARKET_TYPE,
        tick_size="0.1",
        bids=((bid_tick, bid_qty), (bid_tick - 5, 3.0)),
        asks=((ask_tick, ask_qty), (ask_tick + 5, 2.0)),
    )


def _boundary(
    observed_time_ms: int,
    *,
    segment_id: str = "book-1",
    connection_id: str = "book-connection-1",
) -> BookBoundary:
    return BookBoundary(
        capture_seq=observed_time_ms + 1,
        event_time_ms=observed_time_ms,
        transaction_time_ms=observed_time_ms,
        observed_time_ns=observed_time_ms * 1_000_000,
        receive_time="2026-08-27T00:00:00.000000Z",
        receive_monotonic_ns=observed_time_ms + 1,
        segment_id=segment_id,
        connection_id=connection_id,
        market_type=FUTURES_MARKET_TYPE,
        reason="reconnect_boundary",
    )


def _trade(
    observed_time_ms: int,
    *,
    quantity: float = 1.0,
    visible_quantity: float | None = None,
    price_tick: int = 995,
    connection_id: str = "trade-connection-1",
    capture_seq: int | None = None,
    receive_monotonic_ns: int | None = None,
) -> TradeEvent:
    sequence = observed_time_ms + 10 if capture_seq is None else capture_seq
    monotonic = (
        observed_time_ms + 10 if receive_monotonic_ns is None else receive_monotonic_ns
    )
    return TradeEvent(
        capture_seq=sequence,
        event_time_ms=observed_time_ms,
        transaction_time_ms=observed_time_ms,
        observed_time_ns=observed_time_ms * 1_000_000,
        receive_time="2026-08-27T00:00:00.000000Z",
        receive_monotonic_ns=monotonic,
        connection_id=connection_id,
        market_type=FUTURES_MARKET_TYPE,
        tick_size="0.1",
        price_tick=price_tick,
        quantity=quantity,
        visible_quantity=quantity if visible_quantity is None else visible_quantity,
        buyer_is_maker=True,
    )


def _maker_config() -> MakerConfig:
    return MakerConfig(
        order_quantity=1.0,
        max_position=2.0,
        maker_fee_bps=0.0,
        inventory_target=0.0,
        inventory_skew=0.0,
        markout_horizon_ms=100,
    )


def _taker_config(*, latency_ms: int = 0) -> TakerConfig:
    return TakerConfig(
        order_quantity=1.0,
        position_size=1.0,
        latency_ms=latency_ms,
        taker_fee_bps=0.0,
        threshold_bps=0.0,
        markout_horizon_ms=100,
    )


def _state_at_ns(
    observed_time_ns: int,
    *,
    capture_seq: int,
    bid_tick: int = 995,
    ask_tick: int = 1005,
    bid_qty: float = 2.0,
    ask_qty: float = 1.0,
    segment_id: str = "book-1",
) -> BookState:
    observed_time_ms = observed_time_ns // 1_000_000
    return BookState(
        capture_seq=capture_seq,
        event_time_ms=observed_time_ms,
        transaction_time_ms=observed_time_ms,
        observed_time_ns=observed_time_ns,
        receive_time="2026-08-27T00:00:00.000000Z",
        receive_monotonic_ns=capture_seq,
        update_id=capture_seq,
        segment_id=segment_id,
        connection_id=f"connection-{segment_id}",
        market_type=FUTURES_MARKET_TYPE,
        tick_size="0.1",
        bids=((bid_tick, bid_qty), (bid_tick - 5, 3.0)),
        asks=((ask_tick, ask_qty), (ask_tick + 5, 2.0)),
    )


def test_unobserved_state_is_hidden_from_panel_and_taker() -> None:
    state = _state(100, event_time_ms=0, transaction_time_ms=0)

    rows = list(
        _panel_rows(
            [state],
            [],
            start_ms=0,
            end_ms=0,
            book_coverage={"book-1": _coverage(0, 200)},
            trade_coverage={"trade-connection-1": _coverage(0, 200)},
        )
    )
    assert rows == []

    fills: list[dict[str, object]] = []
    diagnostics = simulate_taker(
        [state],
        [{"timestamp_ms": 50, "prediction": 0.1}],
        prediction_column="prediction",
        config=_taker_config(),
        observed_coverage={"book-1": _coverage(0, 200)},
        on_fill=fills.append,
    )
    assert diagnostics["filled_orders"] == 0
    assert fills == []


def test_exact_ns_observation_boundary_is_hidden_from_panel_and_taker() -> None:
    initial = _state_at_ns(0, capture_seq=1)
    just_after_boundary = _state_at_ns(100_000_001, capture_seq=2)
    coverage = {"book-1": (0, 200_000_000)}
    trade_coverage = {"trade-connection-1": (0, 200_000_000)}

    rows = list(
        _panel_rows(
            [initial, just_after_boundary],
            [],
            start_ms=100,
            end_ms=100,
            book_coverage=coverage,
            trade_coverage=trade_coverage,
        )
    )
    assert rows[0]["book_capture_seq"] == initial.capture_seq

    fills: list[dict[str, object]] = []
    diagnostics = simulate_taker(
        [just_after_boundary],
        [{"timestamp_ms": 100, "prediction": 1.0}],
        prediction_column="prediction",
        config=_taker_config(),
        observed_coverage=coverage,
        on_fill=fills.append,
    )
    assert diagnostics["filled_orders"] == 0
    assert fills == []

    maker = simulate_maker(
        [just_after_boundary],
        [],
        [_quote(100)],
        config=_maker_config(),
        observed_coverage=coverage,
        trade_coverage=trade_coverage,
    )
    assert maker["created_orders"] == 0


def test_panel_coverage_uses_exact_half_open_nanoseconds() -> None:
    state = _state_at_ns(100_000_000, capture_seq=1)
    common = {
        "book_events": [state],
        "trades": [],
        "start_ms": 200,
        "end_ms": 200,
        "trade_coverage": {"trade-connection-1": (0, 300_000_000)},
    }

    covered = list(
        _panel_rows(
            **common,
            book_coverage={"book-1": (100_000_000, 300_000_000)},
        )
    )
    not_fully_covered = list(
        _panel_rows(
            **common,
            book_coverage={"book-1": (100_000_001, 300_000_000)},
        )
    )
    assert len(covered) == 1
    assert not_fully_covered == []


def test_trade_gap_is_missing_not_zero_volume() -> None:
    books = [_state(0), _state(300)]
    missing = list(
        _panel_rows(
            books,
            [],
            start_ms=100,
            end_ms=200,
            book_coverage={"book-1": _coverage(0, 300)},
            trade_coverage={"trade-connection-1": _coverage(0, 100)},
        )
    )
    assert missing == []

    complete = list(
        _panel_rows(
            books,
            [],
            start_ms=100,
            end_ms=200,
            book_coverage={"book-1": _coverage(0, 300)},
            trade_coverage={"trade-connection-1": _coverage(0, 300)},
        )
    )
    assert [_row_timestamp_ms(row) for row in complete] == [100, 200]
    assert [row["trade_volume_100ms"] for row in complete] == [0.0, 0.0]


def test_depth_reconnect_does_not_make_first_partial_interval_valid() -> None:
    rows = list(
        _panel_rows(
            [
                _state(0),
                _boundary(150),
                _state(200, segment_id="book-2", connection_id="book-connection-2"),
            ],
            [],
            start_ms=100,
            end_ms=200,
            book_coverage={"book-1": _coverage(0, 150), "book-2": _coverage(200, 300)},
            trade_coverage={"trade-connection-1": _coverage(0, 300)},
        )
    )
    assert [_row_timestamp_ms(row) for row in rows] == [100]


def test_trade_reconnect_does_not_make_first_partial_interval_valid() -> None:
    rows = list(
        _panel_rows(
            [_state(0)],
            [_trade(200, connection_id="trade-connection-2")],
            start_ms=100,
            end_ms=300,
            book_coverage={"book-1": _coverage(0, 400)},
            trade_coverage={
                "trade-connection-1": _coverage(0, 150),
                "trade-connection-2": _coverage(150, 400),
            },
        )
    )
    assert [_row_timestamp_ms(row) for row in rows] == [100, 300]


def test_maker_uses_local_observation_order_for_same_time_events() -> None:
    initial = _state(0, capture_seq=1, receive_monotonic_ns=1)
    changed = _state(
        100,
        bid_tick=994,
        ask_tick=1006,
        capture_seq=2,
        receive_monotonic_ns=20,
    )
    book_first_trade = _trade(
        100,
        quantity=3.0,
        capture_seq=3,
        receive_monotonic_ns=30,
    )
    trade_first_book = _state(
        100,
        bid_tick=994,
        ask_tick=1006,
        capture_seq=3,
        receive_monotonic_ns=30,
    )
    trade_first = _trade(
        100,
        quantity=3.0,
        capture_seq=2,
        receive_monotonic_ns=20,
    )

    book_first = simulate_maker(
        [initial, changed],
        [book_first_trade],
        [_quote(0)],
        config=_maker_config(),
        observed_coverage={"book-1": _coverage(0, 200)},
        trade_coverage={"trade-connection-1": _coverage(0, 200)},
    )
    trade_first_result = simulate_maker(
        [initial, trade_first_book],
        [trade_first],
        [_quote(0)],
        config=_maker_config(),
        observed_coverage={"book-1": _coverage(0, 200)},
        trade_coverage={"trade-connection-1": _coverage(0, 200)},
    )

    assert book_first["filled_orders"] == 0
    assert trade_first_result["filled_orders"] == 1
    assert book_first["same_timestamp_policy"] == "local_observation_order"


def test_total_trade_flow_uses_q_but_maker_queue_uses_visible_quantity() -> None:
    trade = _trade(50, quantity=2.0, visible_quantity=1.0)
    rows = list(
        _panel_rows(
            [_state(0)],
            [trade],
            start_ms=100,
            end_ms=100,
            book_coverage={"book-1": _coverage(0, 200)},
            trade_coverage={"trade-connection-1": _coverage(0, 200)},
        )
    )
    assert rows[0]["trade_volume_100ms"] == 2.0

    result = simulate_maker(
        [_state(0)],
        [trade],
        [_quote(0)],
        config=_maker_config(),
        observed_coverage={"book-1": _coverage(0, 200)},
        trade_coverage={"trade-connection-1": _coverage(0, 200)},
    )
    assert result["filled_orders"] == 0


def test_zero_visible_trade_through_cannot_fill_a_maker_order() -> None:
    result = simulate_maker(
        [_state(0)],
        [_trade(50, quantity=1.0, visible_quantity=0.0, price_tick=994)],
        [_quote(0)],
        config=_maker_config(),
        observed_coverage={"book-1": _coverage(0, 200)},
        trade_coverage={"trade-connection-1": _coverage(0, 200)},
    )
    assert result["filled_orders"] == 0


def test_zero_visible_trade_through_does_not_advance_queue() -> None:
    result = simulate_maker(
        [_state(0)],
        [
            _trade(50, quantity=1.0, visible_quantity=0.0, price_tick=994),
            _trade(100, quantity=1.0, visible_quantity=1.0, price_tick=995),
        ],
        [_quote(0)],
        config=_maker_config(),
        observed_coverage={"book-1": _coverage(0, 200)},
        trade_coverage={"trade-connection-1": _coverage(0, 200)},
    )
    assert result["filled_orders"] == 0


def test_trade_through_consumes_full_remaining_order_not_visible_quantity() -> None:
    fills: list[dict[str, object]] = []
    result = simulate_maker(
        [_state(0)],
        [_trade(50, quantity=1.0, visible_quantity=0.1, price_tick=994)],
        [_quote(0)],
        config=_maker_config(),
        observed_coverage={"book-1": (0, 200_000_000)},
        trade_coverage={"trade-connection-1": (0, 200_000_000)},
        on_fill=fills.append,
    )
    assert result["fill_quantity"] == 1.0
    assert fills[0]["fill_qty"] == 1.0


def test_maker_max_abs_position_updates_on_terminal_fill() -> None:
    result = simulate_maker(
        [_state(0)],
        [_trade(50, quantity=1.0, visible_quantity=1.0, price_tick=994)],
        [_quote(0)],
        config=_maker_config(),
        observed_coverage={"book-1": (0, 200_000_000)},
        trade_coverage={"trade-connection-1": (0, 200_000_000)},
    )
    assert result["max_abs_position"] == 1.0


def test_maker_order_expires_at_first_missing_trade_coverage_boundary() -> None:
    fills: list[dict[str, object]] = []
    result = simulate_maker(
        [_state(0)],
        [
            _trade(50, quantity=1.0, visible_quantity=1.0, price_tick=995),
            _trade(100, quantity=1.0, visible_quantity=1.0, price_tick=994),
        ],
        [_quote(0)],
        config=_maker_config(),
        observed_coverage={"book-1": (0, 200_000_000)},
        trade_coverage={"trade-connection-1": (0, 75_000_000)},
        on_fill=fills.append,
    )
    assert fills == []
    assert result["average_order_lifetime_ms"] == 75.0
    assert result["created_orders"] == 2
    assert result["zero_fill_orders"] == 2


def test_maker_terminal_order_uses_coverage_validity_end() -> None:
    result = simulate_maker(
        [_state(0)],
        [],
        [_quote(0)],
        config=_maker_config(),
        observed_coverage={"book-1": _coverage(0, 200)},
        trade_coverage={"trade-connection-1": _coverage(0, 75)},
    )

    assert result["average_order_lifetime_ms"] == 75.0


def test_maker_requires_matching_book_and_trade_schedule_identity() -> None:
    result = simulate_maker(
        [_state(0)],
        [],
        [_quote(0, book_segment_id="book-2", trade_connection_id="trade-connection-2")],
        config=_maker_config(),
        observed_coverage={"book-1": _coverage(0, 200)},
        trade_coverage={"trade-connection-1": _coverage(0, 200)},
    )
    assert result["created_orders"] == 0


def test_book_state_lease_and_taker_timing_and_terminal_mark() -> None:
    fills: list[dict[str, object]] = []
    equities: list[dict[str, object]] = []
    simulate_taker(
        [
            _state(5, event_time_ms=1, transaction_time_ms=1),
            _state(20, bid_tick=1005, ask_tick=1015),
        ],
        [{"timestamp_ms": 0, "prediction": 0.1}],
        prediction_column="prediction",
        config=_taker_config(latency_ms=10),
        observed_coverage={"book-1": _coverage(0, 20)},
        on_fill=fills.append,
        on_equity=equities.append,
    )

    assert fills[0]["decision_time_ms"] == 0
    assert fills[0]["effective_time_ms"] == 10
    assert fills[0]["state_observed_time_ms"] == 5
    assert fills[0]["timestamp_ms"] == 10
    assert equities[0]["timestamp_ms"] == 10

    terminal = simulate_taker(
        [_state(0), _state(100, bid_tick=1005, ask_tick=1015)],
        [{"timestamp_ms": 0, "prediction": 0.1}],
        prediction_column="prediction",
        config=_taker_config(),
        observed_coverage={"book-1": _coverage(0, 101)},
    )
    assert math.isclose(terminal["gross_pnl_usd"], 0.5)


def test_same_price_book_decrease_without_trade_does_not_deplete_queue() -> None:
    result = simulate_maker(
        [
            _state(0, bid_qty=1.0),
            _state(50, bid_qty=0.5),
        ],
        [_trade(100, quantity=1.0, visible_quantity=1.0)],
        [_quote(0)],
        config=_maker_config(),
        observed_coverage={"book-1": _coverage(0, 200)},
        trade_coverage={"trade-connection-1": _coverage(0, 200)},
    )
    assert result["filled_orders"] == 0


def test_maker_quote_schedule_has_no_prediction_surface() -> None:
    assert "quote_schedule" in inspect.signature(simulate_maker).parameters
    with pytest.raises(ValueError, match="prediction"):
        simulate_maker(
            [_state(0)],
            [],
            [{"timestamp_ms": 0, "prediction_score": 1.0}],
            config=_maker_config(),
            observed_coverage={"book-1": _coverage(0, 100)},
            trade_coverage={"trade-connection-1": _coverage(0, 100)},
        )


def test_hft_grid_is_the_only_model_and_label_step_owner() -> None:
    assert HFT_GRID_MS == 100
    models = build_hft_model_catalog(training_row_bound=1).list()
    assert all(
        model.purge_gap_steps == HFT_MODEL_LABEL_HORIZON_MS // HFT_GRID_MS
        for model in models
    )
    rows = [
        {
            "timestamp_ms": index * HFT_GRID_MS,
            "segment_id": "book-1",
            "mid": 100.0,
        }
        for index in range(2)
    ]
    labels = pd.concat(
        iter_label_frames(
            (pd.DataFrame(rows),),
            horizons_ms=(HFT_GRID_MS,),
        ),
        ignore_index=True,
    ).to_dict("records")
    assert labels[0][f"future_mid_return_{HFT_GRID_MS}ms"] == 0.0
