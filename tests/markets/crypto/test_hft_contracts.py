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
from quant_research_platform.contracts.portfolios import PortfolioCapital
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
from quant_research_platform.sources.binance.ids import FUTURES_MARKET_TYPE
from tests.helpers.binance_hft import (
    KNOWN_NO_FUNDING,
    _coverage,
    _panel_rows,
    _quote,
    _row_timestamp_ms,
)
from tests.helpers.portfolio_ledger import capture_fill_rows
from tests.helpers.research_configuration import model_specs, research_configuration


def _state(
    observed_time_ms: int,
    *,
    event_time_ms: int = 0,
    transaction_time_ms: int = 0,
    bid_index: int = 995,
    ask_index: int = 1005,
    bid_qty: float = 2.0,
    ask_qty: float = 1.0,
    segment_id: str = "book-1",
    observation_id: str = "book-connection-1",
    source_sequence: int | None = None,
    receive_monotonic_ns: int | None = None,
) -> BookState:
    sequence = observed_time_ms + 1 if source_sequence is None else source_sequence
    monotonic = (
        observed_time_ms + 1 if receive_monotonic_ns is None else receive_monotonic_ns
    )
    return BookState(
        source_sequence=sequence,
        event_time_ms=event_time_ms,
        transaction_time_ms=transaction_time_ms,
        observed_time_ns=observed_time_ms * 1000000,
        receive_time="2026-08-27T00:00:00.000000Z",
        receive_monotonic_ns=monotonic,
        update_id=sequence,
        segment_id=segment_id,
        observation_id=observation_id,
        market_type=FUTURES_MARKET_TYPE,
        price_unit="0.1",
        bids=((bid_index, bid_qty), (bid_index - 5, 3.0)),
        asks=((ask_index, ask_qty), (ask_index + 5, 2.0)),
    )


def _boundary(
    observed_time_ms: int,
    *,
    segment_id: str = "book-1",
    observation_id: str = "book-connection-1",
) -> BookBoundary:
    return BookBoundary(
        source_sequence=observed_time_ms + 1,
        event_time_ms=observed_time_ms,
        transaction_time_ms=observed_time_ms,
        observed_time_ns=observed_time_ms * 1000000,
        receive_time="2026-08-27T00:00:00.000000Z",
        receive_monotonic_ns=observed_time_ms + 1,
        segment_id=segment_id,
        observation_id=observation_id,
        market_type=FUTURES_MARKET_TYPE,
        reason="reconnect_boundary",
    )


def _trade(
    observed_time_ms: int,
    *,
    quantity: float = 1.0,
    visible_quantity: float | None = None,
    price_index: int = 995,
    observation_id: str = "trade-connection-1",
    source_sequence: int | None = None,
    receive_monotonic_ns: int | None = None,
) -> TradeEvent:
    sequence = observed_time_ms + 10 if source_sequence is None else source_sequence
    monotonic = (
        observed_time_ms + 10 if receive_monotonic_ns is None else receive_monotonic_ns
    )
    return TradeEvent(
        source_sequence=sequence,
        event_time_ms=observed_time_ms,
        transaction_time_ms=observed_time_ms,
        observed_time_ns=observed_time_ms * 1000000,
        receive_time="2026-08-27T00:00:00.000000Z",
        receive_monotonic_ns=monotonic,
        observation_id=observation_id,
        market_type=FUTURES_MARKET_TYPE,
        price_unit="0.1",
        price_index=price_index,
        quantity=quantity,
        visible_quantity=quantity if visible_quantity is None else visible_quantity,
        buyer_is_maker=True,
    )


def _maker_config() -> MakerConfig:
    return MakerConfig(
        latency_ms=0,
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
        markout_horizon_ms=100,
    )


def _state_at_ns(
    observed_time_ns: int,
    *,
    source_sequence: int,
    bid_index: int = 995,
    ask_index: int = 1005,
    bid_qty: float = 2.0,
    ask_qty: float = 1.0,
    segment_id: str = "book-1",
) -> BookState:
    observed_time_ms = observed_time_ns // 1000000
    return BookState(
        source_sequence=source_sequence,
        event_time_ms=observed_time_ms,
        transaction_time_ms=observed_time_ms,
        observed_time_ns=observed_time_ns,
        receive_time="2026-08-27T00:00:00.000000Z",
        receive_monotonic_ns=source_sequence,
        update_id=source_sequence,
        segment_id=segment_id,
        observation_id=f"connection-{segment_id}",
        market_type=FUTURES_MARKET_TYPE,
        price_unit="0.1",
        bids=((bid_index, bid_qty), (bid_index - 5, 3.0)),
        asks=((ask_index, ask_qty), (ask_index + 5, 2.0)),
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
        funding_forecasts=KNOWN_NO_FUNDING,
        prediction_horizon_ms=500,
        prediction_column="prediction",
        config=_taker_config(),
        observed_coverage={"book-1": _coverage(0, 200)},
        on_ledger=capture_fill_rows(fills.append),
        capital=PortfolioCapital("synthetic", 1000.0, "USDT", "linear_perpetual"),
        asset_id="btc_usdt",
        funding_events=(),
    )
    assert diagnostics["filled_orders"] == 0
    assert fills == []


def test_exact_ns_observation_boundary_is_hidden_from_panel_and_taker() -> None:
    initial = _state_at_ns(0, source_sequence=1)
    just_after_boundary = _state_at_ns(100000001, source_sequence=2)
    coverage = {"book-1": (0, 200000000)}
    trade_coverage = {"trade-connection-1": (0, 200000000)}
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
    assert rows[0]["book_source_sequence"] == initial.source_sequence
    fills: list[dict[str, object]] = []
    diagnostics = simulate_taker(
        [just_after_boundary],
        [{"timestamp_ms": 100, "prediction": 1.0}],
        funding_forecasts=KNOWN_NO_FUNDING,
        prediction_horizon_ms=500,
        prediction_column="prediction",
        config=_taker_config(),
        observed_coverage=coverage,
        on_ledger=capture_fill_rows(fills.append),
        capital=PortfolioCapital("synthetic", 1000.0, "USDT", "linear_perpetual"),
        asset_id="btc_usdt",
        funding_events=(),
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
        capital=PortfolioCapital("synthetic", 1000.0, "USDT", "linear_perpetual"),
        asset_id="btc_usdt",
        funding_events=(),
    )
    assert maker["created_orders"] == 0


def test_panel_coverage_uses_exact_half_open_nanoseconds() -> None:
    state = _state_at_ns(100000000, source_sequence=1)
    common = {
        "book_events": [state],
        "trades": [],
        "start_ms": 200,
        "end_ms": 200,
        "trade_coverage": {"trade-connection-1": (0, 300000000)},
    }
    covered = list(
        _panel_rows(**common, book_coverage={"book-1": (100000000, 300000000)})
    )
    not_fully_covered = list(
        _panel_rows(**common, book_coverage={"book-1": (100000001, 300000000)})
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
    assert [_row_timestamp_ms(row) for row in missing] == [100, 200]
    assert all(pd.isna(row["trade_volume_100ms"]) for row in missing)
    assert all(pd.isna(row["trade_observation_id"]) for row in missing)
    assert all(row["book_spread"] == 1.0 for row in missing)
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
                _state(200, segment_id="book-2", observation_id="book-connection-2"),
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
            [_trade(200, observation_id="trade-connection-2")],
            start_ms=100,
            end_ms=300,
            book_coverage={"book-1": _coverage(0, 400)},
            trade_coverage={
                "trade-connection-1": _coverage(0, 150),
                "trade-connection-2": _coverage(150, 400),
            },
        )
    )
    assert [_row_timestamp_ms(row) for row in rows] == [100, 200, 300]
    assert rows[0]["trade_volume_100ms"] == rows[2]["trade_volume_100ms"] == 0.0
    assert pd.isna(rows[1]["trade_volume_100ms"])
    assert len({row["segment_id"] for row in rows}) == 1


def test_maker_uses_local_observation_order_for_same_time_events() -> None:
    book_first_fills, trade_first_fills = ([], [])
    initial = _state(0, source_sequence=1, receive_monotonic_ns=1)
    changed = _state(
        100, bid_index=994, ask_index=1008, source_sequence=3, receive_monotonic_ns=20
    )
    book_first_trade = _trade(
        100, quantity=3.0, source_sequence=2, receive_monotonic_ns=30
    )
    trade_first_book = _state(
        100, bid_index=994, ask_index=1008, source_sequence=2, receive_monotonic_ns=30
    )
    trade_first = _trade(100, quantity=3.0, source_sequence=3, receive_monotonic_ns=20)
    book_first = simulate_maker(
        [initial, changed],
        [book_first_trade],
        [_quote(0)],
        config=_maker_config(),
        observed_coverage={"book-1": _coverage(0, 200)},
        trade_coverage={"trade-connection-1": _coverage(0, 200)},
        capital=PortfolioCapital("synthetic", 1000.0, "USDT", "linear_perpetual"),
        asset_id="btc_usdt",
        funding_events=(),
        on_ledger=capture_fill_rows(book_first_fills.append),
    )
    trade_first_result = simulate_maker(
        [initial, trade_first_book],
        [trade_first],
        [_quote(0)],
        config=_maker_config(),
        observed_coverage={"book-1": _coverage(0, 200)},
        trade_coverage={"trade-connection-1": _coverage(0, 200)},
        capital=PortfolioCapital("synthetic", 1000.0, "USDT", "linear_perpetual"),
        asset_id="btc_usdt",
        funding_events=(),
        on_ledger=capture_fill_rows(trade_first_fills.append),
    )
    assert book_first["filled_orders"] == 1
    assert trade_first_result["filled_orders"] == 1
    assert book_first_fills[0]["spread_capture_per_unit"] == pytest.approx(0.6)
    assert trade_first_fills[0]["spread_capture_per_unit"] == pytest.approx(0.5)
    assert (
        book_first["same_timestamp_policy"]
        == "local_observation_order_then_delayed_commands"
    )


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
        capital=PortfolioCapital("synthetic", 1000.0, "USDT", "linear_perpetual"),
        asset_id="btc_usdt",
        funding_events=(),
    )
    assert result["filled_orders"] == 0


def test_zero_visible_trade_through_does_not_advance_queue() -> None:
    result = simulate_maker(
        [_state(0)],
        [
            _trade(50, quantity=1.0, visible_quantity=0.0, price_index=994),
            _trade(100, quantity=1.0, visible_quantity=1.0, price_index=995),
        ],
        [_quote(0)],
        config=_maker_config(),
        observed_coverage={"book-1": _coverage(0, 200)},
        trade_coverage={"trade-connection-1": _coverage(0, 200)},
        capital=PortfolioCapital("synthetic", 1000.0, "USDT", "linear_perpetual"),
        asset_id="btc_usdt",
        funding_events=(),
    )
    assert result["filled_orders"] == 0


def test_trade_through_consumes_full_remaining_order_not_visible_quantity() -> None:
    fills: list[dict[str, object]] = []
    result = simulate_maker(
        [_state(0)],
        [_trade(50, quantity=1.0, visible_quantity=0.1, price_index=994)],
        [_quote(0)],
        config=_maker_config(),
        observed_coverage={"book-1": (0, 200000000)},
        trade_coverage={"trade-connection-1": (0, 200000000)},
        on_ledger=capture_fill_rows(fills.append),
        capital=PortfolioCapital("synthetic", 1000.0, "USDT", "linear_perpetual"),
        asset_id="btc_usdt",
        funding_events=(),
    )
    assert result["fill_quantity"] == 1.0
    assert fills[0]["fill_qty"] == 1.0
    assert result["max_abs_position"] == 1.0


def test_maker_order_is_censored_at_first_missing_trade_coverage_boundary() -> None:
    fills: list[dict[str, object]] = []
    result = simulate_maker(
        [_state(0)],
        [
            _trade(50, quantity=1.0, visible_quantity=1.0, price_index=995),
            _trade(100, quantity=1.0, visible_quantity=1.0, price_index=994),
        ],
        [_quote(0)],
        config=_maker_config(),
        observed_coverage={"book-1": (0, 200000000)},
        trade_coverage={"trade-connection-1": (0, 75000000)},
        on_ledger=capture_fill_rows(fills.append),
        capital=PortfolioCapital("synthetic", 1000.0, "USDT", "linear_perpetual"),
        asset_id="btc_usdt",
        funding_events=(),
    )
    assert fills == []
    assert result["average_order_lifetime_ms"] is None
    assert result["created_orders"] == 2
    assert result["account_path_status"] == "PARTIAL"
    assert result["account_stop_ns"] == 75_000_000
    assert result["censored_active_order_count"] == 2
    assert result["zero_fill_orders"] == result["cancelled_orders"] == 0
    assert result["net_pnl"] is None


def test_maker_terminal_order_is_unknown_at_coverage_validity_end() -> None:
    result = simulate_maker(
        [_state(0)],
        [],
        [_quote(0)],
        config=_maker_config(),
        observed_coverage={"book-1": _coverage(0, 200)},
        trade_coverage={"trade-connection-1": _coverage(0, 75)},
        capital=PortfolioCapital("synthetic", 1000.0, "USDT", "linear_perpetual"),
        asset_id="btc_usdt",
        funding_events=(),
    )
    assert result["average_order_lifetime_ms"] is None
    assert result["account_path_status"] == "PARTIAL"
    assert result["account_stop_ns"] == 75_000_000
    assert result["censored_active_order_count"] == 2


def test_maker_independent_lineage_retains_actual_source_lifetime() -> None:
    result = simulate_maker(
        [_state(0)],
        [],
        [
            _quote(
                0, book_segment_id="book-2", trade_observation_id="trade-connection-2"
            )
        ],
        config=_maker_config(),
        observed_coverage={"book-1": _coverage(0, 200)},
        trade_coverage={"trade-connection-1": _coverage(0, 200)},
        capital=PortfolioCapital("synthetic", 1000.0, "USDT", "linear_perpetual"),
        asset_id="btc_usdt",
        funding_events=(),
    )
    assert result["created_orders"] == 2
    assert result["account_path_status"] == "PARTIAL"
    assert result["account_stop_ns"] == 200_000_000
    assert result["censored_active_order_count"] == 2


def test_book_state_taker_timing_and_terminal_mark() -> None:
    fills: list[dict[str, object]] = []
    equities: list[dict[str, object]] = []
    simulate_taker(
        [
            _state(0),
            _state(5, event_time_ms=1, transaction_time_ms=1),
            _state(20, bid_index=1005, ask_index=1015),
        ],
        [{"timestamp_ms": 0, "prediction": 0.1}],
        funding_forecasts=KNOWN_NO_FUNDING,
        prediction_horizon_ms=500,
        prediction_column="prediction",
        config=_taker_config(latency_ms=10),
        observed_coverage={"book-1": _coverage(0, 20)},
        on_ledger=capture_fill_rows(fills.append, equities.append),
        capital=PortfolioCapital("synthetic", 1000.0, "USDT", "linear_perpetual"),
        asset_id="btc_usdt",
        funding_events=(),
    )
    assert fills[0]["decision_time_ms"] == 0
    assert fills[0]["effective_time_ms"] == 10
    assert fills[0]["state_observed_time_ms"] == 5
    assert fills[0]["timestamp_ms"] == 10
    assert equities[0].valuations[0].timestamp_ns == 0
    terminal = simulate_taker(
        [_state(0), _state(100, bid_index=1005, ask_index=1015)],
        [{"timestamp_ms": 0, "prediction": 0.1}],
        funding_forecasts=KNOWN_NO_FUNDING,
        prediction_horizon_ms=500,
        prediction_column="prediction",
        config=_taker_config(),
        observed_coverage={"book-1": _coverage(0, 101)},
        capital=PortfolioCapital("synthetic", 1000.0, "USDT", "linear_perpetual"),
        asset_id="btc_usdt",
        funding_events=(),
    )
    assert math.isclose(terminal["gross_pnl"], 0.5)


def test_same_price_book_decrease_without_trade_does_not_deplete_queue() -> None:
    result = simulate_maker(
        [_state(0, bid_qty=1.0), _state(50, bid_qty=0.5)],
        [_trade(100, quantity=1.0, visible_quantity=1.0)],
        [_quote(0)],
        config=_maker_config(),
        observed_coverage={"book-1": _coverage(0, 200)},
        trade_coverage={"trade-connection-1": _coverage(0, 200)},
        capital=PortfolioCapital("synthetic", 1000.0, "USDT", "linear_perpetual"),
        asset_id="btc_usdt",
        funding_events=(),
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
            capital=PortfolioCapital("synthetic", 1000.0, "USDT", "linear_perpetual"),
            asset_id="btc_usdt",
            funding_events=(),
        )


def test_hft_grid_is_the_only_model_and_label_step_owner() -> None:
    assert HFT_GRID_MS == 100
    config = research_configuration("crypto", "baseline")
    models = model_specs(config["models"])
    assert all(not hasattr(model, "purge_gap_steps") for model in models)
    rows = [
        {"timestamp_ms": index * HFT_GRID_MS, "segment_id": "book-1", "mid": 100.0}
        for index in range(2)
    ]
    labels = pd.concat(
        iter_label_frames((pd.DataFrame(rows),), horizons_ms=(HFT_GRID_MS,)),
        ignore_index=True,
    ).to_dict("records")
    assert labels[0][f"future_mid_return_{HFT_GRID_MS}ms"] == 0.0


def test_source_order_keeps_real_monotonic_ties_and_archive_line_order():
    from dataclasses import replace

    from quant_research_platform.contracts.order_book import observation_order_key

    earlier = _state(100, source_sequence=9, receive_monotonic_ns=1)
    later = _state(100, source_sequence=2, receive_monotonic_ns=2)
    assert observation_order_key(earlier) < observation_order_key(later)
    archive_first = replace(later, receive_monotonic_ns=None)
    archive_last = replace(earlier, receive_monotonic_ns=None)
    assert observation_order_key(archive_first) < observation_order_key(archive_last)
    assert archive_first.receive_monotonic_ns is None


@pytest.mark.parametrize(
    ("value", "places"),
    [
        ("100.100000", 1),
        ("1e3", 0),
        ("1e-9", 9),
        ("1000000000000000000000000000.000000001", 9),
        ("123.123456789012345678901234567890000", 29),
    ],
)
def test_price_unit_precision_never_rounds_through_decimal_context(value, places):
    from quant_research_platform.contracts.order_book import price_decimal_places

    assert price_decimal_places(value) == places


def test_price_coordinates_preserve_exact_integers_before_storage_admission():
    from quant_research_platform.contracts.order_book import price_to_index

    assert price_to_index(str(2**63 - 1), "1", error="price_invalid") == 2**63 - 1
    assert price_to_index(str(2**63), "1", error="price_invalid") == 2**63


@pytest.mark.parametrize("latency", [0, 100])
@pytest.mark.parametrize("trade_arrives_first", [False, True])
def test_maker_observed_trade_gap_preserves_unknown_active_and_pending_risk(
    latency, trade_arrives_first
):
    from dataclasses import replace

    result = simulate_maker(
        [_state(0), _state(80)],
        [_trade(80, quantity=5.0, price_index=994, observation_id="trade-connection-2")]
        if trade_arrives_first
        else [],
        [_quote(0), _quote(80, trade_observation_id="trade-connection-2")],
        config=replace(_maker_config(), latency_ms=latency),
        observed_coverage={"book-1": (0, 200000000)},
        trade_coverage={
            "trade-connection-1": (0, 75000000),
            "trade-connection-2": (75000000, 200000000),
        },
        capital=PortfolioCapital("synthetic", 1000.0, "USDT", "linear_perpetual"),
        asset_id="btc_usdt",
        funding_events=(),
    )
    assert result["fill_quantity"] == 0
    assert result["created_orders"] == 2
    assert result["account_path_status"] == "PARTIAL"
    assert result["account_stop_ns"] == 75_000_000
    assert result["censored_active_order_count"] == 2
    assert result["censored_pending_command_count"] == (2 if latency else 0)
    assert result["cancelled_orders"] == result["zero_fill_orders"] == 0
    assert result["net_pnl"] is None


@pytest.mark.parametrize("kind", ["maker", "taker"])
@pytest.mark.parametrize("stop_ms,expected", [(200, True), (100, False)])
def test_quiet_tail_matures_markout_at_reached_clock_and_persists(
    tmp_path, kind, stop_ms, expected
):
    from dataclasses import replace

    from quant_research_platform.artifacts.portfolio_ledger import (
        read_portfolio_ledger_artifact,
        write_portfolio_ledger_artifact,
    )

    capital = PortfolioCapital("quiet_tail", 1000.0, "USDT", "linear_perpetual")
    comparisons = []
    for has_later_book in (False, True):
        books = [_state(0)]
        if has_later_book:
            books.append(_state(175))
        fills = []
        results = []
        metadata = {
            "window": {
                "start": "1970-01-01T00:00:00",
                "end": "1970-01-01T00:00:00.180",
            },
            "sample_role": "test",
        }

        def produce(emit, books=books, fills=fills, results=results):
            common = dict(
                observed_coverage={"book-1": _coverage(0, stop_ms)},
                capital=capital,
                asset_id="btc_usdt",
                funding_events=(),
                funding_forecasts=KNOWN_NO_FUNDING,
                decision_windows_ns=((0, 180000000),),
                observation_window_ns=(0, stop_ms * 1000000),
                on_ledger=capture_fill_rows(
                    lambda fill: fills.append(dict(fill)), emit
                ),
            )
            if kind == "taker":
                result = simulate_taker(
                    books,
                    [{"timestamp_ms": 0, "prediction": 0.1}],
                    prediction_column="prediction",
                    prediction_horizon_ms=500,
                    config=_taker_config(),
                    **common,
                )
            else:
                result = simulate_maker(
                    books,
                    [_trade(10, quantity=3)],
                    [_quote(0)],
                    config=replace(_maker_config(), markout_horizon_ms=90),
                    trade_coverage={"trade-connection-1": _coverage(0, stop_ms)},
                    **common,
                )
            results.append(result)

        root = write_portfolio_ledger_artifact(
            tmp_path / str(has_later_book),
            capital=capital,
            metadata=metadata,
            produce=produce,
        )
        artifact = read_portfolio_ledger_artifact(root)
        trades, cashflows, markouts = ([], [], [])
        for batch in artifact.iter_chunks():
            trades.extend(batch.trades)
            cashflows.extend(batch.cashflows)
            markouts.extend(
                row.number
                for row in batch.diagnostics
                if row.name == "markout_per_unit"
            )
        assert len(trades) == 1 and len(markouts) == 1
        assert (markouts[0] is not None) is expected
        assert fills[0]["markout_per_unit"] == markouts[0]
        comparisons.append((trades, cashflows, results[0]["net_pnl"], markouts))
    left_trades, left_flows, left_pnl, left_markouts = comparisons[0]
    right_trades, right_flows, right_pnl, right_markouts = comparisons[1]
    for left, right in zip(left_trades, right_trades, strict=True):
        assert replace(left, notional=right.notional) == right
        assert left.notional == pytest.approx(right.notional, rel=2e-15, abs=0)
    for left, right in zip(left_flows, right_flows, strict=True):
        assert replace(left, amount=right.amount) == right
        assert left.amount == pytest.approx(right.amount, rel=2e-15, abs=0)
    assert left_pnl == pytest.approx(right_pnl, rel=2e-15, abs=1e-13)
    assert [value is None for value in left_markouts] == [
        value is None for value in right_markouts
    ]
    for left, right in zip(left_markouts, right_markouts, strict=True):
        if left is not None:
            assert left == pytest.approx(right, rel=2e-15, abs=1e-13)


@pytest.mark.parametrize(
    "quantity,visible,expected_fill",
    [(3.0, 2.0, 0.0), (3.0, 2.25, 0.25), (3.0, 3.0, 1.0)],
)
def test_maker_visible_quantity_controls_persisted_fill_and_nav(
    tmp_path, quantity, visible, expected_fill
):
    from quant_research_platform.artifacts.portfolio_ledger import (
        read_portfolio_ledger_artifact,
        write_portfolio_ledger_artifact,
    )

    capital = PortfolioCapital("visible_quantity", 1000.0, "USDT", "linear_perpetual")
    metadata = {
        "window": {"start": "1970-01-01T00:00:00", "end": "1970-01-01T00:00:00.200"},
        "sample_role": "test",
    }
    result = []

    def produce(emit):
        result.append(
            simulate_maker(
                [_state(0), _state(200)],
                [_trade(50, quantity=quantity, visible_quantity=visible)],
                [_quote(0)],
                config=_maker_config(),
                observed_coverage={"book-1": _coverage(0, 300)},
                trade_coverage={"trade-connection-1": _coverage(0, 300)},
                capital=capital,
                asset_id="btc_usdt",
                decision_windows_ns=((0, 200_000_000),),
                funding_events=(),
                on_ledger=emit,
            )
        )

    root = write_portfolio_ledger_artifact(
        tmp_path / "ledger", capital=capital, metadata=metadata, produce=produce
    )
    artifact = read_portfolio_ledger_artifact(root)
    trades = []
    persisted_nav = None
    for batch in artifact.iter_chunks():
        trades.extend(batch.trades)
        endings = [row for row in batch.valuations if row.phase == "end"]
        if endings:
            persisted_nav = sum(row.value for row in endings)
    assert sum(trade.quantity for trade in trades) == expected_fill
    assert result[0]["net_pnl"] == pytest.approx(0.5 * expected_fill)
    assert result[0]["nav"] == pytest.approx(1000.0 + 0.5 * expected_fill)
    assert persisted_nav == pytest.approx(1000.0 + 0.5 * expected_fill)
