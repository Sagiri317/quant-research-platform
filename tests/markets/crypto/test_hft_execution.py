from __future__ import annotations

import math
from dataclasses import replace

import pandas as pd
import pytest

from quant_research_platform.contracts.order_book import (
    BookBoundary,
    BookState,
    TradeEvent,
    price_to_index,
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
from quant_research_platform.sources.binance.ids import FUTURES_MARKET_TYPE
from tests.helpers.binance_hft import KNOWN_NO_FUNDING, _coverage, _quote
from tests.helpers.portfolio_ledger import capture_fill_rows


@pytest.mark.usefixtures("coordinate_scale")
def test_zero_latency_maker_replacement_releases_old_reservation() -> None:
    from tests.markets.crypto.calibration_test_support import calibration_from_paths

    fills = []
    calibration = calibration_from_paths(
        [
            (
                [_state(0)],
                [
                    _trade(400, price="99.5", quantity="3", maker_buyer=True),
                    _trade(401, price="100.5", quantity="2", maker_buyer=False),
                ],
            )
        ],
        [0.0],
    )
    result = _simulate_maker(
        [_state(stamp) for stamp in (0, 100, 200, 300, 400, 500, 600, 700)],
        [
            _trade(stamp, price="99.5", quantity=quantity, maker_buyer=True)
            for stamp, quantity in ((100, "1"), (350, "2"), (650, "1"))
        ],
        [
            {
                **_quote(stamp, book_segment_id="s1"),
                "fold_id": "fold",
                "price_prediction": 0.0,
                "proxy_prediction": 1.0,
            }
            for stamp in (0, 200)
        ],
        config=replace(
            _maker_config(), max_position=1.0, maker_fee_bps=0.0, latency_ms=0
        ),
        observed_coverage={"s1": (0, 1000000000)},
        trade_coverage={"trade-connection-1": (0, 1000000000)},
        capital=PortfolioCapital("fixture", 1000.0, "USDT", "linear_perpetual"),
        asset_id="btc_usdt",
        funding_events=(),
        funding_forecasts=(FundingForecast(0, 1000000000, ()),),
        calibrations={"fold": calibration},
        on_ledger=capture_fill_rows(fills.append),
    )
    assert result["created_orders"] == 4
    assert [fill["timestamp_ms"] for fill in fills] == [650]
    assert "inventory_target" not in result and "inventory_skew" not in result


def test_fixed_maker_inventory_controls_change_real_fill_quantity():
    quantities = []
    for target, skew in ((0.0, 1.0), (-1.0, 1.0), (-1.0, 0.0)):
        fills = []
        result = simulate_maker(
            [_state(0), _state(200)],
            [_trade(50, price="99.5", quantity="4", maker_buyer=True)],
            [_quote(0, book_segment_id="s1")],
            config=replace(
                _maker_config(), inventory_target=target, inventory_skew=skew
            ),
            observed_coverage={"s1": _coverage(0, 201)},
            trade_coverage={"trade-connection-1": _coverage(0, 201)},
            on_ledger=capture_fill_rows(fills.append),
            capital=PortfolioCapital("synthetic", 1000.0, "USDT", "linear_perpetual"),
            asset_id="btc_usdt",
            funding_events=(),
        )
        assert result["inventory_target"] == target
        assert result["inventory_skew"] == skew
        quantities.append(sum(fill["fill_qty"] for fill in fills))
    assert quantities == [1.0, 0.5, 1.0]


def _state(
    timestamp_ms: int,
    *,
    bid_index: int = 995,
    ask_index: int = 1005,
    bid_levels: tuple[tuple[int, float], ...] | None = None,
    ask_levels: tuple[tuple[int, float], ...] | None = None,
    segment_id: str = "s1",
) -> BookState:
    return BookState(
        source_sequence=timestamp_ms + 1,
        event_time_ms=timestamp_ms + 1,
        transaction_time_ms=timestamp_ms,
        observed_time_ns=timestamp_ms * 1000000,
        receive_time="2026-08-27T00:00:00.000000Z",
        receive_monotonic_ns=timestamp_ms + 1,
        update_id=timestamp_ms + 1,
        segment_id=segment_id,
        observation_id=f"connection-{segment_id}",
        market_type=FUTURES_MARKET_TYPE,
        price_unit="0.1",
        bids=bid_levels or ((bid_index, 2.0), (bid_index - 5, 3.0)),
        asks=ask_levels or ((ask_index, 1.0), (ask_index + 5, 2.0)),
    )


def _boundary(timestamp_ms: int, *, segment_id: str = "s1") -> BookBoundary:
    return BookBoundary(
        source_sequence=timestamp_ms + 1,
        event_time_ms=timestamp_ms + 1,
        transaction_time_ms=timestamp_ms,
        observed_time_ns=timestamp_ms * 1000000,
        receive_time="2026-08-27T00:00:00.000000Z",
        receive_monotonic_ns=timestamp_ms + 1,
        segment_id=segment_id,
        observation_id=f"connection-{segment_id}",
        market_type=FUTURES_MARKET_TYPE,
        reason="sequence_gap",
    )


def _trade(
    timestamp_ms: int, *, price: str, quantity: str, maker_buyer: bool
) -> TradeEvent:
    quantity_value = float(quantity)
    return TradeEvent(
        source_sequence=timestamp_ms + 10,
        event_time_ms=timestamp_ms,
        transaction_time_ms=timestamp_ms,
        observed_time_ns=timestamp_ms * 1000000,
        receive_time="2026-08-27T00:00:00.000000Z",
        receive_monotonic_ns=timestamp_ms + 10,
        observation_id="trade-connection-1",
        market_type=FUTURES_MARKET_TYPE,
        price_unit="0.1",
        price_index=price_to_index(price, "0.1", error="bad"),
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


simulate_taker = _simulate_taker
simulate_maker = _simulate_maker


@pytest.mark.parametrize(
    "wrong_identity", (None, "book_segment_id", "trade_observation_id")
)
def test_common_panel_maker_quotes_use_raw_execution_identities(wrong_identity) -> None:
    from quant_research_platform.markets.implementations.crypto.portfolio import (
        _paired_validation,
    )
    from quant_research_platform.markets.implementations.crypto.processed import (
        producer,
    )

    frames = {
        symbol: pd.DataFrame(
            [
                {
                    "timestamp": pd.Timestamp(0),
                    "asset_id": asset,
                    "segment_id": "s1:trade-connection-1",
                    "book_segment_id": "s1",
                    "trade_observation_id": "trade-connection-1",
                    "prediction_score": 1.0,
                    "label": 0.0,
                    "label_available_at": pd.Timestamp(100000000),
                }
            ]
        )
        for symbol, asset in (("BTCUSDT", "btc_usdt"), ("ETHUSDT", "eth_usdt"))
    }
    panel = pd.concat(
        producer._common_symbol_panel_frames(
            {symbol: iter((frame,)) for symbol, frame in frames.items()},
            columns=tuple(frames["BTCUSDT"].columns),
            batch_rows=2,
        ),
        ignore_index=True,
    )
    assert panel["segment_id"].tolist() == ["s1:trade-connection-1:common@0"] * 2
    if wrong_identity is not None:
        panel[wrong_identity] = "wrong-raw-identity"

    class ValidationArtifact:
        models = tuple(
            {
                "fold_id": "fold",
                "model_id": model_id,
                "model_metadata": {"model_algorithm": "linear_regression"},
            }
            for model_id in ("price_model", "pressure_model")
        )

        def validation_chunks(self, *, fold_id, model_id):
            yield panel

    quotes = list(
        _paired_validation(
            ValidationArtifact(), "fold", "btc_usdt", ("price_model", "pressure_model")
        )
    )
    assert len(quotes) == 1
    assert "segment_id" not in quotes[0]
    fills = []
    result = simulate_maker(
        [_state(0), _state(200)],
        [_trade(50, price="99.4", quantity="0.1", maker_buyer=True)],
        quotes,
        config=_maker_config(),
        observed_coverage={"s1": _coverage(0, 201)},
        trade_coverage={"trade-connection-1": _coverage(0, 201)},
        on_ledger=capture_fill_rows(fills.append),
        capital=PortfolioCapital("synthetic", 1000.0, "USDT", "linear_perpetual"),
        asset_id="btc_usdt",
        funding_events=(),
    )
    assert len(fills) == (1 if wrong_identity is None else 0)
    assert result["created_orders"] == (2 if wrong_identity is None else 0)


@pytest.mark.usefixtures("coordinate_scale")
def test_taker_uses_latest_actual_state_at_effective_time() -> None:
    fills = []
    diagnostics = simulate_taker(
        [_state(0), _state(5), _state(15, bid_index=1005, ask_index=1015)],
        [{"timestamp_ms": 0, "prediction": 0.02, "segment_id": "s1"}],
        funding_forecasts=KNOWN_NO_FUNDING,
        prediction_horizon_ms=500,
        prediction_column="prediction",
        config=_taker_config(latency_ms=10),
        observed_coverage={"s1": _coverage(0, 15)},
        on_ledger=capture_fill_rows(fills.append),
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


def test_failing_fill_observer_is_not_replayed_during_cleanup() -> None:
    calls: list[int] = []

    def fail(fill):
        calls.append(fill["order_id"])
        raise KeyboardInterrupt("observer_primary")

    with pytest.raises(KeyboardInterrupt, match="observer_primary"):
        _simulate_taker(
            [_state(0), _state(200)],
            [{"timestamp_ms": 0, "prediction": 0.02}],
            funding_forecasts=KNOWN_NO_FUNDING,
            prediction_horizon_ms=500,
            prediction_column="prediction",
            config=_taker_config(),
            observed_coverage={"s1": _coverage(0, 300)},
            on_ledger=capture_fill_rows(fail),
            capital=PortfolioCapital("synthetic", 1000.0, "USDT", "linear_perpetual"),
            asset_id="btc_usdt",
            funding_events=(),
        )
    assert calls == [1]


@pytest.mark.usefixtures("coordinate_scale")
def test_taker_walks_known_l2_and_reports_partial_coverage_honestly() -> None:
    fills = []
    diagnostics = simulate_taker(
        [_state(0), _state(5, ask_levels=((1005, 0.4), (1010, 0.6)))],
        [{"timestamp_ms": 0, "prediction": 0.02}],
        funding_forecasts=KNOWN_NO_FUNDING,
        prediction_horizon_ms=500,
        prediction_column="prediction",
        config=_taker_config(latency_ms=10, order_quantity=1.5),
        observed_coverage={"s1": _coverage(0, 11)},
        on_ledger=capture_fill_rows(fills.append),
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
        on_ledger=capture_fill_rows(fills.append),
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
        on_ledger=capture_fill_rows(fills.append),
        capital=PortfolioCapital("synthetic", 1000.0, "USDT", "linear_perpetual"),
        asset_id="btc_usdt",
        funding_events=(),
    )
    assert len(fills) == 1
    assert diagnostics["gross_pnl"] is None
    assert diagnostics["net_pnl"] is None


def test_taker_markout_does_not_use_state_past_observed_coverage() -> None:
    diagnostics = simulate_taker(
        [_state(0), _state(200, bid_index=1005, ask_index=1015)],
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


@pytest.mark.parametrize(
    "at_target, tail", [(False, True), (True, True), (True, False)]
)
def test_taker_markout_uses_latest_state_as_of_target(at_target, tail) -> None:
    fills: list[dict[str, object]] = []
    stamp = 100 if at_target else 50
    books = [
        _state(0, bid_index=1000, ask_index=1005),
        _state(stamp, bid_index=995, ask_index=1005),
        replace(
            _state(stamp, bid_index=1005, ask_index=1015),
            source_sequence=stamp + 2,
            update_id=stamp + 2,
        ),
    ]
    if tail:
        books.append(_state(200, bid_index=1095, ask_index=1105))
    diagnostics = simulate_taker(
        books,
        [{"timestamp_ms": 0, "prediction": 0.1}],
        funding_forecasts=KNOWN_NO_FUNDING,
        prediction_horizon_ms=500,
        prediction_column="prediction",
        config=_taker_config(),
        observed_coverage={"s1": _coverage(0, 300)},
        on_ledger=capture_fill_rows(fills.append),
        capital=PortfolioCapital("synthetic", 1000.0, "USDT", "linear_perpetual"),
        asset_id="btc_usdt",
        funding_events=(),
    )
    assert diagnostics["markout_undefined_fill_events"] == 0
    assert fills[0]["markout_per_unit"] == pytest.approx(0.5)


@pytest.mark.usefixtures("coordinate_scale")
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


@pytest.mark.usefixtures("coordinate_scale")
def test_maker_same_price_depletion_partial_fill_and_order_lifetime() -> None:
    fills = []
    diagnostics = simulate_maker(
        [_state(0), _state(200, bid_index=1000, ask_index=1010)],
        [
            _trade(50, price="99.5", quantity="2.5", maker_buyer=True),
            _trade(100, price="99.5", quantity="0.5", maker_buyer=True),
        ],
        [_quote(0, book_segment_id="s1")],
        config=_maker_config(),
        observed_coverage={"s1": _coverage(0, 201)},
        trade_coverage={"trade-connection-1": _coverage(0, 201)},
        on_ledger=capture_fill_rows(fills.append),
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


@pytest.mark.usefixtures("coordinate_scale")
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
        on_ledger=capture_fill_rows(fills.append),
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
        on_ledger=capture_fill_rows(fills.append),
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
        on_ledger=capture_fill_rows(fills.append),
        capital=PortfolioCapital("synthetic", 1000.0, "USDT", "linear_perpetual"),
        asset_id="btc_usdt",
        funding_events=(),
    )
    assert len(fills) == 2
    assert diagnostics["gross_pnl"] == pytest.approx(1.0)
    assert diagnostics["net_pnl"] == pytest.approx(1.02)


def test_cost_sensitivity_reprices_actual_turnover_and_net_pnl() -> None:
    result = fixed_turnover_cost_sensitivity(
        turnover=10000.0,
        net_pnl=5.0,
        base_fee_bps=1.0,
        fee_scenarios_bps=(1.0, 2.0),
        additional_spread_scenarios_bps=(0.0, 1.0),
    )
    assert [row["net_pnl"] for row in result] == pytest.approx([5.0, 4.0, 4.0, 3.0])
    assert {row["method"] for row in result} == {
        "constant_actual_turnover_additive_sensitivity"
    }


def test_taker_long_stream_returns_counts_without_dataframe_results() -> None:

    def decisions():
        for index in range(20000):
            yield {"timestamp_ms": index, "prediction": 0.0}

    diagnostics = simulate_taker(
        [_state(0), _state(19999)],
        decisions(),
        funding_forecasts=KNOWN_NO_FUNDING,
        prediction_horizon_ms=500,
        prediction_column="prediction",
        config=_taker_config(),
        observed_coverage={"s1": _coverage(0, 19999)},
        capital=PortfolioCapital("synthetic", 1000.0, "USDT", "linear_perpetual"),
        asset_id="btc_usdt",
        funding_events=(),
    )
    assert diagnostics["fill_events"] == 0
    assert diagnostics["equity_rows"] == 20000
    assert not any("dataframe" in key.lower() for key in diagnostics)


def test_taker_freezes_orders_across_overlapping_latency_and_future_book_changes():
    from quant_research_platform.contracts.portfolios import FundingEvent

    selections = []
    fills_by_case = []
    for future_quantity in (2.0, 0.25):
        ledger = []
        fills = []
        result = _simulate_taker(
            [_state(0), _state(25, ask_levels=((1100, future_quantity),)), _state(80)],
            [{"timestamp_ms": stamp, "prediction": 0.02} for stamp in (0, 10, 20)],
            funding_forecasts=KNOWN_NO_FUNDING,
            prediction_column="prediction",
            prediction_horizon_ms=100,
            config=_taker_config(latency_ms=30),
            observed_coverage={"s1": _coverage(0, 100)},
            capital=PortfolioCapital("fixture", 1000.0, "USDT", "linear_perpetual"),
            asset_id="btc_usdt",
            funding_events=(),
            on_ledger=capture_fill_rows(fills.append, ledger.append),
        )
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
        funding_events=(FundingEvent(25000000, None, None),),
    )
    assert result["selection_status"] == "COMPLETE"
    assert result["created_orders"] == 1 and result["funding"] == 0.0
    for rate in (0.001, -0.001):
        ledger = []
        result = simulate_taker(
            [_state(0), _state(50), _state(100)],
            [
                {"timestamp_ms": 0, "prediction": 0.02},
                {"timestamp_ms": 50, "prediction": 0.02},
            ],
            funding_forecasts=(FundingForecast(0, 1000000000, ((75000000, None),)),),
            prediction_column="prediction",
            prediction_horizon_ms=50,
            config=_taker_config(),
            observed_coverage={"s1": _coverage(0, 101)},
            capital=PortfolioCapital("fixture", 1000.0, "USDT", "linear_perpetual"),
            asset_id="btc_usdt",
            funding_events=(FundingEvent(75000000, rate, 100.0),),
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
            on_ledger=capture_fill_rows(fills.append, equity.append),
        )
        decisions = [
            {row.name: row for row in batch.diagnostics}
            for batch in equity
            if any(row.name == "decision_reason" for row in batch.diagnostics)
        ]
        if reason == "decision_book_unavailable":
            assert not decisions
        else:
            assert decisions[0]["decision_requested_quantity"].number == quantity
            assert decisions[0]["decision_reason"].text == reason
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


def test_taker_rejects_nonpositive_forecast_and_preserves_primary():
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

    def books():
        try:
            for stamp in (0, 100):
                yield _state(stamp)
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
            capital=PortfolioCapital("fixture", 1000.0, "USDT", "linear_perpetual"),
            asset_id="btc_usdt",
            funding_events=(),
            on_ledger=emit,
        )
    assert raised.value is primary and closed


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


@pytest.fixture(params=(1, 10, 100))
def coordinate_scale(request, monkeypatch):
    import sys
    from dataclasses import replace

    multiplier = request.param
    unit = {1: "0.1", 10: "0.01", 100: "0.001"}[multiplier]
    original_state, original_trade = (_state, _trade)

    def state(*args, **kwargs):
        value = original_state(*args, **kwargs)
        return replace(
            value,
            price_unit=unit,
            bids=tuple(((index * multiplier, qty) for index, qty in value.bids)),
            asks=tuple(((index * multiplier, qty) for index, qty in value.asks)),
        )

    def trade(*args, **kwargs):
        value = original_trade(*args, **kwargs)
        return replace(
            value, price_unit=unit, price_index=value.price_index * multiplier
        )

    monkeypatch.setattr(sys.modules[__name__], "_state", state)
    monkeypatch.setattr(sys.modules[__name__], "_trade", trade)


def test_observation_stop_does_not_rewrite_maker_decision_or_cancel_pending_quote():
    results = []
    for stop in (400000000, 1000000000):
        batches, fills = ([], [])
        result = _simulate_maker(
            [_state(stamp) for stamp in (0, 350, 390, 500, 900)],
            [],
            [_quote(350, book_segment_id="s1")],
            config=replace(_maker_config(), latency_ms=100),
            observed_coverage={"s1": (0, 2000000001)},
            trade_coverage={"trade-connection-1": (0, 2000000001)},
            capital=PortfolioCapital("fixture", 1000.0, "USDT", "linear_perpetual"),
            asset_id="btc_usdt",
            funding_events=(),
            decision_windows_ns=((0, 2000000000),),
            observation_window_ns=(0, stop),
            on_ledger=capture_fill_rows(fills.append, batches.append),
        )
        decisions = [
            batch
            for batch in batches
            if any(row.name.startswith("maker_") for row in batch.diagnostics)
        ]
        results.append((result, batches, decisions, fills))
    assert results[0][2] == results[1][2]
    assert results[0][0]["cancelled_orders"] == 0
    assert results[0][0]["censored_pending_command_count"] == 2
    assert results[0][0]["average_order_lifetime_ms"] is None
    assert not results[0][3]
    assert (
        max(row.timestamp_ns for batch in results[0][1] for row in batch.valuations)
        == 390000000
    )
    prefixes = [
        [
            batch
            for batch in rows
            if max(value.timestamp_ns for value in batch.valuations) <= 350000000
        ]
        for _result, rows, _equity, _fills in results
    ]
    assert len(prefixes[0]) >= 2
    assert prefixes[0] == prefixes[1]


@pytest.mark.parametrize("kind", ["maker", "taker"])
def test_complete_observation_window_preserves_continuous_economic_batches(kind):
    produced = []
    for observed in (None, (0, 1000000001)):
        batches = []
        common = dict(
            observed_coverage={"s1": (0, 1000000001)},
            capital=PortfolioCapital("fixture", 1000.0, "USDT", "linear_perpetual"),
            asset_id="btc_usdt",
            funding_events=(),
            decision_windows_ns=((0, 1000000000),),
            observation_window_ns=observed,
            on_ledger=batches.append,
        )
        books = [_state(stamp) for stamp in (0, 350, 500, 900, 1000)]
        if kind == "maker":
            _simulate_maker(
                books,
                [],
                [_quote(350, book_segment_id="s1")],
                config=_maker_config(),
                trade_coverage={"trade-connection-1": (0, 1000000001)},
                **common,
            )
        else:
            _simulate_taker(
                books,
                [],
                prediction_column="score",
                prediction_horizon_ms=500,
                config=TakerConfig(
                    order_quantity=1.0,
                    position_size=1.0,
                    latency_ms=100,
                    taker_fee_bps=0.0,
                    markout_horizon_ms=500,
                ),
                funding_forecasts=(),
                **common,
            )
        produced.append(batches)
    assert produced[0] == produced[1]


def test_pending_markout_fifo_preserves_maturity_order_with_pending_suffix():
    completed = []
    with _PendingMarkouts(
        _Metrics(), lambda fill: completed.append(fill["ordinal"])
    ) as pending:

        def add(index):
            pending.add(
                {
                    "ordinal": index,
                    "fill_qty": 1.0,
                    "fill_price": 100.0,
                    "side": "buy",
                    "segment_id": "s1",
                    "markout_target_time_ns": index,
                }
            )

        for index in range(10):
            add(index)
        for index in range(20):
            pending.advance(index, None, {})
            add(index + 10)
        assert completed == list(range(20))
        pending.expire_all()
        assert completed == list(range(30))


def test_fee_stress_keeps_representable_extreme_change():
    from fractions import Fraction

    row = fixed_turnover_cost_sensitivity(
        turnover=100.0,
        net_pnl=0.0,
        base_fee_bps=0.0,
        fee_scenarios_bps=(1e308,),
        additional_spread_scenarios_bps=(0.0,),
    )[0]
    expected = float(Fraction(100) * Fraction(1e308) / 10000)
    assert row["cost_delta"] == expected
    assert row["net_pnl"] == -expected
