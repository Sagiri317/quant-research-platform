"""Round-trip research policy: two-sided costs, exact closes and causal clocks."""

from dataclasses import replace
from fractions import Fraction

import pytest

from quant_research_platform.artifacts.portfolio_ledger import (
    read_portfolio_ledger_artifact,
    write_portfolio_ledger_artifact,
)
from quant_research_platform.contracts.portfolios import (
    FundingEvent,
    FundingForecast,
    PortfolioCapital,
)
from quant_research_platform.markets.implementations.crypto.execution import (
    _taker_order_quantity,
    simulate_taker,
)
from tests.helpers.binance_hft import KNOWN_NO_FUNDING, taker_config
from tests.helpers.binance_hft import execution_boundary as boundary
from tests.helpers.binance_hft import execution_state as state
from tests.helpers.portfolio_ledger import capture_fill_rows


def run(books, *, decisions=None, config=None, **kwargs):
    fills, batches = [], []
    capital = PortfolioCapital("round_trip", 1000.0, "USDT", "linear_perpetual")
    result = simulate_taker(
        books,
        [{"timestamp_ms": 0, "prediction": 0.03}] if decisions is None else decisions,
        prediction_column="prediction",
        prediction_horizon_ms=500,
        config=replace(taker_config(latency_ms=100), position_policy="round_trip")
        if config is None
        else config,
        observed_coverage={"s1": (0, 2_000_000_000)},
        capital=capital,
        asset_id="btc_usdt",
        funding_events=kwargs.pop("funding_events", ()),
        funding_forecasts=kwargs.pop("funding_forecasts", KNOWN_NO_FUNDING),
        on_ledger=capture_fill_rows(fills.append, batches.append),
        **kwargs,
    )
    return result, fills, batches, capital


@pytest.mark.parametrize("direction", (1, -1))
@pytest.mark.parametrize("latency", (0, 100, 200))
def test_exit_uses_prediction_deadline_and_real_future_depth_without_another_prediction(
    direction, latency
):
    shifted = direction * 20
    result, fills, batches, _ = run(
        [
            state(0),
            state(500, bid_index=995 + shifted, ask_index=1005 + shifted),
            state(1000),
        ],
        decisions=[
            {"timestamp_ms": 0, "prediction": direction * 0.03},
            {"timestamp_ms": 250, "prediction": -direction * 0.1},
        ],
        config=replace(taker_config(latency_ms=latency), position_policy="round_trip"),
    )
    assert [(f["timestamp_ms"], f["order_role"]) for f in fills] == [
        (latency, "entry"),
        (500, "horizon_exit"),
    ]
    assert [f["fill_price"] for f in fills] == (
        [100.5, 101.5] if direction > 0 else [99.5, 98.5]
    )
    assert result["completed_round_trips"] == 1
    assert result["exit_unfilled_quantity"] == 0.0
    assert result["gross_pnl"] == 1.0
    assert result["fees"] == pytest.approx(0.0202 if direction > 0 else 0.0198)
    assert result["net_pnl"] == pytest.approx(1.0 - result["fees"])
    assert any(
        d.text == "round_trip_position_active" for b in batches for d in b.diagnostics
    )


@pytest.mark.parametrize("prediction", (0.006, -0.006))
def test_one_sided_mark_gain_does_not_pay_for_a_round_trip(prediction):
    config = taker_config()
    args = dict(
        state=state(0),
        prediction=prediction,
        buy_limit=1.0,
        sell_limit=1.0,
        funding_per_unit=0.0,
    )
    assert _taker_order_quantity(**args, config=config)[0] != 0.0
    assert _taker_order_quantity(
        **args, config=replace(config, position_policy="round_trip")
    ) == (0.0, 0.0)


@pytest.mark.parametrize("direction", (1, -1))
def test_exit_depth_breakpoint_and_both_fees_determine_quantity(direction):
    config = replace(taker_config(order_quantity=2.0), position_policy="round_trip")
    book = state(
        0,
        **(
            {"bid_levels": ((995, 0.4), (970, 3.0))}
            if direction > 0
            else {"ask_levels": ((1005, 0.4), (1030, 3.0))}
        ),
    )
    quantity, gain = _taker_order_quantity(
        book,
        prediction=direction * 0.02,
        config=config,
        buy_limit=2.0,
        sell_limit=2.0,
        funding_per_unit=0.0,
    )
    assert quantity == direction * 0.4
    assert gain == pytest.approx(0.4 * (1.0 - (0.0202 if direction > 0 else 0.0198)))


def test_partial_exit_is_disclosed_and_prevents_new_risk():
    result, fills, batches, _ = run(
        [state(0), state(500, bid_levels=((1015, 0.25),), ask_index=1025), state(1000)],
        decisions=[
            {"timestamp_ms": 0, "prediction": 0.03},
            {"timestamp_ms": 600, "prediction": 0.1},
        ],
    )
    assert [f["fill_qty"] for f in fills] == [1.0, 0.25]
    assert result["exit_unfilled_quantity"] == 0.75
    assert result["completed_round_trips"] == 0
    final = next(
        v
        for v in batches[-1].valuations
        if v.asset_id == "btc_usdt" and v.phase == "end"
    )
    assert final.quantity == Fraction(3, 4)


def test_exit_cannot_cross_an_unknown_market_path():
    result, fills, _, _ = run([state(0), boundary(300), state(500)])
    assert [f["order_role"] for f in fills] == ["entry"]
    assert result["account_path_status"] == "PARTIAL"
    assert result["account_stop_reason"] == "portfolio_pending_execution_unknown"
    assert result["completed_round_trips"] == 0
    assert result["net_pnl"] is None


def test_window_boundaries_and_observation_censoring_do_not_invent_exit_fills():
    result, fills, _, _ = run(
        [state(0), state(1000)], decision_windows_ns=((0, 499_000_000),)
    )
    assert result["created_orders"] == 0 and fills == []
    result, fills, _, _ = run(
        [state(0), state(1000)], observation_window_ns=(0, 400_000_000)
    )
    assert len(fills) == 1
    assert result["censored_pending_command_count"] == 1
    assert result["censored_order_quantity"] == 1.0
    assert result["completed_round_trips"] == 0


def test_exit_settlement_follows_funding_at_same_timestamp():
    result, fills, _, _ = run(
        [state(0), state(500, bid_index=1015, ask_index=1025), state(1000)],
        funding_events=(
            FundingEvent(100_000_000, 0.001, 100.0),
            FundingEvent(500_000_000, 0.001, 102.0),
        ),
        funding_forecasts=(
            FundingForecast(
                0, 2_000_000_000, ((100_000_000, -0.1), (500_000_000, -0.102))
            ),
        ),
    )
    assert len(fills) == 2
    assert result["funding"] == pytest.approx(-0.102)
    assert result["net_pnl"] == pytest.approx(1.0 - 0.0202 - 0.102)


@pytest.mark.parametrize("direction", (1, -1))
def test_exact_partial_entry_quantity_is_fully_closed_and_survives_native_persistence(
    tmp_path, direction
):
    entry_side = "ask_levels" if direction > 0 else "bid_levels"
    exit_side = "bid_levels" if direction > 0 else "ask_levels"
    tick = 1005 if direction > 0 else 995
    books = [
        state(0, **{entry_side: ((tick, 1e16 + 2),), exit_side: ((1000, 1e16 + 2),)}),
        state(
            100,
            **{
                entry_side: ((tick, 1e16), (tick + direction, 1.0)),
                exit_side: ((1000, 1e16 + 2),),
            },
        ),
        state(500, **{exit_side: ((1000, 1e16), (1000 - direction, 1.0))}),
    ]
    result, fills, batches, capital = run(
        books,
        decisions=[{"timestamp_ms": 0, "prediction": direction * 0.03}],
        config=replace(
            taker_config(latency_ms=100, order_quantity=1e16 + 2),
            position_policy="round_trip",
            position_size=1e16 + 2,
            taker_fee_bps=0.0,
        ),
    )
    trades = [t for b in batches for t in b.trades]
    assert [t.quantity for t in trades] == [
        direction * Fraction(10**16 + 1),
        -direction * Fraction(10**16 + 1),
    ]
    assert result["completed_round_trips"] == 1
    assert result["exit_unfilled_quantity"] == 0.0
    path = write_portfolio_ledger_artifact(
        tmp_path / "ledger",
        capital=capital,
        metadata={"window": {"start": "1970-01-01", "end": "1970-01-02"}},
        produce=lambda emit: [emit(b) for b in batches],
    )
    assert list(read_portfolio_ledger_artifact(path).iter_chunks()) == batches


def test_round_trip_requires_time_for_entry_then_exit_latency():
    with pytest.raises(ValueError, match="hft_taker_round_trip_horizon_too_short"):
        run(
            [state(0)],
            config=replace(taker_config(latency_ms=250), position_policy="round_trip"),
        )
