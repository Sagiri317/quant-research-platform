from dataclasses import replace

import pytest

from quant_research_platform.contracts.portfolios import (
    FundingForecast,
    PortfolioCapital,
)
from quant_research_platform.markets.implementations.crypto.execution import (
    MakerCalibration,
    _maker_new_order,
    replay_maker_calibration,
    simulate_maker,
)
from tests.helpers.binance_hft import _quote
from tests.markets.crypto.calibration_test_support import (
    calibration_from_paths,
    calibration_inputs,
)
from tests.markets.crypto.test_hft_execution import _maker_config, _state, _trade


def _run_calibrated(paths, errors, *, books, trades, quotes, latency):
    fills, batches = ([], [])
    model = calibration_from_paths(paths, errors)
    result = simulate_maker(
        books,
        trades,
        [
            {
                **_quote(stamp, book_segment_id="s1"),
                "fold_id": "fold",
                "price_prediction": prediction,
                "proxy_prediction": gate,
            }
            for stamp, prediction, gate in quotes
        ],
        config=replace(
            _maker_config(), max_position=3.0, maker_fee_bps=0.0, latency_ms=latency
        ),
        observed_coverage={"s1": (0, 2000000000)},
        trade_coverage={"trade-connection-1": (0, 2000000000)},
        capital=PortfolioCapital("fixture", 1000.0, "USDT", "linear_perpetual"),
        asset_id="btc_usdt",
        funding_events=(),
        funding_forecasts=(FundingForecast(0, 2000000000, ()),),
        calibrations={"fold": model},
        on_ledger=batches.append,
        on_fill=fills.append,
    )
    gains = [
        row.number
        for batch in batches
        for row in batch.diagnostics
        if row.name == "maker_buy_expected_net_gain"
    ]
    return (result, fills, gains, batches)


def test_action_quantity_stays_inside_limit_after_float_fill_accumulation():
    state = _state(0, bid_levels=((995, 1.0), (990, 3.0)))
    trades = [_trade(1, price="99.5", quantity="1", maker_buyer=True)]
    trades.extend(
        (
            _trade(10 * index, price="99.5", quantity=str(quantity), maker_buyer=True)
            for index, quantity in enumerate(
                (0.01, 0.01, 0.05, 0.01, 0.07, 0.06, 0.01, 0.01, 0.07), start=1
            )
        )
    )
    inputs = calibration_inputs([([state], trades)], [0.0])
    new_order = _maker_new_order(
        0,
        "buy",
        0.3,
        state,
        _quote(0, book_segment_id="s1"),
        _maker_config().latency_ms,
    )
    model = MakerCalibration(*inputs, price_unit="0.1")
    keep, quantity, gain = model.select(
        state=state,
        decision_ns=0,
        new_order=new_order,
        active_order=None,
        commands=(),
        lifetime_ns=500000000,
        prediction=0.0,
        fee_bps=0.0,
        funding=[],
    )
    assert not keep and quantity == new_order.remaining
    assert gain == pytest.approx(0.15)


@pytest.mark.parametrize("depletion,kept", [(10, True), (1, False)])
def test_actual_maker_keep_uses_depleted_queue_and_paired_residuals(depletion, kept):

    def book(stamp):
        return _state(stamp, bid_levels=((995, 10.0), (990, 3.0)))

    paths = [
        (
            [book(0)],
            [_trade(100, price="99.5", quantity=str(quantity), maker_buyer=True)],
        )
        for quantity in (5, 11)
    ]
    result, fills, gains, batches = _run_calibrated(
        paths,
        [0.0, -0.02],
        books=[book(stamp) for stamp in (0, 100, 200, 500, 1000)],
        trades=[
            _trade(50, price="99.5", quantity=str(depletion), maker_buyer=True),
            _trade(200, price="99.5", quantity="1", maker_buyer=True),
        ],
        quotes=[(0, 0.03, 1.0), (100, 0.0055, 1.0)],
        latency=0,
    )
    assert result["created_orders"] == 1
    assert [fill["order_id"] for fill in fills] == ([1] if kept else [])
    assert gains == pytest.approx([0.75, 0.05 if kept else 0.0])
    expected_reason = (
        "keep_existing_schedule_without_renewal"
        if kept
        else "cancel_maximizes_expected_net_gain"
    )
    assert any(
        row.text == expected_reason for batch in batches for row in batch.diagnostics
    )


def test_actual_maker_keeps_legal_nonbest_old_price_in_relative_tick_space():

    def book(stamp, moved=False):
        bid = 990 if moved else 995
        return _state(
            stamp,
            bid_index=bid,
            ask_index=bid + 10,
            bid_levels=((bid, 10.0), (bid - 5, 3.0)),
        )

    paths = [
        (
            [book(0)],
            [_trade(100, price="99.5", quantity=str(quantity), maker_buyer=True)],
        )
        for quantity in (5, 11)
    ]
    result, fills, gains, _batches = _run_calibrated(
        paths,
        [0.0, -0.02],
        books=[book(0), book(100, True), book(200, True), book(1000, True)],
        trades=[
            _trade(50, price="99.5", quantity="10", maker_buyer=True),
            _trade(200, price="99.0", quantity="1", maker_buyer=True),
        ],
        quotes=[(0, 0.03, 1.0), (100, 0.015, 1.0)],
        latency=0,
    )
    assert result["created_orders"] == 1
    assert [
        (fill["order_id"], fill["fill_price"], fill["fill_reason"]) for fill in fills
    ] == [(1, 99.5, "trade_through")]
    assert gains == pytest.approx([0.75, 0.4975])


def test_actual_maker_replace_gain_includes_old_fill_before_cancel_latency():

    def book(stamp):
        return _state(stamp, bid_levels=((995, 1.0), (990, 3.0)))

    paths = [
        (
            [book(0)],
            [
                _trade(stamp, price="99.5", quantity=quantity, maker_buyer=True)
                for stamp, quantity in ((50, "1"), (200, "2"))
            ],
        )
    ]
    result, fills, gains, _batches = _run_calibrated(
        paths,
        [0.0],
        books=[book(stamp) for stamp in (0, 100, 200, 300, 400, 1000)],
        trades=[
            _trade(stamp, price="99.5", quantity=quantity, maker_buyer=True)
            for stamp, quantity in ((150, "1"), (250, "1"), (400, "2"))
        ],
        quotes=[(0, 0.03, 1.0), (200, 0.005505, 1.0)],
        latency=100,
    )
    assert result["created_orders"] == 2
    assert [(fill["order_id"], fill["timestamp_ms"]) for fill in fills] == [
        (1, 250),
        (2, 400),
    ]
    assert gains == pytest.approx([3.5, 2.101])


def test_actual_maker_negative_keep_beats_more_negative_delayed_cancel():

    def book(stamp):
        return _state(stamp, bid_levels=((995, 10.0), (990, 3.0)))

    paths = [
        (
            [book(0)],
            [
                _trade(stamp, price="99.5", quantity=quantity, maker_buyer=True)
                for stamp, quantity in ((50, "1"), (400, "11"))
            ],
        ),
        ([book(0)], [_trade(200, price="99.5", quantity="1", maker_buyer=True)]),
    ]
    result, fills, gains, batches = _run_calibrated(
        paths,
        [-0.02, 0.01],
        books=[book(stamp) for stamp in (0, 100, 200, 250, 1000)],
        trades=[
            _trade(150, price="99.5", quantity="10", maker_buyer=True),
            _trade(250, price="99.5", quantity="1", maker_buyer=True),
        ],
        quotes=[(0, 0.03, 1.0), (200, -0.005, 1.0)],
        latency=100,
    )
    assert result["created_orders"] == 1
    assert [fill["order_id"] for fill in fills] == [1]
    assert gains == pytest.approx([0.75, -0.5])
    assert any(
        row.text == "keep_existing_schedule_without_renewal"
        for batch in batches
        for row in batch.diagnostics
    )


def test_actual_maker_keeps_cashflows_of_all_queued_commands():

    def book(stamp):
        return _state(stamp, bid_levels=((995, 10.0), (990, 3.0)))

    paths = [
        (
            [book(0)],
            [
                _trade(stamp, price="99.5", quantity="11", maker_buyer=True)
                for stamp in (120, 300)
            ],
        )
    ]
    result, fills, gains, _batches = _run_calibrated(
        paths,
        [0.0],
        books=[book(stamp) for stamp in (0, 50, 100, 200, 250, 300, 400, 1000)],
        trades=[
            _trade(stamp, price="99.5", quantity="11", maker_buyer=True)
            for stamp in (220, 400)
        ],
        quotes=[(0, 0.01, 1.0), (50, 0.01, 0.0), (100, 0.01, 1.0)],
        latency=200,
    )
    assert result["created_orders"] == 2
    assert [(fill["order_id"], fill["timestamp_ms"]) for fill in fills] == [
        (1, 220),
        (2, 400),
    ]
    assert gains == pytest.approx([1.5, 0.0, 3.0])


def test_many_known_funding_offsets_complete_under_one_small_query_budget():
    state = _state(0, bid_levels=((995, 1.0), (990, 3.0)))
    paths = [
        ([state], [_trade(stamp, price="99.5", quantity=quantity, maker_buyer=True)])
        for stamp, quantity in ((10, "2"), (20, "1.5"))
    ]
    model = calibration_from_paths(paths, [-0.002, 0.003])
    new_order = _maker_new_order(
        0,
        "buy",
        1.0,
        state,
        _quote(0, book_segment_id="s1"),
        _maker_config().latency_ms,
    )
    model._load()
    for index in range(2048):
        elapsed = (5000000, 15000000, 25000000)[index % 3] + index
        keep, quantity, gain = model.select(
            state=state,
            decision_ns=0,
            new_order=new_order,
            active_order=None,
            commands=(),
            lifetime_ns=500000000,
            prediction=-0.005,
            fee_bps=1.0,
            funding=[(elapsed, 0.02)],
        )
        assert not keep and quantity == 0.5
        assert gain == pytest.approx((0.020025, 0.025025, 0.030025)[index % 3])


@pytest.mark.parametrize("include_zero", (False, True))
def test_paired_event_response_has_interior_optimum_and_keeps_zero_samples(
    include_zero,
):
    state = _state(0, bid_levels=((995, 1.0), (990, 3.0)))
    paths = [
        ([state], [_trade(stamp, price="99.5", quantity=quantity, maker_buyer=True)])
        for stamp, quantity in ((10, "2"), (20, "1.5"))
    ]
    errors = [-0.002, 0.003]
    if include_zero:
        paths.append(([state], []))
        errors.append(10.0)
    model = calibration_from_paths(paths, errors)
    new_order = _maker_new_order(
        0,
        "buy",
        1.0,
        state,
        _quote(0, book_segment_id="s1"),
        _maker_config().latency_ms,
    )
    query = dict(
        state=state,
        decision_ns=0,
        new_order=new_order,
        active_order=None,
        commands=(),
        lifetime_ns=500000000,
        prediction=-0.005,
        fee_bps=1.0,
    )
    keep, quantity, gain = model.select(**query, funding=[])
    assert not keep and quantity == 0.5
    assert gain == pytest.approx(0.04005 / len(paths))
    keep, funded_quantity, _gain = model.select(**query, funding=[(15000000, 1.0)])
    assert not keep and funded_quantity == 1.0


def test_delayed_post_only_rejection_preserves_decision_quantity_and_zero_sample():
    config = replace(_maker_config(), latency_ms=10)
    coverage = {"s1": (0, 600000000)}
    trades_coverage = {"trade-connection-1": (0, 600000000)}
    trade = _trade(20, price="99.5", quantity="3", maker_buyer=True)
    row = {
        **_quote(0, book_segment_id="s1"),
        "price_prediction": 0.0,
        "price_label": 0.003,
        "proxy_prediction": 1.0,
        "label_available_at_ns": 500000000,
    }
    chosen = []
    for crossing in (False, True):
        books = [
            _state(0),
            _state(5, bid_index=980, ask_index=990) if crossing else _state(5),
            _state(30),
            _state(500, bid_index=998, ask_index=1008),
        ]
        batches, observations, events = ([], [], [])
        result = simulate_maker(
            books,
            [trade],
            [_quote(0, book_segment_id="s1")],
            config=config,
            observed_coverage=coverage,
            trade_coverage=trades_coverage,
            capital=PortfolioCapital("fixture", 1000.0, "USDT", "linear_perpetual"),
            asset_id="btc_usdt",
            funding_events=(),
            decision_windows_ns=((0, 500000000),),
            on_ledger=batches.append,
        )
        replay_maker_calibration(
            books,
            [trade],
            [row],
            order_quantity=config.order_quantity,
            latency_ms=config.latency_ms,
            observed_coverage=coverage,
            trade_coverage=trades_coverage,
            cutoff_ns=500000000,
            on_sample=lambda observations=observations, **sample: observations.append(
                sample
            ),
            on_event=lambda event_id, event, events=events: events.append(
                (event_id, event.observed_time_ns)
            ),
        )
        chosen.append(
            next(
                item.number
                for batch in batches
                for item in batch.diagnostics
                if item.name == "maker_buy_requested_quantity"
            )
        )
        buy = next(sample for sample in observations if sample["side"] == "buy")
        assert buy["filled"] == (0.0 if crossing else 1.0)
        assert any(
            (
                stamp == 30000000 and buy["event_start"] <= event_id < buy["event_end"]
                for event_id, stamp in events
            )
        )
        assert result["post_only_rejections"] == int(crossing)
        if crossing:
            assert buy["terminal_reason"] == "post_only_crossing_rejected"
    assert chosen == [1.0, 1.0]


def test_reused_queue_cannot_renew_original_prediction_expiry():
    batches = []
    result = simulate_maker(
        [_state(0), _state(500), _state(600)],
        [
            _trade(499, price="99.5", quantity="1", maker_buyer=True),
            _trade(501, price="99.5", quantity="3", maker_buyer=True),
        ],
        [_quote(stamp, book_segment_id="s1") for stamp in (0, 100, 200, 300, 400)],
        config=_maker_config(),
        observed_coverage={"s1": (0, 600000000)},
        trade_coverage={"trade-connection-1": (0, 600000000)},
        capital=PortfolioCapital("fixture", 1000.0, "USDT", "linear_perpetual"),
        asset_id="btc_usdt",
        funding_events=(),
        on_ledger=batches.append,
    )
    assert result["created_orders"] == 2 and result["filled_orders"] == 0
    assert result["average_order_lifetime_ms"] == 500.0
    assert any(
        row.text == "keep_queue_without_renewal"
        for batch in batches
        for row in batch.diagnostics
    )


@pytest.mark.parametrize(
    "pending, keep_wins", ((False, False), (True, False), (False, True))
)
def test_calibrated_maker_compares_old_expiry_with_real_cancel_replace(
    pending, keep_wins
):
    decision_ms = 100 if pending else 200
    trade_schedule = (
        [(350, "3"), (450, "3")] if pending else [(100, "1"), (350, "2"), (650, "1")]
    )
    fills, batches = ([], [])
    paths = [
        (
            [_state(0)],
            [
                _trade(stamp, price="99.5", quantity=buy, maker_buyer=True),
                _trade(stamp + 1, price="100.5", quantity=sell, maker_buyer=False),
            ],
        )
        for stamp, buy, sell in (
            ((100, "3", "2"), (400, "2.5", "1.5")) if keep_wins else ((400, "3", "2"),)
        )
    ]
    model = calibration_from_paths(paths, [-0.0021, -0.0071] if keep_wins else [0.0])
    result = simulate_maker(
        [_state(stamp) for stamp in (0, 100, 200, 300, 400, 500, 600, 700)],
        [
            _trade(stamp, price="99.5", quantity=quantity, maker_buyer=True)
            for stamp, quantity in trade_schedule
        ],
        [
            {
                **_quote(stamp, book_segment_id="s1"),
                "fold_id": "fold",
                "price_prediction": 0.0,
                "proxy_prediction": 1.0,
            }
            for stamp in (0, decision_ms)
        ],
        config=replace(
            _maker_config(), maker_fee_bps=0.0, latency_ms=300 if pending else 0
        ),
        observed_coverage={"s1": (0, 1000000000)},
        trade_coverage={"trade-connection-1": (0, 1000000000)},
        capital=PortfolioCapital("fixture", 1000.0, "USDT", "linear_perpetual"),
        asset_id="btc_usdt",
        funding_events=(),
        funding_forecasts=(FundingForecast(0, 1000000000, ()),),
        calibrations={"fold": model},
        on_ledger=batches.append,
        on_fill=fills.append,
    )
    assert result["created_orders"] == (3 if keep_wins else 4)
    assert [fill["timestamp_ms"] for fill in fills] == (
        [350] if keep_wins else [350, 450] if pending else [650]
    )
    assert all(fill["estimated_queue_ahead_at_join"] == 2.0 for fill in fills)
    gains = [
        row.number
        for batch in batches
        for row in batch.diagnostics
        if row.name == "maker_buy_expected_net_gain"
    ]
    assert gains == pytest.approx([0.0925, 0.145] if keep_wins else [0.5, 0.5])
    if keep_wins:
        assert fills[0]["order_id"] == 1


def test_decisions_reuse_captured_sample_origins_and_read_history_once(monkeypatch):
    from quant_research_platform.policy.lifecycle import closing_iterator

    decoded = {"book": 0, "trade": 0}
    native_events = MakerCalibration.events

    def observed_events(model, start, stop):
        with closing_iterator(native_events(model, start, stop)) as events:
            for event in events:
                decoded[event.kind] += 1
                yield event

    monkeypatch.setattr(MakerCalibration, "events", observed_events)
    state = _state(0)
    paths = [([state], [_trade(100, price="99.5", quantity="5", maker_buyer=True)])] * 3
    samples, events = calibration_inputs(paths, [0.0] * 3)
    reads = []

    def history():
        reads.append(True)
        yield from events()

    order = _maker_new_order(
        0,
        "buy",
        0.3,
        state,
        _quote(0, book_segment_id="s1"),
        _maker_config().latency_ms,
    )
    model = MakerCalibration(samples, history, price_unit="0.1")
    for prediction in (0.0, 0.01):
        keep, quantity, gain = model.select(
            state=state,
            decision_ns=0,
            new_order=replace(order),
            active_order=None,
            commands=(),
            lifetime_ns=500000000,
            prediction=prediction,
            fee_bps=0.0,
            funding=[],
        )
        assert not keep and quantity == 0.3
        assert gain == pytest.approx(0.15 + 30 * prediction)
    assert reads == [True]
    assert decoded == {"book": 0, "trade": 12}


def test_unknown_historical_visible_quantity_blocks_maker_without_faking_fill():
    from quant_research_platform.contracts.portfolios import PortfolioTargetBlocked
    from quant_research_platform.markets.implementations.crypto.execution import (
        _checked_trades,
    )

    trade = replace(
        _trade(100, price="99.5", quantity="2", maker_buyer=True), visible_quantity=None
    )
    assert list(_checked_trades([trade])) == [trade]
    model = calibration_from_paths([([_state(0)], [trade])], [0.0])
    order = _maker_new_order(
        0,
        "buy",
        1.0,
        _state(0),
        _quote(0, book_segment_id="s1"),
        _maker_config().latency_ms,
    )
    with pytest.raises(PortfolioTargetBlocked, match="visible_quantity_unknown"):
        model.select(
            state=_state(0),
            decision_ns=0,
            new_order=order,
            active_order=None,
            commands=(),
            lifetime_ns=500000000,
            prediction=0.0,
            fee_bps=0.0,
            funding=[],
        )


@pytest.mark.parametrize("calibrated", [False, True])
@pytest.mark.parametrize("short_source", ["book", "trade"])
def test_future_coverage_end_cannot_change_earlier_maker_decision(
    calibrated, short_source
):

    def run(end):
        batches = []
        model = calibration_from_paths(
            [
                (
                    [_state(0)],
                    [_trade(400, price="99.5", quantity="5", maker_buyer=True)],
                )
            ],
            [0.0],
        )
        simulate_maker(
            [_state(0), _state(100)],
            [],
            [
                {
                    **_quote(timestamp, book_segment_id="s1"),
                    "fold_id": "fold",
                    "price_prediction": 0.0,
                    "proxy_prediction": 1.0,
                }
                for timestamp in (0, 100)
            ],
            config=replace(_maker_config(), maker_fee_bps=0.0, latency_ms=0),
            observed_coverage={
                "s1": (0, end if short_source == "book" else 1000000000)
            },
            trade_coverage={
                "trade-connection-1": (
                    0,
                    end if short_source == "trade" else 1000000000,
                )
            },
            capital=PortfolioCapital("fixture", 1000.0, "USDT", "linear_perpetual"),
            asset_id="btc_usdt",
            funding_events=(),
            funding_forecasts=(FundingForecast(0, 2000000000, ()),),
            calibrations={"fold": model} if calibrated else None,
            on_ledger=batches.append,
        )
        return [
            row
            for batch in batches
            for row in batch.diagnostics
            if row.name.startswith("maker_buy_") or row.name.startswith("maker_sell_")
        ]

    short, long = (run(250000000), run(1000000000))
    assert short == long
    assert any(
        row.name == "maker_buy_requested_quantity" and row.number > 0 for row in long
    )


def test_native_history_is_shared_by_sides_and_recomputes_economic_queries(monkeypatch):
    from quant_research_platform.policy.lifecycle import closing_iterator

    decoded = {"book": 0, "trade": 0}
    native_events = MakerCalibration.events

    def observed_events(model, start, stop):
        with closing_iterator(native_events(model, start, stop)) as events:
            for event in events:
                decoded[event.kind] += 1
                yield event

    monkeypatch.setattr(MakerCalibration, "events", observed_events)

    state = _state(0, bid_levels=((995, 1.0), (990, 3.0)))
    paths = [
        ([state], [_trade(stamp, price="99.5", quantity="1.5", maker_buyer=True)])
        for stamp in (10, 20)
    ]
    sample_source, event_source = calibration_inputs(paths, [-0.002, 0.003])
    reads = []

    def samples():
        reads.append("samples")
        yield from sample_source()

    def events():
        reads.append("events")
        yield from event_source()

    order = _maker_new_order(
        0,
        "buy",
        1.0,
        state,
        _quote(0, book_segment_id="s1"),
        _maker_config().latency_ms,
    )
    queries = [
        dict(
            state=state,
            decision_ns=0,
            new_order=replace(order),
            active_order=replace(order, remaining=0.5) if index == 2 else None,
            commands=((15000000, "buy", None),) if index == 3 else (),
            lifetime_ns=20000000 if index == 3 else 500000000,
            prediction=(0.01, -0.005, 0.02, -0.02)[index],
            fee_bps=float(index),
            funding=[] if index == 0 else [(15000000, 0.02)],
        )
        for index in range(4)
    ]
    queries.append(
        {
            **queries[0],
            "new_order": _maker_new_order(
                0,
                "sell",
                1.0,
                state,
                _quote(0, book_segment_id="s1"),
                _maker_config().latency_ms,
            ),
        }
    )
    shared = MakerCalibration(samples, events, price_unit="0.1")
    answers = [shared.select(**query) for query in queries]
    assert reads == ["samples", "events"]
    shared.select(**queries[0])
    assert reads == ["samples", "events"]
    for answer, query in zip(answers, queries, strict=True):
        fresh = MakerCalibration(sample_source, event_source, price_unit="0.1")
        assert fresh.select(**query) == answer
    assert decoded == {"book": 0, "trade": 44}
