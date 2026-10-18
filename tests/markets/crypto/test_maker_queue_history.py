"""Native queue facts across execution, calibration, and historical commands.

Keep the native edge cases together; history cells select distinct command and
coordinate branches rather than repeating the full Cartesian product.
"""

from dataclasses import replace

import pytest

from quant_research_platform.artifacts.crypto_hft import summarize_crypto_ledger
from quant_research_platform.contracts.portfolios import PortfolioCapital
from quant_research_platform.markets.implementations.crypto.execution import (
    replay_maker_calibration,
    simulate_maker,
)
from quant_research_platform.sources.binance.ids import FUTURES_MARKET_TYPE
from tests.helpers.binance_hft import _coverage, _quote
from tests.helpers.binance_hft import execution_trade as _trade
from tests.helpers.binance_hft import maker_config as _maker_config
from tests.markets.crypto.test_hft_execution import _persist_execution
from tests.sources.binance.test_known_price_ranges import _delta, _engine


def native(side, truncated=False, gap=False):
    engine = _engine(FUTURES_MARKET_TYPE)

    def feed(uid, ms, **kw):
        record = _delta(FUTURES_MARKET_TYPE, uid, **kw)
        record["receive_time"] = f"1970-01-01T00:00:00.{ms:03d}000000Z"
        return next(engine.feed(record))

    first = feed(100, 0)
    engine.output_levels = 1 if truncated else None
    updates = [(103, 1), (102, 1)] if side == "buy" else [(108, 1), (109, 1)]
    if not truncated:
        updates += [(101, 0), (100, 0)] if side == "buy" else [(110, 0), (111, 0)]
    second = feed(
        102 if gap else 101, 5, **{"bids" if side == "buy" else "asks": updates}
    )
    return [first, second] if gap else [first, second, feed(102, 200)]


@pytest.mark.parametrize(
    "side,mode",
    [
        ("buy", "known_zero"),
        ("sell", "known_zero"),
        ("buy", "truncated"),
        ("sell", "truncated"),
        ("buy", "gap"),
    ],
)
def test_native_queue_execution_and_calibration(tmp_path, side, mode):
    books = native(side, mode == "truncated", mode == "gap")
    trade = replace(
        _trade(
            20,
            price="101" if side == "buy" else "110",
            quantity="1",
            maker_buyer=side == "buy",
        ),
        price_unit="1",
        price_index=101 if side == "buy" else 110,
    )
    capital = PortfolioCapital("independent", 1000.0, "USDT", "linear_perpetual")
    results = []

    def produce(emit):
        results.append(
            simulate_maker(
                books,
                [trade],
                [_quote(0, book_segment_id="segment")],
                config=replace(
                    _maker_config(),
                    latency_ms=10,
                    inventory_target=2.0 if side == "buy" else -2.0,
                    inventory_skew=1.0,
                ),
                observed_coverage={"segment": _coverage(0, 201)},
                trade_coverage={"trade-connection-1": _coverage(0, 201)},
                capital=capital,
                asset_id="btc_usdt",
                funding_events=(),
                decision_windows_ns=((0, 200_000_000),),
                on_ledger=emit,
            )
        )

    artifact = _persist_execution(tmp_path, capital, produce)
    summary = summarize_crypto_ledger(
        artifact, artifact.iter_chunks(), windows_ns=((0, 200_000_000),)
    )
    result = results[0]
    expected = "COMPLETE" if mode == "known_zero" else "PARTIAL"
    assert result["account_path_status"] == expected
    assert summary["execution"]["account_path_status"] == expected
    if mode == "known_zero":
        assert result["filled_orders"] == 1
        assert result["net_pnl"] is not None
    else:
        assert result["account_stop_ns"] == (5_000_000 if mode == "gap" else 10_000_000)
        assert result["net_pnl"] is None
        assert summary["economics"]["net_pnl"] is None

    row = {
        **_quote(0, book_segment_id="segment"),
        "proxy_prediction": 1.0,
        "price_prediction": 0.0,
        "price_label": 0.0,
        "label_available_at_ns": 500_000_000,
    }
    samples = []
    result = replay_maker_calibration(
        books,
        [trade],
        [row],
        order_quantity=1.0,
        latency_ms=10,
        observed_coverage={"segment": _coverage(0, 501)},
        trade_coverage={"trade-connection-1": _coverage(0, 501)},
        cutoff_ns=500_000_000,
        on_sample=lambda **kw: samples.append(kw),
        on_event=lambda **kw: None,
    )
    if mode == "known_zero":
        assert result["samples"] == {"buy": 1, "sell": 1}
        assert next(s for s in samples if s["side"] == side)["filled"] == 1.0
    else:
        assert result["samples"] == {"buy": 0, "sell": 0}
        assert result["excluded_decisions"] == 1
        assert not samples


@pytest.mark.parametrize(
    "side,scale,keep,truncated",
    [
        ("buy", 2, True, False),
        ("sell", 2, False, False),
        ("buy", 1, True, True),
        ("sell", 2, False, True),
    ],
)
def test_native_history_keep_reprice_scaling(side, scale, keep, truncated):
    from types import SimpleNamespace

    from quant_research_platform.contracts.portfolios import PortfolioTargetBlocked
    from quant_research_platform.markets.implementations.crypto.execution import (
        MakerCalibration,
        _maker_new_order,
    )

    books = native(side, truncated)
    initial = books[0]
    current = replace(
        initial,
        price_unit="1" if scale == 1 else "0.5",
        bids=tuple((p * scale, q) for p, q in initial.bids),
        asks=tuple((p * scale, q) for p, q in initial.asks),
        known_bid_floor_tick=initial.known_bid_floor_tick * scale,
        known_ask_ceiling_tick=initial.known_ask_ceiling_tick * scale,
    )
    new_order = _maker_new_order(
        1, side, 1.0, current, _quote(0, book_segment_id="segment"), 10
    )
    sample = {
        "decision_at": 0,
        "initial_bid_index": 101,
        "initial_ask_index": 110,
        "book_segment_id": "segment",
        "trade_observation_id": "trade-connection-1",
    }
    events = [
        SimpleNamespace(**{**vars(books[1]), "kind": "book", "event_at": 5_000_000}),
        SimpleNamespace(
            kind="trade",
            event_at=20_000_000,
            observation_id="trade-connection-1",
            price_index=101 if side == "buy" else 110,
            visible_quantity=1.0,
            buyer_is_maker=side == "buy",
        ),
    ]
    calibration = MakerCalibration(lambda: (), lambda: (), price_unit="1")
    result = calibration._response(
        sample,
        state=current,
        decision_ns=0,
        new_order=new_order,
        active_order=None,
        commands=[(10_000_000, side, new_order)] if keep else [],
        lifetime_ns=500_000_000,
        replace_quantity=None if keep else 1.0,
        history_scale=scale,
        events=iter(events),
    )
    if truncated:
        with pytest.raises(
            PortfolioTargetBlocked, match="activation_queue_depth_unknown"
        ):
            list(result)
    else:
        fills = list(result)
        assert len(fills) == 1 and fills[0][1] == 1
        assert fills[0][2] == (101.0 if side == "buy" else 110.0)
        assert fills[0][3] is (not keep)


def test_public_unknown_history_preserves_schedule():
    from quant_research_platform.contracts.portfolios import FundingForecast
    from tests.helpers.portfolio_ledger import capture_fill_rows
    from tests.markets.crypto.calibration_test_support import calibration_from_paths

    initial = native("buy")[0]
    historical_trade = replace(
        _trade(20, price="101", quantity="3", maker_buyer=True),
        price_unit="1",
        price_index=101,
    )
    known = calibration_from_paths([([initial], [historical_trade])], [0.0])
    unknown = calibration_from_paths([(native("buy", True), [historical_trade])], [0.0])
    quotes = [
        {
            **_quote(100, book_segment_id="segment"),
            "fold_id": "unknown",
            "price_prediction": 0.1,
            "proxy_prediction": 1.0,
        }
    ]
    quotes.insert(
        0,
        {
            **_quote(0, book_segment_id="segment"),
            "fold_id": "known",
            "price_prediction": 0.1,
            "proxy_prediction": 1.0,
        },
    )
    fills, batches = [], []
    result = simulate_maker(
        [initial],
        [
            replace(
                historical_trade,
                observed_time_ns=300_000_000,
                event_time_ms=300,
                transaction_time_ms=300,
            )
        ],
        quotes,
        config=replace(
            _maker_config(calibrated=True), latency_ms=10, maker_fee_bps=0.0
        ),
        observed_coverage={"segment": _coverage(0, 1001)},
        trade_coverage={"trade-connection-1": _coverage(0, 1001)},
        capital=PortfolioCapital("independent", 1000.0, "USDT", "linear_perpetual"),
        asset_id="btc_usdt",
        funding_events=(),
        funding_forecasts=(FundingForecast(0, 1_001_000_000, ()),),
        calibrations={"known": known, "unknown": unknown},
        on_ledger=capture_fill_rows(fills.append, batches.append),
    )
    assert result["selection_status"] == "PARTIAL"
    assert result["unknown_queue_decisions"] == 1
    assert result["created_orders"] == 1
    assert len(fills) == 1
    assert any(
        d.name == "maker_buy_selection_reason"
        and d.text == "activation_queue_depth_unknown"
        for b in batches
        for d in b.diagnostics
    )
