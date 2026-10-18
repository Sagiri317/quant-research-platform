"""Certified zero queues survive the native producer and calibration tape."""

import pandas as pd

from quant_research_platform.artifacts.statistics import read_statistics_artifact
from quant_research_platform.markets.implementations.crypto import portfolio
from quant_research_platform.markets.implementations.crypto.execution import (
    _maker_join,
    _maker_new_order,
)
from quant_research_platform.sources.binance.ids import FUTURES_MARKET_TYPE
from quant_research_platform.sources.binance.replay import (
    DepthReconstructor,
    ReconstructionStats,
)
from tests.helpers.binance_hft import _quote
from tests.helpers.binance_hft import execution_state as _state
from tests.markets.crypto.test_direct_portfolio import _independent_maker_case
from tests.sources.binance.test_known_price_ranges import _delta, _snapshot


def test_native_certified_zero_queues_survive_calibration_roundtrip(
    tmp_path, monkeypatch
):
    config, _backtest = _independent_maker_case(tmp_path, monkeypatch)

    def books(*_args, stats=None, **_kwargs):
        engine = DepthReconstructor(
            FUTURES_MARKET_TYPE,
            price_unit="0.1",
            snapshot_limit=2,
            required_levels=2,
            connection_close_ns=10_000_000_000,
            stats=stats if stats is not None else ReconstructionStats(),
        )
        snapshot = _snapshot(FUTURES_MARKET_TYPE)
        snapshot["segment_id"] = "s1"
        snapshot["payload"].update(
            bids=[["99.5", "2"], ["99.0", "3"]], asks=[["100.5", "1"], ["101.0", "2"]]
        )
        engine.bootstrap(snapshot)
        for index, stamp in enumerate((2100, 2150, 2650, 2700)):
            record = _delta(
                FUTURES_MARKET_TYPE,
                100 + index,
                bids=[("99.5", 0), ("99.0", 0), ("99.7", 1), ("99.6", 1)]
                if index == 1
                else (),
                asks=[("100.5", 0), ("101.0", 0), ("100.3", 1), ("100.4", 1)]
                if index == 1
                else (),
            )
            record["segment_id"] = "s1"
            record["receive_time"] = pd.Timestamp(
                stamp, unit="ms", tz="UTC"
            ).isoformat()
            yield from engine.feed(record)

    monkeypatch.setattr(portfolio, "iter_reconstructed_depth", books)
    artifact = read_statistics_artifact(
        portfolio.calibrate_maker(config),
        validate_tables=portfolio.validate_maker_calibration,
    )
    with artifact.table("crypto_maker_events.parquet") as frame:
        assert frame.known_price_bound.dtype == "int64"
        assert frame.loc[frame.event_kind.ne("book"), "known_price_bound"].eq(0).all()
    event = next(
        event
        for event in portfolio._calibration_events(artifact, "observed")
        if event.kind == "book" and event.event_at == 2_150_000_000
    )
    assert event.known_bid_floor_tick == 990 < event.bids[-1][0]
    assert event.known_ask_ceiling_tick == 1010 > event.asks[-1][0]
    for side in ("buy", "sell"):
        order = _maker_new_order(
            1, side, 1.0, _state(2100), _quote(2100, book_segment_id="s1"), 100
        )
        assert (
            _maker_join(
                order,
                event,
                {"s1": (0, 10_000_000_000)},
                {"trade-connection-1": (0, 10_000_000_000)},
            )
            is None
        )
        assert order.queue_ahead == 0


def test_legacy_book_producer_bounds_fall_back_to_visible_depth(tmp_path, monkeypatch):
    config, _backtest = _independent_maker_case(tmp_path, monkeypatch)
    artifact = read_statistics_artifact(
        portfolio.calibrate_maker(config),
        validate_tables=portfolio.validate_maker_calibration,
    )
    event = next(
        event
        for event in portfolio._calibration_events(artifact, "observed")
        if event.kind == "book"
    )
    assert event.known_bid_floor_tick == event.bids[-1][0]
    assert event.known_ask_ceiling_tick == event.asks[-1][0]
