from __future__ import annotations

import pytest

from quant_research_platform.contracts.order_book import BookBoundary, BookState
from quant_research_platform.sources.binance.ids import (
    FUTURES_MARKET_TYPE,
    SPOT_MARKET_TYPE,
)
from quant_research_platform.sources.binance.replay import (
    DepthReconstructor,
    ReconstructionStats,
)


def _snapshot(market, update_id=100):
    return dict(
        market_type=market,
        segment_id="segment",
        observation_id="observation",
        payload=dict(
            lastUpdateId=update_id,
            bids=[["101", "1"], ["100", "1"]],
            asks=[["110", "1"], ["111", "1"]],
        ),
    )


def _delta(market, update_id, bids=(), asks=()):
    return dict(
        market_type=market,
        segment_id="segment",
        observation_id="observation",
        source_sequence=update_id,
        receive_time=f"2026-10-01T00:00:00.{update_id:09d}Z",
        receive_monotonic_ns=update_id,
        payload=dict(
            U=update_id,
            u=update_id,
            pu=update_id - 1,
            E=update_id,
            T=update_id,
            b=[[str(p), str(q)] for p, q in bids],
            a=[[str(p), str(q)] for p, q in asks],
        ),
    )


def _engine(market, price_unit="1"):
    engine = DepthReconstructor(
        market,
        price_unit=price_unit,
        snapshot_limit=2,
        required_levels=2,
        connection_close_ns=2_000_000_000_000_000_000,
        stats=ReconstructionStats(),
    )
    engine.bootstrap(_snapshot(market))
    return engine


@pytest.mark.parametrize("market", [SPOT_MARKET_TYPE, FUTURES_MARKET_TYPE])
def test_absolute_updates_extend_only_contiguous_known_price_ranges(market):
    engine = _engine(market)
    first_id = 101 if market == SPOT_MARKET_TYPE else 100
    # The hole at 99/112 keeps individually observed levels non-executable.
    first = list(
        engine.feed(_delta(market, first_id, [(97, 2), (98, 4)], [(113, 3), (114, 5)]))
    )
    assert first[0].bids == ((101, 1), (100, 1))
    assert first[0].asks == ((110, 1), (111, 1))
    assert engine.advance(_delta(market, first_id + 1, [(98, 0)], [(113, 0)])) is True
    assert engine.advance(_delta(market, first_id + 2, [(98, 7)], [(113, 9)])) is True
    # Explicit zeros bridge the hole, using the latest absolute quantities.
    last = list(
        engine.feed(
            _delta(
                market,
                first_id + 3,
                [(99, 0), (100, 0), (101, 0)],
                [(112, 0), (110, 0), (111, 0)],
            )
        )
    )
    assert len(last) == 1 and isinstance(last[0], BookState)
    assert last[0].bids == ((98, 7), (97, 2))
    assert last[0].asks == ((113, 9), (114, 5))
    assert engine.valid and engine.stats.invalid_segments == 0


@pytest.mark.parametrize("market", [SPOT_MARKET_TYPE, FUTURES_MARKET_TYPE])
def test_hole_remains_invalid_until_later_observation_without_backfill(market):
    engine = _engine(market)
    first_id = 101 if market == SPOT_MARKET_TYPE else 100
    first = list(engine.feed(_delta(market, first_id, [(97, 2), (98, 4)])))[0]
    missing = _delta(market, first_id + 1, [(100, 0), (101, 0)])
    boundary = list(engine.feed(missing))[0]
    assert isinstance(boundary, BookBoundary) and boundary.reason == "one_sided_book"
    assert not engine.valid
    assert list(engine.feed(_delta(market, first_id + 2))) == []
    recovered = list(engine.feed(_delta(market, first_id + 3, [(99, 0)])))
    assert len(recovered) == 1 and isinstance(recovered[0], BookState)
    assert recovered[0].bids == ((98, 4), (97, 2))
    assert recovered[0].segment_id != first.segment_id
    assert (
        engine.stats.coverage_segments[first.segment_id][1] == boundary.observed_time_ns
    )
    assert (
        engine.stats.coverage_segments[recovered[0].segment_id][0]
        == recovered[0].observed_time_ns
    )
    assert engine.stats.invalid_segments == 1
    assert engine.stats.valid_segments == 2 and engine.stats.recovered_segments == 1


@pytest.mark.parametrize("market", [SPOT_MARKET_TYPE, FUTURES_MARKET_TYPE])
@pytest.mark.parametrize("previously_expanded", [False, True])
def test_new_snapshot_drops_outside_quantity_and_boundary_knowledge(
    market, previously_expanded
):
    engine = _engine(market)
    first_id = 101 if market == SPOT_MARKET_TYPE else 100
    bids = [(97, 2), (98, 4)] + ([(99, 0)] if previously_expanded else [])
    assert engine.advance(_delta(market, first_id, bids)) is True
    engine.bootstrap(_snapshot(market, 200))
    next_id = 201 if market == SPOT_MARKET_TYPE else 200
    result = engine.advance(_delta(market, next_id, [(99, 0), (100, 0), (101, 0)]))
    assert isinstance(result, BookBoundary) and result.reason == "one_sided_book"


@pytest.mark.parametrize("market", [SPOT_MARKET_TYPE, FUTURES_MARKET_TYPE])
def test_stale_bridge_never_supplies_outside_knowledge(market):
    engine = _engine(market)
    first_id = 101 if market == SPOT_MARKET_TYPE else 100
    assert engine.advance(_delta(market, first_id, [(97, 2), (98, 4)])) is True
    assert engine.advance(_delta(market, first_id, [(99, 0)])) is False
    result = engine.advance(_delta(market, first_id + 1, [(100, 0), (101, 0)]))
    assert isinstance(result, BookBoundary) and result.reason == "one_sided_book"
    assert engine.stats.stale_events == engine.stats.duplicate_events == 1


@pytest.mark.parametrize("market", [SPOT_MARKET_TYPE, FUTURES_MARKET_TYPE])
@pytest.mark.parametrize("damage", ["sequence_gap", "invalid_numeric"])
def test_gap_or_bad_numeric_cannot_promote_outside_updates(market, damage):
    engine = _engine(market)
    first_id = 101 if market == SPOT_MARKET_TYPE else 100
    assert engine.advance(_delta(market, first_id, [(97, 2), (98, 4)])) is True
    row = _delta(market, first_id + 1, [(99, 0)])
    if damage == "sequence_gap":
        row["payload"].update(U=first_id + 2, u=first_id + 2, pu=first_id + 1)
    else:
        row["payload"]["a"] = [["112", "1e309"]]
    boundary = engine.advance(row)
    assert isinstance(boundary, BookBoundary) and boundary.reason == damage
    assert not engine.valid
    assert engine.advance(_delta(market, first_id + 3, [(99, 0)])) is False
    engine.bootstrap(_snapshot(market, 200))
    next_id = 201 if market == SPOT_MARKET_TYPE else 200
    result = engine.advance(_delta(market, next_id, [(99, 0), (100, 0), (101, 0)]))
    assert isinstance(result, BookBoundary) and result.reason == "one_sided_book"


def test_representation_unit_never_infers_exchange_tick_or_unobserved_zeros():
    engine = _engine(FUTURES_MARKET_TYPE, "0.1")
    assert engine.advance(_delta(FUTURES_MARKET_TYPE, 100, [(98, 3), (99, 4)])) is True
    # Whole-number prices do not prove intermediate 0.1 price indices empty.
    result = engine.advance(_delta(FUTURES_MARKET_TYPE, 101, [(100, 0), (101, 0)]))
    assert isinstance(result, BookBoundary) and result.reason == "one_sided_book"


def test_sparse_far_prices_do_not_materialize_unobserved_grid():
    engine = _engine(FUTURES_MARKET_TYPE, "0.000000000000000001")
    assert (
        engine.advance(_delta(FUTURES_MARKET_TYPE, 100, [(1, 3)], [(10**30, 4)]))
        is True
    )
    assert len(engine._bids) == len(engine._asks) == 2
    assert len(engine._pending_bids) == len(engine._pending_asks) == 1


@pytest.mark.parametrize("market", [SPOT_MARKET_TYPE, FUTURES_MARKET_TYPE])
@pytest.mark.parametrize("already_uncovered", [False, True])
def test_crossed_evidence_stays_terminal_even_when_depth_is_insufficient(
    market, already_uncovered
):
    engine = _engine(market)
    first_id = 101 if market == SPOT_MARKET_TYPE else 100
    assert engine.advance(_delta(market, first_id)) is True
    if already_uncovered:
        initial = engine.advance(_delta(market, first_id + 1, [(100, 0)]))
        assert isinstance(initial, BookBoundary)
        assert initial.reason == "unsupported_coverage"
        first_id += 1
    crossed = engine.advance(_delta(market, first_id + 1, [(100, 0)], [(99, 1)]))
    assert isinstance(crossed, BookBoundary) and crossed.reason == "crossed_book"
    assert engine.stats.crossed_books == 1 and not engine.valid
    assert engine.advance(_delta(market, first_id + 2, [(100, 1)], [(99, 0)])) is False
    assert not engine.valid
    engine.bootstrap(_snapshot(market, 200))
    next_id = 201 if market == SPOT_MARKET_TYPE else 200
    assert engine.advance(_delta(market, next_id)) is True


@pytest.mark.parametrize("market", [SPOT_MARKET_TYPE, FUTURES_MARKET_TYPE])
@pytest.mark.parametrize("empty_side", ["bids", "asks"])
def test_crossed_pending_positive_is_terminal_but_pending_zero_is_not(
    market, empty_side
):
    for pending_quantity in (0, 1):
        engine = _engine(market)
        first_id = 101 if market == SPOT_MARKET_TYPE else 100
        pending_bids = [(97, pending_quantity), (98, pending_quantity)]
        pending_asks = [(113, pending_quantity), (114, pending_quantity)]
        assert (
            engine.advance(
                _delta(
                    market,
                    first_id,
                    pending_bids if empty_side == "bids" else (),
                    pending_asks if empty_side == "asks" else (),
                )
            )
            is True
        )
        result = engine.advance(
            _delta(
                market,
                first_id + 1,
                [(100, 0), (101, 0)] if empty_side == "bids" else [(115, 1)],
                [(96, 1)] if empty_side == "bids" else [(110, 0), (111, 0)],
            )
        )
        assert isinstance(result, BookBoundary)
        assert result.reason == (
            "crossed_book" if pending_quantity else "one_sided_book"
        )
        repair = _delta(
            market,
            first_id + 2,
            [(99, 0), (97, 1), (98, 1)] if empty_side == "bids" else [(115, 0)],
            [(96, 0)] if empty_side == "bids" else [(112, 0), (113, 1), (114, 1)],
        )
        assert engine.advance(repair) is (not bool(pending_quantity))
        engine.bootstrap(_snapshot(market, 200))
        next_id = 201 if market == SPOT_MARKET_TYPE else 200
        assert engine.advance(_delta(market, next_id)) is True


@pytest.mark.parametrize("market", [SPOT_MARKET_TYPE, FUTURES_MARKET_TYPE])
def test_insufficient_depth_recovers_only_from_continuous_new_evidence(market):
    engine = _engine(market)
    first_id = 101 if market == SPOT_MARKET_TYPE else 100
    first = list(engine.feed(_delta(market, first_id)))[0]
    boundary = list(engine.feed(_delta(market, first_id + 1, [(100, 0)])))[0]
    assert (
        isinstance(boundary, BookBoundary) and boundary.reason == "unsupported_coverage"
    )
    assert not engine.valid
    restored = list(engine.feed(_delta(market, first_id + 2, [(99, 5)])))[0]
    assert isinstance(restored, BookState) and restored.bids == ((101, 1), (99, 5))
    assert restored.segment_id != first.segment_id
    assert (
        engine.stats.coverage_segments[first.segment_id][1] == boundary.observed_time_ns
    )
    assert (
        engine.stats.coverage_segments[restored.segment_id][0]
        == restored.observed_time_ns
    )
    # A subsequent uncovered interval does not relax the native link rule.
    result = engine.advance(_delta(market, first_id + 3, [(99, 0)]))
    assert isinstance(result, BookBoundary) and result.reason == "unsupported_coverage"
    gap = engine.advance(_delta(market, first_id + 5, [(99, 5)]))
    assert isinstance(gap, BookBoundary) and gap.reason == "sequence_gap"
    assert engine.advance(_delta(market, first_id + 6, [(99, 5)])) is False


def test_public_reconstructor_cannot_trust_caller_supplied_clock():
    engine = _engine(FUTURES_MARKET_TYPE)
    record = _delta(FUTURES_MARKET_TYPE, 100)
    record.update(receive_time="not-a-clock", observed_time_ns=123)
    with pytest.raises(ValueError, match="binance_replay_receive_time_invalid"):
        list(engine.feed(record))


def test_public_reconstructor_clock_remains_receive_time():
    engine = _engine(FUTURES_MARKET_TYPE)
    record = _delta(FUTURES_MARKET_TYPE, 100)
    record["observed_time_ns"] = 123
    state = list(engine.feed(record))[0]
    assert state.observed_time_ns == 1790812800000000100
