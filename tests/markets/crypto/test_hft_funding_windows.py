from contextlib import ExitStack
from dataclasses import replace

import pytest

from quant_research_platform.contracts.portfolios import (
    FundingEvent,
    FundingForecast,
    PortfolioCapital,
    PortfolioTargetBlocked,
)
from quant_research_platform.markets.implementations.crypto.execution import (
    simulate_taker,
)
from tests.helpers.binance_hft import execution_state as _state
from tests.helpers.binance_hft import taker_config as _taker_config
from tests.helpers.portfolio_ledger import capture_fill_rows


def test_taker_known_funding_and_continuous_gap_have_hand_calculated_economics(
    tmp_path,
):
    initial_capital = 1000.0
    from quant_research_platform.artifacts.crypto_hft import summarize_crypto_ledger
    from quant_research_platform.artifacts.portfolio_ledger import (
        read_portfolio_ledger_artifact,
        write_portfolio_ledger_artifact,
    )

    decisions = [
        {"timestamp_ms": 0, "prediction": 0.02},
        {"timestamp_ms": 15, "prediction": 0.02},
        {"timestamp_ms": 60, "prediction": 0.0},
        {"timestamp_ms": 80, "prediction": -0.02},
    ]
    books = [
        _state(stamp, bid_index=995 + step * 10, ask_index=1005 + step * 10)
        for step, stamp in enumerate((0, 20, 30, 40, 60, 80, 100))
    ]
    books.append(_state(1000, bid_index=9995, ask_index=10005))
    forecasts = (
        FundingForecast(
            0, 1000000000, ((40000000, -0.103), (60000000, 0.104), (80000000, -0.21))
        ),
    )

    def run(events, known=forecasts):
        batches, fills = ([], [])
        result = simulate_taker(
            books,
            decisions,
            prediction_column="prediction",
            prediction_horizon_ms=50,
            config=_taker_config(latency_ms=10),
            observed_coverage={"s1": (0, 2000000000)},
            capital=PortfolioCapital(
                "fixture", initial_capital, "USDT", "linear_perpetual"
            ),
            asset_id="btc_usdt",
            funding_events=events,
            funding_forecasts=known,
            decision_windows_ns=((0, 20000000), (60000000, 100000000)),
            on_ledger=capture_fill_rows(
                lambda row: fills.append(dict(row)), batches.append
            ),
        )
        selected = []
        boundaries = []
        for batch in batches:
            diagnostics = {row.name: row for row in batch.diagnostics}
            if "decision_requested_quantity" in diagnostics:
                selected.append(diagnostics["decision_requested_quantity"].number)
                if selected[-1] == 0:
                    assert diagnostics["expected_funding_cashflow"].number == 0.0
            for marker in ("oos_window_start", "oos_window_end"):
                if marker in diagnostics:
                    boundaries.append(
                        sum(v.value for v in batch.valuations if v.phase == "end")
                    )
            assert all(v.timestamp_ns <= 100000000 for v in batch.valuations)
        return (result, selected, boundaries, fills, batches)

    events = (
        FundingEvent(40000000, 0.001, 103.0),
        FundingEvent(60000000, -0.001, 104.0),
        FundingEvent(80000000, 0.002, 105.0),
    )
    result, selected, boundaries, fills, batches = run(events)
    assert selected == [1.0, 1.0, 0.0, -1.0]
    assert [row["timestamp_ms"] for row in fills] == [10, 90]
    assert result["created_orders"] == 3 and result["cancelled_orders"] == 1
    assert result["funding"] == pytest.approx(-0.209)
    assert result["fees"] == pytest.approx(0.0205)
    assert result["net_pnl"] == pytest.approx(3.7705)
    assert boundaries == pytest.approx([1000, 1000.48995, 1003.49095, 1003.7705])
    assert boundaries[2] - boundaries[1] == pytest.approx(3.001)
    metadata = {
        "window": {
            "start": "1970-01-01T00:00:00",
            "end": "1970-01-01T00:00:00.100",
        }
    }
    capital = PortfolioCapital("fixture", initial_capital, "USDT", "linear_perpetual")
    root = write_portfolio_ledger_artifact(
        tmp_path / "ledger",
        capital=capital,
        metadata=metadata,
        produce=lambda emit: [emit(batch) for batch in batches],
    )
    artifact = read_portfolio_ledger_artifact(root)
    intervals = summarize_crypto_ledger(
        artifact,
        artifact.iter_chunks(),
        windows_ns=((0, 20000000), (60000000, 100000000)),
    )["intervals"]
    assert [row["kind"] for row in intervals] == ["oos", "gap", "oos"]
    assert [row["net_pnl"] for row in intervals] == pytest.approx(
        [0.48995, 3.001, 0.27955]
    )
    assert [row["funding"] for row in intervals] == pytest.approx([0, 0.001, -0.21])
    assert sum(row["fees"] for row in intervals) == pytest.approx(result["fees"])
    assert sum(row["net_pnl"] for row in intervals) == pytest.approx(result["net_pnl"])
    with pytest.raises(ValueError, match="crypto_ledger_window_marker_invalid"):
        summarize_crypto_ledger(
            artifact,
            artifact.iter_chunks(),
            windows_ns=((0, 21000000), (60000000, 100000000)),
        )
    altered = tuple(
        replace(event, rate=-event.rate, mark_price=event.mark_price + 1)
        for event in events
    )
    changed, chosen, _, _, _ = run(altered)
    assert chosen == selected and changed["net_pnl"] != result["net_pnl"]
    for amount in (-10.0, 10.0):
        future = FundingForecast(61000000, 1000000000, ((95000000, amount),))
        _, chosen, _, _, _ = run(events, (*forecasts, future))
        assert chosen[:3] == selected[:3]


def test_taker_missing_forecast_and_boundary_price_fail_without_future_fill():
    kwargs = dict(
        prediction_column="prediction",
        prediction_horizon_ms=50,
        config=_taker_config(latency_ms=10),
        observed_coverage={"s1": (0, 18000000)},
        capital=PortfolioCapital("fixture", 1000.0, "USDT", "linear_perpetual"),
        asset_id="btc_usdt",
        funding_events=(),
        decision_windows_ns=((0, 20000000),),
    )
    result = simulate_taker(
        [_state(0), _state(30)],
        [{"timestamp_ms": 0, "prediction": 0.02}],
        funding_forecasts=(),
        **kwargs,
    )
    assert result["selection_status"] == "BLOCKED" and result["created_orders"] == 0
    result = simulate_taker(
        [_state(0), _state(30)],
        [{"timestamp_ms": 0, "prediction": 0.02}],
        funding_forecasts=(FundingForecast(0, 1000000000, ()),),
        **kwargs,
    )
    assert result["account_path_status"] == "PARTIAL"
    assert (
        result["account_stop_reason"] == "portfolio_window_boundary_valuation_missing"
    )
    assert result["net_pnl"] is None
    assert result["known_prefix_net_pnl"] is not None


@pytest.mark.parametrize(
    "window",
    (
        {"start": "1970-01-01", "end": "1970-01-01"},
        {"start": "1970-01-01T00:00:00", "end": "1970-01-01T23:59:59.999999999"},
        {
            "start": "1970-01-01T01:00:00+01:00",
            "end": "1970-01-02T00:59:59.999999999+01:00",
        },
    ),
)
def test_independent_episode_collection_resets_capital_and_funding_boundaries(
    tmp_path, window
):
    from functools import partial
    from types import SimpleNamespace

    from quant_research_platform.artifacts.crypto_hft import (
        read_crypto_episode_artifact,
    )
    from quant_research_platform.interfaces.queries import portfolio_curve
    from quant_research_platform.markets.implementations.crypto.portfolio import (
        _execute_portfolio,
        _window,
        _window_bounds,
        _write_backtest,
    )
    from quant_research_platform.policy.lifecycle import closing_iterator
    from tests.helpers.binance_hft import panel_boundary as _boundary

    capital = PortfolioCapital("fixture", 1000.0, "USDT", "linear_perpetual")
    day_stop = 86_400_000_000_000
    bounds = _window_bounds((("direct", _window(window)),))
    assert bounds == ((0, day_stop - 1),)
    coverage = {"s1": (0, 30000000), "s2": (40000000, day_stop)}
    trade_coverage = SimpleNamespace(intervals=lambda: iter(coverage.values()))
    decisions = [
        {"timestamp_ms": time, "score": prediction}
        for time, prediction in ((10, 0.02), (20, 0.02), (40, -0.02))
    ]
    events = tuple(
        FundingEvent(stamp * 1000000, 0.01, 100.0) for stamp in (0, 20, 30, 40, 50)
    )
    book_rows = [
        *(_state(stamp, bid_index=990, ask_index=1000) for stamp in (0, 10, 20)),
        _state(29, bid_index=990, ask_index=1010),
        _boundary(30),
        _state(40, bid_index=1000, ask_index=1010, segment_id="s2"),
        replace(
            _state(86_399_999, bid_index=990, ask_index=1010, segment_id="s2"),
            observed_time_ns=day_stop - 1,
            receive_time="1970-01-01T23:59:59.999999999Z",
        ),
    ]
    read_count = 0

    def books():
        nonlocal read_count
        for value in book_rows:
            read_count += 1
            yield value

    funding = SimpleNamespace(
        identity="d" * 64,
        coverage=((0, day_stop - 1),),
        settlement_qualification=(
            {
                "coverage": {
                    "start_ns": 0,
                    "end_ns": day_stop - 1,
                    "status": "COMPLETE",
                },
                "settlement_source": None,
                "verification": "declared_archive_not_independently_verified",
            },
        ),
        events=events,
        forecasts=iter((FundingForecast(0, day_stop, ()),)),
    )
    execute = partial(
        simulate_taker,
        prediction_column="score",
        prediction_horizon_ms=500,
        config=replace(
            _taker_config(latency_ms=0), taker_fee_bps=0.0, position_size=2.0
        ),
        observed_coverage=coverage,
        capital=capital,
        asset_id="btc_usdt",
        decision_windows_ns=bounds,
    )
    root = _write_backtest(
        {
            "output": str(tmp_path / "result"),
            "prediction_window": window,
            "execution_mode": "independent_segments",
            "predictions": "fixture",
        },
        capital,
        produce=lambda on_episode: _execute_portfolio(
            execution_mode="independent_segments",
            execute=execute,
            books=books(),
            trades=None,
            decisions=iter(decisions),
            funding=funding,
            coverage=coverage,
            trade_coverage=trade_coverage,
            bounds=bounds,
            on_ledger=None,
            on_episode=on_episode,
        ),
    )
    artifact = read_crypto_episode_artifact(root)
    summaries = []
    indexes = []
    with closing_iterator(artifact.iter_episodes()) as rows:
        for index, ledger in rows:
            from quant_research_platform.artifacts.crypto_hft import (
                summarize_crypto_ledger,
            )

            restored = summarize_crypto_ledger(ledger, ledger.iter_chunks())
            assert restored["funding_qualification"]["sources"] == list(
                funding.settlement_qualification
            )
            summaries.append(restored["economics"])
            assert ledger.capital == capital
            indexes.append(index)
            with closing_iterator(ledger.iter_chunks()) as batches:
                batches = list(batches)
            assert batches[0].sequence == 0
            assert all(
                index["start_ns"] <= row.timestamp_ns < index["stop_ns"]
                for batch in batches
                for row in batch.valuations
            )
            assert (
                sum(row.value for row in batches[0].valuations if row.phase == "before")
                == 1000.0
            )
    assert [row["net_pnl"] for row in summaries] == pytest.approx([-1.0, 1.0])
    assert [row["funding"] for row in summaries] == pytest.approx([-1.0, 1.0])
    assert read_count == len(book_rows)

    assert [
        (
            row["start_ns"],
            row["stop_ns"],
            row["terminal_valuation_ns"],
            row["termination"],
        )
        for row in indexes
    ] == [
        (0, 30_000_000, 29_000_000, "TRUNCATED"),
        (40_000_000, day_stop, day_stop - 1, "PLANNED_END"),
    ]
    assert "planned_oos_end" not in indexes[0]["stop_reasons"]
    assert "planned_oos_end" in indexes[1]["stop_reasons"]
    curve = portfolio_curve(root, execution_mode="independent_segments")
    assert curve["episode_count"] == 2
    assert [row["observed_end_ns"] for row in curve["episodes"]] == [
        29_000_000,
        day_stop - 1,
    ]
    assert [row["points"][-1]["nav"] for row in curve["episodes"]] == pytest.approx(
        [999.0, 1001.0]
    )


def test_episode_atomic_publication_cancel_and_no_overwrite(tmp_path):
    import pandas as pd

    from quant_research_platform.artifacts.crypto_hft import (
        write_crypto_episode_artifact,
    )
    from quant_research_platform.markets.implementations.crypto.execution import (
        PortfolioEconomics,
    )
    from quant_research_platform.storage.errors import OutputAlreadyExistsError

    capital = PortfolioCapital("fixture", 1000.0, "USDT", "linear_perpetual")
    metadata = {
        "window": {
            "start": pd.Timestamp(0).isoformat(),
            "end": pd.Timestamp(30_000_000).isoformat(),
        }
    }

    def produce(emit):
        emit(
            start_ns=0,
            stop_ns=30_000_001,
            termination="PLANNED_END",
            stop_reasons=("planned_oos_end",),
            produce=lambda sink: PortfolioEconomics(capital, "btc_usdt", sink).mark(
                0, 100.0
            ),
        )

    old = write_crypto_episode_artifact(
        tmp_path / "old", capital=capital, metadata=metadata, produce=produce
    )
    before = {
        path.relative_to(old): path.read_bytes()
        for path in old.rglob("*")
        if path.is_file()
    }

    def cancel(emit):
        produce(emit)
        raise KeyboardInterrupt("cancel_after_complete_child")

    with pytest.raises(KeyboardInterrupt, match="cancel_after_complete_child"):
        write_crypto_episode_artifact(
            tmp_path / "new", capital=capital, metadata=metadata, produce=cancel
        )
    assert not (tmp_path / "new").exists()
    with pytest.raises(OutputAlreadyExistsError):
        write_crypto_episode_artifact(
            old, capital=capital, metadata=metadata, produce=produce
        )
    assert before == {
        path.relative_to(old): path.read_bytes()
        for path in old.rglob("*")
        if path.is_file()
    }
    assert not list(tmp_path.glob(".a-*"))


def test_funding_cut_between_books_carries_information_without_account_state():
    from functools import partial
    from types import SimpleNamespace

    from quant_research_platform.markets.implementations.crypto.portfolio import (
        _execute_portfolio,
    )

    count = 0
    observed = []

    def books():
        nonlocal count
        for stamp in (0, 60):
            count += 1
            yield _state(stamp)

    def emit(**episode):
        batches = []
        result = episode.pop("produce")(batches.append)
        observed.append((episode, result, batches))

    coverage = {"s1": (0, 61000000)}
    funding = SimpleNamespace(
        identity="a" * 64,
        coverage=((0, 70000000),),
        settlement_qualification=(
            {
                "coverage": {"start_ns": 0, "end_ns": 70000000, "status": "COMPLETE"},
                "settlement_source": None,
                "verification": "declared_archive_not_independently_verified",
            },
        ),
        events=(
            FundingEvent(30000000, None, None),
            FundingEvent(40000000, 0.01, 100.0),
            FundingEvent(50000000, 0.01, 100.0),
        ),
        forecasts=iter((FundingForecast(0, 1000000000, ()),)),
    )
    execute = partial(
        simulate_taker,
        prediction_column="score",
        prediction_horizon_ms=500,
        config=replace(_taker_config(latency_ms=0), taker_fee_bps=0.0),
        observed_coverage=coverage,
        capital=PortfolioCapital("fixture", 1000.0, "USDT", "linear_perpetual"),
        asset_id="btc_usdt",
        decision_windows_ns=((0, 60000000),),
    )
    _execute_portfolio(
        execution_mode="independent_segments",
        execute=execute,
        books=books(),
        trades=None,
        decisions=iter(
            ({"timestamp_ms": 10, "score": 0.02}, {"timestamp_ms": 40, "score": -0.02})
        ),
        funding=funding,
        coverage=coverage,
        trade_coverage=SimpleNamespace(intervals=lambda: iter(coverage.values())),
        bounds=((0, 60000000),),
        on_ledger=None,
        on_episode=emit,
    )
    assert count == 2 and len(observed) == 3
    assert observed[1][0]["start_ns"] == 30000000
    assert observed[2][0]["start_ns"] == 30000001
    assert [row[1]["funding"] for row in observed] == [0.0, 0.0, 1.0]
    assert [row[1]["initial_capital"] for row in observed] == [1000.0] * 3
    for episode, _result, batches in observed:
        assert all(
            episode["start_ns"] <= value.timestamp_ns < episode["stop_ns"]
            for batch in batches
            for value in batch.valuations
        )
        assert (
            sum(
                value.value
                for value in batches[0].valuations
                if value.phase == "before"
            )
            == 1000.0
        )


@pytest.mark.parametrize("start,stop", ((5000000, 9000000), (12000000, 18000000)))
def test_observed_empty_episode_has_real_initial_value_and_no_trades(start, stop):
    batches = []
    result = simulate_taker(
        [_state(0), _state(20)],
        [],
        prediction_column="score",
        prediction_horizon_ms=500,
        config=_taker_config(latency_ms=0),
        observed_coverage={"s1": (0, 31000000)},
        decision_windows_ns=((0, 10000000), (20000000, 30000000)),
        observation_window_ns=(start, stop),
        funding_events=(),
        funding_forecasts=(),
        capital=PortfolioCapital("fixture", 1000.0, "USDT", "linear_perpetual"),
        asset_id="btc_usdt",
        on_ledger=batches.append,
    )
    assert result["no_eligible_decisions"]
    assert result["nav"] == 1000.0
    assert not [value for batch in batches for value in batch.trades]
    stamps = [value.timestamp_ns for batch in batches for value in batch.valuations]
    assert stamps and min(stamps) == start and (max(stamps) < stop)


def test_episode_qualification_does_not_split_fold_or_day_and_empty_is_blocked():
    from types import SimpleNamespace

    from quant_research_platform.markets.implementations.crypto.portfolio import (
        _execute_episodes,
        _qualified_episodes,
    )

    day = 86400000000000
    bounds = ((0, day // 2), (day, day * 2))
    coverage = {"s1": (0, day * 2 + 1)}
    trade = SimpleNamespace(intervals=lambda: iter(coverage.values()))
    funding = SimpleNamespace(coverage=((0, day * 2),), events=())
    assert list(_qualified_episodes(coverage, trade, funding, bounds)) == [
        (
            0,
            day * 2 + 1,
            "PLANNED_END",
            (
                "book_coverage_end",
                "funding_coverage_end",
                "planned_oos_end",
                "trade_coverage_end",
            ),
        )
    ]
    funding.coverage = ((0, 2**63 - 1),)
    longer_calendar = list(_qualified_episodes(coverage, trade, funding, bounds))
    assert longer_calendar[0][:3] == (0, day * 2 + 1, "PLANNED_END")
    assert "funding_coverage_end" not in longer_calendar[0][3]
    with pytest.raises(ValueError, match="window_overflow"):
        list(_qualified_episodes(coverage, trade, funding, ((0, 2**63 - 1),)))
    with ExitStack() as resources:
        with pytest.raises(PortfolioTargetBlocked, match="qualified_episode_missing"):
            _execute_episodes(
                books=(),
                trades=None,
                decisions=(),
                forecasts=(),
                funding=funding,
                coverage={},
                trade_coverage=trade,
                bounds=bounds,
                resources=resources,
                execute=lambda **_kwargs: pytest.fail("no qualified interval"),
                on_episode=lambda **_kwargs: pytest.fail(
                    "must not publish empty episode"
                ),
            )


def test_persisted_funding_days_intersect_market_coverage_without_bridging(tmp_path):
    from types import SimpleNamespace

    from quant_research_platform.markets.implementations.crypto.portfolio import (
        _qualified_episodes,
    )
    from quant_research_platform.sources.binance.funding import (
        read_funding_observations,
    )
    from tests.sources.test_binance_funding import _write

    roots = []
    for number, start in enumerate((0, 20, 40)):
        raw = tmp_path / str(number) / "raw"
        _write(
            raw,
            metadata_changes={
                "coverage": {
                    "start_ns": start,
                    "end_ns": start + 9,
                    "status": "COMPLETE",
                }
            },
        )
        roots.append(raw.parent / "funding" / "BTCUSDT")
    books = {"first": (0, 25), "second": (27, 60)}
    trades = SimpleNamespace(intervals=lambda: iter(((5, 22), (24, 45))))
    bounds = ((0, 49),)
    with read_funding_observations(tuple(roots), symbol="BTCUSDT") as source:
        episodes = list(_qualified_episodes(books, trades, source, bounds))
        assert [(row[0], row[1]) for row in episodes] == [
            (5, 10),
            (10, 20),
            (20, 22),
            (24, 25),
            (27, 30),
            (30, 40),
            (40, 45),
        ]
        assert all(row[2] == "TRUNCATED" for row in episodes)
        assert episodes[0][3] == ("funding_coverage_end",)
        assert episodes[-1][3] == ("trade_coverage_end",)


@pytest.mark.parametrize("next_start,continuous", ((10, True), (11, False)))
def test_funding_nanosecond_adjacency_keeps_maximal_episode(next_start, continuous):
    from types import SimpleNamespace

    from quant_research_platform.markets.implementations.crypto.portfolio import (
        _qualified_episodes,
    )

    source = SimpleNamespace(
        identity="a" * 64, coverage=((0, 9), (next_start, 20)), events=()
    )
    books = {"whole": (0, 21)}
    trades = SimpleNamespace(intervals=lambda: iter(((0, 21),)))
    bounds = ((0, 20),)
    episodes = list(_qualified_episodes(books, trades, source, bounds))
    assert [(row[0], row[1]) for row in episodes] == (
        [(0, 21)] if continuous else [(0, 10), (10, 11), (11, 21)]
    )


@pytest.mark.parametrize("missing", (False, True))
def test_native_episode_cut_points_keep_half_open_endpoints_across_funding_roots(
    tmp_path, missing
):
    import pandas as pd

    from quant_research_platform.markets.implementations.crypto.portfolio import (
        _qualified_episodes,
    )
    from quant_research_platform.sources.binance import replay
    from quant_research_platform.sources.binance.funding import (
        read_funding_observations,
    )
    from quant_research_platform.sources.binance.ids import FUTURES_MARKET_TYPE
    from tests.sources.binance.test_replay import _write_coverage_history
    from tests.sources.test_binance_funding import _write

    start = pd.Timestamp("2020-01-01", tz="UTC").value
    second = 10**9
    descriptor = _write_coverage_history(tmp_path / "coverage", 3, 32)
    index = replay.ObservationCoverageIndex(
        tmp_path / "coverage",
        descriptor,
        markets={FUTURES_MARKET_TYPE},
        decode_interval=replay._decode_recorder_interval,
    )
    stamps = (
        start,
        start + 1,
        start + second,
        start + 2 * second,
        start + 2 * second + 1,
        start + 3 * second - 1,
        start + 5 * second,
    )
    roots = []
    for number, (left, right) in enumerate(
        ((start, start + 2 * second - 1), (start + 2 * second, start + 5 * second))
    ):
        raw = tmp_path / str(number) / "raw"
        rows = [
            (
                "settlement",
                stamp,
                None,
                None if missing and i % 2 == 0 else 0.0,
                None if missing and i % 2 else 100.0,
                None,
            )
            for i, stamp in enumerate(stamps)
            if left <= stamp <= right
        ]
        _write(
            raw,
            rows=rows,
            metadata_changes={
                "coverage": {"start_ns": left, "end_ns": right, "status": "COMPLETE"}
            },
        )
        roots.append(raw.parent / "funding" / "BTCUSDT")
    books = {"book": (start, start + 5 * second + 1)}
    bounds = ((start, start + 5 * second),)
    with read_funding_observations(tuple(roots), symbol="BTCUSDT") as funding:
        assert tuple(event.timestamp_ns for event in funding.events) == stamps
        coverage = replay.ObservationCoverage(
            index, FUTURES_MARKET_TYPE, "agg_trades", "BTCUSDT"
        )
        actual = list(_qualified_episodes(books, coverage, funding, bounds))
        expected = [
            (start, start + second),
            (start + 2 * second, start + 3 * second),
            (start + 4 * second, start + 5 * second),
        ]
        if missing:
            # Identified zero-rate settlements need no mark and create no cut.
            expected[:2] = [
                (start, start + 1),
                (start + 1, start + second),
                (start + 2 * second, start + 2 * second + 1),
                (start + 2 * second + 1, start + 2 * second + 2),
                (start + 2 * second + 2, start + 3 * second),
            ]
        assert [(left, right) for left, right, *_ in actual] == expected
        assert all(row[2] == "TRUNCATED" for row in actual)
        early = _qualified_episodes(books, coverage, funding, bounds)
        assert next(early) == actual[0]
        early.close()


def test_funding_coverage_cursor_visits_history_only_once():
    from types import SimpleNamespace

    from quant_research_platform.markets.implementations.crypto.execution import (
        _funding_consumer,
    )

    class Coverage(tuple):
        visits = 0

        def __iter__(self):
            for interval in super().__iter__():
                self.visits += 1
                yield interval

        def __getitem__(self, index):
            self.visits += 1
            return super().__getitem__(index)

    coverage = Coverage((i * 100, (i + 1) * 100 - 1) for i in range(32))
    account = SimpleNamespace(quantity=0)
    consume = _funding_consumer((), account, coverage, lambda: None, {})
    consume(3100)
    account.quantity = 1
    for stamp in range(3101, 3133):
        consume(stamp)
        consume(stamp)
        consume(stamp - 1)
    assert coverage.visits <= len(coverage) + 3 * 32


def test_funding_coverage_cursor_preserves_gap_flat_and_cashflows():
    from fractions import Fraction

    from quant_research_platform.markets.implementations.crypto.execution import (
        PortfolioEconomics,
        _AccountPathStop,
        _funding_consumer,
    )

    account = PortfolioEconomics(
        PortfolioCapital("fixture", 1000.0, "USDT", "linear_perpetual"),
        "btc_usdt",
        None,
    )
    account.quantity = Fraction(2)
    events = (
        FundingEvent(3, 0.125, 80.0),
        FundingEvent(8, 0.0, None),
        FundingEvent(12, -0.25, 40.0),
        FundingEvent(19, 0.125, 40.0),
        FundingEvent(25, None, None),
        FundingEvent(32, -0.125, 80.0),
        FundingEvent(39, 0.125, 80.0),
    )
    consume = _funding_consumer(
        events, account, ((0, 10), (11, 20), (30, 40)), lambda: None, {}
    )
    for stamp in (5, 15, 15, 7):
        consume(stamp)
    consume(20)
    assert [row.timestamp_ns for row in account._unvalued_funding] == [3, 12, 19]
    assert [row.amount for row in account._unvalued_funding] == [-20.0, 20.0, -10.0]
    account.quantity = Fraction()
    consume(30)
    account.quantity = Fraction(-1)
    for stamp in (35, 35, 20):
        consume(stamp)
    assert [row.timestamp_ns for row in account._unvalued_funding] == [3, 12, 19, 32]
    assert [row.amount for row in account._unvalued_funding] == [
        -20.0,
        20.0,
        -10.0,
        -10.0,
    ]

    with pytest.raises(_AccountPathStop) as stopped:
        consume(45)
    assert stopped.value.reason == "portfolio_required_funding_coverage_missing"
    assert stopped.value.timestamp_ns == 41
    assert account._unvalued_funding[-1].timestamp_ns == 39
    assert account._unvalued_funding[-1].amount == 10.0


def test_episode_funding_selection_visits_history_linearly():
    count = 32
    from types import SimpleNamespace

    from quant_research_platform.markets.implementations.crypto.portfolio import (
        _execute_episodes,
    )

    class Counted(tuple):
        visits = 0

        def __iter__(self):
            for value in super().__iter__():
                self.visits += 1
                yield value

    events = Counted(FundingEvent(i * 100 + 50, 0.0, None) for i in range(count))
    coverage = Counted((i * 100, i * 100 + 89) for i in range(count))
    selected = []
    with ExitStack() as resources:
        result = _execute_episodes(
            books=(),
            trades=None,
            decisions=(),
            forecasts=(),
            funding=SimpleNamespace(events=events, coverage=coverage),
            coverage={str(i): (i * 100 + 10, i * 100 + 80) for i in range(count)},
            trade_coverage=None,
            bounds=((0, count * 100 - 1),),
            resources=resources,
            execute=lambda **kwargs: selected.append(
                (kwargs["funding_events"], kwargs["funding_coverage"])
            ),
            on_episode=lambda **kwargs: kwargs["produce"](lambda _batch: None),
        )
    assert result == {"episode_count": count}
    assert events.visits <= 2 * count
    assert coverage.visits <= 2 * count
    assert selected == [((events[i],), (coverage[i],)) for i in range(count)]


def test_episode_funding_slices_keep_boundary_events_and_spanning_coverage():
    from fractions import Fraction
    from types import SimpleNamespace

    from quant_research_platform.markets.implementations.crypto.execution import (
        PortfolioEconomics,
        _AccountPathStop,
        _funding_consumer,
    )
    from quant_research_platform.markets.implementations.crypto.portfolio import (
        _execute_episodes,
    )

    events = tuple(
        FundingEvent(
            stamp,
            None if stamp == 10 else 1.0 if stamp == 40 else 0.125,
            None if stamp in (10, 40) else 80.0,
        )
        for stamp in (4, 5, 9, 10, 11, 19, 20, 29, 30, 39, 40, 41, 44, 45)
    )
    coverage = ((-10, 9), (10, 19), (30, 100))
    selected = []
    with ExitStack() as resources:
        _execute_episodes(
            books=(),
            trades=None,
            decisions=(),
            forecasts=(),
            funding=SimpleNamespace(events=events, coverage=coverage),
            coverage={"book": (5, 45)},
            trade_coverage=None,
            bounds=((5, 44),),
            resources=resources,
            execute=lambda **kwargs: selected.append(kwargs),
            on_episode=lambda **kwargs: kwargs["produce"](lambda _batch: None),
        )
    assert [row["observation_window_ns"] for row in selected] == [
        (5, 10),
        (10, 11),
        (11, 20),
        (20, 30),
        (30, 40),
        (40, 41),
        (41, 45),
    ]
    for row in selected:
        start, stop = row["observation_window_ns"]
        assert row["funding_events"] == tuple(
            event for event in events if start <= event.timestamp_ns < stop
        )
        assert row["funding_coverage"] == tuple(
            interval
            for interval in coverage
            if interval[1] >= start and interval[0] < stop
        )
        outcomes = []
        for intervals in (coverage, row["funding_coverage"]):
            account = PortfolioEconomics(
                PortfolioCapital("fixture", 1000.0, "USDT", "linear_perpetual"),
                "btc_usdt",
                None,
            )
            consume = _funding_consumer(
                row["funding_events"], account, intervals, lambda: None, {}
            )
            # A fresh episode settles its start event while flat, before trading.
            consume(start)
            account.quantity = Fraction(2)
            stopped = None
            try:
                consume(stop - 1)
            except _AccountPathStop as error:
                stopped = (error.reason, error.timestamp_ns)
            outcomes.append((account.cash, account._unvalued_funding, stopped))
        assert outcomes[0] == outcomes[1]
        assert outcomes[1][2] == (
            ("portfolio_required_funding_coverage_missing", 21) if start == 20 else None
        )
