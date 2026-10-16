"""Selected-source stops retain exact prefixes and nullable full-window results."""

import gzip
import hashlib
import json
from dataclasses import asdict
from functools import partial
from types import SimpleNamespace

import pytest

from quant_research_platform.artifacts.crypto_hft import (
    read_crypto_episode_artifact,
    summarize_crypto_ledger,
)
from quant_research_platform.artifacts.portfolio_ledger import (
    read_portfolio_ledger_artifact,
    write_portfolio_ledger_artifact,
)
from quant_research_platform.artifacts.statistics import read_statistics_artifact
from quant_research_platform.contracts.portfolios import (
    FundingEvent,
    FundingForecast,
    PortfolioCapital,
)
from quant_research_platform.markets.implementations.crypto import portfolio
from quant_research_platform.markets.implementations.crypto.execution import (
    TakerConfig,
    simulate_taker,
)
from quant_research_platform.markets.implementations.crypto.ids import HFT_PANEL_DATA_ID
from quant_research_platform.markets.prepare import prepare
from quant_research_platform.research.portfolios.diagnostics import crypto_statistics
from tests.helpers.binance_hft import (
    _write_file,
    write_binance_hft_run,
    write_hft_funding_fixture,
)
from tests.markets.crypto.test_hft_execution import _state
from tests.research.test_crypto_funding_gaps import _policy

CAPITAL = PortfolioCapital("partial", 1000.0, "USDT", "linear_perpetual")
WINDOW = {"start": "1970-01-01T00:00:00", "end": "1970-01-01T00:00:01"}


def _close_depth(raw, manifest, stop=500_000_000):
    for index, member in enumerate(manifest["files"]):
        if member["symbol"] == "BTCUSDT" and member["data_kind"] == "depth":
            with gzip.open(raw / member["path"], "rt") as handle:
                rows = [json.loads(line) for line in handle]
            manifest["files"][index] = _write_file(
                raw,
                symbol=member["symbol"],
                data_kind=member["data_kind"],
                stream=member["stream"],
                rows=[row for row in rows if row["observed_time_ns"] < stop],
            )
    sidecar = raw / "connection_intervals.jsonl.gz"
    with gzip.open(sidecar, "rt") as handle:
        intervals = [json.loads(line) for line in handle]
    for row in intervals:
        if row["stream"] == "btcusdt@depth@100ms":
            row["closed_at"] = "1970-01-01T00:00:00.500000Z"
            row["closed_monotonic_ns"] = stop
    payload = gzip.compress(
        "".join(json.dumps(row) + "\n" for row in intervals).encode(), mtime=0
    )
    sidecar.write_bytes(payload)
    manifest["capture"]["connection_intervals"].update(
        compressed_bytes=len(payload), sha256=hashlib.sha256(payload).hexdigest()
    )
    manifest["status"] = "PARTIAL"
    (raw / "run.json").write_text(json.dumps(manifest))


@pytest.mark.parametrize("mode", ["continuous", "independent_segments"])
@pytest.mark.parametrize("closed", [False, True])
def test_actual_source_close_overrides_reference_panel_and_reaches_statistics(
    tmp_path, mode, closed
):
    reference = tmp_path / "reference" / "raw"
    write_binance_hft_run(reference, count=10, maker_liquidity=True)
    panel = prepare(
        {
            "market_id": "crypto",
            "data_id": HFT_PANEL_DATA_ID,
            "inputs": {},
            "source": {"root": str(reference)},
            "time_window": WINDOW,
            "output_root": str(tmp_path / "panel"),
        }
    )
    raw = tmp_path / "selected" / "raw"
    manifest = write_binance_hft_run(raw, count=10, maker_liquidity=True)
    manifest["run_id"] = "independent-selected-execution"
    (raw / "run.json").write_text(json.dumps(manifest))
    if closed:
        _close_depth(raw, manifest)
    write_hft_funding_fixture(raw)
    config = {
        "output": str(tmp_path / "ledger"),
        "inputs": str(panel),
        "panel": str(panel),
        "source": str(raw),
        "asset_id": "btc_usdt",
        "symbol": "BTCUSDT",
        "prediction_window": WINDOW,
        "execution_mode": mode,
        "capital": asdict(CAPITAL),
        "execution": asdict(portfolio.MakerConfig(1.0, 2.0, 1.0, 0.0, 0.0, 100, 0)),
        "models": [],
        "calibration": None,
    }
    root = portfolio.backtest_maker(config)
    summaries, fills = [], []

    def consume(artifact, batches):
        summaries.append(
            summarize_crypto_ledger(
                artifact,
                batches,
                consume_batch=lambda batch: fills.extend(batch.trades),
            )
        )

    if mode == "continuous":
        read_portfolio_ledger_artifact(root, consume_batches=consume)
    else:
        read_crypto_episode_artifact(
            root,
            consume_episode=lambda _row, artifact, batches: consume(artifact, batches),
        )
    assert fills
    assert (
        all(fill.timestamp_ns < 500_000_000 for fill in fills)
        if closed
        else any(fill.timestamp_ns > 500_000_000 for fill in fills)
    )
    summary = summaries[0]
    if closed and mode == "continuous":
        assert summary["execution"]["account_path_status"] == "PARTIAL"
        assert summary["execution"]["censored_active_order_count"] > 0
        assert summary["economics"]["net_pnl"] is None
        assert summary["execution"]["known_prefix_net_pnl"] is not None
    statistic = read_statistics_artifact(
        crypto_statistics(
            {
                "ledger": str(root),
                "output": str(tmp_path / "statistics"),
                "statistics": _policy(),
            }
        )
    )
    with statistic.table("crypto_portfolio_statistics.parquet") as frame:
        if closed and mode == "continuous":
            row = frame.loc[frame.metric_id.eq("net_pnl")].iloc[0]
            assert row.status == "UNDEFINED" and row.value != row.value
        if mode == "independent_segments":
            assert (
                not frame.loc[frame.episode_id.eq("all"), "metric_id"]
                .isin(["net_pnl", "terminal_nav", "sharpe"])
                .any()
            )


@pytest.mark.parametrize("position", [False, True])
@pytest.mark.parametrize("missing", ["coverage", "event"])
def test_funding_unknown_stops_only_exposed_account_and_publishes(
    tmp_path, position, missing
):
    metadata = {
        "window": WINDOW,
        "sample_role": "prediction",
        "execution_mode": "continuous",
        "execution_style": "taker",
        "asset_id": "btc_usdt",
        "symbol": "BTCUSDT",
        "execution": asdict(TakerConfig(1.0, 2.0, 0, 0.0, 100)),
    }
    funding = SimpleNamespace(
        coverage=() if missing == "coverage" else ((0, 1_000_000_001),),
        events=()
        if missing == "coverage"
        else (FundingEvent(500_000_000, None, None),),
        forecasts=(FundingForecast(0, 2_000_000_000, ()),),
        settlement_qualification=(),
    )
    result = []

    def produce(emit):
        result.append(
            portfolio._execute_portfolio(
                execution_mode="continuous",
                execute=partial(
                    simulate_taker,
                    prediction_column="prediction",
                    prediction_horizon_ms=100,
                    config=TakerConfig(1.0, 2.0, 0, 0.0, 100),
                    observed_coverage={"s1": (0, 1_000_000_001)},
                    capital=CAPITAL,
                    asset_id="btc_usdt",
                    decision_windows_ns=((0, 1_000_000_000),),
                ),
                books=[_state(t) for t in (0, 100, 500, 1000)],
                trades=None,
                decisions=[{"timestamp_ms": 0, "prediction": 0.02}] if position else [],
                funding=funding,
                coverage={"s1": (0, 1_000_000_001)},
                trade_coverage=None,
                bounds=((0, 1_000_000_000),),
                on_ledger=emit,
                on_episode=None,
            )
        )

    root = write_portfolio_ledger_artifact(
        tmp_path / "ledger", capital=CAPITAL, metadata=metadata, produce=produce
    )
    artifact = read_portfolio_ledger_artifact(root)
    summary = summarize_crypto_ledger(
        artifact, artifact.iter_chunks(), windows_ns=((0, 1_000_000_000),)
    )
    assert result[0]["account_path_status"] == ("PARTIAL" if position else "COMPLETE")
    assert (
        summary["economics"]["net_pnl"] is None
        if position
        else summary["economics"]["net_pnl"] == 0
    )
    assert result[0]["created_orders"] == int(position)
    read_statistics_artifact(
        crypto_statistics(
            {
                "ledger": str(root),
                "output": str(tmp_path / "statistics"),
                "statistics": _policy(),
            }
        )
    )


@pytest.mark.parametrize("effective_ms", [499, 500, 501])
def test_taker_selected_source_half_open_boundary_censors_committed_command(
    effective_ms,
):
    result = simulate_taker(
        [_state(0), _state(400)],
        [{"timestamp_ms": 400, "prediction": 0.02}],
        prediction_column="prediction",
        prediction_horizon_ms=500,
        config=TakerConfig(1.0, 2.0, effective_ms - 400, 0.0, 100),
        observed_coverage={"s1": (0, 500_000_000)},
        capital=CAPITAL,
        asset_id="btc_usdt",
        funding_events=(),
        funding_forecasts=(FundingForecast(0, 2_000_000_000, ()),),
        decision_windows_ns=((0, 1_000_000_000),),
    )
    assert result["created_orders"] == 1
    assert result["filled_orders"] == int(effective_ms < 500)
    assert result["account_path_status"] == "PARTIAL"
    if effective_ms >= 500:
        assert result["censored_pending_command_count"] == 1
        assert result["zero_fill_orders"] == result["cancelled_orders"] == 0


@pytest.mark.parametrize("position", [False, True])
def test_book_gap_without_pending_orders_preserves_identifiable_endpoint(
    tmp_path, position
):
    from tests.markets.crypto.test_hft import _boundary

    metadata = {
        "window": WINDOW,
        "sample_role": "prediction",
        "execution_mode": "continuous",
        "execution_style": "taker",
        "asset_id": "btc_usdt",
        "symbol": "BTCUSDT",
        "execution": asdict(TakerConfig(1.0, 2.0, 0, 0.0, 100)),
    }
    result = []

    def produce(emit):
        result.append(
            simulate_taker(
                [
                    _state(0),
                    _state(300),
                    _boundary(400),
                    _state(600, segment_id="s2"),
                    _state(1000, segment_id="s2"),
                ],
                [{"timestamp_ms": 0, "prediction": 0.02}] if position else [],
                prediction_column="prediction",
                prediction_horizon_ms=100,
                config=TakerConfig(1.0, 2.0, 0, 0.0, 100),
                observed_coverage={
                    "s1": (0, 400_000_000),
                    "s2": (600_000_000, 1_000_000_001),
                },
                capital=CAPITAL,
                asset_id="btc_usdt",
                funding_events=(),
                funding_forecasts=(FundingForecast(0, 2_000_000_000, ()),),
                funding_coverage=((0, 1_000_000_001),) if position else (),
                decision_windows_ns=((0, 1_000_000_000),),
                on_ledger=emit,
            )
        )

    root = write_portfolio_ledger_artifact(
        tmp_path / "ledger", capital=CAPITAL, metadata=metadata, produce=produce
    )
    assert result[0]["account_path_status"] == "COMPLETE"
    assert result[0]["market_observation_status"] == (
        "PARTIAL" if position else "COMPLETE"
    )
    assert result[0]["net_pnl"] == (-0.5 if position else 0.0)
    statistic = read_statistics_artifact(
        crypto_statistics(
            {
                "ledger": str(root),
                "output": str(tmp_path / "statistics"),
                "statistics": _policy(),
            }
        )
    )
    with statistic.table("crypto_portfolio_statistics.parquet") as frame:
        metrics = frame.set_index("metric_id")
        assert metrics.loc["net_pnl", "status"] == "DEFINED"
        assert metrics.loc["terminal_nav", "status"] == "DEFINED"
        assert metrics.loc["continuous_max_drawdown", "status"] == (
            "UNDEFINED" if position else "DEFINED"
        )
        if position:
            assert (
                metrics.loc["continuous_max_drawdown", "reason"]
                == "crypto_execution_book_coverage_incomplete"
            )
        assert metrics.loc["oos_inventory_mean", "status"] == "DEFINED"
        assert metrics.loc["oos_inventory_mean", "value"] == int(position)


def test_trade_numeric_boundary_censors_pending_execution_without_erasing_book():
    from quant_research_platform.contracts.order_book import TradeBoundary
    from quant_research_platform.markets.implementations.crypto.execution import (
        simulate_maker,
    )
    from tests.helpers.binance_hft import _quote
    from tests.markets.crypto.test_hft_execution import _maker_config, _trade

    trade = _trade(100, price="99.4", quantity="1", maker_buyer=True)
    boundary = TradeBoundary(
        **{
            name: getattr(trade, name)
            for name in TradeBoundary.__dataclass_fields__
            if name != "reason"
        },
        reason="invalid_numeric",
    )
    results = simulate_maker(
        [_state(0), _state(200)],
        [boundary],
        [_quote(0, book_segment_id="s1")],
        config=_maker_config(),
        observed_coverage={"s1": (0, 201_000_000)},
        trade_coverage={"trade-connection-1": (0, 201_000_000)},
        capital=CAPITAL,
        asset_id="btc_usdt",
        funding_events=(),
        decision_windows_ns=((0, 200_000_000),),
    )
    assert results["account_path_status"] == "PARTIAL"
    assert results["account_stop_ns"] == 100_000_000
    assert results["filled_orders"] == results["cancelled_orders"] == 0
    assert results["censored_active_order_count"] == 2
    assert results["net_pnl"] is None


@pytest.mark.parametrize("mode", ["continuous", "independent_segments"])
def test_native_bad_trade_stops_current_account_without_future_fills(tmp_path, mode):
    raw = tmp_path / "raw"
    manifest = write_binance_hft_run(raw, count=10, maker_liquidity=True)
    panel = prepare(
        {
            "market_id": "crypto",
            "data_id": HFT_PANEL_DATA_ID,
            "inputs": {},
            "source": {"root": str(raw)},
            "time_window": WINDOW,
            "output_root": str(tmp_path / "panel"),
        }
    )
    write_hft_funding_fixture(raw)
    config = {
        "output": str(tmp_path / "clean"),
        "inputs": str(panel),
        "panel": str(panel),
        "source": str(raw),
        "asset_id": "btc_usdt",
        "symbol": "BTCUSDT",
        "prediction_window": WINDOW,
        "execution_mode": mode,
        "capital": asdict(CAPITAL),
        "execution": asdict(portfolio.MakerConfig(1.0, 2.0, 1.0, 0.0, 0.0, 100, 0)),
        "models": [],
        "calibration": None,
    }

    def read(root):
        summaries, fills = [], []

        def consume(artifact, batches):
            summaries.append(
                summarize_crypto_ledger(
                    artifact,
                    batches,
                    consume_batch=lambda batch: fills.extend(batch.trades),
                )
            )

        if mode == "continuous":
            read_portfolio_ledger_artifact(root, consume_batches=consume)
        else:
            read_crypto_episode_artifact(
                root,
                consume_episode=lambda _row, artifact, batches: consume(
                    artifact, batches
                ),
            )
        return summaries, fills

    clean_summaries, clean_fills = read(portfolio.backtest_maker(config))
    assert clean_summaries[0]["execution"]["account_path_status"] == "COMPLETE"
    # A local numeric defect keeps clocks, identity and unrelated stream bytes intact.
    for index, member in enumerate(manifest["files"]):
        if member["symbol"] == "BTCUSDT" and member["data_kind"] == "agg_trades":
            with gzip.open(raw / member["path"], "rt") as stream:
                rows = [json.loads(line) for line in stream]
            rows[3]["payload"]["q"] = "NaN"
            manifest["files"][index] = _write_file(
                raw,
                symbol=member["symbol"],
                data_kind=member["data_kind"],
                stream=member["stream"],
                rows=rows,
            )
    (raw / "run.json").write_text(json.dumps(manifest))
    root = portfolio.backtest_maker({**config, "output": str(tmp_path / "bad")})
    summaries, fills = read(root)
    assert len(summaries) == 1  # No implicit restart inside the planned segment.
    execution = summaries[0]["execution"]
    assert execution["account_path_status"] == "PARTIAL"
    assert execution["account_stop_ns"] == 400_000_000
    assert execution["censored_active_order_count"] > 0
    assert summaries[0]["economics"]["net_pnl"] is None
    assert fills == [fill for fill in clean_fills if fill.timestamp_ns < 400_000_000]
    statistic = read_statistics_artifact(
        crypto_statistics(
            {
                "ledger": str(root),
                "output": str(tmp_path / "statistics"),
                "statistics": _policy(),
            }
        )
    )
    with statistic.table("crypto_portfolio_statistics.parquet") as frame:
        metrics = frame.loc[~frame.episode_id.eq("all")].set_index("metric_id")
        for name in (
            "terminal_nav",
            "continuous_max_drawdown"
            if mode == "continuous"
            else "episode_max_drawdown",
            *(["oos_net_pnl"] if mode == "continuous" else []),
            "pnl_per_fill",
            "pnl_per_notional",
        ):
            assert metrics.loc[name, "status"] == "UNDEFINED"
            assert metrics.loc[name, "reason"] == execution["account_stop_reason"]
    with statistic.table("crypto_portfolio_intervals.parquet") as frame:
        assert frame.empty
    with statistic.table("crypto_cost_sensitivity.parquet") as frame:
        assert frame.net_pnl.isna().all()
        assert frame.cost_delta.notna().all()
    with statistic.table("crypto_portfolio_coverage.parquet") as frame:
        covered = frame.loc[frame.kind.eq("covered")]
        assert (
            covered.stop_at.iloc[-1].value
            == execution["known_prefix_valuation_timestamp_ns"] + 1
        )
        assert frame.kind.eq("missing").any()
    # A local account stop must still close and verify the selected source.
    for member in manifest["files"]:
        if member["symbol"] == "BTCUSDT" and member["data_kind"] == "depth":
            member["sha256"] = "0" * 64
    (raw / "run.json").write_text(json.dumps(manifest))
    invalid_output = tmp_path / "invalid_identity"
    with pytest.raises(ValueError, match="binance_replay_file_identity_mismatch"):
        portfolio.backtest_maker({**config, "output": str(invalid_output)})
    assert not invalid_output.exists()
    assert root.exists()


@pytest.mark.parametrize("boundary_kind", ["trade", "book", "visible_quantity"])
def test_calibration_excludes_unknown_outcomes_and_resumes_later_samples(boundary_kind):
    from quant_research_platform.contracts.order_book import TradeBoundary
    from quant_research_platform.markets.implementations.crypto.execution import (
        replay_maker_calibration,
    )
    from tests.helpers.binance_hft import _quote
    from tests.markets.crypto.test_hft_execution import _trade

    trade = _trade(400, price="99.4", quantity="1", maker_buyer=True)
    boundary = TradeBoundary(
        **{
            name: getattr(trade, name)
            for name in TradeBoundary.__dataclass_fields__
            if name != "reason"
        },
        reason="invalid_numeric",
    )
    book_events = [_state(stamp) for stamp in (0, 300, 600, 1100)]
    trade_events = [boundary]
    if boundary_kind == "visible_quantity":
        from dataclasses import replace

        trade_events = [replace(trade, visible_quantity=None)]
    if boundary_kind == "book":
        from tests.markets.crypto.test_hft import _boundary

        book_events.insert(2, _boundary(400))
        trade_events = []
    rows = [
        {
            **_quote(stamp, book_segment_id="s1"),
            "price_prediction": 0.0,
            "price_label": 0.003,
            "proxy_prediction": 1.0,
            "label_available_at_ns": (stamp + 500) * 1_000_000,
        }
        for stamp in (0, 600)
    ]
    samples, events = [], []
    result = replay_maker_calibration(
        book_events,
        trade_events,
        rows,
        order_quantity=1.0,
        latency_ms=0,
        observed_coverage={"s1": (0, 1_200_000_000)},
        trade_coverage={"trade-connection-1": (0, 1_200_000_000)},
        cutoff_ns=1_100_000_000,
        on_sample=lambda **row: samples.append(row),
        on_event=lambda **row: events.append(row),
    )
    assert result == {"samples": {"buy": 1, "sell": 1}, "excluded_decisions": 1}
    assert len(samples) == 2
    assert all(
        row["sample_id"] == 0 and row["decision_ns"] == 600_000_000 for row in samples
    )
    assert all(not isinstance(row["event"], TradeBoundary) for row in events)


def test_unknown_trade_context_excludes_only_affected_calibration_rows():
    import pandas as pd

    from quant_research_platform.markets.implementations.crypto.portfolio import (
        _paired_validation,
    )

    rows = [
        SimpleNamespace(
            timestamp=pd.Timestamp(stamp, unit="ms"),
            segment_id="s1",
            book_segment_id="s1",
            trade_observation_id=identity,
            prediction_score=0.1,
            label=0.01,
            label_available_at=pd.Timestamp(stamp + 500, unit="ms"),
        )
        for stamp, identity in ((0, pd.NA), (100, "trade"))
    ]
    result = list(_paired_validation([iter(rows), iter(rows)], asset_id="btc_usdt"))
    assert [row["timestamp_ms"] for row in result] == [100]


def test_book_only_independent_execution_has_no_trade_coverage_dependency():
    funding = SimpleNamespace(coverage=((0, 1001),), events=())
    result = list(
        portfolio._qualified_episodes({"book": (0, 1001)}, None, funding, ((0, 1000),))
    )
    assert result == [
        (0, 1001, "PLANNED_END", ("book_coverage_end", "planned_oos_end"))
    ]


@pytest.mark.parametrize("cut", ["book", "funding"])
def test_partial_independent_account_does_not_erase_later_declared_episode(
    tmp_path, cut
):
    from quant_research_platform.artifacts.crypto_hft import (
        write_crypto_episode_artifact,
    )
    from quant_research_platform.contracts.order_book import TradeBoundary
    from quant_research_platform.markets.implementations.crypto.execution import (
        simulate_maker,
    )
    from tests.helpers.binance_hft import _quote
    from tests.markets.crypto.test_hft_execution import _trade

    event = _trade(400, price="99.4", quantity="1", maker_buyer=True)
    bad = TradeBoundary(
        **{
            name: getattr(event, name)
            for name in TradeBoundary.__dataclass_fields__
            if name != "reason"
        },
        reason="invalid_numeric",
    )
    books = (
        {"s1": (0, 500_000_000), "s2": (600_000_000, 1_200_000_001)}
        if cut == "book"
        else {"s1": (0, 1_200_000_001)}
    )
    next_segment = "s2" if cut == "book" else "s1"
    trades = {"trade-connection-1": (0, 1_200_000_001)}
    funding = SimpleNamespace(
        coverage=((0, 1_200_000_001),),
        events=() if cut == "book" else (FundingEvent(600_000_000, None, None),),
        forecasts=(FundingForecast(0, 2_000_000_000, ()),),
        settlement_qualification=(),
    )
    config = portfolio.MakerConfig(1.0, 2.0, 1.0, 0.0, 0.0, 100, 0)
    metadata = {
        "window": {**WINDOW, "end": "1970-01-01T00:00:01.200"},
        "execution_mode": "independent_segments",
        "execution_style": "maker",
        "execution": asdict(config),
        "sample_role": "prediction",
    }
    root = write_crypto_episode_artifact(
        tmp_path / "episodes",
        capital=CAPITAL,
        metadata=metadata,
        produce=lambda emit: portfolio._execute_portfolio(
            execution_mode="independent_segments",
            execute=partial(
                simulate_maker,
                config=config,
                observed_coverage=books,
                trade_coverage=trades,
                capital=CAPITAL,
                asset_id="btc_usdt",
                decision_windows_ns=((0, 1_200_000_000),),
            ),
            books=[
                _state(0),
                _state(300 if cut == "book" else 500),
                *([_state(600, segment_id="s2")] if cut == "book" else []),
                _state(1000, segment_id=next_segment),
                _state(1200, segment_id=next_segment),
            ],
            trades=[
                _trade(100, price="99.4", quantity="1", maker_buyer=True),
                bad,
                _trade(800, price="99.4", quantity="1", maker_buyer=True),
            ],
            decisions=[
                _quote(0, book_segment_id="s1"),
                _quote(700, book_segment_id=next_segment),
            ],
            funding=funding,
            coverage=books,
            trade_coverage=SimpleNamespace(intervals=lambda: iter(trades.values())),
            bounds=((0, 1_200_000_000),),
            on_ledger=None,
            on_episode=emit,
        ),
    )
    summaries = []
    read_crypto_episode_artifact(
        root,
        consume_episode=lambda _row, artifact, batches: summaries.append(
            summarize_crypto_ledger(artifact, batches)
        ),
    )
    assert [row["execution"]["account_path_status"] for row in summaries] == [
        "PARTIAL",
        *(["COMPLETE"] * (1 if cut == "book" else 2)),
    ]
    assert summaries[0]["execution"]["account_stop_ns"] == 400_000_000
    assert summaries[0]["economics"]["net_pnl"] is None
    assert (
        summaries[0]["execution"]["known_prefix_valuation_timestamp_ns"] < 400_000_000
    )
    assert summaries[-1]["economics"]["fill_events"] == 1
    assert all(row["economics"]["initial_capital"] == 1000.0 for row in summaries)
    statistic = read_statistics_artifact(
        crypto_statistics(
            {
                "ledger": str(root),
                "output": str(tmp_path / "statistics"),
                "statistics": _policy(),
            }
        )
    )
    with statistic.table("crypto_portfolio_intervals.parquet") as frame:
        assert frame.start_at.iloc[-1].value == 600_000_000 + (cut == "funding")
        assert (frame.start_at.astype("int64") >= 600_000_000).all()
    with statistic.table("crypto_portfolio_statistics.parquet") as frame:
        assert (
            not frame.loc[frame.episode_id.eq("all"), "metric_id"]
            .isin(["net_pnl", "terminal_nav", "sharpe"])
            .any()
        )


@pytest.mark.parametrize("style", ["taker", "maker"])
def test_account_stop_preserves_markouts_mature_before_stop(tmp_path, style):
    from quant_research_platform.contracts.order_book import TradeBoundary
    from quant_research_platform.markets.implementations.crypto.execution import (
        simulate_maker,
    )
    from tests.helpers.binance_hft import _quote
    from tests.markets.crypto.test_hft_execution import _trade

    def produce(emit):
        common = dict(
            capital=CAPITAL,
            asset_id="btc_usdt",
            observed_coverage={"s1": (0, 1_000_000_001)},
            decision_windows_ns=((0, 1_000_000_000),),
            on_ledger=emit,
        )
        if style == "taker":
            return simulate_taker(
                [_state(0), _state(500)],
                [{"timestamp_ms": 0, "prediction": 0.02}],
                prediction_column="prediction",
                prediction_horizon_ms=500,
                config=TakerConfig(1.0, 2.0, 0, 0.0, 100),
                funding_events=(FundingEvent(400_000_000, None, None),),
                funding_forecasts=(FundingForecast(0, 2_000_000_000, ()),),
                **common,
            )
        trade = _trade(400, price="99.4", quantity="1", maker_buyer=True)
        boundary = TradeBoundary(
            **{
                name: getattr(trade, name)
                for name in TradeBoundary.__dataclass_fields__
                if name != "reason"
            },
            reason="invalid_numeric",
        )
        return simulate_maker(
            [_state(0), _state(500)],
            [_trade(100, price="99.4", quantity="1", maker_buyer=True), boundary],
            [_quote(0, book_segment_id="s1")],
            config=portfolio.MakerConfig(1.0, 2.0, 1.0, 0.0, 0.0, 100, 0),
            trade_coverage={"trade-connection-1": (0, 1_000_000_001)},
            funding_events=(),
            **common,
        )

    root = write_portfolio_ledger_artifact(
        tmp_path / style, capital=CAPITAL, metadata={"window": WINDOW}, produce=produce
    )
    artifact = read_portfolio_ledger_artifact(root)
    batches = list(artifact.iter_chunks())
    summary = summarize_crypto_ledger(artifact, iter(batches))
    assert summary["execution"]["account_path_status"] == "PARTIAL"
    assert summary["execution"]["account_stop_ns"] == 400_000_000
    assert summary["execution"]["markout_undefined_fill_events"] == 0
    assert [
        row.number
        for batch in batches
        for row in batch.diagnostics
        if row.name == "markout_per_unit"
    ] == [-0.5 if style == "taker" else 0.5]
