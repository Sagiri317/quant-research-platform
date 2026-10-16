"""Causal funding gaps must survive the native execution, ledger and STAT owners."""

from types import SimpleNamespace

import pytest

from quant_research_platform.artifacts.crypto_hft import summarize_crypto_ledger
from quant_research_platform.artifacts.portfolio_ledger import (
    read_portfolio_ledger_artifact,
    write_portfolio_ledger_artifact,
)
from quant_research_platform.artifacts.statistics import (
    read_statistics_artifact,
)
from quant_research_platform.contracts.portfolios import (
    FundingEvent,
    FundingForecast,
    PortfolioCapital,
)
from quant_research_platform.markets.implementations.crypto.execution import (
    TakerConfig,
    simulate_taker,
)
from quant_research_platform.markets.implementations.crypto.portfolio import (
    _execute_portfolio,
)
from quant_research_platform.research.portfolios.diagnostics import crypto_statistics
from tests.helpers.binance_hft import hft_research_plan
from tests.markets.crypto.test_hft_execution import _state


def _metadata(plan, mode):
    return {
        "window": plan["test"],
        "sample_role": "test",
        "chronology": {"folds": plan["folds"], "test": plan["test"]},
        "execution_mode": mode,
        "execution_style": "taker",
        "asset_id": "btc_usdt",
        "symbol": "BTCUSDT",
        "execution": {
            "order_quantity": 1.0,
            "position_size": 2.0,
            "latency_ms": 50,
            "taker_fee_bps": 0.0,
            "markout_horizon_ms": 100,
        },
    }


def _policy():
    return {
        "annualization_minimum_observations": 2,
        "annualization_periods": 365.0,
        "fee_scenarios_bps": [0.0, 1.0],
        "additional_spread_scenarios_bps": [0.0, 1.0],
    }


def test_partial_funding_gap_native_port_stat_publication(tmp_path):
    plan = hft_research_plan()
    capital = PortfolioCapital(
        "crypto_hft_latency_0", 1000.0, "USDT", "linear_perpetual"
    )
    metadata = _metadata(plan, "continuous")
    funding = SimpleNamespace(
        identity="f" * 64,
        coverage=((8_600_000_000, 9_001_000_000),),
        forecasts=(
            FundingForecast(8_600_000_000, 8_800_000_000, ((8_700_000_000, -0.1),)),
            FundingForecast(8_800_000_000, 10_000_000_000, ()),
        ),
        events=(FundingEvent(8_700_000_000, 0.001, 100.0),),
        settlement_qualification=(),
    )
    results = []

    def produce(emit):
        result = _execute_portfolio(
            execution_mode="continuous",
            execute=lambda **kwargs: simulate_taker(
                **kwargs,
                prediction_column="prediction",
                prediction_horizon_ms=200,
                config=TakerConfig(1.0, 2.0, 50, 0.0, 100),
                observed_coverage={"s1": (8_600_000_000, 9_001_000_000)},
                capital=capital,
                asset_id="btc_usdt",
                decision_windows_ns=((8_600_000_000, 9_000_000_000),),
            ),
            books=[_state(t) for t in (8600, 8700, 8800, 9000)],
            trades=None,
            decisions=[
                {"timestamp_ms": t, "prediction": 0.02} for t in (8600, 8620, 8800)
            ],
            funding=funding,
            coverage={"s1": (8_600_000_000, 9_001_000_000)},
            trade_coverage=SimpleNamespace(covering=lambda *_: ("trade",)),
            bounds=((8_600_000_000, 9_000_000_000),),
            on_ledger=emit,
            on_episode=None,
        )
        results.append(result)

    root = write_portfolio_ledger_artifact(
        tmp_path / "ledger", capital=capital, metadata=metadata, produce=produce
    )
    artifact = read_portfolio_ledger_artifact(root)
    summary = summarize_crypto_ledger(artifact, artifact.iter_chunks())
    result = results[0]
    assert result["selection_status"] == "PARTIAL"
    assert result["funding_required_decisions"] == 3
    assert result["funding_covered_decisions"] == 2
    assert result["funding_decision_coverage"] == pytest.approx(2 / 3)
    assert result["created_orders"] == result["filled_orders"] == 2
    assert result["funding"] == -0.1
    assert summary["economics"]["net_pnl"] == pytest.approx(-1.1)
    trades = [trade for batch in artifact.iter_chunks() for trade in batch.trades]
    assert [trade.timestamp_ns for trade in trades] == [8_650_000_000, 8_850_000_000]
    statistic = read_statistics_artifact(
        crypto_statistics(
            {
                "analyses": [
                    "performance",
                    "annualized",
                    "execution",
                    "cost_sensitivity",
                ],
                "ledger": str(root),
                "output": str(tmp_path / "statistics"),
                "statistics": _policy(),
            }
        )
    )
    with statistic.table("crypto_portfolio_statistics.parquet") as frame:
        assert frame.set_index("metric_id").loc[
            "terminal_nav", "value"
        ] == pytest.approx(998.9)
        continuous = frame.loc[frame.episode_id.eq("continuous")].set_index("metric_id")
        risk = continuous.loc[
            ["oos_mean_daily_return", "sharpe", "annualized_volatility"]
        ]
        assert risk.status.eq("UNDEFINED").all()
        assert risk.reason.eq("insufficient_complete_utc_days").all()
        assert risk.value.isna().all()
        assert continuous.loc["net_pnl", "value"] == pytest.approx(-1.1)
    with statistic.table("crypto_execution_diagnostics.parquet") as frame:
        rows = frame.loc[frame.episode_id.eq("continuous")].set_index("metric_id")
        assert rows.loc["selection_status", "text"] == "PARTIAL"
        assert rows.loc["blocked_decisions", "value"] == 1
        assert rows.loc["funding_decision_coverage", "value"] == pytest.approx(2 / 3)
        assert (
            rows.loc["funding_gap_policy", "text"]
            == "causal_no_new_risk_partial_forecast_v1"
        )
    assert summary["execution"]["selection_status"] == "PARTIAL"
    assert summary["execution"]["funding_covered_decisions"] == 2
    assert any(
        d.text == "portfolio_expected_funding_missing"
        for b in artifact.iter_chunks()
        for d in b.diagnostics
    )


@pytest.mark.parametrize("forecasts", [(), (FundingForecast(0, 1, ()),)])
def test_no_covered_forecast_decision_stays_blocked(forecasts):
    funding = SimpleNamespace(
        identity="f" * 64,
        coverage=((0, 101_000_000),),
        forecasts=forecasts,
        events=(),
        settlement_qualification=(),
    )
    result = _execute_portfolio(
        execution_mode="continuous",
        execute=lambda **kwargs: simulate_taker(
            **kwargs,
            prediction_column="prediction",
            prediction_horizon_ms=50,
            config=TakerConfig(1.0, 2.0, 0, 0.0, 50),
            observed_coverage={"s1": (0, 101_000_000)},
            capital=PortfolioCapital("fixture", 1000.0, "USDT", "linear_perpetual"),
            asset_id="btc_usdt",
            decision_windows_ns=((0, 100_000_000),),
        ),
        books=[_state(0), _state(100)],
        trades=None,
        decisions=[{"timestamp_ms": 0, "prediction": 0.02}],
        funding=funding,
        coverage={"s1": (0, 101_000_000)},
        trade_coverage=SimpleNamespace(covering=lambda *_: ("trade",)),
        bounds=((0, 100_000_000),),
        on_ledger=None,
        on_episode=None,
    )
    assert result["selection_status"] == "BLOCKED"
    assert result["net_pnl"] == 0 and result["created_orders"] == 0


@pytest.mark.parametrize("missing", ["settlement", "book", "trade"])
def test_flat_path_does_not_require_unconsumed_source_coverage(missing):
    funding = SimpleNamespace(
        coverage=() if missing == "settlement" else ((0, 101_000_000),),
        forecasts=(),
        events=(),
        settlement_qualification=(),
    )
    result = _execute_portfolio(
        execution_mode="continuous",
        execute=lambda **kwargs: simulate_taker(
            **kwargs,
            prediction_column="prediction",
            prediction_horizon_ms=50,
            config=TakerConfig(1.0, 2.0, 0, 0.0, 50),
            observed_coverage={} if missing == "book" else {"s1": (0, 101_000_000)},
            capital=PortfolioCapital("fixture", 1000.0, "USDT", "linear_perpetual"),
            asset_id="btc_usdt",
            decision_windows_ns=((0, 100_000_000),),
        ),
        books=() if missing == "book" else [_state(0), _state(100)],
        trades=None,
        decisions=(),
        funding=funding,
        coverage={} if missing == "book" else {"s1": (0, 101_000_000)},
        trade_coverage=None,
        bounds=((0, 100_000_000),),
        on_ledger=None,
        on_episode=None,
    )
    assert result["net_pnl"] == 0 and result["created_orders"] == 0
    assert result["account_path_status"] == "COMPLETE"


@pytest.mark.parametrize(
    "gap_ms,trade_ms",
    [(50, 125), (50, 150), (50, 151), (100, 150), (100, 200), (100, 201)],
)
def test_calibrated_maker_gap_cancel_respects_latency_and_settlement(gap_ms, trade_ms):
    from dataclasses import replace

    from quant_research_platform.markets.implementations.crypto.execution import (
        simulate_maker,
    )
    from tests.helpers.binance_hft import _quote
    from tests.markets.crypto.calibration_test_support import calibration_from_paths
    from tests.markets.crypto.test_hft_execution import _maker_config, _trade

    calibration = calibration_from_paths(
        [([_state(0)], [_trade(150, price="99.5", quantity="4", maker_buyer=True)])],
        [0.0],
    )
    ledger = []
    result = simulate_maker(
        [_state(t) for t in (0, 100, 300, 900)],
        [_trade(trade_ms, price="99.5", quantity="4", maker_buyer=True)],
        [
            {
                **_quote(t, book_segment_id="s1"),
                "fold_id": "fold",
                "price_prediction": 0.02,
                "proxy_prediction": 1.0,
            }
            for t in (0, gap_ms, 300)
        ],
        config=replace(_maker_config(calibrated=True), latency_ms=100),
        observed_coverage={"s1": (0, 901_000_000)},
        trade_coverage={"trade-connection-1": (0, 901_000_000)},
        capital=PortfolioCapital("fixture", 1000.0, "USDT", "linear_perpetual"),
        asset_id="btc_usdt",
        funding_events=(FundingEvent(250_000_000, 0.001, 100.0),),
        funding_forecasts=(
            FundingForecast(0, 500_000_000, ()),
            FundingForecast(300_000_000, 1_000_000_000, ()),
        ),
        calibrations={"fold": calibration},
        on_ledger=ledger.append,
    )
    assert result["selection_status"] == "PARTIAL"
    assert (
        result["blocked_decisions"] == 1 and result["funding_required_decisions"] == 3
    )
    assert result["created_orders"] == 2
    trades = [trade for batch in ledger for trade in batch.trades]
    assert len(trades) == (1 if trade_ms <= gap_ms + 100 else 0)
    assert result["funding"] == (-0.1 if trade_ms <= gap_ms + 100 else 0.0)
    diagnostics = [d for batch in ledger for d in batch.diagnostics]
    assert sum(d.text == "portfolio_expected_funding_missing" for d in diagnostics) == 2
    gains = [d.number for d in diagnostics if d.name == "maker_buy_expected_net_gain"]
    assert gains[1] is None


@pytest.mark.parametrize("all_uncovered", [False, True])
def test_independent_episode_gap_does_not_block_covered_sibling(
    tmp_path, all_uncovered
):
    from functools import partial

    from quant_research_platform.artifacts.crypto_hft import (
        read_crypto_episode_artifact,
        write_crypto_episode_artifact,
    )

    plan = hft_research_plan()
    capital = PortfolioCapital(
        "crypto_hft_latency_0", 1000.0, "USDT", "linear_perpetual"
    )
    metadata = _metadata(plan, "independent_segments")
    coverage = {
        "s1": (8_600_000_000, 8_700_000_000),
        "s2": (8_800_000_000, 9_001_000_000),
    }
    funding = SimpleNamespace(
        identity="f" * 64,
        coverage=((8_600_000_000, 9_001_000_000),),
        forecasts=()
        if all_uncovered
        else (FundingForecast(8_600_000_000, 8_700_000_000, ()),),
        events=(),
        settlement_qualification=(),
    )

    def produce(emit):
        return _execute_portfolio(
            execution_mode="independent_segments",
            execute=partial(
                simulate_taker,
                prediction_column="prediction",
                prediction_horizon_ms=50,
                config=TakerConfig(1.0, 2.0, 0, 0.0, 100),
                observed_coverage=coverage,
                capital=capital,
                asset_id="btc_usdt",
                decision_windows_ns=((8_600_000_000, 9_000_000_000),),
            ),
            books=[
                _state(8600),
                _state(8699),
                _state(8800, segment_id="s2"),
                _state(9000, segment_id="s2"),
            ],
            trades=None,
            decisions=[{"timestamp_ms": t, "prediction": 0.02} for t in (8600, 8800)],
            funding=funding,
            coverage=coverage,
            trade_coverage=SimpleNamespace(intervals=lambda: iter(coverage.values())),
            bounds=((8_600_000_000, 9_000_000_000),),
            on_ledger=None,
            on_episode=emit,
        )

    root = write_crypto_episode_artifact(
        tmp_path / "episodes", capital=capital, metadata=metadata, produce=produce
    )
    summaries = []
    read_crypto_episode_artifact(
        root,
        consume_episode=lambda episode, ledger, batches: summaries.append(
            summarize_crypto_ledger(ledger, batches)
        ),
    )
    assert [s["execution"]["selection_status"] for s in summaries] == [
        "BLOCKED" if all_uncovered else "COMPLETE",
        "BLOCKED",
    ]
    assert [s["execution"]["funding_decision_coverage"] for s in summaries] == [
        0.0 if all_uncovered else 1.0,
        0.0,
    ]
    assert [s["economics"]["fill_events"] for s in summaries] == [
        0 if all_uncovered else 1,
        0,
    ]
    assert [s["economics"]["initial_capital"] for s in summaries] == [1000.0, 1000.0]

    statistic = read_statistics_artifact(
        crypto_statistics(
            {
                "analyses": [
                    "performance",
                    "annualized",
                    "execution",
                    "cost_sensitivity",
                ],
                "ledger": str(root),
                "output": str(tmp_path / "statistics"),
                "statistics": _policy(),
            }
        )
    )
    with statistic.table("crypto_execution_diagnostics.parquet") as frame:
        aggregate = frame.loc[frame.episode_id.eq("all")].set_index("metric_id")
        assert aggregate.loc["funding_selection_status", "text"] == (
            "BLOCKED" if all_uncovered else "PARTIAL"
        )
        assert aggregate.loc["funding_decision_coverage", "value"] == (
            0.0 if all_uncovered else 0.5
        )
        assert set(frame.loc[frame.metric_id.eq("selection_status"), "text"]) == (
            {"BLOCKED"} if all_uncovered else {"COMPLETE", "BLOCKED"}
        )
    with statistic.table("crypto_portfolio_statistics.parquet") as frame:
        assert (
            not frame.loc[frame.episode_id.eq("all"), "metric_id"]
            .isin(("sharpe", "net_pnl", "terminal_nav"))
            .any()
        )


def test_maker_impossible_new_order_without_pending_risk_needs_no_forecast():
    from dataclasses import replace

    from quant_research_platform.markets.implementations.crypto.execution import (
        simulate_maker,
    )
    from tests.helpers.binance_hft import _quote
    from tests.markets.crypto.calibration_test_support import calibration_from_paths
    from tests.markets.crypto.test_hft_execution import _maker_config, _trade

    calibration = calibration_from_paths(
        [([_state(0)], [_trade(150, price="99.5", quantity="4", maker_buyer=True)])],
        [0.0],
    )
    result = simulate_maker(
        [_state(0), _state(600)],
        [],
        [
            {
                **_quote(0, book_segment_id="s1"),
                "fold_id": "fold",
                "price_prediction": 0.02,
                "proxy_prediction": 1.0,
            }
        ],
        config=replace(_maker_config(calibrated=True), latency_ms=500),
        observed_coverage={"s1": (0, 601_000_000)},
        trade_coverage={"trade-connection-1": (0, 601_000_000)},
        capital=PortfolioCapital("fixture", 1000.0, "USDT", "linear_perpetual"),
        asset_id="btc_usdt",
        funding_events=(),
        funding_forecasts=(),
        calibrations={"fold": calibration},
    )
    assert (
        result["created_orders"]
        == result["funding_required_decisions"]
        == result["blocked_decisions"]
        == 0
    )
    assert result["funding_decision_coverage"] is None
