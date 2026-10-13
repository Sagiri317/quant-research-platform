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
    write_statistics_artifact,
)
from quant_research_platform.contracts.portfolios import (
    FundingEvent,
    FundingForecast,
    PortfolioCapital,
    PortfolioTargetBlocked,
)
from quant_research_platform.contracts.time import TimeWindow
from quant_research_platform.markets.implementations.crypto.execution import (
    TakerConfig,
    simulate_taker,
)
from quant_research_platform.markets.implementations.crypto.portfolio import (
    _execute_portfolio,
)
from quant_research_platform.profiles.research import (
    CRYPTO_PROFILE_ID,
    bind_research_configuration,
    research_profile,
)
from quant_research_platform.storage.layout import StorageLayout
from quant_research_platform.storage.software_identity import (
    production_software_identity,
)
from quant_research_platform.workflows.research import _target_envelope_for_spec
from tests.helpers.binance_hft import hft_research_plan
from tests.helpers.research_configuration import research_configuration
from tests.markets.crypto.test_hft_execution import _state


def test_partial_funding_gap_native_port_stat_publication(tmp_path):
    plan = hft_research_plan()
    config = research_configuration(
        "crypto",
        "baseline",
        plan=plan,
        development=(
            plan["folds"][0]["validation"]["start"],
            plan["folds"][-1]["validation"]["end"],
        ),
    )
    config["portfolios"] = [
        item
        for item in config["portfolios"]
        if item["candidate_id"] == "crypto_hft_latency_0"
    ]
    profile = bind_research_configuration(research_profile(CRYPTO_PROFILE_ID), config)
    protocol = profile.protocol(config["scenario_id"], access_window="holdout")
    descriptor = protocol.execution_targets()[0]
    capital = PortfolioCapital(
        descriptor.candidate.candidate_id, 1000.0, "USDT", "linear_perpetual"
    )
    software = production_software_identity()
    layout = StorageLayout(tmp_path)
    window = TimeWindow(**plan["test"])

    def envelope(target_id, upstream):
        spec = next(
            x for x in protocol.target_specs("research run") if x.target_id == target_id
        )
        return _target_envelope_for_spec(
            profile=profile,
            protocol=protocol,
            target_spec=spec,
            data_spec=None,
            research_window={**plan["test"], "access": "holdout"},
            upstream_identities=upstream,
            software_identity=software,
            raw_content_identity=None,
        )

    port_envelope = envelope(descriptor.target_id, {})
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
            expected_funding_identity=funding.identity,
            coverage={"s1": (8_600_000_000, 9_001_000_000)},
            trade_coverage=SimpleNamespace(covering=lambda *_: ("trade",)),
            bounds=((8_600_000_000, 9_000_000_000),),
            on_ledger=emit,
            on_episode=None,
        )
        results.append(result)

    root = write_portfolio_ledger_artifact(
        layout.semantic_artifact_root("crypto", descriptor.target_id),
        envelope=port_envelope,
        produce=produce,
    )
    artifact = read_portfolio_ledger_artifact(
        root, expected_identity=port_envelope.instance_identity
    )
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
    upstream = {descriptor.target_id: port_envelope.instance_identity}
    stat_envelope = envelope(descriptor.statistics_target_id, upstream)
    tables = profile.producer_bindings[descriptor.statistics_target_id].producer(
        layout=layout.with_target_roots({descriptor.target_id: root}),
        protocol_manifest=stat_envelope.payload()["semantic_metadata"][
            "target_contract"
        ]["protocol"],
        time_window=window,
        upstream_identities=upstream,
    )
    stat_root = write_statistics_artifact(
        layout.semantic_artifact_root("crypto", descriptor.statistics_target_id),
        envelope=stat_envelope,
        tables=tables,
        table_group="crypto_portfolio",
    )
    statistic = read_statistics_artifact(
        stat_root, expected_identity=stat_envelope.instance_identity
    )
    with statistic.table("crypto_portfolio_statistics.parquet") as frame:
        assert frame.set_index("metric_id").loc[
            "terminal_nav", "value"
        ] == pytest.approx(998.9)
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
    with pytest.raises(
        PortfolioTargetBlocked, match="portfolio_expected_funding_missing"
    ):
        _execute_portfolio(
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
            expected_funding_identity=funding.identity,
            coverage={"s1": (0, 101_000_000)},
            trade_coverage=SimpleNamespace(covering=lambda *_: ("trade",)),
            bounds=((0, 100_000_000),),
            on_ledger=None,
            on_episode=None,
        )


@pytest.mark.parametrize("missing", ["settlement", "book", "trade"])
def test_forecast_gap_policy_does_not_relax_actual_source_coverage(missing):
    funding = SimpleNamespace(
        identity="f" * 64,
        coverage=() if missing == "settlement" else ((0, 101_000_000),),
    )
    with pytest.raises(PortfolioTargetBlocked, match="coverage_incomplete"):
        _execute_portfolio(
            execution_mode="continuous",
            execute=lambda **_: pytest.fail("unqualified execution"),
            books=(),
            trades=None,
            decisions=(),
            funding=funding,
            expected_funding_identity=funding.identity,
            coverage={} if missing == "book" else {"s1": (0, 101_000_000)},
            trade_coverage=SimpleNamespace(
                covering=lambda *_: () if missing == "trade" else ("trade",)
            ),
            bounds=((0, 100_000_000),),
            on_ledger=None,
            on_episode=None,
        )


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
        config=replace(_maker_config(), latency_ms=100),
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
    plan["execution"]["backtest_modes"] = ["independent_segments"]
    config = research_configuration(
        "crypto",
        "baseline",
        plan=plan,
        development=(
            plan["folds"][0]["validation"]["start"],
            plan["folds"][-1]["validation"]["end"],
        ),
    )
    config["portfolios"] = [
        item
        for item in config["portfolios"]
        if item["candidate_id"] == "crypto_hft_latency_0"
    ]
    profile = bind_research_configuration(research_profile(CRYPTO_PROFILE_ID), config)
    protocol = profile.protocol(config["scenario_id"], access_window="holdout")
    descriptor = protocol.execution_targets()[0]
    capital = PortfolioCapital(
        descriptor.candidate.candidate_id, 1000.0, "USDT", "linear_perpetual"
    )
    software = production_software_identity()
    layout = StorageLayout(tmp_path)

    def make_envelope(target_id, upstream):
        spec = next(
            x for x in protocol.target_specs("research run") if x.target_id == target_id
        )
        return _target_envelope_for_spec(
            profile=profile,
            protocol=protocol,
            target_spec=spec,
            data_spec=None,
            research_window={**plan["test"], "access": "holdout"},
            upstream_identities=upstream,
            software_identity=software,
            raw_content_identity=None,
        )

    envelope = make_envelope(descriptor.target_id, {})
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
            expected_funding_identity=funding.identity,
            coverage=coverage,
            trade_coverage=SimpleNamespace(intervals=lambda: iter(coverage.values())),
            bounds=((8_600_000_000, 9_000_000_000),),
            on_ledger=None,
            on_episode=emit,
        )

    if all_uncovered:
        with pytest.raises(
            PortfolioTargetBlocked, match="portfolio_expected_funding_missing"
        ):
            write_crypto_episode_artifact(
                tmp_path / "episodes", envelope=envelope, produce=produce
            )
        assert not (tmp_path / "episodes" / envelope.instance_identity).exists()
        return
    root = write_crypto_episode_artifact(
        tmp_path / "episodes", envelope=envelope, produce=produce
    )
    summaries = []
    read_crypto_episode_artifact(
        root,
        expected_identity=envelope.instance_identity,
        consume_episode=lambda episode, ledger, batches: summaries.append(
            summarize_crypto_ledger(ledger, batches)
        ),
    )
    assert [s["execution"]["selection_status"] for s in summaries] == [
        "COMPLETE",
        "BLOCKED",
    ]
    assert [s["execution"]["funding_decision_coverage"] for s in summaries] == [
        1.0,
        0.0,
    ]
    assert [s["economics"]["fill_events"] for s in summaries] == [1, 0]
    assert [s["economics"]["initial_capital"] for s in summaries] == [1000.0, 1000.0]

    upstream = {descriptor.target_id: envelope.instance_identity}
    stat_envelope = make_envelope(descriptor.statistics_target_id, upstream)
    tables = profile.producer_bindings[descriptor.statistics_target_id].producer(
        layout=layout.with_target_roots({descriptor.target_id: root}),
        protocol_manifest=stat_envelope.payload()["semantic_metadata"][
            "target_contract"
        ]["protocol"],
        time_window=TimeWindow(**plan["test"]),
        upstream_identities=upstream,
    )
    stat_root = write_statistics_artifact(
        layout.semantic_artifact_root("crypto", descriptor.statistics_target_id),
        envelope=stat_envelope,
        tables=tables,
        table_group="crypto_portfolio",
    )
    statistic = read_statistics_artifact(
        stat_root, expected_identity=stat_envelope.instance_identity
    )
    with statistic.table("crypto_execution_diagnostics.parquet") as frame:
        aggregate = frame.loc[frame.episode_id.eq("all")].set_index("metric_id")
        assert aggregate.loc["funding_selection_status", "text"] == "PARTIAL"
        assert aggregate.loc["funding_decision_coverage", "value"] == 0.5
        assert set(frame.loc[frame.metric_id.eq("selection_status"), "text"]) == {
            "COMPLETE",
            "BLOCKED",
        }
    with statistic.table("crypto_portfolio_statistics.parquet") as frame:
        assert (
            not frame.loc[frame.episode_id.eq("all"), "metric_id"]
            .isin(("sharpe", "net_pnl", "terminal_nav"))
            .any()
        )


def test_forecast_policy_identity_changes_port_stat_preserving_model_prediction(
    tmp_path,
):
    import shutil
    from pathlib import Path

    from quant_research_platform.contracts.targets import ProducerKind
    from quant_research_platform.workflows import research
    from tests.storage.test_software_identity import _identity

    config = research_configuration("crypto", "baseline")
    profile = bind_research_configuration(research_profile(CRYPTO_PROFILE_ID), config)
    protocol = profile.protocol(config["scenario_id"])
    targets = protocol.target_specs("research run")
    package = tmp_path / "package"
    shutil.copytree(
        Path(research.__file__).resolve().parents[1],
        package,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    before = _identity(package)
    owner = package / "markets/implementations/crypto/execution.py"
    owner.write_bytes(owner.read_bytes() + b"\n# Funding gap policy changes.\n")
    after = _identity(package)
    for target in targets:
        binding = profile.producer_bindings[target.target_id]
        changed = research._target_software_identity(
            binding, before
        ) != research._target_software_identity(binding, after)
        if binding.kind in (
            ProducerKind.MODEL,
            ProducerKind.PREDICTION,
            ProducerKind.DATASET,
            ProducerKind.PROCESSED,
            ProducerKind.EVALUATION,
        ):
            assert not changed, target.target_id
        elif binding.kind is ProducerKind.PORTFOLIO:
            assert changed, target.target_id
    descriptor = protocol.execution_targets()[0]

    def port_stat(software):
        identities = []
        upstream = {}
        for target_id in (descriptor.target_id, descriptor.statistics_target_id):
            target = next(item for item in targets if item.target_id == target_id)
            envelope = _target_envelope_for_spec(
                profile=profile,
                protocol=protocol,
                target_spec=target,
                data_spec=None,
                research_window={
                    **config["windows"]["development"],
                    "access": "development",
                },
                upstream_identities=upstream,
                software_identity=software,
                raw_content_identity=None,
            )
            identities.append(envelope.instance_identity)
            upstream = {target_id: envelope.instance_identity}
        return identities

    assert all(
        left != right
        for left, right in zip(port_stat(before), port_stat(after), strict=True)
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
        config=replace(_maker_config(), latency_ms=500),
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
