from copy import deepcopy
from dataclasses import asdict

import numpy as np
import pandas as pd
import pytest

from quant_research_platform.artifacts.portfolio_ledger import (
    write_portfolio_ledger_artifact,
)
from quant_research_platform.artifacts.statistics import (
    read_statistics_artifact,
    write_statistics_artifact,
)
from quant_research_platform.contracts.business_manifest import (
    build_business_manifest_envelope,
)
from quant_research_platform.contracts.portfolios import (
    FundingEvent,
    PortfolioCapital,
    PortfolioDiagnostic,
)
from quant_research_platform.contracts.time import TimeWindow
from quant_research_platform.markets.implementations.crypto.execution import (
    _record_execution_result,
)
from quant_research_platform.profiles.research import (
    CRYPTO_PROFILE_ID,
    bind_research_configuration,
    research_profile,
)
from quant_research_platform.research.portfolios.accounting import PortfolioEconomics
from quant_research_platform.storage.identity import (
    filesystem_glob,
    open_filesystem_file,
)
from quant_research_platform.storage.layout import StorageLayout
from quant_research_platform.storage.software_identity import (
    production_software_identity,
)
from quant_research_platform.workflows.research import _target_envelope_for_spec
from tests.helpers.research_configuration import research_configuration


def _statistics_plan(quantity_scale=1.0):
    return {
        "folds": [
            {
                "fold_id": "first",
                "train": {"start": "2023-12-01T00:00:00", "end": "2023-12-10T00:00:00"},
                "validation": {
                    "start": "2023-12-11T00:00:00",
                    "end": "2023-12-20T00:00:00",
                },
            },
            {
                "fold_id": "second",
                "train": {"start": "2023-12-02T00:00:00", "end": "2023-12-20T00:00:00"},
                "validation": {
                    "start": "2023-12-21T00:00:00",
                    "end": "2023-12-31T00:00:00",
                },
            },
        ],
        "test": {"start": "2024-01-01T00:00:00", "end": "2024-01-06T00:00:00"},
        "initial_capital": 1000.0,
        "valuation_currency": "USDT",
        "exposure_limits": {
            asset: {
                "max_order_quantity": quantity_scale,
                "max_position_quantity": 2.0 * quantity_scale,
            }
            for asset in ("btc_usdt", "eth_usdt")
        },
        "execution": {
            "backtest_modes": ["continuous"],
            "taker_fee_bps": 0.0,
            "latency_ms": 0,
            "markout_horizon_ms": 500,
            "latency_scenarios_ms": [0, 100, 200],
            "fee_scenarios_bps": [0.0, 1.0],
            "additional_spread_scenarios_bps": [0.0, 1.0],
            "maker": {
                "maker_fee_bps": 0.0,
                "latency_ms": 0,
                "min_calibration_samples": 1,
                "inventory_target": 0.0,
                "inventory_skew": 0.0,
            },
        },
    }


@pytest.mark.parametrize("explicit", (True, False))
@pytest.mark.parametrize("kind", ("PORTFOLIO", "STATISTIC"))
def test_taker_identity_projects_consumed_latency_and_preserves_real_scenarios(
    explicit, kind
):
    plan = _statistics_plan()
    configuration = research_configuration(
        "crypto",
        "baseline",
        development=(
            plan["folds"][0]["validation"]["start"],
            plan["folds"][-1]["validation"]["end"],
        ),
        plan=plan,
    )
    candidate = next(item for item in configuration["portfolios"] if "model_id" in item)
    candidate["latency_ms"] = 100 if explicit else None
    configuration["portfolios"] = [candidate]
    software = production_software_identity()

    def bind(config):
        return bind_research_configuration(
            research_profile(CRYPTO_PROFILE_ID), config
        ).protocol(config["scenario_id"], access_window="holdout")

    def envelope(protocol):
        descriptor = protocol.execution_targets()[0]
        target_id = (
            descriptor.target_id
            if kind == "PORTFOLIO"
            else descriptor.statistics_target_id
        )
        target = next(
            item
            for item in protocol.target_specs("research run")
            if item.target_id == target_id
        )
        result = _target_envelope_for_spec(
            profile=bind_research_configuration(
                research_profile(CRYPTO_PROFILE_ID), protocol.configuration()
            ),
            protocol=protocol,
            target_spec=target,
            data_spec=None,
            research_window={**plan["test"], "access": "holdout"},
            upstream_identities={},
            software_identity=software,
            raw_content_identity=None,
        )
        return result.instance_identity, protocol.target_manifest(target_id, kind)

    first = bind(configuration)
    changed = deepcopy(configuration)
    changed["research_plan"]["execution"]["latency_ms"] = 200
    second = bind(changed)
    first_identity, first_projection = envelope(first)
    second_identity, second_projection = envelope(second)
    assert (
        len(first.portfolio_candidates())
        == len(second.portfolio_candidates())
        == (1 if explicit else 3)
    )
    assert first_projection["research_plan"]["execution"]["latency_ms"] == (
        100 if explicit else 0
    )
    assert second_projection["research_plan"]["execution"]["latency_ms"] == (
        100 if explicit else 200
    )
    assert (first_identity == second_identity) is explicit
    if explicit:
        assert first_projection == second_projection
        changed["portfolios"][0]["latency_ms"] = 300
        assert envelope(bind(changed))[0] != first_identity
    else:
        assert {item.latency_ms for item in first.portfolio_candidates()} == {
            None,
            100,
            200,
        }
        changed["research_plan"]["execution"]["latency_scenarios_ms"] = [0, 100, 300]
        assert {item.latency_ms for item in bind(changed).portfolio_candidates()} == {
            None,
            0,
            100,
            300,
        }


@pytest.mark.parametrize("quantity_scale", (1.0, 2.0 ** (-50)))
@pytest.mark.parametrize("missing_boundary", (False, True))
def test_sparse_published_ledgers_project_utc_days_gap_and_endpoint_inventory(
    tmp_path, missing_boundary, quantity_scale, monkeypatch
):
    plan = _statistics_plan(quantity_scale)
    plan["execution"]["fee_scenarios_bps"].append(1e308)
    configuration = research_configuration(
        "crypto",
        "baseline",
        development=(
            plan["folds"][0]["validation"]["start"],
            plan["folds"][-1]["validation"]["end"],
        ),
        plan=plan,
    )
    selected = {
        "crypto_hft_latency_0",
        "crypto_hft_eth_independent_eth_usdt",
        "crypto_hft_maker_baseline",
    }
    configuration["portfolios"] = [
        item for item in configuration["portfolios"] if item["candidate_id"] in selected
    ]
    assert {item["candidate_id"] for item in configuration["portfolios"]} == selected
    profile = bind_research_configuration(
        research_profile(CRYPTO_PROFILE_ID), configuration
    )
    bound = profile.protocol(configuration["scenario_id"], access_window="holdout")
    layout = StorageLayout(tmp_path)
    start = pd.Timestamp("2024-01-01T00:00:00").value
    day = pd.Timedelta(days=1).value
    window = TimeWindow("2024-01-01T00:00:00", "2024-01-06T00:00:00")
    research_window = {
        "start": "2024-01-01T00:00:00",
        "end": "2024-01-06T00:00:00",
        "access": "holdout",
    }
    roots, identities = ({}, {})
    for descriptor in bound.execution_targets():
        candidate = descriptor.candidate
        capital = PortfolioCapital(
            candidate.candidate_id, 1000.0, "USDT", "linear_perpetual"
        )
        envelope = build_business_manifest_envelope(
            logical_artifact_id=descriptor.target_id,
            schema_identity="a" * 64,
            protocol_or_spec_identity="b" * 64,
            software_identity="c" * 64,
            research_window=research_window,
            upstream_identities={},
            semantic_metadata={"portfolio_capital": asdict(capital)},
        )

        def produce(emit, capital=capital, candidate=candidate, *, split_windows=False):
            economic = PortfolioEconomics(capital, candidate.asset_id, emit)

            def mark(offset, price, name=None, number=None):
                diagnostics = (
                    ()
                    if name is None
                    else (
                        PortfolioDiagnostic(
                            candidate.asset_id, None, name, number=number
                        ),
                    )
                )
                economic.mark(
                    start + offset, price / quantity_scale, diagnostics=diagnostics
                )

            mark(0, 100.0, "oos_window_start", 0.0)
            economic.fill(
                timestamp_ns=start,
                order_id="first",
                quantity=quantity_scale,
                price=100.0 / quantity_scale,
                mark_price=100.0 / quantity_scale,
                fee=0.0,
            )
            mark(day, 200.0, "utc_day_boundary", float((start + day) // day))
            if split_windows:
                mark(day, 200.0, "oos_window_end", 0.0)
            economic.funding(
                FundingEvent(start + day + day // 2, -0.5, 200.0 / quantity_scale)
            )
            mark(2 * day, 500.0, "utc_day_boundary", float((start + 2 * day) // day))
            if split_windows:
                mark(2 * day, 500.0, "oos_window_start", 1.0)
            mark(
                3 * day,
                500.0,
                None if missing_boundary else "utc_day_boundary",
                None if missing_boundary else float((start + 3 * day) // day),
            )
            mark(4 * day, 500.0, "utc_day_boundary", float((start + 4 * day) // day))
            mark(5 * day, 350.0)
            economic.fill(
                timestamp_ns=start + 5 * day,
                order_id="terminal",
                quantity=quantity_scale,
                price=350.0 / quantity_scale,
                mark_price=350.0 / quantity_scale,
                fee=0.0,
            )
            mark(5 * day, 350.0, "oos_window_end", 1.0 if split_windows else 0.0)
            _record_execution_result(
                economic,
                {
                    "simulation": "sparse_ledger_fixture",
                    "net_pnl": 350.0,
                    "created_orders": 2,
                    "filled_orders": 2,
                    "selection_status": "COMPLETE",
                },
                start + 5 * day,
                350.0 / quantity_scale,
            )

        roots[descriptor.target_id] = write_portfolio_ledger_artifact(
            layout.semantic_artifact_root("crypto", descriptor.target_id),
            envelope=envelope,
            produce=produce,
        )
        identities[descriptor.target_id] = envelope.instance_identity
        from quant_research_platform.artifacts.crypto_hft import summarize_crypto_ledger
        from quant_research_platform.artifacts.portfolio_ledger import (
            read_portfolio_ledger_artifact,
        )

        separated_windows = ((start, start + day), (start + 2 * day, start + 5 * day))
        separated_envelope = build_business_manifest_envelope(
            logical_artifact_id=f"{descriptor.target_id}_separated",
            schema_identity=envelope.schema_identity,
            protocol_or_spec_identity=envelope.protocol_or_spec_identity,
            software_identity=envelope.software_identity,
            research_window=research_window,
            upstream_identities={},
            semantic_metadata={
                "portfolio_capital": asdict(capital),
                "oos_windows_ns": [list(bounds) for bounds in separated_windows],
            },
        )
        separated_root = write_portfolio_ledger_artifact(
            layout.semantic_artifact_root(
                "crypto", separated_envelope.logical_artifact_id
            ),
            envelope=separated_envelope,
            produce=lambda emit: produce(emit, split_windows=True),
        )
        ledger = read_portfolio_ledger_artifact(
            separated_root, expected_identity=separated_envelope.instance_identity
        )
        separated = summarize_crypto_ledger(
            ledger, ledger.iter_chunks(), windows_ns=separated_windows
        )["intervals"]
        gap = next(row for row in separated if row["kind"] == "gap")
        assert gap["funding"] == 100.0 and gap["net_pnl"] == 400.0
    for descriptor in bound.execution_targets():
        target = descriptor.statistics_target_id
        selected_identities = {descriptor.target_id: identities[descriptor.target_id]}
        spec = next(
            spec
            for spec in bound.target_specs("research run")
            if spec.target_id == target
        )
        envelope = _target_envelope_for_spec(
            profile=profile,
            protocol=bound,
            target_spec=spec,
            data_spec=None,
            research_window=research_window,
            upstream_identities=selected_identities,
            software_identity=production_software_identity(),
            raw_content_identity=None,
        )
        from quant_research_platform.artifacts import portfolio_ledger

        scanned = []
        original_batches = portfolio_ledger._iter_batches

        def count_batches(
            root, *args, scanned=scanned, original_batches=original_batches
        ):
            scanned.append(root)
            yield from original_batches(root, *args)

        with monkeypatch.context() as check:
            check.setattr(portfolio_ledger, "_iter_batches", count_batches)
            owned = profile.producer_bindings[target].producer(
                layout=layout.with_target_roots(roots),
                protocol_manifest=envelope.payload()["semantic_metadata"][
                    "target_contract"
                ]["protocol"],
                time_window=window,
                upstream_identities=selected_identities,
            )
        assert scanned == [roots[descriptor.target_id]]
        root = write_statistics_artifact(
            layout.semantic_artifact_root("crypto", target),
            envelope=envelope,
            tables=owned,
            table_group="crypto_portfolio",
        )
        if (
            quantity_scale == 1.0
            and not missing_boundary
            and descriptor == bound.execution_targets()[0]
        ):
            from quant_research_platform.research.portfolios.diagnostics import (
                fixed_turnover_cost_sensitivity,
            )

            unrepresentable = fixed_turnover_cost_sensitivity(
                turnover=1e308,
                net_pnl=0.0,
                base_fee_bps=0.0,
                fee_scenarios_bps=(1e308,),
                additional_spread_scenarios_bps=(0.0,),
            )[0]
            assert unrepresentable["cost_delta"] == np.inf
            invalid = {name: frame.copy() for name, frame in owned.items()}
            for column, value in unrepresentable.items():
                invalid["crypto_cost_sensitivity"].loc[0, column] = value
            with pytest.raises(ValueError, match="statistics_table_numeric_invalid"):
                write_statistics_artifact(
                    StorageLayout(tmp_path / "unrepresentable").semantic_artifact_root(
                        "crypto", target
                    ),
                    envelope=envelope,
                    tables=invalid,
                    table_group="crypto_portfolio",
                )
        artifact = read_statistics_artifact(
            root, expected_identity=envelope.instance_identity
        )
        with artifact.table("crypto_portfolio_statistics.parquet") as statistics:
            with artifact.table("crypto_portfolio_daily.parquet") as daily:
                with artifact.table("crypto_portfolio_intervals.parquet") as intervals:
                    for candidate in (descriptor.candidate,):
                        metrics = statistics.loc[
                            statistics.candidate_id.eq(candidate.candidate_id)
                            & statistics.episode_id.eq("continuous")
                        ].set_index("metric_id")
                        assert metrics.loc["terminal_nav", "value"] == 1350.0
                        assert metrics.loc[
                            "continuous_max_drawdown", "value"
                        ] == pytest.approx(-0.1)
                        assert metrics.loc["gap_net_pnl", "value"] == 0.0
                        assert metrics.loc["oos_net_pnl", "value"] == 350.0
                        assert (
                            metrics.loc["oos_inventory_max_abs", "value"]
                            == 2.0 * quantity_scale
                        )
                        assert metrics.loc[
                            "oos_inventory_mean", "value"
                        ] == pytest.approx(quantity_scale, abs=0.0)
                        assert (
                            metrics.loc["oos_inventory_time_at_limit_fraction", "value"]
                            == 0.0
                        )
                        rows = daily.loc[daily.candidate_id.eq(candidate.candidate_id)]
                        assert len(rows) == 5 and rows.benchmark_return.eq(0).all()
                        assert rows.iloc[-1].start_nav == 1500.0
                        assert rows.iloc[-1].net_return == -0.1
                        if missing_boundary:
                            assert rows.status.eq("DEFINED").sum() == 3
                            assert metrics.loc["sharpe", "status"] == "UNDEFINED"
                            assert (
                                metrics.loc["annualized_volatility", "status"]
                                == "UNDEFINED"
                            )
                            assert (
                                metrics.loc["sharpe", "reason"]
                                == "planned_utc_day_valuation_undefined"
                            )
                        else:
                            expected = np.array([0.1, 400.0 / 1100.0, 0.0, 0.0, -0.1])
                            assert rows.net_return.to_numpy() == pytest.approx(expected)
                            daily_volatility = np.std(expected, ddof=1)
                            assert metrics.loc["sharpe", "value"] == pytest.approx(
                                expected.mean() / daily_volatility * np.sqrt(365)
                            )
                            assert metrics.loc[
                                "annualized_volatility", "value"
                            ] == pytest.approx(daily_volatility * np.sqrt(365))
                        observed = intervals.loc[
                            intervals.candidate_id.eq(candidate.candidate_id)
                        ]
                        assert observed.kind.tolist() == ["oos"]
                        assert observed.funding.iloc[0] == 100.0
                        assert observed.net_pnl.iloc[0] == 350.0
                        from fractions import Fraction

                        with artifact.table("crypto_cost_sensitivity.parquet") as costs:
                            stressed = costs.loc[
                                costs.fee_bps.eq(1e308)
                                & costs.additional_spread_bps.eq(0.0)
                            ]
                            assert len(stressed) == 1
                            expected_cost = float(
                                Fraction(float(observed.turnover.iloc[0]))
                                * Fraction(1e308)
                                / 10000
                            )
                            assert stressed.cost_delta.iloc[0] == expected_cost
                            assert stressed.net_pnl.iloc[0] == 350.0 - expected_cost


@pytest.mark.parametrize("scale", (1.0, 2.0 ** (-50)))
@pytest.mark.parametrize("direction", (-1.0, 1.0))
@pytest.mark.parametrize(
    "fraction,expected_fraction",
    ((0.0, 0.0), (0.5, 0.0), (float.fromhex("0x1.fffffffffffffp-1"), 0.0), (1.0, 1.0)),
)
def test_inventory_limit_duration_has_exact_execution_boundary(
    tmp_path, scale, direction, fraction, expected_fraction
):
    from quant_research_platform.artifacts.portfolio_ledger import (
        read_portfolio_ledger_artifact,
    )
    from quant_research_platform.research.portfolios.diagnostics import (
        _crypto_observations,
    )

    capital = PortfolioCapital("limit", 1000.0, "USDT", "linear_perpetual")
    window = TimeWindow("1970-01-01T00:00:00", "1970-01-01T00:00:10")
    duration = 10000000000
    envelope = build_business_manifest_envelope(
        logical_artifact_id="limit",
        schema_identity="a" * 64,
        protocol_or_spec_identity="b" * 64,
        software_identity="c" * 64,
        research_window={
            "start": window.start.isoformat(),
            "end": window.end.isoformat(),
            "access": "development",
        },
        upstream_identities={},
        semantic_metadata={"portfolio_capital": asdict(capital)},
    )

    def produce(emit):
        economics = PortfolioEconomics(capital, "btc_usdt", emit)
        economics.mark(0, 1.0 / scale)
        if fraction:
            economics.fill(
                timestamp_ns=0,
                order_id="open",
                quantity=direction * fraction * scale,
                price=1.0 / scale,
                mark_price=1.0 / scale,
                fee=0.0,
            )
        economics.mark(duration, 1.0 / scale)

    root = write_portfolio_ledger_artifact(
        tmp_path / "limit", envelope=envelope, produce=produce
    )
    inventory = {}

    def consume(artifact, batches):
        observe, finish = _crypto_observations(artifact, (("test", window),), scale)
        for batch in batches:
            observe(batch)
        inventory.update(finish()[1])

    read_portfolio_ledger_artifact(
        root, expected_identity=root.name, consume_batches=consume
    )
    assert inventory["oos"]["duration"] == duration
    assert inventory["oos"]["at_limit"] / duration == expected_fraction
    assert inventory["gap"]["duration"] == 0


def test_independent_statistics_keep_returns_per_episode_and_reject_combined_nav(
    tmp_path, monkeypatch
):
    from quant_research_platform.artifacts import statistics as statistics_module
    from quant_research_platform.artifacts.crypto_hft import (
        read_crypto_episode_artifact,
        write_crypto_episode_artifact,
    )
    from quant_research_platform.artifacts.research import (
        read_research_report_artifact,
        write_research_run_record,
    )
    from quant_research_platform.contracts.json_codec import canonical_json_hash
    from quant_research_platform.contracts.targets import (
        ProducerKind,
        ResearchRunRecord,
        ResearchRunTargetResult,
        RunStatus,
        TargetStatus,
        unmeasured_operation_evidence,
    )
    from quant_research_platform.interfaces.queries import portfolio_curve
    from quant_research_platform.workflows.operations import report_build

    plan = _statistics_plan()
    plan["execution"]["backtest_modes"] = ["continuous", "independent_segments"]
    plan["test"]["end"] = "2025-01-01T00:00:00"
    configuration = research_configuration(
        "crypto",
        "baseline",
        development=(
            plan["folds"][0]["validation"]["start"],
            plan["folds"][-1]["validation"]["end"],
        ),
        plan=plan,
    )
    configuration["portfolios"] = configuration["portfolios"][:1]
    profile = bind_research_configuration(
        research_profile(CRYPTO_PROFILE_ID), configuration
    )
    bound = profile.protocol(configuration["scenario_id"], access_window="holdout")
    continuous, descriptor = (
        target
        for target in bound.execution_targets()
        if target.candidate.candidate_id
        == configuration["portfolios"][0]["candidate_id"]
    )
    candidate = descriptor.candidate
    layout = StorageLayout(tmp_path)
    window = TimeWindow(**configuration["windows"]["holdout"])
    research_window = {
        "start": window.start.isoformat(),
        "end": window.end.isoformat(),
        "access": "holdout",
    }
    specs = {spec.target_id: spec for spec in bound.target_specs("research run")}
    execution_envelope = _target_envelope_for_spec(
        profile=profile,
        protocol=bound,
        target_spec=specs[descriptor.target_id],
        data_spec=None,
        research_window=research_window,
        upstream_identities={},
        software_identity=production_software_identity(),
        raw_content_identity=None,
    )
    capital = PortfolioCapital(
        candidate.candidate_id, 1000.0, "USDT", "linear_perpetual"
    )
    first = pd.Timestamp(window.start).value
    day, hour = (pd.Timedelta(days=1).value, pd.Timedelta(hours=1).value)

    def produce(emit_episode, *, price_delta=0.0):
        for offset, fee, price in ((0, 1.0, 200.0), (2 * day, 2.0, 150.0)):
            price += price_delta
            start = first + offset

            def ledger(emit, start=start, fee=fee, price=price):
                economics = PortfolioEconomics(capital, candidate.asset_id, emit)
                economics.mark(start, 100.0)
                economics.fill(
                    timestamp_ns=start,
                    order_id="entry",
                    quantity=1.0,
                    price=100.0,
                    mark_price=100.0,
                    fee=fee,
                )
                economics.mark(start + hour // 2, price)
                _record_execution_result(
                    economics,
                    {
                        "simulation": "independent_fixture",
                        "net_pnl": price - 100.0 - fee,
                        "created_orders": 1,
                        "filled_orders": 1,
                    },
                    start + hour // 2,
                    price,
                )

            emit_episode(
                start_ns=start,
                stop_ns=start + hour,
                termination="TRUNCATED",
                stop_reasons=["book_coverage_end"],
                produce=ledger,
            )

    root = write_crypto_episode_artifact(
        layout.semantic_artifact_root("crypto", descriptor.target_id),
        envelope=execution_envelope,
        produce=produce,
    )
    execution_root = root
    upstream = {descriptor.target_id: execution_envelope.instance_identity}
    statistics_spec = specs[descriptor.statistics_target_id]
    envelope = _target_envelope_for_spec(
        profile=profile,
        protocol=bound,
        target_spec=statistics_spec,
        data_spec=None,
        research_window=research_window,
        upstream_identities=upstream,
        software_identity=production_software_identity(),
        raw_content_identity=None,
    )
    owned = profile.producer_bindings[descriptor.statistics_target_id].producer(
        layout=layout.with_target_roots({descriptor.target_id: root}),
        protocol_manifest=envelope.payload()["semantic_metadata"]["target_contract"][
            "protocol"
        ],
        time_window=window,
        upstream_identities=upstream,
        read_episodes=read_crypto_episode_artifact,
    )
    root = write_statistics_artifact(
        layout.semantic_artifact_root("crypto", descriptor.statistics_target_id),
        envelope=envelope,
        tables=owned,
        table_group="crypto_portfolio",
    )
    invalid = owned["crypto_portfolio_statistics"].copy()
    invalid.loc[
        invalid.episode_id.eq("all") & invalid.metric_id.eq("fees"), "metric_id"
    ] = "net_pnl"
    with pytest.raises(ValueError, match="cross_episode_metric"):
        statistics_module._validated_frame(
            invalid,
            member="crypto_portfolio_statistics.parquet",
            envelope=envelope.payload(),
        )
    artifact = read_statistics_artifact(
        root, expected_identity=envelope.instance_identity
    )
    with artifact.table("crypto_portfolio_statistics.parquet") as frame:
        episode_metrics = frame.loc[frame.episode_id.ne("all")]
        assert sorted(
            episode_metrics.loc[episode_metrics.metric_id.eq("net_pnl"), "value"]
        ) == [48.0, 99.0]
        assert episode_metrics.loc[
            episode_metrics.metric_id.eq("initial_capital"), "value"
        ].tolist() == [1000.0, 1000.0]
        all_metrics = frame.loc[frame.episode_id.eq("all")].set_index("metric_id")
        assert all_metrics.loc["fees", "value"] == 3.0
        assert all_metrics.loc["covered_seconds", "value"] == 7200.0
        assert all_metrics.loc["episode_count", "value"] == 2.0
        assert all_metrics.loc["truncated_episode_count", "value"] == 2.0
        assert not {
            "net_pnl",
            "capital_return",
            "terminal_nav",
            "sharpe",
            "annualized_volatility",
            "episode_max_drawdown",
        }.intersection(all_metrics.index)
    with artifact.table("crypto_portfolio_coverage.parquet") as frame:
        assert frame.kind.tolist() == ["covered", "missing", "covered", "missing"]
        assert (
            frame.stop_at.iloc[:-1].to_numpy() == frame.start_at.iloc[1:].to_numpy()
        ).all()
        assert frame.start_at.iloc[0] == pd.Timestamp(window.start)
        assert frame.stop_at.iloc[-1] == pd.Timestamp(window.end) + pd.Timedelta(
            nanoseconds=1
        )
    with artifact.table("crypto_portfolio_daily.parquet") as frame:
        assert frame.empty
    manifest = bound.manifest()
    references = tuple(
        (
            ResearchRunTargetResult(
                target_id=item.target_id,
                producer_kind=kind,
                status=TargetStatus.BUILT,
                instance_identity=identity.instance_identity,
                schema_identity=identity.schema_identity,
                artifact_root=layout.semantic_artifact_root("crypto", item.target_id)
                .relative_to(tmp_path)
                .as_posix(),
            )
            for item, kind, identity in (
                (
                    specs[descriptor.target_id],
                    ProducerKind.PORTFOLIO,
                    execution_envelope,
                ),
                (statistics_spec, ProducerKind.STATISTIC, envelope),
            )
        )
    ) + tuple(
        (
            ResearchRunTargetResult(
                target_id=target,
                producer_kind=kind,
                status=TargetStatus.BLOCKED,
                reason="book_coverage_missing",
            )
            for target, kind in (
                (continuous.target_id, ProducerKind.PORTFOLIO),
                (continuous.statistics_target_id, ProducerKind.STATISTIC),
            )
        )
    )
    run = ResearchRunRecord(
        run_id="run_independent_statistics",
        profile_id=profile.profile_id,
        scenario_id=configuration["scenario_id"],
        protocol=manifest,
        protocol_identity=canonical_json_hash(manifest),
        research_window=research_window,
        software_identity=production_software_identity().identity_hash,
        terminal_targets=(
            continuous.statistics_target_id,
            descriptor.statistics_target_id,
        ),
        target_results=references,
        status=RunStatus.BLOCKED,
        operation_evidence=unmeasured_operation_evidence("fixture_record"),
    )
    write_research_run_record(layout.research_run_path(run.run_id), run)
    curve = portfolio_curve(
        profile=profile,
        layout=layout,
        run_id=run.run_id,
        target_id=descriptor.target_id,
    )
    assert curve["execution_mode"] == "independent_segments"
    assert "points" not in curve
    assert curve["episode_count"] == 2
    for stride in (2, 100):
        sampled = portfolio_curve(
            profile=profile,
            layout=layout,
            run_id=run.run_id,
            target_id=descriptor.target_id,
            stride=stride,
        )
        assert sampled["source_point_count"] == curve["source_point_count"]
        assert sampled["candidate_id"] == curve["candidate_id"]
        assert sampled["initial_capital"] == curve["initial_capital"]
        for actual, complete in zip(
            sampled["episodes"], curve["episodes"], strict=True
        ):
            last = complete["points"][-1]["sequence"]
            expected = [
                point
                for point in complete["points"]
                if point["sequence"] % stride == 0 or point["sequence"] == last
            ]
            assert actual["points"] == expected
            assert actual["boundaries"] == complete["boundaries"]
            for key in (
                "episode_id",
                "start_ns",
                "stop_ns",
                "terminal_valuation_ns",
                "termination",
                "stop_reasons",
                "instance_identity",
            ):
                assert actual[key] == complete[key]
    assert curve["source_point_count"] == curve["displayed_point_count"]
    assert [episode["points"][0]["nav"] for episode in curve["episodes"]] == [
        1000.0,
        1000.0,
    ]
    assert [episode["points"][-1]["nav"] for episode in curve["episodes"]] == [
        1099.0,
        1048.0,
    ]
    import hashlib
    import json
    from io import BytesIO
    from pathlib import Path

    from quant_research_platform.interfaces import queries

    monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path / "matplotlib"))
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    from matplotlib.dates import AutoDateLocator, ConciseDateFormatter
    from matplotlib.ticker import StrMethodFormatter

    notebook_path = (
        Path(__file__).resolve().parents[2]
        / "notebooks"
        / "research-interview-demo.ipynb"
    )
    notebook_source = notebook_path.read_bytes()
    notebook = json.loads(notebook_source)
    data_before = {
        member.relative_to(tmp_path): hashlib.sha256(member.read_bytes()).digest()
        for member in (tmp_path / "data").rglob("*")
        if member.is_file()
    }
    plot = next(cell for cell in notebook["cells"] if cell["id"] == "research-cell-11")
    namespace = {
        "selected_targets": [descriptor.target_id],
        "queries": queries,
        "profile": profile,
        "layout": layout,
        "RUN_ID": run.run_id,
        "NAV_STRIDE": 1,
        "np": np,
        "pd": pd,
        "plt": plt,
        "is_crypto": True,
        "display": lambda value: None,
        "AutoDateLocator": AutoDateLocator,
        "ConciseDateFormatter": ConciseDateFormatter,
        "StrMethodFormatter": StrMethodFormatter,
    }
    exec(
        compile("".join(plot["source"]), "research-interview-demo.ipynb", "exec"),
        namespace,
    )
    figure = namespace["figure"]
    figure.canvas.draw()
    assert len(figure.axes) == 2
    assert [axis.lines[0].get_ydata()[0] for axis in figure.axes] == [1000.0, 1000.0]
    assert [axis.lines[0].get_ydata()[-1] for axis in figure.axes] == [1099.0, 1048.0]
    png = BytesIO()
    figure.savefig(png, format="png")
    assert png.getvalue().startswith(b"\x89PNG\r\n\x1a\n")
    assert notebook_path.read_bytes() == notebook_source
    assert {
        member.relative_to(tmp_path): hashlib.sha256(member.read_bytes()).digest()
        for member in (tmp_path / "data").rglob("*")
        if member.is_file()
    } == data_before
    figure = namespace = curve = None
    report = report_build(profile=profile, layout=layout, run_id=run.run_id, plan=False)
    result = report["target_result"]
    saved = read_research_report_artifact(
        layout.instance_root(
            layout.semantic_artifact_root("crypto", result["target_id"]),
            result["instance_identity"],
        ),
        expected_identity=result["instance_identity"],
        source_run=run,
    )
    assert saved == report["report"]
    candidates = saved["sections"]["portfolio"]["candidates"]
    assert candidates[0]["execution_mode"] == "continuous"
    assert candidates[0]["status"] == "BLOCKED"
    assert candidates[0]["reason"] == "book_coverage_missing"
    assert candidates[1]["status"] == "DEFINED"
    assert [row["economics"]["net_pnl"] for row in candidates[1]["episodes"]] == [
        99.0,
        48.0,
    ]
    assert "economics" not in candidates[1]
    assert all(row["termination"] == "TRUNCATED" for row in candidates[1]["episodes"])
    assert saved["sections"]["portfolio"]["intervals"]["results"]
    assert saved["sections"]["Data/Coverage"]["execution_coverage"]["results"]
    report = saved = candidates = None
    replacement = write_crypto_episode_artifact(
        tmp_path / "replacement",
        envelope=execution_envelope,
        produce=lambda emit: produce(emit, price_delta=10.0),
    )
    read_curve = queries._ledger_curve
    consumed = 0
    replaced = False

    def swap_after_consumption(*args, **kwargs):
        nonlocal replaced, consumed
        curve = read_curve(*args, **kwargs)
        consumed += 1
        source = next(filesystem_glob(replacement, "**/valuations.parquet"))
        target = execution_root / source.relative_to(replacement)
        with open_filesystem_file(source, "rb") as member:
            with open_filesystem_file(target, "r+b") as destination:
                destination.write(member.read())
                destination.truncate()
        replaced = True
        return curve

    monkeypatch.setattr(queries, "_ledger_curve", swap_after_consumption)
    with pytest.raises(ValueError, match="identity|changed"):
        portfolio_curve(
            profile=profile,
            layout=layout,
            run_id=run.run_id,
            target_id=descriptor.target_id,
        )
    assert replaced and consumed == 1


@pytest.mark.parametrize(
    "initial", (1000.0, np.nextafter(0.0, 1.0), np.finfo(float).max)
)
def test_cash_ledger_nav_endpoints_publish_undefined_risk_without_mutation(
    tmp_path, initial
):
    import hashlib

    initial = float(initial)
    plan = _statistics_plan()
    plan["initial_capital"] = initial
    plan["test"]["end"] = "2024-01-03T00:00:00"
    configuration = research_configuration(
        "crypto",
        "baseline",
        development=(
            plan["folds"][0]["validation"]["start"],
            plan["folds"][-1]["validation"]["end"],
        ),
        plan=plan,
    )
    profile = bind_research_configuration(
        research_profile(CRYPTO_PROFILE_ID), configuration
    )
    protocol = profile.protocol(configuration["scenario_id"], access_window="holdout")
    descriptor = protocol.execution_targets()[0]
    candidate = descriptor.candidate
    capital = PortfolioCapital(
        candidate.candidate_id, initial, "USDT", "linear_perpetual"
    )
    layout = StorageLayout(tmp_path)
    window = TimeWindow(**plan["test"])
    research_window = {**plan["test"], "access": "holdout"}
    start = pd.Timestamp(window.start).value
    day = pd.Timedelta(days=1).value
    ledger_envelope = build_business_manifest_envelope(
        logical_artifact_id=descriptor.target_id,
        schema_identity="a" * 64,
        protocol_or_spec_identity="b" * 64,
        software_identity="c" * 64,
        research_window=research_window,
        upstream_identities={},
        semantic_metadata={"portfolio_capital": asdict(capital)},
    )

    def produce(emit):
        economics = PortfolioEconomics(capital, candidate.asset_id, emit)
        economics.mark(
            start,
            1.0,
            diagnostics=(
                PortfolioDiagnostic(
                    candidate.asset_id, None, "oos_window_start", number=0.0
                ),
            ),
        )
        for offset in (day, 2 * day):
            economics.mark(
                start + offset,
                1.0,
                diagnostics=(
                    PortfolioDiagnostic(
                        candidate.asset_id,
                        None,
                        "utc_day_boundary",
                        number=float((start + offset) // day),
                    ),
                ),
            )
        economics.mark(
            start + 2 * day,
            1.0,
            diagnostics=(
                PortfolioDiagnostic(
                    candidate.asset_id, None, "oos_window_end", number=0.0
                ),
            ),
        )
        _record_execution_result(
            economics,
            {
                "simulation": "cash_only",
                "net_pnl": 0.0,
                "created_orders": 0,
                "filled_orders": 0,
                "selection_status": "COMPLETE",
            },
            start + 2 * day,
            1.0,
        )

    root = write_portfolio_ledger_artifact(
        layout.semantic_artifact_root("crypto", descriptor.target_id),
        envelope=ledger_envelope,
        produce=produce,
    )
    before = {
        p.relative_to(root): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in root.rglob("*")
        if p.is_file()
    }
    target = descriptor.statistics_target_id
    identities = {descriptor.target_id: ledger_envelope.instance_identity}
    spec = next(
        spec
        for spec in protocol.target_specs("research run")
        if spec.target_id == target
    )
    envelope = _target_envelope_for_spec(
        profile=profile,
        protocol=protocol,
        target_spec=spec,
        data_spec=None,
        research_window=research_window,
        upstream_identities=identities,
        software_identity=production_software_identity(),
        raw_content_identity=None,
    )
    owned = profile.producer_bindings[target].producer(
        layout=layout.with_target_roots({descriptor.target_id: root}),
        protocol_manifest=envelope.payload()["semantic_metadata"]["target_contract"][
            "protocol"
        ],
        time_window=window,
        upstream_identities=identities,
    )
    output = write_statistics_artifact(
        layout.semantic_artifact_root("crypto", target),
        envelope=envelope,
        tables=owned,
        table_group="crypto_portfolio",
    )
    artifact = read_statistics_artifact(
        output, expected_identity=envelope.instance_identity
    )
    with artifact.table("crypto_portfolio_statistics.parquet") as metrics:
        risk = metrics.loc[
            metrics.metric_id.isin(("annualized_volatility", "sharpe"))
        ].set_index("metric_id")
        assert risk.observation_count.eq(2).all()
        assert risk.loc["sharpe", "status"] == "UNDEFINED"
        if initial == 1000.0:
            assert risk.loc["annualized_volatility", "value"] == 0.0
            assert risk.loc["annualized_volatility", "status"] == "DEFINED"
        else:
            assert risk.status.eq("UNDEFINED").all()
            assert risk.value.isna().all()
            assert risk.reason.eq("return_risk_unresolved").all()
    after = {
        p.relative_to(root): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in root.rglob("*")
        if p.is_file()
    }
    assert before == after
