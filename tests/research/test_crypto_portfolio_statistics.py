from types import SimpleNamespace

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
from quant_research_platform.contracts.portfolios import (
    FundingEvent,
    PortfolioCapital,
    PortfolioDiagnostic,
)
from quant_research_platform.contracts.time import TimeWindow
from quant_research_platform.markets.implementations.crypto.execution import (
    _record_execution_result,
)
from quant_research_platform.research.portfolios.accounting import PortfolioEconomics
from quant_research_platform.research.portfolios.diagnostics import crypto_statistics


def _metadata(plan, candidate, style="taker", mode="continuous"):
    limits = plan["exposure_limits"][candidate.asset_id]
    execution = (
        {
            "order_quantity": limits["max_order_quantity"],
            "position_size": limits["max_position_quantity"],
            "latency_ms": plan["execution"]["latency_ms"],
            "taker_fee_bps": plan["execution"]["taker_fee_bps"],
            "markout_horizon_ms": plan["execution"]["markout_horizon_ms"],
        }
        if style == "taker"
        else {
            "order_quantity": limits["max_order_quantity"],
            "max_position": limits["max_position_quantity"],
            "maker_fee_bps": plan["execution"]["maker"]["maker_fee_bps"],
            "latency_ms": plan["execution"]["maker"]["latency_ms"],
            "inventory_target": 0.0,
            "inventory_skew": 0.0,
            "markout_horizon_ms": plan["execution"]["markout_horizon_ms"],
        }
    )
    return {
        "window": plan["test"],
        "sample_role": "test",
        "chronology": {"folds": plan["folds"], "test": plan["test"]},
        "asset_id": candidate.asset_id,
        "symbol": "BTCUSDT" if candidate.asset_id == "btc_usdt" else "ETHUSDT",
        "execution_mode": mode,
        "execution_style": style,
        "execution": execution,
    }


def _statistics(ledger, output, plan):
    return crypto_statistics(
        {
            "ledger": str(ledger),
            "output": str(output),
            "statistics": {
                "annualization_minimum_observations": 2,
                "annualization_periods": 365.0,
                "fee_scenarios_bps": plan["execution"]["fee_scenarios_bps"],
                "additional_spread_scenarios_bps": plan["execution"][
                    "additional_spread_scenarios_bps"
                ],
            },
        }
    )


def _tables(artifact):
    result = {}
    for member in artifact._members:
        with artifact.table(member) as frame:
            result[member.removesuffix(".parquet")] = frame
    return result


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


@pytest.mark.parametrize("quantity_scale", (1.0, 2.0 ** (-50)))
@pytest.mark.parametrize("missing_boundary", (False, True))
def test_sparse_published_ledgers_project_utc_days_gap_and_endpoint_inventory(
    tmp_path, missing_boundary, quantity_scale, monkeypatch
):
    plan = _statistics_plan(quantity_scale)
    plan["execution"]["fee_scenarios_bps"].append(1e308)
    candidates = [
        (SimpleNamespace(candidate_id=name, asset_id=asset), style)
        for name, asset, style in (
            ("crypto_hft_latency_0", "btc_usdt", "taker"),
            ("crypto_hft_eth_independent_eth_usdt", "eth_usdt", "taker"),
            ("crypto_hft_maker_baseline", "btc_usdt", "maker"),
        )
    ]
    start = pd.Timestamp("2024-01-01T00:00:00").value
    day = pd.Timedelta(days=1).value
    roots = {}
    for candidate, style in candidates:
        capital = PortfolioCapital(
            candidate.candidate_id, 1000.0, "USDT", "linear_perpetual"
        )
        metadata = _metadata(plan, candidate, style)

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

        roots[candidate.candidate_id] = write_portfolio_ledger_artifact(
            tmp_path / candidate.candidate_id,
            capital=capital,
            metadata=metadata,
            produce=produce,
        )
        from quant_research_platform.artifacts.crypto_hft import summarize_crypto_ledger
        from quant_research_platform.artifacts.portfolio_ledger import (
            read_portfolio_ledger_artifact,
        )

        separated_windows = ((start, start + day), (start + 2 * day, start + 5 * day))
        separated_root = write_portfolio_ledger_artifact(
            tmp_path / (candidate.candidate_id + "-separated"),
            capital=capital,
            metadata={
                **metadata,
                "oos_windows_ns": [list(bounds) for bounds in separated_windows],
            },
            produce=lambda emit: produce(emit, split_windows=True),
        )
        ledger = read_portfolio_ledger_artifact(separated_root)
        separated = summarize_crypto_ledger(
            ledger, ledger.iter_chunks(), windows_ns=separated_windows
        )["intervals"]
        gap = next(row for row in separated if row["kind"] == "gap")
        assert gap["funding"] == 100.0 and gap["net_pnl"] == 400.0
    for candidate, _style in candidates:
        target = candidate.candidate_id + "-statistics"
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
            root = _statistics(roots[candidate.candidate_id], tmp_path / target, plan)
        assert scanned == [roots[candidate.candidate_id]]
        artifact = read_statistics_artifact(root)
        owned = _tables(artifact)
        if (
            quantity_scale == 1.0
            and not missing_boundary
            and candidate == candidates[0][0]
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
                    tmp_path / "unrepresentable",
                    metadata=artifact.metadata,
                    tables=invalid,
                    table_group="crypto_portfolio",
                )
        with artifact.table("crypto_portfolio_statistics.parquet") as statistics:
            with artifact.table("crypto_portfolio_daily.parquet") as daily:
                with artifact.table("crypto_portfolio_intervals.parquet") as intervals:
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
                    assert metrics.loc["oos_inventory_mean", "value"] == pytest.approx(
                        quantity_scale, abs=0.0
                    )
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
    metadata = {
        "window": {"start": window.start.isoformat(), "end": window.end.isoformat()}
    }

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
        tmp_path / "limit", capital=capital, metadata=metadata, produce=produce
    )
    inventory = {}

    def consume(artifact, batches):
        observe, finish = _crypto_observations(artifact, (("test", window),), scale)
        for batch in batches:
            observe(batch)
        inventory.update(finish()[1])

    read_portfolio_ledger_artifact(root, consume_batches=consume)
    assert inventory["oos"]["duration"] == duration
    assert inventory["oos"]["at_limit"] / duration == expected_fraction
    assert inventory["gap"]["duration"] == 0


def test_independent_statistics_keep_returns_per_episode_and_reject_combined_nav(
    tmp_path, monkeypatch
):
    from quant_research_platform.artifacts import statistics as statistics_module
    from quant_research_platform.artifacts.crypto_hft import (
        write_crypto_episode_artifact,
    )
    from quant_research_platform.interfaces.queries import portfolio_curve

    plan = _statistics_plan()
    plan["execution"]["backtest_modes"] = ["continuous", "independent_segments"]
    plan["test"]["end"] = "2025-01-01T00:00:00"
    candidate = SimpleNamespace(candidate_id="independent", asset_id="btc_usdt")
    window = TimeWindow(**plan["test"])
    capital = PortfolioCapital(
        candidate.candidate_id, 1000.0, "USDT", "linear_perpetual"
    )
    metadata = _metadata(plan, candidate, mode="independent_segments")
    first = pd.Timestamp(window.start).value
    day, hour = (pd.Timedelta(days=1).value, pd.Timedelta(hours=1).value)
    from quant_research_platform.contracts.portfolios import PortfolioTargetBlocked
    from quant_research_platform.markets.implementations.crypto.portfolio import (
        _market_coverage_check,
    )

    with pytest.raises(PortfolioTargetBlocked, match="book_coverage_incomplete"):
        _market_coverage_check(
            {
                "first": (first, first + hour),
                "second": (first + 2 * day, first + 2 * day + hour),
            },
            SimpleNamespace(covering=lambda *_: ()),
            ((first, pd.Timestamp(window.end).value),),
        )

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

    execution_root = write_crypto_episode_artifact(
        tmp_path / "episodes", capital=capital, metadata=metadata, produce=produce
    )
    root = _statistics(execution_root, tmp_path / "statistics", plan)
    artifact = read_statistics_artifact(root)
    owned = _tables(artifact)
    invalid = owned["crypto_portfolio_statistics"].copy()
    invalid.loc[
        invalid.episode_id.eq("all") & invalid.metric_id.eq("fees"), "metric_id"
    ] = "net_pnl"
    with pytest.raises(ValueError, match="cross_episode_metric"):
        statistics_module._validated_frame(
            invalid,
            member="crypto_portfolio_statistics.parquet",
            metadata=artifact.metadata,
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
        risk = episode_metrics.loc[
            episode_metrics.metric_id.isin(
                ["sharpe", "annualized_volatility", "oos_mean_daily_return"]
            )
        ]
        assert len(risk) == 6 and risk.status.eq("UNDEFINED").all()
        assert risk.reason.eq("insufficient_complete_utc_days").all()
        assert risk.value.isna().all()
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
    curve = portfolio_curve(execution_root, execution_mode="independent_segments")
    assert curve["execution_mode"] == "independent_segments"
    assert "points" not in curve
    assert curve["episode_count"] == 2
    assert all(item["termination"] == "TRUNCATED" for item in curve["episodes"])
    assert all(
        item["stop_reasons"] == ["book_coverage_end"] for item in curve["episodes"]
    )
    for stride in (2, 100):
        sampled = portfolio_curve(
            execution_root,
            execution_mode="independent_segments",
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


@pytest.mark.parametrize(
    "initial,terminal_price",
    (
        (1000.0, None),
        (np.nextafter(0.0, 1.0), None),
        (np.finfo(float).max, None),
        (1e-300, 1e100),
    ),
)
def test_ledger_nav_endpoints_publish_undefined_risk_without_mutation(
    tmp_path, initial, terminal_price
):
    import hashlib

    initial = float(initial)
    plan = _statistics_plan()
    plan["initial_capital"] = initial
    plan["test"]["end"] = "2024-01-03T00:00:00"
    candidate = SimpleNamespace(candidate_id="endpoints", asset_id="btc_usdt")
    capital = PortfolioCapital(
        candidate.candidate_id, initial, "USDT", "linear_perpetual"
    )
    window = TimeWindow(**plan["test"])
    start = pd.Timestamp(window.start).value
    day = pd.Timedelta(days=1).value
    metadata = _metadata(plan, candidate)

    def produce(emit):
        economics = PortfolioEconomics(capital, candidate.asset_id, emit)
        economics.mark(
            start,
            initial if terminal_price is not None else 1.0,
            diagnostics=(
                PortfolioDiagnostic(
                    candidate.asset_id, None, "oos_window_start", number=0.0
                ),
            ),
        )
        if terminal_price is not None:
            economics.fill(
                timestamp_ns=start,
                order_id="entry",
                quantity=1.0,
                price=initial,
                mark_price=initial,
                fee=0.0,
            )
        for offset in (day, 2 * day):
            economics.mark(
                start + offset,
                terminal_price if terminal_price is not None else 1.0,
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
            terminal_price if terminal_price is not None else 1.0,
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
                "net_pnl": 0.0 if terminal_price is None else terminal_price - initial,
                "created_orders": int(terminal_price is not None),
                "filled_orders": int(terminal_price is not None),
                "selection_status": "COMPLETE",
            },
            start + 2 * day,
            terminal_price if terminal_price is not None else 1.0,
        )

    root = write_portfolio_ledger_artifact(
        tmp_path / "ledger",
        capital=capital,
        metadata=metadata,
        produce=produce,
    )
    before = {
        p.relative_to(root): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in root.rglob("*")
        if p.is_file()
    }
    output = _statistics(root, tmp_path / "statistics", plan)
    artifact = read_statistics_artifact(output)
    with artifact.table("crypto_portfolio_statistics.parquet") as metrics:
        risk = metrics.loc[
            metrics.metric_id.isin(("annualized_volatility", "sharpe"))
        ].set_index("metric_id")
        assert risk.observation_count.eq(2 if terminal_price is None else 1).all()
        assert risk.loc["sharpe", "status"] == "UNDEFINED"
        if terminal_price is not None:
            assert risk.status.eq("UNDEFINED").all()
            assert risk.reason.eq("planned_utc_day_valuation_undefined").all()
            indexed = metrics.set_index("metric_id")
            assert indexed.loc["planned_complete_utc_days", "value"] == 2
            assert indexed.loc["defined_complete_utc_days", "value"] == 1
            assert indexed.loc["terminal_nav", "value"] == terminal_price
            assert indexed.loc["net_pnl", "value"] == terminal_price - initial
            assert indexed.loc["oos_mean_daily_return", "status"] == "UNDEFINED"
        elif initial == 1000.0:
            assert risk.loc["annualized_volatility", "value"] == 0.0
            assert risk.loc["annualized_volatility", "status"] == "DEFINED"
        else:
            assert risk.status.eq("UNDEFINED").all()
            assert risk.value.isna().all()
            assert risk.reason.eq("return_risk_unresolved").all()
    with artifact.table("crypto_portfolio_daily.parquet") as daily:
        assert len(daily) == 2
        if terminal_price is not None:
            assert daily.start_nav.tolist() == [initial, terminal_price]
            assert daily.end_nav.tolist() == [terminal_price, terminal_price]
            assert daily.status.tolist() == ["UNDEFINED", "DEFINED"]
            assert daily.reason.tolist() == ["nonfinite_result", ""]
            assert pd.isna(daily.net_return.iloc[0])
            assert daily.net_return.iloc[1] == 0.0
    after = {
        p.relative_to(root): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in root.rglob("*")
        if p.is_file()
    }
    assert before == after


@pytest.mark.parametrize(
    "start_nav,end_nav,expected",
    (
        (1e308, -1e308, -2.0),
        (1e308, 1e-300, -1.0),
        (1e-300, 2e-300, 1.0),
        (1000.0, 1100.0, 0.1),
    ),
)
def test_daily_return_uses_representable_final_ratio(start_nav, end_nav, expected):
    from types import SimpleNamespace

    from quant_research_platform.research.portfolios.diagnostics import (
        _crypto_observations,
    )

    window = TimeWindow("1970-01-01T00:00:00", "1970-01-02T00:00:00")
    ledger = SimpleNamespace(capital=SimpleNamespace(initial_capital=start_nav))
    observe, finish = _crypto_observations(ledger, (("test", window),), 1.0)
    for timestamp, nav, name in (
        (0, start_nav, "oos_window_start"),
        (86400000000000, end_nav, "oos_window_end"),
    ):
        observe(
            SimpleNamespace(
                valuations=(
                    SimpleNamespace(
                        phase="end",
                        timestamp_ns=timestamp,
                        value=nav,
                        quantity=0.0,
                        asset_id="cash",
                    ),
                ),
                diagnostics=(PortfolioDiagnostic("btc_usdt", None, name, number=0.0),),
                trades=(),
            )
        )
    rows, _, _ = finish()
    assert len(rows) == 1
    assert rows[0]["status"] == "DEFINED"
    assert rows[0]["reason"] == ""
    assert rows[0]["net_return"] == expected
