"""Requested statistics consume only their actual scientific inputs."""

from dataclasses import asdict
from itertools import combinations
from math import comb

import numpy as np
import pandas as pd
import pytest

from quant_research_platform.artifacts.portfolio_ledger import (
    write_portfolio_ledger_artifact,
)
from quant_research_platform.artifacts.statistics import read_statistics_artifact
from quant_research_platform.contracts.portfolios import (
    PortfolioCandidateSpec,
    PortfolioCapital,
    PortfolioDiagnostic,
    PortfolioLedgerBatch,
    PortfolioValuation,
)
from quant_research_platform.research import statistics
from quant_research_platform.research.portfolios.diagnostics import portfolio_statistics
from quant_research_platform.research.statistics_production import factor_statistics


def _cash_ledger(root, *, candidate=None):
    candidate = candidate or PortfolioCandidateSpec(
        candidate_id="cash",
        selection_group="global",
        construction="quantile_equal_weight",
        prediction_selector="explicit",
        rebalance_schedule="daily",
        top_quantile=0.2,
        benchmark_id=None,
    )
    dates = pd.bdate_range("2024-01-02", periods=4)
    capital = PortfolioCapital(
        candidate_id=candidate.candidate_id,
        initial_capital=1000.0,
        valuation_currency="CNY",
        valuation_method="cash_equity",
    )
    window = {"start": str(dates[0].date()), "end": str(dates[-1].date())}

    def produce(emit):
        for index in range(2):
            entry, end = dates[index + 1 : index + 3]
            emit(
                PortfolioLedgerBatch(
                    sequence=index,
                    trades=(),
                    cashflows=(),
                    valuations=tuple(
                        PortfolioValuation(
                            timestamp_ns=(end if phase == "end" else entry).value,
                            asset_id=asset,
                            phase=phase,
                            quantity=1000.0 if asset == "__cash__" else 0.0,
                            value=1000.0 if asset == "__cash__" else 0.0,
                        )
                        for phase in ("before", "after", "end")
                        for asset in ("__cash__", "asset")
                    ),
                    diagnostics=(
                        PortfolioDiagnostic(
                            asset_id="__cash__",
                            fill_id=None,
                            name="decision_delta_weight",
                            number=-0.1,
                        ),
                        PortfolioDiagnostic(
                            asset_id="__cash__",
                            fill_id=None,
                            name="decision_reason",
                            text="simple_rule",
                        ),
                        PortfolioDiagnostic(
                            asset_id="__cash__",
                            fill_id=None,
                            name="target_notional",
                            number=1000.0,
                        ),
                        PortfolioDiagnostic(
                            asset_id="asset",
                            fill_id=None,
                            name="target_notional",
                            number=0.0,
                        ),
                        PortfolioDiagnostic(
                            asset_id="asset",
                            fill_id=None,
                            name="decision_timestamp",
                            text=dates[index].isoformat(),
                        ),
                        PortfolioDiagnostic(
                            asset_id="asset", fill_id=None, name="adv20", number=100.0
                        ),
                        PortfolioDiagnostic(
                            asset_id="asset",
                            fill_id=None,
                            name="decision_delta_weight",
                            number=0.1,
                        ),
                    ),
                )
            )

    write_portfolio_ledger_artifact(
        root,
        capital=capital,
        metadata={"candidate": asdict(candidate), "window": window},
        produce=produce,
    )
    return window


@pytest.mark.parametrize(
    "analyses", [["performance"], ["capacity"], ["sharpe_inference"], ["cscv"]]
)
def test_ledger_only_analyses_do_not_open_market_inputs(
    tmp_path, monkeypatch, analyses
):
    from quant_research_platform.research.portfolios import production
    from quant_research_platform.storage.datasets import reading

    ledger = tmp_path / "ledger"
    window = _cash_ledger(ledger)

    def unexpected(*args, **kwargs):
        raise AssertionError("unrequested market data read")

    monkeypatch.setattr(reading, "read_dataset", unexpected)
    monkeypatch.setattr(production, "read_dataset", unexpected)
    config = {
        "ledgers": [str(ledger)],
        "window": window,
        "sample_role": "test",
        "analyses": analyses,
        "output": str(tmp_path / "statistics"),
    }
    if "capacity" in analyses:
        config["adv_sensitivities"] = [0.1]
    if "cscv" in analyses:
        config["statistics"] = {"cscv_blocks": 4}
    artifact = read_statistics_artifact(portfolio_statistics(config))
    expected = {
        "performance": {"portfolio_daily.parquet", "portfolio_performance.parquet"},
        "capacity": {
            "portfolio_capacity.parquet",
            "portfolio_capacity_summary.parquet",
        },
        "sharpe_inference": {"portfolio_robustness.parquet"},
        "cscv": {"portfolio_robustness.parquet"},
    }[analyses[0]]
    assert set(artifact._members) == expected
    assert {path.name for path in artifact.root.glob("*.parquet")} == expected
    if analyses == ["capacity"]:
        with artifact.table("portfolio_capacity.parquet") as values:
            assert values.capacity.tolist() == pytest.approx([100.0, 100.0])
    elif analyses == ["performance"]:
        with artifact.table("portfolio_performance.parquet") as values:
            assert values.set_index("metric_id").loc["terminal_nav", "value"] == 1000.0
    else:
        assert artifact.metadata["inference_scope"] == "exploratory"
        assert artifact.metadata["formal_selection_qualification"] == "not_established"
        with artifact.table("portfolio_robustness.parquet") as values:
            assert values.status.eq("UNDEFINED").all()
            assert not values.reason.isin(
                ["test_excluded", "global_validation_selection_reuses_validation"]
            ).any()


@pytest.mark.parametrize("sample_role", [None, "walk_forward_oos"])
def test_rank_ic_only_uses_declared_maturity_and_no_market_cap(
    tmp_path, monkeypatch, sample_role
):
    from quant_research_platform.storage.datasets import reading

    dates = pd.bdate_range("2024-01-02", periods=4)
    frame = pd.DataFrame(
        {
            "timestamp": np.repeat(dates, 4),
            "asset_id": ["a", "b", "c", "d"] * 4,
            "eligible": True,
            "signal": [0.0, 1.0, 2.0, 3.0] * 4,
            "return": [0.0, 0.1, 0.3, 0.2] * 4,
            "mature_at": np.repeat(dates + pd.Timedelta(days=2), 4),
            "unused": ["not numeric"] * 16,
        }
    )
    source = tmp_path / "panel.parquet"
    frame.to_parquet(source, index=False)
    declared = {
        "data_id": "standalone",
        "frequency": "1d",
        "grain": "asset_timestamp",
        "primary_key": ["timestamp", "asset_id"],
        "partition_columns": [],
        "column_roles": {
            "timestamp": "timestamp",
            "asset_id": "key",
            "eligible": "universe",
            "signal": "factor",
            "return": "label",
            "mature_at": "context",
            "unused": "factor",
        },
        "label_availability_columns": {"return": "mature_at"},
    }
    projections = []
    original = reading.iter_dataset_chunks

    def observed(session, **kwargs):
        projections.append(session.physical_columns)
        yield from original(session, **kwargs)

    monkeypatch.setattr(
        "quant_research_platform.research.statistics_production.iter_dataset_chunks",
        observed,
    )
    config = {
        "dataset": {
            "path": str(source),
            "spec": declared,
            "columns": ["signal", "return"],
        },
        "output": str(tmp_path / "out"),
        "window": {"start": str(dates[0].date()), "end": str(dates[-1].date())},
        "label_cutoff": "2024-01-10",
        "sample_role": "test",
        "analyses": ["rank_ic"],
        "evaluation": {"factors": ["signal"], "primary_label": "return"},
        "statistics": {
            "hac_lag": 0,
            "factor_fdr_families": {"signals": ["signal"]},
            "fdr_primary": "BY_5pct",
            "fdr_robustness": "BH_5pct",
        },
    }
    if sample_role is None:
        config.pop("sample_role")
    else:
        config["sample_role"] = sample_role
    artifact = read_statistics_artifact(factor_statistics(config))
    assert artifact.metadata["sample_role"] == (sample_role or "exploratory")
    assert artifact.metadata["inference_scope"] == "exploratory"
    assert artifact.metadata["formal_selection_qualification"] == "not_established"
    assert artifact._members == ("factor_diagnostics.parquet",)
    assert len(projections) == 1 and set(projections[0]) == set(frame) - {"unused"}
    with artifact.table("factor_diagnostics.parquet") as values:
        assert values.observation_count.eq(4).all()
        assert values.fdr_method.tolist() == ["BY", "BH"]
    with pytest.raises(ValueError, match="statistics_table_member_invalid"):
        with artifact.table("fama_macbeth.parquet"):
            pass
    for excluded in ("signal", "return"):
        rejected = {
            **config,
            "dataset": {
                **config["dataset"],
                "columns": [name for name in ("signal", "return") if name != excluded],
            },
            "output": str(tmp_path / ("excluded_" + excluded)),
        }
        with pytest.raises(ValueError, match="statistics_required_columns_excluded"):
            factor_statistics(rejected)
    config["output"] = str(tmp_path / "conflict")
    config["evaluation"]["label_availability_columns"] = {"return": "timestamp"}
    with pytest.raises(ValueError, match="statistics_label_availability_conflict"):
        factor_statistics(config)


def test_cscv_complement_reuse_preserves_exhaustive_result_and_work(monkeypatch):
    rng = np.random.default_rng(901)
    matrix = rng.normal(0.001, 0.01, (103, 7))
    blocks = np.array_split(np.arange(len(matrix)), 8)
    total = 0
    for chosen in combinations(range(8), 4):
        train = np.concatenate([blocks[i] for i in chosen])
        test = np.concatenate([blocks[i] for i in range(8) if i not in chosen])
        a = statistics._candidate_ranks(matrix[train], np.zeros_like(matrix[train]))
        b = statistics._candidate_ranks(matrix[test], np.zeros_like(matrix[test]))
        winner = np.argmax(a)
        rank = 1 + np.sum(b < b[winner])
        total += rank / 8 <= 0.5
    calls = []
    original = statistics._candidate_ranks

    def counted(values, errors):
        calls.append(len(values))
        return original(values, errors)

    monkeypatch.setattr(statistics, "_candidate_ranks", counted)
    result = statistics.cscv_pbo(matrix, slice_count=8)
    assert result.status == "DEFINED"
    assert result.pbo == total / comb(8, 4)
    assert result.combination_count == comb(8, 4)
    assert len(calls) == comb(8, 4)


def test_rank_ic_reuses_only_identical_masks(monkeypatch):
    frame = pd.DataFrame(
        {
            "timestamp": [pd.Timestamp("2024-01-02")] * 5,
            "asset_id": list("abcde"),
            "label": [0.0, 2.0, 1.0, 4.0, 3.0],
            "a": [0.0, 1.0, 2.0, 3.0, 4.0],
            "b": [4.0, 3.0, 2.0, 1.0, 0.0],
            "c": [np.nan, 1.0, 4.0, 3.0, 2.0],
        }
    )
    expected = {
        column: frame[[column, "label"]].dropna().corr(method="spearman").iloc[0, 1]
        for column in ("a", "b", "c")
    }
    calls = []
    original = statistics._average_ranks

    def counted(values):
        calls.append(len(values))
        return original(values)

    monkeypatch.setattr(statistics, "_average_ranks", counted)
    result = statistics.cross_sectional_rank_ic(
        frame, value_columns=("a", "b", "c"), label_column="label"
    )
    assert len(calls) == 5
    for column, value in expected.items():
        assert result[column].iloc[0] == pytest.approx(value, abs=1e-15)


@pytest.mark.parametrize("sample_role", [None, "walk_forward_oos"])
def test_explicit_prediction_family_statistics_need_no_models_or_selection(
    tmp_path, sample_role
):
    import shutil

    from quant_research_platform.research.statistics_production import model_statistics
    from tests.research.test_prediction_runner_model_selection import candidates

    cases, scoring = candidates(tmp_path)
    for case in cases:
        shutil.rmtree(case.root)
        trial = case.root.with_name(case.root.name + ".trial")
        if trial.exists():
            shutil.rmtree(trial)
    frame = pd.read_parquet(scoring["labels"])
    frame.loc[frame.timestamp.eq(pd.Timestamp("2020-01-06")), "label_available_at"] = (
        pd.Timestamp("2020-01-10")
    )
    frame.drop(columns="x").to_parquet(scoring["labels"], index=False)
    del scoring["label_spec"]["column_roles"]["x"]
    config = {
        "dataset": scoring["labels"],
        "input_spec": scoring["label_spec"],
        "output": str(tmp_path / "statistics"),
        "window": scoring["scoring_window"],
        "label_cutoff": "2020-01-10",
        "sample_role": "test",
        "root_seed": 13,
        "family_ids": scoring["candidate_ids"],
        "comparison_model_id": scoring["candidate_ids"][0],
        "evaluation": {"primary_label": "label"},
        "statistics": {"hac_lag": 0, "bootstrap_block": 1, "bootstrap_repetitions": 10},
        "periods": [
            {
                "period_id": "explicit",
                "window": scoring["scoring_window"],
                "predictions": scoring["predictions"],
            }
        ],
    }
    if sample_role is None:
        config.pop("sample_role")
    else:
        config["sample_role"] = sample_role
    artifact = read_statistics_artifact(model_statistics(config))
    assert artifact.metadata["sample_role"] == (sample_role or "exploratory")
    assert artifact.metadata["coverage_basis"] == "persisted_prediction_keys"
    assert artifact.metadata["inference_scope"] == "exploratory"
    assert artifact.metadata["formal_selection_qualification"] == "not_established"
    assert set(artifact._members) == {
        "model_losses.parquet",
        "model_prediction_metrics.parquet",
        "model_comparison.parquet",
        "model_reality_checks.parquet",
        "robustness.parquet",
    }
    with artifact.table("model_losses.parquet") as losses:
        assert len(losses) == 6
        assert losses.groupby("family_id").observation_count.first().to_dict() == {
            "linear_regression": 2,
            "ic_weighted_rank_ensemble": 2,
        }
        assert losses.common_observation_count.eq(2).all()
    excluded_label = {
        **config,
        "dataset": {"path": config["dataset"], "columns": ["timestamp", "asset_id"]},
        "output": str(tmp_path / "excluded_label"),
    }
    with pytest.raises(ValueError, match="statistics_required_columns_excluded"):
        model_statistics(excluded_label)
    broken = {
        **config,
        "output": str(tmp_path / "missing"),
        "family_ids": [*config["family_ids"], "absent"],
    }
    with pytest.raises(
        ValueError, match="statistics_prediction_family_coverage_missing"
    ):
        model_statistics(broken)


@pytest.mark.parametrize(
    "mark,allowed", [(np.nextafter(500.0, np.inf), True), (501.0, False)]
)
def test_nav_continuity_uses_economic_amount_tolerance(mark, allowed):
    from types import SimpleNamespace

    from quant_research_platform.contracts.time import TimeWindow
    from quant_research_platform.research.portfolios.diagnostics import (
        _portfolio_statistics_tables,
    )

    dates = pd.bdate_range("2024-01-02", periods=4)
    capital = PortfolioCapital(
        candidate_id="marked",
        initial_capital=1000.0,
        valuation_currency="CNY",
        valuation_method="cash_equity",
    )
    batches = []
    for index, price in enumerate((500.0, mark)):
        entry, end = dates[index + 1 : index + 3]
        batches.append(
            PortfolioLedgerBatch(
                sequence=index,
                trades=(),
                cashflows=(),
                valuations=tuple(
                    PortfolioValuation(
                        timestamp_ns=(end if phase == "end" else entry).value,
                        asset_id=asset,
                        phase=phase,
                        quantity=500.0 if asset == "__cash__" else 1.0,
                        value=500.0 if asset == "__cash__" else price,
                        mark_price=1.0 if asset == "__cash__" else price,
                    )
                    for phase in ("before", "after", "end")
                    for asset in ("__cash__", "asset")
                ),
                diagnostics=tuple(
                    PortfolioDiagnostic(
                        asset_id=asset, fill_id=None, name=name, number=value
                    )
                    for asset in ("__cash__", "asset")
                    for name, value in (
                        ("target_notional", 500.0 if asset == "__cash__" else price),
                        ("decision_delta_weight", 0.0),
                    )
                ),
            )
        )
    ledger = SimpleNamespace(capital=capital, iter_chunks=lambda: iter(batches))
    candidate = SimpleNamespace(
        candidate_id="marked", benchmark_id=None, risk_style_columns=()
    )

    def run():
        return _portfolio_statistics_tables(
            ledgers={"marked": ledger},
            candidates=(candidate,),
            config={"analyses": ["performance"], "sample_role": "test"},
            time_window=TimeWindow(dates[0], dates[-1]),
            benchmark_opens=pd.DataFrame(),
            calendar=None,
            unified_trading_sessions=None,
            holding_inputs={},
        )

    if allowed:
        result = run()["portfolio_daily"]
        assert len(result) == 2
        assert result.nav_end.iloc[1] == pytest.approx(1000.0, rel=0, abs=1e-9)
    else:
        with pytest.raises(ValueError, match="portfolio_statistics_nav_discontinuity"):
            run()


@pytest.mark.parametrize(
    "analyses", [["performance"], ["annualized"], ["execution"], ["cost_sensitivity"]]
)
def test_crypto_analyses_require_only_selected_policy_and_keep_real_coverage(
    tmp_path, analyses
):
    from quant_research_platform.markets.implementations.crypto.execution import (
        _record_execution_result,
    )
    from quant_research_platform.research.portfolios.accounting import (
        PortfolioEconomics,
    )
    from quant_research_platform.research.portfolios.diagnostics import (
        crypto_statistics,
    )

    capital = PortfolioCapital("crypto", 1000.0, "USDT", "linear_perpetual")
    dates = pd.date_range("2024-01-01", periods=3)

    def produce(emit):
        economic = PortfolioEconomics(capital, "btc_usdt", emit)
        for index, timestamp in enumerate(dates):
            name = (
                "oos_window_start"
                if index == 0
                else "oos_window_end"
                if index == 2
                else "utc_day_boundary"
            )
            number = (
                float(timestamp.value // pd.Timedelta(days=1).value)
                if index == 1
                else 0.0
            )
            economic.mark(
                timestamp.value,
                100.0,
                diagnostics=(
                    PortfolioDiagnostic("btc_usdt", None, name, number=number),
                ),
            )
        _record_execution_result(
            economic,
            {
                "simulation": "descriptive",
                "net_pnl": 0.0,
                "created_orders": 0,
                "filled_orders": 0,
                "selection_status": "COMPLETE",
            },
            dates[-1].value,
            100.0,
        )

    source = write_portfolio_ledger_artifact(
        tmp_path / "crypto",
        capital=capital,
        metadata={
            "window": {"start": dates[0].isoformat(), "end": dates[-1].isoformat()},
            "sample_role": "validation",
            "execution_mode": "continuous",
            "execution_style": "taker",
            "execution": {"position_size": 1.0, "taker_fee_bps": 0.0},
        },
        produce=produce,
    )
    config = {
        "ledger": str(source),
        "output": str(tmp_path / "statistics"),
        "analyses": analyses,
    }
    if analyses == ["annualized"]:
        config["statistics"] = {
            "annualization_minimum_observations": 2,
            "annualization_periods": 365.0,
        }
    elif analyses == ["cost_sensitivity"]:
        config["statistics"] = {
            "fee_scenarios_bps": [0.0, 1.0],
            "additional_spread_scenarios_bps": [0.0],
        }
    artifact = read_statistics_artifact(crypto_statistics(config))
    requested = {
        "performance": {
            "crypto_portfolio_intervals.parquet",
            "crypto_portfolio_daily.parquet",
            "crypto_portfolio_statistics.parquet",
        },
        "annualized": {
            "crypto_portfolio_daily.parquet",
            "crypto_portfolio_statistics.parquet",
        },
        "execution": {"crypto_execution_diagnostics.parquet"},
        "cost_sensitivity": {"crypto_cost_sensitivity.parquet"},
    }[analyses[0]] | {"crypto_portfolio_coverage.parquet"}
    assert set(artifact._members) == requested
    with artifact.table("crypto_portfolio_coverage.parquet") as coverage:
        assert coverage.kind.eq("covered").all()
        assert coverage.start_at.iloc[0] == dates[0]
        assert coverage.stop_at.iloc[-1] == dates[-1] + pd.Timedelta(1, "ns")
    if "crypto_portfolio_statistics.parquet" in requested:
        with artifact.table("crypto_portfolio_statistics.parquet") as values:
            assert ("sharpe" in set(values.metric_id)) == ("annualized" in analyses)


@pytest.mark.parametrize("sample_role", [None, "walk_forward_oos"])
def test_portfolio_statistics_role_is_optional_description(tmp_path, sample_role):
    ledger = tmp_path / "ledger"
    window = _cash_ledger(ledger)
    config = {
        "ledgers": [str(ledger)],
        "window": window,
        "analyses": ["performance"],
        "output": str(tmp_path / "statistics"),
    }
    if sample_role is not None:
        config["sample_role"] = sample_role
    artifact = read_statistics_artifact(portfolio_statistics(config))
    assert artifact.metadata["sample_role"] == (sample_role or "exploratory")
    with artifact.table("portfolio_performance.parquet") as values:
        assert values.set_index("metric_id").loc["terminal_nav", "value"] == 1000.0


@pytest.mark.parametrize("mixed", [False, True])
def test_comparison_only_supports_simple_and_enhanced_without_risk(
    tmp_path, monkeypatch, mixed
):
    from quant_research_platform.research.portfolios import production
    from tests.helpers.benchmark_references import reference_input

    ledger = tmp_path / "cash"
    window = _cash_ledger(ledger)
    reference = reference_input(tmp_path / "index.parquet", [100.0, 110.0, 99.0])
    config = {
        "ledgers": [str(ledger)],
        "window": window,
        "analyses": ["benchmark_comparison"],
        "comparison_benchmarks": [reference],
        "output": str(tmp_path / "statistics"),
    }
    if mixed:
        candidate = PortfolioCandidateSpec(
            candidate_id="enhanced",
            selection_group="global",
            construction="benchmark_enhanced",
            prediction_selector="explicit",
            rebalance_schedule="daily",
            top_quantile=None,
            benchmark_id="csi500",
            risk_style_columns=("size",),
        )
        enhanced = tmp_path / "enhanced"
        assert _cash_ledger(enhanced, candidate=candidate) == window
        config["ledgers"].append(str(enhanced))

    def unexpected(*args, **kwargs):
        raise AssertionError("comparison opened construction or risk inputs")

    monkeypatch.setattr(production, "_iter_input_sessions", unexpected)
    monkeypatch.setattr(production, "_read_materialized", unexpected)
    artifact = read_statistics_artifact(portfolio_statistics(config))
    assert set(artifact._members) == {
        "portfolio_benchmark_intervals.parquet",
        "portfolio_benchmark_metrics.parquet",
    }
    with artifact.table("portfolio_benchmark_intervals.parquet") as intervals:
        for _, rows in intervals.groupby("candidate_id"):
            assert rows.benchmark_return.tolist() == pytest.approx([0.1, -0.1])
            assert rows.return_difference.tolist() == pytest.approx([-0.1, 0.1])
            assert rows.relative_wealth.tolist() == pytest.approx([1 / 1.1, 1 / 0.99])
            assert rows.status.eq("DEFINED").all()
    with artifact.table("portfolio_benchmark_metrics.parquet") as metrics:
        for _, rows in metrics.groupby("candidate_id"):
            selected = rows.set_index("metric_id")
            assert selected.loc["relative_wealth", "value"] == pytest.approx(100 / 99)
            assert selected.loc["tracking_error", "value"] == pytest.approx(
                0.1 * 2**0.5 * 252**0.5
            )
            assert selected.loc["maximum_relative_drawdown", "value"] == pytest.approx(
                1 / 1.1 - 1
            )
