"""Provider bytes to frozen models, actual accounts and native inference tables."""

from dataclasses import asdict

import pandas as pd
import pytest

from quant_research_platform.artifacts.outputs.prediction import read_predictions
from quant_research_platform.artifacts.portfolio_ledger import (
    read_portfolio_ledger_artifact,
    summarize_portfolio_ledger,
)
from quant_research_platform.artifacts.statistics import read_statistics_artifact
from quant_research_platform.research.models.trial_selection import (
    read_selection,
)
from quant_research_platform.research.portfolios.diagnostics import portfolio_statistics
from quant_research_platform.research.portfolios.production import backtest
from quant_research_platform.research.portfolios.risk import FactorRiskConfig
from quant_research_platform.research.statistics_production import model_statistics
from tests.helpers.research_acceptance import run_a_share_acceptance
from tests.helpers.statistics_artifacts import (
    portfolio_statistics_config,
    statistics_config,
)

pytestmark = pytest.mark.complete_chain


@pytest.fixture(scope="module")
def source_chain(tmp_path_factory):
    return run_a_share_acceptance(tmp_path_factory.mktemp("source-acceptance"))


def test_a_share_real_source_model_portfolio_chain(source_chain):
    case = source_chain
    assert set(case.selections) == {"global", "quarterly"}
    assert len(case.final_models) == len(case.test_predictions) == 4
    assert len(set(case.final_models.values())) == 2
    assert len(set(case.test_predictions.values())) == 2
    assert len(case.ledgers) == len(case.enhanced_ledgers) == 2
    for (group, family), path in case.test_predictions.items():
        frame, metadata = read_predictions(path)
        assert metadata["sample_role"] == "test"
        assert frame.timestamp.min() == pd.Timestamp(case.test["start"])
        assert frame.timestamp.max() == pd.Timestamp(case.test["end"])
        assert frame.groupby("timestamp").asset_id.nunique().eq(100).all()
        assert metadata["model"] == str(case.final_models[group, family])


def test_native_enhancement_obeys_capacity_and_postcost_limits(source_chain):
    from quant_research_platform.research.portfolios.optimizer import OptimizerConfig

    limits = OptimizerConfig()
    for path in source_chain.enhanced_ledgers.values():
        ledger = read_portfolio_ledger_artifact(path)
        assert ledger.capital.initial_capital == 2e9
        participation, active = [], []
        optimized = 0
        for batch in ledger.iter_chunks():
            rows = {(item.asset_id, item.name): item for item in batch.diagnostics}
            adv = {
                asset: item.number
                for (asset, name), item in rows.items()
                if name == "adv20"
            }
            participation.extend(
                abs(item.notional) / adv[item.asset_id] for item in batch.trades
            )
            tracking = [
                item.number
                for item in batch.diagnostics
                if item.name == "predicted_tracking_error"
            ]
            if not tracking:
                continue
            optimized += 1
            assert 0 < tracking[0] <= limits.tracking_error_limit + 1e-8
            after = {
                item.asset_id: item.value
                for item in batch.valuations
                if item.phase == "after"
            }
            nav = sum(after.values())
            deviation = [
                after[asset] / nav - item.number
                for (asset, name), item in rows.items()
                if name == "benchmark_weight"
            ]
            assert max(map(abs, deviation)) <= limits.active_name_limit + 1e-8
            active.extend(deviation)
        assert optimized > 0
        assert max(map(abs, active)) > 1e-6
        assert participation and max(participation) > 0
        assert all(value <= 0.05 + 1e-8 for value in participation)


def test_native_turnover_boundary_skips_rebalance_and_keeps_marks(
    source_chain, tmp_path, monkeypatch
):
    from quant_research_platform.research.portfolios import production

    rejected = []
    solve = production.optimize_portfolio

    def observe_solution(assets, **kwargs):
        try:
            return solve(assets, **kwargs)
        except ValueError as error:
            if str(error) == "optimizer_turnover_bounds_infeasible":
                rejected.append((assets.copy(), kwargs["config"].target_turnover_limit))
            raise

    monkeypatch.setattr(production, "optimize_portfolio", observe_solution)
    original = source_chain.enhanced_configs["global"]
    # The real initial allocation is exempt from turnover, so it can fill.
    # Later price drift forces sales larger than the tiny turnover budget.
    name_limit, turnover_limit = 1e-8, 1e-10
    window = {**original["output_window"], "end": "2019-09-10"}
    ledger = read_portfolio_ledger_artifact(
        backtest(
            {
                **original,
                "assumptions": {**original["assumptions"], "initial_nav": 1e9},
                "optimizer_config": {
                    "active_name_limit": name_limit,
                    "target_turnover_limit": turnover_limit,
                },
                "output_window": window,
                "prediction_window": window,
                "predictions": [
                    {**item, "window": window} for item in original["predictions"]
                ],
                "output": str(tmp_path / "infeasible"),
            }
        )
    )
    batches = list(ledger.iter_chunks())
    skipped_dates = set()
    for batch in batches:
        if batch.trades:
            after = {
                item.asset_id: item.value
                for item in batch.valuations
                if item.phase == "after"
            }
            nav = sum(after.values())
            active = [
                after[item.asset_id] / nav - item.number
                for item in batch.diagnostics
                if item.name == "benchmark_weight"
            ]
            assert active and max(map(abs, active)) <= name_limit + 1e-8
        facts = {item.name: item.text for item in batch.diagnostics if item.text}
        if facts.get("decision_status") != "SKIPPED":
            continue
        assert facts["decision_skip_reason"] == "optimizer_turnover_bounds_infeasible"
        assert not batch.trades and not batch.cashflows
        assert batch.valuations
        assert any(item.quantity > 0 for item in batch.valuations)
        skipped_dates.add(pd.Timestamp(batch.valuations[0].timestamp_ns).date())
    assert skipped_dates
    summary = summarize_portfolio_ledger(ledger.capital, iter(batches))
    assert summary["fees"] > 0 and summary["fill_events"] > 0
    assert len(rejected) == len(skipped_dates)
    for assets, configured_limit in rejected:
        assert configured_limit == turnover_limit
        stocks = assets.loc[~assets.is_cash]
        # Fees cannot increase NAV. Even these necessary upper-band sales alone
        # exceed the allowed half-L1 turnover, independent of the solver result.
        minimum_sale = (
            (stocks.current_weight - stocks.benchmark_weight - name_limit)
            .clip(lower=0)
            .sum()
        )
        assert 0.5 * minimum_sale > turnover_limit + 1e-8


def test_native_excess_capital_skips_capacity_infeasible_targets(
    source_chain, tmp_path
):
    original = source_chain.enhanced_configs["global"]
    window = {**original["output_window"], "end": "2019-09-05"}
    # At 1e12, the first cash-funded 0.5% per-name lower bound exceeds 5% ADV.
    # Keep the actual risk model, optimizer, costs and raw-share account path.
    ledger = read_portfolio_ledger_artifact(
        backtest(
            {
                **original,
                "assumptions": {**original["assumptions"], "initial_nav": 1e12},
                "output_window": window,
                "prediction_window": window,
                "predictions": [
                    {**item, "window": window} for item in original["predictions"]
                ],
                "output": str(tmp_path / "capacity-infeasible"),
            }
        )
    )
    batches = list(ledger.iter_chunks())
    assert len(batches) > 1
    for batch in batches:
        facts = {item.name: item.text for item in batch.diagnostics if item.text}
        assert facts["decision_status"] == "SKIPPED"
        assert facts["decision_skip_reason"] == "optimizer_capacity_bounds_infeasible"
        assert not batch.trades and not batch.cashflows
        assert batch.valuations
    summary = summarize_portfolio_ledger(ledger.capital, iter(batches))
    assert summary["fees"] == 0
    assert summary["nav"] == ledger.capital.initial_capital


def test_source_models_reach_paired_native_test_statistics(source_chain, tmp_path):
    case = source_chain
    previous = None
    for group in ("global", "quarterly"):
        config = statistics_config(
            factors=case.factors, sample_role="test", selection_group=group
        )
        config["evaluation"] = {"primary_label": config["evaluation"]["primary_label"]}
        config["statistics"] = {
            key: value
            for key, value in config["statistics"].items()
            if key in {"hac_lag", "bootstrap_block", "bootstrap_repetitions"}
            or key == "group_comparison"
            and group == "quarterly"
        }
        config["statistics"]["bootstrap_repetitions"] = 32
        config.update(
            dataset=str(case.dataset),
            output=str(tmp_path / group),
            window=case.test,
            prediction_id="source_predictions",
            comparison_model_id=next(iter(case.models)),
            periods=[
                {
                    "period_id": "frozen_test",
                    "window": case.test,
                    "selection": str(case.selections[group]),
                    "predictions": [
                        str(case.test_predictions[group, model])
                        for model in case.models
                    ],
                }
            ],
        )
        if previous is not None:
            config["global_statistics"] = str(previous)
        previous = model_statistics(config)
        saved = read_statistics_artifact(previous)
        with saved.table("prediction_diagnostics.parquet") as diagnostics:
            assert diagnostics.observation_count.gt(20).all()
            assert not diagnostics.reason.eq("selection_reuses_evaluation_window").any()
        with saved.table("model_losses.parquet") as losses:
            assert set(losses.family_id) == set(case.models)
            assert losses.period_id.eq("frozen_test").all()
        if group == "quarterly":
            with saved.table("model_group_comparison.parquet") as paired:
                assert set(paired.family_id) == set(case.models)
                assert paired.observation_count.gt(20).all()


def test_source_accounts_reconcile_native_portfolio_statistics(source_chain, tmp_path):
    case = source_chain
    _, defaults = portfolio_statistics_config(sample_role="test")
    policy = {
        key: value
        for key, value in defaults["statistics"].items()
        if key
        in {
            "cscv_blocks",
            "hac_lag",
            "bootstrap_block",
            "bootstrap_repetitions",
            "group_comparison",
        }
    }
    policy["bootstrap_repetitions"] = 32
    policy["group_comparison"].update(
        portfolio_ids=["champion_top20", "champion_enhanced"],
        primary_portfolio_id="champion_top20",
    )
    inputs = {
        **case.portfolio_configs["global"]["inputs"],
        "index_market": str(case.paths["index_market"]),
        "benchmark": str(case.paths["benchmark_weights"]),
        "risk_context": [
            str(case.paths[name])
            for name in (
                "price_volume_facts",
                "market_model_factors",
                "fundamental_factors",
            )
        ],
    }
    paths = [*case.ledgers.values(), *case.enhanced_ledgers.values()]
    output = portfolio_statistics(
        {
            "ledgers": list(map(str, paths)),
            "analyses": [
                "performance",
                "attribution",
                "capacity",
                "sharpe_inference",
                "cscv",
                "group_comparison",
            ],
            "inputs": inputs,
            "output": str(tmp_path / "statistics"),
            "window": case.test,
            "sample_role": "test",
            "benchmark_total_return_indices": case.indices,
            "risk": asdict(FactorRiskConfig()),
            "adv_sensitivities": case.portfolio_configs["global"]["assumptions"][
                "adv_sensitivities"
            ],
            "statistics": policy,
        }
    )
    statistics = read_statistics_artifact(output)
    with statistics.table("portfolio_performance.parquet") as metrics:
        assert set(metrics.candidate_id) == {
            "global_champion_top20",
            "quarterly_champion_top20",
            "global_champion_enhanced",
            "quarterly_champion_enhanced",
        }
        for path in paths:
            ledger = read_portfolio_ledger_artifact(path)
            assert ledger.capital.initial_capital == 2e9
            actual = summarize_portfolio_ledger(ledger.capital, ledger.iter_chunks())
            selected = metrics.loc[
                metrics.candidate_id.eq(ledger.capital.candidate_id)
            ].set_index("metric_id")
            assert selected.loc["cost", "value"] == pytest.approx(actual["fees"])
            assert selected.loc["net_pnl", "value"] == pytest.approx(actual["net_pnl"])
    with statistics.table("portfolio_group_comparison.parquet") as pairs:
        assert set(pairs.portfolio_id) == {"champion_top20", "champion_enhanced"}


@pytest.mark.parametrize(
    "mutation", ["future_selection", "future_maturity", "invalid_rank"]
)
def test_account_rejects_invalid_persisted_prediction_facts(
    source_chain, tmp_path, mutation
):
    import json
    import shutil

    case = source_chain
    config = {**case.portfolio_configs["global"], "output": str(tmp_path / "bad")}
    item = dict(config["predictions"][0])
    copied = tmp_path / "predictions"
    shutil.copytree(item["path"], copied)
    item["path"] = str(copied)
    config["predictions"] = [item]
    if mutation == "invalid_rank":
        frame = pd.read_parquet(copied / "prediction.parquet")
        frame.loc[0, "prediction_rank_score"] = 1e100
        frame.to_parquet(copied / "prediction.parquet", index=False)
        reason = "invalid_prediction_rank_values"
    else:
        manifest = json.loads((copied / "manifest.json").read_text())
        metadata = manifest["metadata"]
        if mutation == "future_selection":
            metadata["model_facts"]["selection_cutoff"] = "2099-01-01"
            reason = "model_selection_after_fit_cutoff"
        else:
            metadata["model_metadata"]["label_available_max"] = "2099-01-01"
            reason = "model_training_label_after_fit_cutoff"
        (copied / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match=reason):
        backtest(config)
    assert not (tmp_path / "bad").exists()


def test_frozen_alpha_schedules_use_real_mature_prediction_returns(
    source_chain, tmp_path
):
    import json

    from quant_research_platform.research.portfolios.accounting import (
        alpha_calibration_intervals,
        frozen_alpha_calibration,
    )
    from quant_research_platform.research.portfolios.production import calibrate
    from quant_research_platform.storage.datasets import read_dataset

    case = source_chain
    calendar = read_dataset(
        case.paths["trading_calendar"], columns=("calendar_date", "is_trading_day")
    )
    sessions = pd.DatetimeIndex(
        calendar.loc[calendar.is_trading_day, "calendar_date"]
    ).sort_values()
    sessions = sessions[sessions >= pd.Timestamp(case.validation["start"])]
    prices = (
        read_dataset(
            case.paths["market_panel_1d"], columns=("timestamp", "asset_id", "adj_open")
        )
        .set_index(["timestamp", "asset_id"])
        .adj_open
    )
    counts = {}
    for group in ("global", "quarterly"):
        champion = read_selection(case.selections[group])["selected_model_id"]
        predictions = read_predictions(case.validation_predictions[champion])[0]
        for schedule in ("weekly-last-trading-day", "monthly-last-trading-day"):
            intervals = alpha_calibration_intervals(sessions, schedule)
            joined = predictions.merge(
                intervals, on="timestamp", how="inner", validate="many_to_one"
            )
            entry = prices.reindex(
                pd.MultiIndex.from_frame(joined[["entry_at", "asset_id"]])
            ).to_numpy()
            end = prices.reindex(
                pd.MultiIndex.from_frame(joined[["available_at", "asset_id"]])
            ).to_numpy()
            observations = joined[["timestamp", "prediction_rank_score"]].assign(
                holding_return=end / entry - 1
            )
            root = tmp_path / group / schedule
            root.mkdir(parents=True)
            observations.to_parquet(root / "observations.parquet", index=False)
            intervals.to_parquet(root / "intervals.parquet", index=False)
            config = {
                "observations": str(root / "observations.parquet"),
                "scheduled_intervals": str(root / "intervals.parquet"),
                "rebalance_schedule": schedule,
                "calendar": {
                    "path": str(case.paths["trading_calendar"]),
                    "time_window": {
                        "start": case.validation["start"],
                        "end": case.formal["end"],
                    },
                },
                "cutoff": case.validation["end"],
                "selection_group": group,
                "output": str(root / "calibration.json"),
            }
            output = calibrate(config)
            payload = json.loads(output.read_text())
            frozen, cutoff = frozen_alpha_calibration(
                payload,
                cutoff=case.validation["end"],
                selection_group=group,
                rebalance_schedule=schedule,
            )
            assert frozen.coverage >= 0.8 and frozen.valid_rebalance_dates >= 6
            assert pd.Timestamp(cutoff) == pd.Timestamp(case.validation["end"])
            counts[group, schedule] = frozen.valid_rebalance_dates
            # Raw future returns exist, but are unavailable at freeze.
            future = joined.available_at.gt(case.validation["end"])
            assert future.any()
            observations.loc[future.to_numpy(), "holding_return"] += 1000
            observations.to_parquet(root / "future-observations.parquet", index=False)
            perturbed = calibrate(
                {
                    **config,
                    "observations": str(root / "future-observations.parquet"),
                    "output": str(root / "future-calibration.json"),
                }
            )
            assert (
                json.loads(perturbed.read_text())["calibration"]
                == payload["calibration"]
            )
    for group in ("global", "quarterly"):
        assert (
            counts[group, "weekly-last-trading-day"]
            > counts[group, "monthly-last-trading-day"]
            >= 6
        )
