"""Explicit A-share source-to-account acceptance with real production operations."""

import json
from dataclasses import asdict
from math import ceil
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from quant_research_platform.artifacts.outputs.prediction import read_predictions
from quant_research_platform.artifacts.portfolio_ledger import (
    read_portfolio_ledger_artifact,
    summarize_portfolio_ledger,
)
from quant_research_platform.markets.implementations.a_share.processed.catalog import (
    processed_catalog,
)
from quant_research_platform.markets.prepare import prepare
from quant_research_platform.research.datasets.assembler import assemble
from quant_research_platform.research.models.persisted import load_model
from quant_research_platform.research.models.production import fit
from quant_research_platform.research.models.trial_selection import (
    read_selection,
    score,
    select,
)
from quant_research_platform.research.portfolios.production import backtest, calibrate
from quant_research_platform.research.predictions.runner import predict
from quant_research_platform.research.protocols import a_share
from quant_research_platform.sources.prepare import prepare_source
from quant_research_platform.storage.datasets import read_dataset
from tests.helpers.a_share_raw import write_a_share_raw_fixture
from tests.helpers.paths import StorageLayout
from tests.helpers.research_configuration import research_configuration


def prepare_a_share_acceptance(root: Path):
    """Normalize provider-shaped bytes and explicitly prepare each chosen input."""
    root = Path(root)
    indices = {"csi500": "H00905", "csi1000": "H00852"}
    write_a_share_raw_fixture(
        StorageLayout(root),
        benchmark_total_return_indices=indices,
        # Supply enough earlier risk exposures for 378 fresh factor dates
        # within a complete 504-session history before the unchanged test split.
        formal_dataset_start="2017-07-01",
        history_end="2019-10-31",
        validation_start="2019-01-01",
    )
    processed_names = (
        "market_panel_1d",
        "execution_mask",
        "price_volume_facts",
        "forward_returns",
        "market_model_factors",
        "fundamental_pit",
        "fundamental_factors",
        "benchmark_weights",
    )
    catalog = processed_catalog()
    required_sources = {
        name
        for target in processed_names
        for name in catalog.get(target).source_input_data_ids
    } | {"a_share_fee_schedule", "index_market", "corporate_actions"}
    source_groups = {
        "choice": {
            "financial_balance",
            "financial_income",
            "financial_cashflow",
            "financial_first_disclosure",
        },
        "reference": {"a_share_fee_schedule", "a_share_industry_snapshots"},
    }
    source_groups["csmar"] = required_sources - set.union(*source_groups.values())
    paths = {}
    for source, names in source_groups.items():
        outputs = {name: str(root / "normalized" / name) for name in sorted(names)}
        receipt = prepare_source(
            {
                "source_id": source,
                "raw_root": str(root / "data/raw" / source),
                "outputs": outputs,
            }
        )
        assert not receipt.skipped, receipt.skipped
        paths.update({name: Path(path) for name, path in outputs.items()})
    history = {"start": "2016-04-21", "end": "2019-10-31"}
    for name in processed_names:
        spec = catalog.get(name)
        selected = (*spec.source_input_data_ids, *spec.processed_input_data_ids)
        options = {}
        if name in {"fundamental_pit", "fundamental_factors"}:
            options["factor_columns"] = ["earnings_yield"]
            selected = tuple(
                key
                for key in selected
                if key not in {"financial_balance", "financial_cashflow"}
            )
        paths[name] = prepare(
            {
                **options,
                "market_id": "a_share",
                "data_id": name,
                "inputs": {key: str(paths[key]) for key in selected},
                "output_root": str(root / "prepared" / name),
                "time_window": history,
            }
        )
    risk_rows = read_dataset(
        paths["market_model_factors"],
        columns=("timestamp", "asset_id", "beta_252d", "residual_volatility_60d"),
    )
    calendar = read_dataset(paths["trading_calendar"])
    sessions = pd.DatetimeIndex(
        calendar.loc[
            calendar.source_market_type.eq(4)
            & calendar.is_trading_day
            & calendar.calendar_date.between(history["start"], history["end"]),
            "calendar_date",
        ].sort_values()
    )
    # All 100 fixture assets predate the supplied calendar and quote every day.
    # Market eligibility first becomes provable on session 252. The default
    # 75% policy then needs 189 paired returns and 45 fresh residuals, while
    # both rolling histories must also have their full nominal session spans.
    listing_sessions, beta_window, residual_window = 252, 252, 60
    first_beta = max(beta_window - 1, listing_sessions + ceil(0.75 * beta_window) - 2)
    first_risk = max(residual_window - 1, first_beta + ceil(0.75 * residual_window) - 1)
    for column, first in (
        ("beta_252d", first_beta),
        ("residual_volatility_60d", first_risk),
    ):
        assert (
            risk_rows.loc[risk_rows.timestamp.lt(sessions[first]), column].isna().all()
        )
        first_cross = risk_rows.loc[risk_rows.timestamp.eq(sessions[first])]
        assert first_cross.asset_id.nunique() == 100
        assert first_cross[column].notna().all()
    complete_crosses = risk_rows.dropna().groupby("timestamp").asset_id.nunique()
    assert (
        complete_crosses.loc[complete_crosses.eq(100)].index.min()
        == sessions[first_risk]
    )
    factors = research_configuration()["evaluation"]["factors"]
    dataset = assemble(
        {
            "market_id": "a_share",
            "dataset_id": "pv_all",
            "inputs": {
                key: str(paths[key])
                for key in ("market_panel_1d", "price_volume_facts", "forward_returns")
            },
            "output_root": str(root / "assembled"),
            "time_window": {"start": "2017-10-02", "end": "2019-10-31"},
            "factor_columns": factors,
            "label_columns": ["model_target", "forward_return_20d"],
        }
    )
    return SimpleNamespace(
        root=root,
        paths=paths,
        dataset=dataset,
        history=history,
        formal={"start": "2017-10-02", "end": "2019-10-31"},
        indices=indices,
        factors=factors,
    )


def run_a_share_acceptance(project_root: Path):
    """Exercise saved models, complete selection, frozen test and native accounts."""
    case = prepare_a_share_acceptance(project_root)
    train = {"start": "2017-10-02", "end": "2018-12-31"}
    validation = {"start": "2019-01-01", "end": "2019-08-30"}
    test = {"start": "2019-09-02", "end": "2019-10-31"}
    model_specs = [
        dict(item)
        for item in research_configuration()["models"]
        if item["model_algorithm"] in {"equal_weight_rank_ensemble", "ridge_regression"}
    ]
    models, predictions, configurations = {}, {}, {}
    for model in model_specs:
        candidates = model.pop("parameter_candidates", None)
        config = {
            "market_id": "a_share",
            "inputs": str(case.dataset),
            "model": model,
            "output": str(case.root / "models" / model["model_id"]),
            "fit_windows": [train],
            "fit_cutoff": train["end"],
        }
        if candidates is not None:
            config["candidate_parameters"] = candidates[0]
        identifier = model["model_id"]
        configurations[identifier] = config
        models[identifier] = fit(config)
        predictions[identifier] = predict(
            {
                "model": str(models[identifier]),
                "inputs": str(case.dataset),
                "output": str(case.root / "validation" / identifier),
                "prediction_window": validation,
                "sample_role": "validation",
            }
        )
    case.models, case.validation_predictions = models, predictions
    case.test, case.validation = test, validation
    case.selections, case.final_models, case.test_predictions, case.ledgers = (
        {},
        {},
        {},
        {},
    )
    case.enhanced_ledgers, case.portfolio_configs, case.enhanced_configs = {}, {}, {}
    for group in ("global", "quarterly"):
        scored = score(
            {
                "predictions": list(map(str, predictions.values())),
                "labels": str(case.dataset),
                "candidate_ids": list(models),
                "validation_windows": [validation],
                "scoring_window": validation,
                "cutoff": validation["end"],
                "test_window": test,
                "selection_group": group,
                "validation_axis": "cross_sectional",
                "output": str(case.root / group / "scores.json"),
            }
        )
        refit_window = {"start": train["start"], "end": validation["end"]}
        selected = select(
            {
                "scores": str(scored),
                "output": str(case.root / group / "selection.json"),
                "selection_metric": "validation_rankic",
                "selection_direction": "maximize",
                "tie_break_order": list(models),
                "refit_windows": [refit_window],
                "refit_cutoff": validation["end"],
            }
        )
        case.selections[group] = selected
        decision = read_selection(selected)
        assert {row["family_id"] for row in decision["families"]} == set(models)
        for identifier, original in configurations.items():
            final = fit(
                {
                    **original,
                    "output": str(case.root / group / "final" / identifier),
                    "fit_windows": [refit_window],
                    "fit_cutoff": validation["end"],
                    "test_window": test,
                }
            )
            _, facts = load_model(final)
            assert facts["fit_windows"] == [refit_window]
            predicted = predict(
                {
                    "model": str(final),
                    "inputs": str(case.dataset),
                    "output": str(case.root / group / "test" / identifier),
                    "prediction_window": test,
                    "sample_role": "test",
                }
            )
            case.final_models[group, identifier] = final
            case.test_predictions[group, identifier] = predicted
        champion = decision["selected_model_id"]
        inputs = {
            "dataset": str(case.dataset),
            **{
                role: str(case.paths[name])
                for role, name in {
                    "panel": "market_panel_1d",
                    "execution": "execution_mask",
                    "calendar": "trading_calendar",
                    "fees": "a_share_fee_schedule",
                    "market_bars": "market_bars_1d",
                    "stock_master": "stock_master",
                    "corporate_actions": "corporate_actions",
                }.items()
            },
        }
        # Keep compared accounts equal and reachable from cash at the 5% ADV cap.
        config = {
            "candidate": {
                "candidate_id": group + "_champion_top20",
                "selection_group": group,
                "construction": "quantile_equal_weight",
                "prediction_selector": "champion",
                "top_quantile": 0.2,
                "rebalance_schedule": "weekly-last-trading-day",
                "benchmark_id": None,
            },
            "inputs": inputs,
            "predictions": [
                {
                    "path": str(case.test_predictions[group, champion]),
                    "window": test,
                }
            ],
            "output": str(case.root / group / "ledger"),
            "output_window": test,
            "input_window": {"start": "2019-08-01", "end": "2019-10-31"},
            "formal_dataset_window": case.formal,
            "prediction_window": test,
            "sample_role": "test",
            "assumptions": {
                **asdict(a_share.EXECUTION_ASSUMPTIONS),
                "initial_nav": 2e9,
            },
        }
        config["assumptions"]["adv_sensitivities"] = list(
            config["assumptions"]["adv_sensitivities"]
        )
        ledger_root = backtest(config)
        ledger = read_portfolio_ledger_artifact(ledger_root)
        summary = summarize_portfolio_ledger(ledger.capital, ledger.iter_chunks())
        assert summary["fill_events"] > 0 and summary["fees"] > 0 and summary["nav"] > 0
        assert ledger.metadata["initial_holding_convention"] == "cash"
        case.ledgers[group] = ledger_root
        case.portfolio_configs[group] = config
        enhanced = {
            **config,
            "input_window": case.history,
            "candidate": {
                **config["candidate"],
                "candidate_id": group + "_champion_enhanced",
                "construction": "benchmark_enhanced",
                "benchmark_id": "csi500",
                "top_quantile": None,
                "risk_style_columns": list(a_share.RISK_STYLE_COLUMNS),
            },
            "inputs": {
                **inputs,
                "benchmark": str(case.paths["benchmark_weights"]),
                "risk_context": [
                    str(case.paths[name])
                    for name in (
                        "price_volume_facts",
                        "market_model_factors",
                        "fundamental_factors",
                    )
                ],
            },
        }
        observations, intervals = source_alpha_observations(
            case, predictions[champion], "weekly-last-trading-day"
        )
        calibration_root = case.root / group / "alpha"
        calibration_root.mkdir()
        observations.to_parquet(calibration_root / "observations.parquet", index=False)
        intervals.to_parquet(calibration_root / "intervals.parquet", index=False)
        calibration = calibrate(
            {
                "observations": str(calibration_root / "observations.parquet"),
                "scheduled_intervals": str(calibration_root / "intervals.parquet"),
                "rebalance_schedule": "weekly-last-trading-day",
                "calendar": {
                    "path": str(case.paths["trading_calendar"]),
                    "time_window": {
                        "start": validation["start"],
                        "end": case.formal["end"],
                    },
                },
                "cutoff": validation["end"],
                "selection_group": group,
                "output": str(calibration_root / "calibration.json"),
            }
        )
        frozen = json.loads(calibration.read_text())
        assert frozen["calibration"]["valid_rebalance_dates"] >= 6
        assert frozen["calibration"]["coverage"] >= 0.8
        enhanced["calibration"] = str(calibration)
        enhanced["output"] = str(case.root / group / "enhanced-ledger")
        case.enhanced_configs[group] = enhanced
        enhanced_root = backtest(enhanced)
        account = read_portfolio_ledger_artifact(enhanced_root)
        diagnostics = [
            item for batch in account.iter_chunks() for item in batch.diagnostics
        ]
        assert any(
            item.name == "decision_reason" and item.text == "enhanced_optimizer"
            for item in diagnostics
        )
        assert {item.number for item in diagnostics if item.name == "alpha_slope"} == {
            frozen["calibration"]["slope"]
        }
        assert frozen["calibration"]["slope"] > 0
        assert account.metadata["initial_holding_convention"] == "cash"
        assert (
            summarize_portfolio_ledger(account.capital, account.iter_chunks())[
                "fill_events"
            ]
            > 0
        )
        case.enhanced_ledgers[group] = enhanced_root
    return case


def source_alpha_observations(case, prediction_path, schedule):
    """Pair actual native predictions and execution opens on the declared schedule."""
    from quant_research_platform.research.portfolios.accounting import (
        alpha_calibration_intervals,
    )

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
    intervals = alpha_calibration_intervals(sessions, schedule)
    predictions = read_predictions(prediction_path)[0]
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
    return observations, intervals
