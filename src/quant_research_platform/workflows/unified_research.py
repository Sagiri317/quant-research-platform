"""Persisted synthetic phase-1 smoke across the approved A-share and Crypto owners."""

from __future__ import annotations

from datetime import date
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd

from quant_research_platform.artifacts.outputs.bundle_validation import (
    read_prediction_output_bundle,
)
from quant_research_platform.contracts.experiments import ExperimentPlan
from quant_research_platform.contracts.ids import MarketId
from quant_research_platform.contracts.time import TimeWindow
from quant_research_platform.markets.implementations.a_share.benchmark import (
    materialize_benchmark_weights,
    normalize_benchmark_history,
)
from quant_research_platform.markets.implementations.a_share.calendar import (
    build_a_share_experiment_fold,
)
from quant_research_platform.markets.implementations.a_share.factors import (
    calculations as factor_calculations,
)
from quant_research_platform.markets.implementations.a_share.factors import (
    definitions as factor_definitions,
)
from quant_research_platform.markets.implementations.crypto.execution import (
    MakerConfig,
    TakerConfig,
)
from quant_research_platform.markets.implementations.crypto.hft import (
    build_crypto_grid_experiment_fold,
)
from quant_research_platform.markets.implementations.crypto.ids import (
    HFT_RESEARCH_DATASET_ID,
)
from quant_research_platform.markets.implementations.crypto.research_scenarios import (
    HFT_MULTI_ASSET_SCOPES,
)
from quant_research_platform.reports.unified_research import (
    build_unified_research_markdown,
)
from quant_research_platform.research.experiments import (
    TrialRecord,
    append_trial_record,
    read_trial_ledger,
)
from quant_research_platform.research.portfolios.a_share import (
    FeeSchedule,
    PortfolioOptimizationConfig,
    optimize_a_share_portfolio,
)
from quant_research_platform.research.statistics import (
    cscv_pbo,
    fama_macbeth,
    fdr_adjust,
    hac_mean,
    model_confidence_set,
    probabilistic_and_deflated_sharpe,
    spa_reality_check,
    stationary_bootstrap_mean_ci,
    stepm_superior_models,
)
from quant_research_platform.runtime.resources import (
    current_runtime_resource_policy,
    runtime_resource_policy_scope,
)
from quant_research_platform.sources.csmar.fundamentals import (
    financial_events_from_interim,
    materialize_pit_fundamentals,
)
from quant_research_platform.sources.csmar.interim import normalize_raw_frame
from quant_research_platform.storage.atomic import atomic_write_json, atomic_write_text
from quant_research_platform.storage.datasets.reading import (
    DatasetReadMode,
    read_dataset,
)
from quant_research_platform.storage.layout import StorageLayout
from quant_research_platform.workflows.crypto_hft import run_crypto_hft_experiment


def run_unified_research_smoke(
    *,
    raw_crypto_root: Path,
    output_root: Path,
) -> dict[str, object]:
    """Execute real repository seams on bounded synthetic inputs; never use holdout."""

    with runtime_resource_policy_scope(explicit_cap_bytes=256 * 1024**2):
        return _run_unified_research_smoke(
            raw_crypto_root=raw_crypto_root,
            output_root=output_root,
        )


def _run_unified_research_smoke(
    *,
    raw_crypto_root: Path,
    output_root: Path,
) -> dict[str, object]:
    crypto_report = run_crypto_hft_experiment(
        output_root=output_root,
        raw_root=raw_crypto_root,
        start_ms=0,
        end_ms=6_000,
        taker_config=TakerConfig(1.0, 1.0, 10, 1.0, 0.0, 500),
        maker_config=MakerConfig(1.0, 2.0, -1.0, 0.0, 0.0, 500),
    )
    a_share, a_plan = _a_share_smoke()
    crypto_plan = _crypto_plan()
    statistics = _statistics_from_crypto_artifacts(output_root, crypto_report)
    ledger = _write_ledgers(
        output_root / "experiment-ledger",
        a_plan=a_plan,
        crypto_plan=crypto_plan,
        crypto_report=crypto_report,
        a_share=a_share,
    )
    payload: dict[str, object] = {
        "schema_id": "qrp-unified-research-smoke",
        "run_id": "unified-research-interview-smoke",
        "mode": "SMOKE_ONLY",
        "data_classification": "SYNTHETIC_DATA",
        "status": {
            "implementation": "IMPLEMENTATION_COMPLETE",
            "data": "REAL_DATA_PENDING",
            "holdout": "HOLDOUT_LOCKED",
            "readiness": "READY_FOR_REAL_DATA",
        },
        "holdout_accessed": False,
        "protocols": {
            "a_share": a_plan.protocol_payload(),
            "crypto": crypto_plan.protocol_payload(),
        },
        "a_share": a_share,
        "crypto": {
            "lineage": crypto_report["lineage"],
            "data_quality": crypto_report["data_quality"],
            "evaluation": crypto_report["evaluation"],
            "mode_evaluations": {
                mode: crypto_report["models"]["oos_evaluation"][f"crypto_hft_{mode}"]
                for mode in HFT_MULTI_ASSET_SCOPES
            },
            "execution": crypto_report["execution"],
        },
        "statistics": statistics,
        "ledger": ledger,
    }
    atomic_write_json(output_root / "unified-research-smoke.json", payload)
    atomic_write_text(
        output_root / "unified-research-interview-demo.md",
        build_unified_research_markdown(payload),
    )
    return payload


def _a_share_smoke() -> tuple[dict[str, object], ExperimentPlan]:
    sessions = pd.date_range("2020-01-01", periods=70, freq="B")
    calendar = pd.DataFrame(
        {"calendar_date": sessions, "source_market_type": 1, "is_trading_day": True}
    )
    fold = build_a_share_experiment_fold(
        calendar,
        fold_id="a-share-fold",
        train=TimeWindow(sessions[0].date(), sessions[19].date()),
        validation=TimeWindow(sessions[40].date(), sessions[44].date()),
        development_test=TimeWindow(sessions[45].date(), sessions[49].date()),
        purge_steps=20,
    )
    plan = ExperimentPlan(
        experiment_id="a-share-smoke",
        market_id=MarketId("a_share"),
        research_dataset_id="a-share-synthetic",
        target_id="forward-return-20d",
        candidate_family_id="fixed-a-share-candidates",
        data_snapshot_id="synthetic-a-share-snapshot",
        input_lineage=("csmar-interim-synthetic", "pit-fundamentals-synthetic"),
        folds=(fold,),
        final_holdout=TimeWindow(sessions[50].date(), sessions[69].date()),
        max_label_lookahead_steps=20,
        selection_metric="validation_rankic",
        selection_direction="maximize",
        candidate_ids=("equal-weight", "ridge", "tree"),
        trial_budget=3,
        random_seed=7,
    )
    interim, disclosures = _financial_inputs()
    events = financial_events_from_interim(
        interim, disclosures, trading_sessions=sessions
    )
    observations = pd.DataFrame(
        {
            "asset_id": ["000001", "000002"],
            "timestamp": [sessions[10], sessions[10]],
        }
    )
    pit = materialize_pit_fundamentals(observations, events)
    factors = factor_calculations.calculate_fundamental_factors(
        pit,
        pd.DataFrame(
            {
                "asset_id": ["000001", "000002"],
                "timestamp": [sessions[10], sessions[10]],
                "market_cap": [200.0, 400.0],
            }
        ),
    )
    benchmark = _benchmark_smoke(sessions)
    assets = pd.DataFrame(
        {
            "asset_id": ["a", "b", "c"],
            "alpha": [0.10, 0.02, -1.0],
            "current_weight": [0.5, 0.4, 0.1],
            "benchmark_weight": [0.5, 0.5, 0.0],
            "specific_variance": [0.01, 0.01, 0.02],
            "adv": [1_000_000.0, 1_000_000.0, 100_000.0],
            "eligible": [True, True, False],
            "can_enter": [True, True, True],
            "can_exit": [True, False, True],
            "beta": [1.0, 1.0, 1.2],
            "size": [0.2, -0.2, 1.0],
        }
    )
    covariance = pd.DataFrame(
        [[0.02, 0.0], [0.0, 0.01]],
        index=["beta", "size"],
        columns=["beta", "size"],
    )
    schedules = (FeeSchedule(date(2020, 1, 1), 2.0, 2.0, 10.0, 0.1),)
    portfolios = {}
    for frequency, cost_penalty in (("1d", 2.0), ("1w", 1.0), ("1m", 0.5)):
        result = optimize_a_share_portfolio(
            assets,
            exposure_columns=("beta", "size"),
            factor_covariance=covariance,
            fee_schedules=schedules,
            trade_date=date(2020, 2, 1),
            config=PortfolioOptimizationConfig(
                frequency,
                1_000_000.0,
                0.7,
                0.2,
                0.4,
                0.1,
                0.2,
                1.0,
                cost_penalty,
                2.0,
                1.0,
            ),
        )
        portfolios[frequency] = {
            "weights": result.weights.to_dict(),
            "active_risk": result.predicted_active_risk,
            "factor_variance": result.factor_variance,
            "specific_variance": result.specific_variance,
            "turnover": result.turnover,
            "estimated_cost": result.estimated_cost,
        }
    weights = np.array(list(portfolios["1w"]["weights"].values()))
    benchmark_weights = assets["benchmark_weight"].to_numpy()
    return_matrix = np.array(
        [[0.01, 0.002, -0.004], [-0.003, 0.001, 0.0], [0.005, 0.003, -0.002]]
    )
    portfolio_returns = return_matrix @ weights
    benchmark_returns = return_matrix @ benchmark_weights
    nav = np.cumprod(1.0 + portfolio_returns)
    active_nav = np.cumprod(1.0 + portfolio_returns - benchmark_returns)
    factor_rows = factors.astype("object").where(factors.notna(), None)
    factor_rows["timestamp"] = factor_rows["timestamp"].map(
        lambda value: pd.Timestamp(value).isoformat()
    )
    return (
        {
            "pit_event_count": len(events),
            "pit_defined_rows": int(pit["value"].notna().sum()),
            "pit_timeline": [
                pd.Timestamp(value).isoformat() for value in events["available_at"]
            ],
            "factor_definition_ids": [
                definition.factor_id
                for definition in factor_definitions.fundamental_factor_definitions()
            ],
            "factor_rows": factor_rows.to_dict("records"),
            "benchmark_ids": sorted(benchmark["benchmark_ids"]),
            "benchmark_weight_rows": benchmark["weight_rows"],
            "portfolios": portfolios,
            "nav": nav.tolist(),
            "active_nav": active_nav.tolist(),
            "drawdown": (nav / np.maximum.accumulate(nav) - 1.0).tolist(),
            "model_status": "REAL_DATA_PENDING",
            "horizon_decay_status": "REAL_DATA_PENDING",
        },
        plan,
    )


def _financial_inputs() -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    frames = {}
    for stock, scale in (("000001", 1.0), ("000002", 2.0)):
        common = {
            "Stkcd": stock,
            "ShortName": "synthetic",
            "IfCorrect": "0",
            "DeclareDate": None,
            "Accper": "2019-12-31",
            "Typrep": "A",
        }
        for data_id, values in (
            (
                "financial_balance",
                {
                    "A001000000": 100.0 * scale,
                    "A002000000": 40.0 * scale,
                    "A003000000": 60.0 * scale,
                },
            ),
            (
                "financial_income",
                {"B001101000": 30.0 * scale, "B002000000": 10.0 * scale},
            ),
            ("financial_cashflow", {"C001000000": 15.0 * scale}),
        ):
            frames.setdefault(data_id, []).append({**common, **values})
    interim = {
        data_id: normalize_raw_frame(pd.DataFrame(rows), data_id)
        for data_id, rows in frames.items()
    }
    disclosures = pd.DataFrame(
        [
            {
                "asset_id": stock,
                "table_id": table,
                "report_period": pd.Timestamp("2019-12-31"),
                "first_disclosure_date": pd.Timestamp("2020-01-03"),
            }
            for stock in ("000001", "000002")
            for table in ("FS_Combas", "FS_Comins", "FS_Comscfd")
        ]
    )
    return interim, disclosures


def _benchmark_smoke(sessions: pd.DatetimeIndex) -> dict[str, object]:
    membership_rows = []
    weight_rows = []
    return_rows = []
    for index_code in ("000905", "000852"):
        for stock, weight in (("000001", 60.0), ("000002", 40.0)):
            membership_rows.append(
                {
                    "Indexcd": index_code,
                    "Chgsmp01": sessions[2],
                    "Chgsmp02": stock,
                    "Chgsmp03": "synthetic",
                    "Chgsmp04": "1",
                    "Chgsmp05": "1",
                    "Chgsmp06": sessions[1],
                    "Chgsmp07": "SSE",
                }
            )
            weight_rows.append(
                {
                    "Indexcd": index_code,
                    "Enddt": sessions[2],
                    "Stkcd": stock,
                    "Constdnme": "synthetic",
                    "Weight": weight,
                }
            )
        return_rows.append(
            {
                "Indexcd": index_code,
                "Idxtrd01": sessions[2],
                "Idxtrd02": 100.0,
                "Idxtrd03": 101.0,
                "Idxtrd04": 99.0,
                "Idxtrd05": 100.5,
                "Idxtrd06": 1.0,
                "Idxtrd07": 1.0,
                "Idxtrd08": 1.0,
                "Idxtrd09": "synthetic",
            }
        )
    history = normalize_benchmark_history(
        membership_interim=normalize_raw_frame(
            pd.DataFrame(membership_rows), "index_membership"
        ),
        weight_interim=normalize_raw_frame(pd.DataFrame(weight_rows), "index_weights"),
        return_interim=normalize_raw_frame(pd.DataFrame(return_rows), "index_market"),
        index_id_mapping={"000905": "csi500", "000852": "csi1000"},
    )
    observations = (
        history["weights"].loc[:, ("benchmark_id", "constituent_asset_id")].copy()
    )
    observations["timestamp"] = sessions[3]
    weights = materialize_benchmark_weights(observations, history["weights"])
    weight_rows = weights.copy()
    for column in ("timestamp", "effective_date"):
        weight_rows[column] = weight_rows[column].map(
            lambda value: pd.Timestamp(value).isoformat()
        )
    return {
        "benchmark_ids": set(history["membership"]["benchmark_id"]),
        "weight_rows": weight_rows.to_dict("records"),
    }


def _crypto_plan() -> ExperimentPlan:
    grid = pd.date_range("1970-01-01", periods=81, freq="100ms")
    fold = build_crypto_grid_experiment_fold(
        grid,
        fold_id="crypto-fold",
        train=TimeWindow(grid[0].to_pydatetime(), grid[34].to_pydatetime()),
        validation=TimeWindow(grid[40].to_pydatetime(), grid[44].to_pydatetime()),
        development_test=TimeWindow(grid[45].to_pydatetime(), grid[60].to_pydatetime()),
        purge_steps=5,
    )
    return ExperimentPlan(
        experiment_id="crypto-smoke",
        market_id=MarketId("crypto"),
        research_dataset_id=HFT_RESEARCH_DATASET_ID,
        target_id="future-mid-return-500ms",
        candidate_family_id="fixed-multi-asset-scopes",
        data_snapshot_id="workflow-run",
        input_lineage=("binance-recorder-run", HFT_RESEARCH_DATASET_ID),
        folds=(fold,),
        final_holdout=TimeWindow(grid[70].to_pydatetime(), grid[80].to_pydatetime()),
        max_label_lookahead_steps=5,
        selection_metric="validation_ic",
        selection_direction="maximize",
        candidate_ids=tuple(HFT_MULTI_ASSET_SCOPES),
        trial_budget=len(HFT_MULTI_ASSET_SCOPES),
        random_seed=7,
    )


def _statistics_from_crypto_artifacts(
    output_root: Path, crypto_report: dict[str, object]
) -> dict[str, object]:
    layout = StorageLayout(output_root)
    resource_policy = current_runtime_resource_policy()
    assert resource_policy is not None
    research = read_dataset(
        layout.research_dataset_root("crypto", HFT_RESEARCH_DATASET_ID),
        mode=DatasetReadMode.MATERIALIZED,
        resource_policy=resource_policy,
    )
    matrices = {}
    for asset_id, modes in {
        "btc_usdt": ("btc_independent", "pooled", "leave_btc_out"),
        "eth_usdt": ("eth_independent", "pooled", "leave_eth_out"),
    }.items():
        target = research.loc[
            research["asset_id"].eq(asset_id),
            ("timestamp", "asset_id", "future_mid_return_500ms"),
        ].dropna()
        losses = None
        for mode in modes:
            output_id = crypto_report["models"]["prediction_outputs"][
                f"crypto_hft_{mode}"
            ]
            with read_prediction_output_bundle(
                output_root / "outputs" / output_id,
                expected_output_id=output_id,
                expected_run_id="workflow-run",
            ) as (manifest, lookup):
                metadata = manifest["metadata"]
                scoped_target = target.loc[
                    target["timestamp"].between(
                        pd.Timestamp(metadata["test_start"]),
                        pd.Timestamp(metadata["test_end"]),
                    )
                ]
                with lookup.values_for_frame(scoped_target, mark_matched=False) as (
                    values,
                    found,
                ):
                    if 0 in found:
                        raise ValueError("unified_prediction_scope_incomplete")
                    series = (
                        np.asarray(values)
                        - scoped_target["future_mid_return_500ms"].to_numpy()
                    ) ** 2
            frame = pd.DataFrame({mode: series}, index=scoped_target["timestamp"])
            losses = frame if losses is None else losses.join(frame, how="inner")
        matrices[asset_id] = losses
    family_results = {
        asset_id: _statistics_for_loss_matrix(losses)
        for asset_id, losses in matrices.items()
    }
    fama_rows = []
    for period, slope in enumerate((0.4, 0.5, 0.6, 0.7), start=1):
        for asset, factor in enumerate((-2.0, -1.0, 1.0, 2.0), start=1):
            fama_rows.append(
                {
                    "timestamp": pd.Timestamp(2020, period, 1),
                    "asset_id": f"asset-{asset}",
                    "return": 0.1 + slope * factor + 0.01 * (asset % 2),
                    "factor": factor,
                }
            )
    fama = fama_macbeth(
        pd.DataFrame(fama_rows),
        return_column="return",
        factor_columns=("factor",),
        max_lags=1,
    )
    return {
        "families": family_results,
        "fama_macbeth": fama.summary.reset_index(names="coefficient").to_dict(
            "records"
        ),
    }


def _statistics_for_loss_matrix(losses: pd.DataFrame) -> dict[str, object]:
    benchmark = losses.iloc[:, 0]
    alternatives = losses.iloc[:, 1:]
    differentials = alternatives.subtract(benchmark, axis=0)
    hac_results = [
        hac_mean(differentials[column].to_numpy(), max_lags=1)
        for column in alternatives
    ]
    bootstrap = stationary_bootstrap_mean_ci(
        differentials.iloc[:, 0].to_numpy(),
        block_size=2,
        repetitions=99,
        seed=7,
    )
    spa = spa_reality_check(
        benchmark, alternatives, block_size=2, repetitions=99, seed=7
    )
    stepm = stepm_superior_models(
        benchmark, alternatives, block_size=2, repetitions=99, seed=7
    )
    mcs = model_confidence_set(losses, block_size=2, repetitions=99, seed=7)
    returns = -losses.to_numpy()
    trial_sharpes = [
        float(np.mean(column) / np.std(column, ddof=1)) for column in returns.T
    ]
    sharpe = probabilistic_and_deflated_sharpe(
        returns[:, int(np.argmax(trial_sharpes))],
        benchmark_sharpe=0.0,
        trial_sharpes=trial_sharpes,
    )
    pbo = cscv_pbo(returns, slice_count=4, max_combinations=6)
    fdr = fdr_adjust([result.p_value for result in hac_results], method="BY")
    return {
        "common_observations": len(losses),
        "candidate_ids": list(losses.columns),
        "hac": [
            {"mean": result.mean, "p_value": result.p_value} for result in hac_results
        ],
        "stationary_bootstrap": {
            "lower": bootstrap.lower,
            "upper": bootstrap.upper,
        },
        "fdr_by": fdr.adjusted_p_values.tolist(),
        "spa": {
            "lower": spa.lower_p_value,
            "consistent": spa.consistent_p_value,
            "upper": spa.upper_p_value,
        },
        "stepm": list(stepm.superior_models),
        "mcs": {
            "included": list(mcs.included),
            "excluded": list(mcs.excluded),
        },
        "psr": sharpe.psr,
        "dsr": sharpe.dsr,
        "pbo": pbo.pbo,
        "pbo_status": pbo.status,
    }


def _write_ledgers(
    root: Path,
    *,
    a_plan: ExperimentPlan,
    crypto_plan: ExperimentPlan,
    crypto_report: dict[str, object],
    a_share: dict[str, object],
) -> dict[str, object]:
    for index, candidate in enumerate(a_plan.candidate_ids, start=1):
        append_trial_record(
            root / "a-share",
            a_plan,
            TrialRecord(
                event_id=f"a-share-{index}",
                attempt_index=index,
                experiment_id=a_plan.experiment_id,
                protocol_hash=a_plan.protocol_hash,
                candidate_family_id=a_plan.candidate_family_id,
                candidate_id=candidate,
                trial_id=f"a-share-trial-{index}",
                fold_id=a_plan.folds[0].fold_id,
                market_id="a_share",
                research_dataset_id=a_plan.research_dataset_id,
                target_id=a_plan.target_id,
                horizon_id="20d",
                common_coverage_id="real-data-pending",
                factor_definition_ids=tuple(a_share["factor_definition_ids"]),
                random_seed=7,
                status="abandoned",
                undefined_reasons={"validation": "real_data_pending"},
                rebalance_assumption="1w",
            ),
        )
    evaluations = crypto_report["models"]["oos_evaluation"]
    for index, mode in enumerate(crypto_plan.candidate_ids, start=1):
        append_trial_record(
            root / "crypto",
            crypto_plan,
            TrialRecord(
                event_id=f"crypto-{index}",
                attempt_index=index,
                experiment_id=crypto_plan.experiment_id,
                protocol_hash=crypto_plan.protocol_hash,
                candidate_family_id=crypto_plan.candidate_family_id,
                candidate_id=mode,
                trial_id=f"crypto-trial-{index}",
                fold_id=crypto_plan.folds[0].fold_id,
                market_id="crypto",
                research_dataset_id=crypto_plan.research_dataset_id,
                target_id=crypto_plan.target_id,
                horizon_id="500ms",
                common_coverage_id="btc-eth-common-workflow-run",
                model_definition_id=f"crypto_hft_{mode}",
                random_seed=7,
                backend_versions={"sklearn": version("scikit-learn")},
                thread_settings={"native_threads": 1},
                status="completed",
                validation_metrics=evaluations[f"crypto_hft_{mode}"],
                artifact_lineage=(
                    crypto_report["models"]["prediction_outputs"][f"crypto_hft_{mode}"],
                ),
            ),
        )
    return {
        "a_share": [
            record.status for record in read_trial_ledger(root / "a-share", a_plan)
        ],
        "crypto": [
            record.status for record in read_trial_ledger(root / "crypto", crypto_plan)
        ],
    }
