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
from quant_research_platform.contracts.ids import MarketId, SourceId
from quant_research_platform.contracts.time import TimeWindow
from quant_research_platform.markets.implementations.a_share.benchmark import (
    materialize_benchmark_weights,
    normalize_benchmark_history,
)
from quant_research_platform.markets.implementations.a_share.calendar import (
    build_a_share_experiment_fold,
)
from quant_research_platform.markets.implementations.a_share.factors import (
    definitions as factor_definitions,
)
from quant_research_platform.markets.implementations.a_share.ids import (
    FUNDAMENTAL_FORWARD_RETURNS_DATASET_ID,
    FUNDAMENTAL_PIT_PROCESSED_DATA_ID,
    FUNDAMENTAL_RAW_PROCESSED_DATA_ID,
)
from quant_research_platform.markets.implementations.a_share.processed import (
    materializer as a_share_materializer,
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
    estimate_a_share_factor_risk,
    run_a_share_portfolio_backtest,
)
from quant_research_platform.research.statistics import (
    cscv_pbo,
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
from quant_research_platform.sources.csmar.interim import (
    MARKET_BARS_DATA_ID,
    interim_catalog,
    normalize_raw_frame,
)
from quant_research_platform.storage.atomic import (
    atomic_write_directory,
    atomic_write_json,
    atomic_write_text,
)
from quant_research_platform.storage.datasets.reading import (
    DatasetReadMode,
    read_dataset,
)
from quant_research_platform.storage.datasets.writing import write_dataset
from quant_research_platform.storage.layout import StorageLayout
from quant_research_platform.workflows.crypto_hft import run_crypto_hft_experiment


def run_unified_research_smoke(
    *,
    raw_crypto_root: Path,
    output_root: Path,
) -> dict[str, object]:
    """Execute real repository seams on bounded synthetic inputs; never use holdout."""

    with runtime_resource_policy_scope(explicit_cap_bytes=256 * 1024**2):
        return atomic_write_directory(
            Path(output_root),
            lambda staging: _run_unified_research_smoke(
                raw_crypto_root=raw_crypto_root,
                output_root=staging,
            ),
        )


def _run_unified_research_smoke(
    *,
    raw_crypto_root: Path,
    output_root: Path,
) -> dict[str, object]:
    crypto_root = output_root / "crypto"
    crypto_report = run_crypto_hft_experiment(
        output_root=crypto_root,
        raw_root=raw_crypto_root,
        start_ms=0,
        end_ms=6_000,
        taker_config=TakerConfig(1.0, 1.0, 10, 1.0, 0.0, 500),
        maker_config=MakerConfig(1.0, 2.0, -1.0, 0.0, 0.0, 500),
    )
    a_share, a_plan = _a_share_smoke(output_root / "a_share")
    crypto_plan = _crypto_plan()
    statistics = _statistics_from_crypto_artifacts(
        crypto_root,
        crypto_report,
        a_plan=a_plan,
        crypto_plan=crypto_plan,
    )
    ledger = _write_ledgers(
        output_root / "experiment-ledger",
        a_plan=a_plan,
        crypto_plan=crypto_plan,
        crypto_report=crypto_report,
        a_share=a_share,
        statistics=statistics,
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
            "role_evaluations": statistics["role_metrics"],
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


def _a_share_smoke(output_root: Path) -> tuple[dict[str, object], ExperimentPlan]:
    sessions = pd.date_range("2020-01-01", periods=91, freq="B")
    calendar = pd.DataFrame(
        {"calendar_date": sessions, "source_market_type": 1, "is_trading_day": True}
    )
    fold = build_a_share_experiment_fold(
        calendar,
        fold_id="a-share-fold",
        train=TimeWindow(sessions[0].date(), sessions[19].date()),
        validation=TimeWindow(sessions[40].date(), sessions[44].date()),
        development_test=TimeWindow(sessions[45].date(), sessions[49].date()),
        final_holdout=TimeWindow(sessions[70].date(), sessions[90].date()),
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
        final_holdout=TimeWindow(sessions[70].date(), sessions[90].date()),
        max_label_lookahead_steps=20,
        selection_metric="validation_rankic",
        selection_direction="maximize",
        candidate_ids=("equal-weight", "ridge", "tree"),
        trial_budget=3,
        random_seed=7,
    )
    interim, disclosures = _financial_inputs()
    factors, factor_artifacts = _persist_a_share_fundamental_fixture(
        output_root,
        sessions=sessions,
        interim=interim,
        disclosures=disclosures,
    )
    benchmark = _benchmark_smoke(sessions)
    portfolio_panel = _a_share_portfolio_panel(
        sessions[10:40],
        benchmark_weights=benchmark["portfolio_benchmark_weights"],
    )
    risk_estimate = estimate_a_share_factor_risk(
        portfolio_panel.assign(risk_weight=portfolio_panel["benchmark_weight"]),
        exposure_columns=("beta", "size"),
        weight_column="risk_weight",
        ewma_half_life=10.0,
        diagonal_shrinkage=0.2,
        annualization_factor=252.0,
    )
    portfolio_panel["specific_variance"] = portfolio_panel["asset_id"].map(
        risk_estimate.specific_variance
    )
    schedules = (FeeSchedule(date(2020, 1, 1), 2.0, 2.0, 10.0, 0.1),)
    portfolios = {}
    for frequency in ("1d", "1w", "1m"):
        result = run_a_share_portfolio_backtest(
            portfolio_panel,
            benchmark_returns=benchmark["portfolio_benchmark_returns"],
            exposure_columns=("beta", "size"),
            factor_covariance=risk_estimate.factor_covariance,
            fee_schedules=schedules,
            config=PortfolioOptimizationConfig(
                frequency,
                1_000_000.0,
                0.8,
                0.3,
                1.0,
                0.2,
                0.6,
                10.0,
                0.5,
                2.0,
                1.0,
            ),
        )
        path = result["path"].copy()
        path["timestamp"] = path["timestamp"].map(
            lambda value: pd.Timestamp(value).isoformat()
        )
        exposures = result["active_exposures"].copy()
        exposures["timestamp"] = exposures["timestamp"].map(
            lambda value: pd.Timestamp(value).isoformat()
        )
        portfolios[frequency] = {
            "summary": result["summary"],
            "path": path.to_dict("records"),
            "active_exposures": exposures.to_dict("records"),
        }
    factor_rows = factors.astype("object").where(factors.notna(), None)
    factor_rows["timestamp"] = factor_rows["timestamp"].map(
        lambda value: pd.Timestamp(value).isoformat()
    )
    return (
        {
            "pit_defined_rows": int(factors.notna().all(axis=1).sum()),
            "pit_timeline": [
                pd.Timestamp(value).isoformat()
                for value in sorted(factors["timestamp"].drop_duplicates())
            ],
            "factor_definition_ids": [
                definition.factor_id
                for definition in factor_definitions.fundamental_factor_definitions()
            ],
            "factor_rows": factor_rows.to_dict("records"),
            "factor_artifacts": factor_artifacts,
            "benchmark_ids": sorted(benchmark["benchmark_ids"]),
            "benchmark_weight_rows": benchmark["weight_rows"],
            "benchmark_return_rows": [
                {
                    **row,
                    "timestamp": pd.Timestamp(row["timestamp"]).isoformat(),
                }
                for row in benchmark["portfolio_benchmark_returns"].to_dict("records")
            ],
            "risk_model": {
                "estimator": "cross_sectional_wls_ewma_diagonal_shrinkage",
                "ewma_half_life": risk_estimate.ewma_half_life,
                "diagonal_shrinkage": risk_estimate.diagonal_shrinkage,
                "annualization_factor": risk_estimate.annualization_factor,
                "factor_returns": [
                    {
                        **row,
                        "timestamp": pd.Timestamp(row["timestamp"]).isoformat(),
                    }
                    for row in risk_estimate.factor_returns.to_dict("records")
                ],
                "factor_covariance": risk_estimate.factor_covariance.to_dict(),
                "specific_variance": risk_estimate.specific_variance.to_dict(),
            },
            "portfolios": portfolios,
            "model_status": "REAL_DATA_PENDING",
            "horizon_decay_status": "REAL_DATA_PENDING",
        },
        plan,
    )


def _a_share_portfolio_panel(
    sessions: pd.DatetimeIndex,
    *,
    benchmark_weights: dict[str, float],
) -> pd.DataFrame:
    assets = tuple(benchmark_weights)
    benchmark = tuple(benchmark_weights.values())
    exposures = ((1.0, -0.2), (0.9, 0.3), (1.1, 0.1))
    alpha_cycles = (
        (0.005, -0.002, 0.001),
        (-0.002, 0.006, 0.002),
        (0.003, 0.001, -0.001),
    )
    return_cycles = (
        (0.002, -0.001, 0.0007),
        (-0.001, 0.0025, 0.0004),
        (0.0003, -0.0005, 0.0015),
    )
    rows = []
    for step, timestamp in enumerate(sessions):
        for position, asset_id in enumerate(assets):
            rows.append(
                {
                    "timestamp": timestamp,
                    "asset_id": asset_id,
                    "asset_return": return_cycles[step % 3][position],
                    "alpha": alpha_cycles[step % 3][position],
                    "current_weight": benchmark[position],
                    "benchmark_weight": benchmark[position],
                    "adv": (5_000_000.0, 4_000_000.0, 3_000_000.0)[position],
                    "eligible": True,
                    "can_enter": True,
                    "can_exit": True,
                    "beta": exposures[position][0],
                    "size": exposures[position][1],
                }
            )
    return pd.DataFrame(rows)


def _persist_a_share_fundamental_fixture(
    output_root: Path,
    *,
    sessions: pd.DatetimeIndex,
    interim: dict[str, pd.DataFrame],
    disclosures: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, object]]:
    policy = current_runtime_resource_policy()
    assert policy is not None
    layout = StorageLayout(output_root)
    source_id = SourceId("csmar")
    catalog = interim_catalog()
    target_sessions = sessions[10:13]
    calendar_sessions = pd.bdate_range("2019-01-01", sessions[-1])
    calendar = pd.DataFrame(
        {
            "timestamp": calendar_sessions,
            "calendar_date": calendar_sessions,
            "source_market_type": 1,
            "is_trading_day": True,
            "source_day_of_week": calendar_sessions.dayofweek + 1,
            "source_state": "O",
        }
    )
    market_rows = []
    for asset_id, market_cap in (("000001", 200.0), ("000002", 400.0)):
        for offset, timestamp in enumerate(target_sessions):
            close = 10.0 + offset
            market_rows.append(
                {
                    "source_asset_id": asset_id,
                    "timestamp": timestamp,
                    "trade_date": timestamp,
                    "market_type": 1,
                    "trading_status_code": 1,
                    "filling_flag": "0",
                    "open": close,
                    "high": close + 0.1,
                    "low": close - 0.1,
                    "close": close,
                    "volume": 1_000.0,
                    "amount": 10_000.0,
                    "prev_close": close - 0.1,
                    "limit_status": 0.0,
                    "period_return": 0.001,
                    "adj_close": close,
                    "adj_open": close,
                    "float_market_cap": market_cap * 0.8,
                    "total_market_cap": market_cap,
                    "turnover": 0.01,
                }
            )
    source_frames = {
        **interim,
        "trading_calendar": calendar,
        MARKET_BARS_DATA_ID: pd.DataFrame(market_rows),
        "financial_disclosures": normalize_raw_frame(
            disclosures, "financial_disclosures"
        ),
    }
    for data_id, frame in source_frames.items():
        spec = catalog.get(data_id)
        write_dataset(
            layout.interim_data_root(source_id, data_id),
            frame.loc[:, spec.required_columns],
            spec=spec,
            resource_policy=policy,
            borrowed_input=True,
        )
    target_window = TimeWindow(target_sessions[0].date(), target_sessions[-1].date())
    inspections = a_share_materializer.materialize_processed_data(
        layout=layout,
        source_id=source_id,
        time_window=target_window,
        data_ids=(FUNDAMENTAL_RAW_PROCESSED_DATA_ID,),
        resource_policy=policy,
    )
    factors = read_dataset(
        layout.processed_data_root(
            "a_share", "factor", FUNDAMENTAL_RAW_PROCESSED_DATA_ID
        ),
        mode=DatasetReadMode.MATERIALIZED,
        resource_policy=policy,
    )
    return factors, {
        "pit_data_id": FUNDAMENTAL_PIT_PROCESSED_DATA_ID,
        "raw_factor_data_id": FUNDAMENTAL_RAW_PROCESSED_DATA_ID,
        "research_dataset_id": FUNDAMENTAL_FORWARD_RETURNS_DATASET_ID,
        "research_dataset_status": "REAL_DATA_PENDING",
        "persisted_data_ids": sorted(inspections),
        "neutralized": {
            "status": "REAL_DATA_PENDING",
            "reason": "real_industry_history_pending",
        },
    }


def _financial_inputs() -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    frames = {}
    for stock, scale in (("000001", 1.0), ("000002", 2.0)):
        for report_period, period_scale in (
            ("2018-12-31", 0.8),
            ("2019-12-31", 1.0),
        ):
            common = {
                "Stkcd": stock,
                "ShortName": "synthetic",
                "IfCorrect": "0",
                "DeclareDate": None,
                "Accper": report_period,
                "Typrep": "A",
            }
            for data_id, values in (
                (
                    "financial_balance",
                    {
                        "A001000000": 100.0 * scale * period_scale,
                        "A002000000": 40.0 * scale * period_scale,
                        "A003000000": 60.0 * scale * period_scale,
                    },
                ),
                (
                    "financial_income",
                    {
                        "B001101000": 30.0 * scale * period_scale,
                        "B002000000": 10.0 * scale * period_scale,
                    },
                ),
                (
                    "financial_cashflow",
                    {"C001000000": 15.0 * scale * period_scale},
                ),
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
                "report_period": pd.Timestamp(report_period),
                "first_disclosure_date": pd.Timestamp(disclosure_date),
            }
            for stock in ("000001", "000002")
            for table in ("FS_Combas", "FS_Comins", "FS_Comscfd")
            for report_period, disclosure_date in (
                ("2018-12-31", "2019-01-03"),
                ("2019-12-31", "2020-01-03"),
            )
        ]
    )
    return interim, disclosures


def _benchmark_smoke(sessions: pd.DatetimeIndex) -> dict[str, object]:
    membership_rows = []
    weight_rows = []
    return_rows = []
    for index_code in ("000905", "000852"):
        for stock, weight in (
            ("000001", 50.0),
            ("000002", 30.0),
            ("000003", 20.0),
        ):
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
        for step, timestamp in enumerate(sessions[10:40]):
            return_rows.append(
                {
                    "Indexcd": index_code,
                    "Idxtrd01": timestamp,
                    "Idxtrd02": 100.0,
                    "Idxtrd03": 101.0,
                    "Idxtrd04": 99.0,
                    "Idxtrd05": 100.5,
                    "Idxtrd06": 1.0,
                    "Idxtrd07": 1.0,
                    "Idxtrd08": (0.08, -0.03, 0.05)[step % 3],
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
    weights = materialize_benchmark_weights(
        observations,
        history["weights"],
        history["membership"],
    )
    weight_rows = weights.copy()
    for column in ("timestamp", "effective_date", "membership_effective_date"):
        weight_rows[column] = weight_rows[column].map(
            lambda value: pd.Timestamp(value).isoformat()
        )
    return {
        "benchmark_ids": set(history["membership"]["benchmark_id"]),
        "weight_rows": weight_rows.to_dict("records"),
        "portfolio_benchmark_weights": weights.loc[
            weights["benchmark_id"].eq("csi500"),
            ("constituent_asset_id", "weight"),
        ]
        .set_index("constituent_asset_id")["weight"]
        .to_dict(),
        "portfolio_benchmark_returns": history["returns"]
        .loc[
            history["returns"]["benchmark_id"].eq("csi500"),
            ("timestamp", "period_return"),
        ]
        .rename(columns={"period_return": "benchmark_return"}),
    }


def _crypto_plan() -> ExperimentPlan:
    grid = pd.date_range("1970-01-01", periods=81, freq="100ms")
    fold = build_crypto_grid_experiment_fold(
        grid,
        fold_id="crypto-fold",
        train=TimeWindow(grid[0].to_pydatetime(), grid[34].to_pydatetime()),
        validation=TimeWindow(grid[40].to_pydatetime(), grid[47].to_pydatetime()),
        development_test=TimeWindow(grid[48].to_pydatetime(), grid[55].to_pydatetime()),
        final_holdout=TimeWindow(grid[70].to_pydatetime(), grid[80].to_pydatetime()),
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
    output_root: Path,
    crypto_report: dict[str, object],
    *,
    a_plan: ExperimentPlan,
    crypto_plan: ExperimentPlan,
) -> dict[str, object]:
    layout = StorageLayout(output_root)
    resource_policy = current_runtime_resource_policy()
    assert resource_policy is not None
    research = read_dataset(
        layout.research_dataset_root("crypto", HFT_RESEARCH_DATASET_ID),
        mode=DatasetReadMode.MATERIALIZED,
        resource_policy=resource_policy,
    )
    fold = crypto_plan.folds[0]
    role_windows = {
        "validation": fold.validation,
        "development_test": fold.development_test,
    }
    matrices: dict[str, pd.DataFrame] = {}
    role_asset_ics: dict[str, dict[str, list[float]]] = {
        role: {} for role in role_windows
    }
    role_counts: dict[str, dict[str, int]] = {role: {} for role in role_windows}
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
                    if not all(found):
                        raise ValueError("unified_prediction_scope_incomplete")
                    predictions = np.asarray(values)
                    series = (
                        predictions
                        - scoped_target["future_mid_return_500ms"].to_numpy()
                    ) ** 2
                for role, window in role_windows.items():
                    role_mask = scoped_target["timestamp"].between(
                        pd.Timestamp(window.start), pd.Timestamp(window.end)
                    )
                    labels = scoped_target.loc[
                        role_mask, "future_mid_return_500ms"
                    ].to_numpy()
                    role_predictions = predictions[role_mask.to_numpy()]
                    if (
                        len(labels) < 2
                        or np.std(labels) == 0.0
                        or np.std(role_predictions) == 0.0
                    ):
                        raise ValueError("unified_role_ic_undefined")
                    role_asset_ics[role].setdefault(mode, []).append(
                        float(np.corrcoef(labels, role_predictions)[0, 1])
                    )
                    role_counts[role][mode] = role_counts[role].get(mode, 0) + len(
                        labels
                    )
            frame = pd.DataFrame({mode: series}, index=scoped_target["timestamp"])
            losses = frame if losses is None else losses.join(frame, how="inner")
        matrices[asset_id] = losses
    family_results: dict[str, dict[str, object]] = {}
    for asset_id, losses in matrices.items():
        family_results[asset_id] = {}
        for role, window in role_windows.items():
            role_losses = losses.loc[
                losses.index.to_series().between(
                    pd.Timestamp(window.start), pd.Timestamp(window.end)
                )
            ]
            result = _statistics_for_loss_matrix(role_losses)
            family_results[asset_id][role] = result
    global_candidate_ids = [
        *[f"a_share:{candidate}" for candidate in a_plan.candidate_ids],
        *[f"crypto:{candidate}" for candidate in crypto_plan.candidate_ids],
    ]
    return {
        "families": family_results,
        "role_metrics": {
            role: {
                mode: {
                    "mean_asset_ic": float(np.mean(values)),
                    "asset_count": len(values),
                    "observation_count": role_counts[role][mode],
                }
                for mode, values in role_asset_ics[role].items()
            }
            for role in role_windows
        },
        "global": {
            "candidate_ids": global_candidate_ids,
            "candidate_dispositions": [
                {
                    "candidate_id": candidate_id,
                    "status": "UNDEFINED",
                    "reason": (
                        "real_data_pending"
                        if candidate_id.startswith("a_share:")
                        else "incomparable_evaluation_scope_for_global_matrix"
                    ),
                }
                for candidate_id in global_candidate_ids
            ],
            "multiple_testing": {
                "status": "UNDEFINED",
                "reason": "no_common_time_by_candidate_matrix",
                "included_candidate_ids": [],
                "excluded_candidate_ids": global_candidate_ids,
                "fdr_bh": [],
                "fdr_by": [],
            },
        },
        "fama_macbeth": {
            "status": "UNDEFINED",
            "reason": "real_data_pending",
        },
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
    predictive_scores = -losses.to_numpy()
    trial_sharpes = [
        float(np.mean(column) / np.std(column, ddof=1))
        for column in predictive_scores.T
    ]
    sharpe = None
    sharpe_reason = None
    try:
        sharpe = probabilistic_and_deflated_sharpe(
            predictive_scores[:, int(np.argmax(trial_sharpes))],
            benchmark_sharpe=0.0,
            trial_sharpes=trial_sharpes,
        )
    except ValueError as error:
        if str(error) not in {
            "sharpe_zero_variance",
            "sharpe_variance_invalid",
            "trial_sharpe_dispersion_undefined",
        }:
            raise
        sharpe_reason = str(error)
    pbo = cscv_pbo(predictive_scores, slice_count=4, max_combinations=6)
    p_values = [result.p_value for result in hac_results]
    fdr_bh = fdr_adjust(p_values, method="BH")
    fdr_by = fdr_adjust(p_values, method="BY")
    return {
        "common_observations": len(losses),
        "candidate_ids": list(losses.columns),
        "hypothesis_ids": list(alternatives.columns),
        "hac": [
            {"mean": result.mean, "p_value": result.p_value} for result in hac_results
        ],
        "stationary_bootstrap": {
            "lower": bootstrap.lower,
            "upper": bootstrap.upper,
        },
        "fdr_bh": fdr_bh.adjusted_p_values.tolist(),
        "fdr_by": fdr_by.adjusted_p_values.tolist(),
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
        "predictive_score_overfit": {
            "input_semantics": "negative_squared_error_proxy_not_pnl",
            "trial_candidate_ids": list(losses.columns),
            "trial_count": len(trial_sharpes),
            "psr_dsr_status": "DEFINED" if sharpe is not None else "UNDEFINED",
            "psr_dsr_reason": sharpe_reason,
            "psr": None if sharpe is None else sharpe.psr,
            "dsr": None if sharpe is None else sharpe.dsr,
            "pbo": pbo.pbo,
            "pbo_status": pbo.status,
        },
    }


def _write_ledgers(
    root: Path,
    *,
    a_plan: ExperimentPlan,
    crypto_plan: ExperimentPlan,
    crypto_report: dict[str, object],
    a_share: dict[str, object],
    statistics: dict[str, object],
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
                validation_metrics=statistics["role_metrics"]["validation"][mode],
                development_test_metrics=statistics["role_metrics"]["development_test"][
                    mode
                ],
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
