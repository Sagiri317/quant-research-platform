from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
from datetime import date

import pandas as pd
import pytest

from quant_research_platform.contracts.experiments import (
    ExperimentFold,
    ExperimentPlan,
)
from quant_research_platform.contracts.ids import MarketId
from quant_research_platform.contracts.time import TimeWindow
from quant_research_platform.markets.implementations.a_share.calendar import (
    build_a_share_experiment_fold,
)
from quant_research_platform.markets.implementations.crypto.hft import (
    build_crypto_grid_experiment_fold,
)


def _fold(
    fold_id: str,
    train: tuple[date, date],
    validation: tuple[date, date],
    development_test: tuple[date, date],
) -> ExperimentFold:
    return ExperimentFold(
        fold_id=fold_id,
        train=TimeWindow(*train),
        validation=TimeWindow(*validation),
        development_test=TimeWindow(*development_test),
        purge_steps=20,
        step_kind="test_steps",
        train_end_step=0,
        validation_start_step=21,
        validation_end_step=22,
        development_test_start_step=23,
    )


def _plan() -> ExperimentPlan:
    return ExperimentPlan(
        experiment_id="unified-smoke-1",
        market_id=MarketId("a_share"),
        research_dataset_id="a-share-smoke",
        target_id="forward-return-20d",
        candidate_family_id="fixed-smoke-candidates",
        data_snapshot_id="snapshot-1",
        input_lineage=("market-panel-1", "factor-panel-1"),
        folds=(
            _fold(
                "fold-1",
                (date(2018, 1, 1), date(2020, 12, 31)),
                (date(2021, 2, 1), date(2021, 7, 31)),
                (date(2021, 8, 1), date(2022, 1, 31)),
            ),
            _fold(
                "fold-2",
                (date(2019, 1, 1), date(2021, 12, 31)),
                (date(2022, 2, 1), date(2022, 7, 31)),
                (date(2022, 8, 1), date(2023, 1, 31)),
            ),
        ),
        final_holdout=TimeWindow(date(2023, 2, 1), date(2023, 12, 31)),
        max_label_lookahead_steps=20,
        selection_metric="validation_rankic",
        selection_direction="maximize",
        candidate_ids=("equal-weight", "ridge"),
        trial_budget=4,
        random_seed=7,
        backend_versions={"numpy": "2.3.5", "statsmodels": "0.15.0"},
        thread_settings={"native_threads": 1},
        cost_assumptions={"commission_bps": 2.0},
        risk_assumptions={"tracking_error_limit": 0.05},
    )


def test_experiment_plan_is_locked_immutable_and_hash_stable() -> None:
    plan = _plan()
    reordered = replace(
        plan,
        backend_versions={"statsmodels": "0.15.0", "numpy": "2.3.5"},
    )

    assert plan.holdout_status == "LOCKED"
    assert plan.protocol_hash == reordered.protocol_hash
    assert len(plan.protocol_hash) == 64
    assert plan.development_windows() == tuple(
        fold.development_test for fold in plan.folds
    )
    with pytest.raises(PermissionError, match="^final_holdout_locked$"):
        plan.window("final_holdout")
    with pytest.raises(TypeError):
        plan.backend_versions["numpy"] = "changed"  # type: ignore[index]
    with pytest.raises(FrozenInstanceError):
        plan.holdout_status = "OPEN"  # type: ignore[misc]


def test_experiment_plan_rejects_nonderived_purge_and_holdout_overlap() -> None:
    plan = _plan()

    with pytest.raises(ValueError, match="^experiment_purge_mismatch$"):
        replace(plan, max_label_lookahead_steps=19)
    with pytest.raises(ValueError, match="^experiment_holdout_overlap$"):
        replace(
            plan,
            final_holdout=TimeWindow(date(2022, 12, 1), date(2023, 12, 31)),
        )


def test_market_owners_bind_purge_and_embargo_to_real_steps() -> None:
    sessions = pd.date_range("2020-01-01", periods=10, freq="B")
    calendar = pd.DataFrame(
        {
            "calendar_date": sessions,
            "source_market_type": 1,
            "is_trading_day": True,
        }
    )
    a_share = build_a_share_experiment_fold(
        calendar,
        fold_id="a-share-fold",
        train=TimeWindow(sessions[0].date(), sessions[1].date()),
        validation=TimeWindow(sessions[4].date(), sessions[5].date()),
        development_test=TimeWindow(sessions[7].date(), sessions[8].date()),
        purge_steps=2,
        embargo_steps=1,
    )
    crypto_grid = pd.date_range("2026-01-01", periods=10, freq="100ms")
    crypto = build_crypto_grid_experiment_fold(
        crypto_grid,
        fold_id="crypto-fold",
        train=TimeWindow(
            crypto_grid[0].to_pydatetime(), crypto_grid[1].to_pydatetime()
        ),
        validation=TimeWindow(
            crypto_grid[4].to_pydatetime(), crypto_grid[5].to_pydatetime()
        ),
        development_test=TimeWindow(
            crypto_grid[7].to_pydatetime(), crypto_grid[8].to_pydatetime()
        ),
        purge_steps=2,
        embargo_steps=1,
    )

    assert a_share.step_kind == "a_share_trading_session"
    assert crypto.step_kind == "crypto_100ms_grid"
    with pytest.raises(ValueError, match="^experiment_purge_gap_invalid$"):
        build_a_share_experiment_fold(
            calendar,
            fold_id="bad-fold",
            train=TimeWindow(sessions[0].date(), sessions[1].date()),
            validation=TimeWindow(sessions[4].date(), sessions[5].date()),
            development_test=TimeWindow(sessions[7].date(), sessions[8].date()),
            purge_steps=1,
            embargo_steps=1,
        )
