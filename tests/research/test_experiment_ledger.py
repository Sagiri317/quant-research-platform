from __future__ import annotations

from dataclasses import replace
from datetime import date
from pathlib import Path

import pytest

from quant_research_platform.contracts.experiments import ExperimentFold, ExperimentPlan
from quant_research_platform.contracts.ids import MarketId
from quant_research_platform.contracts.time import TimeWindow
from quant_research_platform.research.experiments import (
    TrialRecord,
    append_trial_record,
    read_trial_ledger,
)


def _plan() -> ExperimentPlan:
    return ExperimentPlan(
        experiment_id="unified-smoke-1",
        market_id=MarketId("a_share"),
        research_dataset_id="a-share-smoke",
        target_id="forward-return-20d",
        candidate_family_id="fixed-smoke-candidates",
        data_snapshot_id="snapshot-1",
        input_lineage=("market-panel-1",),
        folds=(
            ExperimentFold(
                fold_id="fold-1",
                train=TimeWindow(date(2020, 1, 1), date(2020, 1, 1)),
                validation=TimeWindow(date(2020, 1, 2), date(2020, 1, 2)),
                development_test=TimeWindow(date(2020, 1, 3), date(2020, 1, 3)),
                purge_steps=0,
                embargo_steps=0,
                step_kind="test_steps",
                train_end_step=0,
                validation_start_step=1,
                validation_end_step=1,
                development_test_start_step=2,
                development_test_end_step=2,
                final_holdout_start_step=3,
            ),
        ),
        final_holdout=TimeWindow(date(2020, 1, 4), date(2020, 1, 4)),
        max_label_lookahead_steps=0,
        selection_metric="validation_rankic",
        selection_direction="maximize",
        candidate_ids=("ridge",),
        trial_budget=3,
        random_seed=7,
    )


def _record(
    plan: ExperimentPlan, *, event_id: str, attempt_index: int, status: str
) -> TrialRecord:
    return TrialRecord(
        event_id=event_id,
        attempt_index=attempt_index,
        experiment_id=plan.experiment_id,
        protocol_hash=plan.protocol_hash,
        candidate_family_id=plan.candidate_family_id,
        candidate_id="ridge",
        trial_id=f"trial-{attempt_index}",
        fold_id="fold-1",
        market_id="a_share",
        research_dataset_id="a-share-smoke",
        target_id="forward-return-20d",
        horizon_id="20d",
        common_coverage_id="coverage-1",
        factor_definition_ids=("mom-20d",),
        model_definition_id="ridge-v1",
        normalized_parameters={"alpha": 1.0},
        random_seed=7,
        backend_versions={"statsmodels": "0.15.0"},
        thread_settings={"native_threads": 1},
        status=status,
        validation_metrics={"rankic": None},
        development_test_metrics={},
        undefined_reasons={"rankic": "insufficient_observations"},
        cost_assumptions={"commission_bps": 2.0},
        risk_assumptions={"tracking_error_limit": 0.05},
        rebalance_assumption="1w",
        artifact_lineage=("artifact-1",),
    )


def test_trial_ledger_is_append_only_and_retains_failed_and_abandoned(
    tmp_path: Path,
) -> None:
    root = tmp_path / "ledger"
    plan = _plan()
    failed = _record(plan, event_id="event-1", attempt_index=1, status="failed")
    abandoned = _record(plan, event_id="event-2", attempt_index=2, status="abandoned")

    first_path = append_trial_record(root, plan, failed)
    second_path = append_trial_record(root, plan, abandoned)
    records = read_trial_ledger(root, plan)

    assert first_path != second_path
    assert [record.status for record in records] == ["failed", "abandoned"]
    assert [record.attempt_index for record in records] == [1, 2]
    with pytest.raises(ValueError, match="^trial_attempt_not_next$"):
        append_trial_record(root, plan, failed)


def test_trial_record_never_allows_holdout_access() -> None:
    plan = _plan()
    with pytest.raises(ValueError, match="^trial_holdout_access_forbidden$"):
        replace(
            _record(plan, event_id="event-1", attempt_index=1, status="completed"),
            holdout_accessed=True,
        )


def test_trial_record_deep_snapshots_and_ledger_rejects_backfill(
    tmp_path: Path,
) -> None:
    plan = _plan()
    nested = {"grid": [1, {"alpha": 2.0}]}
    record = replace(
        _record(plan, event_id="event-1", attempt_index=1, status="failed"),
        normalized_parameters=nested,
    )
    before = record.payload()
    nested["grid"].append(3)

    assert record.payload() == before
    with pytest.raises(ValueError, match="^trial_attempt_not_next$"):
        append_trial_record(
            tmp_path / "ledger",
            plan,
            _record(plan, event_id="event-2", attempt_index=2, status="failed"),
        )


def test_contaminated_development_test_forces_new_experiment_id(tmp_path: Path) -> None:
    plan = _plan()
    root = tmp_path / "ledger"
    contaminated = replace(
        _record(plan, event_id="event-1", attempt_index=1, status="completed"),
        development_test_contaminated=True,
    )
    append_trial_record(root, plan, contaminated)

    with pytest.raises(
        ValueError, match="^trial_contaminated_experiment_requires_new_id$"
    ):
        append_trial_record(
            root,
            plan,
            _record(plan, event_id="event-2", attempt_index=2, status="completed"),
        )


def test_trial_ledger_uses_exact_bounded_storage_membership(tmp_path: Path) -> None:
    plan = _plan()
    root = tmp_path / "ledger"
    append_trial_record(
        root,
        plan,
        _record(plan, event_id="event-1", attempt_index=1, status="failed"),
    )
    (root / "rogue.txt").write_text("not a ledger member", encoding="utf-8")

    with pytest.raises(ValueError, match="^dataset_metadata_directory_member_invalid$"):
        read_trial_ledger(root, plan)
