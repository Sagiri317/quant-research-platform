from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
from datetime import date

import pytest

from quant_research_platform.contracts.experiments import (
    ExperimentFold,
    ExperimentPlan,
)
from quant_research_platform.contracts.ids import MarketId
from quant_research_platform.contracts.time import TimeWindow


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
        embargo_steps=0,
        step_kind="test_steps",
        train_end_step=0,
        validation_start_step=21,
        validation_end_step=22,
        development_test_start_step=23,
        development_test_end_step=24,
        final_holdout_start_step=45,
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
    assert "protocol_version" not in plan.protocol_payload()
    assert tuple(fold.development_test for fold in plan.folds) == (
        TimeWindow(date(2021, 8, 1), date(2022, 1, 31)),
        TimeWindow(date(2022, 8, 1), date(2023, 1, 31)),
    )
    with pytest.raises(PermissionError, match="^final_holdout_locked$"):
        plan.window("final_holdout")
    with pytest.raises(TypeError):
        plan.backend_versions["numpy"] = "changed"  # type: ignore[index]
    with pytest.raises(FrozenInstanceError):
        plan.holdout_status = "OPEN"  # type: ignore[misc]


def test_experiment_plan_rejects_holdout_overlap_and_keeps_no_purge_surface() -> None:
    plan = _plan()

    assert not hasattr(plan, "max_label_lookahead_steps")
    with pytest.raises(ValueError, match="^experiment_holdout_overlap$"):
        replace(
            plan,
            final_holdout=TimeWindow(date(2022, 12, 1), date(2023, 12, 31)),
        )
    with pytest.raises(ValueError):
        replace(
            plan,
            folds=(
                *plan.folds[:-1],
                replace(plan.folds[-1], final_holdout_start_step=24),
            ),
        )


@pytest.mark.parametrize(
    "field",
    (
        "folds",
        "input_lineage",
        "candidate_ids",
        "tie_break_order",
        "final_holdout",
        "backend_versions",
        "thread_settings",
        "cost_assumptions",
        "risk_assumptions",
        "candidate_parameters",
    ),
)
def test_experiment_plan_json_container_types_are_not_normalized(field):
    plan = _plan()
    payload = plan.protocol_payload()
    assert ExperimentPlan.from_payload(payload).protocol_hash == plan.protocol_hash
    value = payload[field]
    payload[field] = (
        list(value.items())
        if isinstance(value, dict)
        else {str(item): "discarded" for item in value}
    )
    with pytest.raises(ValueError, match="experiment_plan_payload_invalid"):
        ExperimentPlan.from_payload(payload)


def test_experiment_fold_json_object_is_not_a_pairs_array():
    payload = _plan().protocol_payload()
    payload["folds"][0] = list(payload["folds"][0].items())
    with pytest.raises(ValueError, match="experiment_plan_payload_invalid"):
        ExperimentPlan.from_payload(payload)
