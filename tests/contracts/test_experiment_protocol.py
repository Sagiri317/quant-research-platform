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
) -> ExperimentFold:
    return ExperimentFold(
        fold_id=fold_id,
        train=TimeWindow(*train),
        validation=TimeWindow(*validation),
    )


def _plan() -> ExperimentPlan:
    return ExperimentPlan(
        stage="selection",
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
            ),
            _fold(
                "fold-2",
                (date(2019, 1, 1), date(2021, 12, 31)),
                (date(2022, 2, 1), date(2022, 7, 31)),
            ),
        ),
        test=TimeWindow(date(2023, 2, 1), date(2023, 12, 31)),
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


def test_experiment_plan_rejects_duplicate_candidate_ids() -> None:
    with pytest.raises(ValueError, match="^experiment_candidate_ids$"):
        replace(_plan(), candidate_ids=("ridge_model", "ridge_model"))


@pytest.mark.parametrize(
    "order,error",
    (
        (("other",), "experiment_tie_break_order_mismatch"),
        (("base",), "experiment_tie_break_order_mismatch"),
        (("base", "other", "extra"), "experiment_tie_break_order_mismatch"),
        (("base", "other", "other"), "experiment_tie_break_order"),
    ),
)
def test_experiment_plan_rejects_incomplete_or_nonunique_tie_order(
    order: tuple[str, ...], error: str
) -> None:
    plan = replace(_plan(), candidate_ids=("base", "other"))
    with pytest.raises(ValueError, match=f"^{error}$"):
        replace(plan, tie_break_order=order)


def test_experiment_plan_is_locked_immutable_and_hash_stable() -> None:
    plan = _plan()
    reordered = replace(
        plan,
        backend_versions={"statsmodels": "0.15.0", "numpy": "2.3.5"},
    )

    assert plan.holdout_status == "LOCKED"
    assert plan.protocol_hash == reordered.protocol_hash
    assert len(plan.protocol_hash) == 64
    assert (
        ExperimentPlan.from_payload(plan.protocol_payload()).protocol_hash
        == plan.protocol_hash
    )
    changed = replace(plan, random_seed=plan.random_seed + 1)
    assert changed.protocol_hash != plan.protocol_hash
    assert (
        changed.protocol_hash
        == ExperimentPlan.from_payload(changed.protocol_payload()).protocol_hash
    )
    assert "protocol_version" not in plan.protocol_payload()
    with pytest.raises(TypeError):
        plan.backend_versions["numpy"] = "changed"  # type: ignore[index]
    with pytest.raises(FrozenInstanceError):
        plan.holdout_status = "OPEN"  # type: ignore[misc]


def test_experiment_plan_rejects_test_overlap_and_keeps_no_purge_surface() -> None:
    plan = _plan()

    assert not hasattr(plan, "max_label_lookahead_steps")
    with pytest.raises(ValueError, match="^experiment_test_overlap$"):
        replace(
            plan,
            test=TimeWindow(date(2022, 7, 31), date(2023, 12, 31)),
        )
    with pytest.raises(ValueError, match="^experiment_fold_windows_overlap$"):
        replace(
            plan.folds[-1],
            validation=TimeWindow(date(2021, 12, 31), date(2022, 7, 31)),
        )


@pytest.mark.parametrize(
    "field",
    (
        "folds",
        "input_lineage",
        "candidate_ids",
        "tie_break_order",
        "test",
        "fit_windows",
        "selection_ledger_semantic_identities",
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


def test_refit_stage_has_one_resolved_candidate_and_no_selection_score():
    plan = _plan()
    refit = replace(
        plan,
        stage="refit",
        folds=(),
        fit_windows=(
            TimeWindow(date(2018, 1, 1), date(2020, 12, 31)),
            TimeWindow(date(2021, 2, 1), date(2022, 7, 31)),
        ),
        fit_cutoff=date(2023, 1, 31),
        selection_group="global",
        candidate_ids=("ridge",),
        candidate_parameters={"ridge": {"random_seed": 7}},
        trial_budget=1,
        selection_metric=None,
        selection_direction=None,
        tie_break_order=(),
        selection_ledger_semantic_identities=("a" * 64, "b" * 64),
    )
    assert ExperimentPlan.from_payload(refit.protocol_payload()) == refit
    assert refit.protocol_hash != plan.protocol_hash
    for changes in (
        {"selection_metric": "validation_rankic"},
        {"trial_budget": 2},
        {"candidate_parameters": {}},
        {"candidate_parameters": {"ridge": {"random_seed": 8}}},
        {"selection_ledger_semantic_identities": ()},
        {"folds": plan.folds},
        {"fit_windows": ()},
        {"fit_cutoff": None},
        {"fit_cutoff": date(2022, 7, 30)},
        {"fit_cutoff": refit.test.start},
        {"fit_windows": tuple(reversed(refit.fit_windows))},
        {
            "fit_windows": (
                refit.fit_windows[0],
                TimeWindow(date(2020, 12, 31), date(2022, 7, 31)),
            )
        },
        {"tie_break_order": ("ridge",)},
    ):
        with pytest.raises(ValueError, match="experiment_refit_plan_invalid"):
            replace(refit, **changes)
    for changes in (
        {"folds": ()},
        {"selection_ledger_semantic_identities": ("a" * 64,)},
        {"fit_windows": refit.fit_windows},
        {"fit_cutoff": refit.fit_cutoff},
    ):
        with pytest.raises(ValueError, match="^experiment_selection_plan_invalid$"):
            replace(plan, **changes)
    for parents in (("b" * 64, "a" * 64), ("a" * 64, "a" * 64), ("invalid",)):
        with pytest.raises(ValueError, match="^experiment_selection_parents_invalid$"):
            replace(refit, selection_ledger_semantic_identities=parents)
    with pytest.raises(ValueError, match="^experiment_selection_group_required$"):
        replace(refit, selection_group=None)
    with pytest.raises(ValueError, match="^experiment_selection_group_invalid$"):
        replace(plan, selection_group="global")
    assert replace(refit, fit_cutoff=refit.fit_windows[-1].end)
    for changes in (
        {"fit_cutoff": date(2023, 1, 30)},
        {"fit_windows": (refit.fit_windows[0],)},
        {"selection_ledger_semantic_identities": ("a" * 64,)},
        {"selection_group": "quarterly"},
    ):
        changed = replace(refit, **changes)
        assert changed.protocol_hash != refit.protocol_hash
        assert ExperimentPlan.from_payload(changed.protocol_payload()) == changed
    crypto = replace(refit, market_id=MarketId("crypto"), selection_group=None)
    assert ExperimentPlan.from_payload(crypto.protocol_payload()) == crypto
    with pytest.raises(ValueError, match="^experiment_selection_group_invalid$"):
        replace(crypto, selection_group="global")


@pytest.mark.parametrize(
    "field", ("final_holdout", "selection_ledger_semantic_identity")
)
def test_retired_plan_fields_cannot_be_read_as_three_role_payload(field):
    payload = _plan().protocol_payload()
    payload[field] = payload["test"] if field == "final_holdout" else "a" * 64
    with pytest.raises(ValueError, match="^experiment_plan_payload_invalid$"):
        ExperimentPlan.from_payload(payload)


@pytest.mark.parametrize(
    "field", ("development_test", "embargo_steps", "final_holdout_start_step")
)
def test_retired_fold_fields_cannot_recreate_a_fourth_role(field):
    payload = _plan().protocol_payload()
    payload["folds"][0][field] = payload["test"] if field == "development_test" else 0
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        ExperimentPlan.from_payload(payload)
