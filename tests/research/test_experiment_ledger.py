from __future__ import annotations

from dataclasses import replace
from datetime import date
from pathlib import Path

import pytest

from quant_research_platform.contracts.experiments import (
    ExperimentFold,
    ExperimentPlan,
    TrialRecord,
)
from quant_research_platform.contracts.ids import MarketId
from quant_research_platform.contracts.time import TimeWindow
from quant_research_platform.storage.experiments import (
    append_trial_record,
    read_trial_ledger,
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
        input_lineage=("market-panel-1",),
        folds=(
            ExperimentFold(
                fold_id="fold-1",
                train=TimeWindow(date(2020, 1, 1), date(2020, 1, 1)),
                validation=TimeWindow(date(2020, 1, 2), date(2020, 1, 2)),
                development_test=TimeWindow(date(2020, 1, 3), date(2020, 1, 3)),
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
        run_id="run_test",
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
        metrics={"rankic": None},
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
    with read_trial_ledger(root, plan) as (_saved_plan, records):
        assert first_path != second_path
        assert [record.status for record in records] == ["failed", "abandoned"]
        assert [record.attempt_index for record in records] == [1, 2]
    with pytest.raises(ValueError, match="^trial_attempt_not_next$"):
        append_trial_record(root, plan, failed)
    completed = _record(plan, event_id="event-3", attempt_index=3, status="completed")
    append_trial_record(root, plan, completed)
    original = {
        path.relative_to(root): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file()
    }
    with pytest.raises(ValueError, match="^trial_budget_exceeded$"):
        append_trial_record(
            root,
            plan,
            _record(plan, event_id="event-4", attempt_index=4, status="attempted"),
        )
    with read_trial_ledger(root, plan) as (_saved_plan, ledger_records_3):
        assert ledger_records_3 == (failed, abandoned, completed)
    assert {
        path.relative_to(root): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file()
    } == original


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
        with read_trial_ledger(root, plan):
            pass


def test_real_selection_records_terminal_attempts_retry_and_primary_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import pandas as pd

    from quant_research_platform.catalog.models import ModelCatalog
    from quant_research_platform.contracts.models import ModelSpec, model_spec_payload
    from quant_research_platform.contracts.predictions import PredictionSpec
    from quant_research_platform.research.experiments import TrialRecorder
    from quant_research_platform.research.models import trial_selection
    from quant_research_platform.research.models.supervised import preflight_model_specs
    from quant_research_platform.research.models.trial_selection import (
        ModelSampleCoverage,
        model_validation_coverage_scope,
    )
    from quant_research_platform.runtime.resources import runtime_resource_policy_scope
    from quant_research_platform.storage.layout import StorageLayout
    from tests.helpers.research_specs import make_demo_research_dataset_spec

    specs = tuple(
        ModelSpec(
            model_id=name,
            model_algorithm="ridge_regression",
            input_factor_columns=("x",),
            label_column="label",
            random_seed=7,
            hyperparameters={"alpha": alpha},
        )
        for name, alpha in (("ridge", 1.0), ("ridge_other", 0.1))
    )
    preflight_model_specs(specs)
    plan = replace(
        _plan(),
        candidate_ids=tuple(spec.model_id for spec in specs),
        trial_budget=2,
        candidate_parameters={
            spec.model_id: model_spec_payload(spec) for spec in specs
        },
    )
    dataset = make_demo_research_dataset_spec(
        data_id="a-share-smoke",
        input_data_ids=("source",),
        base_input_data_id="source",
        factor_columns=("x",),
        label_columns=("label",),
        filter_columns=("eligible",),
        context_columns=(),
    )
    frame = pd.DataFrame(
        [
            {
                "timestamp": pd.Timestamp(day),
                "asset_id": asset,
                "market_id": "a_share",
                "x": float(index),
                "label": float(2 * index),
                "label_available_at": pd.Timestamp(day),
                "eligible": True,
            }
            for day in ("2020-01-01", "2020-01-02", "2020-01-03", "2020-01-04")
            for index, asset in enumerate(("a", "b", "c"), start=1)
        ]
    )
    prediction = PredictionSpec(
        prediction_id="prediction",
        model_id="ridge",
        candidate_model_ids=("ridge_other",),
        research_dataset_id=dataset.data_id,
        train_end="2020-01-01",
        test_start="2020-01-03",
        test_end="2020-01-03",
        model_selection_validation_start="2020-01-02",
        model_selection_validation_end="2020-01-02",
    )
    layout = StorageLayout(tmp_path)
    coverage = ModelSampleCoverage(
        model_validation_coverage_scope(plan.data_snapshot_id, plan.folds[0], specs)
    )
    coverage.add(frame.loc[frame.timestamp == "2020-01-02", ["timestamp", "asset_id"]])

    def select(run_id: str):
        recorder = TrialRecorder(layout, run_id, plan)
        with runtime_resource_policy_scope() as policy:
            try:
                result = trial_selection.select_prediction_model_spec(
                    prediction_spec=prediction,
                    dataset_spec=dataset,
                    model_catalog=ModelCatalog(specs),
                    chunks=lambda columns, **_kwargs: (frame.loc[:, list(columns)],),
                    trial_recorder=recorder,
                    validation_coverage=coverage,
                )
            finally:
                assert policy.used_memory_bytes == 0
        return result, recorder

    (_, _metadata), first = select("run_first")
    with read_trial_ledger(first.root) as (_saved_plan, records):
        assert [record.status for record in records] == ["completed", "completed"]
        assert all(record.metrics["validation_rows"] == 3 for record in records)
        assert all(record.metrics["n_train"] == 3 for record in records)
        assert all(
            record.normalized_parameters
            == plan.candidate_parameters[record.candidate_id]
            for record in records
        )
        first_event_id = records[0].event_id
    # OOS and holdout labels cannot alter validation or the selected config.
    frame.loc[frame.timestamp >= "2020-01-03", "label"] = -100000.0
    (_, _), second = select("run_retry")
    assert (
        first.reference()["semantic_identity"]
        == second.reference()["semantic_identity"]
    )
    with read_trial_ledger(second.root) as (_saved_plan, ledger_records_11):
        assert first_event_id != ledger_records_11[0].event_id

    for phase in ("fit", "validation", "abandon"):
        primary = (
            KeyboardInterrupt("private text")
            if phase == "abandon"
            else ValueError("private text")
        )

        def fail(*_args, _error=primary, **_kwargs):
            raise _error

        target = trial_selection.fitting if phase == "fit" else trial_selection
        name = (
            "_fit_model_result_owned_from_frames"
            if phase == "fit"
            else "evaluate_model_trial_metrics_from_chunks"
        )
        with monkeypatch.context() as patch:
            patch.setattr(target, name, fail)
            with pytest.raises(type(primary)) as caught:
                select(f"run_{phase}")
        assert caught.value is primary
        with read_trial_ledger(
            layout.trial_ledger_root(f"run_{phase}", plan.experiment_id)
        ) as (_saved_plan, failed):
            assert len(failed) == 1  # second config was registered, never attempted.
            assert failed[0].status == ("abandoned" if phase == "abandon" else "failed")
            assert failed[0].undefined_reasons == {"failure": type(primary).__name__}
            assert "private text" not in str(failed[0].payload())
    import quant_research_platform.research.experiments as experiments

    primary = ValueError("fit primary")

    def fail_fit(*_args, **_kwargs):
        raise primary

    def fail_record(root, plan, record):
        if record.status == "attempted":
            return append_trial_record(root, plan, record)
        from quant_research_platform.runtime.resources import (
            current_runtime_resource_policy,
        )

        policy = current_runtime_resource_policy()
        with policy.lease(
            "test.failure_record_capacity", policy.working_set_headroom_bytes
        ):
            return append_trial_record(root, plan, record)

    with monkeypatch.context() as patch:
        patch.setattr(
            trial_selection.fitting, "_fit_model_result_owned_from_frames", fail_fit
        )
        patch.setattr(experiments, "append_trial_record", fail_record)
        with pytest.raises(ValueError) as caught:
            select("run_record_failure")
    assert caught.value is primary
    assert any("trial failure record also failed" in note for note in primary.__notes__)


def test_model_trial_read_validates_metrics_without_restricting_generic_experiments(
    tmp_path: Path,
) -> None:
    from quant_research_platform.contracts.experiments import (
        trial_ledger_semantic_identity,
    )
    from quant_research_platform.storage.experiments import (
        read_model_trial_ledger_reference,
    )
    from quant_research_platform.storage.layout import StorageLayout

    layout = StorageLayout(tmp_path)
    valid = {
        "validation_rows": 3,
        "n_train": 4,
        "validation_ic": 0.2,
        "validation_rankic": 0.3,
        "validation_ic_reason": None,
        "validation_rankic_reason": None,
        "validation_mse": 1.0,
    }
    for index, bad in enumerate(
        (
            {key: value for key, value in valid.items() if key != "validation_rows"},
            {**valid, "validation_rankic_reason": "not_a_reason"},
            {**valid, "validation_mse": -1.0},
            *(
                {**valid, field: sign * float.fromhex("0x1.0000000000001p+0")}
                for field in ("validation_ic", "validation_rankic")
                for sign in (-1.0, 1.0)
            ),
        )
    ):
        plan = replace(_plan(), experiment_id=f"experiment_{index}")
        record = replace(
            _record(
                plan, event_id=f"event_{index}", attempt_index=1, status="completed"
            ),
            metrics=bad,
        )
        root = layout.trial_ledger_root(record.run_id, plan.experiment_id)
        append_trial_record(root, plan, record)
        with read_trial_ledger(root) as (_saved_plan, ledger_records_9):
            assert ledger_records_9 == (record,)
        reference = {
            "run_id": record.run_id,
            "experiment_id": plan.experiment_id,
            "protocol_hash": plan.protocol_hash,
            "semantic_identity": trial_ledger_semantic_identity(plan, (record,)),
        }
        with pytest.raises(ValueError, match="invalid_model"):
            with read_model_trial_ledger_reference(layout, reference):
                pass


def test_started_trial_survives_without_finish_and_terminal_is_identity_bound(tmp_path):
    plan = _plan()
    root = tmp_path / "ledger"
    start = _record(plan, event_id="started", attempt_index=1, status="attempted")
    started_path = append_trial_record(root, plan, start)
    original = started_path.read_bytes()
    with read_trial_ledger(root, plan) as (_saved_plan, ledger_records_6):
        assert ledger_records_6 == (start,)
    terminal = replace(start, event_id="finished", status="abandoned")
    with pytest.raises(ValueError, match="trial_terminal_identity_mismatch"):
        append_trial_record(root, plan, replace(terminal, random_seed=99))
    with read_trial_ledger(root, plan) as (_saved_plan, ledger_records_7):
        assert ledger_records_7 == (start,)
    append_trial_record(root, plan, terminal)
    assert started_path.read_bytes() == original
    with read_trial_ledger(root, plan) as (_saved_plan, ledger_records_8):
        assert ledger_records_8 == (terminal,)
    with pytest.raises(ValueError, match="trial_attempt_not_next"):
        append_trial_record(root, plan, terminal)


@pytest.mark.parametrize("stage", ("selection", "refit"))
def test_production_begin_survives_hard_exit(tmp_path, stage):
    import json
    import os
    import subprocess
    import sys
    from dataclasses import replace

    from quant_research_platform.contracts.models import ModelSpec, model_spec_payload
    from quant_research_platform.storage.layout import StorageLayout

    spec = ModelSpec(
        model_id="ridge",
        model_algorithm="ridge_regression",
        input_feature_columns=("feature",),
        label_column="label",
        random_seed=7,
        hyperparameters={"alpha": 1.0},
    )
    plan = replace(_plan(), candidate_parameters={"ridge": model_spec_payload(spec)})
    if stage == "refit":
        plan = replace(
            plan,
            stage="refit",
            selection_metric=None,
            selection_direction=None,
            tie_break_order=(),
            trial_budget=1,
            folds=plan.folds[:1],
            candidate_ids=("ridge",),
            selection_ledger_semantic_identity="b" * 64,
        )
    code = """import os, sys, json
from pathlib import Path
from quant_research_platform.contracts.experiments import ExperimentPlan
from quant_research_platform.research.experiments import TrialRecorder
from quant_research_platform.contracts.models import ModelSpec
from quant_research_platform.storage.layout import StorageLayout
plan = ExperimentPlan.from_payload(json.loads(sys.argv[2]))
recorder = TrialRecorder(StorageLayout(Path(sys.argv[1])), 'run_test', plan)
recorder.begin(ModelSpec(
    model_id='ridge', model_algorithm='ridge_regression',
    input_feature_columns=('feature',), label_column='label', random_seed=7,
    hyperparameters={'alpha': 1.0}), common_coverage_id='a' * 64)
os._exit(17)
"""
    environment = os.environ.copy()
    repository = Path(__file__).resolve().parents[2]
    environment["PYTHONPATH"] = os.pathsep.join(
        (str(repository), str(repository / "src"))
    )
    child = subprocess.run(
        [
            sys.executable,
            "-c",
            code,
            str(tmp_path),
            json.dumps(plan.protocol_payload()),
        ],
        env=environment,
        capture_output=True,
        timeout=30,
    )
    assert child.returncode == 17
    with read_trial_ledger(
        StorageLayout(tmp_path).trial_ledger_root("run_test", plan.experiment_id),
        plan,
    ) as (_saved_plan, records):
        assert len(records) == 1 and records[0].status == "attempted"


def test_run_ledger_references_have_no_arbitrary_member_limit(tmp_path):
    from quant_research_platform.storage.experiments import (
        research_run_trial_references,
    )
    from quant_research_platform.storage.layout import StorageLayout

    layout = StorageLayout(tmp_path)
    for index in range(1026):
        plan = replace(_plan(), experiment_id=f"experiment_{index:04d}")
        record = _record(plan, event_id="start", attempt_index=1, status="attempted")
        append_trial_record(
            layout.trial_ledger_root(record.run_id, plan.experiment_id), plan, record
        )
    from quant_research_platform.runtime.resources import runtime_resource_policy_scope
    from tests.helpers.runtime_resources import capture_test_policy

    policy = capture_test_policy(
        available_memory_bytes=1024**3, explicit_cap_bytes=1024
    )
    with runtime_resource_policy_scope(policy):
        with pytest.raises(
            ValueError, match="trial_ledger_working_set_exceeds_memory_capacity"
        ):
            with research_run_trial_references(layout, "run_test"):
                pass
        assert policy.used_memory_bytes == 0
    with research_run_trial_references(layout, "run_test") as references:
        assert len(references) == 1026
        assert len({item["experiment_id"] for item in references}) == 1026
    root = layout.trial_ledger_root("run_test")
    (root / "invalid").write_text("not a ledger", encoding="utf-8")
    with pytest.raises(ValueError, match="trial_run_members_invalid"):
        with research_run_trial_references(layout, "run_test"):
            pass


@pytest.mark.parametrize(
    "field",
    (
        "factor_definition_ids",
        "artifact_lineage",
        "normalized_parameters",
        "backend_versions",
        "thread_settings",
        "metrics",
        "development_test_metrics",
        "undefined_reasons",
        "cost_assumptions",
        "risk_assumptions",
    ),
)
def test_trial_json_container_types_are_not_normalized(field):
    record = _record(
        _plan(), event_id="event-shape", attempt_index=1, status="attempted"
    )
    payload = record.payload()
    assert TrialRecord.from_payload(payload).payload() == payload
    value = payload[field]
    payload[field] = (
        list(value.items())
        if isinstance(value, dict)
        else {str(item): "discarded" for item in value}
    )
    with pytest.raises(ValueError, match="trial_record_payload_invalid"):
        TrialRecord.from_payload(payload)


@pytest.mark.parametrize("cap", (1, None))
def test_trial_recorder_rejects_before_publication_and_preserves_attempt(tmp_path, cap):
    from quant_research_platform.contracts.models import ModelSpec
    from quant_research_platform.research.experiments import TrialRecorder
    from quant_research_platform.runtime.resources import runtime_resource_policy_scope
    from quant_research_platform.storage.layout import StorageLayout
    from tests.helpers.runtime_resources import capture_test_policy

    layout = StorageLayout(tmp_path)
    spec = ModelSpec(
        model_id="ridge",
        model_algorithm="ridge_regression",
        input_factor_columns=("x",),
        label_column="label",
        random_seed=7,
        hyperparameters={"alpha": 1.0},
    )
    recorder = TrialRecorder(layout, "run_low_cap", _plan())
    policy = capture_test_policy(
        available_memory_bytes=16 * 1024, explicit_cap_bytes=cap
    )
    with runtime_resource_policy_scope(policy):
        with pytest.raises(
            ValueError, match="trial_ledger_working_set_exceeds_memory_capacity"
        ):
            recorder.begin(spec, common_coverage_id="coverage")
        assert not recorder.root.exists()
        assert policy.used_memory_bytes == 0
    attempted = recorder.begin(spec, common_coverage_id="coverage")
    original = {path.name: path.read_bytes() for path in recorder.root.rglob("*.json")}
    policy = capture_test_policy(
        available_memory_bytes=16 * 1024, explicit_cap_bytes=cap
    )
    primary = KeyboardInterrupt("synthetic")
    with runtime_resource_policy_scope(policy):
        with pytest.raises(
            ValueError, match="trial_ledger_working_set_exceeds_memory_capacity"
        ):
            recorder.finish(attempted, error=primary)
        assert policy.used_memory_bytes == 0
    assert {
        path.name: path.read_bytes() for path in recorder.root.rglob("*.json")
    } == original
    with read_trial_ledger(recorder.root) as (_, records):
        assert len(records) == 1 and records[0].status == "attempted"


def test_trial_ledger_cumulative_scope_admission_and_model_overlap(
    tmp_path, monkeypatch
):
    from quant_research_platform.runtime.resources import runtime_resource_policy_scope
    from quant_research_platform.storage import manifests
    from tests.helpers.runtime_resources import capture_test_policy

    root, plan = tmp_path / "ledger", _plan()
    for index in range(1, 4):
        append_trial_record(
            root,
            plan,
            _record(
                plan, event_id=f"event-{index}", attempt_index=index, status="failed"
            ),
        )
    original = {path.name: path.read_bytes() for path in root.rglob("*.json")}
    policy = capture_test_policy(available_memory_bytes=64 * 1024**2)
    with runtime_resource_policy_scope(policy):
        with read_trial_ledger(root) as (saved, records):
            held = policy.used_memory_bytes
            assert saved == plan and len(records) == 3 and held > 0
        assert policy.used_memory_bytes == 0
        for primary in (RuntimeError("consumer"), KeyboardInterrupt("consumer")):
            with pytest.raises(type(primary)) as caught:
                with read_trial_ledger(root):
                    raise primary
            assert caught.value is primary
            assert policy.used_memory_bytes == 0
    reads = []
    read = manifests.read_bounded_qrp_json

    def observed(path, **kwargs):
        reads.append(path.name)
        return read(path, **kwargs)

    monkeypatch.setattr(manifests, "read_bounded_qrp_json", observed)
    policy = capture_test_policy(
        available_memory_bytes=64 * 1024**2, explicit_cap_bytes=held
    )
    with runtime_resource_policy_scope(policy):
        with policy.lease("research.model.selection", held // 2):
            with pytest.raises(
                ValueError, match="trial_ledger_working_set_exceeds_memory_capacity"
            ):
                with read_trial_ledger(root):
                    pytest.fail("ledger exceeded model overlap capacity")
            assert policy.used_memory_bytes == held // 2
        assert policy.used_memory_bytes == 0
    assert len(reads) < 3
    assert {path.name: path.read_bytes() for path in root.rglob("*.json")} == original


def test_first_plan_readback_precedes_first_trial_publication(tmp_path, monkeypatch):
    from quant_research_platform.storage import experiments

    plan = _plan()
    record = _record(plan, event_id="first", attempt_index=1, status="attempted")
    write = experiments.atomic_write_json

    def corrupt_first_plan(path, payload):
        if path.name == "plan.json":
            payload = {**payload, "target_id": "other-target"}
        return write(path, payload)

    monkeypatch.setattr(experiments, "atomic_write_json", corrupt_first_plan)
    root = tmp_path / "ledger"
    with pytest.raises(ValueError, match="trial_plan_mismatch"):
        append_trial_record(root, plan, record)
    assert (root / "plan.json").is_file()
    assert not tuple((root / "records").iterdir())


@pytest.mark.parametrize("member", ("plan", "record"))
@pytest.mark.parametrize("extra_byte", (0, 1))
@pytest.mark.parametrize("existing", (False, True))
def test_trial_json_disk_limit_is_checked_before_publication(
    tmp_path, member, extra_byte, existing
):
    from hashlib import sha256

    from quant_research_platform.contracts.json_codec import (
        MAX_QRP_JSON_BYTES,
        render_qrp_json_object,
    )
    from quant_research_platform.runtime.resources import runtime_resource_policy_scope

    root = tmp_path / "ledger"
    plan = _plan()
    padding = "研究" + chr(10) + '"'
    disk_size = MAX_QRP_JSON_BYTES + extra_byte
    if member == "plan":
        plan = replace(plan, input_lineage=(padding,))
        remaining = disk_size - len(
            (render_qrp_json_object(plan.protocol_payload()) + "\n").encode("utf-8")
        )
        plan = replace(plan, input_lineage=(padding + "x" * remaining,))
    record = _record(
        plan, event_id="new", attempt_index=2 if existing else 1, status="failed"
    )
    if member == "record":
        record = replace(record, artifact_lineage=(padding,))
        remaining = disk_size - len(
            (render_qrp_json_object(record.payload()) + "\n").encode("utf-8")
        )
        record = replace(record, artifact_lineage=(padding + "x" * remaining,))
    payload = plan.protocol_payload() if member == "plan" else record.payload()
    assert len((render_qrp_json_object(payload) + "\n").encode("utf-8")) == disk_size

    def identities():
        return {
            path.relative_to(root): sha256(path.read_bytes()).hexdigest()
            for path in root.rglob("*")
            if path.is_file()
        }

    with runtime_resource_policy_scope(explicit_cap_bytes=512 * 1024**2) as policy:
        if existing:
            saved_plan = _plan() if member == "plan" and extra_byte else plan
            append_trial_record(
                root,
                saved_plan,
                _record(saved_plan, event_id="first", attempt_index=1, status="failed"),
            )
        original = identities()
        if extra_byte:
            with pytest.raises(ValueError, match="^qrp_json_render_exceeds_max_bytes$"):
                append_trial_record(root, plan, record)
            assert root.exists() is existing
            assert identities() == original
        else:
            path = append_trial_record(root, plan, record)
            checked = root / "plan.json" if member == "plan" else path
            assert checked.stat().st_size == MAX_QRP_JSON_BYTES
            assert checked.read_bytes().endswith(b"\n")
            with read_trial_ledger(root, plan) as (saved, records):
                assert saved == plan
                assert records[-1] == record
                assert len(records) == (2 if existing else 1)
        assert policy.used_memory_bytes == 0


def test_trial_ledger_root_stops_real_directory_scan_at_third_member(
    tmp_path, monkeypatch
):
    from contextlib import contextmanager

    from quant_research_platform.runtime.resources import runtime_resource_policy_scope
    from quant_research_platform.storage import experiments

    root = tmp_path / "ledger"
    plan = _plan()
    append_trial_record(
        root, plan, _record(plan, event_id="first", attempt_index=1, status="failed")
    )
    for index in range(64):
        (root / f"unexpected_{index:03d}").write_bytes(b"")
    consumed = []
    closed = []
    scan = experiments.filesystem_scandir

    @contextmanager
    def observed(path):
        try:
            with scan(path) as entries:

                def counted():
                    for entry in entries:
                        consumed.append(entry.name)
                        yield entry

                yield counted()
        finally:
            closed.append(True)

    monkeypatch.setattr(experiments, "filesystem_scandir", observed)
    with runtime_resource_policy_scope() as policy:
        with pytest.raises(
            ValueError, match="^dataset_metadata_directory_member_invalid$"
        ):
            with read_trial_ledger(root):
                pytest.fail("unexpected ledger members were accepted")
        assert policy.used_memory_bytes == 0
    assert len(consumed) == 3
    assert closed == [True]
