"""Append-only experiment records owned by storage."""

from __future__ import annotations

import os
from dataclasses import replace
from pathlib import Path

from quant_research_platform.contracts.experiments import (
    ExperimentPlan,
    TrialRecord,
    trial_ledger_semantic_identity,
    validate_trial_ledger_reference,
    validate_trial_record_against_plan,
)
from quant_research_platform.contracts.models import (
    validate_completed_model_trial_metrics,
)
from quant_research_platform.storage.atomic import (
    atomic_write_directory,
    atomic_write_json,
)
from quant_research_platform.storage.identity import require_unaliased_path
from quant_research_platform.storage.layout import StorageLayout
from quant_research_platform.storage.manifests import (
    read_bounded_qrp_json_directory,
    read_manifest,
)


def append_trial_record(
    root: Path,
    plan: ExperimentPlan,
    record: TrialRecord,
) -> Path:
    if not isinstance(record, TrialRecord):
        raise TypeError("trial_record_required")
    validate_trial_record_against_plan(plan, record)
    if not root.exists():

        def write(staging: Path) -> None:
            atomic_write_json(staging / "plan.json", plan.protocol_payload())
            (staging / "records").mkdir()

        atomic_write_directory(root, write)
    existing = read_trial_ledger(root, plan)
    finishing = (
        bool(existing)
        and record.attempt_index == len(existing)
        and existing[-1].status == "attempted"
        and record.status != "attempted"
    )
    if finishing:
        _validate_trial_completion(existing[-1], record)
    elif record.attempt_index != len(existing) + 1:
        raise ValueError("trial_attempt_not_next")
    if any(item.development_test_contaminated for item in existing):
        raise ValueError("trial_contaminated_experiment_requires_new_id")
    suffix = "-terminal" if finishing else ""
    path = Path(root) / "records" / f"{record.attempt_index:08d}{suffix}.json"
    return atomic_write_json(path, record.payload())


def read_trial_ledger(
    root: Path,
    plan: ExperimentPlan | None = None,
) -> tuple[TrialRecord, ...]:
    root = Path(root)
    if not root.exists():
        return ()
    saved = read_trial_plan(root)
    if plan is not None and saved.protocol_payload() != plan.protocol_payload():
        raise ValueError("trial_plan_mismatch")
    plan = saved
    entries = read_bounded_qrp_json_directory(
        root / "records", max_members=2 * plan.trial_budget
    )
    events = {name: TrialRecord.from_payload(payload) for name, payload in entries}
    starts = {
        record.attempt_index: record
        for name, record in events.items()
        if name == f"{record.attempt_index:08d}.json"
    }
    for name, record in events.items():
        if name == f"{record.attempt_index:08d}.json":
            continue
        if (
            name != f"{record.attempt_index:08d}-terminal.json"
            or record.attempt_index not in starts
        ):
            raise ValueError("trial_ledger_filename_mismatch")
        _validate_trial_completion(starts[record.attempt_index], record)
    records = tuple(
        events.get(f"{index:08d}-terminal.json", starts[index])
        for index in sorted(starts)
    )
    indices = tuple(record.attempt_index for record in records)
    if indices != tuple(range(1, len(records) + 1)):
        raise ValueError("trial_ledger_order_invalid")
    for record in records:
        validate_trial_record_against_plan(plan, record)
    if any(record.development_test_contaminated for record in records[:-1]):
        raise ValueError("trial_contaminated_experiment_reused")
    return records


def _validate_trial_completion(start: TrialRecord, terminal: TrialRecord) -> None:
    if start.status != "attempted" or terminal.status == "attempted":
        raise ValueError("trial_terminal_transition_invalid")
    expected = replace(
        start,
        event_id=terminal.event_id,
        status=terminal.status,
        validation_metrics=terminal.validation_metrics,
        thread_settings=terminal.thread_settings,
        undefined_reasons=terminal.undefined_reasons,
    )
    if expected.payload() != terminal.payload():
        raise ValueError("trial_terminal_identity_mismatch")


def read_trial_plan(root: Path) -> ExperimentPlan:
    root = require_unaliased_path(root, error="trial_ledger_path_invalid")
    if {item.name for item in root.iterdir()} != {"plan.json", "records"}:
        raise ValueError("dataset_metadata_directory_member_invalid")
    return ExperimentPlan.from_payload(read_manifest(root / "plan.json"))


def read_trial_ledger_reference(
    layout: StorageLayout, reference: dict[str, str]
) -> tuple[ExperimentPlan, tuple[TrialRecord, ...]]:
    reference = validate_trial_ledger_reference(reference)
    root = layout.trial_ledger_root(reference["run_id"], reference["experiment_id"])
    plan = read_trial_plan(root)
    records = read_trial_ledger(root, plan)
    if (
        plan.protocol_hash != reference["protocol_hash"]
        or any(record.run_id != reference["run_id"] for record in records)
        or trial_ledger_semantic_identity(plan, records)
        != reference["semantic_identity"]
    ):
        raise ValueError("trial_ledger_reference_mismatch")
    return plan, records


def research_run_trial_references(
    layout: StorageLayout, run_id: str
) -> tuple[dict[str, str], ...]:
    root = require_unaliased_path(
        layout.trial_ledger_root(run_id), error="trial_ledger_path_invalid"
    )
    if not root.exists():
        return ()
    names = []
    with os.scandir(root) as entries:
        for entry in entries:
            if len(names) >= 1024 or not entry.is_dir(follow_symlinks=False):
                raise ValueError("trial_run_members_invalid")
            names.append(entry.name)
    references = []
    for name in sorted(names):
        experiment_root = layout.trial_ledger_root(run_id, name)
        plan = read_trial_plan(experiment_root)
        records = read_trial_ledger(experiment_root, plan)
        if plan.experiment_id != name or any(
            record.run_id != run_id for record in records
        ):
            raise ValueError("trial_run_members_invalid")
        references.append(
            {
                "run_id": run_id,
                "experiment_id": name,
                "protocol_hash": plan.protocol_hash,
                "semantic_identity": trial_ledger_semantic_identity(plan, records),
            }
        )
    return tuple(references)


def read_model_trial_ledger_reference(
    layout: StorageLayout, reference: dict[str, str]
) -> tuple[ExperimentPlan, tuple[TrialRecord, ...]]:
    plan, records = read_trial_ledger_reference(layout, reference)
    for record in records:
        if record.status == "completed":
            validate_completed_model_trial_metrics(
                record.candidate_id, record.validation_metrics
            )
    return plan, records
