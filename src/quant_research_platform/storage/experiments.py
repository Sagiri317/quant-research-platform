"""Append-only experiment records owned by storage."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from quant_research_platform.contracts.experiments import (
    ExperimentPlan,
    TrialRecord,
    trial_ledger_semantic_identity,
    validate_trial_ledger_reference,
    validate_trial_record_against_plan,
)
from quant_research_platform.contracts.json_codec import (
    canonical_identity_bytes,
    qrp_json_object_snapshot,
)
from quant_research_platform.contracts.models import (
    validate_completed_model_trial_metrics,
)
from quant_research_platform.policy.share_safety import (
    evidence_json_value_unsafe_reason,
)
from quant_research_platform.storage.atomic import (
    atomic_write_directory,
    atomic_write_json,
)
from quant_research_platform.storage.identity import (
    filesystem_exists,
    filesystem_mkdir,
    filesystem_scandir,
    filesystem_stat,
    require_unaliased_path,
)
from quant_research_platform.storage.layout import StorageLayout
from quant_research_platform.storage.manifests import (
    _directory_identity,
    _qrp_json_file_identity,
    read_bounded_qrp_json_directory,
    read_qrp_json_file,
)


def append_trial_record(root: Path, plan: ExperimentPlan, record: TrialRecord) -> Path:
    if not isinstance(record, TrialRecord):
        raise TypeError("trial_record_required")
    plan_payload = qrp_json_object_snapshot(plan.protocol_payload())
    record_payload = qrp_json_object_snapshot(record.payload())
    validate_trial_record_against_plan(plan, record)
    if evidence_json_value_unsafe_reason([plan_payload, record_payload]) is not None:
        raise ValueError("trial_evidence_unsafe")
    if not filesystem_exists(root):

        def write(staging: Path) -> None:
            atomic_write_json(staging / "plan.json", plan_payload)
            filesystem_mkdir(staging / "records")

        atomic_write_directory(root, write)
    _saved, existing = read_trial_ledger(root, plan)
    finishing = (
        bool(existing)
        and record.attempt_index == len(existing)
        and (existing[-1].status == "attempted")
        and (record.status != "attempted")
    )
    if finishing:
        _validate_trial_completion(existing[-1], record)
    elif record.attempt_index != len(existing) + 1:
        raise ValueError("trial_attempt_not_next")
    suffix = "-terminal" if finishing else ""
    path = Path(root) / "records" / f"{record.attempt_index:08d}{suffix}.json"
    return atomic_write_json(path, record_payload)


def read_trial_ledger(root: Path, plan: ExperimentPlan | None = None):
    """Read and validate the complete append-only trial ledger."""
    root = Path(root)
    if not filesystem_exists(root):
        return (plan, ())
    root = require_unaliased_path(root, error="trial_ledger_path_invalid")
    root_identity = _directory_identity(filesystem_stat(root, follow_symlinks=False))
    members = set()
    with filesystem_scandir(root) as entries:
        for item in entries:
            if len(members) == 2:
                raise ValueError("dataset_metadata_directory_member_invalid")
            members.add(item.name)
    if members != {"plan.json", "records"}:
        raise ValueError("dataset_metadata_directory_member_invalid")
    identity = _qrp_json_file_identity(root / "plan.json")
    plan_payload = read_qrp_json_file(root / "plan.json", expected_identity=identity)[1]
    if evidence_json_value_unsafe_reason(plan_payload) is not None:
        raise ValueError("trial_evidence_unsafe")
    saved = ExperimentPlan.from_payload(plan_payload)
    if plan is not None and canonical_identity_bytes(
        saved.protocol_payload()
    ) != canonical_identity_bytes(plan.protocol_payload()):
        raise ValueError("trial_plan_mismatch")
    plan = saved
    records_root = root / "records"
    records_identity = _directory_identity(
        filesystem_stat(records_root, follow_symlinks=False)
    )
    record_identities = {}
    entries = read_bounded_qrp_json_directory(
        records_root,
        max_members=2 * plan.trial_budget,
        member_identities=record_identities,
    )
    if (
        evidence_json_value_unsafe_reason([payload for _, payload in entries])
        is not None
    ):
        raise ValueError("trial_evidence_unsafe")
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
    if _qrp_json_file_identity(root / "plan.json") != identity or any(
        _qrp_json_file_identity(records_root / name) != expected
        for name, expected in record_identities.items()
    ):
        raise ValueError("dataset_metadata_file_identity_changed")
    for directory, expected_identity, expected_names in (
        (root, root_identity, members),
        (records_root, records_identity, set(record_identities)),
    ):
        with filesystem_scandir(directory) as current:
            names = set()
            for item in current:
                if len(names) == len(expected_names):
                    raise ValueError("dataset_metadata_directory_identity_changed")
                names.add(item.name)
            if names != expected_names:
                raise ValueError("dataset_metadata_directory_identity_changed")
        if (
            _directory_identity(filesystem_stat(directory, follow_symlinks=False))
            != expected_identity
        ):
            raise ValueError("dataset_metadata_directory_identity_changed")
    return (saved, records)


def _validate_trial_completion(start: TrialRecord, terminal: TrialRecord) -> None:
    if start.status != "attempted" or terminal.status == "attempted":
        raise ValueError("trial_terminal_transition_invalid")
    expected = replace(
        start,
        event_id=terminal.event_id,
        status=terminal.status,
        metrics=terminal.metrics,
        thread_settings=terminal.thread_settings,
        undefined_reasons=terminal.undefined_reasons,
    )
    if canonical_identity_bytes(expected.payload()) != canonical_identity_bytes(
        terminal.payload()
    ):
        raise ValueError("trial_terminal_identity_mismatch")


def read_trial_ledger_reference(layout: StorageLayout, reference: dict[str, str]):
    reference = validate_trial_ledger_reference(reference)
    root = layout.trial_ledger_root(reference["run_id"], reference["experiment_id"])
    plan, records = read_trial_ledger(root)
    if plan is None:
        raise FileNotFoundError("trial_ledger_missing")
    if (
        plan.experiment_id != reference["experiment_id"]
        or plan.protocol_hash != reference["protocol_hash"]
        or any(record.run_id != reference["run_id"] for record in records)
        or trial_ledger_semantic_identity(plan, records)
        != reference["semantic_identity"]
    ):
        raise ValueError("trial_ledger_reference_mismatch")
    return (plan, records)


def research_run_trial_references(layout: StorageLayout, run_id: str):
    root = require_unaliased_path(
        layout.trial_ledger_root(run_id), error="trial_ledger_path_invalid"
    )
    if not filesystem_exists(root):
        return ()
    references = []
    with filesystem_scandir(root) as entries:
        for entry in entries:
            if not entry.is_dir(follow_symlinks=False):
                raise ValueError("trial_run_members_invalid")
            name = entry.name
            experiment_root = layout.trial_ledger_root(run_id, name)
            plan, records = read_trial_ledger(experiment_root)
            if plan.experiment_id != name or any(
                record.run_id != run_id for record in records
            ):
                raise ValueError("trial_run_members_invalid")
            reference = {
                "run_id": run_id,
                "experiment_id": name,
                "protocol_hash": plan.protocol_hash,
                "semantic_identity": trial_ledger_semantic_identity(plan, records),
            }
            references.append(reference)
    references.sort(key=lambda reference: reference["experiment_id"])
    return tuple(references)


def read_model_trial_ledger_reference(layout: StorageLayout, reference: dict[str, str]):
    plan, records = read_trial_ledger_reference(layout, reference)
    for record in records:
        if record.status == "completed":
            if plan.stage == "selection":
                validate_completed_model_trial_metrics(
                    record.candidate_id, record.metrics
                )
            elif (
                set(record.metrics) != {"n_train"}
                or type(record.metrics["n_train"]) is not int
                or record.metrics["n_train"] <= 0
            ):
                raise ValueError("refit_metrics_invalid")
    return (plan, records)
