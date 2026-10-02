"""Append-only experiment records owned by storage."""

from __future__ import annotations

import sys
from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import fields, is_dataclass, replace
from pathlib import Path

from quant_research_platform.contracts.experiments import (
    ExperimentPlan,
    TrialRecord,
    trial_ledger_semantic_identity,
    validate_trial_ledger_reference,
    validate_trial_record_against_plan,
)
from quant_research_platform.contracts.json_codec import (
    MAX_QRP_JSON_BYTES,
    bounded_rendered_qrp_json_object_size,
    json_record_working_set_bytes,
)
from quant_research_platform.contracts.models import (
    validate_completed_model_trial_metrics,
)
from quant_research_platform.runtime.resources import runtime_resource_policy_scope
from quant_research_platform.storage.atomic import (
    atomic_write_directory,
    atomic_write_json,
)
from quant_research_platform.storage.identity import (
    filesystem_exists,
    filesystem_mkdir,
    filesystem_scandir,
    require_unaliased_path,
)
from quant_research_platform.storage.layout import StorageLayout
from quant_research_platform.storage.manifests import (
    _qrp_json_file_identity,
    read_bounded_qrp_json,
    read_bounded_qrp_json_directory,
)


def append_trial_record(
    root: Path,
    plan: ExperimentPlan,
    record: TrialRecord,
) -> Path:
    if not isinstance(record, TrialRecord):
        raise TypeError("trial_record_required")
    with runtime_resource_policy_scope() as policy:
        initial = 16 * (_trial_payload_size(plan) + _trial_payload_size(record))
        policy.require_working_set(
            initial, error="trial_ledger_working_set_exceeds_memory_capacity"
        )
        with policy.lease("storage.trial.append", initial) as lease:
            plan_payload, record_payload = plan.protocol_payload(), record.payload()
            plan_bytes, record_bytes = (
                bounded_rendered_qrp_json_object_size(
                    payload, lazy_arrays={}, max_bytes=MAX_QRP_JSON_BYTES - 1
                )
                + 1
                for payload in (plan_payload, record_payload)
            )
            required = sum(
                json_record_working_set_bytes(size)
                for size in (plan_bytes, record_bytes)
            )
            if required > lease.reserved_bytes:
                additional = required - lease.reserved_bytes
                policy.require_working_set(
                    additional, error="trial_ledger_working_set_exceeds_memory_capacity"
                )
                lease.grow(additional)
            validate_trial_record_against_plan(plan, record)
            if not filesystem_exists(root):
                first_read_bytes = (
                    1024
                    + 2 * json_record_working_set_bytes(plan_bytes)
                    + sum(
                        3 * sys.getsizeof(name) + 256
                        for name in ("plan.json", "records")
                    )
                )
                policy.require_working_set(
                    first_read_bytes,
                    error="trial_ledger_working_set_exceeds_memory_capacity",
                )

                def write(staging: Path) -> None:
                    atomic_write_json(staging / "plan.json", plan_payload)
                    filesystem_mkdir(staging / "records")

                atomic_write_directory(root, write)
            with read_trial_ledger(root, plan) as (_saved, existing):
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
                suffix = "-terminal" if finishing else ""
                path = (
                    Path(root) / "records" / f"{record.attempt_index:08d}{suffix}.json"
                )
                return atomic_write_json(path, record_payload)


def _trial_payload_size(value: object) -> int:
    """Bound copies of a borrowed typed contract before rendering its JSON payload."""
    size = sys.getsizeof(value)
    if is_dataclass(value):
        return size + sum(
            sys.getsizeof(field.name) + _trial_payload_size(getattr(value, field.name))
            for field in fields(value)
        )
    if isinstance(value, Mapping):
        return size + sum(
            sys.getsizeof(key) + _trial_payload_size(item)
            for key, item in value.items()
        )
    if isinstance(value, (tuple, list)):
        return size + sum(_trial_payload_size(item) for item in value)
    return size


@contextmanager
def read_trial_ledger(root: Path, plan: ExperimentPlan | None = None):
    """Own cumulative ledger decode and projections until the consumer exits."""
    root = Path(root)
    with runtime_resource_policy_scope() as policy:
        policy.require_working_set(
            1024, error="trial_ledger_working_set_exceeds_memory_capacity"
        )
        with policy.lease("storage.trial.ledger", 1024) as lease:
            if not filesystem_exists(root):
                yield plan, ()
                return
            root = require_unaliased_path(root, error="trial_ledger_path_invalid")
            members = set()
            with filesystem_scandir(root) as entries:
                for item in entries:
                    if len(members) == 2:
                        raise ValueError("dataset_metadata_directory_member_invalid")
                    additional = 3 * sys.getsizeof(item.name) + 256
                    policy.require_working_set(
                        additional,
                        error="trial_ledger_working_set_exceeds_memory_capacity",
                    )
                    lease.grow(additional)
                    members.add(item.name)
            if members != {"plan.json", "records"}:
                raise ValueError("dataset_metadata_directory_member_invalid")
            identity = _qrp_json_file_identity(root / "plan.json")
            additional = 2 * json_record_working_set_bytes(identity.size)
            policy.require_working_set(
                additional, error="trial_ledger_working_set_exceeds_memory_capacity"
            )
            lease.grow(additional)
            saved = ExperimentPlan.from_payload(
                read_bounded_qrp_json(root / "plan.json", expected_identity=identity)[1]
            )
            if plan is not None and saved.protocol_payload() != plan.protocol_payload():
                raise ValueError("trial_plan_mismatch")
            plan = saved
            entries = read_bounded_qrp_json_directory(
                root / "records", max_members=2 * plan.trial_budget, lease=lease
            )
            events = {
                name: TrialRecord.from_payload(payload) for name, payload in entries
            }
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
            yield saved, records


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
    if expected.payload() != terminal.payload():
        raise ValueError("trial_terminal_identity_mismatch")


@contextmanager
def read_trial_ledger_reference(layout: StorageLayout, reference: dict[str, str]):
    reference = validate_trial_ledger_reference(reference)
    root = layout.trial_ledger_root(reference["run_id"], reference["experiment_id"])
    with read_trial_ledger(root) as (plan, records):
        if plan is None:
            raise FileNotFoundError("trial_ledger_missing")
        if (
            plan.protocol_hash != reference["protocol_hash"]
            or any(record.run_id != reference["run_id"] for record in records)
            or trial_ledger_semantic_identity(plan, records)
            != reference["semantic_identity"]
        ):
            raise ValueError("trial_ledger_reference_mismatch")
        yield plan, records


@contextmanager
def research_run_trial_references(layout: StorageLayout, run_id: str):
    root = require_unaliased_path(
        layout.trial_ledger_root(run_id), error="trial_ledger_path_invalid"
    )
    if not filesystem_exists(root):
        yield ()
        return
    with runtime_resource_policy_scope() as policy:
        with policy.lease("storage.trial_run.references", 0) as lease:
            references = []
            with filesystem_scandir(root) as entries:
                for entry in entries:
                    if not entry.is_dir(follow_symlinks=False):
                        raise ValueError("trial_run_members_invalid")
                    name = entry.name
                    experiment_root = layout.trial_ledger_root(run_id, name)
                    with read_trial_ledger(experiment_root) as (plan, records):
                        if plan.experiment_id != name or any(
                            record.run_id != run_id for record in records
                        ):
                            raise ValueError("trial_run_members_invalid")
                        reference = {
                            "run_id": run_id,
                            "experiment_id": name,
                            "protocol_hash": plan.protocol_hash,
                            "semantic_identity": trial_ledger_semantic_identity(
                                plan, records
                            ),
                        }
                        additional = json_record_working_set_bytes(
                            bounded_rendered_qrp_json_object_size(
                                reference, lazy_arrays={}, max_bytes=sys.maxsize
                            )
                        )
                        policy.require_working_set(
                            additional,
                            error="trial_ledger_working_set_exceeds_memory_capacity",
                        )
                        lease.grow(additional)
                    references.append(reference)
            references.sort(key=lambda reference: reference["experiment_id"])
            yield tuple(references)


@contextmanager
def read_model_trial_ledger_reference(layout: StorageLayout, reference: dict[str, str]):
    with read_trial_ledger_reference(layout, reference) as (plan, records):
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
        yield plan, records
