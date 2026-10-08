"""Typed target-DAG control-plane contracts."""

from __future__ import annotations

import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import StrEnum
from math import isfinite
from pathlib import Path, PurePosixPath, PureWindowsPath
from types import MappingProxyType

from quant_research_platform.contracts.business_manifest import (
    _freeze_json_value,
    _plain_json_value,
)
from quant_research_platform.contracts.experiments import (
    validate_trial_ledger_reference,
)
from quant_research_platform.contracts.json_codec import (
    bounded_qrp_json_object_snapshot,
    canonical_json_hash,
)
from quant_research_platform.contracts.run_id import validate_run_id
from quant_research_platform.contracts.time import TimeWindow

_MAX_RESEARCH_RUN_RECORD_BYTES = 1024 * 1024


def unmeasured_operation_evidence(reason):
    return {
        "schema_id": "qrp-run-observations",
        "status": "UNDEFINED",
        "reason": reason,
        "sampling": "target_boundaries_not_full_process_peaks",
        "sample_count": 0,
        "metrics": {},
        "environment": {"status": "UNDEFINED", "reason": reason},
    }


def _operation_evidence(value):
    result = bounded_qrp_json_object_snapshot(dict(value), max_bytes=65536)
    if (
        set(result)
        != {
            "schema_id",
            "status",
            "reason",
            "sampling",
            "sample_count",
            "metrics",
            "environment",
        }
        or result["schema_id"] != "qrp-run-observations"
        or result["sampling"] != "target_boundaries_not_full_process_peaks"
        or type(result["status"]) is not str
        or result["status"] not in {"DEFINED", "UNDEFINED"}
        or type(result["reason"]) is not str
        or type(result["sample_count"]) is not int
        or result["sample_count"] < 0
        or type(result["metrics"]) is not dict
        or type(result["environment"]) is not dict
    ):
        raise ValueError("research_run_operation_evidence_invalid")
    if result["sample_count"] == 0:
        if (
            result["status"] != "UNDEFINED"
            or not result["reason"]
            or result["metrics"]
            or result["environment"]
            != {"status": "UNDEFINED", "reason": result["reason"]}
        ):
            raise ValueError("research_run_operation_evidence_invalid")
        return result
    environment = result["environment"]
    if (
        set(environment)
        != {"created_at_utc", "platform", "host_id", "native_thread_env"}
        or type(environment["created_at_utc"]) is not str
        or type(environment["platform"]) is not str
        or not environment["platform"]
        or type(environment["native_thread_env"]) is not dict
    ):
        raise ValueError("research_run_operation_evidence_invalid")
    try:
        created_at = datetime.fromisoformat(environment["created_at_utc"])
    except ValueError as error:
        raise ValueError("research_run_operation_evidence_invalid") from error
    if created_at.utcoffset() != timedelta(0):
        raise ValueError("research_run_operation_evidence_invalid")
    if set(environment["native_thread_env"]) != {
        "OPENBLAS_NUM_THREADS",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    }:
        raise ValueError("research_run_operation_evidence_invalid")
    for item in (environment["host_id"], *environment["native_thread_env"].values()):
        if (
            type(item) is not dict
            or type(item.get("status")) is not str
            or (
                item.get("status") == "DEFINED"
                and (set(item) != {"status", "value"} or type(item["value"]) is not str)
            )
            or (
                item.get("status") == "UNDEFINED"
                and (
                    set(item) != {"status", "reason"}
                    or type(item["reason"]) is not str
                    or not item["reason"]
                )
            )
            or item.get("status") not in {"DEFINED", "UNDEFINED"}
        ):
            raise ValueError("research_run_operation_evidence_invalid")
    if environment["host_id"]["status"] == "DEFINED":
        _require_hash(
            environment["host_id"]["value"], "research_run_operation_evidence_invalid"
        )
    if set(result["metrics"]) != {
        "wall_seconds",
        "cpu_seconds",
        "read_bytes",
        "write_bytes",
        "sampled_peak_process_threads",
        "sampled_peak_child_processes",
    }:
        raise ValueError("research_run_operation_evidence_invalid")
    for metric in result["metrics"].values():
        if type(metric) is not dict:
            raise ValueError("research_run_operation_evidence_invalid")
        if metric.get("status") == "DEFINED":
            if (
                set(metric) != {"status", "value"}
                or type(metric["value"]) not in {int, float}
                or not isfinite(metric["value"])
                or metric["value"] < 0
            ):
                raise ValueError("research_run_operation_evidence_invalid")
        elif metric.get("status") == "UNDEFINED":
            if (
                set(metric) != {"status", "reason"}
                or type(metric["reason"]) is not str
                or not metric["reason"]
            ):
                raise ValueError("research_run_operation_evidence_invalid")
        else:
            raise ValueError("research_run_operation_evidence_invalid")
    if result["status"] == "DEFINED" and (
        result["reason"]
        or any(metric["status"] != "DEFINED" for metric in result["metrics"].values())
    ):
        raise ValueError("research_run_operation_evidence_invalid")
    if result["status"] == "UNDEFINED" and not result["reason"]:
        raise ValueError("research_run_operation_evidence_invalid")
    return result


class RunStatus(StrEnum):
    COMPLETE = "COMPLETE"
    BLOCKED = "BLOCKED"
    FAILED = "FAILED"


class TargetStatus(StrEnum):
    REUSED = "REUSED"
    BUILT = "BUILT"
    BLOCKED = "BLOCKED"
    FAILED = "FAILED"


class PlanDecision(StrEnum):
    REUSE = "REUSE"
    BUILD = "BUILD"
    BLOCKED = "BLOCKED"


class ProducerKind(StrEnum):
    INTERIM = "INTERIM"
    PROCESSED = "PROCESSED"
    DATASET = "DATASET"
    EVALUATION = "EVALUATION"
    MODEL = "MODEL"
    PREDICTION = "PREDICTION"
    STATISTIC = "STATISTIC"
    REPORT = "REPORT"
    PORTFOLIO = "PORTFOLIO"


@dataclass(frozen=True, slots=True)
class ProducerBinding:
    kind: ProducerKind
    producer: Callable[..., object]
    source_identity: Callable[[Path, str], str | None] | None = None
    raw_root: Path | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", ProducerKind(self.kind))
        if not callable(self.producer):
            raise ValueError("producer_callable_required")
        if self.source_identity is not None and not callable(self.source_identity):
            raise ValueError("source_identity_callable_required")
        if self.raw_root is not None and (
            not isinstance(self.raw_root, Path) or self.source_identity is None
        ):
            raise ValueError("producer_source_root_invalid")


@dataclass(frozen=True, slots=True)
class TargetSpec:
    """One DAG node with ordered semantic dependencies."""

    target_id: str
    dependencies: tuple[str, ...] = ()
    producer_id: str | None = None

    def __post_init__(self) -> None:
        _require_identifier(self.target_id, "target_id_required")
        _require_identifier_tuple(self.dependencies, "target_dependencies_invalid")
        if self.target_id in self.dependencies:
            raise ValueError("target_self_dependency")
        if self.producer_id is not None:
            _require_identifier(self.producer_id, "target_producer_id_invalid")


@dataclass(frozen=True, slots=True)
class TargetPlan:
    target_id: str
    upstream_target_ids: tuple[str, ...]
    producer_id: str | None
    decision: PlanDecision
    blocker: str | None = None
    requested_identity: str | None = None
    matching_instance_identity: str | None = None
    upstream_identities: Mapping[str, str] = MappingProxyType({})
    rebuild: bool = False
    producer_kind: ProducerKind | None = None
    schema_identity: str | None = None
    artifact_root: str | None = None

    def __post_init__(self) -> None:
        _require_identifier(self.target_id, "target_id_required")
        _require_identifier_tuple(
            self.upstream_target_ids,
            "target_dependencies_invalid",
        )
        if self.producer_id is not None:
            _require_identifier(self.producer_id, "target_producer_id_invalid")
        if self.producer_kind is not None:
            object.__setattr__(self, "producer_kind", ProducerKind(self.producer_kind))
        if self.schema_identity is not None:
            _require_hash(self.schema_identity, "target_schema_identity_invalid")
        if self.artifact_root is not None:
            _require_artifact_root(self.artifact_root, self.target_id)
        object.__setattr__(self, "decision", PlanDecision(self.decision))
        if type(self.rebuild) is not bool:
            raise ValueError("target_rebuild_invalid")
        for field_name in ("requested_identity", "matching_instance_identity"):
            value = getattr(self, field_name)
            if value is not None:
                _require_hash(value, f"{field_name}_invalid")
        if self.decision in {PlanDecision.BUILD, PlanDecision.REUSE} and (
            self.requested_identity is None
        ):
            raise ValueError("target_identity_required")
        if self.matching_instance_identity is not None:
            if self.requested_identity != self.matching_instance_identity:
                raise ValueError("matching_instance_identity_mismatch")
            if self.decision not in {PlanDecision.BUILD, PlanDecision.REUSE}:
                raise ValueError("matching_instance_decision_invalid")
            if self.decision is PlanDecision.BUILD and not self.rebuild:
                raise ValueError("build_matching_identity_requires_rebuild")
        if self.decision is PlanDecision.REUSE and (
            self.requested_identity is None or self.matching_instance_identity is None
        ):
            raise ValueError("reuse_identity_required")
        if not isinstance(self.upstream_identities, Mapping):
            raise ValueError("upstream_identities_invalid")
        upstreams = dict(self.upstream_identities)
        if any(
            type(target_id) is not str or type(identity) is not str
            for target_id, identity in upstreams.items()
        ):
            raise ValueError("upstream_identities_invalid")
        for target_id, identity in upstreams.items():
            _require_identifier(target_id, "upstream_target_id_invalid")
            _require_hash(identity, "upstream_identity_invalid")
        object.__setattr__(
            self,
            "upstream_identities",
            MappingProxyType(dict(sorted(upstreams.items()))),
        )
        if self.decision is PlanDecision.BLOCKED and not self.blocker:
            raise ValueError("blocked_target_reason_required")
        if self.decision is not PlanDecision.BLOCKED and self.blocker is not None:
            raise ValueError("target_blocker_not_applicable")

    def payload(self) -> dict[str, object]:
        return {
            "target_id": self.target_id,
            "upstream_target_ids": list(self.upstream_target_ids),
            "requested_identity": self.requested_identity,
            "matching_instance_identity": self.matching_instance_identity,
            "upstream_identities": dict(self.upstream_identities),
            "producer_binding": self.producer_id,
            "decision": self.decision.value,
            "blocker": self.blocker,
            "rebuild": self.rebuild,
        }


@dataclass(frozen=True, slots=True)
class TargetResult:
    target_id: str
    status: TargetStatus
    reason: str | None = None
    instance_identity: str | None = None

    def __post_init__(self) -> None:
        _require_identifier(self.target_id, "target_id_required")
        object.__setattr__(self, "status", TargetStatus(self.status))
        if self.instance_identity is not None:
            _require_hash(self.instance_identity, "instance_identity_invalid")
        if self.status in {TargetStatus.BUILT, TargetStatus.REUSED} and (
            self.instance_identity is None
        ):
            raise ValueError("target_instance_identity_required")
        if self.reason is not None and (
            type(self.reason) is not str or not self.reason
        ):
            raise ValueError("target_result_reason_invalid")
        if (
            self.status in {TargetStatus.BLOCKED, TargetStatus.FAILED}
            and not self.reason
        ):
            raise ValueError("target_result_reason_required")
        if self.status is TargetStatus.BLOCKED and self.instance_identity is not None:
            raise ValueError("blocked_target_identity_not_allowed")
        if self.status is TargetStatus.FAILED and self.instance_identity is not None:
            raise ValueError("failed_target_identity_not_allowed")
        if self.status in {TargetStatus.BUILT, TargetStatus.REUSED} and (
            self.reason is not None
        ):
            raise ValueError("target_result_reason_not_applicable")

    def payload(self) -> dict[str, object]:
        return {
            "target_id": self.target_id,
            "status": self.status.value,
            "reason": self.reason,
            "instance_identity": self.instance_identity,
        }


@dataclass(frozen=True, slots=True)
class ResearchRunTargetResult:
    """One run-local target outcome plus its persisted reference contract."""

    target_id: str
    producer_kind: ProducerKind | None
    status: TargetStatus
    reason: str | None = None
    instance_identity: str | None = None
    schema_identity: str | None = None
    artifact_root: str | None = None

    def __post_init__(self) -> None:
        result = TargetResult(
            target_id=self.target_id,
            status=self.status,
            reason=self.reason,
            instance_identity=self.instance_identity,
        )
        object.__setattr__(self, "status", result.status)
        if self.producer_kind is not None:
            object.__setattr__(self, "producer_kind", ProducerKind(self.producer_kind))
        if self.artifact_root is not None:
            _require_artifact_root(self.artifact_root, self.target_id)
        if self.schema_identity is not None:
            _require_hash(self.schema_identity, "research_run_schema_identity_invalid")
        if self.status in {TargetStatus.BUILT, TargetStatus.REUSED} and (
            self.schema_identity is None
            or self.producer_kind is None
            or self.artifact_root is None
        ):
            raise ValueError("research_run_target_reference_required")
        if self.status in {TargetStatus.BLOCKED, TargetStatus.FAILED} and (
            self.schema_identity is not None or self.artifact_root is not None
        ):
            raise ValueError("research_run_target_reference_not_allowed")

    def payload(self) -> dict[str, object]:
        return {
            "target_id": self.target_id,
            "producer_kind": (
                self.producer_kind.value if self.producer_kind is not None else None
            ),
            "status": self.status.value,
            "reason": self.reason,
            "instance_identity": self.instance_identity,
            "schema_identity": self.schema_identity,
            "path": self.artifact_root,
        }


@dataclass(frozen=True, slots=True)
class ResearchRunRecord:
    """One immutable terminal record for a research-run attempt."""

    run_id: str
    profile_id: str
    scenario_id: str
    protocol: Mapping[str, object]
    protocol_identity: str
    research_window: Mapping[str, object]
    software_identity: str
    terminal_targets: tuple[str, ...]
    target_results: tuple[ResearchRunTargetResult, ...]
    status: RunStatus
    operation_evidence: Mapping[str, object]
    failure_reason: str | None = None
    trial_ledgers: tuple[Mapping[str, str], ...] = ()
    holdout_evidence: Mapping[str, object] = MappingProxyType({})

    def __post_init__(self) -> None:
        validate_run_id(self.run_id)
        object.__setattr__(
            self,
            "operation_evidence",
            _freeze_json_value(_operation_evidence(self.operation_evidence)),
        )
        if type(self.trial_ledgers) is not tuple:
            raise ValueError("research_run_trial_references_invalid")
        references = tuple(
            validate_trial_ledger_reference(dict(reference))
            for reference in self.trial_ledgers
        )
        if any(reference["run_id"] != self.run_id for reference in references) or len(
            {reference["experiment_id"] for reference in references}
        ) != len(references):
            raise ValueError("research_run_trial_references_invalid")
        object.__setattr__(
            self,
            "trial_ledgers",
            tuple(MappingProxyType(reference) for reference in references),
        )
        try:
            validate_run_id(self.profile_id)
        except ValueError as exc:
            raise ValueError("research_run_profile_id_invalid") from exc
        try:
            validate_run_id(self.scenario_id)
        except ValueError as exc:
            raise ValueError("research_run_scenario_id_invalid") from exc
        protocol = bounded_qrp_json_object_snapshot(
            dict(self.protocol), max_bytes=_MAX_RESEARCH_RUN_RECORD_BYTES
        )
        _require_hash(self.protocol_identity, "research_run_protocol_identity_invalid")
        if canonical_json_hash(protocol) != self.protocol_identity:
            raise ValueError("research_run_protocol_identity_mismatch")
        window = bounded_qrp_json_object_snapshot(
            dict(self.research_window), max_bytes=_MAX_RESEARCH_RUN_RECORD_BYTES
        )
        if set(window) != {"start", "end", "access"} or window["access"] not in {
            "development",
            "holdout",
        }:
            raise ValueError("research_run_window_invalid")
        try:
            normalized_window = TimeWindow(window["start"], window["end"])
        except (TypeError, ValueError) as exc:
            raise ValueError("research_run_window_invalid") from exc
        if (
            normalized_window.start.isoformat() != window["start"]
            or normalized_window.end.isoformat() != window["end"]
        ):
            raise ValueError("research_run_window_invalid")
        _require_hash(self.software_identity, "research_run_software_identity_invalid")
        _require_identifier_tuple(
            self.terminal_targets,
            "research_run_terminal_targets_invalid",
        )
        if not self.terminal_targets:
            raise ValueError("research_run_terminal_targets_invalid")
        if type(self.target_results) is not tuple or any(
            not isinstance(result, ResearchRunTargetResult)
            for result in self.target_results
        ):
            raise ValueError("research_run_target_results_invalid")
        target_ids = tuple(result.target_id for result in self.target_results)
        if len(set(target_ids)) != len(target_ids):
            raise ValueError("research_run_target_results_invalid")
        object.__setattr__(self, "status", RunStatus(self.status))
        if self.failure_reason is not None and (
            type(self.failure_reason) is not str or not self.failure_reason
        ):
            raise ValueError("research_run_failure_reason_invalid")
        expected_status = _result_status(self.target_results, self.failure_reason)
        if self.status is not expected_status:
            raise ValueError("research_run_status_mismatch")
        if self.status is RunStatus.COMPLETE and not self.target_results:
            raise ValueError("research_run_target_results_invalid")
        if self.failure_reason is None and not set(self.terminal_targets).issubset(
            target_ids
        ):
            raise ValueError("research_run_terminal_result_missing")
        if (
            protocol.get("scenario_id") != self.scenario_id
            or protocol.get("access_window") != window["access"]
        ):
            raise ValueError("research_run_protocol_request_mismatch")
        object.__setattr__(self, "protocol", _freeze_json_value(protocol))
        object.__setattr__(self, "research_window", MappingProxyType(window))
        evidence = bounded_qrp_json_object_snapshot(
            dict(self.holdout_evidence), max_bytes=sys.maxsize
        )
        if window["access"] == "holdout" and not evidence:
            raise ValueError("research_run_holdout_evidence_invalid")
        if evidence:
            if evidence.get("qualification") == "HOLDOUT_NOT_ACCESSED" and (
                self.status is not RunStatus.BLOCKED
                or self.trial_ledgers
                or any(
                    item.status is not TargetStatus.BLOCKED
                    or item.instance_identity is not None
                    for item in self.target_results
                )
            ):
                raise ValueError("research_run_holdout_evidence_invalid")
            if (
                window["access"] != "holdout"
                or set(evidence) != {"qualification", "prior_runs"}
                or evidence["qualification"]
                not in {
                    "HOLDOUT_FIRST_EVIDENCE",
                    "HOLDOUT_NOT_ACCESSED",
                    "REUSE",
                    "POST_HOLDOUT_CORRECTED",
                    "POST_HOLDOUT_ADDITIONAL_EVIDENCE",
                }
                or type(evidence["prior_runs"]) is not list
            ):
                raise ValueError("research_run_holdout_evidence_invalid")
            for reference in evidence["prior_runs"]:
                if type(reference) is not dict or set(reference) != {
                    "run_id",
                    "identity",
                }:
                    raise ValueError("research_run_holdout_evidence_invalid")
                validate_run_id(reference["run_id"])
                _require_hash(
                    reference["identity"], "research_run_holdout_evidence_invalid"
                )
            prior_ids = [item["run_id"] for item in evidence["prior_runs"]]
            if (
                len(prior_ids) != len(set(prior_ids))
                or self.run_id in prior_ids
                or (
                    evidence["qualification"]
                    in {"HOLDOUT_FIRST_EVIDENCE", "HOLDOUT_NOT_ACCESSED"}
                )
                != (not prior_ids)
            ):
                raise ValueError("research_run_holdout_evidence_invalid")
        object.__setattr__(self, "holdout_evidence", _freeze_json_value(evidence))

    def payload(self) -> dict[str, object]:
        return {
            "schema_id": "qrp-research-run",
            "trial_ledgers": [dict(reference) for reference in self.trial_ledgers],
            "holdout_evidence": _plain_json_value(self.holdout_evidence),
            "run_id": self.run_id,
            "profile_id": self.profile_id,
            "scenario_id": self.scenario_id,
            "protocol": _plain_json_value(self.protocol),
            "protocol_identity": self.protocol_identity,
            "research_window": dict(self.research_window),
            "software_identity": self.software_identity,
            "terminal_targets": list(self.terminal_targets),
            "target_results": [result.payload() for result in self.target_results],
            "status": self.status.value,
            "failure_reason": self.failure_reason,
            "operation_evidence": _plain_json_value(self.operation_evidence),
        }


def research_run_record_from_payload(value: object) -> ResearchRunRecord:
    if type(value) is not dict or set(value) != {
        "schema_id",
        "trial_ledgers",
        "holdout_evidence",
        "run_id",
        "profile_id",
        "scenario_id",
        "protocol",
        "protocol_identity",
        "research_window",
        "software_identity",
        "terminal_targets",
        "target_results",
        "status",
        "failure_reason",
        "operation_evidence",
    }:
        raise ValueError("research_run_record_invalid")
    if value["schema_id"] != "qrp-research-run":
        raise ValueError("research_run_record_invalid")
    if (
        any(
            type(value[key]) is not list
            for key in ("target_results", "terminal_targets", "trial_ledgers")
        )
        or any(
            type(value[key]) is not dict
            for key in (
                "protocol",
                "research_window",
                "holdout_evidence",
                "operation_evidence",
            )
        )
        or any(type(reference) is not dict for reference in value["trial_ledgers"])
    ):
        raise ValueError("research_run_record_invalid")
    raw_results = value["target_results"]
    try:
        results = tuple(
            ResearchRunTargetResult(
                target_id=item["target_id"],
                producer_kind=item["producer_kind"],
                status=item["status"],
                reason=item["reason"],
                instance_identity=item["instance_identity"],
                schema_identity=item["schema_identity"],
                artifact_root=item["path"],
            )
            for item in raw_results
            if type(item) is dict
            and set(item)
            == {
                "target_id",
                "producer_kind",
                "status",
                "reason",
                "instance_identity",
                "schema_identity",
                "path",
            }
        )
        if len(results) != len(raw_results):
            raise ValueError("research_run_record_invalid")
        return ResearchRunRecord(
            run_id=value["run_id"],
            profile_id=value["profile_id"],
            scenario_id=value["scenario_id"],
            protocol=value["protocol"],
            protocol_identity=value["protocol_identity"],
            research_window=value["research_window"],
            software_identity=value["software_identity"],
            terminal_targets=tuple(value["terminal_targets"]),
            target_results=results,
            status=value["status"],
            failure_reason=value["failure_reason"],
            trial_ledgers=tuple(value["trial_ledgers"]),
            holdout_evidence=value["holdout_evidence"],
            operation_evidence=value["operation_evidence"],
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("research_run_record_invalid") from exc


def _result_status(
    results: tuple[ResearchRunTargetResult, ...], failure_reason: str | None
) -> RunStatus:
    if failure_reason is not None or any(
        result.status is TargetStatus.FAILED for result in results
    ):
        return RunStatus.FAILED
    if any(result.status is TargetStatus.BLOCKED for result in results):
        return RunStatus.BLOCKED
    return RunStatus.COMPLETE


def _require_identifier(value: object, error: str) -> None:
    if (
        type(value) is not str
        or not value
        or any(character.isspace() for character in value)
    ):
        raise ValueError(error)


def _require_identifier_tuple(value: object, error: str) -> None:
    if type(value) is not tuple or any(
        type(item) is not str
        or not item
        or any(character.isspace() for character in item)
        for item in value
    ):
        raise ValueError(error)
    if len(set(value)) != len(value):
        raise ValueError(error)


def _require_hash(value: object, error: str) -> None:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(error)


def _require_artifact_root(value: object, target_id: str) -> None:
    if type(value) is not str:
        raise ValueError("research_run_artifact_root_invalid")
    path = PurePosixPath(value)
    windows_path = PureWindowsPath(value)
    parts = path.parts
    if (
        path.is_absolute()
        or windows_path.is_absolute()
        or windows_path.drive
        or "\\" in value
        or ":" in value
        or path.as_posix() != value
        or any(part in {"", ".", ".."} for part in parts)
        or len(parts) not in {4, 5}
        or parts[0] != "data"
        or parts[-1] != target_id
        or parts[1] not in {"interim", "processed", "research", "artifacts"}
    ):
        raise ValueError("research_run_artifact_root_invalid")
