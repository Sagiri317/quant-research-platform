"""Typed target-DAG control-plane contracts."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum


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


@dataclass(frozen=True, slots=True)
class ProducerBinding:
    kind: ProducerKind
    producer: Callable[..., object]
    source_root_selector: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", ProducerKind(self.kind))
        if not callable(self.producer):
            raise ValueError("producer_callable_required")
        if self.source_root_selector is not None:
            _require_identifier(
                self.source_root_selector,
                "producer_source_root_selector_invalid",
            )


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

    def __post_init__(self) -> None:
        _require_identifier(self.target_id, "target_id_required")
        _require_identifier_tuple(
            self.upstream_target_ids,
            "target_dependencies_invalid",
        )
        if self.producer_id is not None:
            _require_identifier(self.producer_id, "target_producer_id_invalid")
        object.__setattr__(self, "decision", PlanDecision(self.decision))
        if self.decision is PlanDecision.BLOCKED and not self.blocker:
            raise ValueError("blocked_target_reason_required")
        if self.decision is not PlanDecision.BLOCKED and self.blocker is not None:
            raise ValueError("target_blocker_not_applicable")

    def payload(self) -> dict[str, object]:
        return {
            "target_id": self.target_id,
            "upstream_target_ids": list(self.upstream_target_ids),
            "requested_identity": None,
            "producer_binding": self.producer_id,
            "decision": self.decision.value,
            "blocker": self.blocker,
        }


@dataclass(frozen=True, slots=True)
class TargetResult:
    target_id: str
    status: TargetStatus
    blocker: str | None = None

    def __post_init__(self) -> None:
        _require_identifier(self.target_id, "target_id_required")
        object.__setattr__(self, "status", TargetStatus(self.status))
        if self.status is TargetStatus.BLOCKED and not self.blocker:
            raise ValueError("blocked_target_reason_required")
        if self.status is not TargetStatus.BLOCKED and self.blocker is not None:
            raise ValueError("target_blocker_not_applicable")

    def payload(self) -> dict[str, object]:
        return {
            "target_id": self.target_id,
            "status": self.status.value,
            "blocker": self.blocker,
        }


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
