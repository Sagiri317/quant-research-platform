"""Workflow contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from quant_research_platform.contracts.ids import MarketId, SourceId
from quant_research_platform.contracts.outputs import OutputScope
from quant_research_platform.contracts.run_id import validate_run_id


class WorkflowRunPolicy(StrEnum):
    VALIDATE_ONLY = "validate_only"
    WRITE_ARTIFACTS = "write_artifacts"

    @classmethod
    def default(cls) -> WorkflowRunPolicy:
        return cls.VALIDATE_ONLY

    @classmethod
    def normalize(cls, value: WorkflowRunPolicy | str | None) -> WorkflowRunPolicy:
        if value is None:
            return cls.default()
        if isinstance(value, cls):
            return value
        try:
            return cls(str(value))
        except ValueError as exc:
            raise ValueError(f"invalid_workflow_run_policy: {value}") from exc


class WorkflowCheckpointPolicy(StrEnum):
    NONE = "none"
    MEMORY = "memory"


class WorkflowArtifactWriterPolicy(StrEnum):
    NONE = "none"
    SPEC_DRIVEN = "spec_driven"


def build_failed_schema_inventory_summary(
    *, failure_codes: list[str], dataset_count: int, total_row_count: int
) -> dict[str, Any]:
    value = {
        "schema_id": "qrp-schema-inventory-summary",
        "status": "FAIL",
        "failure_codes": list(failure_codes),
        "dataset_count": dataset_count,
        "total_row_count": total_row_count,
    }
    return validate_schema_inventory_summary(
        value,
        validation_status="FAIL",
        validation_failure_codes=failure_codes,
        dataset_count=dataset_count,
        total_row_count=total_row_count,
    )


def validate_schema_inventory_summary(
    value: object,
    *,
    validation_status: str,
    validation_failure_codes: list[str],
    dataset_count: int,
    total_row_count: int,
) -> dict[str, Any]:
    if (
        type(value) is not dict
        or type(validation_failure_codes) is not list
        or value.get("schema_id") != "qrp-schema-inventory-summary"
        or type(dataset_count) is not int
        or dataset_count < 0
        or type(value.get("dataset_count")) is not int
        or value.get("dataset_count") != dataset_count
        or type(total_row_count) is not int
        or total_row_count < 0
        or type(value.get("total_row_count")) is not int
        or value.get("total_row_count") != total_row_count
    ):
        raise ValueError("schema_inventory_summary_invalid")
    if validation_status == "PASS" and not validation_failure_codes:
        schema_hash = value.get("schema_hash")
        if (
            dataset_count == 0
            or set(value)
            != {"schema_id", "dataset_count", "total_row_count", "schema_hash"}
            or type(schema_hash) is not str
            or len(schema_hash) != 64
            or any(character not in "0123456789abcdef" for character in schema_hash)
        ):
            raise ValueError("schema_inventory_summary_invalid")
    elif validation_status == "FAIL" and validation_failure_codes:
        if (
            set(value)
            != {
                "schema_id",
                "status",
                "failure_codes",
                "dataset_count",
                "total_row_count",
            }
            or value.get("status") != "FAIL"
            or value.get("failure_codes") != validation_failure_codes
            or any(
                type(code) is not str or not code for code in validation_failure_codes
            )
        ):
            raise ValueError("schema_inventory_summary_invalid")
    else:
        raise ValueError("schema_inventory_summary_invalid")
    return dict(value)


@dataclass(frozen=True)
class WorkflowNode:
    node_id: str
    dependencies: tuple[str, ...] = field(default_factory=tuple)
    checkpoint_policy: WorkflowCheckpointPolicy = WorkflowCheckpointPolicy.NONE


@dataclass(frozen=True)
class WorkflowSpec:
    workflow_id: str
    step_ids: tuple[str, ...] = field(default_factory=tuple)
    nodes: tuple[WorkflowNode, ...] = field(default_factory=tuple)
    workflow_version: str = "0.1"
    market_id: MarketId | None = None
    source_id: SourceId | None = None
    input_spec_ids: tuple[str, ...] = field(default_factory=tuple)
    output_spec_ids: tuple[str, ...] = field(default_factory=tuple)
    run_type: str = "workflow"
    scope: OutputScope = OutputScope.LOCAL_ONLY
    artifact_writer_policy: WorkflowArtifactWriterPolicy = (
        WorkflowArtifactWriterPolicy.NONE
    )
    checkpoint_policy: WorkflowCheckpointPolicy = WorkflowCheckpointPolicy.NONE
    dependency_graph: dict[str, tuple[str, ...]] = field(default_factory=dict)
    contract_version: str = "0.1"

    def __post_init__(self) -> None:
        if self.contract_version != "0.1":
            raise ValueError("unsupported_workflow_contract_version")
        if self.workflow_version != "0.1":
            raise ValueError("unsupported_workflow_version")

        if self.nodes and not self.step_ids:
            object.__setattr__(
                self,
                "step_ids",
                tuple(node.node_id for node in self.nodes),
            )
        elif self.step_ids and not self.nodes:
            object.__setattr__(
                self,
                "nodes",
                tuple(WorkflowNode(node_id=step_id) for step_id in self.step_ids),
            )

        node_ids = tuple(node.node_id for node in self.nodes)
        if len(set(node_ids)) != len(node_ids):
            raise ValueError("duplicate_workflow_node")
        if self.nodes and not self.dependency_graph:
            object.__setattr__(
                self,
                "dependency_graph",
                {node.node_id: node.dependencies for node in self.nodes},
            )


@dataclass(frozen=True)
class WorkflowRunSummary:
    workflow_id: str
    workflow_version: str
    run_policy: WorkflowRunPolicy
    run_id: str
    status: str
    market_id: str | None
    run_type: str
    input_spec_ids: tuple[str, ...]
    output_spec_ids: tuple[str, ...]
    node_count: int
    completed_nodes: tuple[str, ...]
    scope: OutputScope
    artifact_writer_policy: str
    failure_code: str | None = None

    def __post_init__(self) -> None:
        validate_run_id(self.run_id)

    def to_record(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "workflow_id": self.workflow_id,
            "workflow_version": self.workflow_version,
            "run_policy": self.run_policy.value,
            "run_id": self.run_id,
            "status": self.status,
            "market_id": self.market_id,
            "run_type": self.run_type,
            "input_spec_ids": list(self.input_spec_ids),
            "output_spec_ids": list(self.output_spec_ids),
            "node_count": self.node_count,
            "completed_nodes": list(self.completed_nodes),
            "scope": self.scope.value,
            "artifact_writer_policy": self.artifact_writer_policy,
        }
        if self.failure_code is not None:
            payload["failure_code"] = self.failure_code
        return payload


__all__ = [
    "WorkflowArtifactWriterPolicy",
    "WorkflowCheckpointPolicy",
    "WorkflowNode",
    "WorkflowRunPolicy",
    "WorkflowRunSummary",
    "WorkflowSpec",
    "build_failed_schema_inventory_summary",
    "validate_schema_inventory_summary",
]
