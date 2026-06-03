"""Workflow contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from quant_research_platform.contracts.ids import MarketId, SourceId


class WorkflowCheckpointPolicy(StrEnum):
    NONE = "none"
    MEMORY = "memory"


class WorkflowArtifactWriterPolicy(StrEnum):
    NONE = "none"
    SPEC_DRIVEN = "spec_driven"


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
    requires_private_boundary: bool = True
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


__all__ = [
    "WorkflowArtifactWriterPolicy",
    "WorkflowCheckpointPolicy",
    "WorkflowNode",
    "WorkflowSpec",
]
