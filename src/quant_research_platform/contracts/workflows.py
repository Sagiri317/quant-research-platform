"""Workflow contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum


class WorkflowCheckpointPolicy(StrEnum):
    NONE = "none"
    MEMORY = "memory"


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
    contract_version: str = "0.1"

    def __post_init__(self) -> None:
        if self.contract_version != "0.1":
            raise ValueError("unsupported_workflow_contract_version")

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


__all__ = ["WorkflowCheckpointPolicy", "WorkflowNode", "WorkflowSpec"]
