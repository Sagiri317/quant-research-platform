"""Registered workflow node handlers."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from quant_research_platform.contracts.workflows import WorkflowRunPolicy

NodeHandler = Callable[[], Any]


@dataclass(frozen=True)
class WorkflowNodeDefinition:
    node_id: str
    handler: NodeHandler
    contract_version: str = "0.1"

    def __post_init__(self) -> None:
        if self.contract_version != "0.1":
            raise ValueError("unsupported_workflow_node_contract_version")


@dataclass
class WorkflowNodeRegistry:
    _items: dict[str, WorkflowNodeDefinition] = field(default_factory=dict)

    def register(self, definition: WorkflowNodeDefinition) -> None:
        if definition.node_id in self._items:
            raise ValueError(
                f"duplicate_workflow_node_definition: {definition.node_id}"
            )
        self._items[definition.node_id] = definition

    def get(self, node_id: str) -> WorkflowNodeDefinition:
        try:
            return self._items[node_id]
        except KeyError as exc:
            raise ValueError(f"unknown_workflow_node: {node_id}") from exc



def workflow_node_registry(
    *,
    run_policy: WorkflowRunPolicy | str | None = None,
) -> WorkflowNodeRegistry:
    policy = _normalize_workflow_run_policy(run_policy)
    if policy is WorkflowRunPolicy.WRITE_ARTIFACTS:
        raise NotImplementedError("workflow_write_artifacts_not_implemented")
    return _build_workflow_node_registry_for_validation()


def _normalize_workflow_run_policy(
    run_policy: WorkflowRunPolicy | str | None,
) -> WorkflowRunPolicy:
    if run_policy is None:
        return WorkflowRunPolicy.VALIDATE_ONLY
    return WorkflowRunPolicy.normalize(run_policy)


def _build_workflow_node_registry_for_validation() -> WorkflowNodeRegistry:
    registry = WorkflowNodeRegistry()
    for node_id in (
        "validate_interim_spec",
        "validate_processed_specs",
        "validate_inputs",
        "validate_outputs",
    ):
        registry.register(
            WorkflowNodeDefinition(
                node_id=node_id,
                handler=lambda node_id=node_id: f"{node_id}:validated",
            )
        )
    return registry


__all__ = [
    "NodeHandler",
    "WorkflowNodeDefinition",
    "WorkflowNodeRegistry",
    "workflow_node_registry",
]
