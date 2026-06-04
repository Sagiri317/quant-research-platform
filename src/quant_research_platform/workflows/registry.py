"""Registered workflow node handlers."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

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

    def handlers_for(self, node_ids: tuple[str, ...]) -> dict[str, NodeHandler]:
        return {node_id: self.get(node_id).handler for node_id in node_ids}

    @classmethod
    def bootstrap_default(cls) -> WorkflowNodeRegistry:
        registry = cls()
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


__all__ = ["NodeHandler", "WorkflowNodeDefinition", "WorkflowNodeRegistry"]
