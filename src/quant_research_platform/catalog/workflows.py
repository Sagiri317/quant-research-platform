"""Workflow catalog."""

from __future__ import annotations

from dataclasses import dataclass, field

from quant_research_platform.contracts.workflows import WorkflowSpec


@dataclass
class WorkflowCatalog:
    _items: dict[str, WorkflowSpec] = field(default_factory=dict)

    def register(self, spec: WorkflowSpec) -> None:
        if spec.workflow_id in self._items:
            raise ValueError(f"duplicate_workflow: {spec.workflow_id}")
        self._items[spec.workflow_id] = spec

    def get(self, workflow_id: str) -> WorkflowSpec:
        return self._items[workflow_id]

    def list(self) -> tuple[WorkflowSpec, ...]:
        return tuple(self._items[key] for key in sorted(self._items))


__all__ = ["WorkflowCatalog"]
