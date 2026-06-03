"""Workflow catalog."""

from __future__ import annotations

from dataclasses import dataclass, field

from quant_research_platform.contracts.workflows import WorkflowSpec


@dataclass
class WorkflowCatalog:
    _items: dict[str, WorkflowSpec] = field(default_factory=dict)

    def register(self, spec: WorkflowSpec) -> None:
        self._items[spec.workflow_id] = spec


__all__ = ["WorkflowCatalog"]
