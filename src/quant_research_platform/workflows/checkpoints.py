"""In-memory workflow checkpoint state."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class WorkflowCheckpoint:
    node_id: str
    status: str


@dataclass
class InMemoryCheckpointStore:
    _items: dict[str, WorkflowCheckpoint] = field(default_factory=dict)

    def record(self, node_id: str, status: str) -> WorkflowCheckpoint:
        checkpoint = WorkflowCheckpoint(node_id=node_id, status=status)
        self._items[node_id] = checkpoint
        return checkpoint

    def get(self, node_id: str) -> WorkflowCheckpoint:
        return self._items[node_id]

    def list(self) -> tuple[tuple[str, str], ...]:
        return tuple((item.node_id, item.status) for item in self._items.values())


__all__ = ["InMemoryCheckpointStore", "WorkflowCheckpoint"]
