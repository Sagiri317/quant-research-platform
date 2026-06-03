"""Simple topological workflow executor."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from quant_research_platform.contracts.workflows import WorkflowSpec
from quant_research_platform.workflows.checkpoints import InMemoryCheckpointStore
from quant_research_platform.workflows.planner import plan_topological_steps

NodeHandler = Callable[[], Any]


def execute_steps(
    step_ids: tuple[str, ...],
    handlers: dict[str, NodeHandler],
    checkpoints: InMemoryCheckpointStore | None = None,
) -> dict[str, Any]:
    results: dict[str, Any] = {}
    for step_id in step_ids:
        if step_id not in handlers:
            raise ValueError(f"missing_workflow_handler: {step_id}")
        if checkpoints is not None:
            checkpoints.record(step_id, "running")
        results[step_id] = handlers[step_id]()
        if checkpoints is not None:
            checkpoints.record(step_id, "completed")
    return results


def execute_workflow(
    spec: WorkflowSpec,
    handlers: dict[str, NodeHandler],
    checkpoints: InMemoryCheckpointStore | None = None,
) -> dict[str, Any]:
    return execute_steps(
        plan_topological_steps(spec),
        handlers,
        checkpoints=checkpoints,
    )


__all__ = ["NodeHandler", "execute_steps", "execute_workflow"]
