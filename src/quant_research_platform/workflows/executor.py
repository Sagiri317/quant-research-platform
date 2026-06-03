"""Simple topological workflow executor."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from quant_research_platform.contracts.workflows import (
    WorkflowArtifactWriterPolicy,
    WorkflowSpec,
)
from quant_research_platform.runtime.privacy import (
    is_private_boundary_path,
    sanitize_public_object,
)
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
    *,
    allowed_spec_ids: set[str] | None = None,
    output_root: Path | None = None,
    include_summary: bool = False,
) -> dict[str, Any]:
    _validate_workflow_execution(
        spec,
        allowed_spec_ids=allowed_spec_ids,
        output_root=output_root,
    )
    results = execute_steps(
        plan_topological_steps(spec),
        handlers,
        checkpoints=checkpoints,
    )
    if include_summary:
        return {
            "results": results,
            "summary": workflow_run_summary(spec, results),
        }
    return results


def _validate_workflow_execution(
    spec: WorkflowSpec,
    *,
    allowed_spec_ids: set[str] | None,
    output_root: Path | None,
) -> None:
    if allowed_spec_ids is not None:
        requested = set(spec.input_spec_ids).union(spec.output_spec_ids)
        unknown = sorted(requested.difference(allowed_spec_ids))
        if unknown:
            raise ValueError(f"unknown_workflow_spec_id: {', '.join(unknown)}")
    if (
        spec.artifact_writer_policy == WorkflowArtifactWriterPolicy.SPEC_DRIVEN
        and not spec.output_spec_ids
    ):
        raise ValueError("workflow_output_spec_required")
    if output_root is not None and spec.requires_private_boundary:
        if not is_private_boundary_path(output_root):
            raise ValueError("workflow_output_outside_private_boundary")


def workflow_run_summary(
    spec: WorkflowSpec,
    results: dict[str, Any],
) -> dict[str, Any]:
    return sanitize_public_object(
        {
            "workflow_id": spec.workflow_id,
            "workflow_version": spec.workflow_version,
            "market_id": str(spec.market_id) if spec.market_id else None,
            "source_id": str(spec.source_id) if spec.source_id else None,
            "run_type": spec.run_type,
            "input_spec_ids": list(spec.input_spec_ids),
            "output_spec_ids": list(spec.output_spec_ids),
            "node_count": len(spec.nodes),
            "completed_nodes": sorted(results),
            "private_boundary_required": spec.requires_private_boundary,
            "artifact_writer_policy": spec.artifact_writer_policy.value,
        }
    )


__all__ = [
    "NodeHandler",
    "execute_steps",
    "execute_workflow",
    "workflow_run_summary",
]
