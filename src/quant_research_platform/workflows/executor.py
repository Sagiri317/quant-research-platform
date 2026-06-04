"""Simple topological workflow executor."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from quant_research_platform.contracts.execution import ExecutionMode
from quant_research_platform.contracts.workflows import (
    WorkflowArtifactWriterPolicy,
    WorkflowRunSummary,
    WorkflowSpec,
)
from quant_research_platform.runtime.privacy import (
    is_private_boundary_path,
    sanitize_public_object,
)
from quant_research_platform.workflows.checkpoints import InMemoryCheckpointStore
from quant_research_platform.workflows.planner import plan_topological_steps
from quant_research_platform.workflows.registry import WorkflowNodeRegistry

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
    registry: WorkflowNodeRegistry,
    checkpoints: InMemoryCheckpointStore | None = None,
    *,
    allowed_spec_ids: set[str] | None = None,
    output_root: Path | None = None,
    execution_mode: ExecutionMode | str | None = None,
    run_id: str = "workflow-fixture-run",
    include_summary: bool = False,
) -> dict[str, Any]:
    mode = ExecutionMode.normalize(execution_mode)
    _validate_workflow_execution(
        spec,
        allowed_spec_ids=allowed_spec_ids,
        output_root=output_root,
    )
    handlers = registry.handlers_for(spec.step_ids)
    results = execute_steps(
        plan_topological_steps(spec),
        handlers,
        checkpoints=checkpoints,
    )
    if include_summary:
        return {
            "results": results,
            "summary": workflow_run_summary(
                spec,
                results,
                execution_mode=mode,
                run_id=run_id,
            ),
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
    *,
    execution_mode: ExecutionMode | str | None = None,
    run_id: str = "workflow-fixture-run",
) -> dict[str, Any]:
    summary = WorkflowRunSummary(
        workflow_id=spec.workflow_id,
        workflow_version=spec.workflow_version,
        execution_mode=ExecutionMode.normalize(execution_mode),
        run_id=run_id,
        status="executed",
        market_id=str(spec.market_id) if spec.market_id else None,
        data_product_id=spec.data_product_id,
        run_type=spec.run_type,
        input_spec_ids=spec.input_spec_ids,
        output_spec_ids=spec.output_spec_ids,
        node_count=len(spec.nodes),
        completed_nodes=tuple(sorted(results)),
        private_boundary_required=spec.requires_private_boundary,
        artifact_writer_policy=spec.artifact_writer_policy.value,
    )
    return sanitize_public_object(summary.to_public_record())


__all__ = [
    "NodeHandler",
    "execute_steps",
    "execute_workflow",
    "workflow_run_summary",
]
