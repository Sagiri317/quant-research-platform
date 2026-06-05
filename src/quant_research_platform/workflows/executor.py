"""Simple topological workflow executor."""

from __future__ import annotations

from quant_research_platform.contracts.execution import ExecutionMode
from quant_research_platform.contracts.workflows import (
    WorkflowArtifactWriterPolicy,
    WorkflowRunSummary,
    WorkflowSpec,
)
from quant_research_platform.workflows.checkpoints import InMemoryCheckpointStore
from quant_research_platform.workflows.planner import plan_topological_steps
from quant_research_platform.workflows.registry import WorkflowNodeRegistry


def _execute_registered_steps(
    step_ids: tuple[str, ...],
    registry: WorkflowNodeRegistry,
    checkpoints: InMemoryCheckpointStore | None = None,
) -> tuple[str, ...]:
    completed: list[str] = []
    for step_id in step_ids:
        definition = registry.get(step_id)
        if checkpoints is not None:
            checkpoints.record(step_id, "running")
        definition.handler()
        if checkpoints is not None:
            checkpoints.record(step_id, "completed")
        completed.append(step_id)
    return tuple(completed)


def _validate_registered_steps(
    step_ids: tuple[str, ...],
    registry: WorkflowNodeRegistry,
) -> None:
    for step_id in step_ids:
        registry.get(step_id)


def execute_workflow(
    spec: WorkflowSpec,
    registry: WorkflowNodeRegistry,
    checkpoints: InMemoryCheckpointStore | None = None,
    *,
    allowed_spec_ids: set[str] | None = None,
    execution_mode: ExecutionMode | str | None = None,
    run_id: str = "workflow-fixture-run",
) -> WorkflowRunSummary:
    mode = ExecutionMode.normalize(execution_mode)
    _validate_workflow_execution(
        spec,
        allowed_spec_ids=allowed_spec_ids,
    )
    step_ids = plan_topological_steps(spec)
    if mode is ExecutionMode.DRY_RUN:
        _validate_registered_steps(step_ids, registry)
        return workflow_run_summary(
            spec,
            completed_nodes=(),
            execution_mode=mode,
            run_id=run_id,
            status="validated",
        )
    completed_nodes = _execute_registered_steps(
        step_ids,
        registry,
        checkpoints=checkpoints,
    )
    return workflow_run_summary(
        spec,
        completed_nodes=completed_nodes,
        execution_mode=mode,
        run_id=run_id,
    )


def _validate_workflow_execution(
    spec: WorkflowSpec,
    *,
    allowed_spec_ids: set[str] | None,
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

def workflow_run_summary(
    spec: WorkflowSpec,
    *,
    completed_nodes: tuple[str, ...],
    execution_mode: ExecutionMode | str | None = None,
    run_id: str = "workflow-fixture-run",
    status: str = "executed",
) -> WorkflowRunSummary:
    return WorkflowRunSummary(
        workflow_id=spec.workflow_id,
        workflow_version=spec.workflow_version,
        execution_mode=ExecutionMode.normalize(execution_mode),
        run_id=run_id,
        status=status,
        market_id=str(spec.market_id) if spec.market_id else None,
        data_product_id=spec.data_product_id,
        run_type=spec.run_type,
        input_spec_ids=spec.input_spec_ids,
        output_spec_ids=spec.output_spec_ids,
        node_count=len(spec.nodes),
        completed_nodes=tuple(sorted(completed_nodes)),
        private_boundary_required=spec.requires_private_boundary,
        artifact_writer_policy=spec.artifact_writer_policy.value,
    )


__all__ = [
    "execute_workflow",
    "workflow_run_summary",
]
