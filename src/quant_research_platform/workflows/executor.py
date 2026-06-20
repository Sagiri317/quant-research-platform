"""Simple topological workflow executor."""

from __future__ import annotations

from quant_research_platform.contracts.execution import WorkflowRunMode
from quant_research_platform.contracts.workflows import (
    WorkflowArtifactWriterPolicy,
    WorkflowRunSummary,
    WorkflowSpec,
)
from quant_research_platform.workflows.checkpoints import InMemoryCheckpointStore
from quant_research_platform.workflows.planner import plan_topological_steps
from quant_research_platform.workflows.registry import (
    WorkflowNodeRegistry,
    workflow_node_registry,
)


def _validate_registered_steps(
    step_ids: tuple[str, ...],
    registry: WorkflowNodeRegistry,
) -> None:
    for step_id in step_ids:
        registry.get(step_id)


def execute_workflow(
    spec: WorkflowSpec,
    registry: WorkflowNodeRegistry | None = None,
    checkpoints: InMemoryCheckpointStore | None = None,
    *,
    allowed_spec_ids: set[str] | None = None,
    run_mode: WorkflowRunMode | str | None = None,
    run_id: str = "workflow-dry-run",
) -> WorkflowRunSummary:
    mode = _normalize_workflow_run_mode(run_mode)
    if mode is WorkflowRunMode.WRITE_ARTIFACTS:
        raise NotImplementedError("workflow_write_artifacts_not_implemented")
    node_registry = registry or workflow_node_registry(run_mode=mode)
    _validate_workflow_run(
        spec,
        allowed_spec_ids=allowed_spec_ids,
    )
    step_ids = plan_topological_steps(spec)
    if mode is WorkflowRunMode.VALIDATE_ONLY:
        _validate_registered_steps(step_ids, node_registry)
        return workflow_run_summary(
            spec,
            completed_nodes=(),
            run_mode=mode,
            run_id=run_id,
            status="validated",
        )
    raise NotImplementedError("workflow_write_artifacts_not_implemented")


def _validate_workflow_run(
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
    run_mode: WorkflowRunMode | str | None = None,
    run_id: str = "workflow-dry-run",
    status: str = "validated",
) -> WorkflowRunSummary:
    return WorkflowRunSummary(
        workflow_id=spec.workflow_id,
        workflow_version=spec.workflow_version,
        run_mode=WorkflowRunMode.normalize(run_mode),
        run_id=run_id,
        status=status,
        market_id=str(spec.market_id) if spec.market_id else None,
        run_type=spec.run_type,
        input_spec_ids=spec.input_spec_ids,
        output_spec_ids=spec.output_spec_ids,
        node_count=len(spec.nodes),
        completed_nodes=tuple(sorted(completed_nodes)),
        private_boundary_required=spec.requires_private_boundary,
        artifact_writer_policy=spec.artifact_writer_policy.value,
    )


def _normalize_workflow_run_mode(
    run_mode: WorkflowRunMode | str | None,
) -> WorkflowRunMode:
    if run_mode is None:
        return WorkflowRunMode.VALIDATE_ONLY
    return WorkflowRunMode.normalize(run_mode)


__all__ = [
    "execute_workflow",
    "workflow_run_summary",
]
