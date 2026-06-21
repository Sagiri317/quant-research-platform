"""Workflow service adapters built on registered nodes."""

from __future__ import annotations

from quant_research_platform.contracts.workflow_run import WorkflowRunPolicy
from quant_research_platform.contracts.workflows import WorkflowSpec
from quant_research_platform.runtime.workflow_records import (
    workflow_run_summary_to_public_record,
)
from quant_research_platform.workflows.executor import run_workflow
from quant_research_platform.workflows.registry import (
    WorkflowNodeRegistry,
    workflow_node_registry,
)


def run_registered_workflow(
    spec: WorkflowSpec,
    *,
    registry: WorkflowNodeRegistry | None = None,
    allowed_spec_ids: set[str] | None = None,
    run_policy: WorkflowRunPolicy | str | None = None,
    run_id: str = "workflow-validation-run",
) -> dict[str, object]:
    policy = _normalize_workflow_run_policy(run_policy)
    node_registry = registry or workflow_node_registry(run_policy=policy)
    summary = run_workflow(
        spec,
        registry=node_registry,
        allowed_spec_ids=allowed_spec_ids,
        run_policy=policy,
        run_id=run_id,
    )
    public_record = workflow_run_summary_to_public_record(summary)
    return {
        **public_record,
        "status": (
            "artifact_writing_not_implemented"
            if policy is WorkflowRunPolicy.WRITE_ARTIFACTS
            else "validated"
        ),
    }


__all__ = ["run_registered_workflow"]


def _normalize_workflow_run_policy(
    run_policy: WorkflowRunPolicy | str | None,
) -> WorkflowRunPolicy:
    if run_policy is None:
        return WorkflowRunPolicy.VALIDATE_ONLY
    return WorkflowRunPolicy.normalize(run_policy)
