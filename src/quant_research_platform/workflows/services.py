"""Workflow service adapters built on registered nodes."""

from __future__ import annotations

from quant_research_platform.contracts.execution import ExecutionMode
from quant_research_platform.contracts.workflows import WorkflowSpec
from quant_research_platform.runtime.workflow_records import (
    workflow_run_summary_to_public_record,
)
from quant_research_platform.workflows.executor import execute_workflow
from quant_research_platform.workflows.registry import (
    WorkflowNodeRegistry,
    workflow_node_registry,
)


def execute_registered_workflow(
    spec: WorkflowSpec,
    *,
    registry: WorkflowNodeRegistry | None = None,
    allowed_spec_ids: set[str] | None = None,
    execution_mode: ExecutionMode | str | None = None,
    run_id: str = "workflow-dry-run",
) -> dict[str, object]:
    mode = _normalize_workflow_execution_mode(execution_mode)
    node_registry = registry or workflow_node_registry(execution_mode=mode)
    summary = execute_workflow(
        spec,
        registry=node_registry,
        allowed_spec_ids=allowed_spec_ids,
        execution_mode=mode,
        run_id=run_id,
    )
    public_record = workflow_run_summary_to_public_record(summary)
    return {
        **public_record,
        "status": "executed" if mode is ExecutionMode.EXECUTE else "validated",
    }


__all__ = ["execute_registered_workflow"]


def _normalize_workflow_execution_mode(
    execution_mode: ExecutionMode | str | None,
) -> ExecutionMode:
    if execution_mode is None:
        return ExecutionMode.DRY_RUN
    return ExecutionMode.normalize(execution_mode)
