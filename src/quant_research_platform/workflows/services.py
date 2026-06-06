"""Workflow service adapters built on registered nodes."""

from __future__ import annotations

from quant_research_platform.contracts.execution import ExecutionMode
from quant_research_platform.contracts.workflows import WorkflowSpec
from quant_research_platform.runtime.workflow_records import (
    workflow_run_summary_to_public_record,
)
from quant_research_platform.workflows.executor import execute_workflow
from quant_research_platform.workflows.registry import WorkflowNodeRegistry


def execute_registered_workflow(
    spec: WorkflowSpec,
    *,
    registry: WorkflowNodeRegistry | None = None,
    allowed_spec_ids: set[str] | None = None,
    execution_mode: ExecutionMode | str | None = None,
    run_id: str = "workflow-fixture-run",
) -> dict[str, object]:
    node_registry = registry or WorkflowNodeRegistry.bootstrap_default()
    mode = ExecutionMode.normalize(execution_mode)
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
