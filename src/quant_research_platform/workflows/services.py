"""Workflow service adapters built on registered nodes."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from quant_research_platform.contracts.execution import ExecutionMode
from quant_research_platform.contracts.workflows import WorkflowSpec
from quant_research_platform.runtime.privacy import sanitize_public_object
from quant_research_platform.workflows.executor import execute_workflow
from quant_research_platform.workflows.registry import WorkflowNodeRegistry


def execute_registered_workflow(
    spec: WorkflowSpec,
    *,
    registry: WorkflowNodeRegistry | None = None,
    allowed_spec_ids: set[str] | None = None,
    output_root: Path | None = None,
    execution_mode: ExecutionMode | str | None = None,
    run_id: str = "workflow-fixture-run",
) -> dict[str, Any]:
    node_registry = registry or WorkflowNodeRegistry.bootstrap_default()
    mode = ExecutionMode.normalize(execution_mode)
    result = execute_workflow(
        spec,
        registry=node_registry,
        allowed_spec_ids=allowed_spec_ids,
        output_root=output_root,
        execution_mode=mode,
        run_id=run_id,
        include_summary=True,
    )
    return sanitize_public_object(
        {
            **result["summary"],
            "status": "executed" if mode is ExecutionMode.EXECUTE else "validated",
        }
    )


__all__ = ["execute_registered_workflow"]
