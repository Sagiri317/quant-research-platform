"""Workflow service adapters built on registered nodes."""

from __future__ import annotations

from pathlib import Path
from typing import Any

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
) -> dict[str, Any]:
    node_registry = registry or WorkflowNodeRegistry.bootstrap_default()
    handlers = node_registry.handlers_for(spec.step_ids)
    result = execute_workflow(
        spec,
        handlers=handlers,
        allowed_spec_ids=allowed_spec_ids,
        output_root=output_root,
        include_summary=True,
    )
    return sanitize_public_object(
        {
            "status": "validated",
            **result["summary"],
        }
    )


__all__ = ["execute_registered_workflow"]
