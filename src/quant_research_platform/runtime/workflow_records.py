"""Public-safe workflow record helpers."""

from __future__ import annotations

from typing import Any

from quant_research_platform.contracts.workflows import WorkflowRunSummary
from quant_research_platform.runtime.privacy import sanitize_public_object


def workflow_run_summary_to_public_record(
    summary: WorkflowRunSummary,
) -> dict[str, Any]:
    """Return a public-safe workflow run summary record."""
    sanitized = sanitize_public_object(summary.to_record())
    if not isinstance(sanitized, dict):
        raise TypeError("workflow_public_record_must_be_mapping")
    return sanitized


__all__ = ["workflow_run_summary_to_public_record"]
