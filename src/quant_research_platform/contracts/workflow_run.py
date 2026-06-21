"""Workflow run policy contract for validation and artifact-writing paths."""

from __future__ import annotations

from enum import StrEnum


class WorkflowRunPolicy(StrEnum):
    VALIDATE_ONLY = "validate_only"
    WRITE_ARTIFACTS = "write_artifacts"

    @classmethod
    def default(cls) -> WorkflowRunPolicy:
        return cls.VALIDATE_ONLY

    @classmethod
    def normalize(cls, value: WorkflowRunPolicy | str | None) -> WorkflowRunPolicy:
        if value is None:
            return cls.default()
        if isinstance(value, cls):
            return value
        try:
            return cls(str(value))
        except ValueError as exc:
            raise ValueError(f"invalid_workflow_run_policy: {value}") from exc


__all__ = ["WorkflowRunPolicy"]
