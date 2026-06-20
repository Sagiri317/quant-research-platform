"""Workflow run mode contract for validation and artifact-writing paths."""

from __future__ import annotations

from enum import StrEnum


class WorkflowRunMode(StrEnum):
    VALIDATE_ONLY = "validate_only"
    WRITE_ARTIFACTS = "write_artifacts"

    @classmethod
    def default(cls) -> WorkflowRunMode:
        return cls.VALIDATE_ONLY

    @classmethod
    def normalize(cls, value: WorkflowRunMode | str | None) -> WorkflowRunMode:
        if value is None:
            return cls.default()
        if isinstance(value, cls):
            return value
        try:
            return cls(str(value))
        except ValueError as exc:
            raise ValueError(f"invalid_workflow_run_mode: {value}") from exc


__all__ = ["WorkflowRunMode"]
