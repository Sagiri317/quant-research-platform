"""Execution mode contract for public API and CLI entry points."""

from __future__ import annotations

from enum import StrEnum


class ExecutionMode(StrEnum):
    EXECUTE = "execute"
    DRY_RUN = "dry_run"

    @classmethod
    def default(cls) -> ExecutionMode:
        return cls.EXECUTE

    @classmethod
    def normalize(cls, value: ExecutionMode | str | None) -> ExecutionMode:
        if value is None:
            return cls.default()
        if isinstance(value, cls):
            return value
        try:
            return cls(str(value))
        except ValueError as exc:
            raise ValueError(f"invalid_execution_mode: {value}") from exc


__all__ = ["ExecutionMode"]
