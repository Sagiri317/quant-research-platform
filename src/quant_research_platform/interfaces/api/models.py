"""Pydantic models for the QRP 0.1 API surface."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from quant_research_platform.contracts.execution import ExecutionMode


class DataProductInterimRequest(BaseModel):
    """Request body for public data product interim workflow validation/execution."""

    execution_mode: ExecutionMode = Field(default=ExecutionMode.EXECUTE)
    run_id: str | None = None


class ApiErrorPayload(BaseModel):
    """Common API error payload."""

    error_code: str
    message: str
    field: str | None = None
    details: dict[str, Any] = Field(default_factory=dict)


class ApiErrorEnvelope(BaseModel):
    """Common FastAPI error envelope."""

    error: ApiErrorPayload

    @classmethod
    def from_error(
        cls,
        *,
        error_code: str,
        message: str,
        field: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> ApiErrorEnvelope:
        return cls(
            error=ApiErrorPayload(
                error_code=error_code,
                message=message,
                field=field,
                details=details or {},
            )
        )
