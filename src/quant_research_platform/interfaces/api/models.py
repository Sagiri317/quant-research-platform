"""FastAPI-specific error models."""

from __future__ import annotations

from typing import Literal

from pydantic import Field, model_serializer

from quant_research_platform.interfaces.models import _PublicModel


class ApiValidationError(_PublicModel):
    type: str
    loc: tuple[str | int, ...]
    msg: Literal["validation_error"] = "validation_error"


class ApiErrorDetails(_PublicModel):
    errors: list[ApiValidationError] = Field(default_factory=list)

    @model_serializer
    def _serialize(self) -> dict[str, object]:
        if not self.errors:
            return {}
        return {"errors": [item.model_dump() for item in self.errors]}


class ApiErrorPayload(_PublicModel):
    """Common API error payload."""

    error_code: str
    message: str
    field: str | None = None
    details: ApiErrorDetails = Field(default_factory=ApiErrorDetails)


class ApiErrorEnvelope(_PublicModel):
    """Common FastAPI error envelope."""

    error: ApiErrorPayload
