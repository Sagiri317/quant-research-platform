"""Common API error envelope helpers."""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field as dataclass_field
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from quant_research_platform.interfaces.api.models import ApiErrorEnvelope
from quant_research_platform.runtime.privacy import (
    sanitize_public_object,
    sanitize_public_text,
)


@dataclass(frozen=True)
class ApiError(Exception):
    error_code: str
    message: str
    status_code: int = 422
    field: str | None = None
    details: dict[str, Any] = dataclass_field(default_factory=dict)


def api_error_response(error: ApiError) -> JSONResponse:
    envelope = ApiErrorEnvelope.from_error(
        error_code=error.error_code,
        message=sanitize_public_text(error.message),
        field=sanitize_public_text(error.field) if error.field is not None else None,
        details=sanitize_public_object(error.details),
    )
    return JSONResponse(
        status_code=error.status_code,
        content=envelope.model_dump(),
    )


def register_api_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(ApiError)
    async def _handle_api_error(_request: Request, exc: ApiError) -> JSONResponse:
        return api_error_response(exc)

    @app.exception_handler(Exception)
    async def _handle_unexpected_error(
        _request: Request,
        exc: Exception,
    ) -> JSONResponse:
        return api_error_response(
            ApiError(
                error_code="internal_error",
                message=sanitize_public_text(repr(exc)),
                status_code=500,
            )
        )


__all__ = ["ApiError", "api_error_response", "register_api_error_handlers"]
