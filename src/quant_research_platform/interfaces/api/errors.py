"""Common API error envelope helpers."""

from __future__ import annotations

from dataclasses import dataclass

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from quant_research_platform.interfaces.api.models import (
    ApiErrorDetails,
    ApiErrorEnvelope,
    ApiErrorPayload,
    ApiValidationError,
)
from quant_research_platform.policy.share_safety import (
    sanitize_shared_text,
)


@dataclass(frozen=True)
class ApiError(Exception):
    error_code: str
    status_code: int = 422
    field: str | None = None
    details: ApiErrorDetails | None = None


def api_error_response(error: ApiError) -> JSONResponse:
    envelope = ApiErrorEnvelope(
        error=ApiErrorPayload(
            error_code=error.error_code,
            message=error.error_code,
            field=(
                sanitize_shared_text(error.field) if error.field is not None else None
            ),
            details=error.details if error.details is not None else ApiErrorDetails(),
        )
    )
    return JSONResponse(
        status_code=error.status_code,
        content=envelope.model_dump(),
    )


def register_api_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(ApiError)
    async def _handle_api_error(_request: Request, exc: ApiError) -> JSONResponse:
        return api_error_response(exc)

    @app.exception_handler(RequestValidationError)
    async def _handle_request_validation_error(
        _request: Request,
        exc: RequestValidationError,
    ) -> JSONResponse:
        return api_error_response(
            ApiError(
                error_code="request_validation_error",
                status_code=422,
                details=_validation_error_details(exc),
            )
        )

    @app.exception_handler(Exception)
    async def _handle_unexpected_error(
        _request: Request,
        exc: Exception,
    ) -> JSONResponse:
        return api_error_response(
            ApiError(
                error_code="internal_error",
                status_code=500,
            )
        )


def _validation_error_details(exc: RequestValidationError) -> ApiErrorDetails:
    errors: list[ApiValidationError] = []
    for item in exc.errors():
        raw_loc = item.get("loc", ())
        loc = (
            tuple(
                sanitize_shared_text(part) if type(part) is str else part
                for part in raw_loc
                if type(part) in {str, int}
            )[:16]
            if isinstance(raw_loc, (tuple, list))
            else ()
        )
        errors.append(
            ApiValidationError(
                type=sanitize_shared_text(str(item.get("type", "")))[:128],
                loc=loc,
            )
        )
    return ApiErrorDetails(errors=errors)
