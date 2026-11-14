"""Translate query failures into consistent HTTP status codes and sanitized error
bodies.
"""

from __future__ import annotations

from dataclasses import dataclass

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse


@dataclass
class ApiError(Exception):
    error_code: str
    status_code: int = 422
    field: str | None = None
    details: dict[str, object] | None = None


def api_error_response(error: ApiError) -> JSONResponse:
    return JSONResponse(
        status_code=error.status_code,
        content={
            "error": {
                "error_code": error.error_code,
                "message": error.error_code,
                "field": error.field,
                "details": error.details if error.details is not None else {},
            }
        },
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
                details={
                    "errors": [
                        {
                            "type": item["type"],
                            "loc": item["loc"],
                            "msg": "validation_error",
                        }
                        for item in exc.errors()
                    ]
                },
            )
        )

    @app.exception_handler(Exception)
    async def _handle_unexpected_error(
        _request: Request,
        _exc: Exception,
    ) -> JSONResponse:
        return api_error_response(
            ApiError(
                error_code="internal_error",
                status_code=500,
            )
        )
