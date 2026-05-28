from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient

from quant_research_platform.interfaces.api.errors import (
    ApiError,
    api_error_response,
    register_api_error_handlers,
)
from quant_research_platform.interfaces.api.models import ApiErrorEnvelope


def test_api_error_envelope_model_shape() -> None:
    envelope = ApiErrorEnvelope.from_error(
        error_code="QRP-ARTIFACT-001",
        message="schema contract failed",
        field="manifest.display.tables[0]",
        details={"expected": "0.7"},
    )

    assert envelope.model_dump() == {
        "error": {
            "error_code": "QRP-ARTIFACT-001",
            "message": "schema contract failed",
            "field": "manifest.display.tables[0]",
            "details": {"expected": "0.7"},
        }
    }


def test_api_error_response_uses_common_json_shape() -> None:
    response = api_error_response(
        ApiError(
            error_code="QRP-VALIDATION-001",
            message="invalid request",
            status_code=422,
            field="diagnostics.materialized_views[0]",
        )
    )

    assert response.status_code == 422
    assert response.body == (
        b'{"error":{"error_code":"QRP-VALIDATION-001","message":"invalid request",'
        b'"field":"diagnostics.materialized_views[0]","details":{}}}'
    )


def test_fastapi_handler_registers_common_error_shape() -> None:
    app = FastAPI()
    register_api_error_handlers(app)

    @app.get("/boom")
    def boom() -> None:
        raise ApiError(
            error_code="QRP-ARTIFACT-002",
            message="missing run",
            status_code=404,
            field=None,
        )

    response = TestClient(app).get("/boom")

    assert response.status_code == 404
    assert response.json() == {
        "error": {
            "error_code": "QRP-ARTIFACT-002",
            "message": "missing run",
            "field": None,
            "details": {},
        }
    }
