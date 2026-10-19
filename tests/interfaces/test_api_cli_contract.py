"""Read-only API bounds and direct catalog behavior."""

import pytest
from httpx import ASGITransport, AsyncClient

from quant_research_platform.interfaces.api.app import create_app


@pytest.mark.anyio
@pytest.mark.parametrize("anyio_backend", ["asyncio"])
async def test_catalog_queries_and_missing_root_are_read_only(anyio_backend):
    async with AsyncClient(
        transport=ASGITransport(app=create_app()), base_url="http://testserver"
    ) as client:
        for endpoint in ("factors", "labels"):
            response = await client.get(f"/api/catalog/{endpoint}")
            assert response.status_code == 200
            assert response.json()["items"]
            if endpoint == "factors":
                assert all(
                    not {
                        "window",
                        "base_input_columns",
                        "derived_input_columns",
                    }.intersection(item)
                    for item in response.json()["items"]
                )
        response = await client.get("/api/research-datasets/pv_all/requirements")
        assert response.status_code == 200
        assert response.json()["mode"] == "requirements_query_only"
        assert (
            await client.get("/api/inspect", params={"path": "model"})
        ).status_code == 503
        assert (await client.post("/api/fit", json={})).status_code == 404


@pytest.mark.anyio
@pytest.mark.parametrize("anyio_backend", ["asyncio"])
async def test_api_cannot_escape_root_or_read_raw(anyio_backend, tmp_path):
    root = tmp_path / "project"
    root.mkdir()
    (root / "output").mkdir()
    (root / "output" / "manifest.json").write_text('{"kind":"synthetic"}')
    async with AsyncClient(
        transport=ASGITransport(app=create_app(project_root=root)),
        base_url="http://testserver",
    ) as client:
        for path in (
            "../outside",
            "/outside",
            "C:/private",
            "data/raw/input",
            "..\\private",
        ):
            response = await client.get("/api/inspect", params={"path": path})
            assert response.status_code == 422
        response = await client.get("/api/inspect", params={"path": "output"})
        assert response.status_code == 200
        assert response.json() == {"kind": "synthetic"}
        assert (
            await client.get("/api/inspect", params={"path": "missing"})
        ).status_code == 404
        assert (
            await client.post("/api/inspect", json={"path": "output"})
        ).status_code == 405


@pytest.mark.anyio
@pytest.mark.parametrize("anyio_backend", ["asyncio"])
async def test_api_does_not_grant_browser_cors(anyio_backend):
    async with AsyncClient(
        transport=ASGITransport(app=create_app()), base_url="http://testserver"
    ) as client:
        response = await client.get(
            "/api/catalog/labels", headers={"Origin": "http://localhost:3000"}
        )
        assert "access-control-allow-origin" not in response.headers


def test_routes_only_query_explicit_evidence():
    assert {
        (method, route.path)
        for route in create_app().routes
        if route.path.startswith("/api/")
        for method in route.methods
    } == {
        ("GET", "/api/catalog/factors"),
        ("GET", "/api/catalog/labels"),
        ("GET", "/api/research-datasets/{dataset_id}/requirements"),
        ("GET", "/api/inspect"),
        ("GET", "/api/portfolio-curve"),
    }


@pytest.mark.anyio
@pytest.mark.parametrize("anyio_backend", ["asyncio"])
async def test_http_inspection_redacts_private_paths_without_changing_artifact(
    anyio_backend, tmp_path
):
    import json

    output = tmp_path / "output"
    output.mkdir()
    manifest = output / "manifest.json"
    facts = {
        "kind": "synthetic",
        "inputs": [{"path": "/private/project/data.parquet"}],
        "rows": 3,
    }
    manifest.write_text(json.dumps(facts))
    async with AsyncClient(
        transport=ASGITransport(app=create_app(project_root=tmp_path)),
        base_url="http://testserver",
    ) as client:
        response = await client.get("/api/inspect", params={"path": "output"})
        assert response.status_code == 200
        assert response.json()["inputs"] == [{"path": "[local-path]"}]
        assert response.json()["rows"] == 3
        assert json.loads(manifest.read_text()) == facts
        manifest.write_text(json.dumps({"api_key": "sensitive-value"}))
        response = await client.get("/api/inspect", params={"path": "output"})
        assert response.status_code == 422
        assert "sensitive-value" not in response.text


@pytest.mark.anyio
@pytest.mark.parametrize("anyio_backend", ["asyncio"])
async def test_error_handlers_preserve_envelope_and_redact_payload(anyio_backend):
    import json

    from fastapi.exceptions import RequestValidationError

    app = create_app()
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://testserver"
    ) as client:
        response = await client.get(
            "/api/portfolio-curve", params={"path": "unused", "stride": "private-input"}
        )
    assert response.status_code == 422
    assert response.json() == {
        "error": {
            "error_code": "request_validation_error",
            "message": "request_validation_error",
            "field": None,
            "details": {
                "errors": [
                    {
                        "type": "int_parsing",
                        "loc": ["query", "stride"],
                        "msg": "validation_error",
                    }
                ]
            },
        }
    }
    response = await app.exception_handlers[RequestValidationError](
        None,
        RequestValidationError(
            [
                {
                    "type": "x" * 129,
                    "loc": [True, None, *range(20)],
                    "msg": "private-msg",
                    "input": "private-input",
                    "ctx": {"private": 1},
                }
            ]
        ),
    )
    assert json.loads(response.body)["error"]["details"] == {
        "errors": [
            {"type": "x" * 128, "loc": list(range(16)), "msg": "validation_error"}
        ]
    }
    response = await app.exception_handlers[Exception](None, RuntimeError("private"))
    assert response.status_code == 500
    assert json.loads(response.body) == {
        "error": {
            "error_code": "internal_error",
            "message": "internal_error",
            "field": None,
            "details": {},
        }
    }
