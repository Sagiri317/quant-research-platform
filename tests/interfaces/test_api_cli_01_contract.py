from __future__ import annotations

import inspect

from fastapi.testclient import TestClient
from typer.testing import CliRunner

import quant_research_platform.interfaces.api.app as api_app_module
import quant_research_platform.interfaces.cli.commands as cli_commands_module
from quant_research_platform.interfaces.api.errors import ApiError, api_error_response
from quant_research_platform.interfaces.cli import app

create_app = api_app_module.create_app


def test_api_01_routes_are_exact_public_surface() -> None:
    api = create_app()
    routes = {
        (next(iter(route.methods - {"HEAD"})), route.path)
        for route in api.routes
        if route.path.startswith("/api/")
    }

    assert routes == {
        ("GET", "/api/0.1/catalog/factors"),
        ("GET", "/api/0.1/catalog/labels"),
        ("GET", "/api/0.1/catalog/universes"),
        ("GET", "/api/0.1/catalog/data-products"),
        ("POST", "/api/0.1/data-products/a_share_daily_bar/interim"),
        ("POST", "/api/0.1/markets/{market_id}/processed-data"),
        ("POST", "/api/0.1/research-datasets"),
        ("POST", "/api/0.1/workflows"),
        ("GET", "/api/0.1/reports/{run_id}"),
    }


def test_api_01_endpoints_execute_without_private_paths_or_planned_status() -> None:
    client = TestClient(create_app())

    responses = [
        client.get("/api/0.1/catalog/factors"),
        client.get("/api/0.1/catalog/labels"),
        client.get("/api/0.1/catalog/universes"),
        client.get("/api/0.1/catalog/data-products"),
        client.post("/api/0.1/data-products/a_share_daily_bar/interim"),
        client.post("/api/0.1/markets/a_share/processed-data"),
        client.post("/api/0.1/research-datasets"),
        client.post("/api/0.1/workflows"),
        client.get("/api/0.1/reports/example-run"),
    ]

    assert [response.status_code for response in responses] == [200] * len(responses)
    payloads = [response.json() for response in responses]
    assert {payload["api_version"] for payload in payloads} == {"0.1"}
    assert not any("parquet" in str(payload).lower() for payload in payloads)
    assert not any("planned" in str(payload).lower() for payload in payloads)
    assert not any("csmar" in str(payload).lower() for payload in payloads)
    assert not any("source_id" in str(payload).lower() for payload in payloads)
    assert not any("provider_id" in str(payload).lower() for payload in payloads)

    factors = responses[0]
    labels = responses[1]
    universes = responses[2]
    data_products = responses[3]
    assert factors.json()["api_version"] == "0.1"
    assert len(factors.json()["items"]) == 56
    assert labels.json()["items"][0]["definition_version"] == "0.1"
    assert universes.json()["items"][0]["universe_id"] == "a_share_default"
    assert data_products.json()["items"][0]["data_product_id"] == "a_share_daily_bar"


def test_api_cli_adapters_do_not_scan_parquet_paths() -> None:
    module_source = (
        inspect.getsource(api_app_module)
        + "\n"
        + inspect.getsource(cli_commands_module)
    ).lower()

    forbidden_terms = [
        "read_parquet",
        ".parquet",
        "glob(",
        "rglob(",
        "path(",
    ]
    for term in forbidden_terms:
        assert term not in module_source


def test_cli_exposes_clean_rewrite_command_groups() -> None:
    runner = CliRunner()

    commands = [
        ["catalog", "factors"],
        ["catalog", "labels"],
        ["catalog", "universes"],
        ["catalog", "data-products"],
        ["data-product", "build-interim", "--data-product", "a_share_daily_bar"],
        [
            "data-product",
            "build-interim",
            "--data-product",
            "a_share_daily_bar",
            "--dry-run",
        ],
        ["market", "build-processed", "--market", "a_share"],
        ["research", "build-dataset"],
        ["workflow", "run"],
        ["report", "show"],
    ]

    for command in commands:
        result = runner.invoke(app, command)
        assert result.exit_code == 0
        assert "csmar" not in result.output.lower()
        assert "CSMAR" not in result.output
        assert "source_id" not in result.output
        assert "provider_id" not in result.output


def test_cli_rejects_non_fixed_data_product_and_market_names() -> None:
    runner = CliRunner()

    assert runner.invoke(
        app, ["data-product", "build-interim", "--data-product", "other"]
    ).exit_code != 0
    assert runner.invoke(
        app, ["market", "build-processed", "--market", "other"]
    ).exit_code != 0


def test_api_error_envelope_sanitizes_private_details() -> None:
    response = api_error_response(
        ApiError(
            error_code="invalid_private_detail",
            message=r"failed at C:\Users\DELL\data\raw\csmar\000001.SZ",
            field="data_product_id",
            details={
                "traceback": "Traceback (most recent call last): /mnt/data/raw",
                "token": "token=abcdef123456",
                "provider_id": "csmar",
            },
        )
    )
    payload = response.body.decode("utf-8")

    assert "C:" not in payload
    assert "/mnt/" not in payload
    assert "000001" not in payload
    assert "abcdef123456" not in payload
    assert "csmar" not in payload.lower()
    assert "provider_id" not in payload


def test_cli_error_output_is_sanitized() -> None:
    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "data-product",
            "build-interim",
            "--data-product",
            r"C:\Users\DELL\000001.SZ",
        ],
    )

    assert result.exit_code != 0
    assert "C:" not in result.output
    assert "000001" not in result.output
    assert "csmar" not in result.output.lower()


def test_clean_rewrite_cli_does_not_expose_retired_commands() -> None:
    runner = CliRunner()
    retired = [
        ["a-share", "--help"],
        ["ml", "baseline", "train", "--help"],
        ["evaluate" + "-parquet", "--help"],
        ["factor" + "-rebuild", "--help"],
        ["prediction" + "-handoff", "--help"],
        ["compare", "--help"],
    ]

    for command in retired:
        assert runner.invoke(app, command).exit_code != 0
