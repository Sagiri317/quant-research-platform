from __future__ import annotations

import inspect
import os
import re
import subprocess
import sys
from pathlib import Path

from fastapi.testclient import TestClient
from typer.testing import CliRunner

import quant_research_platform.interfaces.api.app as api_app_module
import quant_research_platform.interfaces.cli.commands as cli_commands_module
import quant_research_platform.interfaces.services as services_module
from quant_research_platform.catalog.profiles import research_profile
from quant_research_platform.interfaces.api.errors import ApiError, api_error_response
from quant_research_platform.markets.implementations.a_share.ids import (
    FORWARD_RETURNS_PROCESSED_DATA_ID,
    PRICE_VOLUME_FORWARD_RETURNS_DATASET_ID,
    PRICE_VOLUME_PROCESSED_DATA_ID,
)
from quant_research_platform.research.factors.price_volume import (
    price_volume_factor_columns,
)

app = cli_commands_module.app
create_app = api_app_module.create_app
PROJECT_ROOT = Path(__file__).resolve().parents[2]

APPROVED_API_ROUTES = {
    ("GET", "/api/0.1/catalog/factors"),
    ("GET", "/api/0.1/catalog/labels"),
    ("GET", "/api/0.1/research-datasets/{dataset_id}/requirements"),
}

FORBIDDEN_PUBLIC_MARKERS = (
    "csmar",
    "provider_id",
    "source_id",
    "implementation_package",
    "quant_research_platform.sources.csmar",
    "data/raw",
    "data\\raw",
    "data/interim",
    "data\\interim",
    "data/processed",
    "data\\processed",
    "000001",
    "600000",
    "row sample",
)

RETIRED_API_FRAGMENTS = (
    "/api/0.1/catalog/data-products",
    "/api/0.1/catalog/universes",
    "/api/0.1/data-products/{data_product" + "_id}/interim-runs",
    "/markets/{market_id}/processed-data",
    "/workflows",
    "/reports/{run_id}",
)

RETIRED_CLI_COMMANDS = (
    ("catalog", "data-products"),
    ("catalog", "universes"),
    ("market", "build-processed", "--market", "a_share"),
    ("research", "build-dataset"),
    ("workflow", "run"),
    ("report", "show"),
)


def test_api_01_routes_are_exact_public_surface() -> None:
    api = create_app()
    routes = {
        (next(iter(route.methods - {"HEAD"})), route.path)
        for route in api.routes
        if route.path.startswith("/api/")
    }

    assert routes == APPROVED_API_ROUTES


def test_openapi_uses_approved_operation_ids_and_no_retired_operations() -> None:
    schema = create_app().openapi()
    operation_ids = {
        operation["operationId"]
        for path_item in schema["paths"].values()
        for method, operation in path_item.items()
        if method in {"get", "post"}
    }

    assert operation_ids == {
        "list_catalog_factors",
        "list_catalog_labels",
        "describe_research_dataset_requirements",
    }

    text = str(schema)
    for fragment in RETIRED_API_FRAGMENTS:
        assert fragment not in text


def test_api_01_endpoints_respond_without_private_markers() -> None:
    client = TestClient(create_app())

    responses = [
        client.get("/api/0.1/catalog/factors"),
        client.get("/api/0.1/catalog/labels"),
        client.get(
            "/api/0.1/research-datasets/"
            "price_volume_forward_returns/requirements"
        ),
    ]

    assert [response.status_code for response in responses] == [200] * len(responses)
    payloads = [response.json() for response in responses]
    assert {payload["api_version"] for payload in payloads} == {"0.1"}
    combined = "\n".join(str(payload) for payload in payloads)
    assert "planned" not in combined.lower()
    assert "parquet" not in combined.lower()
    _assert_public_text_has_no_private_markers(combined)

    assert len(responses[0].json()["items"]) == len(price_volume_factor_columns())
    assert responses[1].json()["items"][0]["definition_version"] == "0.1"
    requirements = responses[2].json()
    assert requirements["dataset_id"] == PRICE_VOLUME_FORWARD_RETURNS_DATASET_ID
    assert requirements["input_data_ids"][:2] == [
        PRICE_VOLUME_PROCESSED_DATA_ID,
        FORWARD_RETURNS_PROCESSED_DATA_ID,
    ]
    assert "price_volume" + "_56" not in combined


def test_api_cli_adapters_do_not_scan_parquet_paths() -> None:
    module_source = (
        inspect.getsource(api_app_module)
        + "\n"
        + inspect.getsource(cli_commands_module)
    ).lower()

    for term in ("read_parquet", ".parquet", "glob(", "rglob(", "path("):
        assert term not in module_source


def test_cli_exposes_only_clean_rewrite_public_commands() -> None:
    runner = CliRunner()

    commands = [
        ["catalog", "factors"],
        ["catalog", "labels"],
        [
            "research-datasets",
            "requirements",
            "price_volume_forward_returns",
        ],
    ]

    for command in commands:
        result = runner.invoke(app, command)
        assert result.exit_code == 0
        _assert_public_text_has_no_private_markers(result.output)

    for command in RETIRED_CLI_COMMANDS:
        assert runner.invoke(app, list(command)).exit_code != 0


def test_cli_help_exposes_approved_groups_and_no_retired_commands() -> None:
    runner = CliRunner()
    top_help = runner.invoke(app, ["--help"])
    catalog_help = runner.invoke(app, ["catalog", "--help"])
    research_datasets_help = runner.invoke(app, ["research-datasets", "--help"])

    assert top_help.exit_code == 0
    assert catalog_help.exit_code == 0
    assert research_datasets_help.exit_code == 0

    combined = "\n".join(
        [top_help.output, catalog_help.output, research_datasets_help.output]
    )
    assert "catalog" in top_help.output
    assert "research-datasets" in top_help.output
    assert "factors" in catalog_help.output
    assert "labels" in catalog_help.output
    assert "requirements" in research_datasets_help.output
    assert "data-products" not in combined
    assert "universes" not in combined
    assert "run-interim" not in combined
    assert "--run-id" not in combined
    _assert_public_text_has_no_private_markers(combined)


def test_cli_commands_module_help_invokes_existing_app() -> None:
    env = {**os.environ, "PYTHONPATH": "src", "PYTHONUTF8": "1"}
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "quant_research_platform.interfaces.cli.commands",
            "--help",
        ],
        check=False,
        capture_output=True,
        cwd=PROJECT_ROOT,
        encoding="utf-8",
        env=env,
        text=True,
    )

    assert result.returncode == 0
    assert "catalog" in result.stdout
    assert "research-datasets" in result.stdout
    assert "run-interim" not in result.stdout
    _assert_public_text_has_no_private_markers(result.stdout)


def test_api_and_cli_naming_conventions_are_contractual() -> None:
    schema = create_app().openapi()
    resource_segment_re = re.compile(r"^[a-z][a-z0-9]*(?:-[a-z0-9]+)*s$")
    snake_case_re = re.compile(r"^[a-z][a-z0-9_]*$")
    exempt_segments = {"api", "0.1", "catalog"}

    for path in schema["paths"]:
        for segment in path.strip("/").split("/"):
            if segment in exempt_segments:
                continue
            if segment.startswith("{") and segment.endswith("}"):
                assert snake_case_re.fullmatch(segment.strip("{}"))
                continue
            assert resource_segment_re.fullmatch(segment)

    requirements = services_module.describe_research_dataset_requirements(
        "price_volume_forward_returns",
        profile=research_profile(),
    )
    assert snake_case_re.fullmatch(requirements["dataset_id"])

    help_text = CliRunner().invoke(
        app,
        ["research-datasets", "requirements", "--help"],
    )
    assert help_text.exit_code == 0
    assert "research-datasets" in CliRunner().invoke(app, ["--help"]).output
    assert "requirements" in help_text.output
    assert "--dry_run" not in help_text.output
    assert "--run_id" not in help_text.output


def test_openapi_schema_does_not_expose_provider_markers() -> None:
    schema = create_app().openapi()
    hits: list[str] = []

    for path, path_item in schema["paths"].items():
        for marker in FORBIDDEN_PUBLIC_MARKERS:
            if marker.lower() in path.lower():
                hits.append(f"path:{path}:{marker}")
        for method, operation in path_item.items():
            if method not in {"get", "post", "put", "patch", "delete"}:
                continue
            searchable = [
                str(operation.get("operationId", "")),
                str(operation.get("summary", "")),
                str(operation.get("description", "")),
                " ".join(str(tag) for tag in operation.get("tags", [])),
            ]
            combined = "\n".join(searchable)
            for marker in FORBIDDEN_PUBLIC_MARKERS:
                if marker.lower() in combined.lower():
                    hits.append(f"operation:{method}:{path}:{marker}")

    for schema_name, component_schema in schema["components"]["schemas"].items():
        combined = f"{schema_name}\n{component_schema}"
        for marker in FORBIDDEN_PUBLIC_MARKERS:
            if marker.lower() in combined.lower():
                hits.append(f"schema:{schema_name}:{marker}")

    assert hits == []


def test_api_and_cli_delegate_to_same_shared_service_functions(monkeypatch) -> None:
    calls: dict[str, list[tuple[object, ...]]] = {
        "list_factors": [],
        "list_labels": [],
        "describe_research_dataset_requirements": [],
    }

    def list_factors(profile) -> dict[str, object]:
        assert profile.fields.list_names()
        calls["list_factors"].append(())
        return {
            "api_version": "0.1",
            "items": [
                {
                    "factor_id": "sentinel_factor",
                    "market_id": "a_share",
                    "factor_set": "core",
                    "definition_version": "0.1",
                }
            ],
        }

    def list_labels(profile) -> dict[str, object]:
        assert profile.fields.list_names()
        calls["list_labels"].append(())
        return {
            "api_version": "0.1",
            "items": [
                {
                    "label_id": "sentinel_label",
                    "market_id": "a_share",
                    "horizon": 1,
                    "definition_version": "0.1",
                }
            ],
        }

    def describe_research_dataset_requirements(
        dataset_id: str,
        profile,
    ) -> dict[str, object]:
        assert profile.research_datasets.list()
        calls["describe_research_dataset_requirements"].append((dataset_id,))
        return {
            "api_version": "0.1",
            "status": "sentinel_requirements",
            "dataset_id": dataset_id,
            "input_data_ids": [PRICE_VOLUME_PROCESSED_DATA_ID],
        }

    monkeypatch.setattr(services_module, "list_factors", list_factors)
    monkeypatch.setattr(services_module, "list_labels", list_labels)
    monkeypatch.setattr(
        services_module,
        "describe_research_dataset_requirements",
        describe_research_dataset_requirements,
    )

    client = TestClient(create_app())
    runner = CliRunner()
    api_and_cli = [
        (
            "list_factors",
            lambda: client.get("/api/0.1/catalog/factors"),
            lambda: runner.invoke(app, ["catalog", "factors"]),
        ),
        (
            "list_labels",
            lambda: client.get("/api/0.1/catalog/labels"),
            lambda: runner.invoke(app, ["catalog", "labels"]),
        ),
        (
            "describe_research_dataset_requirements",
            lambda: client.get(
                "/api/0.1/research-datasets/"
                "price_volume_forward_returns/requirements"
            ),
            lambda: runner.invoke(
                app,
                [
                    "research-datasets",
                    "requirements",
                    "price_volume_forward_returns",
                ],
            ),
        ),
    ]

    for service_name, call_api, call_cli in api_and_cli:
        assert call_api().status_code == 200
        cli_result = call_cli()
        assert cli_result.exit_code == 0
        assert len(calls[service_name]) == 2


def test_rejected_dataset_ids_are_sanitized() -> None:
    client = TestClient(create_app())
    runner = CliRunner()

    rejected_values = (
        "other",
        "provider_id",
        "source_id",
        "implementation_package",
    )
    for value in rejected_values:
        dataset_response = client.get(
            f"/api/0.1/research-datasets/{value}/requirements"
        )
        dataset_result = runner.invoke(
            app,
            ["research-datasets", "requirements", value],
        )

        assert dataset_response.status_code == 404
        assert dataset_result.exit_code != 0
        _assert_public_text_has_no_private_markers(str(dataset_response.json()))
        _assert_public_text_has_no_private_markers(dataset_result.output)


def test_api_error_envelope_sanitizes_private_details() -> None:
    response = api_error_response(
        ApiError(
            error_code="invalid_private_detail",
            message="failed with provider-internal detail",
            field="dataset_id",
            details={
                "traceback": "Traceback (most recent call last): provider internals",
                "credential_hint": "redacted credential detail",
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
    result = CliRunner().invoke(
        app,
        [
            "research-datasets",
            "requirements",
            "invalid-dataset-id",
        ],
    )

    assert result.exit_code != 0
    assert "C:" not in result.output
    assert "000001" not in result.output
    assert "csmar" not in result.output.lower()


def test_clean_rewrite_cli_does_not_expose_retired_commands() -> None:
    runner = CliRunner()
    retired = [
        ["private", "--help"],
        ["diagnostics", "--help"],
        ["a-share", "--help"],
        ["ml", "baseline", "train", "--help"],
        ["evaluate" + "-parquet", "--help"],
        ["factor" + "-rebuild", "--help"],
        ["prediction" + "-handoff", "--help"],
        ["compare", "--help"],
        *[list(command[:2]) + ["--help"] for command in RETIRED_CLI_COMMANDS],
    ]

    for command in retired:
        assert runner.invoke(app, command).exit_code != 0


def _assert_public_text_has_no_private_markers(text: str) -> None:
    lowered = text.lower()
    for marker in FORBIDDEN_PUBLIC_MARKERS:
        assert marker.lower() not in lowered
