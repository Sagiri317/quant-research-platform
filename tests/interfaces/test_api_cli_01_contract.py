from __future__ import annotations

import inspect
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from typer.testing import CliRunner

import quant_research_platform.interfaces.api.app as api_app_module
import quant_research_platform.interfaces.cli.commands as cli_commands_module
import quant_research_platform.interfaces.services as services_module
from quant_research_platform.contracts.execution import ExecutionMode
from quant_research_platform.interfaces.api.errors import ApiError, api_error_response
from quant_research_platform.markets.implementations.a_share.ids import (
    A_SHARE_DEFAULT_UNIVERSE_ID,
    A_SHARE_EXECUTION_MASK_ID,
    DAILY_CONTEXT_PROCESSED_DATA_ID,
    FORWARD_RETURNS_PROCESSED_DATA_ID,
    PRICE_VOLUME_PROCESSED_DATA_ID,
)
from quant_research_platform.research.factors.price_volume import (
    price_volume_factor_columns,
)
from quant_research_platform.storage.errors import OutputAlreadyExistsError

app = cli_commands_module.app

create_app = api_app_module.create_app
PROJECT_ROOT = Path(__file__).resolve().parents[2]

APPROVED_API_ROUTES = {
    ("GET", "/api/0.1/catalog/data-products"),
    ("GET", "/api/0.1/catalog/factors"),
    ("GET", "/api/0.1/catalog/labels"),
    ("GET", "/api/0.1/catalog/universes"),
    ("POST", "/api/0.1/data-products/{data_product_id}/interim-runs"),
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
    "reports/private_real",
    "reports\\private_real",
    "000001",
    "600000",
    "row sample",
)

RETIRED_API_FRAGMENTS = (
    "/markets/{market_id}/processed-data",
    "/research-datasets",
    "/workflows",
    "/reports/{run_id}",
    "/data-products/a_share_daily_bar/interim",
)

RETIRED_CLI_COMMANDS = (
    ("data-product", "interim", "a_share_daily_bar"),
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
        "list_catalog_data_products",
        "list_catalog_factors",
        "list_catalog_labels",
        "list_catalog_universes",
        "run_data_product_interim",
        "describe_research_dataset_requirements",
    }

    text = str(schema).lower()
    stale_terms = (
        "build_dataset",
        "workflow_run",
        "report_show",
        "processed_data",
        "build-processed",
        "build-dataset",
        "workflow run",
        "report show",
        "data-product interim",
    )
    for term in stale_terms:
        assert term not in text


def test_api_01_endpoints_execute_without_private_markers_or_planned_status() -> None:
    client = TestClient(create_app())

    responses = [
        client.get("/api/0.1/catalog/factors"),
        client.get("/api/0.1/catalog/labels"),
        client.get("/api/0.1/catalog/universes"),
        client.get("/api/0.1/catalog/data-products"),
        client.post(
            "/api/0.1/data-products/a_share_daily_bar/interim-runs",
            json={"execution_mode": "dry_run"},
        ),
        client.get(
            "/api/0.1/research-datasets/"
            "a_share_price_volume_forward_returns/requirements"
        ),
    ]

    assert [response.status_code for response in responses] == [200] * len(responses)
    payloads = [response.json() for response in responses]
    assert {payload["api_version"] for payload in payloads} == {"0.1"}
    combined = "\n".join(str(payload) for payload in payloads)
    assert "planned" not in combined.lower()
    assert "parquet" not in combined.lower()
    _assert_public_text_has_no_private_markers(combined)

    factors = responses[0]
    labels = responses[1]
    universes = responses[2]
    data_products = responses[3]
    requirements = responses[5]
    assert len(factors.json()["items"]) == len(price_volume_factor_columns())
    assert labels.json()["items"][0]["definition_version"] == "0.1"
    assert universes.json()["items"][0]["universe_id"] == "a_share_default"
    assert data_products.json()["items"][0]["data_product_id"] == "a_share_daily_bar"
    assert requirements.json()["dataset_id"] == "a_share_price_volume_forward_returns"
    assert requirements.json()["required_processed_data_ids"] == [
        PRICE_VOLUME_PROCESSED_DATA_ID,
        FORWARD_RETURNS_PROCESSED_DATA_ID,
        DAILY_CONTEXT_PROCESSED_DATA_ID,
        A_SHARE_DEFAULT_UNIVERSE_ID,
        A_SHARE_EXECUTION_MASK_ID,
    ]
    assert "price_volume" + "_56" not in combined


def test_execution_mode_contract_uses_execute_default_and_dry_run_field() -> None:
    assert ExecutionMode.EXECUTE == "execute"
    assert ExecutionMode.DRY_RUN == "dry_run"
    assert ExecutionMode.default() is ExecutionMode.EXECUTE

    with pytest.raises(
        NotImplementedError,
        match="workflow_execute_not_implemented",
    ):
        services_module.run_data_product_interim("a_share_daily_bar")
    dry_run_payload = services_module.run_data_product_interim(
        "a_share_daily_bar",
        execution_mode=ExecutionMode.DRY_RUN,
    )

    assert dry_run_payload["execution_mode"] == "dry_run"
    assert dry_run_payload["status"] == "validated"


def test_data_product_interim_execute_fails_closed_in_api_and_cli() -> None:
    client = TestClient(create_app())
    runner = CliRunner()

    api_response = client.post(
        "/api/0.1/data-products/a_share_daily_bar/interim-runs"
    )
    cli_result = runner.invoke(
        app, ["data-products", "run-interim", "a_share_daily_bar"]
    )

    assert api_response.status_code == 501
    assert api_response.json()["error"]["error_code"] == (
        "workflow_execute_not_implemented"
    )
    assert api_response.json()["error"]["field"] == "execution_mode"
    assert cli_result.exit_code == 2
    assert "workflow_execute_not_implemented" in cli_result.output
    _assert_public_text_has_no_private_markers(str(api_response.json()))
    _assert_public_text_has_no_private_markers(cli_result.output)


def test_data_product_interim_dry_run_hides_provider_binding_details() -> None:
    client = TestClient(create_app())
    runner = CliRunner()

    api_response = client.post(
        "/api/0.1/data-products/a_share_daily_bar/interim-runs",
        json={"execution_mode": "dry_run"},
    )
    cli_result = runner.invoke(
        app, ["data-products", "run-interim", "a_share_daily_bar", "--dry-run"]
    )

    assert api_response.status_code == 200
    assert cli_result.exit_code == 0
    for text in (str(api_response.json()), cli_result.output):
        assert "a_share_daily_bar_interim" in text
        _assert_public_text_has_no_private_markers(text)


def test_registry_graph_validation_failure_is_public_safe(monkeypatch) -> None:
    from quant_research_platform.catalog.registry_validation import (
        CatalogConsistencyError,
    )

    def fail_validation(*_args, **_kwargs) -> None:
        raise CatalogConsistencyError()

    monkeypatch.setattr(
        services_module,
        "validate_data_product_source_graph",
        fail_validation,
    )

    api_response = TestClient(create_app()).post(
        "/api/0.1/data-products/a_share_daily_bar/interim-runs",
        json={"execution_mode": "dry_run"},
    )
    cli_result = CliRunner().invoke(
        app, ["data-products", "run-interim", "a_share_daily_bar", "--dry-run"]
    )

    assert api_response.status_code == 500
    assert api_response.json()["error"]["error_code"] == (
        "catalog_registry_graph_invalid"
    )
    assert cli_result.exit_code == 2
    assert "catalog_registry_graph_invalid" in cli_result.output
    _assert_public_text_has_no_private_markers(str(api_response.json()))
    _assert_public_text_has_no_private_markers(cli_result.output)


def test_unexpected_api_and_cli_errors_use_stable_public_summary(
    monkeypatch,
) -> None:
    def fail_with_internal_detail(*_args, **_kwargs):
        raise RuntimeError(
            "quant_research_platform.sources.csmar provider_id source_id "
            "implementation_package"
        )

    monkeypatch.setattr(
        services_module,
        "run_data_product_interim",
        fail_with_internal_detail,
    )

    api_response = TestClient(
        create_app(),
        raise_server_exceptions=False,
    ).post("/api/0.1/data-products/a_share_daily_bar/interim-runs")
    cli_result = CliRunner().invoke(
        app, ["data-products", "run-interim", "a_share_daily_bar"]
    )

    assert api_response.status_code == 500
    assert api_response.json()["error"]["error_code"] == "internal_error"
    assert api_response.json()["error"]["message"] == "internal_error"
    assert cli_result.exit_code == 2
    assert "internal_error" in cli_result.output
    _assert_public_text_has_no_private_markers(str(api_response.json()))
    _assert_public_text_has_no_private_markers(cli_result.output)


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


def test_cli_exposes_only_clean_rewrite_public_commands() -> None:
    runner = CliRunner()

    commands = [
        ["catalog", "factors"],
        ["catalog", "labels"],
        ["catalog", "universes"],
        ["catalog", "data-products"],
        ["data-products", "run-interim", "a_share_daily_bar", "--dry-run"],
        [
            "data-products",
            "run-interim",
            "a_share_daily_bar",
            "--dry-run",
            "--run-id",
            "run_01",
        ],
        [
            "research-datasets",
            "requirements",
            "a_share_price_volume_forward_returns",
        ],
    ]

    for command in commands:
        result = runner.invoke(app, command)
        assert result.exit_code == 0
        _assert_public_text_has_no_private_markers(result.output)

    for command in RETIRED_CLI_COMMANDS:
        assert runner.invoke(app, list(command)).exit_code != 0


def test_cli_help_exposes_approved_groups_options_and_no_retired_commands() -> None:
    runner = CliRunner()
    top_help = runner.invoke(app, ["--help"])
    data_products_help = runner.invoke(app, ["data-products", "--help"])
    run_interim_help = runner.invoke(app, ["data-products", "run-interim", "--help"])
    research_datasets_help = runner.invoke(app, ["research-datasets", "--help"])

    assert top_help.exit_code == 0
    assert data_products_help.exit_code == 0
    assert run_interim_help.exit_code == 0
    assert research_datasets_help.exit_code == 0

    combined = "\n".join(
        [
            top_help.output,
            data_products_help.output,
            run_interim_help.output,
            research_datasets_help.output,
        ]
    )
    assert "data-products" in top_help.output
    assert "research-datasets" in top_help.output
    assert "run-interim" in data_products_help.output
    assert "requirements" in research_datasets_help.output
    assert "--dry-run" in run_interim_help.output
    assert "--run-id" in run_interim_help.output
    assert "--execute" not in run_interim_help.output
    assert "--private-real" not in run_interim_help.output
    assert "--source-root" not in run_interim_help.output
    assert "--input-root" not in run_interim_help.output
    assert "--data-root" not in run_interim_help.output
    assert "--provider-path" not in run_interim_help.output
    for command in RETIRED_CLI_COMMANDS:
        assert " ".join(command[:2]) not in combined
    _assert_public_text_has_no_private_markers(combined)


def test_cli_commands_module_help_invokes_existing_app() -> None:
    env = {**os.environ, "PYTHONPATH": "src"}
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
        env=env,
        text=True,
    )

    assert result.returncode == 0
    assert "data-products" in result.stdout
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

    data_product = services_module.list_data_products()["items"][0]["data_product_id"]
    requirements = services_module.describe_research_dataset_requirements(
        "a_share_price_volume_forward_returns"
    )
    assert snake_case_re.fullmatch(data_product)
    assert snake_case_re.fullmatch(requirements["dataset_id"])

    request_schema = schema["components"]["schemas"]["DataProductInterimRunRequest"]
    for field_name in request_schema["properties"]:
        assert snake_case_re.fullmatch(field_name)

    help_text = CliRunner().invoke(app, ["data-products", "run-interim", "--help"])
    assert help_text.exit_code == 0
    assert "data-products" in help_text.output
    assert "run-interim" in help_text.output
    assert "--dry-run" in help_text.output
    assert "--run-id" in help_text.output
    assert "run_interim" not in help_text.output
    assert "--dry_run" not in help_text.output
    assert "--run_id" not in help_text.output


def test_api_schema_uses_execution_mode_not_dry_run_bool() -> None:
    schema = create_app().openapi()
    text = str(schema)
    request_schema = schema["components"]["schemas"]["DataProductInterimRunRequest"]

    assert "execution_mode" in text
    assert "DataProductInterimRunRequest" in text
    assert "dry_run" not in request_schema.get("properties", {})
    assert "dryRun" not in text
    assert "executionMode" not in text


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
        "list_data_products": [],
        "list_factors": [],
        "list_labels": [],
        "list_universes": [],
        "run_data_product_interim": [],
        "describe_research_dataset_requirements": [],
    }

    def list_data_products() -> dict[str, object]:
        calls["list_data_products"].append(())
        return {
            "api_version": "0.1",
            "items": [
                {
                    "data_product_id": "sentinel_product",
                    "market_id": "a_share",
                    "grain": "asset_timestamp",
                    "contract_version": "0.1",
                }
            ],
        }

    def list_factors() -> dict[str, object]:
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

    def list_labels() -> dict[str, object]:
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

    def list_universes() -> dict[str, object]:
        calls["list_universes"].append(())
        return {
            "api_version": "0.1",
            "items": [
                {
                    "universe_id": "sentinel_universe",
                    "market_id": "a_share",
                    "membership_column": "eligible",
                    "definition_version": "0.1",
                }
            ],
        }

    def run_data_product_interim(
        data_product_id: str,
        *,
        execution_mode: ExecutionMode | str | None = None,
        run_id: str | None = None,
    ) -> dict[str, object]:
        calls["run_data_product_interim"].append(
            (data_product_id, execution_mode, run_id)
        )
        return {
            "api_version": "0.1",
            "status": "sentinel_run",
            "data_product_id": data_product_id,
            "execution_mode": ExecutionMode.normalize(execution_mode).value,
            "run_id": run_id or "sentinel_run_id",
        }

    def describe_research_dataset_requirements(dataset_id: str) -> dict[str, object]:
        calls["describe_research_dataset_requirements"].append((dataset_id,))
        return {
            "api_version": "0.1",
            "status": "sentinel_requirements",
            "dataset_id": dataset_id,
            "required_processed_data_ids": [PRICE_VOLUME_PROCESSED_DATA_ID],
        }

    monkeypatch.setattr(services_module, "list_data_products", list_data_products)
    monkeypatch.setattr(services_module, "list_factors", list_factors)
    monkeypatch.setattr(services_module, "list_labels", list_labels)
    monkeypatch.setattr(services_module, "list_universes", list_universes)
    monkeypatch.setattr(
        services_module, "run_data_product_interim", run_data_product_interim
    )
    monkeypatch.setattr(
        services_module,
        "describe_research_dataset_requirements",
        describe_research_dataset_requirements,
    )

    client = TestClient(create_app())
    runner = CliRunner()
    api_and_cli = [
        (
            "list_data_products",
            lambda: client.get("/api/0.1/catalog/data-products"),
            lambda: runner.invoke(app, ["catalog", "data-products"]),
        ),
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
            "list_universes",
            lambda: client.get("/api/0.1/catalog/universes"),
            lambda: runner.invoke(app, ["catalog", "universes"]),
        ),
        (
            "run_data_product_interim",
            lambda: client.post(
                "/api/0.1/data-products/a_share_daily_bar/interim-runs",
                json={"execution_mode": "dry_run", "run_id": "same_service_run"},
            ),
            lambda: runner.invoke(
                app,
                [
                    "data-products",
                    "run-interim",
                    "a_share_daily_bar",
                    "--dry-run",
                    "--run-id",
                    "same_service_run",
                ],
            ),
        ),
        (
            "describe_research_dataset_requirements",
            lambda: client.get(
                "/api/0.1/research-datasets/"
                "a_share_price_volume_forward_returns/requirements"
            ),
            lambda: runner.invoke(
                app,
                [
                    "research-datasets",
                    "requirements",
                    "a_share_price_volume_forward_returns",
                ],
            ),
        ),
    ]

    for service_name, call_api, call_cli in api_and_cli:
        assert call_api().status_code == 200
        cli_result = call_cli()
        assert cli_result.exit_code == 0
        assert len(calls[service_name]) == 2


def test_rejected_data_product_and_dataset_ids_are_sanitized() -> None:
    client = TestClient(create_app())
    runner = CliRunner()

    rejected_values = (
        "other",
        "provider_id",
        "source_id",
        "implementation_package",
    )
    for value in rejected_values:
        product_response = client.post(
            f"/api/0.1/data-products/{value}/interim-runs",
            json={"run_id": "safe_run_id"},
        )
        dataset_response = client.get(
            f"/api/0.1/research-datasets/{value}/requirements"
        )
        product_result = runner.invoke(
            app, ["data-products", "run-interim", value, "--run-id", "safe_run_id"]
        )
        dataset_result = runner.invoke(
            app, ["research-datasets", "requirements", value]
        )

        assert product_response.status_code == 404
        assert dataset_response.status_code == 404
        assert product_result.exit_code != 0
        assert dataset_result.exit_code != 0
        _assert_public_text_has_no_private_markers(str(product_response.json()))
        _assert_public_text_has_no_private_markers(str(dataset_response.json()))
        _assert_public_text_has_no_private_markers(product_result.output)
        _assert_public_text_has_no_private_markers(dataset_result.output)


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


def test_api_request_validation_errors_are_sanitized() -> None:
    response = TestClient(create_app()).post(
        "/api/0.1/data-products/a_share_daily_bar/interim-runs",
        json={"execution_mode": r"C:\Users\DELL\data\raw\csmar\000001.SZ"},
    )
    payload = response.json()

    assert response.status_code == 422
    assert payload["error"]["error_code"] == "request_validation_error"
    assert "C:" not in str(payload)
    assert "000001" not in str(payload)
    assert "csmar" not in str(payload).lower()
    assert "input" not in str(payload).lower()


def test_cli_error_output_is_sanitized() -> None:
    result = CliRunner().invoke(
        app,
        [
            "data-products",
            "run-interim",
            r"C:\Users\DELL\000001.SZ",
        ],
    )

    assert result.exit_code != 0
    assert "C:" not in result.output
    assert "000001" not in result.output
    assert "csmar" not in result.output.lower()


def test_cli_writer_failure_summary_is_sanitized(monkeypatch) -> None:
    def fail_with_existing_output(*_args, **_kwargs):
        raise OutputAlreadyExistsError(
            r"failed at C:\Users\DELL\data\raw\csmar\000001.SZ"
        )

    monkeypatch.setattr(
        services_module,
        "run_data_product_interim",
        fail_with_existing_output,
    )
    result = CliRunner().invoke(
        app, ["data-products", "run-interim", "a_share_daily_bar"]
    )

    assert result.exit_code == 2
    assert "status" in result.output
    assert "failed" in result.output
    assert "failure_code" in result.output
    assert "output_already_exists" in result.output
    assert "C:" not in result.output
    assert "000001" not in result.output
    assert "csmar" not in result.output.lower()


def test_api_writer_failure_summary_is_sanitized(monkeypatch) -> None:
    def fail_with_existing_output(*_args, **_kwargs):
        raise OutputAlreadyExistsError(
            r"failed at C:\Users\DELL\data\raw\csmar\000001.SZ"
        )

    monkeypatch.setattr(
        services_module,
        "run_data_product_interim",
        fail_with_existing_output,
    )
    response = TestClient(create_app()).post(
        "/api/0.1/data-products/a_share_daily_bar/interim-runs"
    )
    payload = response.json()

    assert response.status_code == 409
    assert payload["error"]["error_code"] == "output_already_exists"
    assert payload["error"]["details"] == {
        "status": "failed",
        "failure_code": "output_already_exists",
    }
    assert "C:" not in str(payload)
    assert "000001" not in str(payload)
    assert "csmar" not in str(payload).lower()


def test_run_id_sanitizer_rejects_api_body_and_cli_option_markers() -> None:
    client = TestClient(create_app())
    runner = CliRunner()

    invalid_values = [
        "csmar-run",
        "provider_id-run",
        "source_id-run",
        "implementation_package-run",
        "data-raw-000001.sh",
        "reports/private_real/run",
    ]
    for value in invalid_values:
        response = client.post(
            "/api/0.1/data-products/a_share_daily_bar/interim-runs",
            json={"run_id": value},
        )
        assert response.status_code == 422
        assert response.json()["error"]["error_code"] == "invalid_run_id"
        assert response.json()["error"]["field"] == "run_id"
        _assert_public_text_has_no_private_markers(str(response.json()))

        result = runner.invoke(
            app,
            [
                "data-products",
                "run-interim",
                "a_share_daily_bar",
                "--run-id",
                value,
            ],
        )
        assert result.exit_code == 2
        _assert_public_text_has_no_private_markers(result.output)


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
