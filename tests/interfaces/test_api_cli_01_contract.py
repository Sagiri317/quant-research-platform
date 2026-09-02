from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from typer.main import get_command
from typer.testing import CliRunner

from quant_research_platform.interfaces.api.app import create_app
from quant_research_platform.interfaces.cli import commands as cli_commands
from quant_research_platform.interfaces.cli.commands import app
from quant_research_platform.interfaces.queries import list_factors
from quant_research_platform.profiles.research import research_profile

FORBIDDEN_PUBLIC_MARKERS = (
    "provider_id",
    "source_id",
    "data/raw",
    "data\\raw",
    "data/interim",
    "data\\interim",
    "data/processed",
    "data\\processed",
    "000001",
    "600000",
)


def test_api_routes_are_current_and_exact() -> None:
    api = create_app()
    routes = {
        (next(iter(route.methods - {"HEAD"})), route.path)
        for route in api.routes
        if route.path.startswith("/api/")
    }
    assert routes == {
        ("GET", "/api/0.1/catalog/factors"),
        ("GET", "/api/0.1/catalog/labels"),
        ("GET", "/api/0.1/research-datasets/{dataset_id}/requirements"),
    }


def test_api_catalog_and_requirements_use_current_a_share_protocol() -> None:
    client = TestClient(create_app())
    responses = (
        client.get("/api/0.1/catalog/factors"),
        client.get("/api/0.1/catalog/labels"),
        client.get("/api/0.1/research-datasets/pv_all/requirements"),
    )
    assert [response.status_code for response in responses] == [200, 200, 200]
    factor_items = responses[0].json()["items"]
    assert len(factor_items) == len(list_factors(research_profile()).items)
    assert {item["output_column"] for item in factor_items} >= {
        "reversal_1d",
        "log_float_market_cap",
    }
    requirements = responses[2].json()
    assert requirements["dataset_id"] == "pv_all"
    assert requirements["input_data_ids"] == [
        "market_panel_1d",
        "price_volume_facts",
        "forward_returns",
    ]
    _assert_public_text_is_sanitized(
        "\n".join(str(response.json()) for response in responses)
    )


def test_openapi_has_closed_current_response_models() -> None:
    schema = create_app().openapi()
    assert set(schema["paths"]) == {
        "/api/0.1/catalog/factors",
        "/api/0.1/catalog/labels",
        "/api/0.1/research-datasets/{dataset_id}/requirements",
    }
    assert {
        name: schema["components"]["schemas"][name]["additionalProperties"]
        for name in (
            "FactorCatalogResponse",
            "FactorCatalogItem",
            "LabelCatalogResponse",
            "LabelCatalogItem",
            "ResearchDatasetRequirementsResponse",
        )
    } == {
        "FactorCatalogResponse": False,
        "FactorCatalogItem": False,
        "LabelCatalogResponse": False,
        "LabelCatalogItem": False,
        "ResearchDatasetRequirementsResponse": False,
    }


def test_cli_exposes_only_current_research_surface() -> None:
    command = get_command(app)
    assert set(command.commands) == {
        "catalog",
        "research-datasets",
        "data",
        "dataset",
        "research",
        "report",
        "status",
        "inspect",
    }
    runner = CliRunner()
    for args in (
        ("catalog", "factors"),
        ("catalog", "labels"),
        ("research-datasets", "requirements", "pv_all"),
        ("data", "materialize", "--plan"),
        (
            "dataset",
            "build",
            "pv_all",
            "--start-date",
            "2019-01-01",
            "--end-date",
            "2019-12-31",
            "--plan",
        ),
        (
            "research",
            "run",
            "--start-date",
            "2019-01-01",
            "--end-date",
            "2019-12-31",
            "--plan",
        ),
        ("report", "build", "--plan"),
    ):
        result = runner.invoke(app, list(args))
        assert result.exit_code == 0, result.output
        _assert_public_text_is_sanitized(result.output)

    research_help = runner.invoke(app, ["research", "run", "--help"])
    assert research_help.exit_code == 0, research_help.output
    assert "--holdout" in research_help.output
    assert "--phase" not in research_help.output
    assert "--protocol-frozen" not in research_help.output

    report_help = runner.invoke(app, ["report", "build", "--help"])
    assert report_help.exit_code == 0, report_help.output
    assert "--phase" not in report_help.output
    assert "--protocol-frozen" not in report_help.output

    for args in (
        ["data", "materialize"],
        ["dataset", "build", "pv_all"],
        ["research", "run"],
        ["report", "build"],
        ["status"],
        ["inspect", "pv_all"],
    ):
        result = runner.invoke(app, [*args, "--help"])
        assert result.exit_code == 0, result.output
        assert "--profile" in result.output

    holdout = runner.invoke(
        app,
        [
            "research",
            "run",
            "--holdout",
            "--start-date",
            "2019-01-01",
            "--end-date",
            "2019-12-31",
        ],
    )
    assert holdout.exit_code != 0
    assert "holdout_frozen_development_identity_unavailable" in holdout.output
    assert '"status": "READY"' not in holdout.output


def _assert_public_text_is_sanitized(text: str) -> None:
    lowered = text.lower()
    assert all(marker.lower() not in lowered for marker in FORBIDDEN_PUBLIC_MARKERS)


def test_cli_materialize_leaves_raw_root_resolution_to_project_layout(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, object] = {}

    def materialize(**kwargs: object) -> dict[str, object]:
        captured.update(kwargs)
        return {"status": "BLOCKED"}

    monkeypatch.setattr(cli_commands.research, "data_materialize", materialize)
    result = CliRunner().invoke(
        app,
        [
            "data",
            "materialize",
            "--project-root",
            str(tmp_path),
            "--plan",
        ],
    )

    assert result.exit_code == 0, result.output
    assert captured["raw_root"] is None
    assert captured["layout"].project_root == tmp_path
