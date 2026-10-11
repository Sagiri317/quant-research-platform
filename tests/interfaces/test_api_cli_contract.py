from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import threading
from pathlib import Path

import pytest
from httpx import ASGITransport, AsyncClient
from typer.main import get_command
from typer.testing import CliRunner

from quant_research_platform.interfaces.api.app import create_app
from quant_research_platform.interfaces.cli import commands as cli_commands
from quant_research_platform.interfaces.cli.commands import app
from quant_research_platform.interfaces.queries import list_factors
from quant_research_platform.policy.share_safety import unsafe_for_share_reason
from quant_research_platform.profiles.research import research_profile
from quant_research_platform.workflows import operations


def test_api_routes_are_current_and_exact() -> None:
    api = create_app()
    routes = {
        (next(iter(route.methods - {"HEAD"})), route.path)
        for route in api.routes
        if route.path.startswith("/api/")
    }
    assert routes == {
        ("GET", "/api/catalog/factors"),
        ("GET", "/api/catalog/labels"),
        ("GET", "/api/research-datasets/{dataset_id}/requirements"),
        ("GET", "/api/status"),
        ("GET", "/api/inspect/{target_id}"),
        ("GET", "/api/runs/{run_id}/reports/{report_identity}"),
        ("GET", "/api/runs/{run_id}/portfolios/{target_id}"),
    }


@pytest.mark.parametrize("module_entry", (False, True))
def test_real_cli_entrypoints_emit_safe_failure_without_traceback(
    tmp_path, module_entry
):
    command = (
        [sys.executable, "-m", "quant_research_platform.interfaces.cli"]
        if module_entry
        else [
            sys.executable,
            "-c",
            "from quant_research_platform.interfaces.cli.commands import main; main()",
        ]
    )
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(Path(__file__).resolve().parents[2] / "src")
    result = subprocess.run(
        [
            *command,
            "report",
            "build",
            "--run-id",
            "../synthetic_private_marker",
            "--project-root",
            str(tmp_path),
        ],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=30,
    )
    output = result.stdout + result.stderr
    assert result.returncode != 0
    assert "Traceback" not in output
    assert str(Path(__file__).resolve().parents[2]) not in output
    assert "synthetic_private_marker" not in output
    assert "invalid_run_id" in output


def test_cli_failure_redacts_quoted_credential_text(monkeypatch):
    def fail(*args, **kwargs):
        raise ValueError('{"Authorization": "Bearer SYNTHETIC_SECRET"}')

    monkeypatch.setattr(
        cli_commands.queries, "describe_research_dataset_requirements", fail
    )
    result = CliRunner().invoke(app, ["research-datasets", "requirements", "pv_all"])
    assert result.exit_code == 2
    assert "[credential]" in result.output
    assert "SYNTHETIC_SECRET" not in result.output


def test_cli_main_preserves_keyboard_interrupt(monkeypatch):
    primary = KeyboardInterrupt("cancelled")

    def interrupt():
        raise primary

    monkeypatch.setattr(cli_commands, "app", interrupt)
    with pytest.raises(KeyboardInterrupt) as caught:
        cli_commands.main()
    assert caught.value is primary


@pytest.mark.anyio
@pytest.mark.parametrize("anyio_backend", ["asyncio"])
async def test_api_catalog_and_requirements_use_current_a_share_protocol(
    anyio_backend,
) -> None:
    async with AsyncClient(
        transport=ASGITransport(app=create_app()), base_url="http://testserver"
    ) as client:
        responses = (
            await client.get("/api/catalog/factors"),
            await client.get("/api/catalog/labels"),
            await client.get("/api/research-datasets/pv_all/requirements"),
        )
        assert [response.status_code for response in responses] == [200, 200, 200]
        factor_items = responses[0].json()["items"]
        assert all("api_version" not in response.json() for response in responses)
        assert all("definition_version" not in item for item in factor_items)
        assert all(
            "definition_version" not in item for item in responses[1].json()["items"]
        )
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
            "market_model_factors",
        ]
        _assert_public_text_is_sanitized(
            "\n".join(str(response.json()) for response in responses)
        )


def test_openapi_has_closed_current_response_models() -> None:
    schema = create_app().openapi()
    assert schema["info"]["version"] == "unversioned"
    assert set(schema["paths"]) == {
        "/api/catalog/factors",
        "/api/catalog/labels",
        "/api/research-datasets/{dataset_id}/requirements",
        "/api/status",
        "/api/inspect/{target_id}",
        "/api/runs/{run_id}/reports/{report_identity}",
        "/api/runs/{run_id}/portfolios/{target_id}",
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


def test_cli_exposes_only_current_research_surface(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
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
    for catalog, header in (
        ("factors", "factor_id,market_id,factor_set"),
        ("labels", "label_id,market_id,horizon,horizon_unit"),
    ):
        result = runner.invoke(app, ["catalog", catalog])
        assert result.exit_code == 0
        assert result.output.splitlines()[0] == header
    from tests.helpers.research_configuration import research_configuration

    configuration_path = tmp_path / "research.json"
    configuration_path.write_text(
        json.dumps(research_configuration()), encoding="utf-8"
    )
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
            "--research-plan",
            str(configuration_path),
            "--plan",
        ),
        ("report", "build", "--help"),
    ):
        result = runner.invoke(app, list(args))
        expected = 3 if "--plan" in args else 0
        assert result.exit_code == expected, result.output
        _assert_public_text_is_sanitized(result.output)

    missing_run_id = runner.invoke(app, ["report", "build", "--plan"])
    assert missing_run_id.exit_code != 0

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
        if args == ["research", "run"]:
            assert "--research-plan" in result.output
            assert "--profile" not in result.output
            assert "--start-date" not in result.output
        else:
            assert "--profile" in result.output

    holdout = runner.invoke(
        app,
        [
            "research",
            "run",
            "--holdout",
            "--research-plan",
            str(configuration_path),
        ],
    )
    assert holdout.exit_code != 0
    assert "holdout_frozen_development_identity_unavailable" in holdout.output
    assert '"status": "READY"' not in holdout.output


@pytest.mark.anyio
@pytest.mark.parametrize("anyio_backend", ["asyncio"])
async def test_read_only_status_never_plans_and_client_cannot_bind_a_project(
    anyio_backend, tmp_path, monkeypatch
):
    from quant_research_platform.workflows import research

    def forbidden(*_args, **_kwargs):
        raise AssertionError("read_only_query_started_work")

    for owner, names in (
        (operations, ("plan_targets", "research_run", "report_build")),
        (research, ("preflight_model_specs", "_default_raw_root")),
    ):
        for name in names:
            monkeypatch.setattr(owner, name, forbidden)
    root = tmp_path / "bound"
    outside = tmp_path / "untrusted"
    outside.mkdir()
    (outside / "secret.txt").write_text("must_not_be_returned", encoding="utf-8")
    async with AsyncClient(
        transport=ASGITransport(app=create_app(project_root=root)),
        base_url="http://testserver",
    ) as client:
        response = await client.get(
            "/api/status", params={"project_root": str(outside)}
        )
        assert response.status_code == 200
        assert response.json() == {"profile_id": "a_share_csmar_choice", "runs": []}
        assert not root.exists()
        async with AsyncClient(
            transport=ASGITransport(app=create_app()), base_url="http://testserver"
        ) as unbound:
            assert (await unbound.get("/api/status")).status_code == 503
        for params in (
            {"run_id": "../secret"},
            {"run_id": "missing"},
            {"profile": "../secret"},
        ):
            failed = await client.get("/api/status", params=params)
            assert failed.status_code in {404, 422}
            assert "must_not_be_returned" not in failed.text
            assert str(tmp_path) not in failed.text
        assert not root.exists()


@pytest.mark.anyio
@pytest.mark.parametrize("anyio_backend", ["asyncio"])
async def test_catalog_queries_support_both_existing_profiles(
    anyio_backend,
):
    from quant_research_platform.profiles.research import CRYPTO_PROFILE_ID

    async with AsyncClient(
        transport=ASGITransport(app=create_app()), base_url="http://testserver"
    ) as client:
        profile = research_profile(CRYPTO_PROFILE_ID)
        response = await client.get(
            "/api/catalog/factors", params={"profile": CRYPTO_PROFILE_ID}
        )
        assert response.status_code == 200
        assert len(response.json()["items"]) == len(profile.factors.list())
        dataset_id = profile.research_datasets.list()[0].data_id
        requirements = await client.get(
            f"/api/research-datasets/{dataset_id}/requirements",
            params={"profile": CRYPTO_PROFILE_ID},
        )
        assert requirements.status_code == 200
        assert requirements.json()["market_id"] == "crypto"


@pytest.mark.anyio
@pytest.mark.parametrize("anyio_backend", ["asyncio"])
async def test_overlapping_published_queries_complete_independently(
    anyio_backend, tmp_path, monkeypatch
):
    from quant_research_platform.artifacts import research as artifact
    from quant_research_platform.contracts.json_codec import canonical_json_hash
    from quant_research_platform.contracts.targets import (
        ProducerKind,
        ResearchRunRecord,
        ResearchRunTargetResult,
        RunStatus,
        TargetStatus,
        unmeasured_operation_evidence,
    )
    from quant_research_platform.storage.layout import StorageLayout

    layout = StorageLayout(project_root=tmp_path)
    protocol = {"scenario_id": "pv_all", "access_window": "development"}
    record = ResearchRunRecord(
        run_id="run_one",
        profile_id=research_profile().profile_id,
        scenario_id="pv_all",
        protocol=protocol,
        protocol_identity=canonical_json_hash(protocol),
        research_window={
            "start": "2019-01-01",
            "end": "2019-12-31",
            "access": "development",
        },
        software_identity="c" * 64,
        terminal_targets=("portfolio",),
        target_results=(
            ResearchRunTargetResult(
                target_id="portfolio",
                producer_kind=ProducerKind.PORTFOLIO,
                status=TargetStatus.BLOCKED,
                reason="missing_prediction",
            ),
        ),
        status=RunStatus.BLOCKED,
        operation_evidence=unmeasured_operation_evidence("fixture_record"),
    )
    artifact.write_research_run_record(layout.research_run_path(record.run_id), record)
    entered, release = threading.Event(), threading.Event()
    worker_threads = []
    original_read = artifact.read_research_run_record

    def paused_read(*args, **kwargs):
        worker_threads.append(threading.get_ident())
        if len(worker_threads) == 1:
            entered.set()
            assert release.wait(timeout=5)
        return original_read(*args, **kwargs)

    monkeypatch.setattr(artifact, "read_research_run_record", paused_read)
    async with AsyncClient(
        transport=ASGITransport(
            app=create_app(project_root=tmp_path), raise_app_exceptions=False
        ),
        base_url="http://testserver",
    ) as client:
        url = "/api/runs/run_one/portfolios/portfolio"
        first_request = asyncio.create_task(client.get(url))
        try:
            assert await asyncio.to_thread(entered.wait, 5)
            concurrent = await asyncio.wait_for(client.get(url), timeout=5)
            assert concurrent.status_code == 200
        finally:
            release.set()
            first = await asyncio.wait_for(first_request, timeout=5)
        assert first_request.done() and first.status_code == 200
        assert all(value != threading.get_ident() for value in worker_threads)
        subsequent = await client.get(url)
        assert subsequent.status_code == 200
        assert (
            subsequent.json()
            == first.json()
            == {
                "target_id": "portfolio",
                "status": "BLOCKED",
                "reason": "missing_prediction",
            }
        )
        assert len(worker_threads) == 3
        assert concurrent.json() == first.json()

        def fail_read(*_args, **_kwargs):
            raise RuntimeError("private_runtime_failure")

        monkeypatch.setattr(artifact, "read_research_run_record", fail_read)
        failed = await client.get(url)
        assert failed.status_code == 500
        assert failed.json()["error"]["error_code"] == "internal_error"
        assert "private_runtime_failure" not in failed.text


def _assert_public_text_is_sanitized(text: str) -> None:
    assert unsafe_for_share_reason(text) is None


@pytest.mark.parametrize(
    "status, exit_code", [("COMPLETE", 0), ("FAILED", 1), ("BLOCKED", 3)]
)
def test_cli_materialize_leaves_raw_root_resolution_to_project_layout(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    status,
    exit_code,
) -> None:
    captured: dict[str, object] = {}

    def materialize(**kwargs: object) -> dict[str, object]:
        captured.update(kwargs)
        return {"status": status}

    monkeypatch.setattr(cli_commands.operations, "data_materialize", materialize)
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

    assert result.exit_code == exit_code, result.output
    assert json.loads(result.stdout)["status"] == status
    assert captured["raw_root"] is None
    assert captured["layout"].project_root == tmp_path


def test_research_cli_uses_only_the_configuration_for_scientific_selection(
    tmp_path, monkeypatch
):
    from tests.helpers.research_configuration import research_configuration

    config = research_configuration()
    config["windows"]["development"] = {"start": "2022-01-01", "end": "2022-12-31"}
    path = tmp_path / "research.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    calls = []
    monkeypatch.setattr(
        operations,
        "research_run",
        lambda **kwargs: calls.append(kwargs) or {"status": "COMPLETE"},
    )
    result = CliRunner().invoke(
        app,
        [
            "research",
            "run",
            "--research-plan",
            str(path),
            "--project-root",
            str(tmp_path),
        ],
    )
    assert result.exit_code == 0, result.output
    assert len(calls) == 1
    assert calls[0]["research_plan"] == config
    assert not {"scenario_id", "start_date", "end_date"}.intersection(calls[0])
    assert calls[0]["profile"].profile_id == config["profile_id"]


@pytest.mark.parametrize("profile_id", [None, "unknown"])
def test_research_cli_rejects_invalid_configuration_profile_without_traceback(
    tmp_path, profile_id
):
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps({"profile_id": profile_id}), encoding="utf-8")
    result = CliRunner().invoke(app, ["research", "run", "--research-plan", str(path)])
    assert result.exit_code == 2
    assert "research_configuration_profile_invalid" in result.output
    assert "Traceback" not in result.output


@pytest.mark.parametrize("is_crypto", [False, True])
def test_notebook_metrics_preserve_each_published_scope(is_crypto):
    import pandas as pd

    notebook = json.loads(
        (
            Path(__file__).resolve().parents[2]
            / "notebooks/research-interview-demo.ipynb"
        ).read_text(encoding="utf-8")
    )
    cell = next(item for item in notebook["cells"] if item["id"] == "research-cell-07")
    row = {
        "candidate_id": "synthetic",
        "metric_id": "oos_mean_daily_return",
        "value": 0.01,
        "status": "DEFINED",
        "reason": "",
        "observation_count": 2,
    }
    scopes = [
        ("continuous", "continuous"),
        ("independent_segments", "a" * 64),
        ("independent_segments", "b" * 64),
    ]
    rows = (
        [
            {**row, "execution_mode": mode, "episode_id": episode}
            for mode, episode in scopes
        ]
        if is_crypto
        else [row]
    )
    sections = {"readiness/blockers": {"issues": []}}
    sections.update(
        {"daily statistics": {"results": rows, "policy": {"annualization_days": 365}}}
        if is_crypto
        else {"cost": {"results": rows}, "risk": {"results": []}}
    )
    displayed = []
    namespace = {
        "report": {"sections": sections},
        "is_crypto": is_crypto,
        "pd": pd,
        "display": displayed.append,
        "Markdown": str,
    }
    exec("".join(cell["source"]), namespace)
    visible = namespace["metrics"].loc[:, namespace["columns"]]
    pd.testing.assert_frame_equal(visible, pd.DataFrame(rows).loc[:, visible.columns])
    assert len(visible.drop_duplicates()) == len(rows)
    if is_crypto:
        assert (
            list(zip(visible.execution_mode, visible.episode_id, strict=True)) == scopes
        )
        html = next(item for item in displayed if isinstance(item, str))
        assert all(episode in html for _, episode in scopes)


def _write_metric_run(project_root, value):
    from quant_research_platform.artifacts.research import write_research_run_record
    from quant_research_platform.contracts.json_codec import canonical_json_hash
    from quant_research_platform.contracts.targets import (
        ProducerKind,
        ResearchRunRecord,
        ResearchRunTargetResult,
        RunStatus,
        TargetStatus,
    )
    from quant_research_platform.runtime.execution_metrics import (
        record_run_observations,
    )
    from quant_research_platform.storage.layout import StorageLayout

    observations = {}
    with record_run_observations(observations):
        pass
    observations["status"] = "DEFINED"
    observations["reason"] = ""
    observations["metrics"] = {
        key: {"status": "DEFINED", "value": value} for key in observations["metrics"]
    }
    protocol = {"scenario_id": "pv_all", "access_window": "development"}
    record = ResearchRunRecord(
        run_id="operation_metric",
        profile_id=research_profile().profile_id,
        scenario_id="pv_all",
        protocol=protocol,
        protocol_identity=canonical_json_hash(protocol),
        research_window={
            "start": "2019-01-01",
            "end": "2019-12-31",
            "access": "development",
        },
        software_identity="c" * 64,
        terminal_targets=("portfolio",),
        target_results=(
            ResearchRunTargetResult(
                target_id="portfolio",
                producer_kind=ProducerKind.PORTFOLIO,
                status=TargetStatus.BLOCKED,
                reason="synthetic_missing",
            ),
        ),
        status=RunStatus.BLOCKED,
        operation_evidence=observations,
    )
    path = write_research_run_record(
        StorageLayout(project_root).research_run_path(record.run_id), record
    )
    return record, path


@pytest.mark.anyio
@pytest.mark.parametrize("anyio_backend", ["asyncio"])
@pytest.mark.parametrize(
    "value",
    [10**400 + 1, 10**4096, 2**53 + 1, float.fromhex("0x1.fffffffffffffp+1023")],
    ids=[
        "large_integer",
        "past_removed_integer_limit",
        "exact_integer",
        "maximum_float",
    ],
)
async def test_status_api_preserves_native_operation_metric_precision(
    anyio_backend, tmp_path, value
):
    from quant_research_platform.artifacts.research import read_research_run_record

    record, path = _write_metric_run(tmp_path, value)
    before = path.read_bytes()
    assert read_research_run_record(path).payload() == record.payload()
    async with AsyncClient(
        transport=ASGITransport(
            app=create_app(project_root=tmp_path), raise_app_exceptions=False
        ),
        base_url="http://testserver",
    ) as client:
        response = await client.get("/api/status", params={"run_id": record.run_id})
    assert response.status_code == 200
    assert response.json() == record.payload()
    for metric in response.json()["operation_evidence"]["metrics"].values():
        assert type(metric["value"]) is type(value)
        assert metric["value"] == value
    assert path.read_bytes() == before


@pytest.mark.anyio
@pytest.mark.parametrize("anyio_backend", ["asyncio"])
@pytest.mark.parametrize(
    "bad,error",
    [
        (True, "research_run_record_invalid"),
        (False, "research_run_record_invalid"),
        (-1, "research_run_record_invalid"),
        (-0.5, "research_run_record_invalid"),
        (-(10**400), "research_run_record_invalid"),
        (float("nan"), "qrp_json_nonfinite_number"),
        (float("inf"), "qrp_json_nonfinite_number"),
        (-float("inf"), "qrp_json_nonfinite_number"),
    ],
    ids=[
        "true",
        "false",
        "negative_integer",
        "negative_float",
        "negative_large_integer",
        "nan",
        "positive_infinity",
        "negative_infinity",
    ],
)
async def test_invalid_persisted_operation_metrics_keep_api_failure_projection(
    anyio_backend, tmp_path, bad, error
):
    from quant_research_platform.artifacts.research import read_research_run_record
    from quant_research_platform.storage.layout import StorageLayout

    record, good = _write_metric_run(tmp_path, 1)
    before = good.read_bytes()
    payload = record.payload()
    payload["run_id"] = "operation_bad"
    payload["operation_evidence"]["metrics"]["read_bytes"]["value"] = bad
    candidate = StorageLayout(tmp_path).research_run_path(payload["run_id"])
    raw = (json.dumps(payload, ensure_ascii=False) + "\n").encode("utf-8")
    candidate.write_bytes(raw)
    with pytest.raises(ValueError, match=f"^{error}$"):
        read_research_run_record(candidate)
    async with AsyncClient(
        transport=ASGITransport(
            app=create_app(project_root=tmp_path), raise_app_exceptions=False
        ),
        base_url="http://testserver",
    ) as client:
        response = await client.get("/api/status", params={"run_id": payload["run_id"]})
        valid = await client.get("/api/status", params={"run_id": record.run_id})
    assert response.status_code == 422
    assert response.json()["error"]["error_code"] == "published_evidence_invalid"
    assert valid.status_code == 200 and valid.json() == record.payload()
    assert good.read_bytes() == before
    assert candidate.read_bytes() == raw


def test_api_get_from_former_browser_origin_has_no_cors_grant():
    async def request():
        async with AsyncClient(
            transport=ASGITransport(app=create_app()), base_url="http://test"
        ) as client:
            plain = await client.get("/api/catalog/labels")
            origin = await client.get(
                "/api/catalog/labels", headers={"Origin": "http://localhost:5173"}
            )
            assert origin.status_code == plain.status_code == 200
            assert origin.json() == plain.json()
            assert "access-control-allow-origin" not in origin.headers

    asyncio.run(request())


def test_cli_failure_redacts_embedded_unc_path(monkeypatch):
    def fail(*args, **kwargs):
        raise ValueError("read failed at //nas/private/report.csv")

    monkeypatch.setattr(
        cli_commands.queries, "describe_research_dataset_requirements", fail
    )
    result = CliRunner().invoke(app, ["research-datasets", "requirements", "pv_all"])
    assert result.exit_code == 2
    assert "//nas" not in result.output and "private/report.csv" not in result.output
