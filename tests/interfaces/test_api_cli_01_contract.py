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


@pytest.mark.parametrize("boundary", ["soft", "captured", "explicit"])
def test_curve_output_growth_is_admitted_before_retention(
    tmp_path, monkeypatch, boundary
):
    from types import SimpleNamespace

    from quant_research_platform.contracts.targets import ProducerKind, TargetStatus
    from quant_research_platform.interfaces import queries
    from quant_research_platform.runtime.resources import runtime_resource_policy_scope
    from quant_research_platform.storage.layout import StorageLayout
    from tests.helpers.runtime_resources import capture_test_policy

    policy = capture_test_policy(
        available_memory_bytes=(256 if boundary == "captured" else 400) * 1024,
        explicit_cap_bytes=256 * 1024 if boundary == "explicit" else None,
    )
    target = SimpleNamespace(
        target_id="ledger",
        producer_kind=ProducerKind.PORTFOLIO,
        status=TargetStatus.BUILT,
        instance_identity="b" * 64,
    )
    record = SimpleNamespace(target_results=(target,), payload=lambda: {})
    pulled, closed, observed_bytes = [], [], []

    def chunks():
        try:
            for sequence in range(100):
                pulled.append(sequence)
                observed_bytes.append(policy.used_memory_bytes)
                yield SimpleNamespace(
                    sequence=sequence,
                    valuations=tuple(
                        SimpleNamespace(
                            phase=phase,
                            timestamp_ns=sequence * 10 + offset,
                            value=100.0,
                        )
                        for offset, phase in enumerate(("before", "after", "end"))
                    ),
                    diagnostics=(),
                )
        finally:
            closed.append(True)

    artifact = SimpleNamespace(
        iter_chunks=chunks,
        capital=SimpleNamespace(
            candidate_id="synthetic",
            valuation_currency="USD",
            initial_capital=100.0,
        ),
    )
    monkeypatch.setattr(
        queries.report_artifact,
        "read_research_run_record",
        lambda *args, **kwargs: record,
    )
    monkeypatch.setattr(
        queries.report_artifact,
        "read_validated_reference",
        lambda *args, **kwargs: (tmp_path, {}),
    )
    monkeypatch.setattr(
        queries, "read_portfolio_ledger_artifact", lambda *args, **kwargs: artifact
    )
    with runtime_resource_policy_scope(policy):
        kwargs = dict(
            profile=SimpleNamespace(profile_id="synthetic", market_id="a_share"),
            layout=StorageLayout(tmp_path),
            run_id="run_" + "a" * 32,
            target_id="ledger",
        )
        if boundary == "soft":
            result = queries.portfolio_curve(**kwargs)
            assert (
                result["source_point_count"] == result["displayed_point_count"] == 300
            )
            assert max(observed_bytes) > policy.memory_budget_bytes
        else:
            with pytest.raises(
                ValueError, match="portfolio_curve_exceeds_task_memory_budget"
            ):
                queries.portfolio_curve(**kwargs)
            assert 0 < len(pulled) < 100
        assert closed == [True]
        assert policy.used_memory_bytes == 0


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
        ("GET", "/api/0.1/status"),
        ("GET", "/api/0.1/inspect/{target_id}"),
        ("GET", "/api/0.1/runs/{run_id}/reports/{report_identity}"),
        ("GET", "/api/0.1/runs/{run_id}/portfolios/{target_id}"),
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
            await client.get("/api/0.1/catalog/factors"),
            await client.get("/api/0.1/catalog/labels"),
            await client.get("/api/0.1/research-datasets/pv_all/requirements"),
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
            "market_model_factors",
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
        "/api/0.1/status",
        "/api/0.1/inspect/{target_id}",
        "/api/0.1/runs/{run_id}/reports/{report_identity}",
        "/api/0.1/runs/{run_id}/portfolios/{target_id}",
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
        assert result.exit_code == 0, result.output
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

    for name in (
        "plan_targets",
        "preflight_model_specs",
        "research_run",
        "report_build",
        "_default_raw_root",
    ):
        monkeypatch.setattr(research, name, forbidden)
    root = tmp_path / "bound"
    outside = tmp_path / "untrusted"
    outside.mkdir()
    (outside / "secret.txt").write_text("must_not_be_returned", encoding="utf-8")
    async with AsyncClient(
        transport=ASGITransport(app=create_app(project_root=root)),
        base_url="http://testserver",
    ) as client:
        response = await client.get(
            "/api/0.1/status", params={"project_root": str(outside)}
        )
        assert response.status_code == 200
        assert response.json() == {"profile_id": "a_share_csmar_choice", "runs": []}
        assert not root.exists()
        async with AsyncClient(
            transport=ASGITransport(app=create_app()), base_url="http://testserver"
        ) as unbound:
            assert (await unbound.get("/api/0.1/status")).status_code == 503
        for params in (
            {"run_id": "../secret"},
            {"run_id": "missing"},
            {"profile": "../secret"},
        ):
            failed = await client.get("/api/0.1/status", params=params)
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
            "/api/0.1/catalog/factors", params={"profile": CRYPTO_PROFILE_ID}
        )
        assert response.status_code == 200
        assert len(response.json()["items"]) == len(profile.factors.list())
        dataset_id = profile.research_datasets.list()[0].data_id
        requirements = await client.get(
            f"/api/0.1/research-datasets/{dataset_id}/requirements",
            params={"profile": CRYPTO_PROFILE_ID},
        )
        assert requirements.status_code == 200
        assert requirements.json()["market_id"] == "crypto"


@pytest.mark.anyio
@pytest.mark.parametrize("anyio_backend", ["asyncio"])
async def test_overlapping_published_queries_reject_busy_and_recover(
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
    from quant_research_platform.runtime import resources
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
    roots, worker_threads = [], []
    original_read = artifact.read_research_run_record

    def paused_read(*args, **kwargs):
        root = resources.current_runtime_resource_policy()
        assert root is not None
        roots.append(root)
        worker_threads.append(threading.get_ident())
        with root.lease("test.query_read", 1):
            if len(roots) == 1:
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
        url = "/api/0.1/runs/run_one/portfolios/portfolio"
        first_request = asyncio.create_task(client.get(url))
        try:
            assert await asyncio.to_thread(entered.wait, 5)
            rejected = await client.get(url)
            assert rejected.status_code == 503
            assert (
                rejected.json()["error"]["error_code"] == "runtime_data_scale_lock_busy"
            )
            assert len(roots) == 1 and roots[0].used_memory_bytes == 1
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
        assert len(roots) == 2 and roots[0] is not roots[1]
        assert all(
            root.state == "CLOSED" and root.used_memory_bytes == 0 for root in roots
        )
        assert not resources._PROCESS_POISONED

        def fail_read(*_args, **_kwargs):
            raise RuntimeError("private_runtime_failure")

        monkeypatch.setattr(artifact, "read_research_run_record", fail_read)
        failed = await client.get(url)
        assert failed.status_code == 500
        assert failed.json()["error"]["error_code"] == "internal_error"
        assert "private_runtime_failure" not in failed.text


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


def test_research_cli_uses_only_the_configuration_for_scientific_selection(
    tmp_path, monkeypatch
):
    from quant_research_platform.workflows import research
    from tests.helpers.research_configuration import research_configuration

    config = research_configuration()
    config["windows"]["development"] = {"start": "2022-01-01", "end": "2022-12-31"}
    path = tmp_path / "research.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    calls = []
    monkeypatch.setattr(
        research,
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
