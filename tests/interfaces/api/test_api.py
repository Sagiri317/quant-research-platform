from __future__ import annotations

import json
import zipfile
from io import BytesIO

import pandas as pd
import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from quant_research_platform.artifacts import load_evaluation_set
from quant_research_platform.interfaces.api.app import create_app
from quant_research_platform.interfaces.api.models import RunRoot
from quant_research_platform.interfaces.api.services import download_service
from quant_research_platform.interfaces.api.services.download_service import (
    selected_download_response,
)
from quant_research_platform.interfaces.api.services.run_repository import (
    DEFAULT_RUN_ROOT_ALIAS as REPOSITORY_DEFAULT_RUN_ROOT_ALIAS,
)
from quant_research_platform.interfaces.api.services.run_repository import (
    discover_runs,
    normalize_run_roots,
)
from tests.interfaces.api.artifact_run_fixtures import write_evaluation_set


def test_download_service_does_not_keep_legacy_artifact_helper_entries():
    legacy_helpers = {
        "artifact_relative_path",
        "artifact_relative_path_from_manifest",
        "display_download_items",
        "download_item",
        "download_item_path",
        "ensure_download_item_exposed",
        "json_bytes",
        "table_relative_path",
        "validate_cache_path_segment",
        "validate_download_item_name",
    }

    assert legacy_helpers.isdisjoint(set(dir(download_service)))


def test_api_lists_runs_from_multiple_aliased_roots(tmp_path):
    platform_root = tmp_path / "evaluation_runs"
    crypto_root = tmp_path / "crypto_runs"
    write_evaluation_set(
        platform_root / "demo",
        run_id="demo",
        domain="evaluation_synthetic",
    )
    write_evaluation_set(
        crypto_root / "toy",
        run_id="toy",
        project="crypto-perp-alpha-lab",
        domain="crypto_perp",
        net_returns=[0.02, 0.03],
        turnover=[1.0, 2.0],
        cost=[0.0, 0.0],
    )
    client = TestClient(
        create_app(run_roots=[("platform", platform_root), ("crypto", crypto_root)])
    )

    response = client.get("/api/runs")

    assert response.status_code == 200
    runs = response.json()
    assert [run["public_run_id"] for run in runs] == ["crypto/toy", "platform/demo"]
    assert runs[0]["source_alias"] == "crypto"
    assert "evaluation_id" not in runs[0]
    assert runs[0]["run_id"] == "toy"
    assert runs[0]["project"] == "crypto-perp-alpha-lab"
    assert runs[0]["domain"] == "crypto_perp"
    assert runs[0]["mean_final_return"] == 0.05


def test_api_and_repository_share_default_run_root_alias():
    import quant_research_platform.interfaces.api.app as app_module

    assert app_module.DEFAULT_RUN_ROOT_ALIAS == REPOSITORY_DEFAULT_RUN_ROOT_ALIAS


def test_health_response_does_not_expose_local_run_root_paths(tmp_path):
    runs_root = tmp_path / "runs"
    client = TestClient(create_app(run_roots=[("platform", runs_root)]))

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "run_roots": [{"source_alias": "platform"}, {"source_alias": "pipeline"}],
    }
    assert str(runs_root) not in response.text


def test_run_root_configuration_rejects_duplicate_source_aliases(tmp_path):
    with pytest.raises(ValueError, match="duplicate run root source aliases"):
        create_app(
            run_roots=[("platform", tmp_path / "a"), ("platform", tmp_path / "b")]
        )
    with pytest.raises(ValueError, match="duplicate run root source aliases"):
        normalize_run_roots(
            [("platform", tmp_path / "a"), ("platform", tmp_path / "b")]
        )


@pytest.mark.parametrize(
    "alias",
    ["", ".", "..", "bad.alias", "bad alias", "bad/alias"],
)
def test_run_root_configuration_rejects_unsafe_source_aliases(tmp_path, alias):
    with pytest.raises(ValueError, match="run root aliases"):
        create_app(run_roots=[(alias, tmp_path / "runs")])
    with pytest.raises(ValueError, match="run root aliases"):
        normalize_run_roots([(alias, tmp_path / "runs")])


def test_discover_runs_uses_schema_evaluation_id(tmp_path):
    runs_root = tmp_path / "runs"
    write_evaluation_set(runs_root / "folder_name", run_id="schema_run")

    records = discover_runs([RunRoot("platform", runs_root)])

    assert [record.ref.public_run_id for record in records] == ["platform/schema_run"]


def test_discover_runs_exposes_pipeline_evaluation_run(tmp_path):
    pipeline_root = tmp_path / "pipeline_runs"
    nested_run = pipeline_root / "pipeline_run" / "evaluation"
    write_evaluation_set(nested_run, run_id="demo")

    records = discover_runs([RunRoot("pipeline", pipeline_root)])

    assert [record.ref.public_run_id for record in records] == ["pipeline/demo"]
    assert records[0].ref.path == nested_run


def test_default_api_run_list_exposes_default_pipeline_root(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    platform_root = tmp_path / "reports" / "runs"
    pipeline_run = (
        tmp_path
        / "reports"
        / "private_real"
        / "pipeline_runs"
        / "pipeline_run"
        / "evaluation"
    )
    write_evaluation_set(pipeline_run, run_id="demo")
    client = TestClient(create_app(run_roots=[("platform", platform_root)]))

    response = client.get("/api/runs")

    assert response.status_code == 200
    assert response.json()[0]["public_run_id"] == "pipeline/demo"


def test_create_app_appends_default_pipeline_root_when_run_roots_env_is_unset(
    tmp_path,
    monkeypatch,
):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("QRP_RUN_ROOTS", raising=False)
    monkeypatch.delenv("QRP_PIPELINE_OUTPUT_ROOT", raising=False)

    app = create_app()

    context = app.state.qrp_context
    configured_roots = [
        (root.source_alias, root.path.as_posix()) for root in context.run_roots
    ]
    assert configured_roots == [
        ("platform", "reports/runs"),
        ("pipeline", "reports/private_real/pipeline_runs"),
    ]


def test_create_app_does_not_duplicate_explicit_pipeline_root(tmp_path):
    explicit_pipeline_root = tmp_path / "custom_pipeline_runs"

    app = create_app(
        run_roots=[
            ("platform", tmp_path / "runs"),
            ("pipeline", explicit_pipeline_root),
        ],
        pipeline_output_root=tmp_path / "ignored_pipeline_runs",
    )

    context = app.state.qrp_context
    pipeline_roots = [
        root.path for root in context.run_roots if root.source_alias == "pipeline"
    ]
    assert pipeline_roots == [explicit_pipeline_root]


def test_api_run_list_skips_invalid_run_index_without_exposing_paths(tmp_path):
    runs_root = tmp_path / "runs"
    write_evaluation_set(runs_root / "good", run_id="good")
    broken = runs_root / "broken"
    broken.mkdir(parents=True)
    (broken / "manifest.json").write_text(
        json.dumps({"schema_version": "0.5"}),
        encoding="utf-8",
    )
    client = TestClient(create_app(run_roots=[("platform", runs_root)]))

    response = client.get("/api/runs")

    assert response.status_code == 200
    assert [run["public_run_id"] for run in response.json()] == ["platform/good"]
    assert str(runs_root) not in response.text
    assert "summary.json" not in response.text


def test_api_lists_runs_with_lightweight_index(tmp_path, monkeypatch):
    from quant_research_platform.interfaces.api.services import run_repository

    runs_root = tmp_path / "runs"
    write_evaluation_set(runs_root / "demo", run_id="demo")

    def fail_if_full_artifact_loader_is_used(_path):
        raise AssertionError("/api/runs should not load all artifact CSV files")

    monkeypatch.setattr(
        run_repository,
        "load_evaluation_set",
        fail_if_full_artifact_loader_is_used,
    )
    client = TestClient(create_app(run_roots=[("platform", runs_root)]))

    response = client.get("/api/runs")

    assert response.status_code == 200
    assert response.json()[0]["public_run_id"] == "platform/demo"


def test_api_run_list_skips_unsupported_legacy_schema_runs(tmp_path):
    runs_root = tmp_path / "runs"
    write_evaluation_set(runs_root / "demo", run_id="demo")
    write_evaluation_set(runs_root / "legacy", run_id="legacy")
    for filename in ("manifest.json", "summary.json"):
        path = runs_root / "legacy" / filename
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["artifact_schema_version"] = "0.1"
        path.write_text(json.dumps(payload), encoding="utf-8")
    client = TestClient(create_app(run_roots=[("platform", runs_root)]))

    response = client.get("/api/runs")

    assert response.status_code == 200
    assert [run["public_run_id"] for run in response.json()] == ["platform/demo"]


def test_api_run_list_skips_schema_07_run_missing_required_artifact_file(tmp_path):
    runs_root = tmp_path / "runs"
    write_evaluation_set(runs_root / "good", run_id="good")
    broken = runs_root / "broken"
    write_evaluation_set(broken, run_id="broken")
    (broken / "diagnostics" / "diagnostic_summary.csv").unlink()
    client = TestClient(create_app(run_roots=[("platform", runs_root)]))

    response = client.get("/api/runs")
    target = client.get("/api/runs/platform/broken/evidence")

    assert response.status_code == 200
    assert [run["public_run_id"] for run in response.json()] == ["platform/good"]
    assert target.status_code == 422
    assert target.json()["detail"] == "artifact run index is incomplete or invalid"


def test_api_run_list_skips_schema_07_run_missing_summary_metric_columns(tmp_path):
    runs_root = tmp_path / "runs"
    write_evaluation_set(runs_root / "good", run_id="good")
    broken = runs_root / "broken"
    write_evaluation_set(broken, run_id="broken")
    performance_path = broken / "metrics" / "evaluation_metrics_summary.csv"
    performance = pd.read_csv(performance_path).drop(columns=["metric_value"])
    performance.to_csv(performance_path, index=False)
    client = TestClient(create_app(run_roots=[("platform", runs_root)]))

    response = client.get("/api/runs")
    target = client.get("/api/runs/platform/broken/evidence")

    assert response.status_code == 200
    assert [run["public_run_id"] for run in response.json()] == ["platform/good"]
    assert target.status_code == 422
    assert "evaluation_metrics_summary is missing columns" in target.json()["detail"]


def test_api_run_list_skips_unknown_manifest_artifact_entries(tmp_path):
    runs_root = tmp_path / "runs"
    write_evaluation_set(runs_root / "good", run_id="good")
    broken = runs_root / "broken"
    write_evaluation_set(broken, run_id="broken")
    manifest_path = broken / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifacts"]["private_debug"] = "private/debug.csv"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    client = TestClient(create_app(run_roots=[("platform", runs_root)]))

    response = client.get("/api/runs")
    target = client.get("/api/runs/platform/broken/evidence")

    assert response.status_code == 200
    assert [run["public_run_id"] for run in response.json()] == ["platform/good"]
    assert target.status_code == 422
    assert "unknown artifact group" in target.json()["detail"]


def test_api_returns_summary_metric_table_and_report_for_one_run(tmp_path):
    runs_root = tmp_path / "runs"
    write_evaluation_set(
        runs_root / "demo",
        run_id="demo",
        domain="evaluation_synthetic",
        ic_values=[0.1, 0.2],
        rankic_values=[0.3, 0.4],
    )
    client = TestClient(create_app(run_roots=[("platform", runs_root)]))

    evidence = client.get("/api/runs/platform/demo/evidence")
    ic = client.get("/api/runs/platform/demo/tables/ic/preview")
    returns = client.get("/api/runs/platform/demo/tables/returns/preview")
    performance = client.get(
        "/api/runs/platform/demo/tables/evaluation_metrics_summary/preview"
    )
    retired_performance = client.get("/api/runs/platform/demo/metrics/ic")
    report = client.get("/api/runs/platform/demo/report.md")

    assert evidence.status_code == 200
    assert "evaluation_id" not in evidence.json()["summary"]
    assert evidence.json()["summary"]["run_id"] == "demo"
    assert evidence.json()["summary"]["source_alias"] == "platform"
    assert evidence.json()["summary"]["public_run_id"] == "platform/demo"
    assert evidence.json()["summary"]["domain"] == "evaluation_synthetic"
    assert ic.status_code == 200
    assert {"combination_id", "timestamp", "feature", "label", "ic"}.issubset(
        ic.json()["columns"]
    )
    assert len(ic.json()["rows"]) == 2
    assert returns.status_code == 200
    assert "net_return" in returns.json()["columns"]
    assert performance.status_code == 200
    assert "metric_name" in performance.json()["columns"]
    assert retired_performance.status_code == 404
    assert report.status_code == 200
    assert report.text == "# Report\n"


def test_api_metric_preview_filters_combination_before_limit(tmp_path):
    runs_root = tmp_path / "runs"
    run_dir = runs_root / "demo"
    write_evaluation_set(run_dir, run_id="demo")
    pd.DataFrame(
        [
            {
                "combination_id": "factor__forward_return__dataframe__long_short",
                "timestamp": "2024-01-01",
                "ic": 0.1,
                "rankic": 0.1,
                "n_obs": 4,
            },
            {
                "combination_id": "factor__forward_return__dataframe__long_short",
                "timestamp": "2024-01-02",
                "ic": 0.2,
                "rankic": 0.2,
                "n_obs": 4,
            },
            {
                "combination_id": "factor__forward_return__dataframe__long_short",
                "timestamp": "2024-01-03",
                "ic": 0.3,
                "rankic": 0.3,
                "n_obs": 4,
            },
        ]
    ).assign(
        feature="factor",
        label="forward_return",
        kernel="dataframe",
        portfolio_mode="long_short",
    ).to_csv(run_dir / "metrics" / "ic.csv", index=False)
    client = TestClient(create_app(run_roots=[("platform", runs_root)]))

    response = client.get(
        "/api/runs/platform/demo/tables/ic/preview?"
        "combination_id=factor__forward_return__dataframe__long_short&limit=1"
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["total_rows"] == 3
    assert payload["has_more"] is True
    assert payload["rows"] == [
        {
            "combination_id": "factor__forward_return__dataframe__long_short",
            "timestamp": "2024-01-01",
            "ic": 0.1,
            "rankic": 0.1,
            "n_obs": 4,
            "feature": "factor",
            "label": "forward_return",
            "kernel": "dataframe",
            "portfolio_mode": "long_short",
        }
    ]


def test_api_metric_endpoint_uses_lightweight_table_preview(tmp_path, monkeypatch):
    from quant_research_platform.interfaces.api.services import run_repository

    runs_root = tmp_path / "runs"
    write_evaluation_set(runs_root / "run_a", run_id="run_a")
    write_evaluation_set(runs_root / "run_b", run_id="run_b")
    real_loader = run_repository.load_evaluation_set
    loaded_run_dirs: list[str] = []

    def spy_loader(path, *, table_names=None):
        loaded_run_dirs.append(path.name)
        return real_loader(path, table_names=table_names)

    monkeypatch.setattr(run_repository, "load_evaluation_set", spy_loader)
    client = TestClient(create_app(run_roots=[("platform", runs_root)]))

    response = client.get("/api/runs/platform/run_b/tables/ic/preview")

    assert response.status_code == 200
    assert loaded_run_dirs == []
    assert response.json()["rows"][0]["combination_id"] == (
        "factor__forward_return__dataframe__long_short"
    )


def test_api_metric_endpoint_streams_csv_preview(tmp_path, monkeypatch):
    from quant_research_platform.interfaces.api.services import table_preview_service

    runs_root = tmp_path / "runs"
    write_evaluation_set(runs_root / "demo", run_id="demo")
    read_csv = table_preview_service.pd.read_csv
    chunked_paths: list[str] = []

    def spy_read_csv(path, *args, **kwargs):
        if kwargs.get("chunksize") is not None:
            chunked_paths.append(str(path))
        return read_csv(path, *args, **kwargs)

    monkeypatch.setattr(table_preview_service.pd, "read_csv", spy_read_csv)
    client = TestClient(create_app(run_roots=[("platform", runs_root)]))

    response = client.get("/api/runs/platform/demo/tables/ic/preview?limit=1")

    assert response.status_code == 200
    assert response.json()["limit"] == 1
    assert chunked_paths
    assert chunked_paths[-1].replace("\\", "/").endswith("metrics/ic.csv")


def test_api_metric_endpoint_ignores_unrelated_broken_run_index(tmp_path):
    runs_root = tmp_path / "runs"
    write_evaluation_set(runs_root / "good", run_id="good")
    broken = runs_root / "broken"
    broken.mkdir(parents=True)
    (broken / "manifest.json").write_text(
        json.dumps({"artifact_schema_version": "0.5"}),
        encoding="utf-8",
    )
    client = TestClient(create_app(run_roots=[("platform", runs_root)]))

    response = client.get("/api/runs/platform/good/tables/ic/preview")

    assert response.status_code == 200
    assert response.json()["rows"]


def test_api_discovers_exact_run_root(tmp_path):
    run_dir = tmp_path / "single_run"
    write_evaluation_set(run_dir, run_id="single_run")
    client = TestClient(create_app(run_roots=[("single", run_dir)]))

    response = client.get("/api/runs")

    assert response.status_code == 200
    assert [run["public_run_id"] for run in response.json()] == ["single/single_run"]


def test_api_target_broken_run_index_still_returns_422(tmp_path):
    runs_root = tmp_path / "runs"
    write_evaluation_set(runs_root / "good", run_id="good")
    broken = runs_root / "broken"
    broken.mkdir(parents=True)
    (broken / "manifest.json").write_text(
        json.dumps({"artifact_schema_version": "0.5"}),
        encoding="utf-8",
    )
    client = TestClient(create_app(run_roots=[("platform", runs_root)]))

    response = client.get("/api/runs/platform/broken/evidence")

    assert response.status_code == 422
    assert response.json()["detail"] == "artifact run index is incomplete or invalid"


def test_api_target_duplicate_run_id_within_source_returns_422(tmp_path):
    runs_root = tmp_path / "runs"
    write_evaluation_set(runs_root / "first", run_id="demo")
    write_evaluation_set(runs_root / "second", run_id="demo")
    client = TestClient(create_app(run_roots=[("platform", runs_root)]))

    response = client.get("/api/runs/platform/demo/tables/ic/preview")

    assert response.status_code == 422
    assert response.json()["detail"] == "duplicate run id for source platform: demo"


def test_api_metric_endpoint_returns_limited_preview_rows(tmp_path):
    runs_root = tmp_path / "runs"
    write_evaluation_set(
        runs_root / "demo",
        run_id="demo",
        ic_values=[0.1, 0.2, 0.3],
        rankic_values=[0.3, 0.4, 0.5],
    )
    client = TestClient(create_app(run_roots=[("platform", runs_root)]))

    response = client.get("/api/runs/platform/demo/tables/ic/preview?limit=2")

    assert response.status_code == 200
    payload = response.json()
    assert len(payload["rows"]) == 2
    assert payload["total_rows"] == 3
    assert payload["limit"] == 2
    assert payload["has_more"] is True
    assert payload["total_rows_exact"] is False


def test_api_metric_preview_rejects_oversized_limit(tmp_path):
    runs_root = tmp_path / "runs"
    write_evaluation_set(runs_root / "demo", run_id="demo")
    client = TestClient(create_app(run_roots=[("platform", runs_root)]))

    response = client.get("/api/runs/platform/demo/tables/returns/preview?limit=1001")

    assert response.status_code == 422
    assert response.json()["detail"] == "limit must be between 1 and 1000"


def test_api_metric_preview_allows_maximum_limit(tmp_path):
    runs_root = tmp_path / "runs"
    write_evaluation_set(runs_root / "demo", run_id="demo")
    client = TestClient(create_app(run_roots=[("platform", runs_root)]))

    response = client.get("/api/runs/platform/demo/tables/returns/preview?limit=1000")

    assert response.status_code == 200
    assert response.json()["limit"] == 1000


def test_api_csv_preview_stops_after_page_is_collected(tmp_path, monkeypatch):
    runs_root = tmp_path / "runs"
    run_dir = runs_root / "demo"
    write_evaluation_set(run_dir, run_id="demo")
    client = TestClient(create_app(run_roots=[("platform", runs_root)]))
    original_read_csv = pd.read_csv
    first_chunk = pd.DataFrame(
        [
            {
                "combination_id": "factor__forward_return__dataframe__long_short",
                "timestamp": "2024-01-01",
                "ic": 0.1,
                "rankic": 0.1,
                "n_obs": 4,
            },
            {
                "combination_id": "factor__forward_return__dataframe__long_short",
                "timestamp": "2024-01-02",
                "ic": 0.2,
                "rankic": 0.2,
                "n_obs": 4,
            },
        ]
    )

    def bounded_read_csv(path, *args, **kwargs):
        if str(path).replace("\\", "/").endswith("metrics/ic.csv") and kwargs.get(
            "chunksize"
        ):

            def chunks():
                yield first_chunk
                raise AssertionError(
                    "preview should stop after collecting requested rows"
                )

            return chunks()
        return original_read_csv(path, *args, **kwargs)

    monkeypatch.setattr(pd, "read_csv", bounded_read_csv)

    response = client.get(
        "/api/runs/platform/demo/tables/ic/preview?"
        "combination_id=factor__forward_return__dataframe__long_short&limit=1"
    )

    assert response.status_code == 200
    assert len(response.json()["rows"]) == 1


def test_api_csv_preview_reports_exact_eof_page(tmp_path):
    runs_root = tmp_path / "runs"
    write_evaluation_set(
        runs_root / "demo",
        run_id="demo",
        ic_values=[0.1],
        rankic_values=[0.2],
    )
    client = TestClient(create_app(run_roots=[("platform", runs_root)]))

    response = client.get("/api/runs/platform/demo/tables/ic/preview?limit=1")

    assert response.status_code == 200
    payload = response.json()
    assert len(payload["rows"]) == 1
    assert payload["total_rows"] == 1
    assert payload["has_more"] is False
    assert payload["total_rows_exact"] is True


def test_api_removed_summary_endpoint_returns_404_without_loading_artifacts(
    tmp_path,
    monkeypatch,
):
    from quant_research_platform.interfaces.api.services import run_repository

    runs_root = tmp_path / "runs"
    write_evaluation_set(runs_root / "demo", run_id="demo")

    def fail_if_full_artifact_loader_is_used(_path):
        raise AssertionError("/summary should not load artifact CSV files")

    monkeypatch.setattr(
        run_repository,
        "load_evaluation_set",
        fail_if_full_artifact_loader_is_used,
    )
    client = TestClient(create_app(run_roots=[("platform", runs_root)]))

    response = client.get("/api/runs/platform/demo/summary")

    assert response.status_code == 404


def test_api_removed_metadata_endpoints_return_404_without_loading_artifacts(
    tmp_path,
    monkeypatch,
):
    from quant_research_platform.interfaces.api.services import run_repository

    runs_root = tmp_path / "runs"
    write_evaluation_set(runs_root / "demo", run_id="demo")

    def fail_if_full_artifact_loader_is_used(_path):
        raise AssertionError("metadata endpoints should not load artifact CSV files")

    monkeypatch.setattr(
        run_repository,
        "load_evaluation_set",
        fail_if_full_artifact_loader_is_used,
    )
    client = TestClient(create_app(run_roots=[("platform", runs_root)]))

    manifest = client.get("/api/runs/platform/demo/manifest")
    config = client.get("/api/runs/platform/demo/config")

    assert manifest.status_code == 404
    assert config.status_code == 404


def test_api_returns_aggregated_run_evidence_for_fast_result_switching(tmp_path):
    runs_root = tmp_path / "runs"
    write_evaluation_set(
        runs_root / "demo",
        run_id="demo",
        domain="evaluation_synthetic",
        ic_values=[0.1, 0.2],
        rankic_values=[0.3, 0.4],
    )
    write_evaluation_set(runs_root / "other", run_id="other")
    client = TestClient(create_app(run_roots=[("platform", runs_root)]))

    response = client.get("/api/runs/platform/demo/evidence")

    assert response.status_code == 200
    payload = response.json()
    assert "evaluation_id" not in payload["summary"]
    assert payload["summary"]["run_id"] == "demo"
    assert payload["summary"]["source_alias"] == "platform"
    assert payload["summary"]["public_run_id"] == "platform/demo"
    assert payload["report"]["markdown"] == "# Report\n"
    assert "returns" in payload["tables"]
    assert "net_return" in payload["tables"]["returns"]["columns"]
    assert "performance_summary" not in payload["tables"]
    combination_summary = payload["combination_summaries"]["rows"][0]
    for key, value in {
        "final_cumulative_net_return": 0.029,
        "mean_ic": 0.15,
        "mean_rankic": 0.35,
        "mean_turnover": 2.0,
    }.items():
        assert combination_summary[key] == pytest.approx(value)
    assert {
        key: combination_summary[key]
        for key in (
            "combination_id",
            "feature",
            "label",
            "kernel",
            "portfolio_mode",
        )
    } == {
        "combination_id": "factor__forward_return__dataframe__long_short",
        "feature": "factor",
        "label": "forward_return",
        "kernel": "dataframe",
        "portfolio_mode": "long_short",
    }
    display = payload["display"]
    assert display["schema_version"] == "0.7"
    assert {"combinations", "ic", "returns", "diagnostic_summary"}.issubset(
        {table["key"] for table in display["tables"]}
    )
    display_tables = {table["key"]: table for table in display["tables"]}
    assert display_tables["diagnostic_summary"]["preview_url"] == (
        "/api/runs/platform/demo/tables/diagnostic_summary/preview"
    )
    assert display_tables["diagnostic_summary"]["preview"]["table_key"] == (
        "diagnostic_summary"
    )
    assert display_tables["combinations"]["preview_url"] == (
        "/api/runs/platform/demo/tables/combinations/preview"
    )
    assert [chart["id"] for chart in display["charts"]] == [
        "primary_cumulative_return",
    ]
    all_download = next(
        preset for preset in display["downloads"]["presets"] if preset["key"] == "all"
    )
    assert {"kind": "dynamic", "key": "comparison"} in all_download["items"]
    assert [row["public_run_id"] for row in payload["comparison"]["rows"]] == [
        "platform/demo",
        "platform/other",
    ]


def test_api_run_evidence_summary_keeps_source_alias_for_duplicate_run_ids(tmp_path):
    platform_root = tmp_path / "platform"
    crypto_root = tmp_path / "crypto"
    write_evaluation_set(
        platform_root / "demo",
        run_id="demo",
        domain="evaluation_synthetic",
    )
    write_evaluation_set(crypto_root / "demo", run_id="demo", domain="crypto")
    client = TestClient(
        create_app(run_roots=[("platform", platform_root), ("crypto", crypto_root)])
    )

    response = client.get("/api/runs/crypto/demo/evidence")

    assert response.status_code == 200
    assert response.json()["summary"]["run_id"] == "demo"
    assert response.json()["summary"]["source_alias"] == "crypto"
    assert response.json()["summary"]["public_run_id"] == "crypto/demo"


def test_api_source_alias_scopes_preview_download_and_zip_for_duplicate_run_ids(
    tmp_path,
):
    platform_root = tmp_path / "platform"
    crypto_root = tmp_path / "crypto"
    write_evaluation_set(platform_root / "demo", run_id="demo", ic_values=[0.11])
    write_evaluation_set(crypto_root / "demo", run_id="demo", ic_values=[0.88])
    client = TestClient(
        create_app(run_roots=[("platform", platform_root), ("crypto", crypto_root)])
    )

    preview = client.get("/api/runs/crypto/demo/tables/ic/preview")
    direct = client.get("/api/runs/crypto/demo/artifacts/table:ic")
    zipped = client.post(
        "/api/runs/crypto/demo/download",
        json={"items": ["table:ic"]},
    )

    assert preview.status_code == 200
    assert preview.json()["rows"][0]["ic"] == pytest.approx(0.88)
    assert direct.status_code == 200
    assert "0.88" in direct.text
    assert "0.11" not in direct.text
    assert zipped.status_code == 200
    with zipfile.ZipFile(BytesIO(zipped.content)) as archive:
        assert "0.88" in archive.read("metrics/ic.csv").decode("utf-8")
    assert list((tmp_path / "download_cache" / "crypto" / "demo").glob("*.zip"))


def test_api_evidence_eager_tables_follow_display_charts(tmp_path):
    runs_root = tmp_path / "runs"
    run_dir = runs_root / "demo"
    write_evaluation_set(run_dir, run_id="demo")
    client = TestClient(create_app(run_roots=[("platform", runs_root)]))

    response = client.get("/api/runs/platform/demo/evidence")

    assert response.status_code == 200
    payload = response.json()
    assert "returns" in payload["tables"]
    assert "net_return" in payload["tables"]["returns"]["columns"]


def test_api_evidence_uses_preview_loader_for_chart_tables(tmp_path, monkeypatch):
    from quant_research_platform.artifacts import artifact_loader

    runs_root = tmp_path / "runs"
    run_dir = runs_root / "demo"
    write_evaluation_set(run_dir, run_id="demo")
    read_table = artifact_loader._read_table
    full_load_attempts: list[str] = []

    def fail_if_chart_table_is_full_loaded(path):
        if str(path).replace("\\", "/").endswith("portfolio/cumulative_returns.csv"):
            full_load_attempts.append(str(path))
            raise AssertionError("chart table should use preview loading")
        return read_table(path)

    monkeypatch.setattr(
        artifact_loader,
        "_read_table",
        fail_if_chart_table_is_full_loaded,
    )
    client = TestClient(create_app(run_roots=[("platform", runs_root)]))

    response = client.get("/api/runs/platform/demo/evidence")

    assert response.status_code == 200
    assert "returns" in response.json()["tables"]
    assert full_load_attempts == []


def test_api_serves_factor_csv_template() -> None:
    client = TestClient(create_app(run_roots=[("platform", "reports/runs")]))

    response = client.get("/api/templates/factor-csv")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/csv")
    assert "attachment" in response.headers["content-disposition"]
    assert "timestamp,asset,factor,forward_return,forward_return_1d" in response.text


def test_api_creates_run_from_uploaded_csv_payload(tmp_path):
    runs_root = tmp_path / "runs"
    client = TestClient(create_app(run_roots=[("platform", runs_root)]))
    payload = {
        "run_id": "uploaded_demo",
        "project": "quant-research-platform",
        "kind": "factor",
        "domain": "user_upload",
        "data_stage": "uploaded_panel",
        "uses_external_data": True,
        "timestamp_column": "timestamp",
        "asset_column": "asset",
        "feature_columns": ["factor"],
        "label_columns": ["forward_return"],
        "return_source_column": "forward_return_1d",
        "n_groups": 2,
        "weight_method": "equal_weight_quantile",
        "portfolio_mode": "long_only",
        "direction": "long_high",
        "top_quantile": 0.5,
        "bottom_quantile": 0.5,
        "csv_text": "\n".join(
            [
                "timestamp,asset,factor,forward_return,forward_return_1d",
                "2026-01-01,A,0.1,0.02,0.006",
                "2026-01-01,B,0.2,0.01,0.002",
                "2026-01-02,A,0.3,0.03,0.008",
                "2026-01-02,B,0.1,-0.01,-0.002",
            ]
        ),
    }

    response = client.post("/api/runs", json=payload)

    assert response.status_code == 201
    body = response.json()
    assert body["public_run_id"] == "platform/uploaded_demo"
    assert body["domain"] == "user_upload"
    assert (runs_root / "uploaded_demo" / "portfolio" / "returns.csv").is_file()
    returns = (runs_root / "uploaded_demo" / "portfolio" / "returns.csv").read_text(
        encoding="utf-8"
    )
    assert "long_only" in returns
    assert not (runs_root / "uploaded_demo" / "portfolio" / "long_short.csv").exists()


def test_api_uploaded_csv_rejects_non_uploaded_panel_stage(tmp_path):
    runs_root = tmp_path / "runs"
    client = TestClient(create_app(run_roots=[("platform", runs_root)]))
    payload = {
        "run_id": "uploaded_stage_demo",
        "project": "quant-research-platform",
        "kind": "factor",
        "domain": "user_upload",
        "data_stage": "synthetic_only",
        "uses_external_data": True,
        "timestamp_column": "timestamp",
        "asset_column": "asset",
        "feature_columns": ["factor"],
        "label_columns": ["forward_return"],
        "return_source_column": "forward_return_1d",
        "n_groups": 2,
        "weight_method": "equal_weight_quantile",
        "portfolio_mode": "long_only",
        "direction": "long_high",
        "top_quantile": 0.5,
        "bottom_quantile": 0.5,
        "csv_text": "\n".join(
            [
                "timestamp,asset,factor,forward_return,forward_return_1d",
                "2026-01-01,A,0.1,0.02,0.006",
                "2026-01-01,B,0.2,0.01,0.002",
                "2026-01-02,A,0.3,0.03,0.008",
                "2026-01-02,B,0.1,-0.01,-0.002",
            ]
        ),
    }

    response = client.post("/api/runs", json=payload)

    assert response.status_code == 422
    assert "data_stage" in response.text
    assert not (runs_root / "uploaded_stage_demo").exists()


def test_api_uploaded_csv_rejects_existing_non_manifest_directory(tmp_path):
    runs_root = tmp_path / "runs"
    existing = runs_root / "uploaded_demo"
    existing.mkdir(parents=True)
    (existing / "keep.txt").write_text("keep", encoding="utf-8")
    client = TestClient(create_app(run_roots=[("platform", runs_root)]))
    payload = {
        "run_id": "uploaded_demo",
        "project": "quant-research-platform",
        "kind": "factor",
        "domain": "user_upload",
        "data_stage": "uploaded_panel",
        "uses_external_data": True,
        "timestamp_column": "timestamp",
        "asset_column": "asset",
        "feature_columns": ["factor"],
        "label_columns": ["forward_return"],
        "return_source_column": "forward_return_1d",
        "n_groups": 2,
        "weight_method": "equal_weight_quantile",
        "portfolio_mode": "long_only",
        "direction": "long_high",
        "top_quantile": 0.5,
        "bottom_quantile": 0.5,
        "csv_text": "\n".join(
            [
                "timestamp,asset,factor,forward_return,forward_return_1d",
                "2026-01-01,A,0.1,0.02,0.006",
                "2026-01-01,B,0.2,0.01,0.002",
                "2026-01-02,A,0.3,0.03,0.008",
                "2026-01-02,B,0.1,-0.01,-0.002",
            ]
        ),
    }

    response = client.post("/api/runs", json=payload)

    assert response.status_code == 409
    assert (existing / "keep.txt").read_text(encoding="utf-8") == "keep"
    assert not (existing / "manifest.json").exists()


def test_api_deletes_uploaded_run(tmp_path):
    runs_root = tmp_path / "runs"
    write_evaluation_set(
        runs_root / "uploaded_demo",
        run_id="uploaded_demo",
        domain="user_upload",
        data_stage="uploaded_panel",
        uses_external_data=True,
    )
    client = TestClient(create_app(run_roots=[("platform", runs_root)]))

    response = client.delete("/api/runs/platform/uploaded_demo")

    assert response.status_code == 200
    assert response.json() == {"deleted": "platform/uploaded_demo"}
    assert not (runs_root / "uploaded_demo").exists()


def test_api_rejects_deleting_non_uploaded_run(tmp_path):
    runs_root = tmp_path / "runs"
    write_evaluation_set(runs_root / "demo", run_id="demo", data_stage="synthetic_only")
    client = TestClient(create_app(run_roots=[("platform", runs_root)]))

    response = client.delete("/api/runs/platform/demo")

    assert response.status_code == 403
    assert "只能删除用户上传" in response.json()["detail"]
    assert (runs_root / "demo" / "manifest.json").is_file()


def test_api_rejects_deleting_unknown_source_alias(tmp_path):
    runs_root = tmp_path / "runs"
    write_evaluation_set(
        runs_root / "uploaded_demo",
        run_id="uploaded_demo",
        data_stage="uploaded_panel",
    )
    client = TestClient(create_app(run_roots=[("platform", runs_root)]))

    response = client.delete("/api/runs/private/uploaded_demo")

    assert response.status_code == 404
    assert (runs_root / "uploaded_demo" / "manifest.json").is_file()


def test_api_rejects_deleting_unknown_uploaded_run(tmp_path):
    runs_root = tmp_path / "runs"
    client = TestClient(create_app(run_roots=[("platform", runs_root)]))

    response = client.delete("/api/runs/platform/missing_run")

    assert response.status_code == 404


def test_api_rejects_deleting_directory_without_manifest(tmp_path):
    runs_root = tmp_path / "runs"
    orphan = runs_root / "orphan_uploaded"
    orphan.mkdir(parents=True)
    (orphan / "summary.json").write_text(
        json.dumps({"data_stage": "uploaded_panel"}),
        encoding="utf-8",
    )
    client = TestClient(create_app(run_roots=[("platform", runs_root)]))

    response = client.delete("/api/runs/platform/orphan_uploaded")

    assert response.status_code == 404
    assert orphan.is_dir()


def test_api_rejects_deleting_nested_uploaded_run(tmp_path):
    runs_root = tmp_path / "runs"
    nested = runs_root / "nested" / "uploaded_demo"
    write_evaluation_set(
        nested,
        run_id="uploaded_demo",
        data_stage="uploaded_panel",
    )
    client = TestClient(create_app(run_roots=[("platform", runs_root)]))

    response = client.delete("/api/runs/platform/uploaded_demo")

    assert response.status_code == 404
    assert (nested / "manifest.json").is_file()


def test_api_downloads_artifact_csv(tmp_path):
    runs_root = tmp_path / "runs"
    write_evaluation_set(runs_root / "demo", run_id="demo")
    client = TestClient(create_app(run_roots=[("platform", runs_root)]))

    response = client.get(
        "/api/runs/platform/demo/artifacts/table:evaluation_metrics_summary"
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/csv")
    assert "attachment" in response.headers["content-disposition"]
    assert (
        'filename="evaluation_metrics_summary.csv"'
        in response.headers["content-disposition"]
    )
    assert "final_cumulative_net_return" in response.text
    assert "total_net_portfolio_return" not in response.text


@pytest.mark.parametrize(
    ("item", "expected"),
    [
        ("metadata_file:manifest", "artifact_schema_version"),
        ("metadata_file:summary", "run_id"),
        ("metadata_file:run_context", "provenance"),
    ],
)
def test_api_downloads_display_exposed_metadata_artifacts(tmp_path, item, expected):
    runs_root = tmp_path / "runs"
    write_evaluation_set(runs_root / "demo", run_id="demo")
    client = TestClient(create_app(run_roots=[("platform", runs_root)]))

    response = client.get(f"/api/runs/platform/demo/artifacts/{item}")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/json")
    filename = {
        "metadata_file:manifest": "manifest.json",
        "metadata_file:summary": "summary.json",
        "metadata_file:run_context": "run_context.json",
    }[item]
    assert f'filename="{filename}"' in response.headers["content-disposition"]
    assert expected in response.json()


def test_api_direct_metadata_artifact_must_be_exposed_by_display(tmp_path):
    runs_root = tmp_path / "runs"
    run_dir = runs_root / "demo"
    write_evaluation_set(run_dir, run_id="demo")
    _remove_display_download_item(run_dir / "manifest.json", "metadata_file:summary")
    client = TestClient(create_app(run_roots=[("platform", runs_root)]))

    response = client.get("/api/runs/platform/demo/artifacts/metadata_file:summary")

    assert response.status_code == 422
    assert "missing metadata refs" in response.json()["detail"]


def test_api_direct_metadata_download_uses_canonical_file_path(tmp_path):
    runs_root = tmp_path / "runs"
    run_dir = runs_root / "demo"
    write_evaluation_set(run_dir, run_id="demo")
    (run_dir / "other_summary.json").write_text(
        json.dumps({"evaluation_id": "wrong"}),
        encoding="utf-8",
    )
    client = TestClient(create_app(run_roots=[("platform", runs_root)]))

    response = client.get("/api/runs/platform/demo/artifacts/metadata_file:summary")

    assert response.status_code == 200
    assert response.json()["run_id"] == "demo"


@pytest.mark.parametrize(
    ("item", "filename", "mutate"),
    [
        (
            "metadata_file:manifest",
            "manifest.json",
            lambda payload: payload.update({"local_path": "/home/user/manifest"}),
        ),
        (
            "metadata_file:summary",
            "summary.json",
            lambda payload: payload.update({"input_path": "/Users/user/summary"}),
        ),
        (
            "metadata_file:run_context",
            "run_context.json",
            lambda payload: payload["config"].update(
                {"output_dir": "/mnt/private/run_context"}
            ),
        ),
    ],
)
def test_api_direct_metadata_download_sanitizes_output(
    tmp_path,
    item,
    filename,
    mutate,
):
    runs_root = tmp_path / "runs"
    run_dir = runs_root / "demo"
    write_evaluation_set(run_dir, run_id="demo")
    metadata_path = run_dir / filename
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    mutate(payload)
    metadata_path.write_text(json.dumps(payload), encoding="utf-8")
    client = TestClient(create_app(run_roots=[("platform", runs_root)]))

    response = client.get(f"/api/runs/platform/demo/artifacts/{item}")

    assert response.status_code == 200
    text = response.text
    assert "/home/user" not in text
    assert "/Users/user" not in text
    assert "/mnt/private" not in text
    assert "[redacted-path]" in text


def test_api_downloads_artifact_csv_without_full_artifact_load(
    tmp_path,
    monkeypatch,
):
    from quant_research_platform.interfaces.api.services import run_repository

    runs_root = tmp_path / "runs"
    write_evaluation_set(runs_root / "demo", run_id="demo")

    def fail_if_full_artifact_loader_is_used(_path, *, table_names=None):
        raise AssertionError("artifact CSV download should stream the artifact file")

    monkeypatch.setattr(
        run_repository,
        "load_evaluation_set",
        fail_if_full_artifact_loader_is_used,
    )
    client = TestClient(create_app(run_roots=[("platform", runs_root)]))

    response = client.get(
        "/api/runs/platform/demo/artifacts/table:evaluation_metrics_summary"
    )

    assert response.status_code == 200
    assert (
        response.content
        == (
            runs_root / "demo" / "metrics" / "evaluation_metrics_summary.csv"
        ).read_bytes()
    )


def test_api_direct_artifact_csv_must_be_exposed_by_display(tmp_path):
    runs_root = tmp_path / "runs"
    run_dir = runs_root / "demo"
    write_evaluation_set(run_dir, run_id="demo")
    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["display"]["downloads"]["items"] = [
        item
        for item in manifest["display"]["downloads"]["items"]
        if not (
            item.get("kind") == "table"
            and item.get("key") == "evaluation_metrics_summary"
        )
    ]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    client = TestClient(create_app(run_roots=[("platform", runs_root)]))

    response = client.get(
        "/api/runs/platform/demo/artifacts/table:evaluation_metrics_summary"
    )

    assert response.status_code == 422
    detail = response.json()["detail"]
    assert "missing table refs" in detail


def test_api_direct_artifact_csv_rejects_unsupported_display_schema(tmp_path):
    runs_root = tmp_path / "runs"
    run_dir = runs_root / "demo"
    write_evaluation_set(run_dir, run_id="demo")
    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["display"]["schema_version"] = "0.4"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    client = TestClient(create_app(run_roots=[("platform", runs_root)]))

    response = client.get(
        "/api/runs/platform/demo/artifacts/table:evaluation_metrics_summary"
    )

    assert response.status_code == 422
    assert "display schema_version" in response.json()["detail"]


def test_api_rejects_unsupported_display_schema_at_app_entry_points(tmp_path):
    runs_root = tmp_path / "runs"
    run_dir = runs_root / "demo"
    write_evaluation_set(run_dir, run_id="demo")
    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["display"]["schema_version"] = "0.4"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    client = TestClient(create_app(run_roots=[("platform", runs_root)]))

    evidence = client.get("/api/runs/platform/demo/evidence")
    preview = client.get("/api/runs/platform/demo/tables/ic/preview")
    zip_post = client.post(
        "/api/runs/platform/demo/download",
        json={"items": ["metadata_file:report"]},
    )
    zip_get = client.get("/api/runs/platform/demo/download?items=metadata_file:report")

    assert evidence.status_code == 422
    assert preview.status_code == 422
    assert zip_post.status_code == 422
    assert zip_get.status_code == 422


def test_api_run_index_skips_unsupported_display_schema(tmp_path):
    runs_root = tmp_path / "runs"
    write_evaluation_set(runs_root / "good", run_id="good")
    run_dir = runs_root / "demo"
    write_evaluation_set(run_dir, run_id="demo")
    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["display"]["schema_version"] = "0.4"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    client = TestClient(create_app(run_roots=[("platform", runs_root)]))

    response = client.get("/api/runs")

    assert response.status_code == 200
    assert [run["public_run_id"] for run in response.json()] == ["platform/good"]

    target = client.get("/api/runs/platform/demo/evidence")
    assert target.status_code == 422
    assert "display schema_version" in target.json()["detail"]


def test_api_breakdown_preview_rejects_wrong_group(tmp_path):
    runs_root = tmp_path / "runs"
    write_evaluation_set(
        runs_root / "demo",
        run_id="demo",
        time_breakdowns=("year",),
    )
    client = TestClient(create_app(run_roots=[("platform", runs_root)]))

    response = client.get(
        "/api/runs/platform/demo/breakdowns/not_a_period/performance_summary_by_year"
    )

    assert response.status_code == 404


def test_api_breakdown_preview_reads_period_breakdown_table(tmp_path):
    runs_root = tmp_path / "runs"
    write_evaluation_set(
        runs_root / "demo",
        run_id="demo",
        time_breakdowns=("year",),
    )
    client = TestClient(create_app(run_roots=[("platform", runs_root)]))

    response = client.get(
        "/api/runs/platform/demo/tables/evaluation_metrics_summary_by_year/preview?limit=1"
    )

    assert response.status_code == 200
    payload = response.json()
    assert "period_type" in payload["columns"]
    assert payload["rows"][0]["period_type"] == "year"
    assert payload["rows"][0]["period"] == 2026


def test_api_preview_routes_reject_wrong_display_group(tmp_path):
    runs_root = tmp_path / "runs"
    write_evaluation_set(runs_root / "demo", run_id="demo")
    client = TestClient(create_app(run_roots=[("platform", runs_root)]))

    metrics_for_portfolio = client.get(
        "/api/runs/platform/demo/metrics/cumulative_returns"
    )
    portfolio_for_metric = client.get("/api/runs/platform/demo/portfolio/ic")
    diagnostics_for_portfolio = client.get(
        "/api/runs/platform/demo/diagnostics/cumulative_returns"
    )

    assert metrics_for_portfolio.status_code == 404
    assert portfolio_for_metric.status_code == 404
    assert diagnostics_for_portfolio.status_code == 404


def test_api_downloads_weights_artifact_parquet(tmp_path):
    runs_root = tmp_path / "runs"
    write_evaluation_set(runs_root / "demo", run_id="demo")
    client = TestClient(create_app(run_roots=[("platform", runs_root)]))

    response = client.get("/api/runs/platform/demo/artifacts/table:weights")

    assert response.status_code == 200
    assert response.headers["content-type"] == "application/vnd.apache.parquet"
    assert 'filename="weights.parquet"' in response.headers["content-disposition"]


def test_api_rejects_retired_neutralization_diagnostics_artifact(tmp_path):
    runs_root = tmp_path / "runs"
    write_evaluation_set(runs_root / "demo", run_id="demo")
    client = TestClient(create_app(run_roots=[("platform", runs_root)]))

    response = client.get(
        "/api/runs/platform/demo/artifacts/diagnostics/neutralization_diagnostics.csv"
    )

    assert response.status_code == 422


def test_api_previews_diagnostic_summary_table(tmp_path):
    runs_root = tmp_path / "runs"
    write_evaluation_set(runs_root / "demo", run_id="demo")
    client = TestClient(create_app(run_roots=[("platform", runs_root)]))

    response = client.get("/api/runs/platform/demo/tables/diagnostic_summary/preview")

    assert response.status_code == 200
    payload = response.json()
    assert "policy_family" in payload["columns"]
    assert payload["rows"][0]["diagnostic_status"] == "disabled"


def test_api_generic_table_preview_reads_display_table_by_key(tmp_path):
    runs_root = tmp_path / "runs"
    write_evaluation_set(runs_root / "demo", run_id="demo")
    client = TestClient(create_app(run_roots=[("platform", runs_root)]))

    response = client.get("/api/runs/platform/demo/tables/diagnostic_summary/preview")
    legacy = client.get("/api/runs/platform/demo/diagnostics/tradability_diagnostics")

    assert response.status_code == 200
    assert legacy.status_code == 404
    assert response.json()["limit"] == 20


def test_api_generic_table_preview_supports_offset(tmp_path):
    runs_root = tmp_path / "runs"
    write_evaluation_set(
        runs_root / "demo",
        run_id="demo",
        ic_values=[0.1, 0.2, 0.3],
        rankic_values=[0.3, 0.4, 0.5],
    )
    client = TestClient(create_app(run_roots=[("platform", runs_root)]))

    response = client.get("/api/runs/platform/demo/tables/ic/preview?limit=1&offset=1")

    assert response.status_code == 200
    payload = response.json()
    assert payload["offset"] == 1
    assert payload["rows"][0]["ic"] == pytest.approx(0.2)


def test_api_generic_table_preview_rejects_oversized_limit(tmp_path):
    runs_root = tmp_path / "runs"
    write_evaluation_set(runs_root / "demo", run_id="demo")
    client = TestClient(create_app(run_roots=[("platform", runs_root)]))

    response = client.get("/api/runs/platform/demo/tables/ic/preview?limit=1001")

    assert response.status_code == 422
    assert response.json()["detail"] == "limit must be between 1 and 1000"


def test_api_generic_table_preview_rejects_combination_filter_for_static_table(
    tmp_path,
):
    runs_root = tmp_path / "runs"
    write_evaluation_set(runs_root / "demo", run_id="demo")
    client = TestClient(create_app(run_roots=[("platform", runs_root)]))

    response = client.get(
        "/api/runs/platform/demo/tables/combinations/preview?combination_id=demo"
    )

    assert response.status_code == 422
    assert response.json()["detail"] == (
        "table does not support combination_id filtering"
    )


def test_table_preview_service_rejects_missing_combination_filter_column(tmp_path):
    from quant_research_platform.artifacts import ARTIFACT_SCHEMA_VERSION
    from quant_research_platform.interfaces.api.services.table_preview_service import (
        table_preview_payload,
    )

    pd.DataFrame({"timestamp": ["2026-01-01"], "value": [1.0]}).to_csv(
        tmp_path / "custom.csv",
        index=False,
    )
    manifest = {
        "display": {
            "schema_version": ARTIFACT_SCHEMA_VERSION,
            "tables": [
                {
                    "key": "custom",
                    "path": "custom.csv",
                    "preview": {
                        "kind": "table",
                        "table_key": "custom",
                        "default_limit": 20,
                        "max_limit": 1000,
                        "supports_offset": True,
                        "supports_combination_filter": True,
                        "filter_parameter": "combination_id",
                    },
                }
            ],
        }
    }

    with pytest.raises(HTTPException) as exc:
        table_preview_payload(
            tmp_path,
            manifest,
            "custom",
            combination_id="combo_a",
            limit=None,
            offset=0,
        )

    assert exc.value.status_code == 422
    assert exc.value.detail == "table is missing combination_id column"


def test_api_generic_table_preview_rejects_table_not_in_display(tmp_path):
    runs_root = tmp_path / "runs"
    run_dir = runs_root / "demo"
    write_evaluation_set(run_dir, run_id="demo")
    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifacts"]["metrics"]["private_debug"] = "metrics/private_debug.csv"
    pd.DataFrame({"value": [1]}).to_csv(
        run_dir / "metrics" / "private_debug.csv",
        index=False,
    )
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    client = TestClient(create_app(run_roots=[("platform", runs_root)]))

    response = client.get("/api/runs/platform/demo/tables/private_debug/preview")

    assert response.status_code == 422
    assert "not in schema registry" in response.json()["detail"]


def test_api_rejects_stale_run_missing_display_preview_descriptor(tmp_path):
    runs_root = tmp_path / "runs"
    run_dir = runs_root / "demo"
    write_evaluation_set(run_dir, run_id="demo")
    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["display"]["tables"][0].pop("preview")
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    client = TestClient(create_app(run_roots=[("platform", runs_root)]))

    assert client.get("/api/runs").json() == []
    evidence = client.get("/api/runs/platform/demo/evidence")
    preview = client.get("/api/runs/platform/demo/tables/ic/preview")

    assert evidence.status_code == 422
    assert "preview" in evidence.json()["detail"]
    assert preview.status_code == 422
    assert "preview" in preview.json()["detail"]


def test_api_previews_parquet_weights_table(tmp_path):
    runs_root = tmp_path / "runs"
    write_evaluation_set(runs_root / "demo", run_id="demo")
    client = TestClient(create_app(run_roots=[("platform", runs_root)]))

    response = client.get("/api/runs/platform/demo/tables/weights/preview?limit=20")

    assert response.status_code == 404


def test_api_previews_parquet_weights_defaults_to_display_preview_rows(tmp_path):
    runs_root = tmp_path / "runs"
    write_evaluation_set(runs_root / "demo", run_id="demo")
    client = TestClient(create_app(run_roots=[("platform", runs_root)]))

    response = client.get("/api/runs/platform/demo/tables/weights/preview")

    assert response.status_code == 404


def test_api_previews_parquet_weights_without_full_artifact_load(
    tmp_path,
    monkeypatch,
):
    runs_root = tmp_path / "runs"
    write_evaluation_set(runs_root / "demo", run_id="demo")
    client = TestClient(create_app(run_roots=[("platform", runs_root)]))

    def fail_read_parquet(*_args, **_kwargs):
        raise AssertionError("weights preview should not full-read parquet")

    monkeypatch.setattr(pd, "read_parquet", fail_read_parquet)

    response = client.get("/api/runs/platform/demo/tables/weights/preview?limit=20")

    assert response.status_code == 404


def test_api_parquet_preview_stops_after_filtered_page_is_collected(
    tmp_path,
    monkeypatch,
):
    import pyarrow as pa
    import pyarrow.parquet as pq

    runs_root = tmp_path / "runs"
    write_evaluation_set(runs_root / "demo", run_id="demo")
    client = TestClient(create_app(run_roots=[("platform", runs_root)]))
    original_parquet_file = pq.ParquetFile
    frame = pd.DataFrame(
        [
            {
                "combination_id": "factor__forward_return__dataframe__long_short",
                "timestamp": "2026-01-01",
                "asset": "asset_001",
                "target_weight": 1.0,
            },
            {
                "combination_id": "factor__forward_return__dataframe__long_short",
                "timestamp": "2026-01-02",
                "asset": "asset_001",
                "target_weight": 1.0,
            },
        ]
    )

    class FakeParquetFile:
        schema_arrow = pa.schema(
            [
                ("combination_id", pa.string()),
                ("timestamp", pa.string()),
                ("asset", pa.string()),
                ("target_weight", pa.float64()),
            ]
        )

        def iter_batches(self, *, batch_size):
            del batch_size
            yield pa.RecordBatch.from_pandas(frame)
            raise AssertionError("preview should stop after collecting requested rows")

    def bounded_parquet_file(path, *args, **kwargs):
        if str(path).replace("\\", "/").endswith("portfolio/weights.parquet"):
            return FakeParquetFile()
        return original_parquet_file(path, *args, **kwargs)

    monkeypatch.setattr(pq, "ParquetFile", bounded_parquet_file)

    response = client.get(
        "/api/runs/platform/demo/tables/weights/preview?"
        "combination_id=factor__forward_return__dataframe__long_short&limit=1"
    )

    assert response.status_code == 404


def test_api_parquet_filtered_preview_reports_exact_eof_page(tmp_path):
    runs_root = tmp_path / "runs"
    write_evaluation_set(runs_root / "demo", run_id="demo")
    client = TestClient(create_app(run_roots=[("platform", runs_root)]))

    response = client.get(
        "/api/runs/platform/demo/tables/weights/preview?"
        "combination_id=factor__forward_return__dataframe__long_short&limit=1"
    )

    assert response.status_code == 404


def test_api_downloads_diagnostic_summary_artifact_csv(tmp_path):
    runs_root = tmp_path / "runs"
    write_evaluation_set(runs_root / "demo", run_id="demo")
    client = TestClient(create_app(run_roots=[("platform", runs_root)]))

    response = client.get("/api/runs/platform/demo/artifacts/table:diagnostic_summary")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/csv")
    assert (
        'filename="diagnostic_summary.csv"' in response.headers["content-disposition"]
    )
    assert "diagnostics_contract" in response.text


def test_api_downloads_report_markdown_and_rejects_removed_excel_workbook(tmp_path):
    runs_root = tmp_path / "runs"
    write_evaluation_set(runs_root / "demo", run_id="demo")
    client = TestClient(create_app(run_roots=[("platform", runs_root)]))

    report = client.get("/api/runs/platform/demo/report.md")
    workbook = client.get("/api/runs/platform/demo/artifacts.xlsx")

    assert report.status_code == 200
    assert report.headers["content-type"].startswith("text/markdown")
    assert 'filename="report.md"' in report.headers["content-disposition"]
    assert report.text == "# Report\n"
    assert workbook.status_code == 404


def test_api_downloads_report_markdown_without_full_artifact_load(
    tmp_path,
    monkeypatch,
):
    runs_root = tmp_path / "runs"
    write_evaluation_set(runs_root / "demo", run_id="demo")
    client = TestClient(create_app(run_roots=[("platform", runs_root)]))

    def fail_read_parquet(*_args, **_kwargs):
        raise AssertionError("report.md download should not full-load run tables")

    monkeypatch.setattr(pd, "read_parquet", fail_read_parquet)

    response = client.get("/api/runs/platform/demo/report.md")

    assert response.status_code == 200
    assert response.text == "# Report\n"


def test_api_report_markdown_must_be_exposed_by_display_downloads(tmp_path):
    runs_root = tmp_path / "runs"
    run_dir = runs_root / "demo"
    write_evaluation_set(run_dir, run_id="demo")
    _remove_display_download_item(run_dir / "manifest.json", "report.md")
    client = TestClient(create_app(run_roots=[("platform", runs_root)]))

    response = client.get("/api/runs/platform/demo/report.md")

    assert response.status_code == 422
    assert "missing metadata refs" in response.json()["detail"]


def test_api_report_markdown_requires_canonical_report_artifact(tmp_path):
    runs_root = tmp_path / "runs"
    run_dir = runs_root / "demo"
    write_evaluation_set(run_dir, run_id="demo")
    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifacts"]["report"] = "private_report.md"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    (run_dir / "private_report.md").write_text("# Private Report\n", encoding="utf-8")
    client = TestClient(create_app(run_roots=[("platform", runs_root)]))

    response = client.get("/api/runs/platform/demo/report.md")

    assert response.status_code == 422
    assert "Private Report" not in response.text


def test_api_report_markdown_returns_422_when_file_missing(tmp_path):
    runs_root = tmp_path / "runs"
    run_dir = runs_root / "demo"
    write_evaluation_set(run_dir, run_id="demo")
    (run_dir / "report.md").unlink()
    client = TestClient(create_app(run_roots=[("platform", runs_root)]))

    response = client.get("/api/runs/platform/demo/report.md")

    assert response.status_code == 422


def test_api_evidence_report_must_be_exposed_by_display_downloads(tmp_path):
    runs_root = tmp_path / "runs"
    run_dir = runs_root / "demo"
    write_evaluation_set(run_dir, run_id="demo")
    _remove_display_download_item(run_dir / "manifest.json", "report.md")
    client = TestClient(create_app(run_roots=[("platform", runs_root)]))

    response = client.get("/api/runs/platform/demo/evidence")

    assert response.status_code == 422
    assert "missing metadata refs" in response.json()["detail"]


def test_api_downloads_selected_artifacts_as_zip(tmp_path):
    runs_root = tmp_path / "runs"
    write_evaluation_set(runs_root / "demo", run_id="demo")
    write_evaluation_set(runs_root / "other", run_id="other")
    client = TestClient(create_app(run_roots=[("platform", runs_root)]))

    response = client.post(
        "/api/runs/platform/demo/download",
        json={
            "items": [
                "dynamic:comparison",
                "metadata_file:report",
                "metadata_file:summary",
                "table:evaluation_metrics_summary",
                "table:weights",
                "table:diagnostic_summary",
            ]
        },
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/zip")
    assert (
        'filename="platform_demo_selected_artifacts.zip"'
        in response.headers["content-disposition"]
    )
    with zipfile.ZipFile(BytesIO(response.content)) as archive:
        assert set(archive.namelist()) == {
            "comparison.csv",
            "report.md",
            "summary.json",
            "metrics/evaluation_metrics_summary.csv",
            "portfolio/weights.parquet",
            "diagnostics/diagnostic_summary.csv",
        }
        assert b"public_run_id" in archive.read("comparison.csv")
        assert b"platform/demo" in archive.read("comparison.csv")
        assert b"platform/other" in archive.read("comparison.csv")
        assert (
            archive.read("report.md") == (runs_root / "demo" / "report.md").read_bytes()
        )
        assert b"final_cumulative_net_return" in archive.read(
            "metrics/evaluation_metrics_summary.csv"
        )
        assert archive.read("portfolio/weights.parquet")
        assert b"diagnostics_contract" in archive.read(
            "diagnostics/diagnostic_summary.csv"
        )


def test_api_downloads_selected_artifacts_from_query_items(tmp_path):
    runs_root = tmp_path / "runs"
    write_evaluation_set(runs_root / "demo", run_id="demo")
    client = TestClient(create_app(run_roots=[("platform", runs_root)]))

    response = client.get(
        "/api/runs/platform/demo/download",
        params=[
            ("items", "metadata_file:report"),
            ("items", "metadata_file:summary"),
        ],
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/zip")
    with zipfile.ZipFile(BytesIO(response.content)) as archive:
        assert archive.namelist() == ["report.md", "summary.json"]


def test_api_selected_zip_metadata_download_uses_canonical_file_path(tmp_path):
    runs_root = tmp_path / "runs"
    run_dir = runs_root / "demo"
    write_evaluation_set(run_dir, run_id="demo")
    (run_dir / "evil_summary.json").write_text(
        json.dumps({"run_id": "wrong"}),
        encoding="utf-8",
    )
    client = TestClient(create_app(run_roots=[("platform", runs_root)]))

    response = client.post(
        "/api/runs/platform/demo/download",
        json={"items": ["metadata_file:summary"]},
    )

    assert response.status_code == 200
    with zipfile.ZipFile(BytesIO(response.content)) as archive:
        assert json.loads(archive.read("summary.json"))["run_id"] == "demo"


def test_api_selected_zip_sanitizes_metadata_output(tmp_path):
    runs_root = tmp_path / "runs"
    run_dir = runs_root / "demo"
    write_evaluation_set(run_dir, run_id="demo")
    summary_path = run_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["input_path"] = "/Users/user/private/summary"
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    client = TestClient(create_app(run_roots=[("platform", runs_root)]))

    response = client.post(
        "/api/runs/platform/demo/download",
        json={"items": ["metadata_file:summary"]},
    )

    assert response.status_code == 200
    with zipfile.ZipFile(BytesIO(response.content)) as archive:
        text = archive.read("summary.json").decode("utf-8")
        assert "/Users/user" not in text
        assert "[redacted-path]" in text


def test_api_selected_artifact_zip_reuses_files_and_writes_cache(
    tmp_path,
    monkeypatch,
):
    import pandas as pd

    runs_root = tmp_path / "runs"
    write_evaluation_set(runs_root / "demo", run_id="demo")
    client = TestClient(create_app(run_roots=[("platform", runs_root)]))
    original_to_csv = pd.DataFrame.to_csv
    to_csv_calls: list[list[str]] = []

    def spy_to_csv(self, *args, **kwargs):
        to_csv_calls.append(list(self.columns))
        return original_to_csv(self, *args, **kwargs)

    monkeypatch.setattr(pd.DataFrame, "to_csv", spy_to_csv)

    response = client.post(
        "/api/runs/platform/demo/download",
        json={
            "items": [
                "metadata_file:report",
                "metadata_file:summary",
                "table:evaluation_metrics_summary",
            ]
        },
    )

    assert response.status_code == 200
    assert to_csv_calls == []
    cache_files = sorted(
        (tmp_path / "download_cache" / "platform" / "demo").glob("*.zip")
    )
    assert len(cache_files) == 1
    assert cache_files[0].read_bytes() == response.content


def test_api_rejects_retired_lineage_metadata_download(tmp_path):
    runs_root = tmp_path / "runs"
    write_evaluation_set(runs_root / "demo", run_id="demo")
    client = TestClient(create_app(run_roots=[("platform", runs_root)]))

    direct = client.get("/api/runs/platform/demo/artifacts/lineage.json")
    zipped = client.post(
        "/api/runs/platform/demo/download",
        json={"items": ["lineage.json"]},
    )

    assert direct.status_code == 422
    assert zipped.status_code == 422


def test_api_rejects_retired_config_metadata_download(tmp_path):
    runs_root = tmp_path / "runs"
    write_evaluation_set(runs_root / "demo", run_id="demo")
    client = TestClient(create_app(run_roots=[("platform", runs_root)]))

    direct = client.get("/api/runs/platform/demo/artifacts/config.json")
    zipped = client.post(
        "/api/runs/platform/demo/download",
        json={"items": ["config.json"]},
    )

    assert direct.status_code == 422
    assert zipped.status_code == 422


def test_api_rejects_unknown_selected_artifact_before_writing_cache(tmp_path):
    runs_root = tmp_path / "runs"
    write_evaluation_set(runs_root / "demo", run_id="demo")
    client = TestClient(create_app(run_roots=[("platform", runs_root)]))

    response = client.post(
        "/api/runs/platform/demo/download",
        json={"items": ["metrics/not_exposed.csv"]},
    )

    assert response.status_code == 422
    assert "未知下载项" in response.json()["detail"]
    assert not (tmp_path / "download_cache").exists()


def test_api_selected_artifact_zip_cache_changes_when_run_files_change(tmp_path):
    runs_root = tmp_path / "runs"
    run_dir = runs_root / "demo"
    write_evaluation_set(run_dir, run_id="demo")
    client = TestClient(create_app(run_roots=[("platform", runs_root)]))

    first = client.post(
        "/api/runs/platform/demo/download",
        json={"items": ["metadata_file:report"]},
    )
    assert first.status_code == 200
    with zipfile.ZipFile(BytesIO(first.content)) as archive:
        assert archive.read("report.md").replace(b"\r\n", b"\n") == b"# Report\n"

    (run_dir / "report.md").write_text("# Updated Report\n", encoding="utf-8")
    second = client.post(
        "/api/runs/platform/demo/download",
        json={"items": ["metadata_file:report"]},
    )

    assert second.status_code == 200
    with zipfile.ZipFile(BytesIO(second.content)) as archive:
        assert archive.read("report.md").replace(b"\r\n", b"\n") == (
            b"# Updated Report\n"
        )


def test_api_rejects_unsafe_selected_artifact_zip_entry_names(tmp_path):
    runs_root = tmp_path / "runs"
    run_dir = runs_root / "demo"
    write_evaluation_set(run_dir, run_id="demo")
    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["display"]["downloads"]["items"].append(
        {
            "item": "../evil.csv",
            "label": "Unsafe CSV",
            "kind": "table",
            "table": "ic",
        }
    )
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    client = TestClient(create_app(run_roots=[("platform", runs_root)]))

    response = client.post(
        "/api/runs/platform/demo/download",
        json={"items": ["../evil.csv"]},
    )

    assert response.status_code == 422
    assert "download refs require kind and key" in response.json()["detail"]


def test_api_rejects_unsafe_selected_artifact_zip_cache_identity(tmp_path):
    runs_root = tmp_path / "runs"
    run_dir = runs_root / "demo"
    write_evaluation_set(run_dir, run_id="demo")

    with pytest.raises(HTTPException) as exc_info:
        selected_download_response(
            load_evaluation_set(run_dir),
            "platform",
            "..",
            ["metadata_file:report"],
            tmp_path / "download_cache",
            lambda: b"",
        )

    assert getattr(exc_info.value, "status_code", None) == 422
    assert not (tmp_path / "download_cache" / "platform" / "..").exists()


def test_api_downloads_single_selected_artifact_as_zip(tmp_path):
    runs_root = tmp_path / "runs"
    write_evaluation_set(runs_root / "demo", run_id="demo")
    client = TestClient(create_app(run_roots=[("platform", runs_root)]))

    response = client.post(
        "/api/runs/platform/demo/download",
        json={"items": ["metadata_file:report"]},
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/zip")
    assert (
        'filename="platform_demo_selected_artifacts.zip"'
        in response.headers["content-disposition"]
    )
    with zipfile.ZipFile(BytesIO(response.content)) as archive:
        assert archive.namelist() == ["report.md"]
        assert (
            archive.read("report.md") == (runs_root / "demo" / "report.md").read_bytes()
        )


def test_api_downloads_selected_zip_without_full_artifact_load(
    tmp_path,
    monkeypatch,
):
    runs_root = tmp_path / "runs"
    write_evaluation_set(runs_root / "demo", run_id="demo")
    client = TestClient(create_app(run_roots=[("platform", runs_root)]))

    def fail_read_parquet(*_args, **_kwargs):
        raise AssertionError("selected ZIP should not full-load run tables")

    monkeypatch.setattr(pd, "read_parquet", fail_read_parquet)

    response = client.post(
        "/api/runs/platform/demo/download",
        json={"items": ["metadata_file:report"]},
    )

    assert response.status_code == 200
    with zipfile.ZipFile(BytesIO(response.content)) as archive:
        assert archive.namelist() == ["report.md"]


def test_api_selected_artifact_zip_cache_hashes_long_identity_segments(tmp_path):
    runs_root = tmp_path / "runs"
    run_id = "run_" + ("20240101_" * 12)
    source_alias = "source_" + ("private_acceptance_" * 5)
    write_evaluation_set(runs_root / "demo", run_id="demo")
    response = selected_download_response(
        load_evaluation_set(runs_root / "demo"),
        source_alias,
        run_id,
        ["metadata_file:report"],
        tmp_path / "download_cache",
        lambda: b"",
    )

    assert response.status_code == 200
    cache_files = list((tmp_path / "download_cache").glob("**/*.zip"))
    assert len(cache_files) == 1
    assert len(str(cache_files[0])) < 240


def test_api_returns_chinese_error_for_bad_uploaded_csv(tmp_path):
    runs_root = tmp_path / "runs"
    client = TestClient(create_app(run_roots=[("platform", runs_root)]))
    payload = {
        "run_id": "uploaded_demo",
        "project": "quant-research-platform",
        "kind": "factor",
        "domain": "user_upload",
        "data_stage": "uploaded_panel",
        "uses_external_data": True,
        "timestamp_column": "timestamp",
        "asset_column": "asset",
        "feature_columns": ["factor"],
        "label_columns": ["forward_return"],
        "n_groups": 2,
        "weight_method": "equal_weight_quantile",
        "direction": "long_high",
        "top_quantile": 0.5,
        "bottom_quantile": 0.5,
        "csv_text": "\n".join(
            [
                "timestamp,asset,factor",
                "2026-01-01,A,0.1",
                "2026-01-01,B,0.2",
            ]
        ),
    }

    response = client.post("/api/runs", json=payload)

    assert response.status_code == 422
    assert "CSV 缺少必需列" in response.json()["detail"]


def test_api_returns_chinese_error_for_missing_uploaded_return_source_column(tmp_path):
    runs_root = tmp_path / "runs"
    client = TestClient(create_app(run_roots=[("platform", runs_root)]))
    payload = {
        "run_id": "uploaded_demo",
        "project": "quant-research-platform",
        "kind": "factor",
        "domain": "user_upload",
        "data_stage": "uploaded_panel",
        "uses_external_data": True,
        "timestamp_column": "timestamp",
        "asset_column": "asset",
        "feature_columns": ["factor"],
        "label_columns": ["forward_return"],
        "n_groups": 2,
        "weight_method": "equal_weight_quantile",
        "direction": "long_high",
        "top_quantile": 0.5,
        "bottom_quantile": 0.5,
        "csv_text": "\n".join(
            [
                "timestamp,asset,factor,forward_return",
                "2026-01-01,A,0.1,0.02",
                "2026-01-01,B,0.2,0.01",
            ]
        ),
    }

    response = client.post("/api/runs", json=payload)

    assert response.status_code == 422
    assert "CSV 缺少收益来源列" in response.json()["detail"]


def test_api_returns_comparison_for_discovered_runs(tmp_path):
    runs_root = tmp_path / "runs"
    write_evaluation_set(runs_root / "run_a", run_id="run_a", net_returns=[0.01])
    write_evaluation_set(runs_root / "run_b", run_id="run_b", net_returns=[0.03])
    client = TestClient(create_app(run_roots=[("platform", runs_root)]))

    response = client.get("/api/comparison")

    assert response.status_code == 200
    payload = response.json()
    assert payload["columns"][0] == "source_alias"
    assert [row["public_run_id"] for row in payload["rows"]] == [
        "platform/run_a",
        "platform/run_b",
    ]
    assert "rank_mean_final_return" in payload["columns"]


def test_api_run_list_skips_run_with_missing_weight_method(tmp_path):
    runs_root = tmp_path / "runs"
    write_evaluation_set(runs_root / "good", run_id="good", net_returns=[0.03])
    run_dir = runs_root / "run_a"
    write_evaluation_set(run_dir, run_id="run_a", net_returns=[0.01])
    context_path = run_dir / "run_context.json"
    context = json.loads(context_path.read_text(encoding="utf-8"))
    context["config"]["evaluation_spec"]["weight_configs"] = [{}]
    context_path.write_text(json.dumps(context), encoding="utf-8")
    client = TestClient(create_app(run_roots=[("platform", runs_root)]))

    response = client.get("/api/runs")

    assert response.status_code == 200
    assert [run["public_run_id"] for run in response.json()] == ["platform/good"]


def test_api_comparison_skips_run_with_missing_weight_method(tmp_path):
    runs_root = tmp_path / "runs"
    write_evaluation_set(runs_root / "good", run_id="good", net_returns=[0.03])
    run_dir = runs_root / "run_a"
    write_evaluation_set(run_dir, run_id="run_a", net_returns=[0.01])
    context_path = run_dir / "run_context.json"
    context = json.loads(context_path.read_text(encoding="utf-8"))
    context["config"]["evaluation_spec"]["weight_configs"] = [{}]
    context_path.write_text(json.dumps(context), encoding="utf-8")
    client = TestClient(create_app(run_roots=[("platform", runs_root)]))

    response = client.get("/api/comparison")

    assert response.status_code == 200
    assert [row["public_run_id"] for row in response.json()["rows"]] == [
        "platform/good"
    ]


def test_api_downloads_dynamic_comparison_csv(tmp_path):
    runs_root = tmp_path / "runs"
    write_evaluation_set(runs_root / "run_a", run_id="run_a", net_returns=[0.01])
    write_evaluation_set(runs_root / "run_b", run_id="run_b", net_returns=[0.03])
    client = TestClient(create_app(run_roots=[("platform", runs_root)]))

    response = client.get("/api/comparison.csv")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/csv")
    assert 'filename="comparison.csv"' in response.headers["content-disposition"]
    assert "public_run_id" in response.text
    assert "platform/run_a" in response.text
    assert "platform/run_b" in response.text


def test_api_comparison_csv_includes_runs_without_portable_dynamic_item(tmp_path):
    runs_root = tmp_path / "runs"
    write_evaluation_set(runs_root / "good", run_id="good", net_returns=[0.03])
    run_dir = runs_root / "run_a"
    write_evaluation_set(run_dir, run_id="run_a", net_returns=[0.01])
    _remove_display_download_item(run_dir / "manifest.json", "comparison.csv")
    client = TestClient(create_app(run_roots=[("platform", runs_root)]))

    response = client.get("/api/comparison.csv")

    assert response.status_code == 200
    assert "platform/good" in response.text
    assert "platform/run_a" in response.text


def test_api_comparison_json_includes_runs_without_portable_dynamic_item(tmp_path):
    runs_root = tmp_path / "runs"
    write_evaluation_set(runs_root / "good", run_id="good", net_returns=[0.03])
    run_dir = runs_root / "run_a"
    write_evaluation_set(run_dir, run_id="run_a", net_returns=[0.01])
    _remove_display_download_item(run_dir / "manifest.json", "comparison.csv")
    client = TestClient(create_app(run_roots=[("platform", runs_root)]))

    response = client.get("/api/comparison")

    assert response.status_code == 200
    assert [row["public_run_id"] for row in response.json()["rows"]] == [
        "platform/good",
        "platform/run_a",
    ]


def test_api_evidence_injects_runtime_comparison_download_item(tmp_path):
    runs_root = tmp_path / "runs"
    run_dir = runs_root / "run_a"
    write_evaluation_set(run_dir, run_id="run_a", net_returns=[0.01])
    _remove_display_download_item(run_dir / "manifest.json", "comparison.csv")
    client = TestClient(create_app(run_roots=[("platform", runs_root)]))

    response = client.get("/api/runs/platform/run_a/evidence")

    assert response.status_code == 200
    assert {"kind": "dynamic", "key": "comparison"} in next(
        preset
        for preset in response.json()["display"]["downloads"]["presets"]
        if preset["key"] == "all"
    )["items"]


def test_api_selected_zip_dynamic_comparison_includes_valid_runs(tmp_path):
    runs_root = tmp_path / "runs"
    write_evaluation_set(runs_root / "run_a", run_id="run_a", net_returns=[0.01])
    run_b = runs_root / "run_b"
    write_evaluation_set(run_b, run_id="run_b", net_returns=[0.03])
    _remove_display_download_item(run_b / "manifest.json", "comparison.csv")
    client = TestClient(create_app(run_roots=[("platform", runs_root)]))

    response = client.post(
        "/api/runs/platform/run_a/download",
        json={"items": ["dynamic:comparison"]},
    )

    assert response.status_code == 200
    with zipfile.ZipFile(BytesIO(response.content)) as archive:
        comparison = archive.read("comparison.csv").decode("utf-8")
    assert "platform/run_a" in comparison
    assert "platform/run_b" in comparison


def test_api_comparison_keeps_duplicate_run_ids_from_different_sources(tmp_path):
    platform_root = tmp_path / "platform"
    crypto_root = tmp_path / "crypto"
    write_evaluation_set(platform_root / "demo", run_id="demo", net_returns=[0.01])
    write_evaluation_set(crypto_root / "demo", run_id="demo", net_returns=[0.03])
    client = TestClient(
        create_app(run_roots=[("platform", platform_root), ("crypto", crypto_root)])
    )

    response = client.get("/api/comparison")

    assert response.status_code == 200
    assert [row["public_run_id"] for row in response.json()["rows"]] == [
        "crypto/demo",
        "platform/demo",
    ]


def test_api_comparison_keeps_source_alias_after_run_id_sorting(tmp_path):
    platform_root = tmp_path / "platform"
    crypto_root = tmp_path / "crypto"
    write_evaluation_set(platform_root / "zeta", run_id="zeta", net_returns=[0.01])
    write_evaluation_set(crypto_root / "alpha", run_id="alpha", net_returns=[0.03])
    client = TestClient(
        create_app(run_roots=[("platform", platform_root), ("crypto", crypto_root)])
    )

    response = client.get("/api/comparison")

    assert response.status_code == 200
    rows = response.json()["rows"]
    assert [(row["public_run_id"], row["run_id"]) for row in rows] == [
        ("crypto/alpha", "alpha"),
        ("platform/zeta", "zeta"),
    ]


def test_api_returns_404_for_unknown_run(tmp_path):
    runs_root = tmp_path / "runs"
    write_evaluation_set(runs_root / "demo", run_id="demo")
    client = TestClient(create_app(run_roots=[("platform", runs_root)]))

    response = client.get("/api/runs/platform/missing/evidence")

    assert response.status_code == 404
    assert response.json()["detail"] == "run not found"


def test_api_allows_local_vite_origin(tmp_path):
    runs_root = tmp_path / "runs"
    write_evaluation_set(runs_root / "demo", run_id="demo")
    client = TestClient(create_app(run_roots=[("platform", runs_root)]))

    response = client.options(
        "/api/runs",
        headers={
            "Origin": "http://127.0.0.1:5173",
            "Access-Control-Request-Method": "GET",
        },
    )

    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == ("http://127.0.0.1:5173")

    download_response = client.get(
        "/api/comparison.csv",
        headers={"Origin": "http://127.0.0.1:5173"},
    )

    assert download_response.status_code == 200
    assert (
        download_response.headers["access-control-expose-headers"]
        == "Content-Disposition"
    )


def test_api_run_list_skips_invalid_artifact_contract(tmp_path):
    runs_root = tmp_path / "runs"
    write_evaluation_set(runs_root / "good", run_id="good")
    write_evaluation_set(runs_root / "broken", run_id="broken")
    summary_path = runs_root / "broken" / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary.pop("n_combinations")
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    client = TestClient(create_app(run_roots=[("platform", runs_root)]))

    response = client.get("/api/runs")

    assert response.status_code == 200
    assert [run["public_run_id"] for run in response.json()] == ["platform/good"]

    target = client.get("/api/runs/platform/broken/evidence")
    assert target.status_code == 422
    assert "summary.json is missing required fields" in target.json()["detail"]


def _remove_display_download_item(manifest_path, item_name: str) -> None:
    legacy_ref_map = {
        "manifest.json": {"kind": "metadata_file", "key": "manifest"},
        "summary.json": {"kind": "metadata_file", "key": "summary"},
        "run_context.json": {"kind": "metadata_file", "key": "run_context"},
        "report.md": {"kind": "metadata_file", "key": "report"},
        "comparison.csv": {"kind": "dynamic", "key": "comparison"},
    }
    if ":" in item_name:
        kind, key = item_name.split(":", 1)
        target_ref = {"kind": kind, "key": key}
    else:
        target_ref = legacy_ref_map.get(item_name, {"kind": "table", "key": item_name})
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    downloads = manifest["display"]["downloads"]
    downloads["items"] = [
        item
        for item in downloads["items"]
        if not (
            item.get("kind") == target_ref["kind"]
            and item.get("key") == target_ref["key"]
        )
    ]
    for preset in downloads["presets"]:
        preset["items"] = [
            item
            for item in preset["items"]
            if not (
                isinstance(item, dict)
                and item.get("kind") == target_ref["kind"]
                and item.get("key") == target_ref["key"]
            )
        ]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
