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
    assert runs[0]["evaluation_id"] == "toy"
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


def test_discover_runs_exposes_pipeline_evaluation_runs(tmp_path):
    pipeline_root = tmp_path / "pipeline_runs"
    nested_run = (
        pipeline_root / "pipeline_run" / "evaluation" / "evaluation_runs" / "demo"
    )
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
        / "evaluation_runs"
        / "demo"
    )
    write_evaluation_set(pipeline_run, run_id="demo")
    client = TestClient(create_app(run_roots=[("platform", platform_root)]))

    response = client.get("/api/runs")

    assert response.status_code == 200
    assert response.json()[0]["public_run_id"] == "pipeline/demo"


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
    ic = client.get("/api/runs/platform/demo/metrics/ic")
    pnl = client.get("/api/runs/platform/demo/portfolio/pnl")
    performance = client.get("/api/runs/platform/demo/metrics/performance_summary")
    retired_performance = client.get(
        "/api/runs/platform/demo/metrics/performance-summary"
    )
    report = client.get("/api/runs/platform/demo/report.md")

    assert evidence.status_code == 200
    assert evidence.json()["summary"]["evaluation_id"] == "demo"
    assert evidence.json()["summary"]["run_id"] == "demo"
    assert evidence.json()["summary"]["source_alias"] == "platform"
    assert evidence.json()["summary"]["public_run_id"] == "platform/demo"
    assert evidence.json()["summary"]["domain"] == "evaluation_synthetic"
    assert ic.status_code == 200
    assert {"combination_id", "timestamp", "feature", "label", "ic"}.issubset(
        ic.json()["columns"]
    )
    assert len(ic.json()["rows"]) == 2
    assert pnl.status_code == 200
    assert "cumulative_pnl" in pnl.json()["columns"]
    assert performance.status_code == 200
    assert "max_drawdown" in performance.json()["columns"]
    assert retired_performance.status_code == 404
    assert report.status_code == 200
    assert report.text == "# Report\n"


def test_api_metric_preview_filters_combination_before_limit(tmp_path):
    runs_root = tmp_path / "runs"
    run_dir = runs_root / "demo"
    write_evaluation_set(run_dir, run_id="demo")
    pd.DataFrame(
        [
            {"combination_id": "combo_a", "timestamp": "2024-01-01", "ic": 0.1},
            {"combination_id": "combo_b", "timestamp": "2024-01-02", "ic": 0.2},
            {"combination_id": "combo_b", "timestamp": "2024-01-03", "ic": 0.3},
        ]
    ).assign(
        feature="factor",
        label="forward_return",
        weight_id="w0",
        cost_id="c0",
    ).to_csv(run_dir / "metrics" / "ic.csv", index=False)
    client = TestClient(create_app(run_roots=[("platform", runs_root)]))

    response = client.get(
        "/api/runs/platform/demo/metrics/ic?combination_id=combo_b&limit=1"
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["total_rows"] == 2
    assert payload["rows"] == [
        {
            "combination_id": "combo_b",
            "timestamp": "2024-01-02",
            "ic": 0.2,
            "feature": "factor",
            "label": "forward_return",
            "weight_id": "w0",
            "cost_id": "c0",
        }
    ]


def test_api_metric_endpoint_loads_only_selected_run_artifacts(tmp_path, monkeypatch):
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

    response = client.get("/api/runs/platform/run_b/metrics/ic")

    assert response.status_code == 200
    assert loaded_run_dirs == ["run_b"]


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

    response = client.get("/api/runs/platform/good/metrics/ic")

    assert response.status_code == 200
    assert response.json()["rows"]


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


def test_api_metric_endpoint_returns_limited_preview_rows(tmp_path):
    runs_root = tmp_path / "runs"
    write_evaluation_set(
        runs_root / "demo",
        run_id="demo",
        ic_values=[0.1, 0.2, 0.3],
        rankic_values=[0.3, 0.4, 0.5],
    )
    client = TestClient(create_app(run_roots=[("platform", runs_root)]))

    response = client.get("/api/runs/platform/demo/metrics/ic?limit=2")

    assert response.status_code == 200
    payload = response.json()
    assert len(payload["rows"]) == 2
    assert payload["total_rows"] == 3
    assert payload["limit"] == 2


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
    assert payload["summary"]["evaluation_id"] == "demo"
    assert payload["summary"]["run_id"] == "demo"
    assert payload["summary"]["source_alias"] == "platform"
    assert payload["summary"]["public_run_id"] == "platform/demo"
    assert payload["report"]["markdown"] == "# Report\n"
    assert {"combination_id", "timestamp", "feature", "label", "ic"}.issubset(
        payload["tables"]["ic"]["columns"]
    )
    assert len(payload["tables"]["ic"]["rows"]) == 2
    assert "pnl" in payload["tables"]
    assert "returns" in payload["tables"]
    assert "performance_summary" not in payload["tables"]
    combination_summary = payload["combination_summaries"]["rows"][0]
    for key, value in {
        "final_cumulative_net_return": 0.029,
        "sharpe_like": 0.0,
        "sortino_like": 0.0,
        "max_drawdown": 0.0,
        "win_rate": 1.0,
        "mean_ic": 0.15,
        "icir": 3.0,
        "mean_rankic": 0.35,
        "rankicir": 7.0,
        "mean_turnover": 2.0,
        "total_cost": 0.001,
    }.items():
        assert combination_summary[key] == pytest.approx(value)
    assert {
        key: combination_summary[key]
        for key in (
            "combination_id",
            "feature",
            "label",
            "weight_id",
            "cost_id",
            "weight_method",
            "direction",
            "portfolio_mode",
        )
    } == {
        "combination_id": "factor__forward_return__w0__c0",
        "feature": "factor",
        "label": "forward_return",
        "weight_id": "w0",
        "cost_id": "c0",
        "weight_method": "equal_weight_quantile",
        "direction": "long_high",
        "portfolio_mode": "long_short",
    }
    assert combination_summary["cost_bps"] == pytest.approx(0.0)
    display = payload["display"]
    assert display["schema_version"] == "0.5"
    assert {"combinations", "ic", "returns", "pnl"}.issubset(
        {table["key"] for table in display["tables"]}
    )
    assert [chart["id"] for chart in display["charts"]] == [
        "pnl",
        "ic_rankic",
        "net_portfolio_return",
        "turnover",
        "cost",
    ]
    all_download = next(
        preset
        for preset in display["downloads"]["presets"]
        if preset["key"] == "all"
    )
    assert "portfolio/pnl.csv" in all_download["items"]
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

    preview = client.get("/api/runs/crypto/demo/metrics/ic")
    direct = client.get("/api/runs/crypto/demo/artifacts/metrics/ic.csv")
    zipped = client.post(
        "/api/runs/crypto/demo/download",
        json={"items": ["metrics/ic.csv"]},
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
    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["display"]["charts"].append(
        {
            "id": "cost_breakdown",
            "title": "Cost breakdown",
            "description": "Cost diagnostics",
            "kind": "line",
            "table": "cost_breakdown",
            "x": "timestamp",
            "series": [{"key": "total_cost", "color": "#b42318"}],
        }
    )
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    client = TestClient(create_app(run_roots=[("platform", runs_root)]))

    response = client.get("/api/runs/platform/demo/evidence")

    assert response.status_code == 200
    payload = response.json()
    assert "cost_breakdown" in payload["tables"]
    assert "total_cost" in payload["tables"]["cost_breakdown"]["columns"]


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
        "pnl_return_column": "forward_return_1d",
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
    assert (runs_root / "uploaded_demo" / "portfolio" / "pnl.csv").is_file()
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
        "pnl_return_column": "forward_return_1d",
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
        "pnl_return_column": "forward_return_1d",
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
        "/api/runs/platform/demo/artifacts/metrics/performance_summary.csv"
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/csv")
    assert "attachment" in response.headers["content-disposition"]
    assert 'filename="performance_summary.csv"' in response.headers[
        "content-disposition"
    ]
    assert "final_cumulative_net_return" in response.text
    assert "total_net_portfolio_return" not in response.text


@pytest.mark.parametrize(
    ("item", "expected"),
    [
        ("manifest.json", "artifact_schema_version"),
        ("config.json", "metadata"),
        ("summary.json", "evaluation_id"),
    ],
)
def test_api_downloads_display_exposed_metadata_artifacts(tmp_path, item, expected):
    runs_root = tmp_path / "runs"
    write_evaluation_set(runs_root / "demo", run_id="demo")
    client = TestClient(create_app(run_roots=[("platform", runs_root)]))

    response = client.get(f"/api/runs/platform/demo/artifacts/{item}")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/json")
    assert f'filename="{item}"' in response.headers["content-disposition"]
    assert expected in response.json()


def test_api_direct_metadata_artifact_must_be_exposed_by_display(tmp_path):
    runs_root = tmp_path / "runs"
    run_dir = runs_root / "demo"
    write_evaluation_set(run_dir, run_id="demo")
    _remove_display_download_item(run_dir / "manifest.json", "summary.json")
    client = TestClient(create_app(run_roots=[("platform", runs_root)]))

    response = client.get("/api/runs/platform/demo/artifacts/summary.json")

    assert response.status_code == 422
    assert "summary.json is not exposed" in response.json()["detail"]


def test_api_direct_metadata_download_uses_canonical_file_path(tmp_path):
    runs_root = tmp_path / "runs"
    run_dir = runs_root / "demo"
    write_evaluation_set(run_dir, run_id="demo")
    (run_dir / "other_summary.json").write_text(
        json.dumps({"evaluation_id": "wrong"}),
        encoding="utf-8",
    )
    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifacts"]["summary"] = "other_summary.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    client = TestClient(create_app(run_roots=[("platform", runs_root)]))

    response = client.get("/api/runs/platform/demo/artifacts/summary.json")

    assert response.status_code == 200
    assert response.json()["evaluation_id"] == "demo"


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
        "/api/runs/platform/demo/artifacts/metrics/performance_summary.csv"
    )

    assert response.status_code == 200
    assert response.content == (
        runs_root / "demo" / "metrics" / "performance_summary.csv"
    ).read_bytes()


def test_api_direct_artifact_csv_must_be_exposed_by_display(tmp_path):
    runs_root = tmp_path / "runs"
    run_dir = runs_root / "demo"
    write_evaluation_set(run_dir, run_id="demo")
    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["display"]["downloads"]["items"] = [
        item
        for item in manifest["display"]["downloads"]["items"]
        if item.get("item") != "metrics/performance_summary.csv"
    ]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    client = TestClient(create_app(run_roots=[("platform", runs_root)]))

    response = client.get(
        "/api/runs/platform/demo/artifacts/metrics/performance_summary.csv"
    )

    assert response.status_code == 422
    detail = response.json()["detail"]
    assert "display.downloads.items must include every display table" in detail


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
        "/api/runs/platform/demo/artifacts/metrics/performance_summary.csv"
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
    preview = client.get("/api/runs/platform/demo/metrics/ic")
    zip_post = client.post(
        "/api/runs/platform/demo/download",
        json={"items": ["report.md"]},
    )
    zip_get = client.get("/api/runs/platform/demo/download?items=report.md")

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
        "/api/runs/platform/demo/breakdowns/metrics_by_month/"
        "performance_summary_by_year"
    )

    assert response.status_code == 404
    assert response.json()["detail"] == "breakdown table not found"


def test_api_preview_routes_reject_wrong_display_group(tmp_path):
    runs_root = tmp_path / "runs"
    write_evaluation_set(runs_root / "demo", run_id="demo")
    client = TestClient(create_app(run_roots=[("platform", runs_root)]))

    metrics_for_portfolio = client.get("/api/runs/platform/demo/metrics/pnl")
    portfolio_for_metric = client.get("/api/runs/platform/demo/portfolio/ic")
    diagnostics_for_portfolio = client.get("/api/runs/platform/demo/diagnostics/pnl")

    assert metrics_for_portfolio.status_code == 404
    assert portfolio_for_metric.status_code == 404
    assert diagnostics_for_portfolio.status_code == 404


def test_api_downloads_cost_breakdown_artifact_csv(tmp_path):
    runs_root = tmp_path / "runs"
    write_evaluation_set(runs_root / "demo", run_id="demo")
    client = TestClient(create_app(run_roots=[("platform", runs_root)]))

    response = client.get(
        "/api/runs/platform/demo/artifacts/portfolio/cost_breakdown.csv"
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/csv")
    assert 'filename="cost_breakdown.csv"' in response.headers[
        "content-disposition"
    ]
    assert "buy_turnover" in response.text
    assert "total_cost" in response.text


def test_api_downloads_neutralization_diagnostics_artifact_csv(tmp_path):
    runs_root = tmp_path / "runs"
    write_evaluation_set(runs_root / "demo", run_id="demo")
    client = TestClient(create_app(run_roots=[("platform", runs_root)]))

    response = client.get(
        "/api/runs/platform/demo/artifacts/diagnostics/neutralization_diagnostics.csv"
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/csv")
    assert 'filename="neutralization_diagnostics.csv"' in response.headers[
        "content-disposition"
    ]
    assert "method" in response.text


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


def test_api_report_markdown_must_be_exposed_by_display_downloads(tmp_path):
    runs_root = tmp_path / "runs"
    run_dir = runs_root / "demo"
    write_evaluation_set(run_dir, run_id="demo")
    _remove_display_download_item(run_dir / "manifest.json", "report.md")
    client = TestClient(create_app(run_roots=[("platform", runs_root)]))

    response = client.get("/api/runs/platform/demo/report.md")

    assert response.status_code == 404


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
    assert "report.md is not exposed" in response.json()["detail"]


def test_api_downloads_selected_artifacts_as_zip(tmp_path):
    runs_root = tmp_path / "runs"
    write_evaluation_set(runs_root / "demo", run_id="demo")
    write_evaluation_set(runs_root / "other", run_id="other")
    client = TestClient(create_app(run_roots=[("platform", runs_root)]))

    response = client.post(
        "/api/runs/platform/demo/download",
        json={
            "items": [
                "comparison.csv",
                "report.md",
                "summary.json",
                "metrics/performance_summary.csv",
                "portfolio/cost_breakdown.csv",
            ]
        },
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/zip")
    assert 'filename="platform_demo_selected_artifacts.zip"' in response.headers[
        "content-disposition"
    ]
    with zipfile.ZipFile(BytesIO(response.content)) as archive:
        assert set(archive.namelist()) == {
            "comparison.csv",
            "report.md",
            "summary.json",
            "metrics/performance_summary.csv",
            "portfolio/cost_breakdown.csv",
        }
        assert b"public_run_id" in archive.read("comparison.csv")
        assert b"platform/demo" in archive.read("comparison.csv")
        assert b"platform/other" in archive.read("comparison.csv")
        assert archive.read("report.md") == (
            runs_root / "demo" / "report.md"
        ).read_bytes()
        assert b"final_cumulative_net_return" in archive.read(
            "metrics/performance_summary.csv"
        )
        assert b"buy_turnover" in archive.read("portfolio/cost_breakdown.csv")


def test_api_downloads_selected_artifacts_from_query_items(tmp_path):
    runs_root = tmp_path / "runs"
    write_evaluation_set(runs_root / "demo", run_id="demo")
    client = TestClient(create_app(run_roots=[("platform", runs_root)]))

    response = client.get(
        "/api/runs/platform/demo/download",
        params=[
            ("items", "report.md"),
            ("items", "summary.json"),
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
        json.dumps({"evaluation_id": "wrong"}),
        encoding="utf-8",
    )
    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifacts"]["summary"] = "evil_summary.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    client = TestClient(create_app(run_roots=[("platform", runs_root)]))

    response = client.post(
        "/api/runs/platform/demo/download",
        json={"items": ["summary.json"]},
    )

    assert response.status_code == 200
    with zipfile.ZipFile(BytesIO(response.content)) as archive:
        assert json.loads(archive.read("summary.json"))["evaluation_id"] == "demo"


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
                    "report.md",
                    "summary.json",
                    "metrics/performance_summary.csv",
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
        json={"items": ["report.md"]},
    )
    assert first.status_code == 200
    with zipfile.ZipFile(BytesIO(first.content)) as archive:
        assert archive.read("report.md").replace(b"\r\n", b"\n") == b"# Report\n"

    (run_dir / "report.md").write_text("# Updated Report\n", encoding="utf-8")
    second = client.post(
        "/api/runs/platform/demo/download",
        json={"items": ["report.md"]},
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
    assert "路径" in response.json()["detail"]


def test_api_rejects_unsafe_selected_artifact_zip_cache_identity(tmp_path):
    runs_root = tmp_path / "runs"
    run_dir = runs_root / "demo"
    write_evaluation_set(run_dir, run_id="demo")

    with pytest.raises(HTTPException) as exc_info:
        selected_download_response(
            load_evaluation_set(run_dir),
            "platform",
            "..",
            ["report.md"],
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
        json={"items": ["report.md"]},
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/zip")
    assert 'filename="platform_demo_selected_artifacts.zip"' in response.headers[
        "content-disposition"
    ]
    with zipfile.ZipFile(BytesIO(response.content)) as archive:
        assert archive.namelist() == ["report.md"]
        assert archive.read("report.md") == (
            runs_root / "demo" / "report.md"
        ).read_bytes()


def test_api_selected_artifact_zip_cache_hashes_long_identity_segments(tmp_path):
    runs_root = tmp_path / "runs"
    run_id = "run_" + ("20240101_" * 12)
    source_alias = "source_" + ("private_acceptance_" * 5)
    write_evaluation_set(runs_root / "demo", run_id="demo")
    response = selected_download_response(
        load_evaluation_set(runs_root / "demo"),
        source_alias,
        run_id,
        ["report.md"],
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


def test_api_returns_chinese_error_for_missing_uploaded_pnl_column(tmp_path):
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
    assert "CSV 缺少 PnL 收益列" in response.json()["detail"]


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
    summary_path = run_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["weight_configs"] = [{}]
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    client = TestClient(create_app(run_roots=[("platform", runs_root)]))

    response = client.get("/api/runs")

    assert response.status_code == 200
    assert [run["public_run_id"] for run in response.json()] == ["platform/good"]


def test_api_comparison_skips_run_with_missing_weight_method(tmp_path):
    runs_root = tmp_path / "runs"
    write_evaluation_set(runs_root / "good", run_id="good", net_returns=[0.03])
    run_dir = runs_root / "run_a"
    write_evaluation_set(run_dir, run_id="run_a", net_returns=[0.01])
    summary_path = run_dir / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["weight_configs"] = [{}]
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
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


def test_api_comparison_csv_skips_run_without_display_download_item(tmp_path):
    runs_root = tmp_path / "runs"
    write_evaluation_set(runs_root / "good", run_id="good", net_returns=[0.03])
    run_dir = runs_root / "run_a"
    write_evaluation_set(run_dir, run_id="run_a", net_returns=[0.01])
    _remove_display_download_item(run_dir / "manifest.json", "comparison.csv")
    client = TestClient(create_app(run_roots=[("platform", runs_root)]))

    response = client.get("/api/comparison.csv")

    assert response.status_code == 200
    assert "platform/good" in response.text
    assert "platform/run_a" not in response.text


def test_api_comparison_json_skips_run_without_display_download_item(tmp_path):
    runs_root = tmp_path / "runs"
    write_evaluation_set(runs_root / "good", run_id="good", net_returns=[0.03])
    run_dir = runs_root / "run_a"
    write_evaluation_set(run_dir, run_id="run_a", net_returns=[0.01])
    _remove_display_download_item(run_dir / "manifest.json", "comparison.csv")
    client = TestClient(create_app(run_roots=[("platform", runs_root)]))

    response = client.get("/api/comparison")

    assert response.status_code == 200
    assert [row["public_run_id"] for row in response.json()["rows"]] == [
        "platform/good"
    ]


def test_api_evidence_comparison_requires_display_download_item(tmp_path):
    runs_root = tmp_path / "runs"
    run_dir = runs_root / "run_a"
    write_evaluation_set(run_dir, run_id="run_a", net_returns=[0.01])
    _remove_display_download_item(run_dir / "manifest.json", "comparison.csv")
    client = TestClient(create_app(run_roots=[("platform", runs_root)]))

    response = client.get("/api/runs/platform/run_a/evidence")

    assert response.status_code == 422
    assert "comparison.csv is not exposed" in response.json()["detail"]


def test_api_selected_zip_comparison_csv_skips_unexposed_comparison_runs(tmp_path):
    runs_root = tmp_path / "runs"
    write_evaluation_set(runs_root / "run_a", run_id="run_a", net_returns=[0.01])
    run_b = runs_root / "run_b"
    write_evaluation_set(run_b, run_id="run_b", net_returns=[0.03])
    _remove_display_download_item(run_b / "manifest.json", "comparison.csv")
    client = TestClient(create_app(run_roots=[("platform", runs_root)]))

    response = client.post(
        "/api/runs/platform/run_a/download",
        json={"items": ["comparison.csv"]},
    )

    assert response.status_code == 200
    with zipfile.ZipFile(BytesIO(response.content)) as archive:
        comparison = archive.read("comparison.csv").decode("utf-8")
    assert "platform/run_a" in comparison
    assert "platform/run_b" not in comparison


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
    assert response.headers["access-control-allow-origin"] == (
        "http://127.0.0.1:5173"
    )

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
    summary.pop("domain")
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    client = TestClient(create_app(run_roots=[("platform", runs_root)]))

    response = client.get("/api/runs")

    assert response.status_code == 200
    assert [run["public_run_id"] for run in response.json()] == ["platform/good"]

    target = client.get("/api/runs/platform/broken/evidence")
    assert target.status_code == 422
    assert "summary.json is missing required fields" in target.json()["detail"]


def _remove_display_download_item(manifest_path, item_name: str) -> None:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    downloads = manifest["display"]["downloads"]
    downloads["items"] = [
        item for item in downloads["items"] if item.get("item") != item_name
    ]
    for preset in downloads["presets"]:
        preset["items"] = [item for item in preset["items"] if item != item_name]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
