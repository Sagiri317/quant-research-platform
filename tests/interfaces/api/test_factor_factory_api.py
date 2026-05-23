from __future__ import annotations

import json

from fastapi.testclient import TestClient

from quant_research_platform.interfaces.api.app import create_app
from quant_research_platform.research.factor_factory.candidate import (
    CandidateEvaluationRecord,
)
from quant_research_platform.research.factor_factory.reports import (
    write_candidate_reports,
)


def _write_factor_factory_run(path) -> None:
    write_candidate_reports(
        path,
        (
            CandidateEvaluationRecord(
                candidate_id="reversal_20d@1",
                version="1",
                expression="neg(return(close, 20))",
                dependencies=("close",),
                output_column="rev_20d",
                sample_start="2024-01-01",
                sample_end="2024-12-31",
                label_columns=("forward_return_1d",),
                inputs=("close",),
                outputs=("rev_20d",),
                capabilities=("processed_existing", "evaluate"),
                parameters={"window": 20},
                availability_rule="trade_date_close",
                time_semantics={"label_rule": "forward_return_1d"},
                status="success",
                run_dir="reports/private_real/pipeline_runs/factory_run",
            ),
        ),
        overwrite=True,
    )


def test_factor_factory_runs_are_browsable_from_api(tmp_path):
    factory_root = tmp_path / "factor_factory_runs"
    _write_factor_factory_run(factory_root / "factory_demo")
    client = TestClient(
        create_app(
            run_roots=[("platform", tmp_path / "runs")],
            factor_factory_root=factory_root,
        )
    )

    listed = client.get("/api/factor-factory-runs")
    detail = client.get("/api/factor-factory-runs/factory_demo")
    table = client.get("/api/factor-factory-runs/factory_demo/tables/candidate_summary")
    artifact = client.get(
        "/api/factor-factory-runs/factory_demo/artifacts/candidate_summary.csv"
    )

    assert listed.status_code == 200
    assert listed.json()[0]["run_id"] == "factory_demo"
    assert listed.json()[0]["artifact_type"] == "factor_factory_run"
    assert detail.status_code == 200
    assert detail.json()["display"]["tables"][0]["key"] == "candidate_summary"
    assert table.status_code == 200
    assert table.json()["rows"][0]["candidate_id"] == "reversal_20d@1"
    assert "run_dir" not in table.json()["columns"]
    assert table.json()["rows"][0]["run_ref"] == "factory_run"
    assert artifact.status_code == 200
    assert "reversal_20d@1" in artifact.text
    assert "reports/private_real" not in listed.text
    assert "reports/private_real" not in detail.text
    assert "reports/private_real" not in table.text
    assert "reports/private_real" not in artifact.text


def test_factor_factory_api_rejects_unexposed_artifact_item(tmp_path):
    factory_root = tmp_path / "factor_factory_runs"
    _write_factor_factory_run(factory_root / "factory_demo")
    client = TestClient(
        create_app(
            run_roots=[("platform", tmp_path / "runs")],
            factor_factory_root=factory_root,
        )
    )

    response = client.get("/api/factor-factory-runs/factory_demo/artifacts/hidden.csv")

    assert response.status_code == 422


def test_factor_factory_api_rejects_unsafe_display_paths(tmp_path):
    factory_root = tmp_path / "factor_factory_runs"
    run_dir = factory_root / "factory_demo"
    _write_factor_factory_run(run_dir)
    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["display"]["tables"][0]["path"] = "../private_real.csv"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    client = TestClient(
        create_app(
            run_roots=[("platform", tmp_path / "runs")],
            factor_factory_root=factory_root,
        )
    )

    response = client.get("/api/factor-factory-runs/factory_demo")

    assert response.status_code == 422
    assert "private_real.csv" not in response.text


def test_factor_factory_api_rejects_private_display_text(tmp_path):
    factory_root = tmp_path / "factor_factory_runs"
    run_dir = factory_root / "factory_demo"
    _write_factor_factory_run(run_dir)
    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["display"]["tables"][0]["label"] = "reports/private_real/factory_run"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    client = TestClient(
        create_app(
            run_roots=[("platform", tmp_path / "runs")],
            factor_factory_root=factory_root,
        )
    )

    response = client.get("/api/factor-factory-runs/factory_demo")

    assert response.status_code == 422
    assert "reports/private_real" not in response.text
