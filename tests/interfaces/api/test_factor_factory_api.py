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


def _candidate_record(
    candidate_id: str = "reversal_20d@1",
    *,
    status: str = "success",
    failure_reason: str | None = None,
) -> CandidateEvaluationRecord:
    output_column = candidate_id.replace("@", "_").replace("-", "_")
    return CandidateEvaluationRecord(
        candidate_id=candidate_id,
        version="1",
        expression="neg(return(close, 20))",
        dependencies=("close",),
        output_column=output_column,
        sample_start="2024-01-01",
        sample_end="2024-12-31",
        label_columns=("forward_return_1d",),
        inputs=("close",),
        outputs=(output_column,),
        capabilities=("existing_factors", "evaluate"),
        parameters={"window": 20},
        availability_rule="trade_date_close",
        time_semantics={"label_rule": "forward_return_1d"},
        status=status,
        run_dir="reports/private_real/pipeline_runs/factory_run",
        failure_reason=failure_reason,
        failure_code="factory_failed" if status == "failed" else None,
    )


def _write_factor_factory_run(path, record_count: int = 1) -> None:
    write_candidate_reports(
        path,
        tuple(
            _candidate_record(f"reversal_20d@{index}")
            for index in range(1, record_count + 1)
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
    assert set(detail.json()) == {"display", "run"}
    assert detail.json()["run"] == listed.json()[0]
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


def test_factor_factory_table_preview_uses_limit_and_offset(tmp_path):
    factory_root = tmp_path / "factor_factory_runs"
    _write_factor_factory_run(factory_root / "factory_demo", record_count=5)
    client = TestClient(
        create_app(
            run_roots=[("platform", tmp_path / "runs")],
            factor_factory_root=factory_root,
        )
    )

    response = client.get(
        "/api/factor-factory-runs/factory_demo/tables/candidate_summary"
        "?limit=2&offset=2"
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["offset"] == 2
    assert payload["limit"] == 2
    assert payload["has_more"] is True
    assert [row["candidate_id"] for row in payload["rows"]] == [
        "reversal_20d@3",
        "reversal_20d@4",
    ]


def test_factor_factory_table_preview_uses_bounded_csv_read(tmp_path, monkeypatch):
    import quant_research_platform.interfaces.api.routes.factor_factory as route_module

    factory_root = tmp_path / "factor_factory_runs"
    _write_factor_factory_run(factory_root / "factory_demo", record_count=5)
    read_csv_calls: list[dict[str, object]] = []
    real_read_csv = route_module.pd.read_csv

    def spy_read_csv(*args, **kwargs):
        read_csv_calls.append(kwargs)
        return real_read_csv(*args, **kwargs)

    monkeypatch.setattr(route_module.pd, "read_csv", spy_read_csv)
    client = TestClient(
        create_app(
            run_roots=[("platform", tmp_path / "runs")],
            factor_factory_root=factory_root,
        )
    )

    response = client.get(
        "/api/factor-factory-runs/factory_demo/tables/candidate_summary"
        "?limit=2&offset=2"
    )

    assert response.status_code == 200
    assert len(read_csv_calls) == 1
    assert read_csv_calls[0]["nrows"] == 3
    assert callable(read_csv_calls[0]["skiprows"])


def test_factor_factory_root_can_be_configured_from_environment(tmp_path, monkeypatch):
    factory_root = tmp_path / "factor_factory_runs"
    _write_factor_factory_run(factory_root / "factory_demo")
    monkeypatch.setenv("QRP_FACTOR_FACTORY_ROOT", str(factory_root))
    client = TestClient(create_app(run_roots=[("platform", tmp_path / "runs")]))

    response = client.get("/api/factor-factory-runs")

    assert response.status_code == 200
    assert response.json()[0]["run_id"] == "factory_demo"


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


def test_factor_factory_list_skips_single_invalid_run(tmp_path):
    factory_root = tmp_path / "factor_factory_runs"
    _write_factor_factory_run(factory_root / "good_run")
    bad_run = factory_root / "bad_run"
    bad_run.mkdir(parents=True)
    (bad_run / "manifest.json").write_text("{not-json", encoding="utf-8")
    client = TestClient(
        create_app(
            run_roots=[("platform", tmp_path / "runs")],
            factor_factory_root=factory_root,
        )
    )

    response = client.get("/api/factor-factory-runs")

    assert response.status_code == 200
    assert [run["run_id"] for run in response.json()] == ["good_run"]


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


def test_factor_factory_api_rejects_private_display_text_case_insensitive(tmp_path):
    factory_root = tmp_path / "factor_factory_runs"
    run_dir = factory_root / "factory_demo"
    _write_factor_factory_run(run_dir)
    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["display"]["tables"][0]["label"] = "C:\\DATA\\RAW_PRIVATE\\CSMAR"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    client = TestClient(
        create_app(
            run_roots=[("platform", tmp_path / "runs")],
            factor_factory_root=factory_root,
        )
    )

    response = client.get("/api/factor-factory-runs/factory_demo")

    assert response.status_code == 422
    assert "RAW_PRIVATE" not in response.text


def test_factor_factory_record_sanitizes_private_failure_reason_case_insensitive():
    record = _candidate_record(
        "broken@1",
        status="failed",
        failure_reason="Failed at C:\\DATA\\RAW_PRIVATE\\CSMAR\\input.csv",
    )

    row = record.to_row()

    assert "[private-path]" in row["failure_reason"]
    assert "RAW_PRIVATE" not in row["failure_reason"]


def test_factor_factory_record_sanitizes_local_absolute_failure_reason():
    record = _candidate_record(
        "broken@1",
        status="failed",
        failure_reason="Failed(path=C:\\Users\\DELL\\vendor\\input.csv)",
    )

    row = record.to_row()

    assert "[local-path]" in row["failure_reason"]
    assert "C:/Users" not in row["failure_reason"]


def test_factor_factory_api_rejects_private_markers_in_preview_and_download(tmp_path):
    factory_root = tmp_path / "factor_factory_runs"
    run_dir = factory_root / "factory_demo"
    _write_factor_factory_run(run_dir)
    summary_path = run_dir / "candidate_summary.csv"
    summary_text = summary_path.read_text(encoding="utf-8")
    summary_path.write_text(
        summary_text.replace("factory_run", "C:\\DATA\\RAW_PRIVATE\\CSMAR\\input.csv"),
        encoding="utf-8",
    )
    client = TestClient(
        create_app(
            run_roots=[("platform", tmp_path / "runs")],
            factor_factory_root=factory_root,
        )
    )

    table = client.get("/api/factor-factory-runs/factory_demo/tables/candidate_summary")
    artifact = client.get(
        "/api/factor-factory-runs/factory_demo/artifacts/candidate_summary.csv"
    )

    assert table.status_code == 422
    assert artifact.status_code == 422
    assert "RAW_PRIVATE" not in table.text
    assert "RAW_PRIVATE" not in artifact.text
