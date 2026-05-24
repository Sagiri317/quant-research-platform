from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from quant_research_platform.interfaces.api.app import create_app


def test_api_markets_endpoint_matches_web_contract() -> None:
    client = TestClient(create_app(run_roots=[("platform", "reports/runs")]))

    response = client.get("/api/markets")

    assert response.status_code == 200
    markets = response.json()
    assert isinstance(markets, list)
    assert [market["market"] for market in markets] == ["a_share", "crypto"]
    assert markets[0]["status"] == "enabled"
    assert markets[1]["status"] == "planned"


def test_factor_templates_are_loaded_from_config() -> None:
    client = TestClient(create_app(run_roots=[("platform", "reports/runs")]))

    response = client.get("/api/factor-templates?market=a_share")

    assert response.status_code == 200
    templates = response.json()
    assert [template["id"] for template in templates] == [
        "liquidity_20d",
        "momentum_60d",
        "reversal_20d",
    ]
    reversal = next(
        template for template in templates if template["id"] == "reversal_20d"
    )
    assert reversal["expression"] == "neg(return(close, 20))"
    assert reversal["dependencies"] == ["close"]
    assert reversal["version"] == "1"
    assert reversal["inputs"] == ["close"]
    assert reversal["outputs"] == ["rev_20d"]
    assert "existing_factors" in reversal["capabilities"]
    assert reversal["grid_parameter"] == "window"
    assert reversal["grid_values"] == ["20", "60"]


def test_pipeline_snapshot_writes_config_and_checkpoints(tmp_path) -> None:
    client = TestClient(
        create_app(
            run_roots=[("platform", "reports/runs")],
            pipeline_output_root=tmp_path,
        )
    )
    payload = {
        "market": "a_share",
        "run_id": "api_snapshot_smoke",
        "date_start": "2024-01-01",
        "date_end": "2024-12-31",
        "factor_expression": "neg(return(close, 20))",
        "factor_template_id": "reversal_20d",
        "portfolio_mode": "long_only",
        "cost_bps": 10,
        "pnl_return_column": "forward_return_1d",
        "label_columns": ["forward_return_1d", "forward_return_20d"],
    }

    created = client.post("/api/pipeline-snapshots", json=payload)

    assert created.status_code == 201
    created_json = created.json()
    assert created_json["run_id"] == "api_snapshot_smoke"
    assert created_json["kind"] == "snapshot"
    assert created_json["status"] == "ready"
    run_dir = tmp_path / "api_snapshot_smoke"
    assert created_json["run_ref"] == "api_snapshot_smoke"
    assert "snapshot_path" not in created_json
    assert str(tmp_path) not in json.dumps(created_json)
    assert (run_dir / "run_config.yaml").is_file()
    assert (run_dir / "warnings.json").is_file()

    status = client.get("/api/pipeline-snapshots/api_snapshot_smoke")
    checkpoints = client.get("/api/pipeline-snapshots/api_snapshot_smoke/checkpoints")

    assert status.status_code == 200
    assert status.json()["run_ref"] == "api_snapshot_smoke"
    assert "snapshot_path" not in status.json()
    assert str(tmp_path) not in json.dumps(status.json())
    assert checkpoints.status_code == 200
    checkpoint_rows = checkpoints.json()
    assert checkpoint_rows[0]["label"] == "run_config.yaml"
    assert checkpoint_rows[0]["status"] == "ready"
    assert checkpoint_rows[0]["artifact"] == "run_config.yaml"
    assert "path" not in checkpoint_rows[0]
    assert checkpoint_rows[-1]["label"] == "summary.md"
    assert checkpoint_rows[-1]["status"] == "ready"

    assert client.get("/api/pipeline-runs/api_snapshot_smoke").status_code == 404
    assert (
        client.get("/api/pipeline-runs/api_snapshot_smoke/checkpoints").status_code
        == 404
    )


def test_pipeline_snapshot_returns_effective_run_ref_after_name_collision(
    tmp_path,
) -> None:
    client = TestClient(
        create_app(
            run_roots=[("platform", "reports/runs")],
            pipeline_output_root=tmp_path,
        )
    )
    payload = {
        "market": "a_share",
        "run_id": "api_snapshot_collision",
        "date_start": "2024-01-01",
        "date_end": "2024-12-31",
        "factor_expression": "neg(return(close, 20))",
        "factor_template_id": "reversal_20d",
        "portfolio_mode": "long_only",
        "cost_bps": 10,
        "pnl_return_column": "forward_return_1d",
        "label_columns": ["forward_return_1d"],
    }

    first = client.post("/api/pipeline-snapshots", json=payload)
    second = client.post("/api/pipeline-snapshots", json=payload)

    assert first.status_code == 201
    assert second.status_code == 201
    assert first.json()["run_ref"] == "api_snapshot_collision"
    second_ref = second.json()["run_ref"]
    assert second_ref.startswith("api_snapshot_collision_")
    assert (tmp_path / "api_snapshot_collision" / "run_config.yaml").is_file()
    assert (tmp_path / second_ref / "run_config.yaml").is_file()

    status = client.get("/api/pipeline-snapshots/api_snapshot_collision")
    checkpoints = client.get(
        "/api/pipeline-snapshots/api_snapshot_collision/checkpoints"
    )

    assert status.status_code == 200
    assert status.json()["run_ref"] == second_ref
    assert checkpoints.status_code == 200
    assert str(tmp_path) not in json.dumps(checkpoints.json())


def test_pipeline_run_executes_from_existing_factors(
    tmp_path,
    monkeypatch,
) -> None:
    from quant_research_platform.interfaces.api.services import pipeline_service
    factor_pipeline = pipeline_service

    calls: list[object] = []

    def fake_run_a_share_pipeline(config):
        calls.append(config)
        run_dir = tmp_path / config.run_id
        run_dir.mkdir()
        (run_dir / "checkpoints.json").write_text(
            json.dumps(
                [
                    {"key": "snapshot", "label": "配置快照", "status": "ready"},
                    {
                        "key": "evaluate",
                        "label": "A 股 evaluation",
                        "status": "ready",
                        "duration_seconds": 1.25,
                        "row_count": 42,
                        "failure_code": None,
                        "warnings": [{"code": "sample_warning"}],
                        "blocking_issues": [],
                    },
                ]
            ),
            encoding="utf-8",
        )
        return SimpleNamespace(
            status="ready",
            run_dir=run_dir,
            warnings=[],
        )

    monkeypatch.setattr(
        factor_pipeline,
        "run_a_share_pipeline",
        fake_run_a_share_pipeline,
    )
    client = TestClient(
        create_app(
            run_roots=[("platform", "reports/runs")],
            pipeline_output_root=tmp_path,
        )
    )

    created = client.post(
        "/api/pipeline-runs",
        json={
            "market": "a_share",
            "run_id": "api_existing_factors",
            "date_start": "2024-01-01",
            "date_end": "2024-12-31",
            "factor_expression": "neg(return(close, 20))",
            "factor_template_id": "reversal_20d",
            "portfolio_mode": "long_only",
            "cost_bps": 10,
            "pnl_return_column": "forward_return_1d",
            "label_columns": ["forward_return_1d"],
            "processed_factor_column": "rev_custom_20d",
            "start_from": "existing_factors",
        },
    )

    assert created.status_code == 201
    assert created.json()["status"] == "ready"
    assert created.json()["kind"] == "run"
    assert created.json()["run_ref"] == "api_existing_factors"
    assert "snapshot_path" not in created.json()
    assert str(tmp_path) not in json.dumps(created.json())
    assert len(calls) == 1
    assert calls[0].processed_factor_column == "rev_custom_20d"
    assert calls[0].start_from == "existing_factors"

    checkpoints = client.get("/api/pipeline-runs/api_existing_factors/checkpoints")

    assert checkpoints.status_code == 200
    checkpoint_rows = checkpoints.json()
    assert [checkpoint["key"] for checkpoint in checkpoint_rows] == [
        "snapshot",
        "evaluate",
    ]
    assert checkpoint_rows[1]["duration_seconds"] == 1.25
    assert checkpoint_rows[1]["row_count"] == 42
    assert checkpoint_rows[1]["warnings"] == [{"code": "sample_warning"}]
    assert checkpoint_rows[1]["blocking_issues"] == []
    assert all("path" not in checkpoint for checkpoint in checkpoint_rows)
    assert str(tmp_path) not in json.dumps(checkpoint_rows)

    assert client.get("/api/pipeline-snapshots/api_existing_factors").status_code == 404
    assert (
        client.get("/api/pipeline-snapshots/api_existing_factors/checkpoints")
        .status_code
        == 404
    )


def test_pipeline_snapshot_and_run_status_are_separated_for_same_run_id(
    tmp_path,
    monkeypatch,
) -> None:
    from quant_research_platform.interfaces.api.services import pipeline_service

    def fake_run_a_share_pipeline(config):
        return SimpleNamespace(
            status="ready",
            run_dir=tmp_path / config.run_id,
            warnings=[],
        )

    monkeypatch.setattr(
        pipeline_service,
        "run_a_share_pipeline",
        fake_run_a_share_pipeline,
    )
    client = TestClient(
        create_app(
            run_roots=[("platform", "reports/runs")],
            pipeline_output_root=tmp_path,
        )
    )
    base_payload = {
        "market": "a_share",
        "run_id": "shared_pipeline_id",
        "date_start": "2024-01-01",
        "date_end": "2024-12-31",
        "factor_expression": "neg(return(close, 20))",
        "factor_template_id": "reversal_20d",
        "portfolio_mode": "long_only",
        "cost_bps": 10,
        "pnl_return_column": "forward_return_1d",
        "label_columns": ["forward_return_1d"],
    }

    snapshot = client.post("/api/pipeline-snapshots", json=base_payload)
    run = client.post(
        "/api/pipeline-runs",
        json={**base_payload, "start_from": "existing_factors"},
    )

    assert snapshot.status_code == 201
    assert run.status_code == 201
    snapshot_status = client.get("/api/pipeline-snapshots/shared_pipeline_id")
    run_status = client.get("/api/pipeline-runs/shared_pipeline_id")
    assert snapshot_status.status_code == 200
    assert snapshot_status.json()["kind"] == "snapshot"
    assert run_status.status_code == 200
    assert run_status.json()["kind"] == "run"


def test_pipeline_snapshot_status_can_be_read_by_effective_run_ref_after_collision(
    tmp_path,
) -> None:
    client = TestClient(
        create_app(
            run_roots=[("platform", "reports/runs")],
            pipeline_output_root=tmp_path,
        )
    )
    payload = {
        "market": "a_share",
        "run_id": "status_collision",
        "date_start": "2024-01-01",
        "date_end": "2024-12-31",
        "factor_expression": "neg(return(close, 20))",
        "factor_template_id": "reversal_20d",
        "portfolio_mode": "long_only",
        "cost_bps": 10,
        "pnl_return_column": "forward_return_1d",
        "label_columns": ["forward_return_1d"],
    }

    first = client.post("/api/pipeline-snapshots", json=payload)
    second = client.post("/api/pipeline-snapshots", json=payload)

    assert first.status_code == 201
    assert second.status_code == 201
    second_ref = second.json()["run_ref"]

    status = client.get(f"/api/pipeline-snapshots/{second_ref}")
    checkpoints = client.get(f"/api/pipeline-snapshots/{second_ref}/checkpoints")

    assert status.status_code == 200
    assert status.json()["run_ref"] == second_ref
    assert checkpoints.status_code == 200


def test_pipeline_checkpoints_can_be_read_from_disk_after_restart(tmp_path) -> None:
    run_dir = tmp_path / "disk_run"
    run_dir.mkdir()
    (run_dir / "checkpoints.json").write_text(
        json.dumps(
            [
                {
                    "key": "evaluate",
                    "label": "evaluation",
                    "status": "ready",
                    "row_count": 7,
                }
            ]
        ),
        encoding="utf-8",
    )
    client = TestClient(
        create_app(
            run_roots=[("platform", "reports/runs")],
            pipeline_output_root=tmp_path,
        )
    )

    checkpoints = client.get("/api/pipeline-runs/disk_run/checkpoints")

    assert checkpoints.status_code == 200
    assert checkpoints.json()[0]["key"] == "evaluate"
    assert checkpoints.json()[0]["row_count"] == 7
    assert str(tmp_path) not in checkpoints.text


def test_pipeline_checkpoints_sanitize_private_artifacts_after_restart(
    tmp_path,
) -> None:
    run_dir = tmp_path / "private_artifact_run"
    run_dir.mkdir()
    (run_dir / "checkpoints.json").write_text(
        json.dumps(
            [
                {
                    "key": "build_factors",
                    "label": "factor build",
                    "status": "ready",
                    "artifact": "data/processed_private/csmar/factors.parquet",
                    "message": "failed at data/raw_private/csmar/TRD_Dalyr.csv",
                    "failure_code": "missing reports/private_real/pipeline_runs/x",
                    "warnings": [
                        {
                            "code": "private_warning",
                            "message": "uses data/interim_private/csmar",
                        }
                    ],
                    "blocking_issues": [
                        {
                            "code": "private_blocker",
                            "path": "data/processed_private/csmar/panel.parquet",
                        }
                    ],
                }
            ]
        ),
        encoding="utf-8",
    )
    client = TestClient(
        create_app(
            run_roots=[("platform", "reports/runs")],
            pipeline_output_root=tmp_path,
        )
    )

    checkpoints = client.get("/api/pipeline-runs/private_artifact_run/checkpoints")

    assert checkpoints.status_code == 200
    assert checkpoints.json()[0]["artifact"] is None
    assert checkpoints.json()[0]["message"] is None
    assert checkpoints.json()[0]["failure_code"] is None
    assert "raw_private" not in checkpoints.text
    assert "interim_private" not in checkpoints.text
    assert "processed_private" not in checkpoints.text
    assert "private_real" not in checkpoints.text
    assert str(tmp_path) not in checkpoints.text


def test_pipeline_checkpoints_sanitize_traversal_and_absolute_path_text_after_restart(
    tmp_path,
) -> None:
    run_dir = tmp_path / "unsafe_path_text_run"
    run_dir.mkdir()
    (run_dir / "checkpoints.json").write_text(
        json.dumps(
            [
                {
                    "key": "evaluate",
                    "label": "A-share evaluation",
                    "status": "failed",
                    "artifact": "../outside.csv",
                    "message": "failed(path=C:/Users/DELL/vendor/file.csv)",
                    "failure_code": "see(E:/vendor/secret.csv)",
                    "warnings": [
                        {
                            "code": "local_path",
                            "message": "loaded(path:..\\outside.csv)",
                        }
                    ],
                    "blocking_issues": [
                        {
                            "code": "absolute_path",
                            "path": "C:\\Users\\DELL\\vendor\\panel.parquet",
                        }
                    ],
                }
            ]
        ),
        encoding="utf-8",
    )
    client = TestClient(
        create_app(
            run_roots=[("platform", "reports/runs")],
            pipeline_output_root=tmp_path,
        )
    )

    checkpoints = client.get("/api/pipeline-runs/unsafe_path_text_run/checkpoints")

    assert checkpoints.status_code == 200
    checkpoint = checkpoints.json()[0]
    assert checkpoint["artifact"] is None
    assert checkpoint["message"] is None
    assert checkpoint["failure_code"] is None
    assert checkpoint["warnings"] == [{"code": "local_path"}]
    assert checkpoint["blocking_issues"] == [{"code": "absolute_path"}]
    assert "../outside.csv" not in checkpoints.text
    assert "C:/Users" not in checkpoints.text
    assert "E:/vendor" not in checkpoints.text
    assert "C:\\Users" not in checkpoints.text


def test_pipeline_checkpoints_reject_unsafe_required_fields_after_restart(
    tmp_path,
) -> None:
    run_dir = tmp_path / "unsafe_required_checkpoint_run"
    run_dir.mkdir()
    (run_dir / "checkpoints.json").write_text(
        json.dumps(
            [
                {
                    "key": "path=C:/Users/DELL/vendor/file.csv",
                    "label": "A-share evaluation",
                    "status": "failed",
                }
            ]
        ),
        encoding="utf-8",
    )
    client = TestClient(
        create_app(
            run_roots=[("platform", "reports/runs")],
            pipeline_output_root=tmp_path,
        )
    )

    checkpoints = client.get(
        "/api/pipeline-runs/unsafe_required_checkpoint_run/checkpoints"
    )

    assert checkpoints.status_code == 422
    assert checkpoints.json()["detail"] == "pipeline checkpoints invalid"
    assert "C:/Users" not in checkpoints.text


def test_pipeline_checkpoints_reject_unsafe_disk_run_ref(tmp_path) -> None:
    client = TestClient(
        create_app(
            run_roots=[("platform", "reports/runs")],
            pipeline_output_root=tmp_path,
        )
    )

    checkpoints = client.get("/api/pipeline-runs/..%5Coutside/checkpoints")

    assert checkpoints.status_code == 404


def test_pipeline_run_failure_response_hides_private_paths(
    tmp_path,
    monkeypatch,
) -> None:
    from quant_research_platform.interfaces.api.services import pipeline_service
    factor_pipeline = pipeline_service

    def fake_run_a_share_pipeline(config):
        run_dir = tmp_path / config.run_id
        run_dir.mkdir()
        (run_dir / "checkpoints.json").write_text(
            json.dumps(
                [
                    {
                        "key": "evaluate",
                        "label": "A 股 evaluation",
                        "status": "failed",
                        "artifact": None,
                        "message": "pipeline evaluation failed",
                    }
                ]
            ),
            encoding="utf-8",
        )
        raise factor_pipeline.PipelineRunnerError(
            "data/processed_private/csmar/missing.parquet",
            run_dir=run_dir,
        )

    monkeypatch.setattr(
        factor_pipeline,
        "run_a_share_pipeline",
        fake_run_a_share_pipeline,
    )
    client = TestClient(
        create_app(
            run_roots=[("platform", "reports/runs")],
            pipeline_output_root=tmp_path,
        )
    )

    response = client.post(
        "/api/pipeline-runs",
        json={
            "market": "a_share",
            "run_id": "failed_existing_factors",
            "date_start": "2024-01-01",
            "date_end": "2024-12-31",
            "factor_expression": "neg(return(close, 20))",
            "factor_template_id": "reversal_20d",
            "portfolio_mode": "long_only",
            "cost_bps": 10,
            "pnl_return_column": "forward_return_1d",
            "label_columns": ["forward_return_1d"],
            "start_from": "existing_factors",
        },
    )

    assert response.status_code == 201
    assert response.json()["status"] == "failed"
    assert "processed_private" not in response.text

    checkpoints = client.get("/api/pipeline-runs/failed_existing_factors/checkpoints")

    assert checkpoints.status_code == 200
    assert "processed_private" not in checkpoints.text


def test_pipeline_run_requires_explicit_start_from(tmp_path) -> None:
    client = TestClient(
        create_app(
            run_roots=[("platform", "reports/runs")],
            pipeline_output_root=tmp_path,
        )
    )

    response = client.post(
        "/api/pipeline-runs",
        json={
            "market": "a_share",
            "run_id": "missing_start_from",
            "date_start": "2024-01-01",
            "date_end": "2024-12-31",
            "factor_expression": "neg(return(close, 20))",
            "factor_template_id": "reversal_20d",
            "portfolio_mode": "long_only",
            "cost_bps": 0,
            "pnl_return_column": "forward_return_1d",
            "label_columns": ["forward_return_1d"],
        },
    )

    assert response.status_code == 422
    assert "start_from" in response.text
    assert not (tmp_path / "missing_start_from").exists()


def test_pipeline_run_rejects_retired_mode_and_rebuild_flags(
    tmp_path,
) -> None:
    client = TestClient(
        create_app(
            run_roots=[("platform", "reports/runs")],
            pipeline_output_root=tmp_path,
        )
    )
    payload = {
        "market": "a_share",
        "run_id": "retired_fields",
        "date_start": "2024-01-01",
        "date_end": "2024-12-31",
        "factor_expression": "neg(return(close, 20))",
        "factor_template_id": "reversal_20d",
        "portfolio_mode": "long_only",
        "cost_bps": 0,
        "pnl_return_column": "forward_return_1d",
        "label_columns": ["forward_return_1d"],
        "start_from": "existing_factors",
        "execution_mode": "processed_existing",
        "rebuild_panel": True,
    }

    response = client.post("/api/pipeline-runs", json=payload)

    assert response.status_code == 422
    assert "execution_mode" in response.text
    assert "rebuild_panel" in response.text
    assert not (tmp_path / "retired_fields").exists()


@pytest.mark.parametrize(
    "endpoint,extra_payload",
    [
        ("/api/pipeline-snapshots", {}),
        ("/api/pipeline-runs", {"start_from": "existing_factors"}),
    ],
)
@pytest.mark.parametrize(
    "retired_field",
    [
        {"execution_mode": "full_build"},
        {"execution_mode": "processed_existing"},
        {"full_build": True},
        {"processed_existing": True},
        {"rebuild_interim": True},
        {"rebuild_panel": True},
        {"rebuild_factors": True},
    ],
)
def test_pipeline_endpoints_reject_each_retired_execution_field(
    tmp_path,
    endpoint,
    extra_payload,
    retired_field,
) -> None:
    client = TestClient(
        create_app(
            run_roots=[("platform", "reports/runs")],
            pipeline_output_root=tmp_path,
        )
    )
    field_name = next(iter(retired_field))
    payload = {
        "market": "a_share",
        "run_id": f"reject_{field_name}",
        "date_start": "2024-01-01",
        "date_end": "2024-12-31",
        "factor_expression": "neg(return(close, 20))",
        "factor_template_id": "reversal_20d",
        "portfolio_mode": "long_only",
        "cost_bps": 0,
        "pnl_return_column": "forward_return_1d",
        "label_columns": ["forward_return_1d"],
        **extra_payload,
        **retired_field,
    }

    response = client.post(endpoint, json=payload)

    assert response.status_code == 422
    assert field_name in response.text
    assert not (tmp_path / f"reject_{field_name}").exists()


@pytest.mark.parametrize(
    "endpoint,extra_payload",
    [
        ("/api/pipeline-snapshots", {}),
        ("/api/pipeline-runs", {"start_from": "existing_factors"}),
    ],
)
@pytest.mark.parametrize(
    "template_id,expected_code",
    [
        ("../private_template", "unsafe_factor_template_id"),
        ("not_registered", "unknown_factor_template_id"),
    ],
)
def test_pipeline_endpoints_reject_unsafe_or_unknown_factor_template_id(
    tmp_path,
    endpoint,
    extra_payload,
    template_id,
    expected_code,
) -> None:
    client = TestClient(
        create_app(
            run_roots=[("platform", "reports/runs")],
            pipeline_output_root=tmp_path,
        )
    )
    payload = {
        "market": "a_share",
        "run_id": f"reject_template_{expected_code}",
        "date_start": "2024-01-01",
        "date_end": "2024-12-31",
        "factor_expression": "neg(return(close, 20))",
        "factor_template_id": template_id,
        "portfolio_mode": "long_only",
        "cost_bps": 0,
        "pnl_return_column": "forward_return_1d",
        "label_columns": ["forward_return_1d"],
        **extra_payload,
    }

    response = client.post(endpoint, json=payload)

    assert response.status_code == 422
    assert response.json()["detail"]["blocking"][0]["code"] == expected_code
    assert not (tmp_path / f"reject_template_{expected_code}").exists()


def test_pipeline_snapshot_rejects_retired_mode_and_rebuild_flags(
    tmp_path,
) -> None:
    client = TestClient(
        create_app(
            run_roots=[("platform", "reports/runs")],
            pipeline_output_root=tmp_path,
        )
    )
    payload = {
        "market": "a_share",
        "run_id": "snapshot_retired_fields",
        "date_start": "2024-01-01",
        "date_end": "2024-12-31",
        "factor_expression": "neg(return(close, 20))",
        "factor_template_id": "reversal_20d",
        "portfolio_mode": "long_only",
        "cost_bps": 0,
        "pnl_return_column": "forward_return_1d",
        "label_columns": ["forward_return_1d"],
        "execution_mode": "processed_existing",
        "rebuild_panel": True,
    }

    response = client.post("/api/pipeline-snapshots", json=payload)

    assert response.status_code == 422
    assert "execution_mode" in response.text
    assert "rebuild_panel" in response.text
    assert not (tmp_path / "snapshot_retired_fields").exists()


def test_pipeline_run_rejects_invalid_expression(tmp_path) -> None:
    client = TestClient(
        create_app(
            run_roots=[("platform", "reports/runs")],
            pipeline_output_root=tmp_path,
        )
    )

    response = client.post(
        "/api/pipeline-runs",
        json={
            "market": "a_share",
            "run_id": "bad_expression",
            "date_start": "2024-01-01",
            "date_end": "2024-12-31",
            "factor_expression": "__import__('os').system('echo bad')",
            "factor_template_id": "custom",
            "portfolio_mode": "long_only",
            "cost_bps": 0,
            "pnl_return_column": "forward_return_1d",
            "label_columns": ["forward_return_1d"],
            "start_from": "existing_factors",
        },
    )

    assert response.status_code == 422
    assert response.json()["detail"]["blocking"][0]["code"] == (
        "invalid_factor_expression"
    )
    assert not (tmp_path / "bad_expression").exists()


def test_pipeline_run_rejects_unsafe_run_id(tmp_path) -> None:
    client = TestClient(
        create_app(
            run_roots=[("platform", "reports/runs")],
            pipeline_output_root=tmp_path,
        )
    )

    response = client.post(
        "/api/pipeline-runs",
        json={
            "market": "a_share",
            "run_id": "../outside",
            "date_start": "2024-01-01",
            "date_end": "2024-12-31",
            "factor_expression": "neg(return(close, 20))",
            "factor_template_id": "custom",
            "portfolio_mode": "long_only",
            "cost_bps": 0,
            "pnl_return_column": "forward_return_1d",
            "label_columns": ["forward_return_1d"],
            "start_from": "existing_factors",
        },
    )

    assert response.status_code == 422
    assert response.json()["detail"]["blocking"][0]["code"] == "unsafe_run_id"
    assert not (tmp_path.parent / "outside").exists()


def test_pipeline_run_forbids_client_output_root(tmp_path) -> None:
    client = TestClient(
        create_app(
            run_roots=[("platform", "reports/runs")],
            pipeline_output_root=tmp_path,
        )
    )

    response = client.post(
        "/api/pipeline-runs",
        json={
            "market": "a_share",
            "run_id": "server_controlled",
            "date_start": "2024-01-01",
            "date_end": "2024-12-31",
            "factor_expression": "neg(return(close, 20))",
            "factor_template_id": "custom",
            "portfolio_mode": "long_only",
            "cost_bps": 0,
            "pnl_return_column": "forward_return_1d",
            "label_columns": ["forward_return_1d"],
            "start_from": "existing_factors",
            "output_root": (tmp_path / "client").as_posix(),
        },
    )

    assert response.status_code == 422
    assert "output_root" in str(response.json()["detail"])
    assert not (tmp_path / "server_controlled").exists()
