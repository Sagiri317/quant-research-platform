from __future__ import annotations

import json
from types import SimpleNamespace

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
    assert "processed_existing" in reversal["capabilities"]
    assert reversal["grid_parameter"] == "window"
    assert reversal["grid_values"] == ["20", "60"]


def test_pipeline_run_writes_snapshot_and_checkpoints(tmp_path) -> None:
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

    created = client.post("/api/pipeline-runs", json=payload)

    assert created.status_code == 201
    created_json = created.json()
    assert created_json["run_id"] == "api_snapshot_smoke"
    assert created_json["status"] == "ready"
    run_dir = tmp_path / "api_snapshot_smoke"
    assert created_json["run_ref"] == "api_snapshot_smoke"
    assert "snapshot_path" not in created_json
    assert str(tmp_path) not in json.dumps(created_json)
    assert (run_dir / "run_config.yaml").is_file()
    assert (run_dir / "warnings.json").is_file()

    status = client.get("/api/pipeline-runs/api_snapshot_smoke")
    checkpoints = client.get("/api/pipeline-runs/api_snapshot_smoke/checkpoints")

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
    assert checkpoint_rows[-1]["label"] == "评估清单"
    assert checkpoint_rows[-1]["status"] == "pending"


def test_pipeline_run_returns_effective_run_ref_after_name_collision(tmp_path) -> None:
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

    first = client.post("/api/pipeline-runs", json=payload)
    second = client.post("/api/pipeline-runs", json=payload)

    assert first.status_code == 201
    assert second.status_code == 201
    assert first.json()["run_ref"] == "api_snapshot_collision"
    second_ref = second.json()["run_ref"]
    assert second_ref.startswith("api_snapshot_collision_")
    assert (tmp_path / "api_snapshot_collision" / "run_config.yaml").is_file()
    assert (tmp_path / second_ref / "run_config.yaml").is_file()

    status = client.get("/api/pipeline-runs/api_snapshot_collision")
    checkpoints = client.get("/api/pipeline-runs/api_snapshot_collision/checkpoints")

    assert status.status_code == 200
    assert status.json()["run_ref"] == second_ref
    assert checkpoints.status_code == 200
    assert str(tmp_path) not in json.dumps(checkpoints.json())


def test_pipeline_run_can_execute_processed_existing_mode(
    tmp_path,
    monkeypatch,
) -> None:
    from quant_research_platform.interfaces.api.services import pipeline_service
    factor_pipeline = pipeline_service

    calls: list[object] = []

    def fake_run_processed_existing_pipeline(config):
        calls.append(config)
        run_dir = tmp_path / config.run_id
        run_dir.mkdir()
        (run_dir / "checkpoints.json").write_text(
            json.dumps(
                [
                    {"key": "snapshot", "label": "配置快照", "status": "ready"},
                    {
                        "key": "processed_evaluation",
                        "label": "A 股 processed evaluation",
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
        "run_processed_existing_pipeline",
        fake_run_processed_existing_pipeline,
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
            "run_id": "api_processed_existing",
            "date_start": "2024-01-01",
            "date_end": "2024-12-31",
            "factor_expression": "neg(return(close, 20))",
            "factor_template_id": "reversal_20d",
            "portfolio_mode": "long_only",
            "cost_bps": 10,
            "pnl_return_column": "forward_return_1d",
            "label_columns": ["forward_return_1d"],
            "processed_factor_column": "rev_custom_20d",
            "execution_mode": "processed_existing",
        },
    )

    assert created.status_code == 201
    assert created.json()["status"] == "ready"
    assert created.json()["run_ref"] == "api_processed_existing"
    assert "snapshot_path" not in created.json()
    assert str(tmp_path) not in json.dumps(created.json())
    assert len(calls) == 1
    assert calls[0].processed_factor_column == "rev_custom_20d"

    checkpoints = client.get("/api/pipeline-runs/api_processed_existing/checkpoints")

    assert checkpoints.status_code == 200
    checkpoint_rows = checkpoints.json()
    assert [checkpoint["key"] for checkpoint in checkpoint_rows] == [
        "snapshot",
        "processed_evaluation",
    ]
    assert checkpoint_rows[1]["duration_seconds"] == 1.25
    assert checkpoint_rows[1]["row_count"] == 42
    assert checkpoint_rows[1]["warnings"] == [{"code": "sample_warning"}]
    assert checkpoint_rows[1]["blocking_issues"] == []
    assert all("path" not in checkpoint for checkpoint in checkpoint_rows)
    assert str(tmp_path) not in json.dumps(checkpoint_rows)


def test_pipeline_run_status_can_be_read_by_effective_run_ref_after_collision(
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

    first = client.post("/api/pipeline-runs", json=payload)
    second = client.post("/api/pipeline-runs", json=payload)

    assert first.status_code == 201
    assert second.status_code == 201
    second_ref = second.json()["run_ref"]

    status = client.get(f"/api/pipeline-runs/{second_ref}")
    checkpoints = client.get(f"/api/pipeline-runs/{second_ref}/checkpoints")

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
                    "key": "processed_evaluation",
                    "label": "processed evaluation",
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
    assert checkpoints.json()[0]["key"] == "processed_evaluation"
    assert checkpoints.json()[0]["row_count"] == 7
    assert str(tmp_path) not in checkpoints.text


def test_pipeline_checkpoints_reject_unsafe_disk_run_ref(tmp_path) -> None:
    client = TestClient(
        create_app(
            run_roots=[("platform", "reports/runs")],
            pipeline_output_root=tmp_path,
        )
    )

    checkpoints = client.get("/api/pipeline-runs/..%5Coutside/checkpoints")

    assert checkpoints.status_code == 404


def test_processed_existing_failure_response_hides_private_paths(
    tmp_path,
    monkeypatch,
) -> None:
    from quant_research_platform.interfaces.api.services import pipeline_service
    factor_pipeline = pipeline_service

    def fake_run_processed_existing_pipeline(config):
        run_dir = tmp_path / config.run_id
        run_dir.mkdir()
        (run_dir / "checkpoints.json").write_text(
            json.dumps(
                [
                    {
                        "key": "processed_evaluation",
                        "label": "A 股 processed evaluation",
                        "status": "failed",
                        "artifact": None,
                        "message": "processed evaluation failed",
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
        "run_processed_existing_pipeline",
        fake_run_processed_existing_pipeline,
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
            "run_id": "failed_processed_existing",
            "date_start": "2024-01-01",
            "date_end": "2024-12-31",
            "factor_expression": "neg(return(close, 20))",
            "factor_template_id": "reversal_20d",
            "portfolio_mode": "long_only",
            "cost_bps": 10,
            "pnl_return_column": "forward_return_1d",
            "label_columns": ["forward_return_1d"],
            "execution_mode": "processed_existing",
        },
    )

    assert response.status_code == 201
    assert response.json()["status"] == "failed"
    assert "processed_private" not in response.text

    checkpoints = client.get("/api/pipeline-runs/failed_processed_existing/checkpoints")

    assert checkpoints.status_code == 200
    assert "processed_private" not in checkpoints.text


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
            "output_root": (tmp_path / "client").as_posix(),
        },
    )

    assert response.status_code == 422
    assert "output_root" in str(response.json()["detail"])
    assert not (tmp_path / "server_controlled").exists()
