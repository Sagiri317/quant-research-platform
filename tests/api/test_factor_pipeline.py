from __future__ import annotations

from fastapi.testclient import TestClient

from quant_research_platform.api.app import create_app


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
    assert created_json["snapshot_path"] == run_dir.as_posix()
    assert (run_dir / "run_config.yaml").is_file()
    assert (run_dir / "warnings.json").is_file()

    status = client.get("/api/pipeline-runs/api_snapshot_smoke")
    checkpoints = client.get("/api/pipeline-runs/api_snapshot_smoke/checkpoints")

    assert status.status_code == 200
    assert status.json()["snapshot_path"] == run_dir.as_posix()
    assert checkpoints.status_code == 200
    checkpoint_rows = checkpoints.json()
    assert checkpoint_rows[0]["label"] == "run_config.yaml"
    assert checkpoint_rows[0]["status"] == "ready"
    assert checkpoint_rows[-1]["label"] == "评估清单"
    assert checkpoint_rows[-1]["status"] == "pending"


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
