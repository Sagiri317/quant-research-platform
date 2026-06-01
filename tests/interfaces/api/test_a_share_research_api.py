from __future__ import annotations

import json

import pandas as pd
from fastapi.testclient import TestClient

from quant_research_platform.interfaces.api.app import create_app


def test_a_share_factor_rebuild_api_dry_run_and_status(tmp_path) -> None:
    panel_path = tmp_path / "data" / "processed_private" / "csmar" / "panel.parquet"
    panel_path.parent.mkdir(parents=True)
    _panel().to_parquet(panel_path, engine="pyarrow", compression="zstd", index=False)
    output_root = tmp_path / "reports" / "private_real" / "a_share_factor_rebuilds"
    client = TestClient(
        create_app(
            run_roots=[("platform", tmp_path / "runs")],
            factor_rebuild_root=output_root,
        )
    )

    response = client.post(
        "/api/a-share/factor-rebuild-runs",
        json={
            "run_id": "api_dry_run",
            "panel_parquet": str(panel_path),
            "dry_run": True,
            "memory_budget_gb": 1,
        },
    )

    assert response.status_code == 201, response.text
    payload = response.json()
    assert payload == {
        "run_id": "api_dry_run",
        "status": "dry_run",
        "row_count": 0,
        "factor_pack": "a_share_literature_price_volume_rebuild_v1",
    }
    assert str(tmp_path) not in response.text
    assert "processed_private" not in response.text

    status_response = client.get("/api/a-share/factor-rebuild-runs/api_dry_run")
    assert status_response.status_code == 200
    assert status_response.json()["status"] == "dry_run"
    assert str(tmp_path) not in status_response.text

    checkpoints = client.get(
        "/api/a-share/factor-rebuild-runs/api_dry_run/checkpoints"
    )
    assert checkpoints.status_code == 200
    assert [item["key"] for item in checkpoints.json()] == [
        "schema_inventory",
        "memory_preflight",
        "factor_rebuild",
        "feature_set_manifest",
        "overwrite_manifest",
        "eda",
        "runtime_summary",
    ]

    eda = client.get("/api/a-share/factor-rebuild-runs/api_dry_run/eda")
    assert eda.status_code == 200
    assert eda.json() == {"run_id": "api_dry_run", "status": "not_generated"}


def test_a_share_factor_rebuild_api_rejects_duplicate_run_id_before_read(
    tmp_path,
) -> None:
    panel_path = tmp_path / "data" / "processed_private" / "csmar" / "panel.parquet"
    panel_path.parent.mkdir(parents=True)
    _panel().to_parquet(panel_path, engine="pyarrow", compression="zstd", index=False)
    output_root = tmp_path / "reports" / "private_real" / "a_share_factor_rebuilds"
    (output_root / "api_dry_run").mkdir(parents=True)
    client = TestClient(
        create_app(
            run_roots=[("platform", tmp_path / "runs")],
            factor_rebuild_root=output_root,
        )
    )

    response = client.post(
        "/api/a-share/factor-rebuild-runs",
        json={
            "run_id": "api_dry_run",
            "panel_parquet": str(panel_path),
            "dry_run": True,
        },
    )

    assert response.status_code == 409
    assert response.json()["error"]["error_code"] == "duplicate_run_id"
    assert str(tmp_path) not in response.text
    assert "processed_private" not in response.text


def test_a_share_prediction_handoff_api_creates_discoverable_evaluation(
    tmp_path,
) -> None:
    input_path = tmp_path / "features.parquet"
    _prediction_panel().to_parquet(
        input_path,
        engine="pyarrow",
        compression="zstd",
        index=False,
    )
    prediction_root = (
        tmp_path / "reports" / "private_real" / "a_share_prediction_handoffs"
    )
    client = TestClient(
        create_app(
            run_roots=[("platform", tmp_path / "runs")],
            prediction_handoff_root=prediction_root,
        )
    )

    created = client.post(
        "/api/a-share/prediction-handoffs",
        json={
            "run_id": "prediction_api_run",
            "input_parquet": str(input_path),
            "timestamp_column": "timestamp",
            "entity_column": "asset",
            "feature_columns": ["feature_a", "feature_b"],
            "label_column": "forward_return_2d",
            "return_source_column": "forward_return_1d",
            "label_horizon_days": 2,
            "model_ids": ["ridge"],
            "folds": [
                {
                    "fold_id": "fold_001",
                    "train_end": "2024-01-05",
                    "test_start": "2024-01-08",
                    "test_end": "2024-01-09",
                }
            ],
        },
    )

    assert created.status_code == 201, created.text
    assert created.json()["status"] == "ready"
    assert str(tmp_path) not in created.text

    status = client.get("/api/a-share/prediction-handoffs/prediction_api_run")
    assert status.status_code == 200
    assert status.json()["status"] == "ready"

    schema = client.get("/api/a-share/prediction-handoffs/prediction_api_run/schema")
    assert schema.status_code == 200
    assert schema.json()["prediction_handoff_version"] == (
        "n6_a_share_prediction_handoff_v1"
    )

    comparison = client.get(
        "/api/a-share/prediction-handoffs/prediction_api_run/comparison"
    )
    assert comparison.status_code == 200
    assert comparison.json()["status"] == "ready"
    assert comparison.json()["model_count"] >= 1

    eda = client.get("/api/a-share/prediction-handoffs/prediction_api_run/eda")
    assert eda.status_code == 200
    assert eda.json()["status"] == "ready"

    runs = client.get("/api/runs")
    assert runs.status_code == 200
    assert "a_share_prediction/prediction_api_run" in {
        row["public_run_id"] for row in runs.json()
    }


def test_default_app_discovers_default_prediction_handoff_root(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("QRP_RUN_ROOTS", raising=False)
    monkeypatch.delenv("QRP_PREDICTION_HANDOFF_ROOT", raising=False)
    input_path = tmp_path / "features.parquet"
    _prediction_panel().to_parquet(
        input_path,
        engine="pyarrow",
        compression="zstd",
        index=False,
    )
    client = TestClient(create_app())

    created = client.post(
        "/api/a-share/prediction-handoffs",
        json={
            "run_id": "default_prediction_api_run",
            "input_parquet": str(input_path),
            "timestamp_column": "timestamp",
            "entity_column": "asset",
            "feature_columns": ["feature_a", "feature_b"],
            "label_column": "forward_return_2d",
            "return_source_column": "forward_return_1d",
            "label_horizon_days": 2,
            "model_ids": ["ridge"],
            "folds": [
                {
                    "fold_id": "fold_001",
                    "train_end": "2024-01-05",
                    "test_start": "2024-01-08",
                    "test_end": "2024-01-09",
                }
            ],
        },
    )

    assert created.status_code == 201, created.text
    runs = client.get("/api/runs")
    assert runs.status_code == 200
    assert "a_share_prediction/default_prediction_api_run" in {
        row["public_run_id"] for row in runs.json()
    }


def test_a_share_prediction_handoff_api_rejects_horizon_mismatch(
    tmp_path,
) -> None:
    input_path = tmp_path / "features.parquet"
    _prediction_panel().to_parquet(
        input_path,
        engine="pyarrow",
        compression="zstd",
        index=False,
    )
    prediction_root = (
        tmp_path / "reports" / "private_real" / "a_share_prediction_handoffs"
    )
    client = TestClient(
        create_app(
            run_roots=[("platform", tmp_path / "runs")],
            prediction_handoff_root=prediction_root,
        )
    )

    response = client.post(
        "/api/a-share/prediction-handoffs",
        json={
            "run_id": "prediction_bad_horizon",
            "input_parquet": str(input_path),
            "timestamp_column": "timestamp",
            "entity_column": "asset",
            "feature_columns": ["feature_a", "feature_b"],
            "label_column": "forward_return_20d",
            "return_source_column": "forward_return_1d",
            "label_horizon_days": 2,
            "model_ids": ["ridge"],
            "folds": [
                {
                    "fold_id": "fold_001",
                    "train_end": "2024-01-05",
                    "test_start": "2024-01-08",
                    "test_end": "2024-01-09",
                }
            ],
        },
    )

    assert response.status_code == 422
    assert response.json()["error"]["error_code"] == "invalid_prediction_config"
    status = client.get("/api/a-share/prediction-handoffs/prediction_bad_horizon")
    assert status.status_code == 200
    assert status.json()["failure_code"] == "invalid_prediction_config"


def test_a_share_status_endpoints_sanitize_disk_loaded_strings(tmp_path) -> None:
    factor_root = tmp_path / "reports" / "private_real" / "a_share_factor_rebuilds"
    factor_run = factor_root / "bad_factor_status"
    factor_run.mkdir(parents=True)
    (factor_run / "runtime_summary.json").write_text(
        json.dumps(
            {
                "status": "credential_token=abc123",
                "factor_pack": "credential_token=abc123",
                "failure_code": "ValueError: raw unstable exception",
                "row_count": 3,
            }
        ),
        encoding="utf-8",
    )
    prediction_root = (
        tmp_path / "reports" / "private_real" / "a_share_prediction_handoffs"
    )
    prediction_run = prediction_root / "bad_prediction_status"
    prediction_run.mkdir(parents=True)
    (prediction_run / "runtime_summary.json").write_text(
        json.dumps(
            {
                "status": "../reports/private_real/failed",
                "model_suite_id": "api_key=abc123",
                "failure_code": "Traceback RuntimeError raw exception text",
                "row_count": 5,
            }
        ),
        encoding="utf-8",
    )
    client = TestClient(
        create_app(
            run_roots=[("platform", tmp_path / "runs")],
            factor_rebuild_root=factor_root,
            prediction_handoff_root=prediction_root,
        )
    )

    factor_status = client.get(
        "/api/a-share/factor-rebuild-runs/bad_factor_status"
    )
    prediction_status = client.get(
        "/api/a-share/prediction-handoffs/bad_prediction_status"
    )

    assert factor_status.status_code == 200
    assert prediction_status.status_code == 200
    assert factor_status.json() == {
        "run_id": "bad_factor_status",
        "status": "unknown",
        "factor_pack": "a_share_literature_price_volume_rebuild_v1",
        "row_count": 3,
    }
    assert prediction_status.json() == {
        "run_id": "bad_prediction_status",
        "status": "unknown",
        "row_count": 5,
        "model_suite_id": "n6_a_share_sklearn_boosting_suite_v1",
    }
    assert str(tmp_path) not in factor_status.text
    assert str(tmp_path) not in prediction_status.text
    assert "credential_token" not in factor_status.text
    assert "ValueError" not in factor_status.text
    assert "api_key" not in prediction_status.text
    assert "Traceback" not in prediction_status.text
    assert "private_real" not in prediction_status.text


def _panel() -> pd.DataFrame:
    rows = []
    for stock_code, base in (("000001", 10.0), ("000002", 20.0)):
        for offset in range(3):
            close = base + offset
            rows.append(
                {
                    "stock_code": stock_code,
                    "trade_date": pd.Timestamp("2024-01-01")
                    + pd.offsets.BDay(offset),
                    "market_type": "A",
                    "is_a_share_scope": True,
                    "eligible": True,
                    "can_enter": True,
                    "can_exit": True,
                    "adj_open": close - 0.1,
                    "adj_close": close,
                    "return_1d": 0.01 * offset,
                    "amount": 100.0 + offset,
                    "turnover": 0.1 + offset / 100.0,
                    "high": close + 1.0,
                    "low": close - 1.0,
                    "prev_close": close - 0.5,
                    "float_market_cap": 1000.0 + offset,
                    "total_market_cap": 2000.0 + offset,
                }
            )
    return pd.DataFrame(rows)


def _prediction_panel() -> pd.DataFrame:
    rows = []
    dates = pd.bdate_range("2024-01-01", periods=8)
    for asset, offset in (("A", 0.0), ("B", 0.5), ("C", 1.0)):
        for idx, date in enumerate(dates):
            rows.append(
                {
                    "timestamp": date,
                    "asset": asset,
                    "feature_a": float(idx) + offset,
                    "feature_b": float(idx % 3) - offset,
                    "forward_return_2d": 0.01 * idx + offset / 100.0,
                    "forward_return_20d": 0.02 * idx + offset / 100.0,
                    "forward_return_1d": 0.001 * idx,
                }
            )
    return pd.DataFrame(rows)
