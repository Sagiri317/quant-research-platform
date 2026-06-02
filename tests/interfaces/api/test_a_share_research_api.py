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
        "warmup_sufficiency",
        "factor_rebuild",
        "feature_set_manifest",
        "overwrite_manifest",
        "eda",
        "runtime_summary",
    ]
    schema_inventory = client.get(
        "/api/a-share/factor-rebuild-runs/api_dry_run/schema-inventory"
    )
    assert schema_inventory.status_code == 200
    assert schema_inventory.json()["schema_inventory_version"] == (
        "a_share_schema_inventory_v1"
    )

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
    schema_inventory = client.get(
        "/api/a-share/prediction-handoffs/prediction_api_run/schema-inventory"
    )
    assert schema_inventory.status_code == 200
    assert schema_inventory.json()["schema_inventory_version"] == (
        "a_share_schema_inventory_v1"
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


def test_a_share_prediction_handoff_api_resolves_feature_preset(
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
            "run_id": "prediction_api_preset_run",
            "input_parquet": str(input_path),
            "timestamp_column": "timestamp",
            "entity_column": "asset",
            "feature_preset": "a_share_literature_price_volume_preset_v1",
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

    assert response.status_code == 201, response.text
    manifest = json.loads(
        (
            prediction_root
            / "prediction_api_preset_run"
            / "feature_set_manifest.json"
        ).read_text("utf-8")
    )
    assert manifest["feature_resolution"] == "preset"
    assert manifest["feature_columns"] == ["rev_1d", "mom_20d"]


def test_a_share_prediction_handoff_api_rejects_missing_feature_selection(
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
            "run_id": "prediction_missing_features",
            "input_parquet": str(input_path),
            "timestamp_column": "timestamp",
            "entity_column": "asset",
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

    assert response.status_code == 422
    assert response.json()["error"]["error_code"] == "invalid_feature_preset"
    assert not (prediction_root / "prediction_missing_features").exists()


def test_a_share_prediction_handoff_api_rejects_preset_with_no_usable_columns(
    tmp_path,
) -> None:
    input_path = tmp_path / "features.parquet"
    _prediction_panel_without_literature_features().to_parquet(
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
            "run_id": "prediction_no_preset_columns",
            "input_parquet": str(input_path),
            "timestamp_column": "timestamp",
            "entity_column": "asset",
            "feature_preset": "a_share_literature_price_volume_preset_v1",
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

    assert response.status_code == 422
    assert response.json()["error"]["error_code"] == "invalid_feature_preset"
    assert not (prediction_root / "prediction_no_preset_columns").exists()


def test_a_share_prediction_handoff_api_explicit_features_override_preset_metadata(
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
            "run_id": "prediction_explicit_over_preset",
            "input_parquet": str(input_path),
            "timestamp_column": "timestamp",
            "entity_column": "asset",
            "feature_columns": ["feature_a", "feature_b"],
            "feature_preset": "unsupported_preset",
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

    assert response.status_code == 201, response.text
    manifest = json.loads(
        (
            prediction_root
            / "prediction_explicit_over_preset"
            / "feature_set_manifest.json"
        ).read_text("utf-8")
    )
    assert manifest["feature_resolution"] == "explicit"
    assert manifest["feature_set_id"] == "explicit_feature_columns"
    assert manifest["feature_columns"] == ["feature_a", "feature_b"]
    run_context = json.loads(
        (
            prediction_root
            / "prediction_explicit_over_preset"
            / "evaluation"
            / "run_context.json"
        ).read_text("utf-8")
    )
    metadata = run_context["config"]["evaluation_spec"]["metadata"]
    assert metadata["source_ref"] == "explicit_feature_columns"
    leaderboard = json.loads(
        (
            prediction_root
            / "prediction_explicit_over_preset"
            / "comparison"
            / "model_leaderboard.json"
        ).read_text("utf-8")
    )
    assert {row["feature_set_id"] for row in leaderboard} == {
        "explicit_feature_columns"
    }


def test_a_share_prediction_handoff_api_uses_configured_alias_root_for_writes(
    tmp_path,
) -> None:
    input_path = tmp_path / "features.parquet"
    _prediction_panel().to_parquet(
        input_path,
        engine="pyarrow",
        compression="zstd",
        index=False,
    )
    configured_prediction_root = tmp_path / "configured_prediction_root"
    client = TestClient(
        create_app(
            run_roots=[
                ("platform", tmp_path / "runs"),
                ("a_share_prediction", configured_prediction_root),
            ],
        )
    )

    response = client.post(
        "/api/a-share/prediction-handoffs",
        json={
            "run_id": "prediction_configured_alias_root",
            "input_parquet": str(input_path),
            "timestamp_column": "timestamp",
            "entity_column": "asset",
            "feature_preset": "a_share_literature_price_volume_preset_v1",
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

    assert response.status_code == 201, response.text
    assert (configured_prediction_root / "prediction_configured_alias_root").is_dir()
    runs = client.get("/api/runs")
    assert runs.status_code == 200
    assert "a_share_prediction/prediction_configured_alias_root" in {
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


def test_a_share_prediction_handoff_api_discovers_default_root_with_explicit_run_roots(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("QRP_PREDICTION_HANDOFF_ROOT", raising=False)
    input_path = tmp_path / "features.parquet"
    _prediction_panel().to_parquet(
        input_path,
        engine="pyarrow",
        compression="zstd",
        index=False,
    )
    client = TestClient(create_app(run_roots=[("platform", tmp_path / "runs")]))

    created = client.post(
        "/api/a-share/prediction-handoffs",
        json={
            "run_id": "default_registered_prediction",
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
    public_run_ids = [row["public_run_id"] for row in runs.json()]
    assert public_run_ids.count("a_share_prediction/default_registered_prediction") == 1


def test_a_share_prediction_handoff_api_rejects_duplicate_run_id_before_input_read(
    tmp_path,
) -> None:
    prediction_root = (
        tmp_path / "reports" / "private_real" / "a_share_prediction_handoffs"
    )
    (prediction_root / "prediction_duplicate").mkdir(parents=True)
    client = TestClient(
        create_app(
            run_roots=[("platform", tmp_path / "runs")],
            prediction_handoff_root=prediction_root,
        )
    )

    response = client.post(
        "/api/a-share/prediction-handoffs",
        json={
            "run_id": "prediction_duplicate",
            "input_parquet": str(tmp_path / "missing.parquet"),
            "timestamp_column": "timestamp",
            "entity_column": "asset",
            "feature_preset": "a_share_literature_price_volume_preset_v1",
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

    assert response.status_code == 409
    assert response.json()["error"]["error_code"] == "duplicate_run_id"
    assert "missing.parquet" not in response.text
    assert str(tmp_path) not in response.text


def test_a_share_prediction_handoff_api_rejects_malformed_folds_before_run_dir(
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
            "run_id": "prediction_malformed_folds",
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
                    "test_start": "2024-01-08",
                }
            ],
        },
    )

    assert response.status_code == 422
    assert response.json()["error"]["error_code"] == "invalid_prediction_config"
    assert "KeyError" not in response.text
    assert "train_end" not in response.text
    assert not (prediction_root / "prediction_malformed_folds").exists()


def test_a_share_prediction_handoff_api_rejects_overlapping_folds_without_row_outputs(
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
            "run_id": "prediction_overlap_folds",
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
                },
                {
                    "fold_id": "fold_002",
                    "train_end": "2024-01-05",
                    "test_start": "2024-01-09",
                    "test_end": "2024-01-10",
                },
            ],
        },
    )

    assert response.status_code == 422
    assert response.json()["error"]["error_code"] == "invalid_prediction_config"
    run_dir = prediction_root / "prediction_overlap_folds"
    assert not (run_dir / "predictions.parquet").exists()
    assert not (run_dir / "prediction_records.parquet").exists()


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


def test_a_share_status_and_schema_endpoints_reject_malformed_numeric_fields(
    tmp_path,
) -> None:
    factor_root = tmp_path / "reports" / "private_real" / "a_share_factor_rebuilds"
    factor_run = factor_root / "bad_factor_status_numeric"
    factor_run.mkdir(parents=True)
    (factor_run / "runtime_summary.json").write_text(
        json.dumps(
            {
                "status": "ready",
                "factor_pack": "a_share_literature_price_volume_rebuild_v1",
                "row_count": "not_an_int",
            }
        ),
        encoding="utf-8",
    )
    prediction_root = (
        tmp_path / "reports" / "private_real" / "a_share_prediction_handoffs"
    )
    prediction_status_run = prediction_root / "bad_prediction_status_numeric"
    prediction_status_run.mkdir(parents=True)
    (prediction_status_run / "runtime_summary.json").write_text(
        json.dumps(
            {
                "status": "ready",
                "row_count": "not_an_int",
                "model_suite_id": "n6_a_share_sklearn_boosting_suite_v1",
            }
        ),
        encoding="utf-8",
    )
    prediction_schema_run = prediction_root / "bad_prediction_schema_numeric"
    prediction_schema_run.mkdir(parents=True)
    (prediction_schema_run / "model_metadata.json").write_text(
        json.dumps(
            {
                "prediction_handoff_version": "n6_a_share_prediction_handoff_v1",
                "model_suite_id": "n6_a_share_sklearn_boosting_suite_v1",
                "prediction_columns": ["prediction_score_ridge_20d"],
                "target_column": "forward_return_20d",
                "label_horizon_days": "not_an_int",
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
        "/api/a-share/factor-rebuild-runs/bad_factor_status_numeric"
    )
    prediction_status = client.get(
        "/api/a-share/prediction-handoffs/bad_prediction_status_numeric"
    )
    prediction_schema = client.get(
        "/api/a-share/prediction-handoffs/bad_prediction_schema_numeric/schema"
    )

    assert factor_status.status_code == 422
    assert prediction_status.status_code == 422
    assert prediction_schema.status_code == 422
    assert factor_status.json()["error"]["error_code"] == (
        "factor_rebuild_status_invalid"
    )
    assert prediction_status.json()["error"]["error_code"] == (
        "prediction_handoff_status_invalid"
    )
    assert prediction_schema.json()["error"]["error_code"] == (
        "prediction_handoff_status_invalid"
    )
    combined = factor_status.text + prediction_status.text + prediction_schema.text
    assert "not_an_int" not in combined


def test_a_share_schema_and_eda_endpoints_sanitize_disk_loaded_strings(
    tmp_path,
) -> None:
    factor_root = tmp_path / "reports" / "private_real" / "a_share_factor_rebuilds"
    factor_run = factor_root / "bad_factor_eda"
    (factor_run / "eda").mkdir(parents=True)
    (factor_run / "eda" / "eda_manifest.json").write_text(
        json.dumps(
            {
                "status": "credential_token=abc123",
                "artifact_count": 4,
                "chart_count": 2,
                "chart_dependency_status": "ready",
                "chart_files": [
                    "factor_coverage_by_year.png",
                    "../processed_private/leak.png",
                    "credential_token.png",
                ],
            }
        ),
        encoding="utf-8",
    )
    prediction_root = (
        tmp_path / "reports" / "private_real" / "a_share_prediction_handoffs"
    )
    prediction_run = prediction_root / "bad_prediction_schema"
    (prediction_run / "eda").mkdir(parents=True)
    (prediction_run / "model_metadata.json").write_text(
        json.dumps(
            {
                "prediction_handoff_version": "credential_token=abc123",
                "model_suite_id": "api_key=abc123",
                "prediction_columns": [
                    "prediction_score_ridge_20d",
                    "C:/private/path",
                    "secret_token=abc123",
                ],
                "target_column": "../processed_private/forward_return_20d",
                "label_horizon_days": 20,
            }
        ),
        encoding="utf-8",
    )
    (prediction_run / "eda" / "eda_manifest.json").write_text(
        json.dumps(
            {
                "status": "reports/private_real/leak",
                "artifact_count": 6,
                "chart_count": 5,
                "chart_dependency_status": "reports/private_real/leak",
                "chart_files": ["coverage_by_year.png", "C:/private/path.png"],
            }
        ),
        encoding="utf-8",
    )
    (prediction_run / "schema_inventory.json").write_text(
        json.dumps(
            {
                "schema_inventory_version": "a_share_schema_inventory_v1",
                "row_count": 3,
                "row_group_count": 1,
                "columns": [
                    {"name": "timestamp", "dtype": "timestamp[ns]", "role": "key"},
                    {
                        "name": "processed_private/leak",
                        "dtype": "string",
                        "role": "credential_token",
                    },
                    {"name": "vendor_raw_flag", "dtype": "string", "role": "excluded"},
                    {
                        "name": "source_download_code",
                        "dtype": "string",
                        "role": "excluded",
                    },
                    {"name": "local_path", "dtype": "string", "role": "excluded"},
                ],
                "missing_expected_columns": [
                    "secret_token",
                    "vendor_missing",
                    "local_missing_path",
                ],
                "unexpected_columns": ["rev_1d", "source_download_code", "local_path"],
                "duplicate_key_count": 0,
                "date_range": {"min": "2024-01-01", "max": "2024-01-02"},
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

    factor_eda = client.get("/api/a-share/factor-rebuild-runs/bad_factor_eda/eda")
    schema = client.get(
        "/api/a-share/prediction-handoffs/bad_prediction_schema/schema"
    )
    prediction_eda = client.get(
        "/api/a-share/prediction-handoffs/bad_prediction_schema/eda"
    )
    schema_inventory = client.get(
        "/api/a-share/prediction-handoffs/bad_prediction_schema/schema-inventory"
    )

    assert factor_eda.status_code == 200
    assert schema.status_code == 200
    assert prediction_eda.status_code == 200
    assert schema_inventory.status_code == 200
    assert factor_eda.json() == {
        "run_id": "bad_factor_eda",
        "status": "unknown",
        "artifact_count": 4,
        "chart_count": 2,
        "chart_dependency_status": "ready",
        "chart_files": ["factor_coverage_by_year.png"],
    }
    assert schema.json() == {
        "prediction_handoff_version": "",
        "model_suite_id": "n6_a_share_sklearn_boosting_suite_v1",
        "prediction_columns": ["prediction_score_ridge_20d"],
        "target_column": "",
        "label_horizon_days": 20,
    }
    assert prediction_eda.json() == {
        "run_id": "bad_prediction_schema",
        "status": "unknown",
        "artifact_count": 6,
        "chart_count": 5,
        "chart_files": ["coverage_by_year.png"],
    }
    assert schema_inventory.json()["columns"] == [
        {"name": "timestamp", "dtype": "timestamp[ns]", "role": "key"}
    ]
    assert schema_inventory.json()["missing_expected_columns"] == []
    combined = (
        factor_eda.text
        + schema.text
        + prediction_eda.text
        + schema_inventory.text
    )
    assert str(tmp_path) not in combined
    assert "credential" not in combined
    assert "api_key" not in combined
    assert "secret" not in combined
    assert "token" not in combined
    assert "vendor_raw_flag" not in combined
    assert "source_download_code" not in combined
    assert "local_path" not in combined
    assert "local_missing_path" not in combined
    assert "processed_private" not in combined
    assert "private_real" not in combined


def test_a_share_schema_inventory_endpoint_rejects_malformed_numeric_fields(
    tmp_path,
) -> None:
    prediction_root = (
        tmp_path / "reports" / "private_real" / "a_share_prediction_handoffs"
    )
    prediction_run = prediction_root / "bad_schema_inventory_numeric"
    prediction_run.mkdir(parents=True)
    (prediction_run / "schema_inventory.json").write_text(
        json.dumps(
            {
                "schema_inventory_version": "a_share_schema_inventory_v1",
                "row_count": "not_an_int",
                "row_group_count": 1,
                "columns": [],
            }
        ),
        encoding="utf-8",
    )
    client = TestClient(
        create_app(
            run_roots=[("platform", tmp_path / "runs")],
            prediction_handoff_root=prediction_root,
        )
    )

    response = client.get(
        "/api/a-share/prediction-handoffs/"
        "bad_schema_inventory_numeric/schema-inventory"
    )

    assert response.status_code == 422
    assert response.json()["error"]["error_code"] == (
        "prediction_schema_inventory_invalid"
    )
    assert "not_an_int" not in response.text


def test_a_share_eda_endpoints_reject_malformed_numeric_fields(tmp_path) -> None:
    factor_root = tmp_path / "reports" / "private_real" / "a_share_factor_rebuilds"
    factor_run = factor_root / "bad_factor_eda_numeric"
    (factor_run / "eda").mkdir(parents=True)
    (factor_run / "eda" / "eda_manifest.json").write_text(
        json.dumps(
            {
                "status": "ready",
                "artifact_count": "not_an_int",
                "chart_count": 1,
            }
        ),
        encoding="utf-8",
    )
    prediction_root = (
        tmp_path / "reports" / "private_real" / "a_share_prediction_handoffs"
    )
    prediction_run = prediction_root / "bad_prediction_eda_numeric"
    (prediction_run / "eda").mkdir(parents=True)
    (prediction_run / "eda" / "eda_manifest.json").write_text(
        json.dumps(
            {
                "status": "ready",
                "artifact_count": 1,
                "chart_count": "not_an_int",
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

    factor_response = client.get(
        "/api/a-share/factor-rebuild-runs/bad_factor_eda_numeric/eda"
    )
    prediction_response = client.get(
        "/api/a-share/prediction-handoffs/bad_prediction_eda_numeric/eda"
    )

    assert factor_response.status_code == 422
    assert prediction_response.status_code == 422
    assert factor_response.json()["error"]["error_code"] == "factor_rebuild_eda_invalid"
    assert prediction_response.json()["error"]["error_code"] == "prediction_eda_invalid"
    assert "not_an_int" not in factor_response.text
    assert "not_an_int" not in prediction_response.text


def test_a_share_prediction_handoff_status_reports_missing_summary(
    tmp_path,
) -> None:
    prediction_root = (
        tmp_path / "reports" / "private_real" / "a_share_prediction_handoffs"
    )
    (prediction_root / "missing_prediction_status").mkdir(parents=True)
    client = TestClient(
        create_app(
            run_roots=[("platform", tmp_path / "runs")],
            prediction_handoff_root=prediction_root,
        )
    )

    response = client.get(
        "/api/a-share/prediction-handoffs/missing_prediction_status"
    )

    assert response.status_code == 404
    assert response.json()["error"]["error_code"] == (
        "missing_prediction_handoff_status"
    )


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
                    "rev_1d": -(0.001 * idx + offset / 1000.0),
                    "mom_20d": 0.002 * idx + offset / 1000.0,
                    "eligible": True,
                    "can_enter": True,
                    "is_a_share_scope": True,
                    "forward_return_2d": 0.01 * idx + offset / 100.0,
                    "forward_return_20d": 0.02 * idx + offset / 100.0,
                    "forward_return_1d": 0.001 * idx,
                }
            )
    return pd.DataFrame(rows)


def _prediction_panel_without_literature_features() -> pd.DataFrame:
    frame = _prediction_panel()
    return frame.drop(columns=["rev_1d", "mom_20d"])
