from __future__ import annotations

import json

import pandas as pd
import pytest

from quant_research_platform.research.prediction_handoff import (
    PredictionHandoffConfig,
    _write_optional_charts,
    refresh_prediction_handoff_eda,
    run_prediction_handoff,
)


def test_prediction_handoff_writes_predictions_evaluation_and_comparison(
    tmp_path,
) -> None:
    input_path = tmp_path / "features.parquet"
    output_root = tmp_path / "reports" / "private_real" / "a_share_prediction_handoffs"
    _panel().to_parquet(input_path, engine="pyarrow", compression="zstd", index=False)

    result = run_prediction_handoff(
        PredictionHandoffConfig(
            input_path=input_path,
            output_root=output_root,
            run_id="prediction_service_run",
            timestamp_column="timestamp",
            entity_column="asset",
            feature_columns=("feature_a", "feature_b"),
            label_column="forward_return_2d",
            return_source_column="forward_return_1d",
            label_horizon_days=2,
            folds=(
                {
                    "fold_id": "fold_001",
                    "train_end": "2024-01-05",
                    "test_start": "2024-01-08",
                    "test_end": "2024-01-09",
                },
            ),
            model_ids=("ridge", "elastic_net"),
        )
    )

    run_dir = output_root / "prediction_service_run"
    assert result.status == "ready"
    assert (run_dir / "predictions.parquet").is_file()
    assert (run_dir / "prediction_records.parquet").is_file()
    assert (run_dir / "model_metadata.json").is_file()
    assert (run_dir / "checkpoints.json").is_file()
    assert (run_dir / "evaluation" / "manifest.json").is_file()
    assert (run_dir / "comparison" / "model_leaderboard.csv").is_file()
    assert (run_dir / "comparison" / "prediction_vs_single_factors.json").is_file()
    assert (run_dir / "eda" / "eda_manifest.json").is_file()
    assert (run_dir / "eda" / "data_coverage_by_month.csv").is_file()
    eda_manifest = json.loads(
        (run_dir / "eda" / "eda_manifest.json").read_text("utf-8")
    )
    assert eda_manifest["chart_dependency_status"] == "ready"
    assert eda_manifest["chart_count"] >= 5
    assert "coverage_by_year.png" in eda_manifest["chart_files"]
    assert "model_leaderboard_top20.png" in eda_manifest["chart_files"]
    assert (run_dir / "eda" / "coverage_by_year.png").is_file()
    assert (run_dir / "eda" / "model_leaderboard_top20.png").is_file()

    summary = json.loads((run_dir / "runtime_summary.json").read_text("utf-8"))
    eval_summary = json.loads(
        (run_dir / "evaluation" / "summary.json").read_text("utf-8")
    )
    run_context = json.loads(
        (run_dir / "evaluation" / "run_context.json").read_text("utf-8")
    )
    metadata = run_context["config"]["evaluation_spec"]["metadata"]
    model_metadata = json.loads((run_dir / "model_metadata.json").read_text("utf-8"))
    assert summary["status"] == "ready"
    assert eval_summary["kind"] == "prediction"
    assert metadata["prediction_handoff_version"] == "n6_a_share_prediction_handoff_v1"
    assert metadata["source_ref"] == "explicit_feature_columns"
    assert metadata["model_ref"] == "n6_a_share_sklearn_boosting_suite_v1"
    assert metadata["training_execution_scope"] == "local_qrp_research_layer"
    assert metadata["evaluation_training_boundary"] == "training_outside_evaluation"
    assert model_metadata["timestamp_column"] == "timestamp"
    assert model_metadata["entity_column"] == "asset"
    assert model_metadata["label_column"] == "forward_return_2d"
    assert model_metadata["return_source_column"] == "forward_return_1d"
    assert model_metadata["feature_columns"] == ["feature_a", "feature_b"]
    assert model_metadata["feature_preset"] is None
    assert model_metadata["model_ids"] == ["ridge", "elastic_net"]
    assert model_metadata["model_suite_id"] == "n6_a_share_sklearn_boosting_suite_v1"
    assert model_metadata["prediction_handoff_version"] == (
        "n6_a_share_prediction_handoff_v1"
    )
    assert set(model_metadata["prediction_columns"]) == {
        "prediction_score_ridge_2d",
        "prediction_score_elastic_net_2d",
    }
    assert model_metadata["label_horizon_days"] == 2
    assert set(metadata["prediction_columns"]) == {
        "prediction_score_ridge_2d",
        "prediction_score_elastic_net_2d",
    }
    records = pd.read_parquet(run_dir / "prediction_records.parquet")
    assert {"model_id", "target_horizon", "fold_id", "split_name"}.issubset(
        records.columns
    )
    assert str(tmp_path) not in json.dumps(summary)
    assert "row_sample" not in json.dumps(summary)
    assert "stock_sample" not in json.dumps(summary)


def test_refresh_prediction_handoff_eda_regenerates_static_charts(tmp_path) -> None:
    input_path = tmp_path / "features.parquet"
    output_root = tmp_path / "reports" / "private_real" / "a_share_prediction_handoffs"
    _panel().to_parquet(input_path, engine="pyarrow", compression="zstd", index=False)
    run_prediction_handoff(
        PredictionHandoffConfig(
            input_path=input_path,
            output_root=output_root,
            run_id="prediction_refresh_run",
            timestamp_column="timestamp",
            entity_column="asset",
            feature_columns=("feature_a", "feature_b"),
            label_column="forward_return_2d",
            return_source_column="forward_return_1d",
            label_horizon_days=2,
            folds=(
                {
                    "fold_id": "fold_001",
                    "train_end": "2024-01-05",
                    "test_start": "2024-01-08",
                    "test_end": "2024-01-09",
                },
            ),
            model_ids=("ridge", "elastic_net"),
        )
    )

    run_dir = output_root / "prediction_refresh_run"
    (run_dir / "eda" / "coverage_by_year.png").unlink()
    result = refresh_prediction_handoff_eda(run_dir)
    manifest = json.loads((run_dir / "eda" / "eda_manifest.json").read_text("utf-8"))

    assert result["chart_dependency_status"] == "ready"
    assert result["chart_count"] >= 5
    assert "coverage_by_year.png" in manifest["chart_files"]
    assert "prediction_ic_by_year.png" in manifest["chart_files"]
    assert "model_leaderboard_top20.png" in manifest["chart_files"]
    assert (run_dir / "eda" / "coverage_by_year.png").is_file()


def test_refresh_prediction_handoff_eda_recovers_from_predictions_with_metadata_columns(
    tmp_path,
) -> None:
    input_path = tmp_path / "features.parquet"
    output_root = tmp_path / "reports" / "private_real" / "a_share_prediction_handoffs"
    frame = _panel().rename(columns={"timestamp": "trade_date", "asset": "stock_code"})
    frame.to_parquet(input_path, engine="pyarrow", compression="zstd", index=False)
    run_prediction_handoff(
        PredictionHandoffConfig(
            input_path=input_path,
            output_root=output_root,
            run_id="prediction_refresh_trade_date_run",
            timestamp_column="trade_date",
            entity_column="stock_code",
            feature_columns=("feature_a", "feature_b"),
            feature_preset="a_share_literature_price_volume_preset_v1",
            label_column="forward_return_2d",
            return_source_column="forward_return_1d",
            label_horizon_days=2,
            folds=(
                {
                    "fold_id": "fold_001",
                    "train_end": "2024-01-05",
                    "test_start": "2024-01-08",
                    "test_end": "2024-01-09",
                },
            ),
            model_ids=("ridge", "random_forest_lite"),
        )
    )

    run_dir = output_root / "prediction_refresh_trade_date_run"
    metadata = json.loads((run_dir / "model_metadata.json").read_text("utf-8"))
    assert metadata["timestamp_column"] == "trade_date"
    assert metadata["entity_column"] == "stock_code"
    assert metadata["label_column"] == "forward_return_2d"
    assert metadata["return_source_column"] == "forward_return_1d"

    for filename in (
        "data_coverage_by_year.csv",
        "data_coverage_by_month.csv",
        "missingness_by_feature.csv",
        "prediction_ic_by_year.csv",
        "prediction_ic_by_month.csv",
    ):
        (run_dir / "eda" / filename).unlink()

    result = refresh_prediction_handoff_eda(run_dir)
    manifest = json.loads((run_dir / "eda" / "eda_manifest.json").read_text("utf-8"))

    assert (run_dir / "eda" / "data_coverage_by_year.csv").is_file()
    assert (run_dir / "eda" / "prediction_ic_by_year.csv").is_file()
    assert (run_dir / "eda" / "eda_report.md").is_file()
    assert result["chart_dependency_status"] in {"ready", "dependency_unavailable"}
    if result["chart_dependency_status"] == "ready":
        assert result["chart_count"] > 0
        assert manifest["chart_files"]
    else:
        assert result["chart_count"] == 0


def test_refresh_prediction_handoff_eda_rejects_public_run_root(tmp_path) -> None:
    run_dir = tmp_path / "reports" / "public" / "prediction_run"
    run_dir.mkdir(parents=True)
    with pytest.raises(ValueError, match="private_output_policy_violation"):
        refresh_prediction_handoff_eda(run_dir)


def test_refresh_prediction_handoff_eda_rejects_unapproved_private_marker_root(
    tmp_path,
) -> None:
    run_dir = tmp_path / "reports" / "private_tmp" / "prediction_run"
    run_dir.mkdir(parents=True)
    with pytest.raises(ValueError, match="private_output_policy_violation"):
        refresh_prediction_handoff_eda(run_dir)


def test_refresh_prediction_handoff_eda_rejects_traversal_out_of_private_root(
    tmp_path,
) -> None:
    run_dir = (
        tmp_path
        / "reports"
        / "private_real"
        / "a_share_prediction_handoffs"
        / ".."
        / ".."
        / "public"
        / "prediction_run"
    )
    with pytest.raises(ValueError, match="private_output_policy_violation"):
        refresh_prediction_handoff_eda(run_dir)


def test_prediction_handoff_dry_run_writes_no_row_level_outputs(tmp_path) -> None:
    input_path = tmp_path / "features.parquet"
    output_root = tmp_path / "reports" / "private_real" / "a_share_prediction_handoffs"
    _panel().to_parquet(input_path, engine="pyarrow", compression="zstd", index=False)

    result = run_prediction_handoff(
        PredictionHandoffConfig(
            input_path=input_path,
            output_root=output_root,
            run_id="prediction_dry_run",
            timestamp_column="timestamp",
            entity_column="asset",
            feature_columns=("feature_a", "feature_b"),
            label_column="forward_return_2d",
            return_source_column="forward_return_1d",
            label_horizon_days=2,
            folds=(
                {
                    "fold_id": "fold_001",
                    "train_end": "2024-01-05",
                    "test_start": "2024-01-08",
                    "test_end": "2024-01-09",
                },
            ),
            model_ids=("ridge",),
            dry_run=True,
        )
    )

    run_dir = output_root / "prediction_dry_run"
    assert result.status == "dry_run"
    assert (run_dir / "schema_inventory.json").is_file()
    assert (run_dir / "memory_preflight.json").is_file()
    assert (run_dir / "runtime_summary.json").is_file()
    assert (run_dir / "checkpoints.json").is_file()
    assert not (run_dir / "predictions.parquet").exists()
    assert not (run_dir / "prediction_records.parquet").exists()
    assert not (run_dir / "evaluation").exists()


def test_prediction_handoff_rejects_sensitive_feature_names_before_artifacts(
    tmp_path,
) -> None:
    input_path = tmp_path / "features.parquet"
    output_root = tmp_path / "reports" / "private_real" / "a_share_prediction_handoffs"
    _panel().assign(api_key_feature=1.0).to_parquet(
        input_path,
        engine="pyarrow",
        compression="zstd",
        index=False,
    )

    with pytest.raises(ValueError, match="invalid_feature_preset"):
        run_prediction_handoff(
            PredictionHandoffConfig(
                input_path=input_path,
                output_root=output_root,
                run_id="prediction_sensitive_feature",
                timestamp_column="timestamp",
                entity_column="asset",
                feature_columns=("api_key_feature",),
                label_column="forward_return_2d",
                return_source_column="forward_return_1d",
                label_horizon_days=2,
                folds=(
                    {
                        "fold_id": "fold_001",
                        "train_end": "2024-01-05",
                        "test_start": "2024-01-08",
                        "test_end": "2024-01-09",
                    },
                ),
                model_ids=("ridge",),
                dry_run=True,
            )
        )

    assert not (output_root / "prediction_sensitive_feature").exists()


@pytest.mark.parametrize(
    "feature_columns",
    [
        ("feature_a", "forward_return_20d"),
        ("feature_a", "label_score"),
        ("feature_a", "target_return"),
    ],
)
def test_prediction_handoff_rejects_leakage_feature_names_before_artifacts(
    tmp_path,
    feature_columns,
) -> None:
    input_path = tmp_path / "features.parquet"
    output_root = tmp_path / "reports" / "private_real" / "a_share_prediction_handoffs"
    _panel().assign(label_score=1.0, target_return=1.0).to_parquet(
        input_path,
        engine="pyarrow",
        compression="zstd",
        index=False,
    )

    with pytest.raises(ValueError, match="invalid_feature_preset"):
        run_prediction_handoff(
            PredictionHandoffConfig(
                input_path=input_path,
                output_root=output_root,
                run_id="prediction_leakage_feature",
                timestamp_column="timestamp",
                entity_column="asset",
                feature_columns=feature_columns,
                label_column="forward_return_2d",
                return_source_column="forward_return_1d",
                label_horizon_days=2,
                folds=(
                    {
                        "fold_id": "fold_001",
                        "train_end": "2024-01-05",
                        "test_start": "2024-01-08",
                        "test_end": "2024-01-09",
                    },
                ),
                model_ids=("ridge",),
            )
        )

    assert not (output_root / "prediction_leakage_feature").exists()


@pytest.mark.parametrize(
    "folds",
    [
        ({"fold_id": "fold_001", "test_start": "2024-01-08"},),
        ({"train_end": "2024-01-05", "test_start": "2024-01-08"},),
        ({"fold_id": "fold_001", "train_end": "2024-01-05"},),
        (
            {
                "fold_id": "",
                "train_end": "2024-01-05",
                "test_start": "2024-01-08",
                "test_end": "2024-01-09",
            },
        ),
        (
            {
                "fold_id": "fold_001",
                "train_end": "not-a-date",
                "test_start": "2024-01-08",
                "test_end": "2024-01-09",
            },
        ),
        (
            {
                "fold_id": "fold_001",
                "train_end": "2024-01-05",
                "test_start": "2024-01-08",
            },
            {
                "fold_id": "fold_002",
                "train_end": "2024-01-06",
                "test_start": "2024-01-09",
                "test_end": "2024-01-10",
            },
        ),
        (
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
        ),
        (
            {
                "fold_id": "fold_001",
                "train_end": "2024-01-05",
                "test_start": "2024-01-08",
                "test_end": "2024-01-09",
            },
            {
                "fold_id": "fold_001",
                "train_end": "2024-01-06",
                "test_start": "2024-01-10",
                "test_end": "2024-01-10",
            },
        ),
        (
            {
                "fold_id": "fold_001",
                "train_end": "2024-01-05",
                "test_start": "2024-01-08",
                "test_end": "2024-01-09",
            },
            {
                "fold_id": "fold_002",
                "train_end": "2024-01-06",
                "test_start": "2024-01-08",
                "test_end": "2024-01-09",
            },
        ),
    ],
)
def test_prediction_handoff_rejects_malformed_folds_before_artifacts(
    tmp_path,
    folds,
) -> None:
    input_path = tmp_path / "features.parquet"
    output_root = tmp_path / "reports" / "private_real" / "a_share_prediction_handoffs"
    _panel().to_parquet(input_path, engine="pyarrow", compression="zstd", index=False)

    with pytest.raises(ValueError, match="invalid_prediction_config") as exc:
        run_prediction_handoff(
            PredictionHandoffConfig(
                input_path=input_path,
                output_root=output_root,
                run_id="prediction_bad_folds",
                timestamp_column="timestamp",
                entity_column="asset",
                feature_columns=("feature_a", "feature_b"),
                label_column="forward_return_2d",
                return_source_column="forward_return_1d",
                label_horizon_days=2,
                folds=folds,
                model_ids=("ridge",),
            )
        )

    assert "fold_id" not in str(exc.value)
    assert "train_end" not in str(exc.value)
    assert "test_start" not in str(exc.value)
    assert not (output_root / "prediction_bad_folds").exists()


def test_prediction_handoff_rejects_duplicate_run_id_before_input_read(
    tmp_path,
) -> None:
    input_path = tmp_path / "missing_features.parquet"
    output_root = tmp_path / "reports" / "private_real" / "a_share_prediction_handoffs"
    (output_root / "prediction_duplicate").mkdir(parents=True)

    with pytest.raises(ValueError, match="duplicate_run_id") as exc:
        run_prediction_handoff(
            PredictionHandoffConfig(
                input_path=input_path,
                output_root=output_root,
                run_id="prediction_duplicate",
                timestamp_column="timestamp",
                entity_column="asset",
                feature_columns=("feature_a", "feature_b"),
                label_column="forward_return_2d",
                return_source_column="forward_return_1d",
                label_horizon_days=2,
                folds=(
                    {
                        "fold_id": "fold_001",
                        "train_end": "2024-01-05",
                        "test_start": "2024-01-08",
                        "test_end": "2024-01-09",
                    },
                ),
                model_ids=("ridge",),
            )
        )

    assert "missing_features.parquet" not in str(exc.value)


def test_prediction_handoff_rejects_unsupported_feature_preset(tmp_path) -> None:
    input_path = tmp_path / "features.parquet"
    output_root = tmp_path / "reports" / "private_real" / "a_share_prediction_handoffs"
    _panel().to_parquet(input_path, engine="pyarrow", compression="zstd", index=False)

    with pytest.raises(ValueError, match="invalid_feature_preset"):
        run_prediction_handoff(
            PredictionHandoffConfig(
                input_path=input_path,
                output_root=output_root,
                run_id="prediction_invalid_preset",
                timestamp_column="timestamp",
                entity_column="asset",
                feature_columns=("feature_a", "feature_b"),
                feature_preset="not/a/supported/preset",
                label_column="forward_return_2d",
                return_source_column="forward_return_1d",
                label_horizon_days=2,
                folds=(
                    {
                        "fold_id": "fold_001",
                        "train_end": "2024-01-05",
                        "test_start": "2024-01-08",
                        "test_end": "2024-01-09",
                    },
                ),
                model_ids=("ridge",),
            )
        )


def test_prediction_handoff_rejects_private_input_to_public_output_before_run_dir(
    tmp_path,
) -> None:
    input_path = (
        tmp_path / "reports" / "private_real" / "a_share_factor_rebuilds" / "x.parquet"
    )
    input_path.parent.mkdir(parents=True)
    _panel().to_parquet(input_path, engine="pyarrow", compression="zstd", index=False)
    public_output = tmp_path / "reports" / "public_prediction_handoffs"

    with pytest.raises(ValueError, match="private_output_policy_violation"):
        run_prediction_handoff(
            PredictionHandoffConfig(
                input_path=input_path,
                output_root=public_output,
                run_id="public_output_blocked",
                timestamp_column="timestamp",
                entity_column="asset",
                feature_columns=("feature_a", "feature_b"),
                label_column="forward_return_2d",
                return_source_column="forward_return_1d",
                label_horizon_days=2,
                folds=(
                    {
                        "fold_id": "fold_001",
                        "train_end": "2024-01-05",
                        "test_start": "2024-01-08",
                        "test_end": "2024-01-09",
                    },
                ),
                model_ids=("ridge",),
            )
        )

    assert not public_output.exists()


def test_prediction_handoff_chart_dependency_failure_does_not_crash(
    tmp_path,
    monkeypatch,
) -> None:
    import builtins

    original_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "matplotlib" or name.startswith("matplotlib."):
            raise OSError("DLL load failed while importing _imaging")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    chart_count, status, chart_files = _write_optional_charts(
        tmp_path,
        pd.DataFrame({"year": [2024], "row_count": [3]}),
        pd.DataFrame({"feature": ["rev_1d"], "missing_fraction": [0.0]}),
    )

    assert chart_count == 0
    assert status == "dependency_unavailable"
    assert chart_files == ()
    assert not (tmp_path / "coverage_by_year.png").exists()


def _panel() -> pd.DataFrame:
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
