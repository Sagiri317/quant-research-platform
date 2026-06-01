from __future__ import annotations

import json

import pandas as pd
import pytest

from quant_research_platform.research.prediction_handoff import (
    PredictionHandoffConfig,
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

    summary = json.loads((run_dir / "runtime_summary.json").read_text("utf-8"))
    eval_summary = json.loads(
        (run_dir / "evaluation" / "summary.json").read_text("utf-8")
    )
    run_context = json.loads(
        (run_dir / "evaluation" / "run_context.json").read_text("utf-8")
    )
    metadata = run_context["config"]["evaluation_spec"]["metadata"]
    assert summary["status"] == "ready"
    assert eval_summary["kind"] == "prediction"
    assert metadata["prediction_handoff_version"] == "n6_a_share_prediction_handoff_v1"
    assert metadata["source_ref"] == "a_share_literature_price_volume_preset_v1"
    assert metadata["model_ref"] == "n6_a_share_sklearn_boosting_suite_v1"
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
