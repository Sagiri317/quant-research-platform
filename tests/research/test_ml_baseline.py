from __future__ import annotations

import json

import pandas as pd
import pytest
from typer.testing import CliRunner

from quant_research_platform.interfaces.cli.commands import app
from quant_research_platform.research.ml_baseline import (
    BaselineTrainingConfig,
    run_baseline_training,
)


def test_baseline_training_writes_test_predictions_and_metadata(tmp_path) -> None:
    input_path = tmp_path / "features.parquet"
    prediction_path = tmp_path / "predictions.parquet"
    metadata_path = tmp_path / "metadata.json"
    _synthetic_panel().to_parquet(
        input_path,
        engine="pyarrow",
        compression="zstd",
        index=False,
    )

    result = run_baseline_training(
        BaselineTrainingConfig(
            input_path=input_path,
            output_prediction_path=prediction_path,
            output_metadata_path=metadata_path,
            timestamp_column="timestamp",
            entity_column="asset",
            feature_columns=("feature_a", "feature_b"),
            label_column="forward_return",
            train_end="2026-01-02",
            test_start="2026-01-03",
        )
    )

    predictions = pd.read_parquet(prediction_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    assert result.train_row_count == 4
    assert result.test_row_count == 4
    assert result.prediction_column == "prediction_score"
    assert prediction_path.is_file()
    assert metadata_path.is_file()
    assert set(predictions.columns) == {
        "timestamp",
        "asset",
        "prediction_score",
        "forward_return",
        "forward_return_1d",
    }
    assert predictions["timestamp"].min() == pd.Timestamp("2026-01-03")
    assert predictions["timestamp"].max() == pd.Timestamp("2026-01-04")
    assert predictions["prediction_score"].notna().all()
    assert metadata["model_type"] == "numpy_linear_ridge_baseline"
    assert metadata["feature_columns"] == ["feature_a", "feature_b"]
    assert metadata["label_column"] == "forward_return"
    assert metadata["train_row_count"] == 4
    assert metadata["test_row_count"] == 4
    assert len(metadata["coefficients"]) == 2
    assert isinstance(metadata["intercept"], float)
    assert "not investment advice" in metadata["boundary"]
    assert "not live trading evidence" in metadata["boundary"]


def test_baseline_training_rejects_overlapping_split(tmp_path) -> None:
    input_path = tmp_path / "features.parquet"
    _synthetic_panel().to_parquet(input_path, engine="pyarrow", index=False)

    with pytest.raises(ValueError, match="train_end must be before test_start"):
        run_baseline_training(
            BaselineTrainingConfig(
                input_path=input_path,
                output_prediction_path=tmp_path / "predictions.parquet",
                output_metadata_path=tmp_path / "metadata.json",
                timestamp_column="timestamp",
                entity_column="asset",
                feature_columns=("feature_a",),
                label_column="forward_return",
                train_end="2026-01-03",
                test_start="2026-01-03",
            )
        )


def test_baseline_training_rejects_missing_required_column(tmp_path) -> None:
    input_path = tmp_path / "features.parquet"
    _synthetic_panel().drop(columns=["feature_b"]).to_parquet(
        input_path,
        engine="pyarrow",
        index=False,
    )

    with pytest.raises(ValueError, match="missing required columns"):
        run_baseline_training(
            BaselineTrainingConfig(
                input_path=input_path,
                output_prediction_path=tmp_path / "predictions.parquet",
                output_metadata_path=tmp_path / "metadata.json",
                timestamp_column="timestamp",
                entity_column="asset",
                feature_columns=("feature_a", "feature_b"),
                label_column="forward_return",
                train_end="2026-01-02",
                test_start="2026-01-03",
            )
        )


def test_baseline_training_metadata_is_private_safe(tmp_path) -> None:
    input_path = tmp_path / "features.parquet"
    metadata_path = tmp_path / "metadata.json"
    _synthetic_panel().to_parquet(input_path, engine="pyarrow", index=False)

    run_baseline_training(
        BaselineTrainingConfig(
            input_path=input_path,
            output_prediction_path=tmp_path / "predictions.parquet",
            output_metadata_path=metadata_path,
            timestamp_column="timestamp",
            entity_column="asset",
            feature_columns=("feature_a", "feature_b"),
            label_column="forward_return",
            train_end="2026-01-02",
            test_start="2026-01-03",
        )
    )

    metadata_text = metadata_path.read_text(encoding="utf-8")
    for forbidden in (
        str(tmp_path),
        "raw_private",
        "interim_private",
        "processed_private",
        "private_real",
        "stock_sample",
        "row_sample",
        "sample_stocks",
    ):
        assert forbidden not in metadata_text


def test_generated_predictions_enter_existing_prediction_handoff(tmp_path) -> None:
    input_path = tmp_path / "features.parquet"
    prediction_path = tmp_path / "predictions.parquet"
    output_dir = tmp_path / "evaluation"
    _synthetic_panel().to_parquet(input_path, engine="pyarrow", index=False)
    run_baseline_training(
        BaselineTrainingConfig(
            input_path=input_path,
            output_prediction_path=prediction_path,
            output_metadata_path=tmp_path / "metadata.json",
            timestamp_column="timestamp",
            entity_column="asset",
            feature_columns=("feature_a", "feature_b"),
            label_column="forward_return",
            train_end="2026-01-02",
            test_start="2026-01-03",
        )
    )

    result = CliRunner().invoke(
        app,
        [
            "evaluate-parquet",
            "--parquet-path",
            str(prediction_path),
            "--feature",
            "prediction_score",
            "--label",
            "forward_return",
            "--return-source-column",
            "forward_return_1d",
            "--output",
            str(output_dir),
            "--run-id",
            "baseline_prediction_handoff",
            "--kind",
            "prediction",
            "--domain",
            "prediction_handoff",
            "--data-stage",
            "synthetic_only",
            "--no-uses-external-data",
            "--n-groups",
            "2",
            "--prediction-handoff-version",
            "n6_mvs_v1",
            "--prediction-source-ref",
            "synthetic_baseline_prediction_panel_v1",
            "--prediction-model-ref",
            "numpy_linear_ridge_baseline_v1",
        ],
    )

    assert result.exit_code == 0, result.output
    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    run_context = json.loads(
        (output_dir / "run_context.json").read_text(encoding="utf-8")
    )
    assert summary["artifact_schema_version"] == "0.7"
    assert summary["artifact_type"] == "evaluation_set"
    assert summary["kind"] == "prediction"
    assert run_context["config"]["evaluation_spec"]["metadata"][
        "training_scope"
    ] == "external_or_upstream_not_run_by_qrp"


def _synthetic_panel() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                [
                    "2026-01-01",
                    "2026-01-01",
                    "2026-01-02",
                    "2026-01-02",
                    "2026-01-03",
                    "2026-01-03",
                    "2026-01-04",
                    "2026-01-04",
                ]
            ),
            "asset": ["A", "B", "A", "B", "A", "B", "A", "B"],
            "feature_a": [1.0, 2.0, 2.0, 3.0, 3.0, 1.0, 4.0, 0.5],
            "feature_b": [0.0, 1.0, 0.5, 1.5, 0.25, 2.0, 0.75, 1.25],
            "forward_return": [0.02, -0.01, 0.03, -0.02, 0.04, -0.03, 0.02, -0.01],
            "forward_return_1d": [0.02, -0.01, 0.03, -0.02, 0.04, -0.03, 0.02, -0.01],
        }
    )
