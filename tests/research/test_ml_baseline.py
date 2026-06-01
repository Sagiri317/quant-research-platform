from __future__ import annotations

import json

import pandas as pd
import pytest
from typer.testing import CliRunner

from quant_research_platform.interfaces.cli import commands as commands_module
from quant_research_platform.interfaces.cli.commands import app
from quant_research_platform.research.ml_baseline import (
    BaselineFoldConfig,
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


def test_baseline_training_records_oos_split_embargo_and_fold(tmp_path) -> None:
    input_path = tmp_path / "features.parquet"
    prediction_path = tmp_path / "predictions.parquet"
    metadata_path = tmp_path / "metadata.json"
    _synthetic_panel().to_parquet(input_path, engine="pyarrow", index=False)

    run_baseline_training(
        BaselineTrainingConfig(
            input_path=input_path,
            output_prediction_path=prediction_path,
            output_metadata_path=metadata_path,
            timestamp_column="timestamp",
            entity_column="asset",
            feature_columns=("feature_a", "feature_b"),
            label_column="forward_return",
            train_end="2026-01-01",
            test_start="2026-01-03",
            test_end="2026-01-03",
            embargo_days=1,
            fold_id="fold_001",
            split_name="test",
        )
    )

    predictions = pd.read_parquet(prediction_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    assert predictions["timestamp"].min() == pd.Timestamp("2026-01-03")
    assert predictions["timestamp"].max() == pd.Timestamp("2026-01-03")
    assert predictions["split_name"].tolist() == ["test", "test"]
    assert predictions["fold_id"].tolist() == ["fold_001", "fold_001"]
    assert metadata["train_end"] == "2026-01-01"
    assert metadata["test_start"] == "2026-01-03"
    assert metadata["test_end"] == "2026-01-03"
    assert metadata["embargo_days"] == 1
    assert metadata["fold_id"] == "fold_001"
    assert metadata["split_name"] == "test"
    assert metadata["training_time"]["train_end"] == "2026-01-01"
    assert metadata["prediction_time"]["test_start"] == "2026-01-03"
    assert metadata["feature_summary"] == {
        "columns": ["feature_a", "feature_b"],
        "count": 2,
    }
    assert metadata["target_column"] == "forward_return"


def test_baseline_training_writes_walk_forward_predictions_and_metadata(
    tmp_path,
) -> None:
    input_path = tmp_path / "features.parquet"
    prediction_path = tmp_path / "predictions.parquet"
    metadata_path = tmp_path / "metadata.json"
    _synthetic_panel().to_parquet(input_path, engine="pyarrow", index=False)

    result = run_baseline_training(
        BaselineTrainingConfig(
            input_path=input_path,
            output_prediction_path=prediction_path,
            output_metadata_path=metadata_path,
            timestamp_column="timestamp",
            entity_column="asset",
            feature_columns=("feature_a", "feature_b"),
            label_column="forward_return",
            folds=(
                BaselineFoldConfig(
                    fold_id="fold_001",
                    train_end="2026-01-01",
                    test_start="2026-01-02",
                    test_end="2026-01-02",
                    split_name="walk_forward",
                ),
                BaselineFoldConfig(
                    fold_id="fold_002",
                    train_end="2026-01-02",
                    test_start="2026-01-03",
                    test_end="2026-01-03",
                    split_name="walk_forward",
                ),
            ),
        )
    )

    predictions = pd.read_parquet(prediction_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    assert result.train_row_count == 6
    assert result.test_row_count == 4
    assert predictions["timestamp"].tolist() == [
        pd.Timestamp("2026-01-02"),
        pd.Timestamp("2026-01-02"),
        pd.Timestamp("2026-01-03"),
        pd.Timestamp("2026-01-03"),
    ]
    assert predictions["fold_id"].tolist() == [
        "fold_001",
        "fold_001",
        "fold_002",
        "fold_002",
    ]
    assert set(predictions["split_name"]) == {"walk_forward"}
    assert predictions["prediction_score"].notna().all()
    assert metadata["fold_count"] == 2
    assert metadata["test_row_count"] == 4
    assert metadata["folds"] == [
        {
            "fold_id": "fold_001",
            "split_name": "walk_forward",
            "train_end": "2026-01-01",
            "test_start": "2026-01-02",
            "test_end": "2026-01-02",
            "train_row_count": 2,
            "test_row_count": 2,
        },
        {
            "fold_id": "fold_002",
            "split_name": "walk_forward",
            "train_end": "2026-01-02",
            "test_start": "2026-01-03",
            "test_end": "2026-01-03",
            "train_row_count": 4,
            "test_row_count": 2,
        },
    ]
    assert metadata["training_time"]["folds"] == metadata["folds"]
    assert metadata["prediction_time"]["folds"] == metadata["folds"]


def test_baseline_training_rejects_overlapping_walk_forward_test_windows(
    tmp_path,
) -> None:
    input_path = tmp_path / "features.parquet"
    _synthetic_panel().to_parquet(input_path, engine="pyarrow", index=False)

    with pytest.raises(ValueError, match="walk-forward test windows must not overlap"):
        run_baseline_training(
            BaselineTrainingConfig(
                input_path=input_path,
                output_prediction_path=tmp_path / "predictions.parquet",
                output_metadata_path=tmp_path / "metadata.json",
                timestamp_column="timestamp",
                entity_column="asset",
                feature_columns=("feature_a", "feature_b"),
                label_column="forward_return",
                folds=(
                    BaselineFoldConfig(
                        fold_id="fold_001",
                        train_end="2026-01-01",
                        test_start="2026-01-02",
                        test_end="2026-01-03",
                    ),
                    BaselineFoldConfig(
                        fold_id="fold_002",
                        train_end="2026-01-02",
                        test_start="2026-01-03",
                        test_end="2026-01-04",
                    ),
                ),
            )
        )


def test_baseline_training_rejects_non_final_open_ended_walk_forward_fold(
    tmp_path,
) -> None:
    input_path = tmp_path / "features.parquet"
    prediction_path = tmp_path / "predictions.parquet"
    metadata_path = tmp_path / "metadata.json"
    _synthetic_panel().to_parquet(input_path, engine="pyarrow", index=False)

    with pytest.raises(
        ValueError,
        match="open-ended walk-forward test window must be the final fold",
    ):
        run_baseline_training(
            BaselineTrainingConfig(
                input_path=input_path,
                output_prediction_path=prediction_path,
                output_metadata_path=metadata_path,
                timestamp_column="timestamp",
                entity_column="asset",
                feature_columns=("feature_a", "feature_b"),
                label_column="forward_return",
                folds=(
                    BaselineFoldConfig(
                        fold_id="fold_001",
                        train_end="2026-01-01",
                        test_start="2026-01-02",
                    ),
                    BaselineFoldConfig(
                        fold_id="fold_002",
                        train_end="2026-01-02",
                        test_start="2026-01-03",
                    ),
                ),
            )
        )

    assert not prediction_path.exists()
    assert not metadata_path.exists()


def test_baseline_training_allows_final_open_ended_walk_forward_fold(tmp_path) -> None:
    input_path = tmp_path / "features.parquet"
    prediction_path = tmp_path / "predictions.parquet"
    metadata_path = tmp_path / "metadata.json"
    _synthetic_panel().to_parquet(input_path, engine="pyarrow", index=False)

    run_baseline_training(
        BaselineTrainingConfig(
            input_path=input_path,
            output_prediction_path=prediction_path,
            output_metadata_path=metadata_path,
            timestamp_column="timestamp",
            entity_column="asset",
            feature_columns=("feature_a", "feature_b"),
            label_column="forward_return",
            folds=(
                BaselineFoldConfig(
                    fold_id="fold_001",
                    train_end="2026-01-01",
                    test_start="2026-01-02",
                    test_end="2026-01-02",
                ),
                BaselineFoldConfig(
                    fold_id="fold_002",
                    train_end="2026-01-02",
                    test_start="2026-01-03",
                ),
            ),
        )
    )

    predictions = pd.read_parquet(prediction_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert predictions["timestamp"].tolist() == [
        pd.Timestamp("2026-01-02"),
        pd.Timestamp("2026-01-02"),
        pd.Timestamp("2026-01-03"),
        pd.Timestamp("2026-01-03"),
        pd.Timestamp("2026-01-04"),
        pd.Timestamp("2026-01-04"),
    ]
    assert predictions["fold_id"].tolist() == [
        "fold_001",
        "fold_001",
        "fold_002",
        "fold_002",
        "fold_002",
        "fold_002",
    ]
    assert metadata["folds"][1]["test_end"] is None
    assert metadata["folds"][1]["test_row_count"] == 4


def test_baseline_training_rejects_mixed_top_level_split_and_walk_forward_folds(
    tmp_path,
) -> None:
    input_path = tmp_path / "features.parquet"
    prediction_path = tmp_path / "predictions.parquet"
    metadata_path = tmp_path / "metadata.json"
    _synthetic_panel().to_parquet(input_path, engine="pyarrow", index=False)

    with pytest.raises(
        ValueError,
        match="top-level split parameters must not be provided with walk-forward folds",
    ):
        run_baseline_training(
            BaselineTrainingConfig(
                input_path=input_path,
                output_prediction_path=prediction_path,
                output_metadata_path=metadata_path,
                timestamp_column="timestamp",
                entity_column="asset",
                feature_columns=("feature_a", "feature_b"),
                label_column="forward_return",
                train_end="2026-01-01",
                test_start="2026-01-02",
                folds=(
                    BaselineFoldConfig(
                        fold_id="fold_001",
                        train_end="2026-01-01",
                        test_start="2026-01-02",
                    ),
                ),
            )
        )

    assert not prediction_path.exists()
    assert not metadata_path.exists()


def test_baseline_training_rejects_duplicate_walk_forward_fold_ids(tmp_path) -> None:
    input_path = tmp_path / "features.parquet"
    _synthetic_panel().to_parquet(input_path, engine="pyarrow", index=False)

    with pytest.raises(ValueError, match="walk-forward fold_id values must be unique"):
        run_baseline_training(
            BaselineTrainingConfig(
                input_path=input_path,
                output_prediction_path=tmp_path / "predictions.parquet",
                output_metadata_path=tmp_path / "metadata.json",
                timestamp_column="timestamp",
                entity_column="asset",
                feature_columns=("feature_a", "feature_b"),
                label_column="forward_return",
                folds=(
                    BaselineFoldConfig(
                        fold_id="fold_001",
                        train_end="2026-01-01",
                        test_start="2026-01-02",
                    ),
                    BaselineFoldConfig(
                        fold_id="fold_001",
                        train_end="2026-01-02",
                        test_start="2026-01-03",
                    ),
                ),
            )
        )


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


def test_baseline_training_rejects_test_end_before_test_start(tmp_path) -> None:
    input_path = tmp_path / "features.parquet"
    _synthetic_panel().to_parquet(input_path, engine="pyarrow", index=False)

    with pytest.raises(ValueError, match="test_end must be on or after test_start"):
        run_baseline_training(
            BaselineTrainingConfig(
                input_path=input_path,
                output_prediction_path=tmp_path / "predictions.parquet",
                output_metadata_path=tmp_path / "metadata.json",
                timestamp_column="timestamp",
                entity_column="asset",
                feature_columns=("feature_a",),
                label_column="forward_return",
                train_end="2026-01-01",
                test_start="2026-01-03",
                test_end="2026-01-02",
            )
        )


def test_baseline_training_rejects_embargo_overlap(tmp_path) -> None:
    input_path = tmp_path / "features.parquet"
    _synthetic_panel().to_parquet(input_path, engine="pyarrow", index=False)

    with pytest.raises(
        ValueError,
        match="train_end plus embargo_days must be on or before test_start",
    ):
        run_baseline_training(
            BaselineTrainingConfig(
                input_path=input_path,
                output_prediction_path=tmp_path / "predictions.parquet",
                output_metadata_path=tmp_path / "metadata.json",
                timestamp_column="timestamp",
                entity_column="asset",
                feature_columns=("feature_a",),
                label_column="forward_return",
                train_end="2026-01-02",
                test_start="2026-01-03",
                embargo_days=2,
            )
        )


def test_baseline_training_rejects_invalid_split_identifiers(tmp_path) -> None:
    input_path = tmp_path / "features.parquet"
    _synthetic_panel().to_parquet(input_path, engine="pyarrow", index=False)

    with pytest.raises(ValueError, match="fold_id must be a portable identifier"):
        run_baseline_training(
            BaselineTrainingConfig(
                input_path=input_path,
                output_prediction_path=tmp_path / "predictions.parquet",
                output_metadata_path=tmp_path / "metadata.json",
                timestamp_column="timestamp",
                entity_column="asset",
                feature_columns=("feature_a",),
                label_column="forward_return",
                train_end="2026-01-01",
                test_start="2026-01-03",
                fold_id="../fold",
            )
        )


def test_baseline_training_rejects_forward_return_feature_leakage(tmp_path) -> None:
    input_path = tmp_path / "features.parquet"
    _synthetic_panel().to_parquet(input_path, engine="pyarrow", index=False)

    with pytest.raises(ValueError, match="feature_columns must not include"):
        run_baseline_training(
            BaselineTrainingConfig(
                input_path=input_path,
                output_prediction_path=tmp_path / "predictions.parquet",
                output_metadata_path=tmp_path / "metadata.json",
                timestamp_column="timestamp",
                entity_column="asset",
                feature_columns=("feature_a", "forward_return_1d"),
                label_column="forward_return",
                train_end="2026-01-01",
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
        str(tmp_path).replace("\\", "/"),
        "raw_private",
        "interim_private",
        "processed_private",
        "private_real",
        "stock_sample",
        "row_sample",
        "sample_stocks",
    ):
        assert forbidden not in metadata_text


def test_baseline_training_rejects_private_input_with_ordinary_outputs(
    tmp_path,
) -> None:
    private_dir = tmp_path / "vendor_private" / "inputs"
    private_dir.mkdir(parents=True)
    input_path = private_dir / "features.parquet"
    prediction_path = tmp_path / "public_outputs" / "predictions.parquet"
    metadata_path = tmp_path / "public_outputs" / "metadata.json"
    _synthetic_panel().to_parquet(input_path, engine="pyarrow", index=False)

    with pytest.raises(ValueError, match="private input policy"):
        run_baseline_training(
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

    assert not prediction_path.exists()
    assert not metadata_path.exists()


def test_walk_forward_rejects_private_input_with_ordinary_outputs(tmp_path) -> None:
    private_dir = tmp_path / "vendor_private" / "inputs"
    private_dir.mkdir(parents=True)
    input_path = private_dir / "features.parquet"
    prediction_path = tmp_path / "public_outputs" / "predictions.parquet"
    metadata_path = tmp_path / "public_outputs" / "metadata.json"
    _synthetic_panel().to_parquet(input_path, engine="pyarrow", index=False)

    with pytest.raises(ValueError, match="private input policy"):
        run_baseline_training(
            BaselineTrainingConfig(
                input_path=input_path,
                output_prediction_path=prediction_path,
                output_metadata_path=metadata_path,
                timestamp_column="timestamp",
                entity_column="asset",
                feature_columns=("feature_a", "feature_b"),
                label_column="forward_return",
                folds=(
                    BaselineFoldConfig(
                        fold_id="fold_001",
                        train_end="2026-01-01",
                        test_start="2026-01-02",
                    ),
                ),
            )
        )

    assert not prediction_path.exists()
    assert not metadata_path.exists()


def test_baseline_training_allows_private_input_with_private_outputs(tmp_path) -> None:
    private_dir = tmp_path / "vendor_private" / "inputs"
    output_dir = tmp_path / "private_outputs"
    private_dir.mkdir(parents=True)
    input_path = private_dir / "features.parquet"
    prediction_path = output_dir / "predictions.parquet"
    metadata_path = output_dir / "metadata.json"
    _synthetic_panel().to_parquet(input_path, engine="pyarrow", index=False)

    run_baseline_training(
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

    assert prediction_path.is_file()
    assert metadata_path.is_file()
    metadata_text = metadata_path.read_text(encoding="utf-8")
    for forbidden in (
        str(tmp_path),
        str(tmp_path).replace("\\", "/"),
        "vendor_private",
        "private_outputs",
        "features.parquet",
        "predictions.parquet",
        "metadata.json",
        "row_sample",
        "stock_sample",
    ):
        assert forbidden not in metadata_text


def test_ml_baseline_cli_rejects_private_input_with_ordinary_outputs_before_schema_read(
    tmp_path,
    monkeypatch,
) -> None:
    private_dir = tmp_path / "vendor_private" / "inputs"
    private_dir.mkdir(parents=True)
    input_path = private_dir / "features.parquet"
    prediction_path = tmp_path / "public_outputs" / "predictions.parquet"
    metadata_path = tmp_path / "public_outputs" / "metadata.json"
    _synthetic_panel().to_parquet(input_path, engine="pyarrow", index=False)
    read_schema_calls = []

    def fail_read_schema(path):
        read_schema_calls.append(path)
        raise AssertionError("schema read should happen after private output policy")

    monkeypatch.setattr(commands_module.pq, "read_schema", fail_read_schema)

    result = CliRunner().invoke(
        app,
        _baseline_cli_args(
            input_path,
            prediction_path,
            metadata_path,
        ),
    )

    assert result.exit_code == 2
    assert read_schema_calls == []
    assert not prediction_path.exists()
    assert not metadata_path.exists()
    assert "private input policy" in result.output
    for forbidden in (
        str(input_path),
        str(input_path).replace("\\", "/"),
        "vendor_private",
        "features.parquet",
    ):
        assert forbidden not in result.output


def test_ml_baseline_cli_rejects_private_input_with_explicit_features(
    tmp_path,
) -> None:
    private_dir = tmp_path / "vendor_private" / "inputs"
    private_dir.mkdir(parents=True)
    input_path = private_dir / "features.parquet"
    prediction_path = tmp_path / "public_outputs" / "predictions.parquet"
    metadata_path = tmp_path / "public_outputs" / "metadata.json"
    _synthetic_panel().to_parquet(input_path, engine="pyarrow", index=False)

    result = CliRunner().invoke(
        app,
        _baseline_cli_args(
            input_path,
            prediction_path,
            metadata_path,
            feature_columns=("feature_a", "feature_b"),
        ),
    )

    assert result.exit_code == 2
    assert not prediction_path.exists()
    assert not metadata_path.exists()
    assert "private input policy" in result.output
    assert "vendor_private" not in result.output


def test_ml_baseline_cli_allows_private_input_with_private_outputs(tmp_path) -> None:
    private_dir = tmp_path / "vendor_private" / "inputs"
    output_dir = tmp_path / "private_outputs"
    private_dir.mkdir(parents=True)
    input_path = private_dir / "features.parquet"
    prediction_path = output_dir / "predictions.parquet"
    metadata_path = output_dir / "metadata.json"
    _synthetic_panel().to_parquet(input_path, engine="pyarrow", index=False)

    result = CliRunner().invoke(
        app,
        _baseline_cli_args(
            input_path,
            prediction_path,
            metadata_path,
        ),
    )

    assert result.exit_code == 0, result.output
    assert prediction_path.is_file()
    assert metadata_path.is_file()
    assert "private output files" in result.output


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


def test_generated_predictions_support_split_segment_handoff(tmp_path) -> None:
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
            train_end="2026-01-01",
            test_start="2026-01-03",
            split_name="oos_test",
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
            "--segment-column",
            "split_name",
            "--output",
            str(output_dir),
            "--run-id",
            "baseline_prediction_segment_handoff",
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
    ic = pd.read_csv(output_dir / "metrics" / "ic.csv")
    assert "split_name" in ic.columns
    assert set(ic["split_name"]) == {"oos_test"}


def test_baseline_training_outputs_are_deterministic(tmp_path) -> None:
    input_path = tmp_path / "features.parquet"
    _synthetic_panel().to_parquet(input_path, engine="pyarrow", index=False)
    config_kwargs = {
        "input_path": input_path,
        "timestamp_column": "timestamp",
        "entity_column": "asset",
        "feature_columns": ("feature_a", "feature_b"),
        "label_column": "forward_return",
        "train_end": "2026-01-01",
        "test_start": "2026-01-03",
        "test_end": "2026-01-04",
        "embargo_days": 1,
        "fold_id": "fold_001",
        "split_name": "test",
    }

    run_baseline_training(
        BaselineTrainingConfig(
            output_prediction_path=tmp_path / "predictions_a.parquet",
            output_metadata_path=tmp_path / "metadata_a.json",
            **config_kwargs,
        )
    )
    run_baseline_training(
        BaselineTrainingConfig(
            output_prediction_path=tmp_path / "predictions_b.parquet",
            output_metadata_path=tmp_path / "metadata_b.json",
            **config_kwargs,
        )
    )

    pd.testing.assert_frame_equal(
        pd.read_parquet(tmp_path / "predictions_a.parquet"),
        pd.read_parquet(tmp_path / "predictions_b.parquet"),
    )
    assert (tmp_path / "metadata_a.json").read_text(encoding="utf-8") == (
        tmp_path / "metadata_b.json"
    ).read_text(encoding="utf-8")


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


def _baseline_cli_args(
    input_path,
    prediction_path,
    metadata_path,
    *,
    feature_columns=(),
) -> list[str]:
    args = [
        "ml",
        "baseline",
        "train",
        "--input-parquet",
        str(input_path),
        "--output-prediction",
        str(prediction_path),
        "--output-metadata",
        str(metadata_path),
        "--target-column",
        "forward_return",
        "--train-end",
        "2026-01-02",
        "--test-start",
        "2026-01-03",
    ]
    for column in feature_columns:
        args.extend(["--feature-column", column])
    return args
