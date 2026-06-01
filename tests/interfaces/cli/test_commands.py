from __future__ import annotations

import json

import pandas as pd
from typer.testing import CliRunner

from quant_research_platform.interfaces.cli.commands import app


def test_cli_package_exports_main_entrypoint() -> None:
    from quant_research_platform.interfaces.cli import main

    assert callable(main)


def test_cli_exposes_a_share_group() -> None:
    runner = CliRunner()

    result = runner.invoke(app, ["a-share", "--help"])

    assert result.exit_code == 0
    assert "build-factors" in result.output
    assert "evaluate" in result.output


def test_cli_exposes_ml_baseline_train() -> None:
    runner = CliRunner()

    result = runner.invoke(app, ["ml", "baseline", "--help"])

    assert result.exit_code == 0
    assert "train" in result.output


def test_evaluate_parquet_rejects_multiple_costs_with_schema_07_wording() -> None:
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "evaluate-parquet",
            "--parquet-path",
            "panel.parquet",
            "--feature",
            "factor",
            "--label",
            "forward_return",
            "--cost-bps",
            "1",
            "--cost-bps",
            "2",
        ],
    )

    assert result.exit_code == 2
    assert "schema 0.7" in result.output
    assert "schema 0.6" not in result.output


def test_evaluate_parquet_rejects_unknown_diagnostic_view() -> None:
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "evaluate-parquet",
            "--parquet-path",
            "panel.parquet",
            "--feature",
            "factor",
            "--label",
            "forward_return",
            "--diagnostics-enabled",
            "--diagnostic-view",
            "capacity_curve",
        ],
    )

    assert result.exit_code == 2
    assert "unknown diagnostics materialized views" in result.output


def test_evaluate_parquet_writes_prediction_handoff_metadata(tmp_path) -> None:
    panel_path = tmp_path / "prediction_panel.parquet"
    output_dir = tmp_path / "prediction_run"
    pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                ["2026-01-01", "2026-01-01", "2026-01-02", "2026-01-02"]
            ),
            "asset": ["A", "B", "A", "B"],
            "prediction_score": [0.1, 0.2, 0.3, 0.1],
            "forward_return": [0.01, -0.02, 0.03, -0.01],
            "forward_return_1d": [0.01, -0.02, 0.03, -0.01],
        }
    ).to_parquet(panel_path, engine="pyarrow", compression="zstd", index=False)
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "evaluate-parquet",
            "--parquet-path",
            str(panel_path),
            "--feature",
            "prediction_score",
            "--label",
            "forward_return",
            "--return-source-column",
            "forward_return_1d",
            "--output",
            str(output_dir),
            "--run-id",
            "prediction_cli_run",
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
            "synthetic_prediction_panel_v1",
            "--prediction-model-ref",
            "opaque_model_ref_v1",
        ],
    )

    assert result.exit_code == 0, result.output
    assert (output_dir / "summary.json").is_file()
    assert (output_dir / "run_context.json").is_file()
    assert (output_dir / "manifest.json").is_file()
    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    run_context = json.loads(
        (output_dir / "run_context.json").read_text(encoding="utf-8")
    )
    persisted = run_context["config"]["evaluation_spec"]["metadata"]
    assert summary["artifact_schema_version"] == "0.7"
    assert summary["artifact_type"] == "evaluation_set"
    assert manifest["artifact_schema_version"] == "0.7"
    assert manifest["artifact_type"] == "evaluation_set"
    assert summary["kind"] == "prediction"
    assert persisted["prediction_handoff_version"] == "n6_mvs_v1"
    assert persisted["prediction_columns"] == ["prediction_score"]
    assert persisted["label_columns"] == ["forward_return"]
    assert persisted["prediction_time_semantics"] == {
        "decision_time_column": "timestamp",
        "as_of_time_columns": {"prediction_score": "timestamp"},
    }
    assert persisted["source_ref"] == "synthetic_prediction_panel_v1"
    assert persisted["model_ref"] == "opaque_model_ref_v1"
    assert persisted["training_scope"] == "external_or_upstream_not_run_by_qrp"
    assert "not trading" in persisted["prediction_boundary"]
    assert "paper trading" in persisted["prediction_boundary"]
    assert "live trading" in persisted["prediction_boundary"]
    persisted_text = json.dumps(persisted)
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
        assert forbidden not in persisted_text


def test_ml_baseline_train_cli_writes_predictions_and_metadata(tmp_path) -> None:
    input_path = tmp_path / "features.parquet"
    prediction_path = tmp_path / "predictions.parquet"
    metadata_path = tmp_path / "metadata.json"
    evaluation_output = tmp_path / "evaluation"
    pd.DataFrame(
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
    ).to_parquet(input_path, engine="pyarrow", compression="zstd", index=False)
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
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
            "--feature-column",
            "feature_a",
            "--feature-column",
            "feature_b",
            "--train-end",
            "2026-01-01",
            "--test-start",
            "2026-01-03",
            "--test-end",
            "2026-01-03",
            "--embargo-days",
            "1",
            "--fold-id",
            "fold_001",
            "--split-name",
            "test",
        ],
    )

    assert result.exit_code == 0, result.output
    predictions = pd.read_parquet(prediction_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert predictions["timestamp"].min() == pd.Timestamp("2026-01-03")
    assert predictions["timestamp"].max() == pd.Timestamp("2026-01-03")
    assert set(predictions["split_name"]) == {"test"}
    assert set(predictions["fold_id"]) == {"fold_001"}
    assert metadata["embargo_days"] == 1
    assert metadata["fold_id"] == "fold_001"
    assert metadata["split_name"] == "test"

    handoff = runner.invoke(
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
            "--segment-column",
            "fold_id",
            "--output",
            str(evaluation_output),
            "--run-id",
            "cli_baseline_prediction_handoff",
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

    assert handoff.exit_code == 0, handoff.output
    summary = json.loads((evaluation_output / "summary.json").read_text("utf-8"))
    assert summary["artifact_schema_version"] == "0.7"
    assert summary["kind"] == "prediction"


def test_ml_baseline_train_cli_writes_walk_forward_predictions(tmp_path) -> None:
    input_path = tmp_path / "features.parquet"
    prediction_path = tmp_path / "predictions.parquet"
    metadata_path = tmp_path / "metadata.json"
    evaluation_output = tmp_path / "evaluation"
    _baseline_panel().to_parquet(
        input_path,
        engine="pyarrow",
        compression="zstd",
        index=False,
    )
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
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
            "--feature-column",
            "feature_a",
            "--walk-forward-fold",
            "fold_001,2026-01-01,2026-01-02,2026-01-02,walk_forward",
            "--walk-forward-fold",
            "fold_002,2026-01-02,2026-01-03,2026-01-03,walk_forward",
        ],
    )

    assert result.exit_code == 0, result.output
    predictions = pd.read_parquet(prediction_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert predictions["fold_id"].tolist() == [
        "fold_001",
        "fold_001",
        "fold_002",
        "fold_002",
    ]
    assert set(predictions["split_name"]) == {"walk_forward"}
    assert metadata["fold_count"] == 2
    assert metadata["folds"][0]["test_start"] == "2026-01-02"
    assert metadata["folds"][1]["test_start"] == "2026-01-03"
    metadata_text = metadata_path.read_text(encoding="utf-8")
    for forbidden in (
        str(tmp_path),
        str(tmp_path).replace("\\", "/"),
        "vendor_private",
        "row_sample",
        "stock_sample",
    ):
        assert forbidden not in metadata_text

    handoff = runner.invoke(
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
            "fold_id",
            "--output",
            str(evaluation_output),
            "--run-id",
            "cli_walk_forward_prediction_handoff",
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

    assert handoff.exit_code == 0, handoff.output
    summary = json.loads((evaluation_output / "summary.json").read_text("utf-8"))
    assert summary["kind"] == "prediction"


def test_ml_baseline_train_cli_fails_cleanly_for_invalid_walk_forward_fold(
    tmp_path,
) -> None:
    input_path = tmp_path / "features.parquet"
    _baseline_panel().to_parquet(input_path, engine="pyarrow", index=False)
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "ml",
            "baseline",
            "train",
            "--input-parquet",
            str(input_path),
            "--output-prediction",
            str(tmp_path / "predictions.parquet"),
            "--output-metadata",
            str(tmp_path / "metadata.json"),
            "--target-column",
            "forward_return",
            "--feature-column",
            "feature_a",
            "--walk-forward-fold",
            "fold_001,2026-01-02,2026-01-02",
        ],
    )

    assert result.exit_code == 2
    assert "ML baseline input error" in result.output
    assert "train_end must be before test_start" in result.output
    assert str(tmp_path) not in result.output
    assert str(tmp_path).replace("\\", "/") not in result.output


def test_ml_baseline_train_cli_fails_cleanly_for_embargo_overlap(tmp_path) -> None:
    input_path = tmp_path / "features.parquet"
    pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-03"]),
            "asset": ["A", "A", "A"],
            "feature_a": [1.0, 2.0, 3.0],
            "forward_return": [0.01, 0.02, 0.03],
            "forward_return_1d": [0.01, 0.02, 0.03],
        }
    ).to_parquet(input_path, engine="pyarrow", compression="zstd", index=False)
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "ml",
            "baseline",
            "train",
            "--input-parquet",
            str(input_path),
            "--output-prediction",
            str(tmp_path / "predictions.parquet"),
            "--output-metadata",
            str(tmp_path / "metadata.json"),
            "--target-column",
            "forward_return",
            "--feature-column",
            "feature_a",
            "--train-end",
            "2026-01-02",
            "--test-start",
            "2026-01-03",
            "--embargo-days",
            "2",
        ],
    )

    assert result.exit_code == 2
    assert "ML baseline input error" in result.output
    assert "embargo_days" in result.output


def test_ml_baseline_train_cli_rejects_private_input_to_ordinary_outputs(
    tmp_path,
) -> None:
    private_dir = tmp_path / "vendor_private" / "inputs"
    private_dir.mkdir(parents=True)
    input_path = private_dir / "features.parquet"
    prediction_path = tmp_path / "public_outputs" / "predictions.parquet"
    metadata_path = tmp_path / "public_outputs" / "metadata.json"
    _baseline_panel().to_parquet(input_path, engine="pyarrow", index=False)
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
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
            "--feature-column",
            "feature_a",
            "--train-end",
            "2026-01-02",
            "--test-start",
            "2026-01-03",
        ],
    )

    assert result.exit_code == 2
    assert "private input policy" in result.output
    assert str(tmp_path) not in result.output
    assert str(tmp_path).replace("\\", "/") not in result.output
    assert "vendor_private" not in result.output
    assert "public_outputs" not in result.output
    assert not prediction_path.exists()
    assert not metadata_path.exists()


def test_ml_baseline_train_cli_allows_private_input_to_private_outputs(
    tmp_path,
) -> None:
    private_dir = tmp_path / "vendor_private" / "inputs"
    output_dir = tmp_path / "private_outputs"
    private_dir.mkdir(parents=True)
    input_path = private_dir / "features.parquet"
    prediction_path = output_dir / "predictions.parquet"
    metadata_path = output_dir / "metadata.json"
    _baseline_panel().to_parquet(input_path, engine="pyarrow", index=False)
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
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
            "--feature-column",
            "feature_a",
            "--train-end",
            "2026-01-02",
            "--test-start",
            "2026-01-03",
        ],
    )

    assert result.exit_code == 0, result.output
    assert "private output" in result.output
    assert str(tmp_path) not in result.output
    assert str(tmp_path).replace("\\", "/") not in result.output
    assert "vendor_private" not in result.output
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


def test_ml_baseline_train_cli_can_infer_feature_columns(tmp_path) -> None:
    input_path = tmp_path / "features.parquet"
    prediction_path = tmp_path / "predictions.parquet"
    metadata_path = tmp_path / "metadata.json"
    pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                ["2026-01-01", "2026-01-01", "2026-01-03", "2026-01-03"]
            ),
            "asset": ["A", "B", "A", "B"],
            "feature_a": [1.0, 2.0, 3.0, 1.0],
            "feature_b": [0.0, 1.0, 0.25, 2.0],
            "forward_return": [0.02, -0.01, 0.04, -0.03],
            "forward_return_1d": [0.02, -0.01, 0.04, -0.03],
        }
    ).to_parquet(input_path, engine="pyarrow", compression="zstd", index=False)
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
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
            "2026-01-01",
            "--test-start",
            "2026-01-03",
        ],
    )

    assert result.exit_code == 0, result.output
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["feature_columns"] == ["feature_a", "feature_b"]
    assert metadata["target_column"] == "forward_return"


def _baseline_panel() -> pd.DataFrame:
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
                ]
            ),
            "asset": ["A", "B", "A", "B", "A", "B"],
            "feature_a": [1.0, 2.0, 2.0, 3.0, 3.0, 1.0],
            "forward_return": [0.02, -0.01, 0.03, -0.02, 0.04, -0.03],
            "forward_return_1d": [0.02, -0.01, 0.03, -0.02, 0.04, -0.03],
        }
    )
