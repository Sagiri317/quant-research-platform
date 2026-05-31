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
