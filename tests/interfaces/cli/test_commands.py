from __future__ import annotations

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
