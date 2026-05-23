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
