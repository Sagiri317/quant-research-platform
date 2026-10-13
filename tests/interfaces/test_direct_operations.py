"""CLI configuration is passed unchanged to the same explicit Python operation."""

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from quant_research_platform.interfaces.cli import commands


@pytest.mark.parametrize(
    "text",
    ['{"inputs":"first","inputs":"second"}', '{"seed":NaN}', '{"seed":Infinity}', "[]"],
)
def test_cli_rejects_non_strict_json_before_computation(tmp_path, monkeypatch, text):
    path = tmp_path / "config.json"
    path.write_text(text)
    # Parsing is the production JSON boundary, also used by every CLI command.
    with pytest.raises((ValueError, TypeError)):
        commands.read_qrp_json_file(path)


def test_cli_has_independent_operations_without_orchestration_options():
    from typer.main import get_command

    command = get_command(commands.app)
    assert {"fit", "predict", "score", "select", "prepare", "assemble"} <= set(
        command.commands
    )
    assert {"research", "report", "data", "status"}.isdisjoint(command.commands)
    for name in ("fit", "predict", "score", "select", "prepare", "assemble"):
        assert [param.name for param in command.commands[name].params] == ["config"]


def test_cli_main_preserves_interrupt_and_redacts_failure(monkeypatch, capsys):
    primary = KeyboardInterrupt("cancelled")

    def interrupt():
        raise primary

    monkeypatch.setattr(commands, "app", interrupt)
    with pytest.raises(KeyboardInterrupt) as caught:
        commands.main()
    assert caught.value is primary

    def fail():
        raise ValueError('{"Authorization": "Bearer SYNTHETIC_SECRET"}')

    monkeypatch.setattr(commands, "app", fail)
    with pytest.raises(SystemExit) as caught:
        commands.main()
    assert caught.value.code == 2
    output = capsys.readouterr().err
    assert "[credential]" in output
    assert "SYNTHETIC_SECRET" not in output


def test_catalog_cli_is_read_only_and_outputs_native_catalog():
    result = CliRunner().invoke(commands.app, ["catalog", "dataset", "pv_all"])
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["dataset_id"] == "pv_all"


def test_python_and_cli_fit_use_exact_same_configuration(tmp_path, monkeypatch):
    from quant_research_platform.research.models import production

    configuration = {"inputs": "selected.parquet", "output": "new-model", "seed": 17}
    path = tmp_path / "fit.json"
    path.write_text(json.dumps(configuration))
    received = []

    def fit(value):
        received.append(value)
        return Path(value["output"])

    monkeypatch.setattr(production, "fit", fit)
    result = CliRunner().invoke(commands.app, ["fit", "--config", str(path)])
    assert result.exit_code == 0, result.exception
    assert received == [configuration]
    assert json.loads(result.output) == {"output": "new-model"}


@pytest.mark.parametrize(
    "command,module,function",
    (
        ("prepare-source", "sources.prepare", "prepare_source"),
        ("prepare", "markets.prepare", "prepare"),
        ("assemble", "research.datasets.assembler", "assemble"),
        ("predict", "research.predictions.runner", "predict"),
        ("score", "research.models.trial_selection", "score"),
        ("select", "research.models.trial_selection", "select"),
        ("backtest", "research.portfolios.production", "backtest"),
        ("calibrate", "research.portfolios.production", "calibrate"),
        ("plot", "reports.plots", "plot"),
        (
            "backtest-crypto-taker",
            "markets.implementations.crypto.portfolio",
            "backtest_taker",
        ),
        (
            "backtest-crypto-maker",
            "markets.implementations.crypto.portfolio",
            "backtest_maker",
        ),
        (
            "calibrate-crypto-maker",
            "markets.implementations.crypto.portfolio",
            "calibrate_maker",
        ),
        ("statistics-factors", "research.statistics_production", "factor_statistics"),
        ("statistics-model", "research.statistics_production", "model_statistics"),
        (
            "statistics-portfolio",
            "research.portfolios.diagnostics",
            "portfolio_statistics",
        ),
        ("statistics-crypto", "research.portfolios.diagnostics", "crypto_statistics"),
        ("evaluate", "research.evaluations.runner", "evaluate"),
    ),
)
def test_every_cli_operation_passes_the_same_config_without_overrides(
    tmp_path, monkeypatch, command, module, function
):
    import importlib

    domain = importlib.import_module("quant_research_platform." + module)
    value = {
        "inputs": ["first", "second"],
        "output": "new-output",
        "sample_role": "validation",
    }
    path = tmp_path / "config.json"
    path.write_text(json.dumps(value))
    observed = []

    def operation(config):
        observed.append(config)
        if command == "prepare-source":
            from quant_research_platform.sources.csmar.interim import (
                InterimImportResult,
            )

            return InterimImportResult({"selected": Path(config["output"])}, ("empty",))
        return Path(config["output"])

    monkeypatch.setattr(domain, function, operation)
    result = CliRunner().invoke(commands.app, [command, "--config", str(path)])
    assert result.exit_code == 0, result.exception
    assert observed == [value]
    expected = (
        {"outputs": {"selected": "new-output"}, "skipped": ["empty"]}
        if command == "prepare-source"
        else {"output": "new-output"}
    )
    assert json.loads(result.output) == expected


def test_inspect_reads_only_the_selected_native_dataset_metadata(tmp_path):
    from quant_research_platform.interfaces.queries import inspect
    from quant_research_platform.storage.datasets import write_dataset
    from tests.storage.low_level_fixtures import (
        make_default_dataset_frame,
        make_processed_dataset_spec,
    )

    root = tmp_path / "arbitrary-output"
    write_dataset(
        root, make_default_dataset_frame(), spec=make_processed_dataset_spec("selected")
    )
    (root / "unrelated.json").write_text("not JSON")
    result = inspect(root)
    assert result["row_count"] == 3
    assert "common_manifest" not in result
    assert "instance_identity" not in result
    with pytest.raises(ValueError, match="inspection_requires_native_json_metadata"):
        inspect(root / "unrelated.json")
