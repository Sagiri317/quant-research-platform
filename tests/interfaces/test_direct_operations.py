"""CLI configuration is passed unchanged to the same explicit Python operation."""

import json
import sys
from pathlib import Path

import pytest
from typer.testing import CliRunner

from quant_research_platform.interfaces.cli import commands

CONFIG_OPERATIONS = (
    ("prepare-source", "sources.prepare", "prepare_source"),
    ("prepare", "markets.prepare", "prepare"),
    ("assemble", "research.datasets.assembler", "assemble"),
    ("fit", "research.models.production", "fit"),
    ("predict", "research.predictions.runner", "predict"),
    ("explain", "research.models.explanations", "explain"),
    ("score", "research.models.trial_selection", "score"),
    ("select", "research.models.trial_selection", "select"),
    ("backtest", "research.portfolios.production", "backtest"),
    ("calibrate", "research.portfolios.production", "calibrate"),
    ("plot", "reports.plots", "plot"),
    ("backtest-hft", "research.portfolios.high_frequency", "backtest"),
    ("run-quarterly", "research.rolling", "run_quarterly"),
    ("run-sessions", "research.rolling", "run_sessions"),
    ("normalize-book", "research.features.book_normalization", "normalize_book"),
    ("sample-hft", "research.datasets.high_frequency", "sample_hft"),
    (
        "evaluate-time-series",
        "research.evaluations.time_series",
        "evaluate_time_series",
    ),
    (
        "analyze-inventory",
        "research.portfolios.inventory_analysis",
        "analyze_inventory",
    ),
    ("fit-hawkes", "research.models.hawkes", "fit_hawkes"),
    ("statistics-factors", "research.factors.statistics", "factor_statistics"),
    ("statistics-model", "research.models.statistics", "model_statistics"),
    ("statistics-portfolio", "research.portfolios.diagnostics", "portfolio_statistics"),
    ("evaluate", "research.evaluations.runner", "evaluate"),
    ("train-policy", "research.policies.reinforcement", "train_policy"),
    ("backtest-policy", "research.policies.reinforcement", "backtest_policy"),
    ("search-factors", "research.factors.search", "search_factors"),
    ("apply-formula", "research.factors.search", "apply_formula"),
    (
        "calibrate-uncertainty",
        "research.predictions.uncertainty",
        "calibrate_uncertainty",
    ),
    ("apply-uncertainty", "research.predictions.uncertainty", "apply_uncertainty"),
    (
        "evaluate-uncertainty",
        "research.predictions.uncertainty",
        "evaluate_uncertainty",
    ),
    ("fit-cost-model", "research.policies.cost_aware", "fit_cost_model"),
    ("backtest-cost-model", "research.policies.cost_aware", "backtest_cost_model"),
)


def test_cli_rejects_non_strict_json_before_computation(tmp_path, monkeypatch):
    from quant_research_platform.research.models import production

    path = tmp_path / "config.json"
    path.write_text('{"inputs":"first","inputs":"second"}')
    received = []

    def fit(config):
        received.append(config)

    monkeypatch.setattr(production, "fit", fit)
    result = CliRunner().invoke(commands.app, ["fit", "--config", str(path)])

    assert result.exit_code != 0
    assert isinstance(result.exception, ValueError)
    assert "qrp_json_duplicate_key" in str(result.exception)
    assert received == []


def test_cli_registers_independent_operations_with_help():
    from typer.main import get_command

    command = get_command(commands.app)
    assert set(command.commands) == {
        *(name for name, _, _ in CONFIG_OPERATIONS),
        "agent",
        "inspect",
        "catalog",
        "schema-inventory",
    }
    for name, _, _ in CONFIG_OPERATIONS:
        assert [param.name for param in command.commands[name].params] == ["config"]
    assert all(item.help for item in command.commands.values())
    assert all(item.help for item in command.commands["catalog"].commands.values())
    result = CliRunner().invoke(commands.app, ["explain", "--help"])
    assert result.exit_code == 0
    assert "frozen model" in result.output


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


@pytest.mark.parametrize("command,module,function", CONFIG_OPERATIONS)
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
    with pytest.raises(ValueError, match="qrp_json_invalid"):
        inspect(root / "unrelated.json")


def test_inspect_prediction_uses_native_manifest_without_reading_scores(tmp_path):
    import json

    from quant_research_platform.interfaces.queries import inspect
    from tests.helpers.output_frames import native_bundle

    root, _, _ = native_bundle(tmp_path)
    native = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    (root / "prediction.json").write_text("not a native metadata member")
    assert inspect(root) == native
    assert inspect(root / "manifest.json") == native
    with pytest.raises(ValueError, match="qrp_json_invalid"):
        inspect(root / "prediction.json")


@pytest.mark.parametrize("kind", ["date", "path", "nonfinite"])
def test_source_stage_json_boundary_precedes_materialization(
    tmp_path, monkeypatch, kind
):
    from datetime import date

    from quant_research_platform.sources.csmar import interim
    from quant_research_platform.sources.prepare import prepare_source

    def forbidden(**kwargs):
        pytest.fail("non-JSON configuration reached source I/O")

    monkeypatch.setattr(interim, "materialize_interim_data", forbidden)
    config = {
        "source_id": "csmar",
        "raw_root": str(tmp_path / "missing"),
        "outputs": {"market_bars_1d": str(tmp_path / "output")},
    }
    config.update(
        {
            "date": {"start_date": date(2020, 1, 1)},
            "path": {"raw_root": tmp_path / "missing"},
            "nonfinite": {"chunksize": float("nan")},
        }[kind]
    )
    with pytest.raises(ValueError):
        prepare_source(config)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("kind", ["missing", "not_a_directory", "malformed"])
def test_cli_main_classifies_config_errors(tmp_path, monkeypatch, capsys, kind):
    path = tmp_path / "config.json"
    if kind == "malformed":
        path.write_text("not JSON")
    elif kind == "not_a_directory":
        path.write_text("{}")
        path = path / "config.json"
    monkeypatch.setattr(sys, "argv", ["qrp", "fit", "--config", str(path)])

    with pytest.raises(SystemExit) as caught:
        commands.main()

    assert caught.value.code == (2 if kind == "malformed" else 3)
    output = capsys.readouterr().err
    if kind != "malformed":
        assert output == "required_input_missing\n"
    assert str(tmp_path) not in output


@pytest.mark.parametrize(
    "error,cause,expected",
    [
        (ValueError("dataset_metadata_file_invalid"), None, 2),
        (ValueError("invalid_source"), FileNotFoundError("private_source"), 3),
        (ValueError("invalid_source"), NotADirectoryError("private_source"), 3),
        (ValueError("invalid_source"), PermissionError("private_source"), 2),
        (FileNotFoundError("private_source"), None, 3),
        (NotADirectoryError("private_source"), None, 3),
        (RuntimeError("execution_failed"), None, 1),
    ],
)
def test_cli_main_classifies_operation_errors(
    tmp_path, monkeypatch, capsys, error, cause, expected
):
    from quant_research_platform.research.models import production

    path = tmp_path / "config.json"
    path.write_text("{}")

    def fail(config):
        assert config == {}
        raise error from cause

    monkeypatch.setattr(production, "fit", fail)
    monkeypatch.setattr(sys, "argv", ["qrp", "fit", "--config", str(path)])
    with pytest.raises(SystemExit) as caught:
        commands.main()

    assert caught.value.code == expected
    output = capsys.readouterr().err
    assert output == ("required_input_missing\n" if expected == 3 else f"{error}\n")
    assert "private_source" not in output
