from __future__ import annotations

import importlib
import json
import tomllib
from pathlib import Path

import pandas as pd
from typer.testing import CliRunner

from quant_research_platform.interfaces.private_cli import app as private_app
from quant_research_platform.reports.schema_inventory import (
    build_schema_inventory_payload,
)
from tests.helpers import frequency_ids
from tests.storage.low_level_fixtures import write_low_level_parquet_artifact

ALLOWED_INVENTORY_FIELDS = {
    "artifact_id",
    "layer",
    "kind",
    "logical_relative_root",
    "columns",
    "dtypes",
    "primary_key",
    "partition_columns",
    "file_count",
    "row_count",
    "row_count_by_year",
    "min_timestamp",
    "max_timestamp",
    "duplicate_primary_key_count",
    "null_key_counts",
    "selected_important_column_null_counts",
    "schema_hash",
    "warnings",
}


def test_qrp_private_script_entry_is_the_only_private_cli_addition() -> None:
    payload = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))

    assert payload["project"]["scripts"]["qrp-private"] == (
        "quant_research_platform.interfaces.private_cli:main"
    )
    assert payload["project"]["scripts"]["qrp"] == (
        "quant_research_platform.interfaces.cli.commands:main"
    )
    for target in payload["project"]["scripts"].values():
        module_name, function_name = target.split(":", maxsplit=1)
        module = importlib.import_module(module_name)
        assert callable(getattr(module, function_name))


def test_schema_inventory_scans_sealed_dataset_roots(tmp_path) -> None:
    _write_inventory_fixture(tmp_path)

    inventory = build_schema_inventory_payload(tmp_path)
    roots = {item["logical_relative_root"] for item in inventory}
    combined = json.dumps(inventory, ensure_ascii=False)

    assert roots == {
        f"data/interim/csmar/{frequency_ids.MARKET_BARS_DATA_ID}",
        "data/interim/csmar/stock_master",
        "data/processed/a_share/factor/price_volume",
        "data/processed/a_share/annotation/default",
        "data/research/a_share/demo_research_dataset",
    }
    assert all(set(item) == ALLOWED_INVENTORY_FIELDS for item in inventory)
    market_bars_daily = _inventory_item(
        inventory,
        f"data/interim/csmar/{frequency_ids.MARKET_BARS_DATA_ID}",
    )
    stock_master = _inventory_item(inventory, "data/interim/csmar/stock_master")
    price_volume = _inventory_item(
        inventory,
        "data/processed/a_share/factor/price_volume",
    )
    annotation = _inventory_item(
        inventory,
        "data/processed/a_share/annotation/default",
    )
    research = _inventory_item(
        inventory,
        "data/research/a_share/demo_research_dataset",
    )

    assert market_bars_daily["kind"] == "interim"
    assert price_volume["kind"] == "factor"
    assert annotation["kind"] == "annotation"
    assert research["kind"] == "research_dataset"
    assert market_bars_daily["row_count_by_year"] == {
        "2023": 1,
        "2024": 1,
        "2025": 2,
    }
    assert market_bars_daily["min_timestamp"] == "2023-12-29"
    assert market_bars_daily["max_timestamp"] == "2025-01-06"
    assert stock_master["row_count_by_year"] == {"2020": 1, "2021": 1}
    assert stock_master["min_timestamp"] == "2020-01-01"
    assert stock_master["max_timestamp"] == "2021-06-01"
    assert price_volume["row_count_by_year"] == {"2024": 3, "2025": 1}
    assert price_volume["duplicate_primary_key_count"] == 1
    assert {"mom_20d", "mom_60d", "rev_20d"}.issubset(price_volume["columns"])
    assert annotation["selected_important_column_null_counts"] == {"eligible": 1}
    assert research["row_count_by_year"] == {"2024": 2}
    assert "data/raw" not in combined
    assert "reports/private_research" not in combined
    assert str(tmp_path) not in combined
    assert "asset_a" not in combined
    assert "source_asset_a" not in combined
    assert "raw_ignored_value" not in combined


def test_schema_inventory_infers_time_aware_reference_primary_key(tmp_path) -> None:
    index_info_root = tmp_path / "data" / "interim" / "csmar" / "index_info"
    write_low_level_parquet_artifact(
        index_info_root,
        pd.DataFrame(
            {
                "source_asset_id": ["source_index_a"],
                "timestamp": pd.to_datetime(["2020-01-01"]),
                "index_name": ["Synthetic Index"],
            }
        ),
        primary_key=[],
        partition_columns=["year"],
    )

    inventory = build_schema_inventory_payload(tmp_path)
    item = _inventory_item(inventory, "data/interim/csmar/index_info")

    assert item["primary_key"] == [
        "source_asset_id",
        "timestamp",
    ]
    assert "primary_key_inferred" in item["warnings"]


def test_qrp_private_schema_inventory_requires_output_and_refuses_overwrite(
    tmp_path,
) -> None:
    _write_inventory_fixture(tmp_path)
    output = tmp_path / "inventory.json"
    runner = CliRunner()

    missing_output = runner.invoke(
        private_app,
        ["schema-inventory", "--project-root", str(tmp_path)],
    )
    created = runner.invoke(
        private_app,
        [
            "schema-inventory",
            "--project-root",
            str(tmp_path),
            "--output",
            str(output),
        ],
    )
    overwrite = runner.invoke(
        private_app,
        [
            "schema-inventory",
            "--project-root",
            str(tmp_path),
            "--output",
            str(output),
        ],
    )

    assert missing_output.exit_code != 0
    assert created.exit_code == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert isinstance(payload, list)
    assert len(payload) == 5
    assert overwrite.exit_code == 2
    assert "output_already_exists" in overwrite.output


def test_qrp_private_schema_inventory_empty_roots_fail_without_output_file(
    tmp_path,
) -> None:
    output = tmp_path / "nested" / "inventory.json"
    runner = CliRunner()

    result = runner.invoke(
        private_app,
        [
            "schema-inventory",
            "--project-root",
            str(tmp_path),
            "--output",
            str(output),
        ],
    )

    assert result.exit_code != 0
    assert "schema_inventory_no_datasets_found" in result.output
    assert not output.exists()


def test_qrp_private_schema_inventory_sanitizes_unexpected_failures(
    tmp_path,
    monkeypatch,
) -> None:
    output = tmp_path / "inventory.json"

    def fail_with_private_context(project_root):
        raise RuntimeError(
            "boom at " + str(project_root / "data" / "research" / "a_share" / "demo")
        )

    monkeypatch.setattr(
        "quant_research_platform.interfaces.private_cli.build_schema_inventory_payload",
        fail_with_private_context,
    )

    result = CliRunner().invoke(
        private_app,
        [
            "schema-inventory",
            "--project-root",
            str(tmp_path),
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 2
    assert "boom" in result.output
    assert "data/research" not in result.output
    assert str(tmp_path) not in result.output
    assert "Traceback" not in result.output
    assert not output.exists()


def test_qrp_private_help_exposes_only_approved_private_commands() -> None:
    runner = CliRunner()

    root_help = runner.invoke(private_app, ["--help"])
    schema_inventory_help = runner.invoke(
        private_app,
        ["schema-inventory", "--help"],
    )
    evidence_run_help = runner.invoke(private_app, ["evidence", "run", "--help"])

    assert root_help.exit_code == 0
    assert "Private local verification harness commands." in root_help.output
    assert "schema-inventory" in root_help.output
    assert "evidence" in root_help.output
    assert "private-rebuild" not in root_help.output
    assert schema_inventory_help.exit_code == 0
    assert "private local verification harness" in schema_inventory_help.output
    assert "sealed dataset" in schema_inventory_help.output
    assert "data/interim" in schema_inventory_help.output
    assert "data/processed" in schema_inventory_help.output
    assert "data/research" in schema_inventory_help.output
    assert "--project-root" in schema_inventory_help.output
    assert "--output" in schema_inventory_help.output
    assert "--overwrite" not in schema_inventory_help.output
    assert evidence_run_help.exit_code == 0
    assert "private local verification harness" in evidence_run_help.output
    assert "--output-root" in evidence_run_help.output
    assert "--start-date" in evidence_run_help.output
    assert "--end-date" in evidence_run_help.output


def test_qrp_verification_evidence_run_accepts_required_options(monkeypatch) -> None:
    calls = []

    def fake_run(config):
        calls.append(config)
        return type(
            "Result",
            (),
            {
                "status": "PASS",
                "run_id": "s1",
                "output_root": "verification-output/s1",
                "summary": {"status": "PASS", "output_root": "verification-output/s1"},
            },
        )()

    monkeypatch.setattr(
        "quant_research_platform.interfaces.private_cli.run_local_verification",
        fake_run,
    )
    runner = CliRunner()

    result = runner.invoke(
        private_app,
        [
            "evidence",
            "run",
            "--output-root",
            "verification-output/s1",
            "--start-date",
            "2014-01-01",
            "--end-date",
            "2016-12-31",
            "--prediction-train-end",
            "2014-06-30",
            "--prediction-test-start",
            "2014-07-01",
            "--portfolio-frequency",
            frequency_ids.DAILY_FREQUENCY_SLUG,
            "--portfolio-execution-lag-steps",
            "1",
        ],
    )

    assert result.exit_code == 0
    assert calls[0].output_root == Path("verification-output/s1")
    assert calls[0].start_date == "2014-01-01"
    assert calls[0].end_date == "2016-12-31"
    assert calls[0].prediction_train_end == "2014-06-30"
    assert calls[0].prediction_test_start == "2014-07-01"
    assert calls[0].portfolio_frequency == frequency_ids.DAILY_FREQUENCY_SLUG
    assert calls[0].portfolio_execution_lag_steps == 1
    assert calls[0].profile.fields.list_names()
    assert calls[0].run_id is None
    assert "PASS" in result.output
    assert "run_id=s1" in result.output


def test_qrp_verification_evidence_run_accepts_window_overrides(monkeypatch) -> None:
    calls = []

    def fake_run(config):
        calls.append(config)
        return type(
            "Result",
            (),
            {
                "status": "PASS",
                "run_id": "s2",
                "output_root": "verification-output/s2",
                "summary": {"status": "PASS", "output_root": "verification-output/s2"},
            },
        )()

    monkeypatch.setattr(
        "quant_research_platform.interfaces.private_cli.run_local_verification",
        fake_run,
    )
    runner = CliRunner()

    result = runner.invoke(
        private_app,
        [
            "evidence",
            "run",
            "--output-root",
            "verification-output/s2",
            "--start-date",
            "2014-01-01",
            "--end-date",
            "2019-12-31",
            "--evaluation-start-date",
            "2014-01-01",
            "--evaluation-end-date",
            "2019-12-31",
            "--prediction-train-end",
            "2017-12-31",
            "--prediction-test-start",
            "2018-01-01",
            "--prediction-test-end",
            "2019-12-31",
            "--portfolio-frequency",
            frequency_ids.DAILY_FREQUENCY_SLUG,
            "--portfolio-execution-lag-steps",
            "1",
        ],
    )

    assert result.exit_code == 0
    assert calls[0].evaluation_start_date == "2014-01-01"
    assert calls[0].evaluation_end_date == "2019-12-31"
    assert calls[0].prediction_train_end == "2017-12-31"
    assert calls[0].prediction_test_start == "2018-01-01"
    assert calls[0].prediction_test_end == "2019-12-31"
    assert calls[0].portfolio_frequency == frequency_ids.DAILY_FREQUENCY_SLUG
    assert calls[0].portfolio_execution_lag_steps == 1


def test_qrp_verification_evidence_run_rejects_missing_required_options() -> None:
    runner = CliRunner()

    result = runner.invoke(private_app, ["evidence", "run"])
    help_result = runner.invoke(private_app, ["evidence", "run", "--help"])

    assert result.exit_code != 0
    assert help_result.exit_code == 0
    assert "--output-root" in help_result.output
    assert "--start-date" in help_result.output
    assert "--end-date" in help_result.output
    assert "--evaluation-start-date" in help_result.output
    assert "--evaluation-end-date" in help_result.output
    assert "--prediction-train-end" in help_result.output
    assert "--prediction-test-start" in help_result.output
    assert "--prediction-test-end" in help_result.output
    assert "--portfolio-frequency" in help_result.output
    assert "Portfolio signal execution" in help_result.output


def _write_inventory_fixture(project_root: Path) -> None:
    interim_root = frequency_ids.csmar_market_bars_interim_root(project_root)
    processed_root = (
        project_root / "data" / "processed" / "a_share" / "factor" / "price_volume"
    )
    stock_master_root = project_root / "data" / "interim" / "csmar" / "stock_master"
    universe_root = (
        project_root / "data" / "processed" / "a_share" / "annotation" / "default"
    )
    research_root = (
        project_root / "data" / "research" / "a_share" / "demo_research_dataset"
    )
    raw_root = project_root / "data" / "raw" / "vendor" / "ignored"
    report_root = (
        project_root / "reports" / "private_research" / "a_share" / "evaluation" / "run"
    )

    write_low_level_parquet_artifact(
        interim_root,
        pd.DataFrame(
            {
                "timestamp": pd.to_datetime(
                    ["2023-12-29", "2024-01-02", "2025-01-03", "2025-01-06"]
                ),
                "source_asset_id": [
                    "source_asset_a",
                    "source_asset_a",
                    "source_asset_b",
                    "source_asset_c",
                ],
                "close": [1.0, 2.0, 3.0, 4.0],
            }
        ),
        primary_key=["source_asset_id", "timestamp"],
        partition_columns=["year"],
    )
    write_low_level_parquet_artifact(
        stock_master_root,
        pd.DataFrame(
            {
                "source_asset_id": ["source_asset_a", "source_asset_b"],
                "timestamp": pd.to_datetime(["2020-01-01", "2021-06-01"]),
                "listing_date": pd.to_datetime(["2020-01-01", "2021-06-01"]),
            }
        ),
        primary_key=["source_asset_id", "timestamp"],
        partition_columns=["year"],
    )
    write_low_level_parquet_artifact(
        processed_root,
        pd.DataFrame(
            {
                "market_id": ["a_share", "a_share", "a_share", "a_share"],
                "asset_id": ["asset_a", "asset_a", "asset_b", "asset_c"],
                "timestamp": pd.to_datetime(
                    ["2024-01-02", "2024-01-02", "2024-12-31", "2025-01-03"]
                ),
                "mom_20d": [0.1, 0.2, 0.3, 0.4],
                "mom_60d": [None, 0.6, 0.7, 0.8],
                "rev_20d": [0.05, 0.04, 0.03, 0.02],
            }
        ),
        primary_key=["market_id", "asset_id", "timestamp"],
        partition_columns=["year"],
    )
    write_low_level_parquet_artifact(
        universe_root,
        pd.DataFrame(
            {
                "market_id": ["a_share", "a_share"],
                "asset_id": ["asset_a", "asset_b"],
                "timestamp": pd.to_datetime(["2024-01-02", "2025-01-03"]),
                "eligible": pd.Series([True, pd.NA], dtype="boolean"),
            }
        ),
        primary_key=["market_id", "asset_id", "timestamp"],
        partition_columns=["year"],
    )
    write_low_level_parquet_artifact(
        research_root,
        pd.DataFrame(
            {
                "market_id": ["a_share", "a_share"],
                "asset_id": ["asset_a", "asset_b"],
                "timestamp": pd.to_datetime(["2024-01-02", "2024-01-03"]),
                "mom_20d": [0.1, 0.2],
                "forward_return_20d": [0.03, 0.04],
                "eligible": pd.Series([True, False], dtype="boolean"),
            }
        ),
        primary_key=["market_id", "asset_id", "timestamp"],
        partition_columns=["year"],
    )
    write_low_level_parquet_artifact(
        raw_root,
        pd.DataFrame({"raw_value": ["raw_ignored_value"]}),
        primary_key=[],
        partition_columns=[],
    )
    write_low_level_parquet_artifact(
        report_root,
        pd.DataFrame({"asset_id": ["asset_a"], "metric": [1.0]}),
        primary_key=[],
        partition_columns=[],
    )


def _inventory_item(
    inventory: list[dict[str, object]],
    logical_root: str,
) -> dict[str, object]:
    return next(
        item for item in inventory if item["logical_relative_root"] == logical_root
    )
