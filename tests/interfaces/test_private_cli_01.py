from __future__ import annotations

import importlib
import json
import tomllib
from pathlib import Path

import pandas as pd
from typer.testing import CliRunner

from quant_research_platform.interfaces.private_cli import app as private_app
from quant_research_platform.reports.schema_inventory import build_schema_inventory
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

    inventory = build_schema_inventory(tmp_path)
    roots = {item["logical_relative_root"] for item in inventory}
    combined = json.dumps(inventory, ensure_ascii=False)

    assert roots == {
        "data/interim/csmar/daily_market",
        "data/interim/csmar/stock_master",
        "data/processed/a_share/factor/price_volume",
        "data/processed/a_share/annotation/a_share_default",
        "data/research/a_share/demo_research_dataset",
    }
    assert all(set(item) == ALLOWED_INVENTORY_FIELDS for item in inventory)
    daily_market = _inventory_item(inventory, "data/interim/csmar/daily_market")
    stock_master = _inventory_item(inventory, "data/interim/csmar/stock_master")
    price_volume = _inventory_item(
        inventory,
        "data/processed/a_share/factor/price_volume",
    )
    annotation = _inventory_item(
        inventory,
        "data/processed/a_share/annotation/a_share_default",
    )
    research = _inventory_item(
        inventory,
        "data/research/a_share/demo_research_dataset",
    )

    assert daily_market["kind"] == "interim"
    assert price_volume["kind"] == "factor"
    assert annotation["kind"] == "annotation"
    assert research["kind"] == "research_dataset"
    assert daily_market["row_count_by_year"] == {"2023": 1, "2024": 1, "2025": 2}
    assert daily_market["min_timestamp"] == "2023-12-29"
    assert daily_market["max_timestamp"] == "2025-01-06"
    assert stock_master["row_count_by_year"] == {"2020": 1, "2021": 1}
    assert stock_master["min_timestamp"] == "2020-01-01"
    assert stock_master["max_timestamp"] == "2021-06-01"
    assert price_volume["row_count_by_year"] == {"2024": 3, "2025": 1}
    assert price_volume["duplicate_primary_key_count"] == 1
    assert {"mom_20d", "mom_60d", "rev_20d"}.issubset(price_volume["columns"])
    assert annotation["selected_important_column_null_counts"] == {"eligible": 1}
    assert research["row_count_by_year"] == {"2024": 2}
    assert "data/raw" not in combined
    assert "reports/private_real" not in combined
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

    inventory = build_schema_inventory(tmp_path)
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


def test_qrp_private_schema_inventory_help_exposes_only_schema_inventory() -> None:
    runner = CliRunner()

    root_help = runner.invoke(private_app, ["--help"])
    command_help = runner.invoke(private_app, ["schema-inventory", "--help"])

    assert root_help.exit_code == 0
    assert "schema-inventory" in root_help.output
    assert "evidence" in root_help.output
    assert "private-rebuild" not in root_help.output
    assert command_help.exit_code == 0
    assert "--project-root" in command_help.output
    assert "--output" in command_help.output
    assert "--overwrite" not in command_help.output


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
        "quant_research_platform.interfaces.private_cli.run_verification_evidence",
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
        ],
    )

    assert result.exit_code == 0
    assert calls[0].output_root == Path("verification-output/s1")
    assert calls[0].start_date == "2014-01-01"
    assert calls[0].end_date == "2016-12-31"
    assert calls[0].run_id is None
    assert "PASS" in result.output
    assert "run_id=s1" in result.output


def test_qrp_verification_evidence_run_rejects_missing_required_options() -> None:
    runner = CliRunner()

    result = runner.invoke(private_app, ["evidence", "run"])
    help_result = runner.invoke(private_app, ["evidence", "run", "--help"])

    assert result.exit_code != 0
    assert help_result.exit_code == 0
    assert "--output-root" in help_result.output
    assert "--start-date" in help_result.output
    assert "--end-date" in help_result.output


def _write_inventory_fixture(project_root: Path) -> None:
    interim_root = project_root / "data" / "interim" / "csmar" / "daily_market"
    processed_root = (
        project_root
        / "data"
        / "processed"
        / "a_share"
        / "factor"
        / "price_volume"
    )
    stock_master_root = project_root / "data" / "interim" / "csmar" / "stock_master"
    universe_root = (
        project_root
        / "data"
        / "processed"
        / "a_share"
        / "annotation"
        / "a_share_default"
    )
    research_root = (
        project_root / "data" / "research" / "a_share" / "demo_research_dataset"
    )
    raw_root = project_root / "data" / "raw" / "vendor" / "ignored"
    report_root = (
        project_root / "reports" / "private_real" / "a_share" / "evaluation" / "run"
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
