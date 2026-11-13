from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from typer.testing import CliRunner

from quant_research_platform.interfaces.cli.commands import app
from quant_research_platform.storage.identity import (
    filesystem_exists,
    native_filesystem_path,
)
from quant_research_platform.storage.schema_inventory import (
    iter_schema_inventory_items,
)
from tests.storage.low_level_fixtures import write_low_level_parquet_artifact


def test_schema_inventory_command_writes_current_local_artifact(
    tmp_path: Path,
) -> None:
    write_low_level_parquet_artifact(
        tmp_path / "data" / "interim" / "csmar" / "market_bars_1d",
        pd.DataFrame(
            {
                "source_asset_id": ["source_a"],
                "timestamp": pd.to_datetime(["2024-01-02"]),
                "close": [1.0],
            }
        ),
        primary_key=["source_asset_id", "timestamp"],
        partition_columns=["year"],
    )
    output = tmp_path / "inventory.json"
    result = CliRunner().invoke(
        app,
        [
            "schema-inventory",
            "--project-root",
            str(tmp_path),
            "--output",
            str(output),
        ],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(
        Path(native_filesystem_path(output)).read_text(encoding="utf-8")
    )
    assert payload["schema_id"] == "qrp-schema-inventory"
    assert payload["datasets"]


def test_schema_inventory_rejects_overwrite_and_empty_roots(tmp_path: Path) -> None:
    output = tmp_path / "inventory.json"
    runner = CliRunner()
    missing = runner.invoke(
        app,
        ["schema-inventory", "--project-root", str(tmp_path), "--output", str(output)],
    )
    assert missing.exit_code != 0
    assert "schema_inventory_no_datasets_found" in str(missing.exception)
    assert not filesystem_exists(output)

    root = tmp_path / "data" / "interim" / "csmar" / "market_bars_1d"
    write_low_level_parquet_artifact(
        root,
        pd.DataFrame(
            {
                "source_asset_id": ["source_a"],
                "timestamp": pd.to_datetime(["2024-01-02"]),
                "close": [1.0],
            }
        ),
        primary_key=["source_asset_id", "timestamp"],
        partition_columns=["year"],
    )
    created = runner.invoke(
        app,
        ["schema-inventory", "--project-root", str(tmp_path), "--output", str(output)],
    )
    overwrite = runner.invoke(
        app,
        ["schema-inventory", "--project-root", str(tmp_path), "--output", str(output)],
    )
    assert created.exit_code == 0, created.output
    assert overwrite.exit_code != 0
    assert "output_already_exists" in str(overwrite.exception)


def test_schema_inventory_uses_relative_artifact_paths(tmp_path: Path) -> None:
    root = tmp_path / "data" / "interim" / "csmar" / "market_bars_1d"
    write_low_level_parquet_artifact(
        root,
        pd.DataFrame(
            {
                "source_asset_id": ["source_a"],
                "timestamp": pd.to_datetime(["2024-01-02"]),
                "close": [1.0],
            }
        ),
        primary_key=["source_asset_id", "timestamp"],
        partition_columns=["year"],
    )
    with iter_schema_inventory_items(tmp_path) as items:
        payload = json.dumps(list(items), ensure_ascii=False)
    assert str(tmp_path) not in payload
