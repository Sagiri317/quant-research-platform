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
from tests.storage.low_level_fixtures import write_low_level_parquet_artifact


def test_schema_inventory_cli_publishes_relative_paths_once(tmp_path: Path) -> None:
    output = tmp_path / "inventory.json"
    runner = CliRunner()
    args = [
        "schema-inventory",
        "--project-root",
        str(tmp_path),
        "--output",
        str(output),
    ]
    missing = runner.invoke(app, args)
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
    created = runner.invoke(app, args)
    overwrite = runner.invoke(app, args)
    assert created.exit_code == 0, created.output
    assert overwrite.exit_code != 0
    assert "output_already_exists" in str(overwrite.exception)

    payload = json.loads(
        Path(native_filesystem_path(output)).read_text(encoding="utf-8")
    )
    assert payload["schema_id"] == "qrp-schema-inventory"
    (item,) = payload["datasets"]
    assert item["logical_relative_root"] == root.relative_to(tmp_path).as_posix()
    assert str(tmp_path) not in json.dumps(payload, ensure_ascii=False)
