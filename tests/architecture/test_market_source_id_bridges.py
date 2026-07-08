from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path

import pytest

from quant_research_platform.contracts.columns import ColumnRole
from quant_research_platform.contracts.data import (
    DataFrequency,
    DataGrain,
    ProcessedDataKind,
    ProcessedDataSpec,
)
from quant_research_platform.contracts.ids import MarketId
from quant_research_platform.markets.implementations.crypto.processed import source_ids

PROJECT_SRC = Path("src/quant_research_platform")


@dataclass(frozen=True)
class ImportRecord:
    path: Path
    line_number: int
    module: str
    name: str | None = None
    asname: str | None = None

    @property
    def qualified_name(self) -> str:
        if self.name is None:
            return self.module
        return f"{self.module}.{self.name}"


def test_market_processed_source_id_bridges_are_only_source_imports() -> None:
    allowed = {
        PROJECT_SRC / "markets/implementations/a_share/processed/source_ids.py": (
            "quant_research_platform.sources.csmar",
            "ids",
            "csmar_ids",
        ),
        PROJECT_SRC / "markets/implementations/crypto/processed/source_ids.py": (
            "quant_research_platform.sources.binance",
            "ids",
            "binance_ids",
        ),
    }
    roots = (
        PROJECT_SRC / "markets/implementations/a_share",
        PROJECT_SRC / "markets/implementations/crypto",
    )
    hits: list[str] = []
    for root in roots:
        for path in sorted(root.rglob("*.py")):
            for record in _import_records(path):
                if not record.module.startswith("quant_research_platform.sources"):
                    continue
                if _allowed_source_bridge_import(record) != allowed.get(path):
                    hits.append(
                        f"{path}:{record.line_number}:"
                        f"{record.qualified_name} as {record.asname}"
                    )

    assert hits == []


def test_processed_catalogs_and_materializers_do_not_import_sources() -> None:
    paths = (
        PROJECT_SRC / "markets/implementations/a_share/processed/catalog.py",
        PROJECT_SRC / "markets/implementations/crypto/processed/catalog.py",
        PROJECT_SRC / "markets/implementations/crypto/processed/materializer.py",
        PROJECT_SRC / "markets/implementations/crypto/processed/materialize_panel.py",
        PROJECT_SRC / "markets/implementations/crypto/processed/materialize_factors.py",
        PROJECT_SRC / "markets/implementations/crypto/processed/materialize_labels.py",
    )
    hits = [
        f"{path}:{record.line_number}:{module}"
        for path in paths
        for record in _import_records(path)
        if (module := record.qualified_name).startswith(
            "quant_research_platform.sources"
        )
    ]

    assert hits == []


def test_processed_catalogs_use_source_id_bridge_constants() -> None:
    a_share_source = (
        PROJECT_SRC / "markets/implementations/a_share/processed/catalog.py"
    ).read_text(encoding="utf-8")
    crypto_source = (
        PROJECT_SRC / "markets/implementations/crypto/processed/catalog.py"
    ).read_text(encoding="utf-8")

    assert "source_ids.MARKET_BARS_SOURCE_DATA_ID" in a_share_source
    assert "MARKET_BARS_SOURCE_DATA_ID" in a_share_source
    assert "source_ids.MARKET_BARS_SOURCE_DATA_ID" in crypto_source
    assert "market_bars_1m" not in crypto_source
    assert "aggregate_trades_tick" not in crypto_source
    assert "depth_levels_tick" not in crypto_source


@pytest.mark.parametrize(
    "source_input_data_ids",
    [
        ("market_bars_1m", "aggregate_trades_tick"),
        ("market_bars_1m", "aggregate_trades_tick", "depth_levels_tick", "extra"),
        (
            "market_bars_1m",
            "aggregate_trades_tick",
            "depth_levels_tick",
            "depth_levels_tick",
        ),
    ],
)
def test_crypto_source_input_bridge_requires_exact_source_input_set(
    source_input_data_ids: tuple[str, ...],
) -> None:
    spec = ProcessedDataSpec(
        data_id="market_panel_1m",
        market_id=MarketId("crypto"),
        kind=ProcessedDataKind.PANEL,
        frequency=DataFrequency.MINUTE,
        grain=DataGrain.ASSET_TIMESTAMP,
        column_roles={"timestamp": ColumnRole.TIMESTAMP},
        primary_key=("timestamp",),
        partition_columns=("year",),
        source_input_data_ids=source_input_data_ids,
    )

    with pytest.raises(ValueError, match="panel_source_input_data_ids_mismatch"):
        source_ids.market_bars_source_data_id(spec)


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        (
            "from quant_research_platform.sources.binance import ids as binance_ids",
            [],
        ),
        (
            "from quant_research_platform.sources.binance import ids",
            ["quant_research_platform.sources.binance.ids as None"],
        ),
        (
            "from quant_research_platform.sources.binance import catalog",
            ["quant_research_platform.sources.binance.catalog as None"],
        ),
    ],
)
def test_source_bridge_import_records_require_ids_alias(
    source: str,
    expected: list[str],
) -> None:
    path = PROJECT_SRC / "markets/implementations/crypto/processed/source_ids.py"
    records = _import_records_from_source(source, path=path)
    hits = [
        f"{record.qualified_name} as {record.asname}"
        for record in records
        if _allowed_source_bridge_import(record)
        != ("quant_research_platform.sources.binance", "ids", "binance_ids")
    ]

    assert hits == expected


def _allowed_source_bridge_import(
    record: ImportRecord,
) -> tuple[str, str, str] | None:
    if record.name is None:
        return None
    return (record.module, record.name, record.asname or "")


def _import_records(path: Path) -> tuple[ImportRecord, ...]:
    return _import_records_from_source(path.read_text(encoding="utf-8"), path=path)


def _import_records_from_source(source: str, *, path: Path) -> tuple[ImportRecord, ...]:
    tree = ast.parse(source, filename=str(path))
    modules: list[ImportRecord] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module is not None:
            modules.extend(
                ImportRecord(
                    path=path,
                    line_number=node.lineno,
                    module=node.module,
                    name=alias.name,
                    asname=alias.asname,
                )
                for alias in node.names
            )
        elif isinstance(node, ast.Import):
            modules.extend(
                ImportRecord(
                    path=path,
                    line_number=node.lineno,
                    module=alias.name,
                    asname=alias.asname,
                )
                for alias in node.names
            )
    return tuple(modules)
