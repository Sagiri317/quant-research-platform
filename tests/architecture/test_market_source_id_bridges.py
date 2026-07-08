from __future__ import annotations

import ast
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


def test_market_processed_source_id_bridges_are_only_source_imports() -> None:
    allowed = {
        PROJECT_SRC / "markets/implementations/a_share/processed/source_ids.py": {
            "quant_research_platform.sources.csmar.ids",
            "quant_research_platform.sources.csmar",
        },
        PROJECT_SRC / "markets/implementations/crypto/processed/source_ids.py": {
            "quant_research_platform.sources.binance.ids",
            "quant_research_platform.sources.binance",
        },
    }
    roots = (
        PROJECT_SRC / "markets/implementations/a_share",
        PROJECT_SRC / "markets/implementations/crypto",
    )
    hits: list[str] = []
    for root in roots:
        for path in sorted(root.rglob("*.py")):
            for module, line_number in _imported_modules(path):
                if not module.startswith("quant_research_platform.sources"):
                    continue
                if module not in allowed.get(path, set()):
                    hits.append(f"{path}:{line_number}:{module}")

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
        f"{path}:{line_number}:{module}"
        for path in paths
        for module, line_number in _imported_modules(path)
        if module.startswith("quant_research_platform.sources")
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


def _imported_modules(path: Path) -> tuple[tuple[str, int], ...]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    modules: list[tuple[str, int]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module is not None:
            modules.append((node.module, node.lineno))
        elif isinstance(node, ast.Import):
            modules.extend((alias.name, node.lineno) for alias in node.names)
    return tuple(modules)
