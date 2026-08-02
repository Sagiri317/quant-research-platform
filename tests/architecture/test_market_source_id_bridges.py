from __future__ import annotations

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
from tests.helpers.import_graph import ImportEdge, is_relative_to

SOURCE_ROOT = Path("src/quant_research_platform")
MARKET_ROOT = SOURCE_ROOT / "markets/implementations"


def test_market_implementations_import_only_source_id_contracts(
    import_edges: tuple[ImportEdge, ...],
) -> None:
    violations = [
        f"{edge.path}:{edge.lineno}:{edge.module}"
        for edge in import_edges
        if is_relative_to(edge.path, MARKET_ROOT)
        and edge.module.startswith("quant_research_platform.sources")
        and not edge.module.endswith(".ids")
    ]

    assert violations == []


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
def test_crypto_source_bridge_requires_exact_input_set(
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
