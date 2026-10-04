from __future__ import annotations

from pathlib import Path

from tests.helpers.import_graph import (
    ImportEdge,
    is_ids_contract_import,
)

SOURCE_ROOT = Path("src/quant_research_platform")
MARKET_ROOT = SOURCE_ROOT / "markets/implementations"


def test_market_implementations_import_only_source_id_contracts(
    import_edges: tuple[ImportEdge, ...],
) -> None:
    violations = [
        f"{edge.path}:{edge.lineno}:{edge.module}"
        for edge in import_edges
        if edge.path.is_relative_to(MARKET_ROOT)
        and edge.module.startswith("quant_research_platform.sources")
        and not is_ids_contract_import(edge.module)
    ]

    assert violations == []
