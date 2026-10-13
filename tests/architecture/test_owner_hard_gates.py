from __future__ import annotations

import ast
from pathlib import Path

from tests.helpers.import_graph import import_edges_under

SOURCE_ROOT = Path("src/quant_research_platform")


def test_query_owner_has_no_workflow_or_source_traversal(import_edges):
    from tests.helpers.import_graph import import_edges_under

    for edge in import_edges_under(import_edges, SOURCE_ROOT / "interfaces/queries.py"):
        assert not edge.module.startswith(
            ("quant_research_platform.workflows", "quant_research_platform.sources")
        )
        assert not edge.module.rsplit(".", 1)[-1].startswith("write_")


def test_research_and_profile_owners_have_no_provider_traversal(import_edges) -> None:
    from tests.helpers.import_graph import import_edges_under

    for edge in import_edges_under(import_edges, SOURCE_ROOT / "research"):
        assert not edge.module.startswith("quant_research_platform.sources")
    for edge in import_edges_under(import_edges, SOURCE_ROOT / "profiles/research.py"):
        assert edge.module.rsplit(".", 1)[-1] != "write_dataset"
        assert not edge.module.startswith("pandas")


def test_a_share_pit_owner_and_lifecycle_inputs_have_one_direction(
    import_edges,
) -> None:
    factor_path = (
        SOURCE_ROOT / "research" / "factors" / "fundamental" / "calculations.py"
    )
    factor_modules = (
        edge.module for edge in import_edges_under(import_edges, factor_path)
    )
    assert all(
        not module.startswith("quant_research_platform.markets")
        for module in factor_modules
    )

    producer_path = (
        SOURCE_ROOT
        / "markets"
        / "implementations"
        / "a_share"
        / "processed"
        / "producer.py"
    )
    producer_tree = ast.parse(producer_path.read_text(encoding="utf-8"))
    assert (
        "quant_research_platform.markets.implementations.a_share.fundamentals.construct_ttm_events"
        in _canonical_references(producer_tree)
    )

    from quant_research_platform.markets.implementations.a_share.processed import (
        catalog,
    )
    from quant_research_platform.sources.choice.interim import first_disclosure_spec

    panel_sources = (
        catalog.processed_catalog().get("market_panel_1d").source_input_data_ids
    )
    # Industry announcements do not replace stock_master as listing truth.
    assert {"listing_status_changes", "stock_info_annual"}.isdisjoint(panel_sources)
    assert {"stock_master", "stock_info_changes"}.issubset(panel_sources)
    assert first_disclosure_spec().primary_key == (
        "asset_id",
        "report_period",
        "first_disclosure_date",
    )


def test_pit_reference_check_accepts_owner_aliases_and_rejects_wrong_owner():
    owner = "quant_research_platform.markets.implementations.a_share.fundamentals"
    for source in (
        f"from {owner} import construct_ttm_events as construct\nconstruct()",
        "from quant_research_platform.markets.implementations.a_share "
        "import fundamentals as facts\nfacts.construct_ttm_events()",
        f"import {owner} as facts\nfacts.construct_ttm_events()",
    ):
        assert owner + ".construct_ttm_events" in _canonical_references(
            ast.parse(source)
        )
    source = "from wrong_owner import construct_ttm_events\nconstruct_ttm_events()"
    assert owner + ".construct_ttm_events" not in _canonical_references(
        ast.parse(source)
    )


def test_configured_model_algorithms_use_the_single_capability_owner() -> None:
    import json

    from quant_research_platform.contracts.models import _MODEL_ALGORITHM_FACTS

    root = Path(__file__).resolve().parents[2] / "docs" / "research-rosters"
    configured = {
        model["model_algorithm"]
        for path in root.rglob("*.json")
        for model in json.loads(path.read_text(encoding="utf-8"))["models"]
    }
    assert configured == set(_MODEL_ALGORITHM_FACTS)


def _defined_symbols(tree: ast.AST) -> set[str]:
    symbols: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            symbols.add(node.name)
        elif isinstance(node, ast.arg):
            symbols.add(node.arg)
        elif isinstance(node, ast.Name):
            symbols.add(node.id)
        elif isinstance(node, ast.Attribute):
            symbols.add(node.attr)
        elif isinstance(node, ast.keyword) and node.arg is not None:
            symbols.add(node.arg)
    return symbols


def _canonical_references(tree):
    bindings = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            for alias in node.names:
                bindings[alias.asname or alias.name] = f"{node.module}.{alias.name}"
        elif isinstance(node, ast.Import):
            for alias in node.names:
                bindings[alias.asname or alias.name.split(".")[0]] = (
                    alias.name if alias.asname else alias.name.split(".")[0]
                )
    references = set()
    for node in ast.walk(tree):
        parts = []
        while isinstance(node, ast.Attribute):
            parts.append(node.attr)
            node = node.value
        if isinstance(node, ast.Name) and node.id in bindings:
            references.add(".".join((bindings[node.id], *reversed(parts))))
    return references
