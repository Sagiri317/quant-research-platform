from __future__ import annotations

import ast
from pathlib import Path

from tests.helpers.import_graph import ImportEdge, import_edges_under

PROJECT_SRC = Path("src/quant_research_platform")
AGENTS_SRC = PROJECT_SRC / "agents"
PROVIDER_BINDINGS = PROJECT_SRC / "catalog" / "provider_bindings.py"
STORAGE_DATASETS = PROJECT_SRC / "storage" / "datasets.py"
DELETED_STORAGE_PARQUET = PROJECT_SRC / "storage" / "parquet.py"
WORKFLOW_INTERFACE_ROOTS = (
    PROJECT_SRC / "workflows",
    PROJECT_SRC / "interfaces",
)
OLD_PRIVATE_HELPER = "private" + "_interim_root_for_data_product"
FORBIDDEN_PROVIDER_BINDING_IMPORT_PREFIXES = (
    "quant_research_platform.storage",
    "quant_research_platform.sources",
    "quant_research_platform.markets",
)
FORBIDDEN_PROVIDER_BINDING_HELPERS = {
    OLD_PRIVATE_HELPER,
    "raw_source_root",
    "interim_source_root",
    "processed_market_root",
    "layout_from_processed_market_root",
    "layout_from_processed_root",
}


def test_agents_do_not_import_data_or_market_implementation_layers(
    import_edges: tuple[ImportEdge, ...],
) -> None:
    blocked_prefixes = (
        "quant_research_platform.storage",
        "quant_research_platform.sources",
        "quant_research_platform.markets.implementations",
    )
    hits = [
        f"{edge.path}:{edge.lineno} imports {edge.module}"
        for edge in import_edges_under(import_edges, AGENTS_SRC)
        if edge.module.startswith(blocked_prefixes)
    ]

    assert not hits, "agent_data_layer_import: " + "; ".join(hits)


def test_provider_bindings_catalog_remains_pure_container() -> None:
    assert not PROVIDER_BINDINGS.exists()


def test_storage_datasets_exposes_single_artifact_root_api() -> None:
    import quant_research_platform.storage.datasets as datasets

    source = STORAGE_DATASETS.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(STORAGE_DATASETS))
    public_functions = {
        node.name for node in tree.body if isinstance(node, ast.FunctionDef)
    }

    assert not DELETED_STORAGE_PARQUET.exists()
    assert datasets.__all__ == [
        "DatasetReadMode",
        "DatasetInspection",
        "DatasetFingerprint",
        "DatasetComparison",
        "write_dataset",
        "read_dataset",
        "inspect_dataset",
        "fingerprint_dataset",
        "compare_datasets",
    ]
    assert {
        "write_dataset",
        "read_dataset",
        "inspect_dataset",
        "fingerprint_dataset",
        "compare_datasets",
    }.issubset(public_functions)


def test_workflow_and_interface_public_names_use_validate_only_wording() -> None:
    forbidden_terms = (
        "workflow" + "-fixture",
        "fixture" + "_backed",
        "fixture" + "_workflow_node_registry",
        "validate" + "_only_workflow_node_registry",
    )
    hits: list[str] = []
    for root in WORKFLOW_INTERFACE_ROOTS:
        for path in sorted(root.rglob("*.py")):
            text = path.read_text(encoding="utf-8")
            for lineno, line in enumerate(text.splitlines(), start=1):
                for term in forbidden_terms:
                    if term in line:
                        hits.append(f"{path}:{lineno}: {term}")

    assert not hits, "workflow_public_fixture_wording_found: " + "; ".join(hits)


def _referenced_names(tree: ast.Module) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, ast.Attribute):
            names.add(node.attr)
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            names.add(node.value)
    return names
