from __future__ import annotations

import ast
from pathlib import Path

PROJECT_SRC = Path("src/quant_research_platform")
AGENTS_SRC = PROJECT_SRC / "agents"
PROVIDER_BINDINGS = PROJECT_SRC / "catalog" / "provider_bindings.py"
STORAGE_PARQUET = PROJECT_SRC / "storage" / "parquet.py"
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


def test_agents_do_not_import_market_implementations() -> None:
    hits = [
        f"{path}:{lineno} imports {module}"
        for path, lineno, module in _iter_imports(AGENTS_SRC)
        if module.startswith("quant_research_platform.markets.implementations")
    ]

    assert not hits, "agent_market_implementation_import: " + "; ".join(hits)


def test_provider_bindings_catalog_remains_pure_container() -> None:
    source = PROVIDER_BINDINGS.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(PROVIDER_BINDINGS))
    imported_modules = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }
    top_level_functions = {
        node.name for node in tree.body if isinstance(node, ast.FunctionDef)
    }
    forbidden_imports = sorted(
        module
        for module in imported_modules
        if module.startswith(FORBIDDEN_PROVIDER_BINDING_IMPORT_PREFIXES)
    )
    referenced_helpers = sorted(
        FORBIDDEN_PROVIDER_BINDING_HELPERS.intersection(_referenced_names(tree))
    )

    assert "quant_research_platform.catalog.defaults" not in imported_modules
    assert forbidden_imports == []
    assert OLD_PRIVATE_HELPER not in top_level_functions
    assert referenced_helpers == []


def test_storage_parquet_uses_data_spec_provider_protocol() -> None:
    tree = ast.parse(
        STORAGE_PARQUET.read_text(encoding="utf-8"),
        filename=str(STORAGE_PARQUET),
    )
    classes = {node.name for node in tree.body if isinstance(node, ast.ClassDef)}

    assert "DataSpecProvider" in classes
    assert "Data" + "Spec" + "Catalog" not in classes


def test_workflow_and_interface_public_names_use_dry_run_wording() -> None:
    forbidden_terms = (
        "workflow" + "-fixture",
        "fixture" + "_backed",
        "fixture" + "_workflow_node_registry",
        "dry" + "_run_workflow_node_registry",
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


def _iter_imports(root: Path) -> tuple[tuple[Path, int, str], ...]:
    imports: list[tuple[Path, int, str]] = []
    for path in sorted(root.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module is not None:
                imports.append((path, node.lineno, node.module))
            elif isinstance(node, ast.Import):
                imports.extend((path, node.lineno, alias.name) for alias in node.names)
    return tuple(imports)


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
