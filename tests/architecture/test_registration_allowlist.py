from __future__ import annotations

import ast
from pathlib import Path

PROJECT_SRC = Path("src/quant_research_platform")
WORKFLOW_REGISTRY = PROJECT_SRC / "workflows" / "registry.py"
GENERIC_CATALOG_BOOTSTRAP_FORBIDDEN_FILES = {
    "src/quant_research_platform/catalog/data_products.py",
    "src/quant_research_platform/catalog/labels.py",
    "src/quant_research_platform/catalog/markets.py",
    "src/quant_research_platform/catalog/provider_bindings.py",
    "src/quant_research_platform/catalog/reports.py",
    "src/quant_research_platform/catalog/sources.py",
}


def test_generic_catalogs_do_not_define_bootstrap_defaults() -> None:
    hits = [
        f"{path}:{lineno}"
        for path, lineno in _iter_bootstrap_default_definitions()
        if path.as_posix() in GENERIC_CATALOG_BOOTSTRAP_FORBIDDEN_FILES
    ]

    assert not hits, (
        "generic_catalog_bootstrap_default_forbidden: " + "; ".join(hits)
    )


def test_workflow_registry_uses_execution_mode_factory() -> None:
    """dry_run is official; execute fails closed until real execute lands.

    Future execute support must use workflow_node_registry(execution_mode=...)
    as the same public entrypoint. Do not introduce fixture_workflow_node_registry()
    or a long-term public dry_run_workflow_node_registry().
    """

    tree = ast.parse(WORKFLOW_REGISTRY.read_text(encoding="utf-8"))
    top_level_functions = {
        node.name for node in tree.body if isinstance(node, ast.FunctionDef)
    }

    assert "workflow_node_registry" in top_level_functions
    assert "fixture_workflow_node_registry" not in top_level_functions
    assert "dry_run_workflow_node_registry" not in top_level_functions


def test_active_workflow_code_avoids_public_fixture_factory_names() -> None:
    forbidden_terms = (
        "fixture_backed_execution",
        "workflow-fixture-run",
        "fixture_workflow_node_registry",
        "dry_run_workflow_node_registry",
    )
    hits: list[str] = []
    for path in sorted((PROJECT_SRC / "workflows").rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        for lineno, line in enumerate(text.splitlines(), start=1):
            for term in forbidden_terms:
                if term in line:
                    hits.append(f"{path}:{lineno}: {term}")

    assert not hits, "workflow_public_fixture_name_found: " + "; ".join(hits)


def _iter_bootstrap_default_definitions() -> tuple[tuple[Path, int], ...]:
    found: list[tuple[Path, int]] = []
    for path in sorted(PROJECT_SRC.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "bootstrap_default":
                found.append((path, node.lineno))
    return tuple(found)
