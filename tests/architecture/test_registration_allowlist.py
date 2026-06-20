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


def test_workflow_registry_uses_run_mode_factory() -> None:
    """validate_only is official; write_artifacts fails closed until it lands.

    Future artifact writing support must use workflow_node_registry(run_mode=...)
    as the same entrypoint. Do not introduce separate fixture or long-term
    validate-only factories.
    """

    tree = ast.parse(WORKFLOW_REGISTRY.read_text(encoding="utf-8"))
    top_level_functions = {
        node.name for node in tree.body if isinstance(node, ast.FunctionDef)
    }
    fixture_factory = "_".join(("fixture", "workflow", "node", "registry"))
    validate_only_factory = "_".join(
        ("validate", "only", "workflow", "node", "registry")
    )

    assert "workflow_node_registry" in top_level_functions
    assert fixture_factory not in top_level_functions
    assert validate_only_factory not in top_level_functions


def test_active_workflow_code_avoids_public_fixture_factory_names() -> None:
    forbidden_terms = (
        "_".join(("fixture", "backed", "run")),
        "-".join(("workflow", "fixture", "run")),
        "_".join(("fixture", "workflow", "node", "registry")),
        "_".join(("validate", "only", "workflow", "node", "registry")),
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
