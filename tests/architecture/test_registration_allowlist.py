from __future__ import annotations

import ast
from pathlib import Path

PROJECT_SRC = Path("src/quant_research_platform")
WORKFLOW_REGISTRY = PROJECT_SRC / "workflows" / "registry.py"
BATCH_3_BOOTSTRAP_TARGET = (
    "Batch 3: move concrete bootstrap payloads into owner/default composition."
)

KNOWN_BOOTSTRAP_DEFAULTS = {
    "src/quant_research_platform/catalog/data_products.py": {
        "owner": "P1",
        "target": BATCH_3_BOOTSTRAP_TARGET,
    },
    "src/quant_research_platform/catalog/labels.py": {
        "owner": "P1",
        "target": BATCH_3_BOOTSTRAP_TARGET,
    },
    "src/quant_research_platform/catalog/markets.py": {
        "owner": "P1",
        "target": BATCH_3_BOOTSTRAP_TARGET,
    },
    "src/quant_research_platform/catalog/provider_bindings.py": {
        "owner": "P1",
        "target": BATCH_3_BOOTSTRAP_TARGET,
    },
    "src/quant_research_platform/catalog/reports.py": {
        "owner": "P1",
        "target": BATCH_3_BOOTSTRAP_TARGET,
    },
    "src/quant_research_platform/catalog/sources.py": {
        "owner": "P1",
        "target": BATCH_3_BOOTSTRAP_TARGET,
    },
}


def test_bootstrap_default_occurrences_are_known_and_classified() -> None:
    found = {
        path.as_posix(): lineno
        for path, lineno in _iter_bootstrap_default_definitions()
    }
    unclassified = sorted(set(found) - set(KNOWN_BOOTSTRAP_DEFAULTS))
    stale = sorted(set(KNOWN_BOOTSTRAP_DEFAULTS) - set(found))

    assert not unclassified, "unclassified_bootstrap_default: " + repr(
        [(path, found[path]) for path in unclassified]
    )
    assert not stale, "stale_bootstrap_default_allowlist_entry: " + repr(stale)


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
