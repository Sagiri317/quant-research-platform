from __future__ import annotations

import ast
from pathlib import Path

PROJECT_SRC = Path("src/quant_research_platform")

KNOWN_BOOTSTRAP_DEFAULTS = {
    "src/quant_research_platform/catalog/data_products.py": {
        "owner": "P1",
        "target": "catalog/defaults.py platform composition root",
    },
    "src/quant_research_platform/catalog/labels.py": {
        "owner": "P1",
        "target": "catalog/defaults.py platform composition root",
    },
    "src/quant_research_platform/catalog/markets.py": {
        "owner": "P1",
        "target": "catalog/defaults.py platform composition root",
    },
    "src/quant_research_platform/catalog/provider_bindings.py": {
        "owner": "P1",
        "target": "catalog/defaults.py platform composition root",
    },
    "src/quant_research_platform/catalog/reports.py": {
        "owner": "P1",
        "target": "catalog/defaults.py platform composition root",
    },
    "src/quant_research_platform/catalog/sources.py": {
        "owner": "P1",
        "target": "catalog/defaults.py platform composition root",
    },
    "src/quant_research_platform/workflows/registry.py": {
        "owner": "P1",
        "target": "workflow_node_registry(execution_mode=...)",
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
    assert (
        KNOWN_BOOTSTRAP_DEFAULTS[
            "src/quant_research_platform/workflows/registry.py"
        ]["target"]
        == "workflow_node_registry(execution_mode=...)"
    )


def _iter_bootstrap_default_definitions() -> tuple[tuple[Path, int], ...]:
    found: list[tuple[Path, int]] = []
    for path in sorted(PROJECT_SRC.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "bootstrap_default":
                found.append((path, node.lineno))
    return tuple(found)
